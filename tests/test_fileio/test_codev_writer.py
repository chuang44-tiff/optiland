"""Tests for the CODE V writer path.

Covers OpticToCodeVConverter, CodeVFileEncoder, CodeVWriter,
and the save_codev_file public entry point.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from optiland.fileio import load_codev_file, save_codev_file
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.writer.encoder import CodeVFileEncoder
from optiland.fileio.codev.writer.exporter import CodeVWriter
from optiland.fileio.codev.writer.formatter import OpticToCodeVConverter
from optiland.optic import Optic
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codev_files")


def _seq(name: str) -> str:
    return os.path.join(_DIR, name)


def _make_singlet() -> Optic:
    """Build a minimal singlet optic for writer tests."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=1e10)
    optic.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=5.0)
    optic.add_wavelength(value=0.5876, is_primary=True)
    optic.add_wavelength(value=0.4861)
    optic.add_wavelength(value=0.6563)
    return optic


def _make_conic_sphere(conic: float = 0.0, material=None) -> Optic:
    """Singlet built in code (no reader) with a conic and any surface-1 material."""
    from optiland.materials import IdealMaterial

    optic = Optic()
    optic.surfaces.add(index=0, thickness=1e10)
    optic.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        conic=conic,
        material=material if material is not None else IdealMaterial(1.5),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


# ---------------------------------------------------------------------------
# OpticToCodeVConverter
# ---------------------------------------------------------------------------


class TestOpticToCodeVConverter:
    def test_convert_returns_model(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert isinstance(model, CodeVDataModel)

    def test_aperture_epd(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert "EPD" in model.aperture
        assert_allclose(model.aperture["EPD"], 10.0)

    def test_wavelengths_in_microns(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        data = model.wavelengths["data"]
        assert len(data) == 3
        assert_allclose(data[0], 0.5876, atol=1e-4)

    def test_primary_wavelength_index(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert model.wavelengths["primary_index"] == 0

    def test_fields(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert model.fields["type"] == "angle"
        assert model.fields["y"] == [0.0, 5.0]

    def test_surfaces_contain_object_and_image(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        surf_types = [s["type"] for s in model.surfaces.values()]
        assert "object" in surf_types
        assert "image" in surf_types

    def test_catalog_glass_format(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        # Find the surface with glass
        glass_surfs = [s for s in model.surfaces.values() if s.get("glass") is not None]
        assert len(glass_surfs) >= 1
        glass = glass_surfs[0]["glass"]
        assert "name" in glass

    def test_pickups_warning(self):
        """Solves should issue a UserWarning."""
        optic = _make_singlet()
        optic.solves.add("marginal_ray_height", surface_idx=2, height=0.0)
        with pytest.warns(UserWarning, match="solve"):
            OpticToCodeVConverter(optic).convert()


# ---------------------------------------------------------------------------
# CodeVFileEncoder
# ---------------------------------------------------------------------------


class TestCodeVFileEncoder:
    def _make_model(self) -> CodeVDataModel:
        optic = _make_singlet()
        return OpticToCodeVConverter(optic).convert()

    def test_encode_returns_lines(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert isinstance(lines, list)
        assert len(lines) > 0
        assert all(isinstance(line, str) for line in lines)

    def test_header_starts_with_len_new_dim_m(self):
        lines = CodeVFileEncoder(self._make_model()).encode()
        assert lines[0] == "LEN NEW"
        assert lines[1] == "DIM M"
        assert lines.index("RDM Y") > 1

    def test_ends_with_go(self):
        lines = CodeVFileEncoder(self._make_model()).encode()
        assert lines[-1] == "GO"
        assert lines[-2].startswith("SI ")

    def test_header_has_rdm(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("RDM") for line in lines)

    def test_header_has_wl(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("WL") for line in lines)

    def test_wavelengths_in_nm(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        wl_line = next(l for l in lines if l.strip().startswith("WL"))
        tokens = wl_line.split()
        nm_vals = [float(t) for t in tokens[1:]]
        # Primary wavelength ~587.6 nm
        assert any(abs(v - 587.6) < 1.0 for v in nm_vals)

    def test_has_ref_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("REF") for line in lines)

    def test_has_yan_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("YAN") for line in lines)

    def test_has_epd_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("EPD") for line in lines)

    def test_has_so_and_si(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("SO") for line in lines)
        assert any(line.strip().startswith("SI") for line in lines)

    def test_title_quoted(self):
        model = self._make_model()
        model.name = "My Optic"
        lines = CodeVFileEncoder(model).encode()
        assert any("TITLE" in line and "My Optic" in line for line in lines)

    def test_aspheric_coeff_A(self):
        """Even-asphere coefficients are refused until their powers are fixed."""
        optic = load_codev_file(_seq("asphere.seq"))
        with pytest.raises(NotImplementedError, match="even-asphere"):
            OpticToCodeVConverter(optic).convert()

    def test_conic_K(self):
        optic = _make_conic_sphere(conic=-0.5)
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        k_idx = next(i for i, line in enumerate(lines) if line.strip() == "K -0.5")
        assert lines[k_idx - 1].strip() == "CON"

    def test_stop_sto(self):
        optic = load_codev_file(_seq("cooke_triplet.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip() == "STO" for line in lines)

    def test_mirror_refl(self):
        optic = load_codev_file(_seq("mirror.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any("REFL" in line for line in lines)

    def test_fictitious_glass_nd_vd(self):
        from optiland.materials import AbbeMaterial

        optic = _make_conic_sphere(material=AbbeMaterial(1.5168, 64.17))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        # Fictitious glass is written as Nd:Vd on the surface line
        s_lines = [
            l for l in lines if l.strip().startswith("S ") or l.startswith("S  ")
        ]
        assert any(":" in line for line in s_lines)


# ---------------------------------------------------------------------------
# save_codev_file / CodeVWriter
# ---------------------------------------------------------------------------


class TestSaveCodeVFile:
    def test_save_creates_file(self):
        optic = _make_singlet()
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_saved_file_ends_with_go_newline(self, tmp_path):
        path = tmp_path / "out.seq"
        save_codev_file(_make_singlet(), str(path))
        assert path.read_text(encoding="utf-8").endswith("GO\n")

    def test_save_utf8_encoding(self):
        optic = _make_singlet()
        with tempfile.NamedTemporaryFile(
            suffix=".seq", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic, path)
            # Should open without error with utf-8
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert len(content) > 0
        finally:
            os.unlink(path)

    def test_codev_writer_interface(self):
        optic = _make_singlet()
        writer = CodeVWriter()
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            warnings_list = writer.write(optic, path)
            assert isinstance(warnings_list, list)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _round_trip(self, seq_name: str) -> tuple[Optic, Optic]:
        optic1 = load_codev_file(_seq(seq_name))
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic1, path)
            optic2 = load_codev_file(path)
        finally:
            os.unlink(path)
        return optic1, optic2

    def test_roundtrip_wavelength_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.wavelengths.num_wavelengths == o2.wavelengths.num_wavelengths

    def test_roundtrip_primary_wavelength(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert_allclose(
            float(o1.primary_wavelength), float(o2.primary_wavelength), rtol=1e-5
        )

    def test_roundtrip_fields_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.fields.num_fields == o2.fields.num_fields

    def test_roundtrip_aperture_value(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert_allclose(float(o1.aperture.value), float(o2.aperture.value), rtol=1e-5)

    def test_roundtrip_surface_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.surface_group.num_surfaces == o2.surface_group.num_surfaces

    def test_roundtrip_asphere(self):
        optic = load_codev_file(_seq("asphere.seq"))
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            with pytest.raises(NotImplementedError):
                save_codev_file(optic, path)
        finally:
            os.unlink(path)

    def test_roundtrip_fno(self):
        o1, o2 = self._round_trip("fno_fields.seq")
        assert_allclose(float(o1.aperture.value), float(o2.aperture.value), rtol=1e-4)


# ---------------------------------------------------------------------------
# OpticToCodeVConverter Extended
# ---------------------------------------------------------------------------


class TestOpticToCodeVConverterExtended:
    def test_convert_aperture_codev(self):
        from optiland.aperture import ImageFNOAperture

        optic = Optic()
        optic.aperture = ImageFNOAperture(4.0)
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.aperture["FNO"] == 4.0

    def test_warn_unknown_aperture_codev(self):
        optic = Optic()

        class UnknownAp:
            ap_type = "unknown"
            value = 0.0

        optic.aperture = UnknownAp()
        optic.surfaces.add(index=0, thickness=0.0)
        conv = OpticToCodeVConverter(optic)
        with pytest.warns(UserWarning, match="Unknown aperture type"):
            conv.convert()

    def test_no_surfaces_codev(self):
        optic = Optic()
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.surfaces == {}

    def test_glass_fictitious_codev(self):
        from optiland.materials import IdealMaterial

        optic = Optic()
        # Add 3 surfaces: Obj, Real, Img
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(
            index=1, radius=50.0, thickness=5.0, material=IdealMaterial(1.7)
        )
        optic.surfaces.add(index=2, thickness=0.0)
        optic.add_wavelength(0.5876, is_primary=True)
        conv = OpticToCodeVConverter(optic)
        with pytest.warns(UserWarning, match="writing as fictitious glass"):
            model = conv.convert()
        # Obj(0) + Surf(1) + Img(2)
        assert "glass" in model.surfaces[1]
        assert float(model.surfaces[1]["glass"]["nd"]) == pytest.approx(1.7)

    def test_reflective_surface_codev(self):
        optic = Optic()
        # Add object and reflective surface
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(index=1, radius=100.0, thickness=-50.0, material="mirror")
        optic.surfaces.add(index=2, thickness=0.0)
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.surfaces[1]["glass"]["name"] == "REFL"


# ---------------------------------------------------------------------------
# Non-finite values (CODE V has no inf / nan)
# ---------------------------------------------------------------------------


def _make_lens(
    object_thickness: float = float("inf"),
    conic: float = 0.0,
    field_weight: float = 1.0,
    radius: float = 50.0,
) -> Optic:
    """Singlet with an object at ``object_thickness``, built via the public API."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=object_thickness)
    optic.surfaces.add(
        index=1,
        radius=radius,
        thickness=5.0,
        conic=conic,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0, weight=field_weight)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


def _lens_with_aperture(aperture) -> Optic:
    """Singlet whose first surface carries ``aperture``."""
    optic = _make_lens()
    optic.surfaces[1].aperture = aperture
    return optic


def _encode(optic: Optic) -> list[str]:
    return CodeVFileEncoder(OpticToCodeVConverter(optic).convert()).encode()


def _make_asphere(coefficients, conic: float = 0.0) -> Optic:
    """Singlet whose first surface is an even asphere."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=float("inf"))
    optic.surfaces.add(
        index=1,
        surface_type="even_asphere",
        radius=50.0,
        thickness=5.0,
        conic=conic,
        coefficients=coefficients,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


class TestEvenAsphereRefusal:
    @pytest.mark.parametrize(
        "coefficients",
        [[1e-6], [0.0, 1.5e-6], [0.0, 0.0, -2.3e-9], [0.0, 0.0, 0.0, 0.0, 1e-20]],
    )
    def test_nonzero_coefficient_refused(
        self, coefficients, tmp_path, set_test_backend
    ):
        optic = _make_asphere(coefficients, conic=-0.5)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(NotImplementedError, match=r"Surface 1: even-asphere"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_nan_coefficient_refused(self, set_test_backend):
        optic = _make_asphere([0.0, float("nan")])
        with pytest.raises(NotImplementedError, match=r"Surface 1: even-asphere"):
            OpticToCodeVConverter(optic).convert()

    @pytest.mark.parametrize("coefficients", [[], [0.0], [0.0, 0.0, 0.0], [0.0, 1e-31]])
    def test_conic_only_asphere_written_as_con(self, coefficients, set_test_backend):
        lines = _encode(_make_asphere(coefficients, conic=-0.5))
        k = lines.index("  K -0.5")
        assert lines[k - 1] == "  CON"
        assert "  ASP" not in lines
        letters = ("A", "B", "C", "D", "E", "F", "G", "H", "J")
        assert not any(line.split()[:1] == [c] for line in lines for c in letters)

    def test_zero_coefficients_and_zero_conic_write_nothing_extra(
        self, set_test_backend
    ):
        lines = _encode(_make_asphere([0.0, 0.0]))
        assert "  CON" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)

    def test_aspheric_singlet_sample_refused(self, set_test_backend):
        from optiland.samples.simple import AsphericSinglet

        with pytest.raises(NotImplementedError, match=r"Surface \d+: even-asphere"):
            OpticToCodeVConverter(AsphericSinglet()).convert()


class TestProfileKeyword:
    def test_conic_writes_con_before_k(self, set_test_backend):
        lines = _encode(_make_lens(conic=-1.0))
        k = lines.index("  K -1")
        assert lines[k - 1] == "  CON"
        assert lines.count("  CON") == 1

    def test_sphere_no_profile_keyword(self, set_test_backend):
        lines = _encode(_make_lens())
        assert "  CON" not in lines
        assert "  ASP" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)

    def test_profile_keyword_per_surface(self, set_test_backend):
        from optiland.materials import Material

        optic = _make_lens()
        optic.surfaces.add(
            index=2,
            radius=80.0,
            thickness=2.0,
            conic=-0.5,
            material=Material("N-BK7", "schott"),
        )
        lines = _encode(optic)
        s_idx = [i for i, line in enumerate(lines) if line.startswith("S ")]
        assert len(s_idx) == 3
        # CON follows only the conic surface (second S line), directly before K
        con = lines.index("  CON")
        assert s_idx[1] < con < s_idx[2]
        assert lines[con + 1] == "  K -0.5"
        assert lines.count("  CON") == 1

    def test_tiny_conic_below_threshold_writes_nothing(self, set_test_backend):
        lines = _encode(_make_lens(conic=1e-18))
        assert "  CON" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)


class TestNonFinite:
    def test_infinite_object_written_as_1e10(self, set_test_backend):
        lines = _encode(_make_lens())
        so = next(line for line in lines if line.startswith("SO "))
        assert float(so.split()[2]) == 1e10
        text = "\n".join(lines).lower()
        assert "inf" not in text
        assert "nan" not in text

    def test_finite_object_distance_unchanged(self, set_test_backend):
        lines = _encode(_make_lens(object_thickness=250.0))
        so = next(line for line in lines if line.startswith("SO "))
        assert float(so.split()[2]) == 250.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"object_thickness": float("-inf")}, "Object"),
            ({"object_thickness": float("nan")}, "Object"),
            ({"conic": float("nan")}, "Surface 1"),
            ({"radius": float("nan")}, "Surface 1"),
            ({"field_weight": float("nan")}, "header"),
        ],
    )
    def test_nonfinite_values_raise_and_leave_file(
        self, kwargs, match, tmp_path, set_test_backend
    ):
        optic = _make_lens(**kwargs)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=match):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_planar_radius_not_flagged(self, set_test_backend):
        lines = _encode(_make_lens(radius=float("inf")))
        text = "\n".join(lines).lower()
        assert "inf" not in text
        s1 = next(line for line in lines if line.startswith("S "))
        assert s1.split()[1] == "0"

    def test_infinite_aperture_writes_no_cir(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        lines = _encode(_lens_with_aperture(RadialAperture(r_max=float("inf"))))
        text = "\n".join(lines)
        assert "CIR" not in text
        assert "inf" not in text.lower()

    def test_finite_aperture_written(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        lines = _encode(_lens_with_aperture(RadialAperture(5.0)))
        assert lines.count("  CIR CLR 5") == 1

    def test_nan_aperture_raises(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        with pytest.raises(ValueError, match="non-finite"):
            _encode(_lens_with_aperture(RadialAperture(r_max=float("nan"))))

    @pytest.mark.parametrize("case", ["ideal_nan", "hikari_30um"])
    def test_nonfinite_glass_index_raises_and_leaves_file(
        self, case, tmp_path, set_test_backend
    ):
        import warnings

        from optiland.materials import IdealMaterial, Material

        if case == "ideal_nan":
            optic = _make_conic_sphere(material=IdealMaterial(float("nan")))
        else:
            optic = _make_conic_sphere(material=Material("SK16", "hikari"))
            optic.wavelengths.wavelengths.clear()
            optic.wavelengths.add(value=30.0, is_primary=True)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"Surface 1: non-finite (nd|vd)"):
                save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("bad_vd", [float("nan"), float("inf")])
    def test_nonfinite_vd_with_finite_nd_raises_and_leaves_file(
        self, bad_vd, tmp_path, set_test_backend
    ):
        import warnings

        import numpy as np

        from optiland.materials import IdealMaterial

        class _OddDispersion(IdealMaterial):
            """Finite n at the 0.55 um primary; F/C/d lines chosen to break Vd."""

            def n(self, wavelength, **kwargs):
                wl = np.asarray(wavelength, dtype=float)
                nan_f = np.isnan(bad_vd)
                n_f = np.nan if nan_f else 1.6
                n_d = 1.5 if nan_f else np.inf
                return np.where(
                    np.isclose(wl, 0.4861),
                    n_f,
                    np.where(np.isclose(wl, 0.5876), n_d, 1.5),
                )

        optic = _make_conic_sphere(material=_OddDispersion(1.5))
        optic.wavelengths.wavelengths.clear()
        optic.wavelengths.add(value=0.55, is_primary=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = OpticToCodeVConverter(optic).convert()
        glass = model.surfaces[1]["glass"]
        assert np.isfinite(glass["nd"])
        assert not np.isfinite(glass["vd"])

        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"Surface 1: non-finite vd"):
                save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("r_max", [float("nan"), float("-inf")])
    def test_nonfinite_aperture_names_surface_and_leaves_file(
        self, r_max, tmp_path, set_test_backend
    ):
        from optiland.physical_apertures import RadialAperture

        optic = _lens_with_aperture(RadialAperture(r_max=r_max))
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=r"Surface 1: non-finite aperture r_max"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_fmt_rejects_nonfinite(self):
        from optiland.fileio.codev.writer.encoder import _fmt, _fmt_list

        with pytest.raises(ValueError, match="non-finite"):
            _fmt(float("inf"))
        with pytest.raises(ValueError, match="non-finite"):
            _fmt_list([1.0, float("nan")])


# ---------------------------------------------------------------------------
# CODE V glass names
# ---------------------------------------------------------------------------


def _surface_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("S ")]


def _glass_token(line: str) -> str | None:
    """Fourth field of an ``S R T GLASS`` line, or None for air."""
    fields = line.split()
    return fields[3] if len(fields) > 3 else None


class TestGlassNames:
    @staticmethod
    def _lines_with(material) -> list[str]:
        return _encode(_make_conic_sphere(material=material))

    @pytest.mark.parametrize(
        "args, expected",
        [
            (("N-BK7", "schott"), "NBK7_SCHOTT"),
            (("N-SF11",), "NSF11_SCHOTT"),
            (("BK7",), "NBK7_SCHOTT"),
            (("SF5", "schott"), "NSF5_SCHOTT"),
            (("LAC9", "hoya"), "LAC9_HOYA"),
            (("S-BSL7", "ohara"), "SBSL7_OHARA"),
            (("H-K9L", "cdgm"), "HK9L_CDGM"),
            (("K-BK7", "sumita"), "KBK7_SUMITA"),
        ],
    )
    def test_name_from_resolved_file(self, args, expected, set_test_backend):
        from optiland.materials import Material

        lines = self._lines_with(Material(*args))
        assert _glass_token(_surface_lines(lines)[0]) == expected

    @pytest.mark.parametrize(
        "args",
        [("SK16", "hikari"), ("LITHOTEC-CAF2", "schott"), ("S-APL", "ohara")],
    )
    def test_unnamed_materials_use_index_and_warn(self, args, set_test_backend):
        from optiland.fileio.codev.writer.glass_names import codev_glass_name
        from optiland.materials import Material

        mat = Material(*args)
        assert codev_glass_name(mat) is None
        with pytest.warns(UserWarning, match="fictitious glass"):
            lines = self._lines_with(mat)
        token = _glass_token(_surface_lines(lines)[0])
        assert ":" in token
        assert "HIKARI" not in "\n".join(lines).upper()

    def test_catalog_air_not_named(self, set_test_backend):
        from optiland.materials import Material

        with pytest.warns(UserWarning, match="fictitious glass"):
            lines = self._lines_with(Material("air"))
        text = "\n".join(lines).upper()
        assert "AIR_" not in text
        assert "CIDDOR" not in text

    def test_filename_mapping_rules(self):
        from optiland.fileio.codev.writer.glass_names import (
            NOT_IN_CODEV,
            codev_name_for_filename,
        )

        assert codev_name_for_filename("glass/schott/N-BK7.yml") == ("NBK7", "SCHOTT")
        assert codev_name_for_filename(r"glass\schott\N-BK7.yml") == (
            "NBK7",
            "SCHOTT",
        )
        # sub-folders, other catalogs and non-glass folders are never named
        assert codev_name_for_filename("glass/schott/misc/ZERODUR.yml") is None
        assert codev_name_for_filename("glass/hikari/SK16.yml") is None
        assert codev_name_for_filename("other/mixed gases/air/Ciddor.yml") is None
        assert codev_name_for_filename("main/SiO2/Malitson.yml") is None
        # path-shape checks: each must go unnamed on its own
        assert codev_name_for_filename("glass/schott/N-BK7.txt") is None
        assert codev_name_for_filename("glass/schott/N-BK7") is None
        assert codev_name_for_filename("other/schott/N-BK7.yml") is None
        assert codev_name_for_filename("schott/misc/N-BK7.yml") is None
        assert codev_name_for_filename("glass/schott/misc/N-BK7.yml") is None
        assert codev_name_for_filename("glass/schott.yml") is None
        # the denylist is matched on the upper-cased stem
        assert codev_name_for_filename("glass/schott/sf5.yml") is None
        assert len(NOT_IN_CODEV) == 10
        for catalog, stem in NOT_IN_CODEV:
            assert codev_name_for_filename(f"glass/{catalog}/{stem}.yml") is None

    def test_non_material_classes_unnamed(self):
        from optiland.fileio.codev.writer.glass_names import codev_glass_name
        from optiland.materials import AbbeMaterial, IdealMaterial

        assert codev_glass_name(IdealMaterial(1.5)) is None
        assert codev_glass_name(AbbeMaterial(1.5, 60.0, model="buchdahl")) is None
        assert codev_glass_name("N-BK7") is None
        assert codev_glass_name(None) is None

    def test_name_charset_whole_db(self):
        import re

        from optiland.fileio.codev.writer.glass_names import (
            CODEV_CATALOGS,
            codev_name_for_filename,
        )
        from optiland.materials import Material

        named = 0
        for filename in Material._load_dataframe()["filename"]:
            result = codev_name_for_filename(filename)
            if result is None:
                continue
            named += 1
            assert re.fullmatch(r"[A-Z0-9]+", result[0]), filename
            assert result[1] in CODEV_CATALOGS.values()
        assert named > 100

    @pytest.mark.parametrize(
        "sample",
        [
            "CookeTriplet",
            "DoubleGauss",
            "EyepieceErfle",
            "ObjectiveUS008879901",
            "HubbleTelescope",
        ],
    )
    def test_samples_have_no_unsupported_tokens(
        self, sample, tmp_path, set_test_backend
    ):
        import re
        import warnings

        from optiland import samples

        cls = next(
            getattr(module, sample)
            for module in (
                samples.objectives,
                samples.eyepieces,
                samples.telescopes,
            )
            if hasattr(module, sample)
        )
        path = tmp_path / f"{sample}.seq"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_codev_file(cls(), str(path))
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "inf" not in lowered
        assert "nan" not in lowered
        assert "mfl" not in lowered
        for line in _surface_lines(text.splitlines()):
            token = _glass_token(line)
            if token is None or token == "REFL":
                continue
            assert re.fullmatch(
                r"[A-Z0-9]+_(SCHOTT|OHARA|HOYA|CDGM|SUMITA)|\d+\.\d+:\d+\.\d+", token
            ), line
            assert "-" not in token


# ---------------------------------------------------------------------------
# Non-refractive interaction models are refused
# ---------------------------------------------------------------------------


class TestInteractionRefusal:
    @staticmethod
    def _thin_lens_optic() -> Optic:
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1, surface_type="paraxial", f=100.0, thickness=50.0, is_stop=True
        )
        optic.surfaces.add(index=2)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    def test_thin_lens_refused_and_file_untouched(self, tmp_path, set_test_backend):
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=r"Surface 1: thin-lens surfaces cannot be exported to CODE V yet",
        ):
            save_codev_file(self._thin_lens_optic(), str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("interaction_type", ["diffractive", "phase"])
    def test_diffractive_and_phase_refused(
        self, interaction_type, monkeypatch, set_test_backend
    ):
        optic = _make_lens()
        monkeypatch.setattr(
            optic.surfaces[1].interaction_model, "interaction_type", interaction_type
        )
        with pytest.raises(
            NotImplementedError,
            match=rf"Surface 1: {interaction_type} surfaces cannot be exported",
        ):
            OpticToCodeVConverter(optic).convert()

    def test_refractive_and_mirror_still_exported(self, set_test_backend):
        lines = _encode(_make_lens())
        assert not any(line.strip().startswith("MFL") for line in lines)
        optic = Optic()
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(index=1, radius=100.0, thickness=-50.0, material="mirror")
        optic.surfaces.add(index=2, thickness=0.0)
        assert any("REFL" in line for line in _encode(optic))
