"""Optic to CODE V Converter

Converts an Optiland Optic object into a CodeVDataModel. This is the mirror
of CodeVToOpticConverter and is the first stage of the write pipeline.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.surfaces import (
    _raw_from_radius,
    get_handler_for_optiland_type,
)
from optiland.fileio.codev.writer.encoder import wavelength_write_order
from optiland.fileio.codev.writer.glass_names import codev_glass_name
from optiland.fileio.common import (
    FIELD_CLASS_TO_TYPE,
    field_type_string,
    is_air,
    reject_unsupported_ideal_absorption,
    validate_material_propagation,
)
from optiland.materials.data import DataMaterial
from optiland.physical_apertures import RadialAperture, UnclippedAperture

if TYPE_CHECKING:
    from optiland.optic import Optic

# Map from Optiland geometry str() to surface type string
_GEOM_STR_TO_TYPE: dict[str, str] = {
    "Planar": "standard",
    "Standard": "standard",
    "Even Asphere": "even_asphere",
}

# CODE V has no infinity; an object distance of 1e10 or more is treated as infinite.
_INFINITE_OBJECT_DISTANCE = 1e10


# CODE V 2025.03 accepts at most 21 wavelengths and 25 fields and silently drops
# the rest (with only a warning), so a larger optic is refused instead.
_CODEV_MAX_WAVELENGTHS = 21
_CODEV_MAX_FIELDS = 25

# Coordinate system entries that move or tilt a surface (z is its axial position).
_OFFSET_KEYS = ("x", "y", "rx", "ry", "rz")
_OFFSET_TOLERANCE = 1e-12


def _object_distance(thickness: float) -> float:
    """Map an Optiland object distance to a value CODE V can read.

    Args:
        thickness: Object surface thickness; ``+inf`` means an object at infinity.

    Returns:
        ``1e10`` for ``+inf``, otherwise ``thickness`` unchanged.

    Raises:
        ValueError: If ``thickness`` is NaN or ``-inf``.
    """
    if math.isnan(thickness) or thickness == -math.inf:
        raise ValueError(
            f"Object surface thickness is not finite ({thickness!r}); "
            "only +inf (object at infinity) can be exported"
        )
    return _INFINITE_OBJECT_DISTANCE if thickness == math.inf else thickness


def _scalar(value: Any) -> float:
    """Return the first element of a backend scalar or array as a float."""
    return float(be.to_numpy(be.atleast_1d(be.array(value))).ravel()[0])


class OpticToCodeVConverter:
    """Converts an Optic object to a CodeVDataModel.

    This is the mirror of CodeVToOpticConverter and constitutes the first
    stage of the write pipeline (Optic -> CodeVDataModel -> text lines).

    Args:
        optic: The Optic to convert.
    """

    def __init__(self, optic: Optic):
        self._optic = optic
        # private glasses of the conversion in progress: formatted index row -> name
        self._private: dict[tuple[str, ...], str] = {}
        self._private_model: CodeVDataModel | None = None

    def convert(self) -> CodeVDataModel:
        """Build and return the CodeVDataModel for the optic.

        Returns:
            A populated CodeVDataModel ready for CodeVFileEncoder.
        """
        validate_material_propagation(self._optic)
        self._check_object_medium()
        self._check_offsets()
        self._check_counts()
        model = CodeVDataModel()
        model.name = self._optic.name
        model.radius_mode = True  # always write radii
        self._private = {}
        self._private_model = model
        self._convert_aperture(model)
        self._convert_fields(model)
        self._convert_wavelengths(model)
        self._warn_pickups_solves()
        self._convert_surfaces(model)
        return model

    def _check_object_medium(self) -> None:
        """Refuse a non-air medium in object space (``SO`` has no glass here)."""
        surfaces = list(self._optic.surfaces)
        if surfaces and not is_air(surfaces[0].material_post):
            raise NotImplementedError(
                "Surface 0: an object-space medium cannot be exported to CODE V yet"
            )

    def _check_counts(self) -> None:
        """Refuse more wavelengths or fields than CODE V loads.

        Raises:
            ValueError: If the optic has more than 21 wavelengths or 25 fields.
        """
        n_wl = len(self._optic.wavelengths.wavelengths)
        if n_wl > _CODEV_MAX_WAVELENGTHS:
            raise ValueError(
                f"CODE V supports at most {_CODEV_MAX_WAVELENGTHS} wavelengths; "
                f"this optic has {n_wl}"
            )
        n_fields = self._optic.fields.num_fields
        if n_fields > _CODEV_MAX_FIELDS:
            raise ValueError(
                f"CODE V supports at most {_CODEV_MAX_FIELDS} fields; "
                f"this optic has {n_fields}"
            )

    def _check_offsets(self) -> None:
        """Refuse decentered, tilted or chained surfaces.

        The writer emits no ``XDE``/``YDE``/``ADE``... lines, and a basic
        decenter in CODE V would carry to the later surfaces, so an offset can
        not be written faithfully yet. ``z`` is the axial position, not an
        offset, and is not checked.

        Raises:
            NotImplementedError: If a coordinate system has a non-zero ``x``,
                ``y``, ``rx``, ``ry`` or ``rz`` or a reference coordinate system.
            ValueError: If one of those values is not finite.
        """
        for i, surface in enumerate(self._optic.surfaces):
            cs = surface.geometry.cs
            if cs.reference_cs is not None:
                raise NotImplementedError(
                    f"Surface {i}: chained coordinate systems (decenters and "
                    "tilts) cannot be exported to CODE V yet"
                )
            offsets = {k: _scalar(getattr(cs, k)) for k in _OFFSET_KEYS}
            for key, value in offsets.items():
                if not math.isfinite(value):
                    raise ValueError(f"Surface {i}: non-finite {key} ({value!r})")
            moved = [k for k, v in offsets.items() if abs(v) > _OFFSET_TOLERANCE]
            if moved:
                raise NotImplementedError(
                    f"Surface {i}: decenters and tilts cannot be exported to CODE V "
                    f"yet (non-zero {', '.join(moved)})"
                )

    # ------------------------------------------------------------------
    # Aperture
    # ------------------------------------------------------------------

    def _convert_aperture(self, model: CodeVDataModel) -> None:
        ap = self._optic.aperture
        if ap is None:
            return
        if ap.ap_type == "float_by_stop_size":
            # The stop-size constraint has no CODE V form: write the EPD it gives.
            epd = _scalar(self._optic.paraxial.EPD())
            model.aperture["EPD"] = epd
            warnings.warn(
                f"Aperture float_by_stop_size (stop diameter {float(ap.value):.8g}, "
                f"surface {self._optic.surfaces.stop_index}) is exported as a "
                f"fixed EPD of {epd:.8g}; the stop-size constraint is not kept.",
                UserWarning,
                stacklevel=3,
            )
            return
        # Map Optiland aperture types to CODE V keys
        mapping = {
            "EPD": "EPD",
            "imageFNO": "FNO",
            "objectNA": "NAO",
        }
        cv_key = mapping.get(ap.ap_type)
        if cv_key is None:
            warnings.warn(
                f"Unknown aperture type '{ap.ap_type}'; skipping aperture export.",
                UserWarning,
                stacklevel=3,
            )
            return
        model.aperture[cv_key] = float(ap.value)

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def _convert_fields(self, model: CodeVDataModel) -> None:
        fields = self._optic.fields
        definition = fields.field_definition
        if (
            definition is not None
            and type(definition).__name__ not in FIELD_CLASS_TO_TYPE
        ):
            raise NotImplementedError(
                f"field definition '{type(definition).__name__}' has no CODE V "
                "equivalent"
            )
        field_type = field_type_string(self._optic)
        n = fields.num_fields

        x_vals = [float(f.x) for f in fields]
        y_vals = [float(f.y) for f in fields]

        weights = [float(f.weight) for f in fields]

        vignetted = [
            i
            for i, f in enumerate(fields)
            if _scalar(f.vx) != 0.0 or _scalar(f.vy) != 0.0
        ]
        if vignetted:
            warnings.warn(
                "Field vignetting factors (vx, vy) are not exported yet; "
                f"non-zero at field(s) {', '.join(map(str, vignetted))}.",
                UserWarning,
                stacklevel=3,
            )

        model.fields = {
            "num_fields": n,
            "type": field_type,
            "x": x_vals,
            "y": y_vals,
            "weights": weights,
        }

    # ------------------------------------------------------------------
    # Wavelengths
    # ------------------------------------------------------------------

    def _convert_wavelengths(self, model: CodeVDataModel) -> None:
        wls = self._optic.wavelengths
        data: list[float] = []
        weights: list[float] = []
        primary_index = 0
        for i, w in enumerate(wls):
            data.append(float(w.value))
            weights.append(float(w.weight))
            if w.is_primary:
                primary_index = i

        model.wavelengths = {
            "data": data,  # in um; encoder converts to nm
            "weights": weights,
            "num_wavelengths": len(data),
            "primary_index": primary_index,
        }

    # ------------------------------------------------------------------
    # Pickups / Solves warning
    # ------------------------------------------------------------------

    def _warn_pickups_solves(self) -> None:
        pickups = list(self._optic.pickups.pickups)
        solves = list(self._optic.solves.solves)
        if pickups:
            warnings.warn(
                f"Optic has {len(pickups)} pickup(s) that cannot be represented "
                "in a .seq file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )
        if solves:
            warnings.warn(
                f"Optic has {len(solves)} solve(s) that cannot be represented "
                "in a .seq file; resolved values will be exported instead.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------

    def _convert_surfaces(self, model: CodeVDataModel) -> None:
        """Iterate optic surfaces and populate model.surfaces.

        The ObjectSurface (index 0) is written as SO, the remaining real
        surfaces as S, and the last surface as SI.
        """
        all_surfaces = list(self._optic.surfaces)

        if not all_surfaces:
            return

        spacings = self._vertex_spacings()

        # Split: object surface (first), real surfaces (middle), image (last)
        obj_surface = all_surfaces[0]
        real_surfaces = all_surfaces[1:-1] if len(all_surfaces) > 2 else []

        output_idx = 0

        # Object surface (SO)
        self._warn_end_surface_aperture(obj_surface, 0)
        obj_thickness = _object_distance(
            spacings[0] if spacings else _scalar(obj_surface.thickness)
        )
        model.surfaces[output_idx] = {
            "type": "object",
            "radius": 0.0,
            "thickness": obj_thickness,
        }
        output_idx += 1

        for surface in real_surfaces:
            model.surfaces[output_idx] = self._convert_real_surface(
                surface, output_idx, spacings[output_idx]
            )
            output_idx += 1

        # Image surface
        radius = 0.0
        if len(all_surfaces) > 1:
            radius = self._image_radius(all_surfaces[-1], output_idx)
            self._warn_end_surface_aperture(all_surfaces[-1], output_idx)
        model.surfaces[output_idx] = {
            "type": "image",
            "radius": radius,
            "thickness": 0.0,
        }

    @staticmethod
    def _warn_not_exported(aperture: Any, surf_idx: int) -> None:
        """Warn that an aperture class has no verified CODE V equivalent."""
        warnings.warn(
            f"Surface {surf_idx}: aperture {type(aperture).__name__} has no "
            "verified CODE V equivalent yet; not exported",
            UserWarning,
            stacklevel=4,
        )

    def _warn_end_surface_aperture(self, surface: Any, surf_idx: int) -> None:
        """Warn about an aperture on the object or image surface (not written)."""
        aperture = surface.aperture
        if aperture is None or isinstance(aperture, UnclippedAperture):
            return
        kind = "object" if surf_idx == 0 else "image"
        warnings.warn(
            f"Surface {surf_idx}: an aperture on the {kind} surface is not "
            "exported to CODE V",
            UserWarning,
            stacklevel=4,
        )

    def _aperture_keys(self, aperture: Any, surf_idx: int) -> dict[str, float]:
        """Return the ``cir``/``cir_obs`` entries for a surface aperture.

        Only a plain ``RadialAperture`` is exported: a finite ``r_max`` becomes
        ``cir`` (``CIR CLR``) and ``r_min > 0`` becomes ``cir_obs``
        (``CIR OBS``). An infinite ``r_max`` means "no limit" and writes no
        ``CIR CLR``. Every other class is warned about and not written.

        Raises:
            ValueError: If ``r_max`` is NaN or ``-inf``, or ``r_min`` is not a
                finite non-negative number below ``r_max``.
        """
        if aperture is None or isinstance(aperture, UnclippedAperture):
            return {}
        if type(aperture) is not RadialAperture:
            self._warn_not_exported(aperture, surf_idx)
            return {}
        r_max = _scalar(aperture.r_max)
        r_min = _scalar(aperture.r_min)
        if math.isnan(r_max) or r_max == -math.inf:
            raise ValueError(
                f"Surface {surf_idx}: non-finite aperture r_max ({r_max!r})"
            )
        if not (0.0 <= r_min < r_max):
            raise ValueError(
                f"Surface {surf_idx}: invalid aperture, need 0 <= r_min < r_max "
                f"(r_min={r_min!r}, r_max={r_max!r})"
            )
        keys: dict[str, float] = {}
        if r_max != math.inf:
            keys["cir"] = r_max
        if r_min > 0.0:
            keys["cir_obs"] = r_min
        return keys

    @staticmethod
    def _image_radius(surface: Any, output_idx: int) -> float:
        """Return the raw radius of the image surface (0 for a plane).

        Raises:
            NotImplementedError: If the image surface is not planar or a sphere,
                or has a conic constant.
        """
        geom = surface.geometry
        geom_str = str(geom)
        if geom_str == "Even Asphere":
            raise NotImplementedError(
                f"Surface {output_idx}: an aspheric image surface cannot be "
                "exported to CODE V yet"
            )
        if geom_str not in ("Planar", "Standard"):
            raise NotImplementedError(
                f"Surface {output_idx}: image surface geometry '{geom_str}' "
                "cannot be exported to CODE V yet"
            )
        if not abs(_scalar(getattr(geom, "k", 0.0))) <= 1e-16:
            raise NotImplementedError(
                f"Surface {output_idx}: an image surface conic cannot be "
                "exported to CODE V yet"
            )
        return _raw_from_radius(_scalar(getattr(geom, "radius", math.inf)))

    def _vertex_spacings(self) -> list[float]:
        """Return the axial vertex spacing after each surface.

        The spacing is the difference of the global z positions of consecutive
        vertices, so absolute ``z`` positions and solved thicknesses are written
        as they are in the system, not as a stale ``thickness`` attribute. Where
        the attribute agrees with the spacing to 8 machine epsilons of the dtype
        of the global z (scaled by the z values), the attribute is written,
        because the global z of a float32 lens is a cumulative sum.
        It is the CODE V ``THI`` only because decenters and tilts are refused.

        Returns:
            One value per surface but the last; ``+inf`` after an object at
            infinity. Spacings below 1e-12 are returned as 0.0.
        """
        z = be.to_numpy(self._optic.surfaces.global_z_positions).ravel()
        eps = float(np.finfo(z.dtype).eps)  # precision the z values were computed in
        z = z.astype(np.float64)
        spacings = []
        for i, d in enumerate(np.diff(z)):
            # A relative lens keeps its exact attribute (global z of a float32
            # lens is a cumulative sum); absolute z and solves differ from it.
            t = self._optic.surfaces[i].thickness
            # a plain number is exact; _scalar would round it to the backend dtype
            t = float(t) if isinstance(t, int | float) else _scalar(t)
            tol = 8 * eps * max(1.0, abs(z[i]), abs(z[i + 1]))
            if math.isfinite(t) and abs(d - t) <= tol:
                d = t
            spacings.append(0.0 if abs(d) < 1e-12 else float(d))
        return spacings

    def _convert_real_surface(
        self, surface: Any, output_idx: int, thickness: float
    ) -> dict[str, Any]:
        """Build the raw CODE V surface dict for a single non-object/image surface.

        Args:
            surface: The Optiland surface.
            output_idx: Output surface index (used in messages).
            thickness: Axial vertex spacing after the surface.
        """
        geom = surface.geometry
        geom_str = str(geom)
        optiland_type = _GEOM_STR_TO_TYPE.get(geom_str)

        if optiland_type is None:
            raise NotImplementedError(
                f"Surface {output_idx}: geometry type '{geom_str}' "
                "is not supported by the CODE V writer."
            )

        interaction_type = surface.interaction_model.interaction_type
        if interaction_type != "refractive_reflective":
            label = "thin-lens" if interaction_type == "thin_lens" else interaction_type
            raise NotImplementedError(
                f"Surface {output_idx}: {label} surfaces cannot be exported "
                "to CODE V yet"
            )

        handler = get_handler_for_optiland_type(optiland_type)
        try:
            raw = handler.format(surface)
        except NotImplementedError as exc:
            raise NotImplementedError(f"Surface {output_idx}: {exc}") from None

        raw["thickness"] = thickness
        raw["type"] = "standard"

        if surface.is_stop:
            raw["is_stop"] = True

        raw.update(self._aperture_keys(surface.aperture, output_idx))

        # Glass — check reflective flag first (mirror)
        is_reflective = getattr(
            getattr(surface, "interaction_model", None), "is_reflective", False
        )
        glass_entry = self._format_glass(
            surface.material_post, output_idx, is_reflective
        )
        if glass_entry is not None:
            raw["glass"] = glass_entry

        return raw

    def _format_glass(
        self,
        mat: Any,
        surf_idx: int,
        is_reflective: bool = False,
    ) -> dict[str, Any] | None:
        """Build a glass specification dict for a material.

        Args:
            mat: The surface material (post).
            surf_idx: Output surface index (used in warnings).
            is_reflective: True if the surface is a reflective (mirror) surface.

        Returns:
            A dict with key ``"name"`` (and optionally ``"catalog"``),
            or None for air.
        """
        # Mirror surface — detected via interaction_model.is_reflective
        if is_reflective:
            return {"name": "REFL"}

        reject_unsupported_ideal_absorption(mat)
        if isinstance(mat, DataMaterial):
            raise NotImplementedError(
                "This writer cannot preserve DataMaterial optical data; use native JSON"
            )

        if is_air(mat):
            return None

        # Mirror material string fallback
        if isinstance(mat, str) and mat.lower() == "mirror":
            return {"name": "REFL"}

        # Catalog glass with a verified CODE V name
        named = codev_glass_name(mat)
        if named is not None:
            return {"name": named[0], "catalog": named[1]}

        # Everything else -> private glass with the exact indices
        return {"private": self._register_private_glass(mat, surf_idx)}

    def _register_private_glass(self, mat: Any, surf_idx: int) -> str:
        """Add a material to the private glass catalog and return its name.

        The index is evaluated at the wavelengths in the order they are written
        (the ``PWL`` list). Materials with the same formatted row share one name.

        Args:
            mat: The surface material.
            surf_idx: Output surface index (used in messages).

        Returns:
            The private glass name, ``OPT1``, ``OPT2``, ...

        Raises:
            ValueError: If the system has no wavelengths, the material has no
                index at one of them or the index is not positive.
        """
        model = self._private_model
        wl_data = model.wavelengths
        data = wl_data.get("data", [])
        label = getattr(mat, "name", type(mat).__name__)
        if not data:
            raise ValueError(
                f"Surface {surf_idx}: material '{label}' needs a private glass but "
                "the system has no wavelengths"
            )
        indices: list[float] = []
        k_max = 0.0
        for i in wavelength_write_order(wl_data):
            wavelength = data[i]
            try:
                n = _scalar(mat.n(wavelength))
                k = _scalar(mat.k(wavelength))
            except Exception as exc:
                raise ValueError(
                    f"Surface {surf_idx}: material '{label}' has no index at "
                    f"{wavelength} um; cannot export ({exc})"
                ) from None
            if n <= 0:  # NaN and inf are refused by the encoder
                raise ValueError(
                    f"Surface {surf_idx}: non-physical index ({n!r}) of private "
                    f"glass 'OPT{len(self._private) + 1}' at "
                    f"{wavelength * 1000.0:.8g} nm"
                )
            indices.append(float(f"{n:.10g}"))
            k_max = max(k_max, k)

        if k_max > 0:
            warnings.warn(
                f"Surface {surf_idx}: extinction k up to {k_max:.3g} of '{label}' "
                "is not exported; refractive indices are exact.",
                UserWarning,
                stacklevel=5,
            )

        key = tuple(f"{n:.10g}" for n in indices)
        name = self._private.get(key)
        if name is not None:
            return name
        name = f"OPT{len(self._private) + 1}"  # at most 8 characters
        self._private[key] = name
        model.private_glasses[name] = indices
        warnings.warn(
            f"Surface {surf_idx}: glass '{label}' has no verified CODE V catalog "
            f"entry; written as private glass '{name}' with n at the system "
            "wavelengths.",
            UserWarning,
            stacklevel=5,
        )
        return name
