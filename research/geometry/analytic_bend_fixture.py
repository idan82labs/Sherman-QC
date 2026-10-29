"""Small, in-memory analytical bend surfaces; not a detector or scan simulator.

The model is an annular cylindrical sector with tangent planar flanges, extruded
over a finite axial interval. It has no end caps, material law or visibility
inference. Labels and exact normals are construction truth, not inference inputs.
See docs/research/2026-09-07_analytic_bend_testing_design.md for conventions.
"""

from dataclasses import dataclass, replace
import math

import numpy as np


MAX_SAMPLES = 4096
_METRES_PER_UNIT = {"m": 1.0, "mm": 0.001}
_FRAME_ATOL = 1e-12  # Dimensionless floating-point check, not metrology tolerance.


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class BendSpec:
    """One stipulated component; all lengths, including origin, share ``unit``.

    ``angle_rad`` is the signed rotation of start_radial about directed axis,
    not the opening between outward flange rays. The admissible range is
    0 < abs(angle_rad) <= pi; a flat sheet is intentionally a different model.
    ``span`` bounds the cylinder centre axis, not a visible line on the sheet.
    """

    inner_radius: float
    thickness: float
    angle_rad: float
    span: tuple[float, float]
    flange_lengths: tuple[float, float]
    component_id: str = "component-0"
    unit: str = "mm"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    start_radial: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        for name in ("inner_radius", "thickness", "angle_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.inner_radius <= 0 or self.thickness <= 0:
            raise ValueError("inner_radius and thickness must be positive")
        if not math.isfinite(self.inner_radius + self.thickness):
            raise ValueError("outer radius must be finite")
        if not 0 < abs(self.angle_rad) <= math.pi:
            raise ValueError("require 0 < abs(angle_rad) <= pi")
        for name, size in (("span", 2), ("flange_lengths", 2), ("origin", 3),
                           ("axis", 3), ("start_radial", 3)):
            object.__setattr__(self, name, tuple(_vector(getattr(self, name), size, name)))
        length = float(self.span[1]) - float(self.span[0])
        if not math.isfinite(length) or length <= 0:
            raise ValueError("span must be finite and strictly increasing")
        if min(self.flange_lengths) <= 0:
            raise ValueError("flange_lengths must be positive")
        a, n = np.asarray(self.axis), np.asarray(self.start_radial)
        if not np.allclose([a @ a, n @ n, a @ n], [1, 1, 0],
                           atol=_FRAME_ATOL, rtol=0):
            raise ValueError("axis and start_radial must be perpendicular unit vectors")
        if self.unit not in _METRES_PER_UNIT:
            raise ValueError("unit must be 'm' or 'mm'")
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id must be a nonempty string")

    def radius(self, surface):
        """Return the inner or outer face radius, never a neutral-axis radius."""
        if surface not in ("inner", "outer"):
            raise ValueError("surface must be 'inner' or 'outer'")
        return self.inner_radius + (self.thickness if surface == "outer" else 0)

    def axis_segment(self):
        """Ordered finite centre-axis endpoints; not evidence of visible support."""
        return np.asarray(self.origin) + np.outer(self.span, self.axis)

    def rigidly_transformed(self, rotation, translation):
        """Apply x' = Q x + d; normals/directions get Q only, with Q in SO(3)."""
        q = np.asarray(rotation, dtype=float)
        d = _vector(translation, 3, "translation")
        if (q.shape != (3, 3) or not np.isfinite(q).all()
                or not np.allclose(q.T @ q, np.eye(3), atol=_FRAME_ATOL, rtol=0)
                or not math.isclose(float(np.linalg.det(q)), 1, abs_tol=_FRAME_ATOL,
                                    rel_tol=0)):
            raise ValueError("rotation must be a proper orthogonal 3x3 matrix")
        return replace(self, origin=q @ self.origin + d, axis=q @ self.axis,
                       start_radial=q @ self.start_radial)

    def in_units(self, unit):
        """Convert every dimensional parameter; neither geometry nor angle changes."""
        if unit not in _METRES_PER_UNIT:
            raise ValueError("unit must be 'm' or 'mm'")
        factor = _METRES_PER_UNIT[self.unit] / _METRES_PER_UNIT[unit]
        return replace(self, unit=unit, inner_radius=self.inner_radius * factor,
                       thickness=self.thickness * factor,
                       span=tuple(x * factor for x in self.span),
                       flange_lengths=tuple(x * factor for x in self.flange_lengths),
                       origin=tuple(x * factor for x in self.origin))


@dataclass(frozen=True)
class SurfaceSamples:
    """Per-patch samples; seam duplicates are retained with distinct patch labels.

    Fraction and component fields are oracle metadata. Omitting samples models
    only a stipulated observation subset, not camera occlusion or missing material.
    Arrays are ordinary caller-owned NumPy arrays, not immutable evidence records.
    """

    points: np.ndarray
    normals: np.ndarray
    patches: np.ndarray
    surfaces: np.ndarray
    component_ids: np.ndarray
    axial_fractions: np.ndarray
    patch_fractions: np.ndarray
    unit: str


def _fractions(value, name):
    result = np.asarray(value, dtype=float)
    if (result.ndim != 1 or result.size == 0 or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError(f"{name} must be nonempty 1D fractions in [0, 1]")
    return result


def _evaluate(spec, patch, surface, axial_fraction, fraction):
    a, n0 = np.asarray(spec.axis), np.asarray(spec.start_radial)
    k0 = np.cross(a, n0)
    angle = (spec.angle_rad * fraction if patch == "bend" else
             np.full_like(fraction, 0 if patch == "start" else spec.angle_rad))
    radial = np.cos(angle)[:, None] * n0 + np.sin(angle)[:, None] * k0
    axial = spec.span[0] + axial_fraction * (spec.span[1] - spec.span[0])
    points = np.asarray(spec.origin) + axial[:, None] * a + spec.radius(surface) * radial
    sign = math.copysign(1.0, spec.angle_rad)
    if patch == "start":
        points -= sign * spec.flange_lengths[0] * fraction[:, None] * k0
    elif patch == "end":
        points += sign * spec.flange_lengths[1] * fraction[:, None] * np.cross(a, radial)
    normals = radial * (1 if surface == "outer" else -1)
    if not np.isfinite(points).all():
        raise ValueError("sample coordinates exceed finite floating-point range")
    return points, normals


def sample_surfaces(spec, *, axial_fractions=(0.0, 0.5, 1.0),
                    bend_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
                    flange_fractions=(0.0, 0.5, 1.0),
                    patches=("start", "bend", "end"), surfaces=("inner", "outer")):
    """Sample explicit patches, default 66 points, at most MAX_SAMPLES points.

    Axial fractions interpolate the finite span. Bend fractions interpolate the
    signed angle. Flange fractions run from contact (0) to free edge (1).
    Fractions may be permuted/repeated; patch/surface names must be unique.
    No random noise, filesystem, fitting, state/count estimates or detector calls.
    """
    for names, allowed, label in ((patches, ("start", "bend", "end"), "patches"),
                                  (surfaces, ("inner", "outer"), "surfaces")):
        if not names or len(set(names)) != len(names) or any(x not in allowed for x in names):
            raise ValueError(f"{label} must be a nonempty unique selection from {allowed}")
    axial = _fractions(axial_fractions, "axial_fractions")
    bend = _fractions(bend_fractions, "bend_fractions")
    flange = _fractions(flange_fractions, "flange_fractions")
    fractions = {"start": flange, "bend": bend, "end": flange}
    count = len(axial) * sum(len(fractions[p]) for p in patches) * len(surfaces)
    if count > MAX_SAMPLES:
        raise ValueError(f"fixture is limited to {MAX_SAMPLES} samples")
    rows = []
    for surface in surfaces:
        for patch in patches:
            u = np.repeat(axial, len(fractions[patch]))
            f = np.tile(fractions[patch], len(axial))
            points, normals = _evaluate(spec, patch, surface, u, f)
            rows.append((points, normals, np.full(len(u), patch),
                         np.full(len(u), surface), u, f))
    points, normals, patch_names, surface_names, u, f = (
        np.concatenate(column) for column in zip(*rows)
    )
    return SurfaceSamples(points, normals, patch_names, surface_names,
                          np.full(count, spec.component_id), u, f, spec.unit)


def surface_generator(spec, *, bend_fraction=0.5, surface="outer"):
    """Finite ruling on a chosen face, not the centre axis or a qualified marker."""
    return sample_surfaces(spec, axial_fractions=(0, 1), bend_fractions=(bend_fraction,),
                           patches=("bend",), surfaces=(surface,)).points
