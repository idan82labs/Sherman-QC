"""Pure numpy geometry helpers for the onboarding compile (no OCP, no IO).

Conventions: lengths in mm, angles in radians unless a name ends in ``_deg``.
4x4 homogeneous matrices are row-major ``numpy`` arrays applied to column
vectors (``x' = M @ [x, 1]``). Every helper is deterministic.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "unit",
    "orthonormal_frame",
    "rotation_matrix",
    "rotation_about_line",
    "translation",
    "identity4",
    "apply4",
    "apply_dir4",
    "compose",
    "rigid_inverse",
    "line_distance_parallel",
    "signed_angle_about",
    "polygon_area_2d",
    "station_crossings",
    "canonical_direction",
    "mat4_tuple",
    "vec3_tuple",
    "rotation_angle_deg",
    "REGULAR_SWEEP_MAX_DEG",
    "oriented_bend_axis",
    "bend_bisector",
    "wrap_bend_angle",
]

#: Below this sweep a bend's two end radials ``r_a``, ``r_b`` are far from
#: antiparallel, so ``unit(r_a + r_b)`` is its well-conditioned bisector and the
#: in-plane outward directions follow from the radials alone. At and above it
#: (near-reverse hems up to 180 deg, reflex teardrops beyond) the bisector and
#: the orientation come from the material's angular middle instead.
REGULAR_SWEEP_MAX_DEG = 179.0


def unit(v: Sequence[float] | np.ndarray) -> np.ndarray:
    """Unit vector of ``v``; raises ``ValueError`` for a (near) zero vector."""
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1e-15:
        raise ValueError("cannot normalise a zero or non-finite vector")
    return a / n


def orthonormal_frame(normal: np.ndarray, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors ``(x, y)`` spanning the plane orthogonal to ``normal``.

    ``x`` is ``hint`` projected into the plane when given and not parallel to
    ``normal``; otherwise the world axis least aligned with ``normal``.
    ``y = normal x x`` so ``(x, y, normal)`` is right-handed.
    """
    n = unit(normal)
    x = None
    if hint is not None:
        h = np.asarray(hint, dtype=np.float64)
        h = h - float(h @ n) * n
        if float(np.linalg.norm(h)) > 1e-9:
            x = unit(h)
    if x is None:
        axis = int(np.argmin(np.abs(n)))
        e = np.zeros(3)
        e[axis] = 1.0
        x = unit(e - float(e @ n) * n)
    y = np.cross(n, x)
    return x, y


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 right-handed rotation by ``angle`` (rad) about unit ``axis`` (Rodrigues)."""
    k = unit(axis)
    x, y, z = k
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def identity4() -> np.ndarray:
    return np.eye(4)


def translation(offset: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(offset, dtype=np.float64)
    return m


def rotation_about_line(point: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation by ``angle`` (rad, right-hand rule) about the line ``point + s * axis``."""
    r = rotation_matrix(axis, angle)
    p = np.asarray(point, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = p - r @ p
    return m


def compose(*mats: np.ndarray) -> np.ndarray:
    """``mats[0] @ mats[1] @ ...`` (the last matrix is applied first)."""
    out = np.eye(4)
    for m in mats:
        out = out @ m
    return out


def apply4(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points (or a single (3,) point)."""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim == 1:
        return m[:3, :3] @ p + m[:3, 3]
    return p @ m[:3, :3].T + m[:3, 3]


def apply_dir4(m: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Apply only the linear part of a 4x4 transform to direction vectors."""
    d = np.asarray(dirs, dtype=np.float64)
    if d.ndim == 1:
        return m[:3, :3] @ d
    return d @ m[:3, :3].T


def rigid_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a rigid (or mirror) 4x4 transform with orthonormal linear part."""
    out = np.eye(4)
    r = m[:3, :3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


def rotation_angle_deg(m: np.ndarray) -> float:
    """Rotation angle (deg) of the proper part of a 3x3/4x4 orthonormal matrix."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    if np.linalg.det(r) < 0:
        r = -r
    c = (float(np.trace(r)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def line_distance_parallel(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray) -> float:
    """Distance from point ``p2`` to the line ``p1 + s * d1`` (``d1`` unit)."""
    v = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    return float(np.linalg.norm(v - float(v @ d1) * d1))


def signed_angle_about(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (rad, in (-pi, pi]) rotating unit ``a`` onto unit ``b`` about ``axis``."""
    k = unit(axis)
    return math.atan2(float(np.cross(a, b) @ k), float(a @ b))


def polygon_area_2d(poly: np.ndarray) -> float:
    """Signed area of a closed 2D polygon (first point not repeated); CCW positive."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def station_crossings(
    seg_start: np.ndarray, seg_end: np.ndarray, u: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crossings of 2D segments with the line ``x = u``.

    ``seg_start``/``seg_end`` are (S,2) arrays in (u, w) coordinates. Returns
    ``(w_values, segment_indices)`` for segments that straddle ``u`` using the
    half-open rule ``min(u0,u1) <= u < max(u0,u1)``, so a vertex exactly on the
    station line is counted once (even-odd consistent).
    """
    u0, u1 = seg_start[:, 0], seg_end[:, 0]
    lo, hi = np.minimum(u0, u1), np.maximum(u0, u1)
    mask = (lo <= u) & (u < hi)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return np.zeros(0), idx
    a, b = seg_start[idx], seg_end[idx]
    frac = (u - a[:, 0]) / (b[:, 0] - a[:, 0])
    w = a[:, 1] + frac * (b[:, 1] - a[:, 1])
    return w, idx


def oriented_bend_axis(r_start: np.ndarray, axis: np.ndarray, mid_radial: np.ndarray) -> np.ndarray:
    """``axis`` signed so that a positive right-hand rotation of ``r_start`` sweeps into the material.

    ``r_start`` is the unit radial from the cylinder axis to one tangent end of
    a bend, ``mid_radial`` any unit radial inside the bend's angular range (for
    example toward the centroid of its cylinder faces). Valid for every sweep in
    (0, 360) deg, including exactly 180 deg where ``r_start x r_end`` vanishes.
    """
    k = unit(axis)
    s = float(np.cross(np.asarray(r_start, dtype=np.float64), np.asarray(mid_radial, dtype=np.float64)) @ k)
    if abs(s) < 1e-9:
        raise ValueError("the material mid radial is parallel to the bend's end radial")
    return k if s > 0 else -k


def bend_bisector(r_a: np.ndarray, r_b: np.ndarray, axis: np.ndarray, sweep_deg: float,
                  mid_radial: np.ndarray | None = None) -> np.ndarray:
    """Unit radial of a bend's angular middle (its apex direction) at the final pose.

    Below :data:`REGULAR_SWEEP_MAX_DEG` this is ``unit(r_a + r_b)``. At and above
    it the end radials are (nearly) antiparallel or the sweep is reflex, where
    ``r_a + r_b`` vanishes or points into the hem's eye; the bisector is then
    ``r_a`` rotated by ``sweep / 2`` toward the material (``mid_radial``).
    """
    if sweep_deg < REGULAR_SWEEP_MAX_DEG:
        return unit(np.asarray(r_a, dtype=np.float64) + np.asarray(r_b, dtype=np.float64))
    if mid_radial is None:
        raise ValueError("a near-reverse or reflex bend needs the radial of its material's angular middle")
    a = oriented_bend_axis(r_a, axis, mid_radial)
    return unit(rotation_matrix(a, math.radians(sweep_deg) / 2.0) @ np.asarray(r_a, dtype=np.float64))


def wrap_bend_angle(phi: np.ndarray, sweep_rad: float) -> np.ndarray:
    """Map angles from ``atan2`` (in (-pi, pi]) onto a bend's range [0, sweep].

    An angle below ``sweep / 2 - pi`` lies closer to the far end of the bend
    going round the long way and gets ``+ 2 pi``. Material points of a bend
    below 180 deg are never wrapped (they sit in [0, sweep] up to rounding),
    so the map is the identity there; it only matters for sweeps >= 180 deg.
    """
    p = np.asarray(phi, dtype=np.float64)
    return np.where(p < 0.5 * float(sweep_rad) - math.pi, p + 2.0 * math.pi, p)


def canonical_direction(d: np.ndarray) -> np.ndarray:
    """Sign-normalised unit direction: the dominant component is positive."""
    v = unit(d)
    k = int(np.argmax(np.abs(v)))
    return -v if v[k] < 0 else v


def vec3_tuple(v: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Plain-float 3-tuple, -0.0 folded to 0.0 (contract ``Vec3``)."""
    a = np.asarray(v, dtype=np.float64)
    out = []
    for c in a[:3]:
        f = float(c)
        if not math.isfinite(f):
            raise ValueError("non-finite coordinate")
        out.append(0.0 if f == 0.0 else f)
    return (out[0], out[1], out[2])


def mat4_tuple(m: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    """Row-major nested tuple of a 4x4 matrix (contract ``Mat4``), exact bottom row."""
    a = np.asarray(m, dtype=np.float64)
    rows = []
    for i in range(3):
        rows.append(tuple(0.0 if float(v) == 0.0 else float(v) for v in a[i]))
    rows.append((0.0, 0.0, 0.0, 1.0))
    return tuple(rows)  # type: ignore[return-value]
