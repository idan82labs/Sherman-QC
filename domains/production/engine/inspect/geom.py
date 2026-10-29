"""Small, deterministic numpy geometry helpers for the per-scan engine (E3).

Conventions: lengths in mm, angles in radians unless a name ends in ``_deg``.
4x4 homogeneous matrices are row-major and act on column vectors
(``x' = M @ [x, 1]``). Planes are ``(n, d)`` with unit ``n`` and ``n . x = d``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "unit",
    "orthonormal_frame",
    "rotation_matrix",
    "rotation_between",
    "rigid",
    "apply",
    "apply_dir",
    "inverse",
    "rotation_about_line",
    "angle_between_deg",
    "signed_sweep_deg",
    "Plane",
    "fit_plane",
    "robust_plane",
    "plane_intersection",
    "closest_point_on_line",
    "rotation_angle_deg",
    "skin_peaks",
]


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1e-15:
        raise ValueError("cannot normalise a zero or non-finite vector")
    return a / n


def orthonormal_frame(normal, hint=None) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` spanning the plane orthogonal to ``normal``; ``(x, y, normal)`` is right-handed."""
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
    return x, np.cross(n, x)


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """3x3 right-handed rotation by ``angle`` about ``axis`` (Rodrigues)."""
    k = unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) * c + s * kx + (1.0 - c) * np.outer(k, k)


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation taking unit ``a`` onto unit ``b`` (antiparallel: 180 deg about any normal)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        x, _ = orthonormal_frame(a)
        return rotation_matrix(x, math.pi)
    return rotation_matrix(v / s, math.atan2(s, c))


def rigid(r: np.ndarray, t) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def apply(m: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def apply_dir(m: np.ndarray, dirs) -> np.ndarray:
    return np.asarray(dirs, dtype=np.float64) @ m[:3, :3].T


def inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a rigid (or rigid-with-reflection) 4x4 transform."""
    r = m[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


def rotation_about_line(point, axis, angle: float) -> np.ndarray:
    r = rotation_matrix(axis, angle)
    p = np.asarray(point, dtype=np.float64)
    return rigid(r, p - r @ p)


def angle_between_deg(a, b) -> float:
    a, b = unit(a), unit(b)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)))


def signed_sweep_deg(n_parent, n_child, axis, wrap_margin_deg: float = 30.0) -> float:
    """Sweep (deg) between a bend's parent and child OML normals, signed by the side the child lies on.

    ``axis`` is the hinge axis oriented so that the CAD fold turns the child from the
    parent's flat continuation ``u`` toward ``v = axis x u``. The magnitude is the
    unsigned angle between the normals (``acos`` of their dot product, exactly as
    before); it is negative when the child turned the other way about ``axis`` (it
    lies on the other side of its parent's plane: formed the wrong way). A reading
    within ``wrap_margin_deg`` of 180 keeps its positive sign, whatever the noise in
    its sense of turn (a hem's 180 deg is no wrong-way fold).
    """
    a = np.asarray(n_parent, dtype=np.float64)
    b = np.asarray(n_child, dtype=np.float64)
    mag = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))
    turn = float(np.asarray(axis, dtype=np.float64) @ np.cross(a, b))
    if turn < 0.0 and mag < 180.0 - float(wrap_margin_deg):
        return -mag
    return mag


def rotation_angle_deg(m: np.ndarray) -> float:
    c = (float(np.trace(m[:3, :3])) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass
class Plane:
    """``n . x = d``; ``rms`` is the residual RMS of the fit, ``count`` its support."""

    n: np.ndarray
    d: float
    rms: float = 0.0
    count: int = 0
    centroid: np.ndarray | None = None

    def distance(self, pts) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) @ self.n - self.d

    def flipped(self) -> "Plane":
        return Plane(-self.n, -self.d, self.rms, self.count, self.centroid)

    def oriented(self, direction) -> "Plane":
        """The same plane with its normal on the side of ``direction``."""
        return self if float(self.n @ np.asarray(direction)) >= 0 else self.flipped()

    def offset(self, delta: float) -> "Plane":
        """The parallel plane moved by ``delta`` along ``n``."""
        c = None if self.centroid is None else self.centroid + delta * self.n
        return Plane(self.n.copy(), self.d + delta, self.rms, self.count, c)


def fit_plane(pts: np.ndarray, weights: np.ndarray | None = None) -> Plane:
    """Weighted least-squares plane (smallest principal axis)."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        raise ValueError("a plane needs at least 3 points")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = float(w.sum())
    if sw <= 0:
        raise ValueError("plane weights sum to zero")
    c = (p * w[:, None]).sum(axis=0) / sw
    q = (p - c) * np.sqrt(w)[:, None]
    cov = q.T @ q / sw
    vals, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    # deterministic sign: largest |component| positive
    if n[int(np.argmax(np.abs(n)))] < 0:
        n = -n
    d = float(n @ c)
    res = (p - c) @ n
    rms = float(math.sqrt(max(0.0, float((w * res * res).sum() / sw))))
    return Plane(n, d, rms, int(len(p)), c)


def robust_plane(pts: np.ndarray, iterations: int = 4, scale_floor: float = 0.02, cutoff: float = 3.0) -> Plane:
    """Tukey-biweight IRLS plane fit (robust to outliers, flare and duplicate skins)."""
    plane = fit_plane(pts)
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(iterations):
        r = plane.distance(p)
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) * 1.4826
        s = max(mad, scale_floor)
        u = (r - med) / (cutoff * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if float(w.sum()) < 3:
            break
        new = fit_plane(p, w)
        if float(new.n @ plane.n) < 0:
            new = new.flipped()
        plane = new
    r = plane.distance(p)
    inl = np.abs(r) < cutoff * max(scale_floor, float(np.median(np.abs(r))) * 1.4826)
    plane.rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float(np.sqrt(np.mean(r ** 2)))
    plane.count = int(inl.sum())
    return plane


def plane_intersection(a: Plane, b: Plane) -> tuple[np.ndarray, np.ndarray] | None:
    """Line ``(point, unit direction)`` where two planes meet; None when (near) parallel."""
    d = np.cross(a.n, b.n)
    s = float(np.linalg.norm(d))
    if s < 1e-6:
        return None
    d = d / s
    # point on both planes closest to the origin
    m = np.array([a.n, b.n, d])
    rhs = np.array([a.d, b.d, 0.0])
    p = np.linalg.solve(m, rhs)
    return p, d


def closest_point_on_line(point, line_point, line_dir) -> np.ndarray:
    lp = np.asarray(line_point, dtype=np.float64)
    ld = unit(line_dir)
    return lp + float((np.asarray(point) - lp) @ ld) * ld


def skin_peaks(h: np.ndarray, t: float) -> tuple[float, float | None]:
    """Skin levels of a sheet from heights ``h`` along its normal: (top, bottom).

    The densest 0.05 mm level is one skin; a second level one thickness ``t``
    away holding at least 15 % of its count is the other skin. With one skin
    seen, returns (its level, None).
    """
    hb = 0.05
    lo = float(h.min()) - 1.0
    hist = np.bincount(np.floor((h - lo) / hb).astype(np.int64))
    sm = np.convolve(hist, np.ones(5), mode="same")
    p1 = int(np.argmax(sm))
    h1 = lo + (p1 + 0.5) * hb
    c1 = float(sm[p1])
    best = None
    for sign in (1.0, -1.0):
        h2 = h1 + sign * t
        j = int(np.floor((h2 - lo) / hb))
        if j < 0 or j >= len(sm):
            continue
        win = sm[max(0, j - 3): j + 4]
        c2 = float(win.max())
        if c2 >= 0.15 * c1 and (best is None or c2 > best[1]):
            jj = max(0, j - 3) + int(np.argmax(win))
            best = (lo + (jj + 0.5) * hb, c2)
    if best is None:
        return h1, None
    top, bottom = (h1, best[0]) if best[0] < h1 else (best[0], h1)
    return top, bottom
