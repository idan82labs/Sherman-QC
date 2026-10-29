"""Scan-side cross-section evidence for one candidate line (v2, research only).

Same idea as ``broad_ab_v1.line_features`` (cross-section "arms" around the line
at stations along it), with the measurements the v1 A/B lacked:

- per-arm 2-D line fits: flange direction, fit RMS (flatness), visible flange width
  (reach, capped at :data:`RING_OUTER_MM`), point count;
- dihedral (included angle between fitted flange directions) and its stability
  along the line (IQR and the share of slices within 5 deg of the median);
- corner geometry: the distance from the line to the sharp corner of the two
  fitted flanges and the gap between that corner and the nearest surface point
  (a rounded formed bend leaves a gap; a sharp cut edge does not);
- apex side: on which side of the material the line sits. Positive values put
  the line on the convex (outer-skin) side of the fold; an inner-skin duplicate
  reads about minus one sheet thickness;
- near-line support on the full-density cloud, and the distance to the surface.

Two clouds are used: a coarse one (<= :data:`COARSE_POINTS`) for the wide rings
and a fine one (<= :data:`FINE_POINTS`) for near-line support. All outputs are
coordinate-free scalars. Heavy: run only under the shared geometry lock.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FEATURE_VERSION", "ScanClouds", "load_clouds", "line_features_v2", "STATIONS"]

FEATURE_VERSION = "jev_v2_scan_line_features_v1"
FINE_POINTS = 1_500_000
COARSE_POINTS = 200_000
STATIONS = 15
RING_INNER_MM = 4.0
RING_OUTER_MM = 40.0
HIST_BINS = 36
ARM_MIN_FRACTION = 0.08
NEAR_APEX_MM = 3.0


class ScanClouds:
    def __init__(self, pts: np.ndarray):
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(0)
        self.fine = pts
        self.fine_tree = cKDTree(pts)
        coarse = pts if len(pts) <= COARSE_POINTS else pts[rng.choice(len(pts), COARSE_POINTS, replace=False)]
        self.coarse = coarse
        self.coarse_tree = cKDTree(coarse)
        probe = pts[rng.choice(len(pts), min(5000, len(pts)), replace=False)]
        self.fine_spacing = float(np.median(self.fine_tree.query(probe, k=2)[0][:, 1]))
        self.coarse_spacing = float(np.median(self.coarse_tree.query(probe, k=2)[0][:, 1]))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.extent = float(np.linalg.norm(hi - lo))


def load_clouds(path: Path) -> ScanClouds:
    from research.jev.broad_ab_v1 import load_points

    return ScanClouds(load_points(Path(path)))


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _arm_clusters(theta: np.ndarray) -> list[np.ndarray] | str:
    """Boolean masks of angular clusters (contiguous occupied 10-deg bins, >= 8 % of ring mass)."""
    bins = ((theta + math.pi) / (2 * math.pi) * HIST_BINS).astype(int) % HIST_BINS
    hist = np.bincount(bins, minlength=HIST_BINS).astype(float)
    occupied = hist >= max(2.0, 0.02 * len(theta))
    if occupied.all():
        return "enclosed"
    start = int(np.argmin(occupied))
    clusters, current = [], []
    for k in range(HIST_BINS):
        b = (start + k) % HIST_BINS
        if occupied[b]:
            current.append(b)
        elif current:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    masks = []
    for cl in clusters:
        mask = np.isin(bins, cl)
        if mask.sum() >= ARM_MIN_FRACTION * len(theta):
            masks.append(mask)
    masks.sort(key=lambda m: -int(m.sum()))
    return masks


def _fit_arm(uv: np.ndarray) -> dict[str, float] | None:
    """Fit a 2-D line to one arm. Direction points away from the origin (the candidate line)."""
    if len(uv) < 4:
        return None
    c = uv.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((uv - c).T))
    d = v[:, 1]
    if np.dot(d, c) < 0:
        d = -d
    n = np.array([-d[1], d[0]])
    resid = (uv - c) @ n
    return {"dir": d, "point": c, "rms": float(np.sqrt(np.mean(resid ** 2))),
            "reach": float(np.linalg.norm(uv, axis=1).max()), "count": int(len(uv))}


def _intersect(p1, d1, p2, d2) -> np.ndarray | None:
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    s, _ = np.linalg.solve(a, p2 - p1)
    return p1 + s * d1


def _med(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _iqr(values: Sequence[float]) -> float | None:
    return float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) >= 2 else None


def line_features_v2(clouds: ScanClouds, start: Sequence[float], end: Sequence[float]) -> dict[str, Any]:
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    axis = (b - a) / length
    u, w = _basis(axis)
    half_slab = max(1.5, 2.0 * clouds.coarse_spacing)
    ball = math.hypot(RING_OUTER_MM, half_slab)
    near_r = max(0.75, 3.0 * clouds.fine_spacing)
    ts = np.linspace(0.04, 0.96, STATIONS)
    centers = a[None, :] + ts[:, None] * (b - a)[None, :]
    near_counts = np.array([len(x) for x in clouds.fine_tree.query_ball_point(centers, near_r)])
    surf_dist = clouds.fine_tree.query(centers, k=1)[0]
    ball_idx = clouds.coarse_tree.query_ball_point(centers, ball)
    kinds, dihedral, rms_lo, rms_hi, reach_lo, reach_hi, balance = [], [], [], [], [], [], []
    corner_offset, corner_gap, apex_side, arm_count_lo = [], [], [], []
    for k, c in enumerate(centers):
        idx = ball_idx[k]
        if not idx:
            kinds.append(0)
            continue
        d = clouds.coarse[idx] - c
        d = d[np.abs(d @ axis) <= half_slab]
        uv = np.stack([d @ u, d @ w], axis=1) if len(d) else np.zeros((0, 2))
        r = np.linalg.norm(uv, axis=1) if len(uv) else np.zeros(0)
        ring = (r >= RING_INNER_MM) & (r <= RING_OUTER_MM)
        if ring.sum() < 6:
            kinds.append(0)
            continue
        ring_uv = uv[ring]
        masks = _arm_clusters(np.arctan2(ring_uv[:, 1], ring_uv[:, 0]))
        if isinstance(masks, str):
            kinds.append(9)
            continue
        kinds.append(len(masks))
        if len(masks) == 1:
            arm = _fit_arm(ring_uv[masks[0]])
            if arm:
                rms_hi.append(arm["rms"])
                reach_hi.append(arm["reach"])
            continue
        if len(masks) != 2:
            continue
        arms = [_fit_arm(ring_uv[m]) for m in masks[:2]]
        if not all(arms):
            continue
        cosang = float(np.clip(np.dot(arms[0]["dir"], arms[1]["dir"]), -1.0, 1.0))
        dihedral.append(math.degrees(math.acos(cosang)))
        rl, rh = sorted([arms[0]["rms"], arms[1]["rms"]])
        rms_lo.append(rl)
        rms_hi.append(rh)
        el, eh = sorted([arms[0]["reach"], arms[1]["reach"]])
        reach_lo.append(el)
        reach_hi.append(eh)
        cl, ch = sorted([arms[0]["count"], arms[1]["count"]])
        balance.append(cl / ch)
        arm_count_lo.append(cl)
        corner = _intersect(arms[0]["point"], arms[0]["dir"], arms[1]["point"], arms[1]["dir"])
        if corner is not None and np.linalg.norm(corner) < 3 * RING_OUTER_MM:
            corner_offset.append(float(np.linalg.norm(corner)))
            corner_gap.append(float(np.min(np.linalg.norm(uv - corner, axis=1))) if len(uv) else float("nan"))
        bis = arms[0]["dir"] + arms[1]["dir"]
        nb = np.linalg.norm(bis)
        near = uv[r <= NEAR_APEX_MM]
        if nb > 1e-6 and len(near) >= 3:
            apex_side.append(float((near @ (bis / nb)).mean()))
    kinds_a = np.asarray(kinds)
    sup = near_counts >= 3
    run = best = 0
    for s in sup:
        run = run + 1 if s else 0
        best = max(best, run)
    dih_med = _med(dihedral)
    stable = (float(np.mean(np.abs(np.asarray(dihedral) - dih_med) <= 5.0)) if dihedral else None)
    return {
        "line_length_mm": length,
        "near_support_fraction": float(sup.mean()),
        "near_support_longest_run_fraction": best / STATIONS,
        "median_surface_distance_mm": float(np.median(surf_dist)),
        "empty_slice_fraction": float((kinds_a == 0).mean()),
        "one_flange_slice_fraction": float((kinds_a == 1).mean()),
        "two_flange_slice_fraction": float((kinds_a == 2).mean()),
        "multi_flange_slice_fraction": float(((kinds_a >= 3) & (kinds_a < 9)).mean()),
        "enclosed_slice_fraction": float((kinds_a == 9).mean()),
        "median_included_angle_deg": dih_med,
        "median_bend_angle_deg": None if dih_med is None else 180.0 - dih_med,
        "included_angle_iqr_deg": _iqr(dihedral),
        "included_angle_stable_fraction": stable,
        "median_flatter_flange_fit_rms_mm": _med(rms_lo),
        "median_rougher_flange_fit_rms_mm": _med(rms_hi),
        "median_shorter_flange_width_mm": _med(reach_lo),
        "median_longer_flange_width_mm": _med(reach_hi),
        "median_flange_balance_ratio": _med(balance),
        "median_line_to_corner_mm": _med(corner_offset),
        "median_corner_gap_mm": _med([g for g in corner_gap if math.isfinite(g)]),
        "median_apex_side_mm": _med(apex_side),
        "two_flange_slices_count": int(len(dihedral)),
    }
