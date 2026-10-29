"""Bend line from two labelled planar patches.

Locus definitions (both reported):
- ``sharp``: intersection line of the two fitted skin planes (virtual sharp;
  for outer-skin patches this is the OML sharp, off the material).
- ``apex``: the sharp translated along the corner bisector onto the scanned
  bend surface (visible-surface apex). Default label locus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segment import fit_plane


@dataclass
class LineParams:
    margin_mm: float = 3.0            # D = (Ri + t) * max(1, tan(bend/2)) + margin
    fit_extra_mm: float = 20.0        # planes refit on points within D + fit_extra
    percentile: float = 1.0           # robust extent percentile (uniform-corrected)
    min_support: int = 30
    min_length_mm: float = 2.0
    expand_steps: int = 4
    apex_lateral_mm: float = 0.6
    min_parallel_deg: float = 2.0
    fold_support_min_points: int = 3  # override validation: points per half-space per station
    fold_support_min_frac: float = 0.8  # override validation: supported station fraction


APEX_MODES = ("envelope", "station_median")


class BendLineError(ValueError):
    pass


def _unit(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise BendLineError("degenerate direction")
    return v / norm


def plane_intersection(n1, c1, n2, c2, near):
    d = np.cross(n1, n2)
    sin = np.linalg.norm(d)
    if sin < 1e-9:
        raise BendLineError("planes are parallel")
    d = d / sin
    A = np.stack([n1, n2])
    b = np.array([n1 @ c1, n2 @ c2])
    p0 = near + A.T @ np.linalg.solve(A @ A.T, b - A @ near)
    return p0, d


def _canonical(d):
    return d if d[int(np.argmax(np.abs(d)))] > 0 else -d


def _perp(points, p0, d):
    rel = points - p0
    s = rel @ d
    w = rel - np.outer(s, d)
    return s, w


def _side_dir(points, p0, d, n):
    _, w = _perp(points, p0, d)
    w = w - np.outer(w @ n, n)
    lengths = np.linalg.norm(w, axis=1)
    good = lengths > 1e-6
    if not good.any():
        raise BendLineError("patch has no points off the bend line")
    return _unit((w[good] / lengths[good, None]).mean(axis=0))


def robust_extent(s, pct):
    if len(s) == 0:
        return None
    lo, hi = np.percentile(s, [pct, 100.0 - pct])
    pad = (hi - lo) * (pct / 100.0) / max(1e-9, 1.0 - 2 * pct / 100.0)
    return max(float(s.min()), lo - pad), min(float(s.max()), hi + pad)


def zone_radius(ri, t, bend_deg, margin):
    half = np.radians(min(max(bend_deg, 0.0), 170.0)) / 2.0
    return (ri + t) * max(1.0, float(np.tan(half))) + margin


def compute_bend(all_points, side_a, side_b, *, ri_mm, thickness_mm, params=None,
                 outward_a=None, outward_b=None, extent_t=None, extent_xyz=None,
                 apex_mode="envelope"):
    """Compute one bend line. ``side_a``/``side_b`` are (K,3) raw points of each patch.

    ``outward_a``/``outward_b`` are mean outward scan normals (oriented scans only),
    used to classify the pair as outer-skin / inner-skin / mixed.

    ``extent_t=(t0, t1)`` replaces the automatic extent: the line runs from
    ``line_origin + t0 * direction`` to ``line_origin + t1 * direction`` (both
    reported in the result; ``line_origin`` is the foot of the perpendicular
    from the fit-point centroid onto the plane-intersection line, ``direction``
    is the canonical unit direction).  ``extent_xyz=(P0, P1)`` does the same
    with two 3D points projected onto the line.  The plane fit never depends
    on the extent, so ``line_origin``/``direction`` equal those of an
    automatic run with the same sides and margin.  Overrides are NOT validated
    here; see :func:`fold_support`.
    """
    if extent_t is not None and extent_xyz is not None:
        raise BendLineError("give extent_t or extent_xyz, not both")
    params = params or LineParams()
    if len(side_a) < params.min_support or len(side_b) < params.min_support:
        raise BendLineError("a patch has too few points")
    na, ca, _, _ = fit_plane(side_a)
    nb, cb, _, _ = fit_plane(side_b)
    angle_between = np.degrees(np.arccos(min(1.0, abs(float(na @ nb)))))
    if angle_between < params.min_parallel_deg:
        raise BendLineError(f"patches are parallel within {angle_between:.2f} deg")
    near = (ca + cb) / 2.0
    p0, d = plane_intersection(na, ca, nb, cb, near)
    d = _canonical(d)
    ua = _side_dir(side_a, p0, d, na)
    ub = _side_dir(side_b, p0, d, nb)
    included = float(np.degrees(np.arccos(np.clip(ua @ ub, -1.0, 1.0))))
    radius = zone_radius(ri_mm, thickness_mm, 180.0 - included, params.margin_mm)
    rms_a = rms_b = None
    for _ in range(2):
        fit_r = radius + params.fit_extra_mm
        _, wa = _perp(side_a, p0, d)
        _, wb = _perp(side_b, p0, d)
        sel_a = np.linalg.norm(wa, axis=1) <= fit_r
        sel_b = np.linalg.norm(wb, axis=1) <= fit_r
        pa = side_a[sel_a] if sel_a.sum() >= 50 else side_a
        pb = side_b[sel_b] if sel_b.sum() >= 50 else side_b
        na, ca, rms_a, _ = fit_plane(pa)
        nb, cb, rms_b, _ = fit_plane(pb)
        near = np.concatenate([pa, pb]).mean(axis=0)
        p0, d = plane_intersection(na, ca, nb, cb, near)
        d = _canonical(d)
        ua = _side_dir(pa, p0, d, na)
        ub = _side_dir(pb, p0, d, nb)
        included = float(np.degrees(np.arccos(np.clip(ua @ ub, -1.0, 1.0))))
        radius = zone_radius(ri_mm, thickness_mm, 180.0 - included, params.margin_mm)
    bend_deg = 180.0 - included

    used_radius = radius
    extent = None
    support = (0, 0)
    first_supported = None  # (used_radius, sa, za, sb, zb, support) at the first radius with support
    for step in range(params.expand_steps + 1):
        used_radius = radius * (1.5 ** step)
        sa, wa = _perp(side_a, p0, d)
        sb, wb = _perp(side_b, p0, d)
        za = (np.linalg.norm(wa, axis=1) <= used_radius) & (wa @ ua > 0)
        zb = (np.linalg.norm(wb, axis=1) <= used_radius) & (wb @ ub > 0)
        support = (int(za.sum()), int(zb.sum()))
        if min(support) < params.min_support:
            continue
        if first_supported is None:
            first_supported = (used_radius, sa, za, sb, zb, support)
        ea = robust_extent(sa[za], params.percentile)
        eb = robust_extent(sb[zb], params.percentile)
        lo, hi = max(ea[0], eb[0]), min(ea[1], eb[1])
        if hi - lo >= params.min_length_mm:
            extent = (lo, hi)
            break
    override = None
    if extent_xyz is not None:
        pts = np.asarray(extent_xyz, dtype=float)
        if pts.shape != (2, 3) or not np.isfinite(pts).all():
            raise BendLineError("extent_xyz must be two finite [x, y, z] points")
        override = tuple(sorted(float((q - p0) @ d) for q in pts))
    elif extent_t is not None:
        override = (float(extent_t[0]), float(extent_t[1]))
        if not (np.isfinite(override).all() and override[1] > override[0]):
            raise BendLineError("extent_t must be [t0, t1] with t1 > t0")
    auto_extent = extent
    if override is not None:
        if extent is None:
            if first_supported is None:
                raise BendLineError(
                    f"no support within {used_radius:.1f} mm of the line "
                    f"(support {support[0]}/{support[1]}); an extent override cannot fix this")
            used_radius, sa, za, sb, zb, support = first_supported
        if override[1] - override[0] < params.min_length_mm:
            raise BendLineError(f"override extent shorter than {params.min_length_mm} mm")
        extent = override
    if extent is None:
        raise BendLineError(
            f"no overlapping support within {used_radius:.1f} mm of the line "
            f"(support {support[0]}/{support[1]})")
    lo, hi = extent
    start = p0 + lo * d
    end = p0 + hi * d
    bins = np.arange(lo, hi + 1.0, 1.0)
    occ_a = np.histogram(sa[za], bins=bins)[0] > 0 if len(bins) > 1 else np.array([True])
    occ_b = np.histogram(sb[zb], bins=bins)[0] > 0 if len(bins) > 1 else np.array([True])
    coverage = float((occ_a & occ_b).mean())

    # apex: nearest scanned surface along the corner bisector
    bis = _unit(ua + ub)
    s_all, w_all = _perp(all_points, p0, d)
    inside = (s_all >= lo + 0.5) & (s_all <= hi - 0.5)
    h = w_all @ bis
    lateral = np.linalg.norm(w_all - np.outer(h, bis), axis=1)
    max_h = (ri_mm + thickness_mm) / max(np.cos(np.radians(bend_deg) / 2.0), 0.2) + thickness_mm + 2.0
    apex_h = None
    apex_n = 0
    for lat in (params.apex_lateral_mm, 2 * params.apex_lateral_mm, 4 * params.apex_lateral_mm):
        cand = inside & (lateral <= lat) & (h >= -1.0) & (h <= max_h)
        if cand.sum() < 10:
            continue
        hv = h[cand]
        if apex_mode == "station_median":
            sv = s_all[cand]
            per_station = []
            for a in np.arange(lo + 0.5, hi - 0.5, 1.0):
                m = (sv >= a) & (sv < a + 1.0)
                if m.sum() < 3:
                    continue
                q = hv[m]
                per_station.append(float(np.median(q[q <= q.min() + 0.3])))
            if len(per_station) >= 3:
                apex_h = float(np.median(per_station))
                apex_n = len(per_station)
                break
        floor = np.percentile(hv, 2.0)
        near_surface = hv[hv <= floor + max(0.3, 0.3 * thickness_mm)]
        apex_h = float(np.median(near_surface))
        apex_n = int(len(near_surface))
        break

    skin = "unknown"
    if outward_a is not None and outward_b is not None:
        ma = np.sign(na @ ub) * na  # plane-A normal pointing toward flange B side
        mb = np.sign(nb @ ua) * nb
        side_a_cls = "outer" if float(outward_a @ ma) < 0 else "inner"
        side_b_cls = "outer" if float(outward_b @ mb) < 0 else "inner"
        skin = side_a_cls if side_a_cls == side_b_cls else "mixed"
    half = np.radians(bend_deg) / 2.0
    sec_minus_1 = 1.0 / max(np.cos(half), 1e-6) - 1.0
    result = {
        "sharp": {"start": start.tolist(), "end": end.tolist()},
        "apex": None,
        "direction": d.tolist(),
        "bisector": bis.tolist(),
        "side_dirs": {"plane_a": ua.tolist(), "plane_b": ub.tolist()},
        "included_angle_deg": included,
        "bend_angle_deg": bend_deg,
        "dihedral_normals_deg": float(np.degrees(np.arccos(np.clip(abs(na @ nb), -1, 1)))),
        "length_mm": float(hi - lo),
        "zone_radius_mm": float(radius),
        "zone_radius_used_mm": float(used_radius),
        "support_points": {"a": support[0], "b": support[1]},
        "coverage": coverage,
        "plane_a": {"normal": na.tolist(), "point": ca.tolist(), "rms_mm": rms_a},
        "plane_b": {"normal": nb.tolist(), "point": cb.tolist(), "rms_mm": rms_b},
        "skin": skin,
        "apex_offset_mm": apex_h,
        "apex_points": apex_n,
        "expected_apex_offset_mm": {
            "outer": (ri_mm + thickness_mm) * sec_minus_1,
            "inner": ri_mm * sec_minus_1,
        },
        "line_origin": p0.tolist(),
        "t_range": [float(lo), float(hi)],
        "t_range_auto": None if auto_extent is None else [float(v) for v in auto_extent],
        "extent_source": "auto" if override is None else "override",
    }
    if apex_mode != "envelope":
        result["apex_mode"] = apex_mode
    if apex_h is not None:
        result["apex"] = {"start": (start + apex_h * bis).tolist(),
                          "end": (end + apex_h * bis).tolist()}
    return result


def fold_support(points, origin, direction, side_dir_a, side_dir_b, t0, t1, *, reach_mm,
                 min_points=3, half_window_mm=0.5, arc_band_mm=0.5):
    """Evidence that a formed fold runs along ``[t0, t1]`` of a line.

    The line is ``origin + t * direction`` (``direction`` unit).  Stations are
    spaced ~1 mm along ``[t0, t1]`` (``n = max(1, round(t1 - t0))`` equal
    steps, centre of each step).  Around each station, points with
    ``|t - station| <= half_window_mm`` and perpendicular distance
    ``<= reach_mm`` from the line are split by the bisector plane (the plane
    through the line and the corner bisector) into the flange-A half-space
    (``w . e > 0``) and the flange-B half-space (``w . e < 0``), with
    ``e = unit(side_dir_a - side_dir_b)`` and ``w`` the perpendicular offset
    from the line.  A station is supported when both half-spaces hold
    ``>= min_points`` points, or when the fold-arc band ``|w . e| <=
    arc_band_mm`` holds ``>= min_points`` points that straddle the bisector
    plane (a lone flange edge reaching the bisector plane from one side does
    not count).
    Returns ``frac``, ``n_stations`` and the unsupported ``gaps`` as
    ``[t_start, t_end]`` intervals.
    """
    d = _unit(np.asarray(direction, dtype=float))
    e = np.asarray(side_dir_a, dtype=float) - np.asarray(side_dir_b, dtype=float)
    e = e - (e @ d) * d
    e = _unit(e)
    t0, t1 = float(t0), float(t1)
    n = max(1, int(round(t1 - t0)))
    step = (t1 - t0) / n
    centres = t0 + (np.arange(n) + 0.5) * step
    s, w = _perp(np.asarray(points, dtype=float), np.asarray(origin, dtype=float), d)
    near = (np.linalg.norm(w, axis=1) <= reach_mm) & (s >= t0 - half_window_mm) & \
        (s <= t1 + half_window_mm)
    s, h = s[near], w[near] @ e
    order = np.argsort(s)
    s, h = s[order], h[order]
    supported = np.zeros(n, dtype=bool)
    for k, c in enumerate(centres):
        i0, i1 = np.searchsorted(s, [c - half_window_mm, c + half_window_mm + 1e-12])
        hk = h[i0:i1]
        n_a = int((hk > 0).sum())
        n_b = int((hk < 0).sum())
        band = hk[np.abs(hk) <= arc_band_mm]
        arc_ok = len(band) >= min_points and (band > 0).any() and (band < 0).any()
        supported[k] = (n_a >= min_points and n_b >= min_points) or arc_ok
    gaps = []
    k = 0
    while k < n:
        if supported[k]:
            k += 1
            continue
        j = k
        while j < n and not supported[j]:
            j += 1
        gaps.append([round(t0 + k * step, 2), round(t0 + j * step, 2)])
        k = j
    return {"frac": float(supported.mean()), "n_stations": n, "gaps": gaps}
