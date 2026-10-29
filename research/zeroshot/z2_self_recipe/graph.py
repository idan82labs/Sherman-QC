"""Z2 steps 2-3: flange adjacency graph with bend-strip evidence, then E3-style local metrology.

For each pair of flanges whose material comes close, with a dihedral of at least ``min_sweep_deg``:

* the outer skin of each flange is the one with the other flange on its inner side (a one-sided
  flange whose seen skin is the inner one gets the outer plane by moving it t outward);
* the sharp line is the intersection of the two outer planes, axis u = n_A x n_B;
* bend-strip evidence: points whose normals lie in the plane perpendicular to u and sweep between
  the two flanges' normals (outer arc, or the inner arc antiparallel to it), inside the bend
  corner, sliced along u (2 mm); the span is the union of occupied slices (gaps bridged) where
  both flanges have material near the line;
* metrology: per side, the flat material past the arc's tangent line (E3 ``metrology._flat_start``)
  is fitted per station with E3's skin-aware ``_fit_side`` (OML level, or the inner skin moved by
  t); the station sharps' median moves the line (as E3's ``sharp_line_consistency`` step); the
  reported line is the outer arc apex (measured on the arc points near the bisector, else the
  sharp moved back by Ro (sec(sweep/2) - 1)).

Imports of ``domains.production.engine.inspect`` are read-only library calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from research.zeroshot.z2_self_recipe import common  # noqa: F401
from research.zeroshot.z2_self_recipe.segment import Scan, Flange
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import Plane, robust_plane
from domains.production.engine.inspect import metrology as E3M

CFG = InspectConfig()


@dataclass
class GraphParams:
    min_sweep_deg: float = 8.0
    max_sweep_deg: float = 175.0
    near_mm: float = 6.0            # + 3 t: flanges closer than this are candidate neighbours
    slice_mm: float = 2.0
    gap_bridge_mm: float = 12.0
    arc_tol_deg: float = 20.0       # |n . u| < sin(arc_tol)
    arc_margin_deg: float = 4.0     # strictly inside the sweep
    corner_mm: float = 6.0          # + 3 t: arc points within this cross-section distance of the sharp
    contact_mm: float = 2.5         # + 1.5 t: flange material this close to the line counts as contact
    min_arc_slices: int = 3
    min_span_mm: float = 8.0         # it07 nested grouped CV picked (max_ro_t 12, min_span 8) in 4 of 5 folds
    d1_snap: bool = True             # final placement: D1 gated outer-apex snap (it07 ablation +1 TP / -1 FP)
    min_cover: float = 0.25
    local_depth_mm: float = 30.0
    nms_offset_mm: float = 3.0
    risers: bool = False            # virtual riser discovery (it06 experiment: net negative, off)
    max_ro_t: float = 12.0          # measured outer radius beyond this many t: a rolled kink / facet, not a press-brake bend
    nms_deg: float = 5.0


@dataclass
class Edge:
    a: int
    b: int
    ok: bool = False
    reason: str = ""
    sweep_deg: float = 0.0
    line_start: list | None = None
    line_end: list | None = None
    sharp_start: list | None = None
    sharp_end: list | None = None
    span_mm: float = 0.0
    arc_points: int = 0
    arc_slices: int = 0
    cover: float = 0.0
    ro_mm: float | None = None
    stations: int = 0
    score: float = 0.0
    info: dict = field(default_factory=dict)
    geom: dict = field(default_factory=dict)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _intersect(a: Plane, b: Plane):
    return E3M._intersect(a, b)


def outer_plane(scan: Scan, f: Flange, other_pts: np.ndarray) -> tuple[Plane, str]:
    """The flange's outer-skin plane (outward normal) relative to a neighbouring flange."""
    best = None
    for rid, pl in f.skins:
        s = float(np.median(pl.distance(other_pts)))
        cand = (pl, "seen_outer") if s < 0 else (Plane(-pl.n, -pl.d + scan.t, pl.rms, pl.count), "inner_moved")
        area = scan.regions[rid].area
        # prefer a skin that is the outer one itself; among them the largest
        key = (cand[1] == "seen_outer", area)
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1]


def candidate_pairs(scan: Scan, prm: GraphParams) -> dict[tuple[int, int], int]:
    P = scan.P
    pts, lab = [], []
    for f in scan.flanges:
        idx = f.idx
        # boundary-ish sample: points with an unlabelled or foreign neighbour
        nb = scan.flange_of_point[scan.nbr[idx]]
        bd = idx[(nb != f.fid).any(1)]
        # thin to one point per ~1 mm voxel
        if len(bd):
            key = np.floor(P[bd] / 1.0).astype(np.int64)
            _, first = np.unique(key, axis=0, return_index=True)
            bd = bd[np.sort(first)]
        if len(bd) > 4000:
            bd = bd[np.linspace(0, len(bd) - 1, 4000).astype(int)]
        pts.append(P[bd])
        lab.append(np.full(len(bd), f.fid))
    if not pts:
        return {}
    Q = np.vstack(pts)
    L = np.concatenate(lab)
    tree = cKDTree(Q)
    r = prm.near_mm + 3.0 * scan.t
    pairs = tree.query_pairs(r, output_type="ndarray")
    out: dict[tuple[int, int], int] = {}
    if len(pairs) == 0:
        return out
    la, lb = L[pairs[:, 0]], L[pairs[:, 1]]
    m = la != lb
    a, b = np.minimum(la[m], lb[m]), np.maximum(la[m], lb[m])
    key, cnt = np.unique(a * 100000 + b, return_counts=True)
    for k, c in zip(key, cnt):
        out[(int(k // 100000), int(k % 100000))] = int(c)
    return out


def _runs(occ: np.ndarray, gap: int) -> list[tuple[int, int]]:
    runs, start, last = [], None, None
    for i, o in enumerate(occ):
        if not o:
            continue
        if start is None:
            start = last = i
        elif i - last - 1 <= gap:
            last = i
        else:
            runs.append((start, last))
            start = last = i
    if start is not None:
        runs.append((start, last))
    return runs


def _thin(p, n=3000):
    return p if len(p) <= n else p[np.linspace(0, len(p) - 1, n).astype(int)]


def _local(p, q, r):
    """Points of p within r of q (falls back to p)."""
    d, _ = cKDTree(_thin(q)).query(_thin(p, 20000), k=1, distance_upper_bound=r)
    sel = _thin(p, 20000)[np.isfinite(d)]
    return sel if len(sel) >= 20 else p


def _mid_local(scan, f, other, prm):
    """Mid-surface points of f's material near ``other`` (within 3x the candidate reach)."""
    t = scan.t
    idx = f.idx if len(f.idx) <= 20000 else f.idx[np.linspace(0, len(f.idx) - 1, 20000).astype(int)]
    d, _ = cKDTree(_thin(scan.P[other.idx])).query(scan.P[idx], k=1, distance_upper_bound=3.0 * (prm.near_mm + 3.0 * t))
    sel = idx[np.isfinite(d)]
    if len(sel) < 20:
        sel = idx
    return scan.P[sel] - 0.5 * t * scan.Ns[sel]


def _local_centre(p, q_loc, r):
    """Median of p's material within r of the other flange's nearby material."""
    d, _ = cKDTree(_thin(q_loc)).query(_thin(p, 20000), k=1, distance_upper_bound=r)
    sel = _thin(p, 20000)[np.isfinite(d)]
    return np.median(sel if len(sel) >= 20 else p, 0)


def measure_edge(scan: Scan, fa: Flange, fb: Flange, prm: GraphParams, tree: cKDTree) -> Edge:
    P, Ns, t = scan.P, scan.Ns, scan.t
    e = Edge(fa.fid, fb.fid)
    pa, pb = P[fa.idx], P[fb.idx]
    # the material of each flange near the other one decides the outer skin and the in-plane directions
    # (a large flange's far side, e.g. around a window a tab is bent out of, must not)
    pa_loc, pb_loc = _local(pa, pb, prm.near_mm + 3.0 * t), _local(pb, pa, prm.near_mm + 3.0 * t)
    # skin choice on the other flange's mid-surface (its points moved t/2 inward): at small sweeps the other
    # flange's skins sit within a thickness of this flange's planes
    ma, mb = _mid_local(scan, fa, fb, prm), _mid_local(scan, fb, fa, prm)
    pla, ka = outer_plane(scan, fa, mb)
    plb, kb = outer_plane(scan, fb, ma)
    li = _intersect(pla, plb)
    if li is None:
        e.reason = "PARALLEL"
        return e
    lp, u = li
    # cut wall: a one-sided region whose extent along the other flange's normal is about one thickness
    for f, pts, other in ((fa, pa, plb), (fb, pb, pla)):
        if f.two_sided:
            continue
        # a cut wall stands square to its flange's skins
        if abs(float(f.normal @ other.n)) > math.sin(math.radians(20.0)):
            continue
        h = _thin(pts, 20000) @ other.n
        if float(np.quantile(h, 0.98) - np.quantile(h, 0.02)) < 1.3 * t + 0.5:
            e.reason = "CUT_WALL"
            return e
    cs = float(np.clip(pla.n @ plb.n, -1, 1))
    sweep = math.degrees(math.acos(cs))
    e.sweep_deg = sweep
    if sweep < prm.min_sweep_deg or sweep > prm.max_sweep_deg:
        e.reason = "SWEEP_OUT_OF_RANGE"
        return e
    # in-plane directions from the line into each flange
    ca, cb = _local_centre(pa, pb_loc, 3.0 * (prm.near_mm + 3.0 * t)), _local_centre(pb, pa_loc, 3.0 * (prm.near_mm + 3.0 * t))
    wa = _unit(np.cross(u, pla.n))
    if float((ca - lp) @ wa) < 0:
        wa = -wa
    wb = _unit(np.cross(u, plb.n))
    if float((cb - lp) @ wb) < 0:
        wb = -wb
    # candidate points near the line (cross-section distance)
    R = prm.corner_mm + 3.0 * t
    # gather points near the infinite line within the extent of the two flanges' projections
    za, zb = (pa - lp) @ u, (pb - lp) @ u
    zlo = max(float(np.quantile(za, 0.0)), float(np.quantile(zb, 0.0))) - 1.0
    zhi = min(float(np.quantile(za, 1.0)), float(np.quantile(zb, 1.0))) + 1.0
    if zhi - zlo < prm.min_span_mm:
        e.reason = "NO_COMMON_EXTENT"
        return e
    nseg = max(2, int(math.ceil((zhi - zlo) / max(R, 1.0))))
    cand = set()
    for k in range(nseg + 1):
        c = lp + u * (zlo + (zhi - zlo) * k / nseg)
        cand.update(tree.query_ball_point(c, r=math.hypot(R, 0.5 * (zhi - zlo) / nseg) + 0.5))
    idx = np.fromiter(cand, dtype=np.int64)
    if len(idx) == 0:
        e.reason = "NO_POINTS"
        return e
    rel = P[idx] - lp
    z = rel @ u
    cross = rel - np.outer(z, u)
    rr = np.linalg.norm(cross, axis=1)
    inz = (z >= zlo) & (z <= zhi) & (rr < R)
    idx, rel, z, rr = idx[inz], rel[inz], z[inz], rr[inz]
    fl = scan.flange_of_point[idx]
    nn = Ns[idx]
    # arc evidence: normal in the plane perpendicular to u, between the two outer normals (or their negatives)
    perp = np.abs(nn @ u) < math.sin(math.radians(prm.arc_tol_deg))
    e1 = pla.n
    e2 = _unit(np.cross(u, e1))
    if float(plb.n @ e2) < 0:
        e2 = -e2
    ang_b = math.degrees(math.atan2(float(plb.n @ e2), float(plb.n @ e1)))
    a_pt = np.degrees(np.arctan2(nn @ e2, nn @ e1))
    a_in = np.degrees(np.arctan2(-(nn @ e2), -(nn @ e1)))
    mg = prm.arc_margin_deg
    out_arc = perp & (a_pt > mg) & (a_pt < ang_b - mg)
    in_arc = perp & (a_in > mg) & (a_in < ang_b - mg)
    # corner region: not deep inside either flange
    sa, sb = rel @ wa, rel @ wb
    corner = (sa < R) & (sb < R)
    other = (fl >= 0) & (fl != fa.fid) & (fl != fb.fid)
    # strip evidence = mid-sweep normals (the arc's middle half): cut walls and corner seams carry normals near
    # either flange's direction, never the bisector
    q4 = 0.25 * ang_b
    mid = perp & ((np.abs(a_pt - 0.5 * ang_b) < q4) | (np.abs(a_in - 0.5 * ang_b) < q4))
    arc = mid & corner & ~other
    # contact: both flanges have material near the line
    cnear = prm.contact_mm + 1.5 * t
    near_a = (fl == fa.fid) & (rr < R)
    near_b = (fl == fb.fid) & (rr < R)
    nb_bins = int(math.ceil((zhi - zlo) / prm.slice_mm))
    bins = np.clip(((z - zlo) / prm.slice_mm).astype(np.int64), 0, nb_bins - 1)
    occ_arc = np.bincount(bins[arc], minlength=nb_bins) >= 2
    occ_a = np.bincount(bins[near_a], minlength=nb_bins) >= 2
    occ_b = np.bincount(bins[near_b], minlength=nb_bins) >= 2
    both = occ_a & occ_b
    ev = occ_arc & both
    e.arc_points = int(arc.sum())
    e.arc_slices = int(ev.sum())
    if e.arc_slices < prm.min_arc_slices:
        e.reason = "NO_STRIP"
        return e
    gap = int(round(prm.gap_bridge_mm / prm.slice_mm))
    runs = _runs(ev, gap)
    # keep runs; span = union of runs carrying most of the evidence
    tot = ev.sum()
    runs = [r for r in runs if ev[r[0]:r[1] + 1].sum() >= max(prm.min_arc_slices, 0.1 * tot)]
    if not runs:
        e.reason = "NO_STRIP"
        return e
    b0, b1 = runs[0][0], runs[-1][1]
    span_bins = b1 - b0 + 1
    e.cover = float(ev[b0:b1 + 1].mean())
    # refine the ends with the arc points themselves
    in_span = arc & (bins >= b0) & (bins <= b1)
    zz = np.sort(z[in_span])
    s_lo, s_hi = float(np.quantile(zz, 0.005)), float(np.quantile(zz, 0.995))
    e.span_mm = s_hi - s_lo
    if e.span_mm < prm.min_span_mm or e.cover < prm.min_cover:
        e.reason = "SHORT_OR_SPARSE"
        return e
    # ---------------- metrology (E3 style) ----------------
    # outer radius from the outer-arc apex: points near the bisector
    bis = _unit(-(pla.n + plb.n)) if np.linalg.norm(pla.n + plb.n) > 1e-9 else -pla.n
    # bisector of the normals points into the corner (toward material); the outer apex lies along -bis from the sharp
    half = 0.5 * ang_b
    near_bis = out_arc & corner & (np.abs(a_pt - half) < max(3.0, 0.2 * ang_b)) & (z >= s_lo) & (z <= s_hi)
    ro = None
    apex_off = None
    if near_bis.sum() >= 5:
        apex_off = float(np.median(rel[near_bis] @ bis))      # >= 0: apex sits inside the sharp along bis
        k = 1.0 / math.cos(math.radians(min(sweep, 170.0)) / 2.0) - 1.0
        if k > 1e-3 and apex_off > 0:
            ro = apex_off / k
    # radius from the strip's own chord as well (the mid-half of an arc of radius R spans 2 R sin(sweep/4))
    mo = mid & out_arc & corner & ~other & (z >= s_lo) & (z <= s_hi)
    ro_chord = None
    if mo.sum() >= 10:
        perp_b = _unit(np.cross(u, bis))
        c2 = rel[mo] @ perp_b
        chord = float(np.quantile(c2, 0.95) - np.quantile(c2, 0.05))
        ro_chord = chord / max(2.0 * math.sin(math.radians(sweep) / 4.0), 1e-6)
    e.info["ro_chord"] = ro_chord
    e.ro_mm = ro
    r_test = ro if ro is not None else ro_chord
    if r_test is not None and r_test > prm.max_ro_t * t:
        e.reason = "ROLLED"
        return e
    ro_use = ro if (ro is not None and 0.2 * t <= ro <= 15.0 * t + 10) else max(t, 1.0) * 2.0
    d_t = ro_use * math.tan(math.radians(min(sweep, 170.0)) / 2.0)
    lp2, st_info = _station_refine(scan, fa, fb, pla, plb, lp, u, wa, wb, d_t, s_lo, s_hi, prm)
    e.stations = st_info.get("valid", 0)
    # E3's own acceptance: a bend is measured only when both sides show flat material past the arc at enough
    # stations (metrology.measure_bend_geometry: valid >= max(3, ceil(min_station_fraction * n)), else LOW_DENSITY)
    n_st = st_info.get("n_stations", CFG.stations_min)
    if e.stations < max(3, int(math.ceil(CFG.min_station_fraction * n_st))):
        e.reason = "LOW_DENSITY"
        return e
    pla2, plb2 = st_info.get("planes", (pla, plb))
    # apex line
    a0, a1 = lp2 + u * s_lo, lp2 + u * s_hi
    e.sharp_start, e.sharp_end = a0.tolist(), a1.tolist()
    if apex_off is not None and ro is not None:
        back = apex_off                      # measured outer apex, relative to the region-plane sharp
        base = lp
    else:
        back = ro_use * (1.0 / math.cos(math.radians(min(sweep, 170.0)) / 2.0) - 1.0)
        base = lp2
    e.line_start = (base + u * s_lo + bis * back).tolist()
    e.line_end = (base + u * s_hi + bis * back).tolist()
    e.ok = True
    e.geom = {"lp": lp2, "u": u, "na": pla.n, "nb": plb.n, "ro": ro_use, "sweep": sweep,
              "dens": e.arc_points * scan.point_area / max(e.span_mm, 1e-6)}
    e.score = e.cover * min(1.0, e.span_mm / 20.0)
    e.info = {"outer_a": ka, "outer_b": kb, "apex_measured": apex_off is not None, "back_mm": back,
              "station_shift_mm": float(np.linalg.norm(lp2 - lp)), **{k: v for k, v in st_info.items() if k != "planes"}}
    return e


def _station_refine(scan, fa, fb, pla, plb, lp, u, wa, wb, d_t, zlo, zhi, prm):
    """E3 ``measure_bend_geometry`` station loop on a scan-derived pair (no CAD, no registration)."""
    P, t = scan.P, scan.t
    depth = prm.local_depth_mm
    info = {"valid": 0}
    res = {}
    for name, f, pl, w in (("a", fa, pla, wa), ("b", fb, plb, wb)):
        idx = f.idx
        rel = P[idx] - lp
        z = rel @ u
        s = rel @ w
        d = pl.distance(P[idx])
        inz = (z > zlo - 1) & (z < zhi + 1)
        start0 = d_t + 0.5
        try:
            start = E3M._flat_start(s, d, z, start0, depth, t, zlo, zhi, CFG)
        except Exception:  # noqa: BLE001
            start = start0
        strip = inz & (s > start) & (s < depth) & (np.abs(d + 0.5 * t) < 0.5 * t + 0.6)
        # re-seat the region-wide plane on the local strip (bowed / long panels)
        q = None
        if strip.sum() >= CFG.min_station_points * CFG.stations_min:
            try:
                q = E3M._reseat_plane(P[idx[strip]], pl, t, CFG)
            except Exception:  # noqa: BLE001
                q = None
        res[name] = (idx[strip], z[strip], q if q is not None else pl, start)
    ia, za, pla2, sa0 = res["a"]
    ib, zb, plb2, sb0 = res["b"]
    # E3: a joggle riser placed by its step keeps its plane at every station (model plane)
    model_a, model_b = bool(getattr(fa, "virtual", False)), bool(getattr(fb, "virtual", False))
    if model_a:
        pla2 = pla
    if model_b:
        plb2 = plb
    info["flat_start"] = (float(sa0), float(sb0))
    li = _intersect(pla2, plb2)
    if li is not None:
        p0, d0 = li
        if float(d0 @ u) < 0:
            d0 = -d0
        if float(d0 @ u) > math.cos(math.radians(3.0)):
            # project the new line's point to the old axial origin
            lp_new = p0 + float((lp - p0) @ d0) * d0
            lp, u_new = lp_new, d0
        else:
            pla2, plb2 = pla, plb
    need = CFG.min_station_points
    if (not model_a and len(ia) < need * 3) or (not model_b and len(ib) < need * 3) or (model_a and model_b):
        info["planes"] = (pla2, plb2)
        return lp, info
    zs, wst = E3M._station_layout(zlo, zhi, CFG)
    info["n_stations"] = len(zs)
    sharps = []
    for zi in zs:
        ma = np.abs(za - zi) <= 0.5 * wst
        mb = np.abs(zb - zi) <= 0.5 * wst
        if (not model_a and ma.sum() < need) or (not model_b and mb.sum() < need):
            continue
        fpa = pla2 if model_a else E3M._fit_side(P[ia[ma]], pla2, t, CFG)[0]
        fpb = plb2 if model_b else E3M._fit_side(P[ib[mb]], plb2, t, CFG)[0]
        if fpa is None or fpb is None:
            continue
        q = E3M._station_sharp(fpa, fpb, lp, u, zi)
        if q is not None:
            sharps.append(q)
    info["valid"] = len(sharps)
    info["planes"] = (pla2, plb2)
    if len(sharps) >= 3:
        vs = np.asarray(sharps)
        rel = vs - lp
        off = np.median(rel - np.outer(rel @ u, u), axis=0)
        if float(np.linalg.norm(off)) > CFG.sharp_line_consistency_mm and float(np.linalg.norm(off)) < 2.0 * t + 2.0:
            lp = lp + off
            info["station_moved_mm"] = float(np.linalg.norm(off))
    return lp, info


def build_graph(scan: Scan, prm: GraphParams | None = None) -> tuple[list[Edge], dict]:
    prm = prm or GraphParams()
    cands = candidate_pairs(scan, prm)
    tree = cKDTree(scan.P)
    edges = []
    for (a, b), c in sorted(cands.items()):
        e = measure_edge(scan, scan.flanges[a], scan.flanges[b], prm, tree)
        e.info["near_pairs"] = c
        edges.append(e)
    return edges, {"candidates": len(cands)}


def _line_metrics(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da, db = _unit(a[1] - a[0]), _unit(b[1] - b[0])
    ang = math.degrees(math.acos(min(1.0, abs(float(da @ db)))))
    md = b.mean(0) - a.mean(0)
    off = float(np.linalg.norm(md - da * float(md @ da)))
    za = sorted([0.0, float((a[1] - a[0]) @ da)])
    zb = sorted([float((b[0] - a[0]) @ da), float((b[1] - a[0]) @ da)])
    ov = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    return ang, off, ov / max(1e-9, min(za[1] - za[0], zb[1] - zb[0]))


def nms(edges: list[Edge], prm: GraphParams, drop: set[int] | None = None) -> list[Edge]:
    keep: list[Edge] = []
    drop = drop or set()
    for e in sorted([e for e in edges if e.ok and e.a not in drop and e.b not in drop], key=lambda e: -(e.score * 1000 + e.span_mm)):
        dup = False
        for k in keep:
            ang, off, ov = _line_metrics([e.line_start, e.line_end], [k.line_start, k.line_end])
            if ang < prm.nms_deg and off < prm.nms_offset_mm and ov > 0.5:
                dup = True
                break
        if not dup:
            keep.append(e)
    return keep


def suppress_facets(scan: Scan, edges: list[Edge], prm: GraphParams) -> set[int]:
    """Flanges that are facets of another pair's bend arc (a large-radius arc split into planar strips).

    X is a facet of the bend A-B when X is narrower than that arc, nearly all of X lies within the arc's reach of
    the A-B sharp line, and X's normal lies inside the A-B sweep. Every edge of a facet is dropped.
    """
    P, t = scan.P, scan.t
    facets: set[int] = set()
    ok = [e for e in edges if e.ok and e.geom]
    for x in scan.flanges:
        px = P[x.idx]
        if len(px) > 3000:
            px = px[np.linspace(0, len(px) - 1, 3000).astype(int)]
        for e in ok:
            if x.fid in (e.a, e.b):
                continue
            g = e.geom
            arc_len = (g["ro"] + t) * math.radians(g["sweep"])
            if x.width > 1.5 * arc_len + 2.0:
                continue
            u = g["u"]
            if abs(float(x.normal @ u)) > math.sin(math.radians(prm.arc_tol_deg)):
                continue
            # normal inside the sweep (either sign)
            na, nb = g["na"], g["nb"]
            ang_ab = math.degrees(math.acos(float(np.clip(na @ nb, -1, 1))))
            ok_dir = False
            for nx in (x.normal, -x.normal):
                a1 = math.degrees(math.acos(float(np.clip(nx @ na, -1, 1))))
                a2 = math.degrees(math.acos(float(np.clip(nx @ nb, -1, 1))))
                if a1 + a2 < ang_ab + 6.0 and a1 > 2.0 and a2 > 2.0:
                    ok_dir = True
            if not ok_dir:
                continue
            rel = px - g["lp"]
            cr = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
            if float(np.mean(cr < g["ro"] + 2.0 * t + 1.0)) >= 0.8:
                facets.add(x.fid)
                break
    return facets


def break_triangles(kept: list[Edge]) -> tuple[list[Edge], list[tuple[int, int]]]:
    """Box-corner seams: two flanges each bent off a common base meet at a corner whose rounding looks like a
    short bend. In a 3-cycle of bends with mutually non-parallel axes the shortest edge is dropped when it is
    short (< half the others' span) or thin (strip density < half the strongest one's), repeatedly."""
    removed = []
    edges = list(kept)
    while True:
        by = {}
        for e in edges:
            by[(min(e.a, e.b), max(e.a, e.b))] = e
        adj: dict[int, set[int]] = {}
        for a, b in by:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        worst = None
        for (a, b), e in by.items():
            for c in adj[a] & adj[b]:
                tri = [by[(min(a, b), max(a, b))], by[(min(a, c), max(a, c))], by[(min(b, c), max(b, c))]]
                us = [np.asarray(t.geom.get("u")) for t in tri if t.geom]
                if len(us) == 3 and all(abs(float(us[i] @ us[j])) < math.cos(math.radians(10)) for i, j in ((0, 1), (0, 2), (1, 2))):
                    w = min(tri, key=lambda t: t.span_mm)
                    rest = [t for t in tri if t is not w]
                    short = w.span_mm < 0.5 * min(t.span_mm for t in rest)
                    thin = w.geom["dens"] < 0.5 * max(t.geom["dens"] for t in rest)
                    if not (short or thin):
                        continue        # three real bends (e.g. an end panel bent off the base and both sides)
                    if worst is None or w.span_mm < worst.span_mm:
                        worst = w
        if worst is None:
            return edges, removed
        edges = [e for e in edges if e is not worst]
        removed.append((worst.a, worst.b))


def discover_risers(scan: Scan, prm: GraphParams | None = None) -> list[int]:
    """Short risers (joggle webs, the bottom of a narrow U) that segmentation could not hold as a flange.

    For two near-parallel flanges whose mid-planes are offset by at least t and whose material comes close, the
    unlabelled points near both are collected. Their normals' deviation from the flanges' normal peaks at the
    riser's angle; the points within 4 deg of that peak are the riser's skins (split by normal sign, one robust
    plane each). The riser becomes a virtual flange (E3 model plane): its bends to both flanges are then found
    and measured like any other edge, with stations fitted on the real side only.
    """
    from research.zeroshot.z2_self_recipe.segment import Region
    prm = prm or GraphParams()
    P, Ns, t = scan.P, scan.Ns, scan.t
    cands = candidate_pairs(scan, prm)
    made = []
    taken = np.zeros(len(P), bool)
    trees = {}

    def tree(f):
        if f.fid not in trees:
            trees[f.fid] = cKDTree(_thin(P[f.idx], 20000))
        return trees[f.fid]
    for (a, b), _c in sorted(cands.items(), key=lambda kv: -kv[1]):
        fa, fb = scan.flanges[a], scan.flanges[b]
        if getattr(fa, "virtual", False) or getattr(fb, "virtual", False):
            continue
        if abs(float(fa.normal @ fb.normal)) < math.cos(math.radians(10.0)):
            continue
        n = fa.normal
        # mid-plane offset (each flange's largest skin moved t/2 inward)
        def mid_d(f):
            rid, pl = max(f.skins, key=lambda s: scan.regions[s[0]].area)
            pl = pl if float(pl.n @ n) > 0 else pl.flipped()
            inward = 1.0 if float(scan.regions[rid].plane.n @ n) > 0 else -1.0
            return pl.d - inward * 0.5 * t
        off = abs(mid_d(fa) - mid_d(fb))
        if off < 1.0 * t:
            continue
        g = off + 3.0 * t + 6.0
        un = np.nonzero((scan.flange_of_point < 0) & ~taken)[0]
        if len(un) < 30:
            continue
        da, _ = tree(fa).query(P[un], k=1, distance_upper_bound=g)
        db, _ = tree(fb).query(P[un], k=1, distance_upper_bound=g)
        sel = un[np.isfinite(da) & np.isfinite(db)]
        if len(sel) < 30:
            continue
        th = np.degrees(np.arccos(np.clip(np.abs(Ns[sel] @ n), 0, 1)))
        hist, edges = np.histogram(th, bins=41, range=(8.0, 90.0))
        sm = np.convolve(hist, [1, 2, 1], mode="same")
        k = int(np.argmax(sm))
        thp = 0.5 * (edges[k] + edges[k + 1])
        rs = sel[np.abs(th - thp) < 4.0]
        if len(rs) < 30:
            continue
        r0 = _unit(np.median(Ns[rs] * np.sign(Ns[rs] @ Ns[rs[0]])[:, None], 0))
        skins = []
        for sg in (1.0, -1.0):
            m = rs[(Ns[rs] @ r0) * sg > 0.8]
            if len(m) >= 15:
                pl = robust_plane(P[m], iterations=3)
                if float(pl.n @ (sg * r0)) < 0:
                    pl = pl.flipped()
                skins.append((m, pl))
        if not skins:
            continue
        idx = np.concatenate([m for m, _ in skins])
        regs = []
        for m, pl in skins:
            rid = len(scan.regions)
            q = P[m] - P[m].mean(0)
            q = q - np.outer(q @ pl.n, pl.n)
            ev = np.linalg.eigvalsh(q.T @ q / max(len(q), 1))
            reg = Region(rid, m, pl, len(m) * scan.point_area, float(4 * math.sqrt(max(ev[1], 0))),
                         float(4 * math.sqrt(max(ev[2], 0))))
            scan.regions.append(reg)
            regs.append(rid)
        fid = len(scan.flanges)
        big = max(regs, key=lambda r: scan.regions[r].area)
        fl = Flange(fid, regs, idx, scan.regions[big].plane.n.copy(), [(r, scan.regions[r].plane) for r in regs],
                    float(sum(scan.regions[r].area for r in regs)), len(skins) == 2,
                    scan.regions[big].width, scan.regions[big].length, True)
        for r in regs:
            scan.regions[r].flange = fid
        scan.flanges.append(fl)
        scan.flange_of_point[idx] = fid
        taken[idx] = True
        made.append(fid)
    return made
