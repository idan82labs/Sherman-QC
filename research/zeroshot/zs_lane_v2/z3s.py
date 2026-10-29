"""Z3 axis-family slicing: a scan-only birth channel for press-brake bend lines (research only).

zs_lane_v2 vendored copy of research/zeroshot/z3_slicing/slicer.py (it7, frozen with zs_lane_v1; that file is never
edited). v2 changes, all off by default so that default Params reproduce v1 exactly:
  * ``peak_min_area_rel`` > 0: the absolute Gauss-map peak floor becomes scale-relative,
    min(peak_min_area_mm2, peak_min_area_rel * total area) (small / coarse parts, TEST-B class 3).
  * the total mesh area is reported (``area_total``).
  * ``fallback_frac_only``: when the default floors yield no axis family at all, the Gauss-map peaks are searched
    again with the fractional floor only (peak_min_area_frac * total area, no absolute mm^2 floor), so small or
    coarse parts still get families. Scans with at least one family are unchanged.

Pipeline (no CAD, no part-specific logic; every threshold is in mm, degrees or a multiple of t):
  1. mesh -> area-weighted face normals -> dominant flange normal peaks on the projective Gauss sphere;
     axis families = clustered cross products of peak pairs, scored with great-circle (bend-zone) support.
  2. per family u: exact mesh/plane sections every ``step`` mm (planes perpendicular to u), chained by
     shared mesh edges into 2-D polylines that carry the outward face normal of every segment.
  3. per polyline: Douglas-Peucker + collinear merge -> straight runs ("flats"); two consecutive flats
     joined by a short arc = a corner. Included angle from the two flat directions, outer-skin apex =
     intersection of the polyline with the corner bisector. Skin from the outward normals: outer when
     the flats' outward normals point away from the corner interior; an inner-skin corner is moved
     t along -bisector (concentric arcs) to its outer-skin equivalent.
  4. corners tracked across slices (gap bridging), TLS 3-D line through the apices, span = slice support.
  5. per candidate: bend-strip (cylinder) evidence = fraction of 1 mm stations along the line whose
     surface normals sweep the arc between the two flange normals.
Output lines are in the raw scan frame on the outer-skin apex locus.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- parameters

@dataclass
class Params:
    step_mm: float = 2.0             # slice spacing
    dp_eps_mm: float = 0.15          # Douglas-Peucker tolerance (scaled up with t below)
    merge_deg: float = 4.0           # merge consecutive DP segments turning less than this
    min_flat_mm: float = 2.0         # minimum straight run that counts as a flange in a slice
    max_arc_mm: float = 25.0         # max polyline length between two flats of one corner
    min_bend_deg: float = 6.0        # bend angle (180 - included) floor at birth
    max_bend_deg: float = 165.0      # beyond this (hem / sheet edge U-turn) not a corner
    flat_rms_mm: float = 0.25        # flat straightness (rms of TLS residual)
    link_pos_mm: float = 1.5         # tracking: apex move per slice
    link_ang_deg: float = 7.0        # tracking: included-angle change
    link_bis_deg: float = 25.0       # tracking: bisector change
    bridge_mm: float = 4.5           # bridge gaps in a track up to this length
    min_track_slices: int = 2        # minimum supported slices per track
    max_axis_dev_deg: float = 15.0   # fitted line vs family axis
    peak_sep_deg: float = 5.0        # normal peaks NMS separation
    peak_win_deg: float = 2.5        # normal peak window
    peak_min_area_frac: float = 0.0005
    peak_min_area_mm2: float = 20.0
    peak_min_area_rel: float = 0.0   # v2: > 0 makes the absolute floor scale-relative
    fallback_frac_only: bool = False  # v2: zero families -> retry with the fractional floor only
    max_families: int = 20
    family_merge_deg: float = 3.0
    dedupe_pos_mm: float = 1.2       # per-slice outer/inner duplicate merge


# ----------------------------------------------------------------------------- mesh

@dataclass
class Mesh:
    V: np.ndarray
    F: np.ndarray
    fn: np.ndarray       # outward unit face normals (winding)
    fa: np.ndarray       # face areas
    vn: np.ndarray       # area-weighted vertex normals
    _gauss: Any = None


def load_mesh(path: str) -> Mesh:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    if len(F) == 0:
        return mesh_from_point_cloud(str(path))
    return mesh_from_arrays(V, F)


def mesh_from_point_cloud(path: str) -> Mesh:
    """Point-cloud scans (no faces): kNN normals, tangent-plane orientation, ball-pivoting surface."""
    import open3d as o3d

    pc = o3d.io.read_point_cloud(path)
    P = np.asarray(pc.points)
    if len(P) < 100:
        raise ValueError(f"{path}: too few points")
    sp = float(np.median(cKDTree(P).query(P[:: max(1, len(P) // 5000)], k=2)[0][:, 1]))
    if len(P) > 200000:
        pc = pc.voxel_down_sample(max(1.5 * sp, 0.4))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    pc.orient_normals_consistent_tangent_plane(12)
    r = float(np.median(pc.compute_nearest_neighbor_distance()))
    m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector([1.5 * r, 3 * r, 6 * r]))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    if len(F) == 0:
        raise ValueError(f"{path}: surface reconstruction failed")
    return mesh_from_arrays(V, F)


def mesh_from_arrays(V: np.ndarray, F: np.ndarray) -> Mesh:
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nr = np.linalg.norm(cr, axis=1)
    keep = nr > 1e-12
    F, cr, nr = F[keep], cr[keep], nr[keep]
    fn = cr / nr[:, None]
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], cr)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-300)[:, None]
    return Mesh(V, F, fn, 0.5 * nr, vn)


def estimate_thickness(mesh: Mesh, n_samples: int = 20000, seed: int = 0) -> dict[str, float]:
    """Cast rays from face centroids along -n; the opposite skin (antiparallel normal) is hit at ~t."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    p = mesh.fa / mesh.fa.sum()
    idx = rng.choice(len(mesh.F), size=min(n_samples, len(mesh.F)), replace=False, p=p)
    c = mesh.V[mesh.F[idx]].mean(1)
    d = -mesh.fn[idx]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.V.astype(np.float32)), o3d.core.Tensor(mesh.F.astype(np.uint32)))
    rays = np.concatenate([c + d * 1e-3, d], 1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    th = ans["t_hit"].numpy().astype(np.float64)
    pid = ans["primitive_ids"].numpy().astype(np.int64)
    ok = np.isfinite(th) & (pid >= 0) & (pid < len(mesh.F))
    anti = np.zeros(len(th), bool)
    anti[ok] = np.einsum("ij,ij->i", mesh.fn[pid[ok]], mesh.fn[idx[ok]]) < -0.9
    good = ok & anti & (th > 0.2) & (th < 12.0)
    if good.sum() < 50:
        return {"t_mm": 1.5, "confidence": 0.0, "two_sided": float(good.mean())}
    h, edges = np.histogram(th[good], bins=np.arange(0.2, 12.0, 0.02))
    hs = np.convolve(h, np.ones(5) / 5, mode="same")
    k = int(np.argmax(hs))
    mode = 0.5 * (edges[k] + edges[k + 1])
    near = th[good][np.abs(th[good] - mode) < 0.15 * mode + 0.05]
    t = float(np.median(near)) if len(near) else float(mode)
    return {"t_mm": t, "confidence": float(len(near) / good.sum()), "two_sided": float(good.mean())}


# ----------------------------------------------------------------------------- axis families

def _fib_hemisphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = i / n                      # z in (0,1): upper hemisphere
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def gauss_samples(mesh: Mesh, n_max: int = 150000, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed surface normals for the Gauss map: vertex normals averaged over k nearest vertices
    (scanner noise on sub-mm triangles spreads raw face normals over several degrees), with vertex areas."""
    if getattr(mesh, "_gauss", None) is not None:
        return mesh._gauss
    va = np.zeros(len(mesh.V))
    for j in range(3):
        np.add.at(va, mesh.F[:, j], mesh.fa / 3.0)
    rng = np.random.default_rng(1)
    idx = np.arange(len(mesh.V)) if len(mesh.V) <= n_max else rng.choice(len(mesh.V), n_max, replace=False)
    tree = cKDTree(mesh.V)
    _, nb = tree.query(mesh.V[idx], k=min(k, len(mesh.V)))
    ns = mesh.vn[nb].sum(1)
    ns /= np.maximum(np.linalg.norm(ns, axis=1), 1e-12)[:, None]
    w = va[idx] * (va.sum() / max(va[idx].sum(), 1e-12))
    mesh._gauss = (ns, w)
    return mesh._gauss


def normal_peaks(mesh: Mesh, prm: Params) -> list[dict[str, Any]]:
    """Dominant flange normals (projective: n ~ -n) by kernel density on a hemisphere grid + NMS."""
    n, a = gauss_samples(mesh)
    grid = _fib_hemisphere(6000)
    cw = math.cos(math.radians(prm.peak_win_deg))
    tree = cKDTree(np.concatenate([grid, -grid]))
    chord = math.sqrt(2 - 2 * cw)
    dens = np.zeros(len(grid))
    nbrs = tree.query_ball_point(n, chord)
    for j, lst in enumerate(nbrs):
        for g in lst:
            dens[g % len(grid)] += a[j]
    total = a.sum()
    order = np.argsort(-dens)
    peaks: list[np.ndarray] = []
    out = []
    cs = math.cos(math.radians(prm.peak_sep_deg))
    floor = prm.peak_min_area_mm2
    if prm.peak_min_area_rel > 0:
        floor = min(floor, prm.peak_min_area_rel * total)
    thr = max(prm.peak_min_area_frac * total, floor)
    for g in order:
        if dens[g] < thr:
            break
        d = grid[g]
        if any(abs(d @ q) > cs for q in peaks):
            continue
        # refine: weighted mean of sign-aligned normals within the window
        dots = n @ d
        m = np.abs(dots) > cw
        if m.sum() < 3:
            continue
        v = (n[m] * (np.sign(dots[m]) * a[m])[:, None]).sum(0)
        v /= np.linalg.norm(v)
        if any(abs(v @ q) > cs for q in peaks):
            continue
        # sharpness: a flange is a compact peak; a large-radius arc is a band (ring ~ core)
        ad = np.abs(n @ v)
        core = a[ad > cw].sum()
        ring = a[(ad <= cw) & (ad > math.cos(math.radians(2 * prm.peak_win_deg)))].sum()
        if core < 1.2 * ring:
            continue
        peaks.append(v)
        out.append({"normal": v, "area": float(dens[g])})
        if len(out) >= 40:
            break
    return out


def axis_families(mesh: Mesh, prm: Params, peaks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    peaks = normal_peaks(mesh, prm) if peaks is None else peaks
    cands = []
    smin = math.sin(math.radians(8.0))
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            c = np.cross(peaks[i]["normal"], peaks[j]["normal"])
            s = np.linalg.norm(c)
            if s < smin:
                continue
            c = c / s
            w = min(peaks[i]["area"], peaks[j]["area"])
            cands.append((w, c, i, j))
    cands.sort(key=lambda x: -x[0])
    fams: list[dict[str, Any]] = []
    cm = math.cos(math.radians(prm.family_merge_deg))
    for w, c, i, j in cands:
        for f in fams:
            if abs(c @ f["axis"]) > cm:
                f["weight"] += w
                f["pairs"].append((i, j))
                break
        else:
            fams.append({"axis": c, "weight": w, "pairs": [(i, j)]})
    # great-circle support: area of normals perpendicular to u that are NOT on a peak (bend-zone arcs)
    n, a = gauss_samples(mesh)
    pk = np.array([p["normal"] for p in peaks]) if peaks else np.zeros((0, 3))
    on_peak = (np.abs(n @ pk.T) > math.cos(math.radians(3.0))).any(1) if len(pk) else np.zeros(len(n), bool)
    s2 = math.sin(math.radians(2.0))
    for f in fams:
        gc = np.abs(n @ f["axis"]) < s2
        f["arc_area"] = float(a[gc & ~on_peak].sum())
        f["gc_area"] = float(a[gc].sum())
    total = float(a.sum())
    for f in fams:
        f["score"] = f["weight"] / total + 2.0 * f["arc_area"] / total
    fams.sort(key=lambda f: -f["score"])
    return fams[: prm.max_families]


def orthonormal_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


# ----------------------------------------------------------------------------- slicing

def slice_family(mesh: Mesh, u: np.ndarray, step: float) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    """Exact sections of the mesh by planes u.x = c_k. Returns levels and, per level, chains:
    {"pts": (m,2) polyline in (e1,e2), "nrm": (m-1,2) outward normal per segment (projected), "face": (m-1,)}"""
    V, F = mesh.V, mesh.F
    h = V @ u
    hf = h[F]
    lo, hi = hf.min(1), hf.max(1)
    c0 = float(h.min()) + 0.5 * step + 1e-4
    nlev = int(math.floor((float(h.max()) - c0) / step)) + 1
    levels = c0 + step * np.arange(max(nlev, 0))
    klo = np.ceil((lo - c0) / step).astype(np.int64)
    khi = np.floor((hi - c0) / step).astype(np.int64)
    span = khi - klo + 1
    cross = span > 0
    fids = np.nonzero(cross)[0]
    reps = span[fids]
    fid_rep = np.repeat(fids, reps)
    first = np.repeat(klo[fids], reps)
    offs = np.arange(len(fid_rep)) - np.repeat(np.cumsum(reps) - reps, reps)
    kk = first + offs
    keep = (kk >= 0) & (kk < len(levels))
    fid_rep, kk = fid_rep[keep], kk[keep]
    e1, e2 = orthonormal_frame(u)
    P2 = np.stack([V @ e1, V @ e2], 1)
    N2 = np.stack([mesh.fn @ e1, mesh.fn @ e2], 1)
    out: list[list[dict[str, Any]]] = [[] for _ in range(len(levels))]
    if len(fid_rep) == 0:
        return levels, out
    order = np.argsort(kk, kind="stable")
    fid_rep, kk = fid_rep[order], kk[order]
    bounds = np.searchsorted(kk, np.arange(len(levels) + 1))
    nv = len(V)
    for k in range(len(levels)):
        s, e = bounds[k], bounds[k + 1]
        if e - s < 2:
            continue
        fi = fid_rep[s:e]
        c = levels[k]
        tri = F[fi]                                 # (m,3)
        above = h[tri] > c                          # (m,3)
        cnt = above.sum(1)
        # lone vertex: the one whose side differs from the other two
        lone_is_above = cnt == 1
        lone = np.where(lone_is_above[:, None], above, ~above).argmax(1)
        a_idx = tri[np.arange(len(fi)), lone]
        b_idx = tri[np.arange(len(fi)), (lone + 1) % 3]
        d_idx = tri[np.arange(len(fi)), (lone + 2) % 3]

        def ipt(p, q):
            hp, hq = h[p], h[q]
            t = (c - hp) / (hq - hp)
            return P2[p] + (P2[q] - P2[p]) * t[:, None]

        pa = ipt(a_idx, b_idx)
        pb = ipt(a_idx, d_idx)
        ka = np.minimum(a_idx, b_idx) * nv + np.maximum(a_idx, b_idx)
        kb = np.minimum(a_idx, d_idx) * nv + np.maximum(a_idx, d_idx)
        out[k] = _chain(pa, pb, ka, kb, N2[fi], fi)
    return levels, out


def _chain(pa, pb, ka, kb, nrm, fi) -> list[dict[str, Any]]:
    m = len(pa)
    keys = np.concatenate([ka, kb])
    seg = np.concatenate([np.arange(m), np.arange(m)])
    end = np.concatenate([np.zeros(m, np.int8), np.ones(m, np.int8)])
    o = np.argsort(keys, kind="stable")
    ks = keys[o]
    # neighbour over the shared edge key: pairs of equal consecutive keys
    nb = -np.ones((m, 2), np.int64)        # nb[s, end] = neighbour segment
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    for i in same:
        s1, e1_, s2, e2_ = seg[o[i]], end[o[i]], seg[o[i + 1]], end[o[i + 1]]
        if s1 != s2:
            nb[s1, e1_] = s2
            nb[s2, e2_] = s1
    pts_end = (pa, pb)
    keys_end = (ka, kb)
    visited = np.zeros(m, bool)
    chains = []
    # start from open ends first, then closed loops
    starts = list(np.nonzero((nb[:, 0] < 0) | (nb[:, 1] < 0))[0]) + list(range(m))
    for s0 in starts:
        if visited[s0]:
            continue
        # orient: begin at the free end if any
        if nb[s0, 0] < 0:
            entry = 0
        elif nb[s0, 1] < 0:
            entry = 1
        else:
            entry = 0
        seq, ent = [], []
        s, en = s0, entry
        while s >= 0 and not visited[s]:
            visited[s] = True
            seq.append(s)
            ent.append(en)
            ex = 1 - en
            nxt = nb[s, ex]
            if nxt < 0:
                break
            key = keys_end[ex][s]
            en = 0 if ka[nxt] == key else 1
            s = nxt
        if len(seq) < 3:
            continue
        seq = np.array(seq)
        ent = np.array(ent)
        p_in = np.where(ent[:, None] == 0, pa[seq], pb[seq])
        p_out = np.where(ent[:, None] == 0, pb[seq], pa[seq])
        pts = np.vstack([p_in, p_out[-1:]])
        chains.append({"pts": pts, "nrm": nrm[seq], "face": fi[seq]})
    return chains


# ----------------------------------------------------------------------------- corners

def _dp(Q: np.ndarray, eps: float) -> list[int]:
    keep = [0, len(Q) - 1]
    stack = [(0, len(Q) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = Q[b] - Q[a]
        L = np.linalg.norm(d)
        rel = Q[a + 1:b] - Q[a]
        if L < 1e-9:
            dist = np.linalg.norm(rel, axis=1)
        else:
            dist = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep.append(m)
            stack.append((a, m))
            stack.append((m, b))
    return sorted(keep)


def _tls(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    c = Q.mean(0)
    X = Q - c
    w, v = np.linalg.eigh(X.T @ X)
    d = v[:, 1]
    rms = math.sqrt(max(w[0], 0.0) / max(len(Q), 1))
    return c, d, rms


def chain_corners(ch: dict[str, Any], prm: Params, t: float) -> list[dict[str, Any]]:
    Q = ch["pts"]
    if len(Q) < 6:
        return []
    seglen = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    if arc[-1] < 2 * prm.min_flat_mm:
        return []
    eps = max(prm.dp_eps_mm, 0.06 * t)
    idx = _dp(Q, eps)
    # runs = merged DP segments
    runs = []          # (i0, i1) point indices
    for a, b in zip(idx[:-1], idx[1:]):
        if runs:
            i0, i1 = runs[-1]
            d_prev = Q[i1] - Q[i0]
            d_cur = Q[b] - Q[a]
            cosang = (d_prev @ d_cur) / (np.linalg.norm(d_prev) * np.linalg.norm(d_cur) + 1e-12)
            if cosang > math.cos(math.radians(prm.merge_deg)):
                runs[-1] = (i0, b)
                continue
        runs.append((a, b))
    flats = []
    for i0, i1 in runs:
        L = arc[i1] - arc[i0]
        if L < prm.min_flat_mm:
            continue
        # trim the ends a little before the TLS fit (arc tangency)
        trim = min(0.3, 0.15 * L)
        sel = np.nonzero((arc >= arc[i0] + trim) & (arc <= arc[i1] - trim))[0]
        if len(sel) < 3:
            sel = np.arange(i0, i1 + 1)
        c, d, rms = _tls(Q[sel])
        if rms > prm.flat_rms_mm:
            continue
        if d @ (Q[i1] - Q[i0]) < 0:
            d = -d
        # arc piece (a chord of a large-radius bend), not a flat: significant quadratic bow, small radius
        if len(sel) >= 6:
            X = Q[sel] - c
            x = X @ d
            r = d[0] * X[:, 1] - d[1] * X[:, 0]
            A = np.stack([x * x, x, np.ones_like(x)], 1)
            coef, *_ = np.linalg.lstsq(A, r, rcond=None)
            noise = float(np.sqrt(np.mean((A @ coef - r) ** 2)))
            sag = abs(coef[0]) * (0.5 * L) ** 2
            if abs(coef[0]) > 1e-9:
                Rfit = 1.0 / (2.0 * abs(coef[0]))
                if sag > max(0.05, 3.0 * noise) and Rfit < min(8.0 * L, 40.0):
                    continue
        segs = np.arange(i0, i1)
        nn = (ch["nrm"][segs] * seglen[segs, None]).sum(0)
        nl = np.linalg.norm(nn)
        flats.append({"i0": i0, "i1": i1, "c": c, "d": d, "len": L, "rms": rms,
                      "n": nn / nl if nl > 1e-9 else np.zeros(2), "nfrac": nl / max(L, 1e-9)})
    # sheet-edge walls: a short flat (~t) between two antiparallel flats (U-turn), or at a chain end
    wall_max = 1.5 * t + 0.8
    for k, f in enumerate(flats):
        if f["len"] > wall_max:
            continue
        prv = flats[k - 1] if k > 0 and arc[f["i0"]] - arc[flats[k - 1]["i1"]] < 2 * t + 2 else None
        nxt = flats[k + 1] if k + 1 < len(flats) and arc[flats[k + 1]["i0"]] - arc[f["i1"]] < 2 * t + 2 else None
        if prv is not None and nxt is not None and prv["d"] @ nxt["d"] < -0.85:
            f["wall"] = True
        elif (prv is None) != (nxt is None) and f["len"] <= 1.3 * t + 0.6:
            nb = prv if prv is not None else nxt
            end_gap = arc[-1] - arc[f["i1"]] if nxt is None else arc[f["i0"]]
            if abs(nb["d"] @ f["d"]) < 0.35 and end_gap < 1.5:
                f["wall"] = True
    # multi-chord arcs: a short flat between two small same-direction turns is part of one bend arc
    flats = [f for f in flats if not f.get("wall")] if any(f.get("wall") for f in flats) else flats
    def _turn(f1, f2):
        cr = f1["d"][0] * f2["d"][1] - f1["d"][1] * f2["d"][0]
        return math.degrees(math.atan2(cr, float(f1["d"] @ f2["d"])))
    short = 2.5 * t + 2.0
    changed = True
    while changed and len(flats) >= 3:
        changed = False
        for k in range(1, len(flats) - 1):
            f0, f1, f2 = flats[k - 1], flats[k], flats[k + 1]
            if f1["len"] >= short:
                continue
            if arc[f1["i0"]] - arc[f0["i1"]] > prm.max_arc_mm or arc[f2["i0"]] - arc[f1["i1"]] > prm.max_arc_mm:
                continue
            a1, a2 = _turn(f0, f1), _turn(f1, f2)
            if a1 * a2 > 0 and min(abs(a1), abs(a2)) < 35.0 and abs(a1) + abs(a2) < 150.0:
                flats.pop(k)
                changed = True
                break
    out = []
    for fa, fb in zip(flats[:-1], flats[1:]):
        if fa.get("wall") or fb.get("wall"):
            continue
        gap = arc[fb["i0"]] - arc[fa["i1"]]
        if gap > prm.max_arc_mm:
            continue
        # joggle: two parallel flats offset sideways, joined by a short web -> two opposite bends
        if abs(float(fa["d"] @ fb["d"])) > math.cos(math.radians(6.0)) and float(fa["d"] @ fb["d"]) > 0:
            rel = Q[fb["i0"]] - Q[fa["i1"]]
            lat = abs(fa["d"][0] * rel[1] - fa["d"][1] * rel[0])
            if 0.5 * t <= lat <= 4.0 * t + 4.0 and gap <= max(4.0 * t + 6.0, 12.0):
                cl = np.linalg.norm(rel)
                if cl > 1e-6:
                    segs = np.arange(fa["i1"], fb["i0"])
                    nn = ch["nrm"][segs].sum(0) if len(segs) else np.zeros(2)
                    nl = np.linalg.norm(nn)
                    web = {"i0": fa["i1"], "i1": fb["i0"], "c": 0.5 * (Q[fa["i1"]] + Q[fb["i0"]]), "d": rel / cl,
                           "len": cl, "rms": 0.0, "n": nn / nl if nl > 1e-9 else np.zeros(2), "web": True}
                    for x, y in ((fa, web), (web, fb)):
                        c = _make_corner(x, y, Q, arc, prm, t)
                        if c is not None:
                            c["joggle"] = True
                            out.append(c)
            continue
        c = _make_corner(fa, fb, Q, arc, prm, t)
        if c is not None:
            out.append(c)
    return out


def _make_corner(fa, fb, Q, arc, prm: Params, t: float) -> dict[str, Any] | None:
    gap = arc[fb["i0"]] - arc[fa["i1"]]
    if True:
        da = -fa["d"]          # from the corner into flat a
        db = fb["d"]           # from the corner into flat b
        cosi = float(np.clip(da @ db, -1, 1))
        incl = math.degrees(math.acos(cosi))
        bend = 180.0 - incl
        if bend < prm.min_bend_deg or bend > prm.max_bend_deg:
            return None
        # virtual sharp: intersection of the two flat lines
        A = np.array([fa["d"], -fb["d"]]).T
        try:
            st = np.linalg.solve(A, fb["c"] - fa["c"])
        except np.linalg.LinAlgError:
            return None
        sharp = fa["c"] + st[0] * fa["d"]
        # the sharp must sit near the junction (not far beyond the flats)
        jn = 0.5 * (Q[fa["i1"]] + Q[fb["i0"]])
        if np.linalg.norm(sharp - jn) > gap + 4 * t + 6.0:
            return None
        bis = da + db
        bl = np.linalg.norm(bis)
        if bl < 1e-6:
            return None
        bis /= bl
        # apex: crossing of the polyline (between the flats, with margin) with the bisector line
        j0 = max(fa["i0"], np.searchsorted(arc, arc[fa["i1"]] - 2.0 * t - 1.0))
        j1 = min(fb["i1"], np.searchsorted(arc, arc[fb["i0"]] + 2.0 * t + 1.0))
        W = Q[j0:j1 + 1] - sharp
        side = bis[0] * W[:, 1] - bis[1] * W[:, 0]
        hh = W @ bis
        apex = None
        sc = np.nonzero(np.sign(side[:-1]) * np.sign(side[1:]) <= 0)[0]
        if len(sc):
            best = None
            for i in sc:
                s0, s1 = side[i], side[i + 1]
                tt = 0.0 if s0 == s1 else s0 / (s0 - s1)
                hv = hh[i] + tt * (hh[i + 1] - hh[i])
                if hv < -1.0:
                    return None
                if best is None or hv < best[0]:
                    best = (hv, W[i] + tt * (W[i + 1] - W[i]))
            if best is not None:
                apex = sharp + best[1]
                apex_h = best[0]
        if apex is None:
            k = int(np.argmin(np.abs(side) + 0.1 * np.maximum(hh, 0)))
            apex = sharp + W[k]
            apex_h = float(hh[k])
        sa = float(fa["n"] @ bis)
        sb = float(fb["n"] @ bis)
        if sa < -0.2 and sb < -0.2:
            skin = "outer"
        elif sa > 0.2 and sb > 0.2:
            skin = "inner"
        else:
            skin = "mixed"
        la_, lb_, na_, nb_ = fa["len"], fb["len"], fa["n"], fb["n"]
        if da[0] * db[1] - da[1] * db[0] < 0:      # canonical order: cross(da, db) along +u
            da, db, la_, lb_, na_, nb_ = db, da, lb_, la_, nb_, na_
        return ({"apex": apex, "sharp": sharp, "bis": bis, "incl": incl, "bend": bend,
                    "la": la_, "lb": lb_, "rms": max(fa["rms"], fb["rms"]), "gap": gap,
                    "apex_h": apex_h, "skin": skin, "da": da, "db": db,
                    "na": na_, "nb": nb_})




def slice_corners(chains: list[dict[str, Any]], prm: Params, t: float) -> list[dict[str, Any]]:
    """All corners of one slice, inner-skin corners moved t to the outer skin, duplicates merged."""
    cs = []
    for ch in chains:
        cs.extend(chain_corners(ch, prm, t))
    for c in cs:
        c["apex_o"] = c["apex"] - t * c["bis"] if c["skin"] == "inner" else c["apex"].copy()
    # merge outer/inner duplicates (same bend seen on both skins)
    cs.sort(key=lambda c: {"outer": 0, "mixed": 1, "inner": 2}[c["skin"]])
    kept: list[dict[str, Any]] = []
    for c in cs:
        dup = None
        for k in kept:
            if (np.linalg.norm(k["apex_o"] - c["apex_o"]) < prm.dedupe_pos_mm + 0.5 * t
                    and abs(k["incl"] - c["incl"]) < 8.0 and k["bis"] @ c["bis"] > 0.8):
                dup = k
                break
        if dup is None:
            c["both_skins"] = False
            kept.append(c)
        else:
            dup["both_skins"] = dup.get("both_skins", False) or (dup["skin"] != c["skin"])
            dup["la"] = max(dup["la"], c["la"])
            dup["lb"] = max(dup["lb"], c["lb"])
    return kept


# ----------------------------------------------------------------------------- tracking

def track_corners(levels: np.ndarray, per_level: list[list[dict[str, Any]]], prm: Params) -> list[list[tuple[int, dict]]]:
    step = prm.step_mm
    max_gap = max(1, int(round(prm.bridge_mm / step)))
    active: list[dict[str, Any]] = []
    done: list[list[tuple[int, dict]]] = []
    for k, cs in enumerate(per_level):
        # retire stale tracks
        still = []
        for tr in active:
            if k - tr["last"] > max_gap + 1:
                done.append(tr["pts"])
            else:
                still.append(tr)
        active = still
        if not cs:
            continue
        used = set()
        if active:
            cost = np.full((len(active), len(cs)), np.inf)
            for i, tr in enumerate(active):
                gapk = k - tr["last"]
                lc = tr["pts"][-1][1]
                tol = prm.link_pos_mm + 0.15 * step * gapk
                for j, c in enumerate(cs):
                    dp = np.linalg.norm(c["apex_o"] - lc["apex_o"])
                    da = abs(c["incl"] - lc["incl"])
                    if dp > tol or da > prm.link_ang_deg or c["bis"] @ lc["bis"] < math.cos(math.radians(prm.link_bis_deg)):
                        continue
                    cost[i, j] = dp / tol + da / prm.link_ang_deg + 0.3 * (gapk - 1)
            from scipy.optimize import linear_sum_assignment
            fin = np.isfinite(cost)
            if fin.any():
                cc = np.where(fin, cost, 1e6)
                ri, ci = linear_sum_assignment(cc)
                for i, j in zip(ri, ci):
                    if fin[i, j]:
                        active[i]["pts"].append((k, cs[j]))
                        active[i]["last"] = k
                        used.add(j)
        for j, c in enumerate(cs):
            if j not in used:
                active.append({"pts": [(k, c)], "last": k})
    done.extend(tr["pts"] for tr in active)
    return done


def _lift(p2: np.ndarray, level: float, u, e1, e2) -> np.ndarray:
    return level * u + p2[0] * e1 + p2[1] * e2


def tracks_to_segments(tracks, levels, u, prm: Params, t: float, fam_id: int) -> list[dict[str, Any]]:
    """Each track -> a segment carrying its 3-D apex points and per-point attributes."""
    e1, e2 = orthonormal_frame(u)
    out = []
    for tr in tracks:
        if len(tr) < prm.min_track_slices:
            continue
        seg = {
            "family": fam_id, "axis": u,
            "P": np.array([_lift(c["apex_o"], levels[k], u, e1, e2) for k, c in tr]),
            "lev": np.array([levels[k] for k, _ in tr]),
            "incl": np.array([c["incl"] for _, c in tr]),
            "fmin": np.array([min(c["la"], c["lb"]) for _, c in tr]),
            "fmax": np.array([max(c["la"], c["lb"]) for _, c in tr]),
            "fla": np.array([c["la"] for _, c in tr]),
            "flb": np.array([c["lb"] for _, c in tr]),
            "outer": np.array([c["skin"] == "outer" for _, c in tr]),
            "inner": np.array([c["skin"] == "inner" for _, c in tr]),
            "both": np.array([bool(c.get("both_skins", False)) for _, c in tr]),
            "rms": np.array([c["rms"] for _, c in tr]),
            "gap": np.array([c["gap"] for _, c in tr]),
            "bis": np.array([c["bis"][0] * e1 + c["bis"][1] * e2 for _, c in tr]),
            "da": np.array([c["da"][0] * e1 + c["da"][1] * e2 for _, c in tr]),
            "db": np.array([c["db"][0] * e1 + c["db"][1] * e2 for _, c in tr]),
        }
        out.append(seg)
    return out


def _unit3(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def segment_line(seg: dict[str, Any], prm: Params) -> dict[str, Any] | None:
    """TLS line through the apex points; None when the points do not line up along the family axis."""
    u = np.asarray(seg["axis"], float)
    P = seg["P"]
    c0 = P.mean(0)
    n = len(P)
    d = u.copy()
    if n >= 4:
        X = P - c0
        w, v = np.linalg.eigh(X.T @ X)
        spread = float(np.sqrt(max(w[2], 0) / n))
        if spread >= 2.5:            # long enough for its own direction (family axes of small flanges are coarse)
            dp = v[:, 2]
            if abs(dp @ u) < math.cos(math.radians(prm.max_axis_dev_deg)):
                return None          # apex points run obliquely: a cut through another family's bend
            d = dp
    if d @ u < 0:
        d = -d
    s = (P - c0) @ d
    res = np.linalg.norm((P - c0) - np.outer(s, d), axis=1)
    if np.sqrt((res ** 2).mean()) > 1.5:
        return None
    half = 0.5 * prm.step_mm / max(abs(d @ u), 0.5)
    lo, hi = s.min() - half, s.max() + half
    nsl = int(round((seg["lev"].max() - seg["lev"].min()) / prm.step_mm)) + 1
    nsup = len(np.unique(np.round(seg["lev"] / prm.step_mm)))
    bis = _unit3(seg["bis"].mean(0))
    da = _unit3(seg["da"].mean(0))
    db = _unit3(seg["db"].mean(0))
    return {
        "family": int(seg["family"]), "axis": u.tolist(),
        "start": (c0 + lo * d).tolist(), "end": (c0 + hi * d).tolist(), "dir": d.tolist(),
        "length": float(hi - lo), "n_support": int(nsup), "n_slices": int(nsl),
        "support_frac": float(min(1.0, nsup / max(nsl, 1))), "incl": float(np.median(seg["incl"])),
        "incl_std": float(seg["incl"].std()),
        "flank_min": float(np.median(seg["fmin"])), "flank_max": float(np.median(seg["fmax"])),
        "flank_a": float(np.median(seg["fla"])), "flank_b": float(np.median(seg["flb"])),
        "res_rms": float(np.sqrt((res ** 2).mean())),
        "axis_dev_deg": float(math.degrees(math.acos(min(1.0, abs(d @ u))))),
        "frac_outer": float(seg["outer"].mean()), "frac_inner": float(seg["inner"].mean()),
        "frac_both": float(seg["both"].mean()), "flat_rms": float(np.median(seg["rms"])),
        "gap_med": float(np.median(seg["gap"])), "bis": bis.tolist(), "da": da.tolist(), "db": db.tolist(),
        "n_parts": int(seg.get("n_parts", 1)),
    }


def _concat(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"family": a["family"], "axis": a["axis"], "n_parts": a.get("n_parts", 1) + b.get("n_parts", 1)}
    for k in ("P", "lev", "incl", "fmin", "fmax", "fla", "flb", "outer", "inner", "both", "rms", "gap", "bis", "da", "db"):
        out[k] = np.concatenate([a[k], b[k]])
    return out


def flank_continuity(mesh: Mesh, tree: cKDTree, p0, d, da, db, s0: float, s1: float, t: float,
                     reach: float) -> tuple[float, float]:
    """Fraction of 1 mm stations in the gap [s0, s1] along the line with surface on flank A / flank B
    (beyond the bend zone, within reach), i.e. whether the same two plates continue across the gap."""
    if s1 - s0 < 1.0:
        return 1.0, 1.0
    a0 = 2.0 * t + 2.0
    a1 = a0 + max(3.0, reach)
    mid = p0 + 0.5 * (s0 + s1) * d
    rad = math.sqrt((0.5 * (s1 - s0)) ** 2 + a1 ** 2) + 1.0
    idx = tree.query_ball_point(mid, rad)
    if not idx:
        return 0.0, 0.0
    idx = np.asarray(idx)
    Q = mesh.V[idx] - p0
    NV = mesh.vn[idx]
    s = Q @ d
    m = (s >= s0) & (s <= s1)
    Q, s, NV = Q[m], s[m], NV[m]
    nst = max(1, int(math.ceil(s1 - s0)))
    bins = np.clip((s - s0).astype(int), 0, nst - 1)
    res = []
    for f in (da, db):
        nrm = _unit3(np.cross(d, f))
        a = Q @ f
        h = np.abs(Q @ nrm)
        on = (a >= a0) & (a <= a1) & (h <= t + 1.5 + 0.05 * a) & (np.abs(NV @ nrm) > 0.9)
        res.append(float((np.bincount(bins[on], minlength=nst) > 0).mean()))
    return res[0], res[1]


def same_plates(mesh: Mesh, tree: cKDTree, p0, d, da, db, s0: float, s1: float, t: float,
                a_max: float, cell: float = 1.5, flank_len: tuple[float, float] | None = None) -> tuple[bool, bool]:
    """Are the flank surfaces on both sides of the gap [s0, s1] the same plates? For each flank: the mesh
    vertices in the flank plane band (beyond the bend zone, up to a_max) are rasterised in (s, a) and the
    two segment ends must fall in one 8-connected component (windows / reliefs are bridged around)."""
    from scipy import ndimage

    pad = 8.0
    a0 = 2.0 * t + 2.0
    a1 = max(a0 + 6.0, a_max)
    mid = p0 + 0.5 * (s0 + s1) * d
    rad = math.sqrt((0.5 * (s1 - s0) + pad) ** 2 + a1 ** 2) + 1.0
    idx = tree.query_ball_point(mid, rad)
    if not idx:
        return False, False
    idx = np.asarray(idx)
    Q = mesh.V[idx] - p0
    NV = mesh.vn[idx]
    s = Q @ d
    m = (s >= s0 - pad) & (s <= s1 + pad)
    Q, s, NV = Q[m], s[m], NV[m]
    out = []
    ns = int(math.ceil((s1 - s0 + 2 * pad) / cell)) + 1
    na = int(math.ceil((a1 - a0) / cell)) + 1
    for fi_, f in enumerate((da, db)):
        b0, b1 = a0, a1
        if flank_len is not None and flank_len[fi_] < a0 + 6.0:
            # short flank (joggle web, small tab): test the flank itself, not the region beyond it
            b0 = min(a0, 0.4 * flank_len[fi_])
            b1 = max(b0 + 2.0, 1.1 * flank_len[fi_])
        nrm = _unit3(np.cross(d, f))
        a = Q @ f
        h = np.abs(Q @ nrm)
        on = (a >= b0) & (a <= b1) & (h <= t + 1.5 + 0.03 * a) & (np.abs(NV @ nrm) > 0.9)
        if on.sum() < 4:
            out.append(False)
            continue
        gi = np.clip(((s[on] - (s0 - pad)) / cell).astype(int), 0, ns - 1)
        gj = np.clip(((a[on] - b0) / cell).astype(int), 0, na - 1)
        grid = np.zeros((ns, na), bool)
        grid[gi, gj] = True
        grid = ndimage.binary_closing(grid, structure=np.ones((2, 2)))
        lab, _ = ndimage.label(grid, structure=np.ones((3, 3)))
        left = set(np.unique(lab[: max(1, int(pad / cell)), :])) - {0}
        right = set(np.unique(lab[ns - max(1, int(pad / cell)):, :])) - {0}
        out.append(bool(left & right))
    return out[0], out[1]


def merge_segments(segs: list[dict[str, Any]], prm: Params, mesh: Mesh, tree: cKDTree, t: float,
                   max_gap_mm: float = 80.0) -> list[dict[str, Any]]:
    """Chain collinear segments of one family: adjacent ones directly, gapped ones only when both flanks
    continue across the gap (a window / relief between the same two plates)."""
    items = []
    for sg in segs:
        ln = segment_line(sg, prm)
        if ln is not None:
            items.append([sg, ln])
    changed = True
    while changed:
        changed = False
        items.sort(key=lambda it: -it[1]["length"])
        for i in range(len(items)):
            a = items[i][1]
            pa0, pa1 = np.asarray(a["start"]), np.asarray(a["end"])
            d = _unit3(pa1 - pa0)
            for j in range(i + 1, len(items)):
                b = items[j][1]
                if a["family"] != b["family"]:
                    continue
                db_ = _unit3(np.asarray(b["end"]) - np.asarray(b["start"]))
                if abs(d @ db_) < math.cos(math.radians(3.0)):
                    continue
                if abs(a["incl"] - b["incl"]) > 7.0 or np.dot(a["bis"], b["bis"]) < 0.9:
                    continue
                # transverse offset of b's points from a's line
                Pb = items[j][0]["P"] - pa0
                off = np.linalg.norm(Pb - np.outer(Pb @ d, d), axis=1)
                if np.median(off) > 1.2 + 0.3 * t:
                    continue
                sa = sorted([0.0, float((pa1 - pa0) @ d)])
                sb = sorted([float((np.asarray(b["start"]) - pa0) @ d), float((np.asarray(b["end"]) - pa0) @ d)])
                gap0, gap1 = (sa[1], sb[0]) if sb[0] >= sa[0] else (sb[1], sa[0])
                gap = gap1 - gap0
                if gap > max_gap_mm:
                    continue
                if gap > 2.5 * prm.step_mm:
                    a_max = min(80.0, 1.2 * max(3.0, min(a["flank_max"], b["flank_max"])))
                    ok_a, ok_b = same_plates(mesh, tree, pa0, d, np.asarray(a["da"]), np.asarray(a["db"]),
                                             gap0, gap1, t, a_max,
                                             flank_len=(min(a["flank_a"], b["flank_a"]), min(a["flank_b"], b["flank_b"])))
                    if not (ok_a and ok_b):
                        continue
                merged = _concat(items[i][0], items[j][0])
                ln = segment_line(merged, prm)
                if ln is None:
                    continue
                items[i] = [merged, ln]
                items.pop(j)
                changed = True
                break
            if changed:
                break
    return [it[1] for it in items]


# ----------------------------------------------------------------------------- bend-strip evidence

def strip_evidence(mesh: Mesh, tree: cKDTree, line: dict[str, Any], t: float) -> dict[str, float]:
    """Fraction of 1 mm stations along the line whose nearby surface normals sweep the arc between
    the two flange normals (cylinder-strip evidence), and the two-sided flank presence."""
    a = np.asarray(line["start"])
    b = np.asarray(line["end"])
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-6:
        return {"strip_frac": 0.0, "flank_frac": 0.0}
    d /= L
    da = np.asarray(line["da"])
    db = np.asarray(line["db"])
    na = np.cross(d, da)
    nb = np.cross(d, db)
    na /= max(np.linalg.norm(na), 1e-9)
    nb /= max(np.linalg.norm(nb), 1e-9)
    r = 2.5 * t + 3.0
    mid = 0.5 * (a + b)
    idx = tree.query_ball_point(mid, 0.5 * L + r)
    if not idx:
        return {"strip_frac": 0.0, "flank_frac": 0.0}
    idx = np.asarray(idx)
    P = mesh.V[idx] - a
    s = P @ d
    w = P - np.outer(s, d)
    dist = np.linalg.norm(w, axis=1)
    m = (s >= 0) & (s <= L) & (dist <= r)
    if m.sum() < 5:
        return {"strip_frac": 0.0, "flank_frac": 0.0}
    s, n = s[m], mesh.vn[idx[m]]
    # angle of each normal in the plane perpendicular to d, relative to na; arc between na and +-nb
    nperp = n - np.outer(n @ d, d)
    pl = np.linalg.norm(nperp, axis=1)
    okp = pl > 0.94                     # normal within ~20 deg of the plane perpendicular to the line
    x = np.cross(d, na)
    ang = np.degrees(np.arctan2(n @ x, n @ na))
    tot = abs(math.degrees(math.atan2(nb @ x, nb @ na)))
    tot2 = abs(math.degrees(math.atan2(-nb @ x, -nb @ na)))
    # skins may have either sign: test both orientations of na
    a1 = np.abs(ang)
    a2 = np.abs(np.degrees(np.arctan2(n @ x, -(n @ na))))
    lo_m = 0.15
    mid1 = okp & (a1 > lo_m * tot) & (a1 < (1 - lo_m) * tot) if tot > 8 else np.zeros(len(n), bool)
    mid2 = okp & (a2 > lo_m * tot2) & (a2 < (1 - lo_m) * tot2) if tot2 > 8 else np.zeros(len(n), bool)
    mid = mid1 | mid2
    nst = max(1, int(math.ceil(L)))
    bins = np.clip(s.astype(int), 0, nst - 1)
    strip = np.bincount(bins[mid], minlength=nst) > 0
    onA = okp & ((np.abs(n @ na) > 0.985))
    onB = okp & ((np.abs(n @ nb) > 0.985))
    fl = (np.bincount(bins[onA], minlength=nst) > 0) & (np.bincount(bins[onB], minlength=nst) > 0)
    return {"strip_frac": float(strip.mean()), "flank_frac": float(fl.mean())}


def refine_span(tree: cKDTree, line: dict[str, Any], step: float, t: float) -> None:
    """Sub-slice endpoints: walk 0.25 mm stations outward from each end (up to 1.5 slices beyond, 1 slice
    inside) while the scanned surface still touches the apex line (the bend surface ends there)."""
    a = np.asarray(line["start"], float)
    b = np.asarray(line["end"], float)
    L = np.linalg.norm(b - a)
    if L < 1e-6:
        return
    d = (b - a) / L
    r = 0.5 + 0.15 * t
    new = []
    for end, sgn in ((a, -1.0), (b, 1.0)):
        ss = np.arange(-min(step, 0.4 * L), 1.5 * step + 1e-9, 0.25)
        pts = end[None, :] + sgn * ss[:, None] * d[None, :]
        occ = np.array([len(x) > 0 for x in tree.query_ball_point(pts, r)])
        k0 = np.searchsorted(ss, 0.0)
        # last occupied station reachable from inside without a gap > 0.75 mm
        last = None
        miss = 0
        for k in range(len(ss)):
            if occ[k]:
                last = k
                miss = 0
            else:
                miss += 1
                if k > k0 and miss > 3:
                    break
        shift = ss[last] if last is not None else 0.0
        new.append(end + sgn * shift * d)
    if np.linalg.norm(new[1] - new[0]) > 1.0 and (new[1] - new[0]) @ d > 0:
        line["start"], line["end"] = new[0].tolist(), new[1].tolist()
        line["length"] = float(np.linalg.norm(new[1] - new[0]))


# ----------------------------------------------------------------------------- merge

def _pair(a: dict, b: dict) -> tuple[float, float, float]:
    """(angle deg, transverse offset mm, min overlap fraction) between two candidate lines."""
    pa0, pa1 = np.asarray(a["start"]), np.asarray(a["end"])
    pb0, pb1 = np.asarray(b["start"]), np.asarray(b["end"])
    da = pa1 - pa0
    db = pb1 - pb0
    la, lb = np.linalg.norm(da), np.linalg.norm(db)
    da /= la
    db /= lb
    ang = math.degrees(math.acos(min(1.0, abs(da @ db))))
    ma, mb = 0.5 * (pa0 + pa1), 0.5 * (pb0 + pb1)
    dm = mb - ma
    off = max(np.linalg.norm(dm - da * (dm @ da)), np.linalg.norm(dm - db * (dm @ db)))
    ia = sorted([(pa0 - ma) @ da, (pa1 - ma) @ da])
    ib = sorted([(pb0 - ma) @ da, (pb1 - ma) @ da])
    ov = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
    return ang, off, ov / max(min(la, lb), 1e-9)


def merge_candidates(cands: list[dict[str, Any]], key: str = "quality") -> list[dict[str, Any]]:
    """Greedy NMS: a weaker candidate within 4 deg / 2.5 mm / 50 % (of the shorter) of a kept one is dropped."""
    cands = sorted(cands, key=lambda c: -c.get(key, 0.0))
    kept: list[dict[str, Any]] = []
    for c in cands:
        dup = False
        for k in kept:
            ang, off, ov = _pair(c, k)
            if ang < 4.0 and off < 2.5 and ov > 0.5:
                dup = True
                k["n_dups"] = k.get("n_dups", 0) + 1
                break
        if not dup:
            kept.append(c)
    return kept


# ----------------------------------------------------------------------------- driver

def detect(mesh: Mesh, prm: Params | None = None, *, thickness: dict | None = None,
           families: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    prm = prm or Params()
    tm = {}
    t0 = time.time()
    th = thickness or estimate_thickness(mesh)
    t = float(np.clip(th["t_mm"], 0.5, 6.0))
    tm["thickness"] = time.time() - t0
    t1 = time.time()
    fams = families if families is not None else axis_families(mesh, prm)
    fallback = False
    if families is None and not fams and prm.fallback_frac_only:
        from dataclasses import replace as _replace
        fams = axis_families(mesh, _replace(prm, peak_min_area_mm2=0.0, peak_min_area_rel=0.0))
        fallback = True
    tm["families"] = time.time() - t1
    t2 = time.time()
    tree = cKDTree(mesh.V)
    raw = []
    for fi, f in enumerate(fams):
        u = np.asarray(f["axis"], float)
        levels, chains = slice_family(mesh, u, prm.step_mm)
        per_level = [slice_corners(ch, prm, t) for ch in chains]
        tracks = track_corners(levels, per_level, prm)
        segs = tracks_to_segments(tracks, levels, u, prm, t, fi)
        raw.extend(merge_segments(segs, prm, mesh, tree, t))
    tm["slicing"] = time.time() - t2
    t3 = time.time()
    for c in raw:
        refine_span(tree, c, prm.step_mm, t)
        c.update(strip_evidence(mesh, tree, c, t))
        c["quality"] = c["n_support"] * c["support_frac"] * (0.5 + c["frac_outer"] + c["frac_both"])
    tm["strip"] = time.time() - t3
    merged = merge_candidates(raw)
    tm["total"] = time.time() - t0
    return {"thickness": th, "t_mm": t, "area_total": float(mesh.fa.sum()), "family_fallback": fallback,
            "families": [{"axis": np.asarray(f["axis"]).tolist(), "score": f.get("score"), "weight": f.get("weight"),
                          "arc_area": f.get("arc_area")} for f in fams],
            "raw_count": len(raw), "candidates": merged, "timing_s": tm, "params": asdict(prm)}
