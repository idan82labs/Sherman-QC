"""zs_lane_v2 geometric evidence on the scan mesh (no labels, no CAD, no part-specific inputs).

Three per-line tests, all in mm / degrees / multiples of the measured sheet thickness t:

1. ``plate_connected``: are the flank plates on both sides of a gap [g0, g1] along a line the *same plates*?
   For one flank (in-plane direction f away from the bend, plate normal n = d x f) the mesh vertices that lie
   on that plate (|h| <= t + 1.5 + 0.03 a, normal within ~25 deg of n, a beyond the bend zone) are rasterised in
   (s along the line, a across the plate) at a cell size tied to the mesh spacing, dilated by one cell, and
   labelled (8-connected). The plate is continuous across the gap when one component reaches both sides.
   A window or relief between the same two plates is bridged around; separate tabs are not.
2. ``station_profile``: per 1 mm station along a line: bend-zone material present (surface touching the apex
   line), plate material present on flank A / flank B just beyond the bend zone, and the normal sweep test.
3. Normal sweep across the bisector (per station): the normals of the material within the bend zone are
   expressed as an angle phi in the plane perpendicular to the line, from flank A's normal (0) to flank B's
   normal (theta, the bend angle). A formed bend has normals filling the arc: the station sweeps when at
   least 3 of the 4 inner bins of [0.1 theta, 0.9 theta] hold a normal. A seam, a cut wall or two plates
   meeting at a sharp edge have normals only at the ends of the arc (or off the arc).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def mesh_spacing(V: np.ndarray, F: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Median mesh edge length (sampled)."""
    if len(F) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=min(n, len(F)), replace=False)
    e = np.linalg.norm(V[F[idx, 0]] - V[F[idx, 1]], axis=1)
    return float(np.median(e))


class MeshCtx:
    """Vertices, vertex normals, kd-tree, thickness and spacing of one scan mesh."""

    def __init__(self, V: np.ndarray, VN: np.ndarray, F: np.ndarray, t: float, tree: cKDTree | None = None,
                 spacing: float | None = None):
        self.V = np.asarray(V, float)
        self.VN = np.asarray(VN, float)
        self.t = float(t)
        self.tree = tree if tree is not None else cKDTree(self.V)
        self.spacing = float(spacing) if spacing is not None else mesh_spacing(self.V, np.asarray(F), 20000)
        self.cell = float(np.clip(1.25 * self.spacing, 0.4, 1.5))

    def near(self, center: np.ndarray, radius: float) -> np.ndarray:
        return np.asarray(self.tree.query_ball_point(center, radius), dtype=np.int64)


def _line(c: dict) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(c["start"], float)
    b = np.asarray(c["end"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / max(L, 1e-9), L


def flank_band(t: float, flank_len: float | None, a_hi: float) -> tuple[float, float]:
    """Plate band [a_lo, a_hi] tested on one flank. Normally beyond the bend zone (a >= 2t + 2); a short flank
    (lip, joggle web, small tab: shorter than 2t + 6 mm) is tested on the flank itself, 0.4 to 1.1 x its length."""
    a_lo = 2.0 * t + 2.0
    if flank_len is not None and flank_len == flank_len and flank_len < a_lo + 4.0:
        lo = max(0.5, 0.4 * flank_len)
        return lo, max(lo + 2.0, 1.1 * flank_len)
    return a_lo, float(max(a_lo + 4.0, a_hi))


def plate_connected(ctx: MeshCtx, p0: np.ndarray, d: np.ndarray, f: np.ndarray, g0: float, g1: float,
                    a_hi: float, pad: float = 8.0, idx: np.ndarray | None = None,
                    flank_len: float | None = None) -> dict[str, Any]:
    """Is the plate on flank f one connected piece across the gap [g0, g1] (s along d from p0)?"""
    t = ctx.t
    a_lo, a_hi = flank_band(t, flank_len, a_hi)
    if idx is None:
        mid = p0 + 0.5 * (g0 + g1) * d
        rad = math.sqrt((0.5 * (g1 - g0) + pad) ** 2 + a_hi ** 2) + t + 2.0
        idx = ctx.near(mid, rad)
    out = {"connected": False, "left": 0, "right": 0, "gap_cover": 0.0}
    if len(idx) == 0:
        return out
    Q = ctx.V[idx] - p0
    NV = ctx.VN[idx]
    nrm = unit(np.cross(d, f))
    s = Q @ d
    a = Q @ f
    h = np.abs(Q @ nrm)
    on = ((s >= g0 - pad) & (s <= g1 + pad) & (a >= a_lo) & (a <= a_hi) & (h <= t + 1.5 + 0.03 * a)
          & (np.abs(NV @ nrm) > 0.9))
    if on.sum() < 4:
        return out
    cell = ctx.cell
    ns = int(math.ceil((g1 - g0 + 2 * pad) / cell)) + 1
    na = int(math.ceil((a_hi - a_lo) / cell)) + 1
    gi = np.clip(((s[on] - (g0 - pad)) / cell).astype(int), 0, ns - 1)
    gj = np.clip(((a[on] - a_lo) / cell).astype(int), 0, na - 1)
    grid = np.zeros((ns, na), bool)
    grid[gi, gj] = True
    grid = ndimage.binary_dilation(grid, structure=ndimage.generate_binary_structure(2, 1))
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3)))
    k0 = max(1, int(round(min(pad, 4.0) / cell)))
    il = int((pad - min(pad, 4.0)) / cell)
    ir = int((g1 - g0 + pad) / cell) + 1
    left = set(np.unique(lab[il: il + k0, :])) - {0}
    right = set(np.unique(lab[ir: ir + k0, :])) - {0}
    gs, ge = int(pad / cell), int((g1 - g0 + pad) / cell)
    cov = grid[gs: ge + 1].any(1).mean() if ge >= gs else 1.0
    raw = np.zeros(ns, bool)
    raw[gi] = True
    gsr, ger = int(math.ceil(pad / cell)), int((g1 - g0 + pad) / cell) - 1
    cov_raw = raw[gsr: ger + 1].mean() if ger >= gsr else float(raw[min(gs, ns - 1)])
    out.update(connected=bool(left & right), left=len(left), right=len(right), gap_cover=float(cov),
               gap_cover_raw=float(cov_raw))
    return out


def station_profile(ctx: MeshCtx, c: dict, s_lo: float | None = None, s_hi: float | None = None) -> dict[str, np.ndarray]:
    """Per 1 mm station along candidate c (over [s_lo, s_hi], default the whole line): bend-zone material,
    flank A / flank B plate material just beyond the bend zone, and the normal-sweep test (see module doc).
    Needs c['da'], c['db'] (in-plane flank directions away from the bend)."""
    t = ctx.t
    p0, d, L = _line(c)
    s_lo = 0.0 if s_lo is None else s_lo
    s_hi = L if s_hi is None else s_hi
    nst = max(1, int(math.ceil(s_hi - s_lo)))
    da = unit(np.asarray(c["da"], float))
    db = unit(np.asarray(c["db"], float))
    da = unit(da - (da @ d) * d)
    db = unit(db - (db @ d) * d)
    a_lo = 2.0 * t + 2.0
    a_w = a_lo + max(4.0, 2.0 * t)
    r_zone = 2.5 * t + 3.0
    reach = max(a_w, r_zone) + t + 2.0
    mid = p0 + 0.5 * (s_lo + s_hi) * d
    idx = ctx.near(mid, math.sqrt((0.5 * (s_hi - s_lo)) ** 2 + reach ** 2) + 1.0)
    empty = np.zeros(nst, bool)
    res = {"bend": empty.copy(), "fa": empty.copy(), "fb": empty.copy(), "sweep": empty.copy(),
           "sweep_bins": np.zeros(nst), "n_zone": np.zeros(nst), "s0": s_lo}
    if len(idx) == 0:
        return res
    Q = ctx.V[idx] - p0
    NV = ctx.VN[idx]
    s = Q @ d
    m = (s >= s_lo) & (s < s_lo + nst)
    Q, NV, s = Q[m], NV[m], s[m]
    k = np.clip((s - s_lo).astype(int), 0, nst - 1)
    w = Q - np.outer(s, d)
    dist = np.linalg.norm(w, axis=1)
    # bend-zone material touching the apex line
    r_touch = 0.6 + 0.2 * t + 1.25 * ctx.spacing
    res["bend"] = np.bincount(k[dist <= r_touch], minlength=nst) > 0
    for key, f in (("fa", da), ("fb", db)):
        nrm = unit(np.cross(d, f))
        a = Q @ f
        h = np.abs(Q @ nrm)
        on = (a >= a_lo) & (a <= a_w) & (h <= t + 1.5) & (np.abs(NV @ nrm) > 0.9)
        res[key] = np.bincount(k[on], minlength=nst) > 0
    # normal sweep across the bisector: the angle of each zone normal from flank A's normal to flank B's
    na_ = unit(np.cross(d, da))
    nb_ = unit(np.cross(d, db))
    if na_ @ (da + db) > 0:          # outward normals point away from the corner interior
        na_ = -na_
    if nb_ @ (da + db) > 0:
        nb_ = -nb_
    theta = math.degrees(math.acos(float(np.clip(na_ @ nb_, -1, 1))))
    zone = dist <= r_zone
    nperp = NV - np.outer(NV @ d, d)
    okp = np.linalg.norm(nperp, axis=1) > 0.9
    x = unit(nb_ - (nb_ @ na_) * na_)
    ang = np.degrees(np.arctan2(NV @ x, NV @ na_))
    # inner skin normals point inward: fold them onto the outer orientation
    ang_in = np.degrees(np.arctan2(-(NV @ x), -(NV @ na_)))
    sel = zone & okp
    res["n_zone"] = np.bincount(k[sel], minlength=nst).astype(float)
    if theta >= 8.0 and sel.any():
        bins = np.zeros((nst, 4), bool)
        for A in (ang, ang_in):
            fr = A / theta
            inside = sel & (fr > 0.1) & (fr < 0.9)
            b = np.clip(((fr[inside] - 0.1) / 0.2).astype(int), 0, 3)
            bins[k[inside], b] = True
        res["sweep_bins"] = bins.sum(1).astype(float)
        res["sweep"] = res["sweep_bins"] >= 3
    res["theta"] = theta
    return res


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in mask as (start, end_exclusive)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def sweep_features(prof: dict) -> dict[str, float]:
    """Per-candidate summaries of the station profile (features for the selector)."""
    n = max(len(prof["bend"]), 1)
    return {"sw_full": float(prof["sweep"].mean()) if len(prof["sweep"]) else 0.0,
            "sw_bins": float(prof["sweep_bins"].mean() / 4.0) if len(prof["sweep_bins"]) else 0.0,
            "st_bend": float(prof["bend"].mean()) if len(prof["bend"]) else 0.0,
            "st_fa": float(prof["fa"].mean()) if len(prof["fa"]) else 0.0,
            "st_fb": float(prof["fb"].mean()) if len(prof["fb"]) else 0.0,
            "st_both": float((prof["fa"] & prof["fb"]).mean()) if len(prof["fa"]) else 0.0,
            "st_n": float(n)}


SWEEP_FEATS = ["sw_full", "sw_bins", "st_bend", "st_fa", "st_fb", "st_both"]


def bend_touch(ctx: MeshCtx, c: dict, s_lo: float, s_hi: float, step: float = 0.5) -> np.ndarray:
    """Per ``step`` mm station: does scanned surface touch the apex line (tight radius 0.35 mm + 0.5 spacing)?
    A slot between tabs cuts through the bend, so the apex line leaves the surface there."""
    p0, d, L = _line(c)
    nst = max(1, int(math.ceil((s_hi - s_lo) / step)))
    s = s_lo + step * (np.arange(nst) + 0.5)
    r = 0.35 + 0.5 * ctx.spacing
    pts = p0[None, :] + s[:, None] * d[None, :]
    return np.array([len(x) > 0 for x in ctx.tree.query_ball_point(pts, r)])


def split_runs(ctx: MeshCtx, c: dict, prof: dict, min_run: float = 1.0, a_hi: float = 60.0) -> list[dict]:
    """Interior runs where the bend itself is interrupted (the apex line leaves the scanned surface over at least
    ``min_run`` mm, tight-radius test at 0.5 mm stations), each with the plate-connectivity verdict of both flanks
    across the run and the plate coverage inside it. End runs are span trimming, not interruptions."""
    p0, d, L = _line(c)
    step = 0.5
    touch = bend_touch(ctx, c, 0.0, L, step)
    out = []
    n = len(touch)
    da = unit(np.asarray(c["da"], float))
    db = unit(np.asarray(c["db"], float))
    da = unit(da - (da @ d) * d)
    db = unit(db - (db @ d) * d)
    fl = [float(c.get("flank_a", np.nan)), float(c.get("flank_b", np.nan))]
    fmax = [x for x in fl if x == x]
    a_hi = float(np.clip(1.25 * max(fmax) + 3.0 if fmax else a_hi, 12.0, 150.0))
    for i, j in runs(~touch):
        if i == 0 or j >= n or (j - i) * step < min_run:
            continue
        g0, g1 = i * step, j * step
        pad = float(max(2.0, min(8.0, g0, L - g1)))
        ca = plate_connected(ctx, p0, d, da, g0, g1, a_hi, pad=pad, flank_len=fl[0])
        cb = plate_connected(ctx, p0, d, db, g0, g1, a_hi, pad=pad, flank_len=fl[1])
        out.append({"g0": float(g0), "g1": float(g1), "ca": ca["connected"], "cb": cb["connected"],
                    "cov_a": ca["gap_cover"], "cov_b": cb["gap_cover"], "a_hi": a_hi,
                    "raw_a": ca.get("gap_cover_raw", 1.0), "raw_b": cb.get("gap_cover_raw", 1.0),
                    "sides_a": int(ca["left"] > 0) + int(ca["right"] > 0),
                    "sides_b": int(cb["left"] > 0) + int(cb["right"] > 0)})
    return out


def collinear_pairs(C: list[dict], t: float, max_gap: float = 250.0, ang_deg: float = 3.0) -> list[tuple[int, int, float, float, float]]:
    """Consecutive collinear compatible candidates (i before j along i's direction) with the gap between
    them: (i, j, g0, g1, overlap). Each candidate keeps at most its nearest compatible neighbour per side."""
    n = len(C)
    lines = [_line(c) for c in C]
    best: dict[tuple[int, int], tuple[float, int]] = {}
    out = []
    for i in range(n):
        p0, d, L = lines[i]
        if L < 1.0:
            continue
        for j in range(n):
            if j == i:
                continue
            q0, e, M = lines[j]
            if M < 1.0 or abs(d @ e) < math.cos(math.radians(ang_deg)):
                continue
            ia, ib = C[i].get("incl"), C[j].get("incl")
            if ia is not None and ib is not None and abs(float(ia) - float(ib)) > 10.0:
                continue
            q1 = q0 + M * e
            off = max(np.linalg.norm((q0 - p0) - ((q0 - p0) @ d) * d), np.linalg.norm((q1 - p0) - ((q1 - p0) @ d) * d))
            if off > 1.5 + 0.4 * t:
                continue
            sb = sorted([float((q0 - p0) @ d), float((q1 - p0) @ d)])
            if sb[0] < L - 0.2 * min(L, M):
                continue           # j must lie beyond i's end (small overlap allowed)
            g0, g1 = L, sb[0]
            if g1 - g0 > max_gap:
                continue
            key = (i, 1)
            if key not in best or g1 - g0 < best[key][0]:
                best[key] = (g1 - g0, j)
    seen = set()
    for (i, _), (gap, j) in best.items():
        p0, d, L = lines[i]
        q0, e, M = lines[j]
        q1 = q0 + M * e
        sb = sorted([float((q0 - p0) @ d), float((q1 - p0) @ d)])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        out.append((i, j, L, sb[0], max(0.0, L - sb[0])))
    return out


def pair_evidence(ctx: MeshCtx, ci: dict, cj: dict, g0: float, g1: float) -> dict[str, Any]:
    """Join evidence between candidate i and the next collinear candidate j (gap [g0, g1] along i)."""
    p0, d, L = _line(ci)
    q0, e, M = _line(cj)
    t = ctx.t
    out: dict[str, Any] = {"gap": float(g1 - g0), "len_i": L, "len_j": M}
    fl = []
    for c in (ci, cj):
        fl.append((float(c.get("flank_a", np.nan)), float(c.get("flank_b", np.nan))))
    out["flank_i"], out["flank_j"] = fl
    if "da" not in ci or "db" not in ci:
        out["has_flanks"] = False
        return out
    out["has_flanks"] = True
    da = unit(np.asarray(ci["da"], float))
    db = unit(np.asarray(ci["db"], float))
    da = unit(da - (da @ d) * d)
    db = unit(db - (db @ d) * d)
    fmax = [x for x in (fl[0][0], fl[0][1], fl[1][0], fl[1][1]) if x == x]
    a_hi = float(np.clip(1.25 * max(fmax) + 3.0 if fmax else 40.0, 12.0, 150.0))
    if g1 - g0 < 0.5:
        g0c, g1c = g0 - 0.25, g0 + 0.25
    else:
        g0c, g1c = g0, g1
    pad = float(max(2.0, min(8.0, L, M)))
    mid = p0 + 0.5 * (g0c + g1c) * d
    rad = math.sqrt((0.5 * (g1c - g0c) + pad) ** 2 + a_hi ** 2) + t + 2.0
    idx = ctx.near(mid, rad)
    fla = [x for x in (fl[0][0], fl[1][0]) if x == x]
    flb = [x for x in (fl[0][1], fl[1][1]) if x == x]
    ca = plate_connected(ctx, p0, d, da, g0c, g1c, a_hi, pad, idx, flank_len=max(fla) if fla else None)
    cb = plate_connected(ctx, p0, d, db, g0c, g1c, a_hi, pad, idx, flank_len=max(flb) if flb else None)
    out.update(ca=ca["connected"], cb=cb["connected"], cov_a=ca["gap_cover"], cov_b=cb["gap_cover"], a_hi=a_hi)
    # bend-zone material in the gap (relief / window vs a continuous corner the tracker lost)
    if g1 - g0 >= 1.0:
        prof = station_profile(ctx, {"start": p0.tolist(), "end": (p0 + L * d).tolist(), "da": da, "db": db},
                               s_lo=g0, s_hi=g1)
        out["gap_bend"] = float(prof["bend"].mean())
        out["gap_fa"] = float(prof["fa"].mean())
        out["gap_fb"] = float(prof["fb"].mean())
    else:
        out["gap_bend"] = out["gap_fa"] = out["gap_fb"] = 1.0
    return out
