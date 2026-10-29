"""Rolled-region detection and measurement on a scan mesh, without CAD (research prototype).

Pipeline (every threshold in mm, degrees or multiples of the measured sheet thickness t; no part logic):
  1. Candidate roll axes. Smoothed surface normals (z3 ``gauss_samples``) minus the compact flange peaks
     (z3 ``normal_peaks`` rejects band-like peaks, so a roll's normals stay in) vote for great circles: a
     cylinder of axis u puts its normals on the great circle perpendicular to u. Hough over a hemisphere
     grid + NMS, refined by the smallest eigenvector of sum(a n n^T) over the band.
  2. Per axis: exact mesh sections perpendicular to u (z3 ``slice_family``). Every 2-D chain is resampled,
     its signed curvature (convex > 0 w.r.t. the outward face normal) computed at a smoothing scale, and cut
     into constant-sign curved runs. A run is kept as an ARC when, on the RAW points, a circle fits with
     R in [R_min, R_max], geometric sweep >= min_sweep, rms << sagitta, and its curvature is uniform
     (a blurred sharp bend fails the rms/sagitta and uniformity tests; that keeps rolls disjoint from the
     bend lane's sharp-corner lines). Co-circular arcs in one section are merged (scan holes -> coverage).
  3. Arcs are tracked across sections into regions; convex (outer skin, R_out) and concave (inner skin,
     R_in) tracks that are concentric and t apart are paired.
  4. The axis is re-estimated from the station centres (TLS 3-D line) and the mesh re-sliced
     perpendicular to it until it stops moving.
  5. Measurements per region with per-station values and bootstrap-over-stations uncertainties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from research.zeroshot.z3_slicing import slicer as SL
from research.zeroshot.roll_measure import geom as G


@dataclass
class RollParams:
    r_min_mm: float = 12.0          # rolls start here (and at r_min_t * t); sharper = press-brake bend lane
    r_min_t: float = 5.0
    r_max_mm: float = 1500.0
    min_sweep_deg: float = 15.0     # per merged arc in one section
    piece_sweep_deg: float = 4.0    # a piece of an arc (before co-circular merge)
    min_arc_mm: float = 12.0
    min_len_mm: float = 20.0        # region length along the axis
    min_stations: int = 4
    ds_mm: float = 0.5              # chain resampling
    sigma_pos_mm: float = 1.0       # position smoothing before the tangent angle
    sigma_k_mm: float = 4.0         # curvature smoothing scale
    max_rms_over_sag: float = 0.06  # circle-fit rms / arc sagitta
    max_rms_over_r: float = 0.02
    uniform_min: float = 0.45       # min/max of the curvature of the run's thirds
    axis_band_deg: float = 2.5
    n_axes: int = 8
    axis_nms_deg: float = 6.0
    step_mm: float | None = None    # station spacing (auto: extent / 90, clipped to [1.5, 8])
    refine_iters: int = 3
    n_boot: int = 300
    flat_depth_mm: float = 0.3      # flat-spot report threshold (also >= 5 x station fit rms)


# ----------------------------------------------------------------------------- 1. axis candidates

def _fib_hemi(n):
    i = np.arange(n) + 0.5
    z = i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def axis_candidates(mesh: SL.Mesh, prm: RollParams) -> list[dict[str, Any]]:
    n, a = SL.gauss_samples(mesh)
    peaks = SL.normal_peaks(mesh, SL.Params())
    pk = np.array([p["normal"] for p in peaks]) if peaks else np.zeros((0, 3))
    off = ~((np.abs(n @ pk.T) > math.cos(math.radians(3.0))).any(1)) if len(pk) else np.ones(len(n), bool)
    n, a = n[off], a[off]
    total = float(SL.gauss_samples(mesh)[1].sum())
    if len(n) < 50:
        return []
    rng = np.random.default_rng(0)
    if len(n) > 40000:
        idx = rng.choice(len(n), 40000, replace=False)
        n, a = n[idx], a[idx] * (a.sum() / a[idx].sum())
    grid = _fib_hemi(5000)
    sb = math.sin(math.radians(prm.axis_band_deg))
    out: list[dict[str, Any]] = []
    cn = math.cos(math.radians(prm.axis_nms_deg))
    thr = max(150.0, 0.002 * total)
    alive = np.ones(len(n), bool)
    for _ in range(prm.n_axes):
        # greedy: each accepted axis explains away its band, so a roll does not spawn tilted duplicates
        nn, aa = n[alive], a[alive]
        if len(nn) < 10:
            break
        score = np.zeros(len(grid))
        for s0 in range(0, len(grid), 250):
            score[s0:s0 + 250] = ((np.abs(nn @ grid[s0:s0 + 250].T) < sb) * aa[:, None]).sum(0)
        g = int(np.argmax(score))
        if score[g] < thr:
            break
        u = grid[g]
        m = np.abs(nn @ u) < 2 * sb
        S = (nn[m] * aa[m, None]).T @ nn[m]
        w, v = np.linalg.eigh(S)
        u2 = v[:, 0]
        alive[np.nonzero(alive)[0][np.abs(nn @ u2) < 2 * sb]] = False
        if any(abs(u2 @ o["axis"]) > cn for o in out):
            continue
        out.append({"axis": u2, "band_area": float(score[g]), "band_frac": float(score[g] / total)})
    return out


# ----------------------------------------------------------------------------- 2. arcs in one section

def _resample(pts, ds):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]
    if L < 3 * ds:
        return None, None, None
    sn = np.arange(0, L, ds)
    x = np.interp(sn, s, pts[:, 0])
    y = np.interp(sn, s, pts[:, 1])
    segi = np.clip(np.searchsorted(s, sn, side="right") - 1, 0, len(seg) - 1)
    return np.c_[x, y], sn, segi


def _curvature(Q, nrm_s, prm, sigma_k=None):
    ds = prm.ds_mm
    sp = prm.sigma_pos_mm / ds
    x = gaussian_filter1d(Q[:, 0], sp, mode="nearest")
    y = gaussian_filter1d(Q[:, 1], sp, mode="nearest")
    th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    kg = np.gradient(th) / ds
    T = np.c_[np.cos(th), np.sin(th)]
    nl = np.c_[-T[:, 1], T[:, 0]]
    sg = np.sign(gaussian_filter1d(np.sum(nl * nrm_s, 1), 2.0 / ds, mode="nearest"))
    k = -kg * sg
    sk = prm.sigma_k_mm if sigma_k is None else sigma_k
    return gaussian_filter1d(k, sk / ds, mode="nearest"), th


def _refine_end(Q, k_small, c, R, a, b, end, prm, arc_rms, k_stop, th=None, t=1.5):
    """Tangent point at one end of an arc by a 1-D model search on the RAW points: circle (c, R) up to the
    split, the circle's tangent line after it. The outward window stops at the chain end or at a sharp
    feature (small-scale curvature above k_stop). Returns (index, end type, run-out flat length mm)."""
    ds = prm.ds_mm
    n = len(Q)
    L_in = min(3 * prm.sigma_k_mm, 0.3 * (b - a) * ds)
    w_out = 3 * prm.sigma_k_mm + 6.0
    if end == 1:
        i_in = max(a, b - int(L_in / ds))
        j = b
        while j + 1 < n and abs(k_small[j + 1]) < k_stop and (j + 1 - b) * ds <= w_out:
            j += 1
        hit_end = j + 1 >= n
        stopped = (j - b) * ds < w_out            # cut by the chain end or a sharp feature, not by the window
        jj = j - int(round(2.0 / ds)) if stopped else j    # keep clear of that feature's blur / edge rounding
        idx = np.arange(i_in, max(jj, b) + 1)
    else:
        i_in = min(b, a + int(L_in / ds))
        j = a
        while j - 1 >= 0 and abs(k_small[j - 1]) < k_stop and (a - j + 1) * ds <= w_out:
            j -= 1
        hit_end = j - 1 < 0
        stopped = (a - j) * ds < w_out
        jj = j + int(round(2.0 / ds)) if stopped else j
        idx = np.arange(i_in, min(jj, a) - 1, -1)
    P = Q[idx]
    if len(P) < 6:
        return (b if end == 1 else a), ("edge" if hit_end else "sharp"), 0.0
    # local circle near this end (rolled profiles are often multi-radius, e.g. pre-bent ends)
    Lw = min(0.3 * (b - a) * ds, max(20.0, 0.15 * R * max(0.3, (b - a) * ds / R)))
    nw = max(8, int(Lw / ds))
    W = Q[max(a, b - nw):b + 1] if end == 1 else Q[a:min(b, a + nw) + 1]
    if len(W) >= 8:
        cl, Rl, rl = G.fit_circle(W)
        rg = float(np.sqrt(np.mean((np.linalg.norm(W - c, axis=1) - R) ** 2)))
        # only when the global circle clearly misfits this end (multi-radius profile); a short local fit is
        # noisier than the global one and would make the tangent split wander
        if cl is not None and 0.5 * R < Rl < 2.0 * R and rg > 2.0 * max(rl, 1e-3) and rg > 0.05:
            c, R, arc_rms = cl, Rl, max(rl, 1e-3)
    rad = np.linalg.norm(P - c, axis=1)
    e_arc = (rad - R) ** 2
    ca = np.cumsum(e_arc)
    best = None
    for m in range(2, len(P) - 1):
        v = P[m] - c
        T0 = c + R * v / np.linalg.norm(v)
        tau = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        if (P[min(m + 1, len(P) - 1)] - P[m - 1]) @ tau < 0:
            tau = -tau
        D = P[m + 1:] - T0
        # beyond the tangent point the line; points that fall behind it (along tau) count as arc
        along = D @ tau
        perp = D[:, 0] * tau[1] - D[:, 1] * tau[0]
        e_line = np.where(along >= 0, perp ** 2, (np.linalg.norm(P[m + 1:] - c, axis=1) - R) ** 2)
        sse = ca[m] + e_line.sum()
        if best is None or sse < best[0]:
            best = (sse, m, e_line)
    _, m, e_line = best
    flat_len = (len(P) - 1 - m) * ds + (2.0 if stopped else 0.0)
    line_rms = math.sqrt(float(np.mean(e_line))) if len(e_line) else float("inf")
    stop = j  # first index past which the window was cut (chain end or a sharp feature)
    u_turn = False
    if th is not None and not hit_end:
        k_end = int(round((2 * t + 3.0) / ds))
        j2 = min(n - 1, stop + k_end) if end == 1 else max(0, stop - k_end)
        u_turn = abs(th[j2] - th[stop]) > math.radians(150)
    if flat_len >= 8.0 and line_rms < max(0.06, 4 * arc_rms):
        typ = "tangent"
    elif hit_end or u_turn:
        typ = "edge"
    elif flat_len < 8.0:
        typ = "sharp"
    else:
        typ = "other"
    return int(idx[m]), typ, flat_len


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def chain_arcs(pts, nrm, prm: RollParams, t: float) -> list[dict[str, Any]]:
    """Curved constant-curvature runs of one section chain (pieces; sweep >= piece_sweep_deg)."""
    pts = np.asarray(pts, float)
    closed = len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
    Q, sn, segi = _resample(pts, prm.ds_mm)
    if Q is None or sn[-1] < prm.min_arc_mm:
        return []
    nr = G.unit(np.asarray(nrm, float)[segi])
    k, th = _curvature(Q, nr, prm)
    if closed:
        # start the loop at its sharpest point (a sheet-edge U-turn or a sharp bend) so no run wraps
        r = int(np.argmax(np.abs(k)))
        Q, nr = np.roll(Q, -r, 0), np.roll(nr, -r, 0)
        k, th = _curvature(Q, nr, prm)
    k_small, _ = _curvature(Q, nr, prm, sigma_k=1.0)
    ds = prm.ds_mm
    r_min = max(prm.r_min_mm, prm.r_min_t * t)
    k_hi, k_lo = 1.0 / (0.5 * r_min), 1.0 / prm.r_max_mm
    out = []
    for sgn in (1, -1):
        ks = sgn * k
        cand = (ks > k_lo) & (ks < k_hi)
        for i0, i1 in _runs(cand):
            if (i1 - i0) * ds < 0.5 * prm.min_arc_mm:
                continue
            k0 = float(np.median(ks[i0:i1 + 1]))
            # half-maximum boundaries (unbiased tangent point for a curvature step under symmetric blur)
            a, b = i0, i1
            # the longest stretch at >= half the run's curvature, bridging dips <= 2 sigma_k (noise on a large
            # radius); a noise blip farther out in an adjacent flat does not stretch the arc, and a nearer one is
            # inside the tangent-point search window (3 sigma_k) and is removed there
            hm = ks[i0:i1 + 1] >= 0.5 * k0
            gap = int(round(2.0 * prm.sigma_k_mm / ds))
            best, cur0, last = None, None, None
            for q in np.nonzero(hm)[0]:
                if cur0 is None or q - last > gap + 1:
                    if cur0 is not None and (best is None or last - cur0 > best[1] - best[0]):
                        best = (cur0, last)
                    cur0 = q
                last = q
            if cur0 is not None and (best is None or last - cur0 > best[1] - best[0]):
                best = (cur0, last)
            if best is None or best[1] - best[0] < 4:
                continue
            a, b = i0 + best[0], i0 + best[1]
            P = Q[a:b + 1]
            L = (b - a) * ds
            if L < 0.5 * prm.min_arc_mm:
                continue
            m = max(1, int(round(min(1.0, 0.1 * L) / ds)))
            c, R, rms = G.fit_circle(P[m:len(P) - m] if len(P) > 2 * m + 5 else P)
            if c is None or not (r_min <= R <= prm.r_max_mm):
                continue
            ph = G.angle_of(P - c)
            st, sw = G.angular_extent(ph)
            if math.degrees(sw) < prm.piece_sweep_deg:
                continue
            sag = R * (1 - math.cos(min(sw, math.pi) / 2))
            if rms > prm.max_rms_over_sag * sag or rms > prm.max_rms_over_r * R:
                continue
            thirds = np.array_split(ks[a:b + 1], 3)
            mk = [float(np.mean(x)) for x in thirds]
            if min(mk) < prm.uniform_min * max(mk):
                continue
            # convexity check from the fitted centre: convex skin has the centre behind the outward normal
            rad = G.unit(P - c)
            conv = float(np.mean(np.sum(rad * nr[a:b + 1], 1)))
            if conv * sgn < 0.5:
                continue
            # tangent points on the raw geometry (the smoothed-curvature boundary is blurred by neighbours)
            k_stop = max(3.0 / R, 1.0 / r_min)
            a2, ty0, fl0 = _refine_end(Q, k_small, c, R, a, b, 0, prm, rms, k_stop, th, t)
            b2, ty1, fl1 = _refine_end(Q, k_small, c, R, a, b, 1, prm, rms, k_stop, th, t)
            if b2 - a2 < 6:
                continue
            P = Q[a2:b2 + 1]
            m = max(1, int(round(min(1.0, 0.1 * len(P) * ds) / ds)))
            c, R, rms = G.fit_circle(P[m:len(P) - m] if len(P) > 2 * m + 5 else P)
            st, sw = G.angular_extent(G.angle_of(P - c))
            # free-edge ends: the arc runs to the sheet edge; the smoothed curvature of the edge U-turn cuts it
            # short by a few mm, so extend along the raw points that stay on the circle
            tol = max(3.0 * rms, 0.08)
            if ty0 == "edge":
                j = a2
                while j - 1 >= 0 and abs(np.linalg.norm(Q[j - 1] - c) - R) < tol:
                    j -= 1
                fl0 = max(0.0, fl0 - (a2 - j) * ds)
                a2 = j
            if ty1 == "edge":
                j = b2
                while j + 1 < len(Q) and abs(np.linalg.norm(Q[j + 1] - c) - R) < tol:
                    j += 1
                fl1 = max(0.0, fl1 - (j - b2) * ds)
                b2 = j
            P = Q[a2:b2 + 1]
            st, sw = G.angular_extent(G.angle_of(P - c))
            # re-check after the end refinement (the arc may now reach into a sharp bend)
            sag = R * (1 - math.cos(min(sw, math.pi) / 2))
            if not (r_min <= R <= prm.r_max_mm) or rms > prm.max_rms_over_sag * sag or rms > prm.max_rms_over_r * R:
                continue
            out.append({"sign": sgn, "c": c, "R": R, "rms": rms, "start": st, "sweep": sw, "pts": P,
                        "nrm": nr[a2:b2 + 1], "k_profile": ks[a2:b2 + 1], "ends": [ty0, ty1],
                        "flat_len": [fl0, fl1], "L": (b2 - a2) * ds})
    return out


def merge_section_arcs(arcs, prm: RollParams, r_min: float = 0.0):
    """Merge co-circular pieces (same convexity, concentric, same radius) of one section."""
    arcs = sorted(arcs, key=lambda x: -x["sweep"])
    used = [False] * len(arcs)
    out = []
    for i, A in enumerate(arcs):
        if used[i]:
            continue
        grp = [A]
        used[i] = True
        for j in range(i + 1, len(arcs)):
            B = arcs[j]
            if used[j] or B["sign"] != A["sign"]:
                continue
            same = np.linalg.norm(B["c"] - A["c"]) < 0.05 * A["R"] + 1.0 and abs(B["R"] - A["R"]) < 0.04 * A["R"] + 0.5
            if not same:
                # a piece cut off by a flat spot / dent fits its own circle badly: test its points on A's circle
                dev = np.abs(np.linalg.norm(B["pts"] - A["c"], axis=1) - A["R"])
                same = np.median(dev) < max(0.6, 0.01 * A["R"]) and np.percentile(dev, 90) < max(2.0, 0.03 * A["R"])
            if same:
                grp.append(B)
                used[j] = True
        if len(grp) == 1:
            M = dict(A)
            M["cover_pts"] = A["pts"]
        else:
            P = np.vstack([g["pts"] for g in grp])
            c, R, rms = G.fit_circle(P)
            st, sw = G.angular_extent(G.angle_of(P - c))
            M = {"sign": A["sign"], "c": c, "R": R, "rms": rms, "start": st, "sweep": sw, "pts": P,
                 "nrm": np.vstack([g["nrm"] for g in grp]), "k_profile": np.concatenate([g["k_profile"] for g in grp]),
                 "L": sum(g["L"] for g in grp), "cover_pts": P}
            # end types: from the pieces holding the extreme angles
            ph0 = [G.wrap_from(G.angle_of(g["pts"] - c), st) for g in grp]
            lo = int(np.argmin([p.min() for p in ph0]))
            hi = int(np.argmax([p.max() for p in ph0]))
            M["ends"] = [_end_at(grp[lo], c, st, low=True), _end_at(grp[hi], c, st, low=False)]
            M["flat_len"] = [_end_at(grp[lo], c, st, low=True, key="flat_len"), _end_at(grp[hi], c, st, low=False, key="flat_len")]
            M["n_pieces"] = len(grp)
        sag = M["R"] * (1 - math.cos(min(M["sweep"], math.pi) / 2))
        if (math.degrees(M["sweep"]) >= prm.min_sweep_deg and M["L"] >= prm.min_arc_mm and M["R"] >= r_min
                and M["rms"] <= max(prm.max_rms_over_sag * sag, 1e-9) and M["rms"] <= prm.max_rms_over_r * M["R"]):
            out.append(M)
    return out


def _end_at(arc, c, st, low, key="ends"):
    """End type (or flat length) of a piece at its low-angle or high-angle end (pieces store chain order)."""
    v = arc.get(key, [None, None])
    ph = G.wrap_from(G.angle_of(arc["pts"] - c), st)
    first_is_low = ph[0] < ph[-1]
    return v[0] if (first_is_low == low) else v[1]


def orient_arc(arc):
    """Express an arc with start angle + sweep (counter-clockwise) and the end types in that order."""
    c, st = arc["c"], arc["start"]
    ph = G.wrap_from(G.angle_of(arc["pts"] - c), st)
    if arc.get("n_pieces", 1) == 1 and ph[0] > ph[-1]:
        arc["ends"] = arc["ends"][::-1]
        if "flat_len" in arc:
            arc["flat_len"] = arc["flat_len"][::-1]
    return arc


# ----------------------------------------------------------------------------- 3. tracking

def section_arcs(mesh, u, step, prm, t):
    levels, per = SL.slice_family(mesh, u, step)
    res = []
    for k, chains in enumerate(per):
        arcs = []
        for ch in chains:
            arcs += chain_arcs(ch["pts"], ch["nrm"], prm, t)
        merged = [orient_arc(a) for a in merge_section_arcs(arcs, prm, max(prm.r_min_mm, prm.r_min_t * t))]
        if merged and chains:
            P = np.vstack([ch["pts"][:-1] for ch in chains])
            Nn = G.unit(np.vstack([ch["nrm"] for ch in chains]))
            merged = [extend_across_gaps(a, P, Nn) for a in merged]
        res.append(merged)
    return levels, res


def extend_across_gaps(a, P, N, max_gap_deg=None):
    """Holes, cut-outs or unscanned strips split a roll's section into pieces too short to be arcs on their own.
    Walk outward from each arc end over 1-degree bins of the section's raw points that lie on the arc's circle
    (|rho - R| < max(3 rms, 0.15 mm)) with the same skin orientation; across an empty gap of at most
    max(10 deg, 30 mm / R) the arc continues if on-circle points resume. Contiguous points right at an end are
    NOT used (the tangent-point search already placed that end), so only real gaps are bridged."""
    c, R = a["c"], a["R"]
    X = P - c
    rho = np.linalg.norm(X, axis=1)
    rad = X / np.maximum(rho, 1e-9)[:, None]
    facing = np.sum(rad * N, 1) * a["sign"]
    on = (np.abs(rho - R) < max(3 * a["rms"], 0.15)) & (facing > 0.7)
    if not on.any():
        return a
    ph_all = (G.angle_of(X) - a["start"]) % (2 * math.pi)        # 0..2pi from the arc start, CCW
    ph = ph_all[on]
    bw = math.radians(1.0)
    nb = int(round(2 * math.pi / bw))
    occ = np.zeros(nb, bool)
    occ[np.clip((ph / bw).astype(int), 0, nb - 1)] = True
    # any material near the circle (either skin, or departing from it): a real gap has none
    # material of the SAME skin that is off the circle (a departing flat / bend); edge walls and the other skin
    # (normals not facing like this skin) do not count
    tol = max(3 * a["rms"], 0.15)
    near = (np.abs(rho - R) < 6.0) & (np.abs(rho - R) >= tol) & (facing > 0.7)
    anyo = np.zeros(nb, bool)
    anyo[np.clip((ph_all[near] / bw).astype(int), 0, nb - 1)] = True
    gmax = int(max(10.0, math.degrees(30.0 / R)))
    sw_b = int(a["sweep"] / bw)

    def walk(seq):
        """seq: bin indices going outward; returns the last bin of the farthest on-circle run reached across
        gaps <= gmax (None if nothing beyond the contiguous run at the end)."""
        k = 0
        while k < len(seq) and occ[seq[k]]:
            k += 1                      # contiguous with the end: already judged by the tangent-point search
        far, gap, run = None, 0, 0
        while k < len(seq):
            if occ[seq[k]]:
                run += 1
                if run >= 2:
                    far = k
                gap = 0
            else:
                if anyo[seq[k]] and not occ[seq[k]]:
                    break                   # material that is off the circle: the roll ended here, not a gap
                run = 0
                gap += 1
                if gap > gmax:
                    break
            k += 1
        return far

    room = nb - sw_b - 2
    fwd = list(range(sw_b + 1, sw_b + 1 + room))
    f = walk(fwd)
    hi = fwd[f] if f is not None else sw_b
    bwd = [nb - 1 - i for i in range(nb - 1 - hi - 1)]
    b_ = walk(bwd)
    lo = (bwd[b_] - nb) if b_ is not None else 0
    if hi == sw_b and lo == 0:
        return a
    b = dict(a)
    new_lo = lo * bw if lo < 0 else 0.0
    new_hi = (hi + 1) * bw if hi > sw_b else a["sweep"]
    b["start"] = a["start"] + new_lo
    b["sweep"] = new_hi - new_lo
    ends = list(a["ends"])
    if lo < 0:
        ends[0] = "edge"
    if hi > sw_b:
        ends[1] = "edge"
    b["ends"] = ends
    b["cover_pts"] = np.vstack([a["cover_pts"], P[on][(ph > a["sweep"] + bw) & ((ph <= new_hi) | (ph >= 2 * math.pi + new_lo))]])
    b["gap_extended_deg"] = math.degrees(b["sweep"] - a["sweep"])
    return b


def _ang_overlap(a, b):
    """Fraction of the smaller angular interval covered by the other."""
    s = min(a["sweep"], b["sweep"])
    if s <= 0:
        return 0.0
    n = 180
    xa = (a["start"] + np.linspace(0, a["sweep"], n)) % (2 * math.pi)
    d = G.wrap_from(xa, b["start"])
    fa = np.mean(d <= b["sweep"] + 1e-9)
    xb = (b["start"] + np.linspace(0, b["sweep"], n)) % (2 * math.pi)
    fb = np.mean(G.wrap_from(xb, a["start"]) <= a["sweep"] + 1e-9)
    return float(max(fa, fb))


def track(levels, per, step, max_gap=3):
    tracks: list[list[tuple[int, dict]]] = []
    open_: list[int] = []
    for k, arcs in enumerate(per):
        taken = set()
        cands = []
        for ti in open_:
            kl, A = tracks[ti][-1]
            if k - kl > max_gap + 1:
                continue
            for j, B in enumerate(arcs):
                if B["sign"] != A["sign"]:
                    continue
                dc = np.linalg.norm(B["c"] - A["c"])
                tol_c = 0.08 * A["R"] + 2.0 + (k - kl) * step * math.tan(math.radians(3))
                if dc > tol_c or abs(B["R"] - A["R"]) > 0.1 * A["R"] + 1.0:
                    continue
                if _ang_overlap(A, B) < 0.3:
                    continue
                cands.append((dc / tol_c, ti, j))
        cands.sort()
        used_t = set()
        for _, ti, j in cands:
            if ti in used_t or j in taken:
                continue
            tracks[ti].append((k, arcs[j]))
            used_t.add(ti)
            taken.add(j)
        for j, B in enumerate(arcs):
            if j not in taken:
                tracks.append([(k, B)])
                open_.append(len(tracks) - 1)
        open_ = [ti for ti in open_ if k - tracks[ti][-1][0] <= max_gap]
    return tracks


def join_tracks(tracks, max_gap_levels=8):
    """Join same-sign tracks that continue one another along the axis on the same cylinder (a stepped
    outline or a hole can break a track). Where they overlap, the arc with the larger sweep is kept."""
    tracks = [list(t) for t in tracks]
    changed = True
    while changed:
        changed = False
        tracks.sort(key=lambda t: t[0][0])
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                if i == j or not tracks[i] or not tracks[j]:
                    continue
                A, B = tracks[i], tracks[j]
                if A[0][1]["sign"] != B[0][1]["sign"]:
                    continue
                a1, b0 = A[-1][0], B[0][0]
                if not (A[0][0] < b0 and B[-1][0] > a1 and b0 - a1 <= max_gap_levels):
                    continue
                ea = [a for _, a in A[-3:]]
                eb = [a for _, a in B[:3]]
                Ra, Rb = np.median([a["R"] for a in ea]), np.median([a["R"] for a in eb])
                ca = np.median([a["c"] for a in ea], 0)
                cb = np.median([a["c"] for a in eb], 0)
                if abs(Ra - Rb) > 0.1 * Ra + 2 or np.linalg.norm(ca - cb) > 0.1 * Ra + 3:
                    continue
                if _ang_overlap({"start": ea[-1]["start"], "sweep": ea[-1]["sweep"]},
                                {"start": eb[0]["start"], "sweep": eb[0]["sweep"]}) < 0.3:
                    continue
                d = {}
                for k, a in A + B:
                    if k not in d or a["sweep"] > d[k]["sweep"]:
                        d[k] = a
                tracks[i] = sorted(d.items(), key=lambda x: x[0])
                tracks[j] = []
                changed = True
        tracks = [t for t in tracks if t]
    return tracks


def _track_summary(tr, levels):
    R = np.median([a["R"] for _, a in tr])
    c = np.median(np.array([a["c"] for _, a in tr]), 0)
    return R, c, levels[tr[0][0]], levels[tr[-1][0]]


def pair_tracks(tracks, levels, t, prm, step):
    """Group convex/concave tracks that are concentric and ~t apart into regions."""
    good = [tr for tr in tracks if len(tr) >= prm.min_stations and (len(tr) - 1) * step >= prm.min_len_mm * 0.8]
    conv = [tr for tr in good if tr[0][1]["sign"] > 0]
    conc = [tr for tr in good if tr[0][1]["sign"] < 0]
    regions, used = [], set()
    for a in conv:
        da = dict(a)
        best = None
        for j, b in enumerate(conc):
            if j in used:
                continue
            # per-level concentric test: same section, shared centre, radii ~t apart
            common = [k for k, _ in b if k in da]
            if len(common) < max(3, 0.3 * min(len(a), len(b))):
                continue
            db = dict(b)
            ok = 0
            dts = []
            for k in common:
                A, B = da[k], db[k]
                dt = A["R"] - B["R"]
                if np.linalg.norm(A["c"] - B["c"]) < 0.02 * A["R"] + 1.0 and abs(dt - t) < max(1.0, 0.6 * t) + 0.01 * A["R"]:
                    ok += 1
                    dts.append(abs(dt - t))
            if ok < 0.5 * len(common):
                continue
            sc = -ok
            if best is None or sc < best[0]:
                best = (sc, j)
        if best is not None:
            used.add(best[1])
            regions.append({"outer": a, "inner": conc[best[1]]})
        else:
            regions.append({"outer": a, "inner": None})
    for j, b in enumerate(conc):
        if j not in used:
            regions.append({"outer": None, "inner": b})
    return regions


# ----------------------------------------------------------------------------- 4/5. measurement

def _lift(c2, level, u, e1, e2):
    return c2[0] * e1 + c2[1] * e2 + level * u


def _stations(reg, levels):
    """Per station dict: level, outer arc, inner arc."""
    st: dict[int, dict] = {}
    for key in ("outer", "inner"):
        tr = reg.get(key)
        if tr is None:
            continue
        for k, a in tr:
            st.setdefault(k, {"k": k, "level": float(levels[k])})[key] = a
    return [st[k] for k in sorted(st)]


def _station_centre(s):
    if "outer" in s and "inner" in s:
        c, ro, ri, rms = G.fit_concentric(s["outer"]["pts"], s["inner"]["pts"])
        s["concentric"] = {"c": c, "R_out": ro, "R_in": ri, "rms": rms}
        return c
    a = s.get("outer") or s.get("inner")
    return a["c"]


def region_axis(reg, levels, u, e1, e2):
    S = _stations(reg, levels)
    C3 = np.array([_lift(_station_centre(s), s["level"], u, e1, e2) for s in S])
    p0, d, rms = G.fit_line3(C3)
    if d @ u < 0:
        d = -d
    return p0, d, S, C3


def detect_regions(mesh, prm: RollParams, t: float, axes=None, log=print):
    """Detection only (cheap): candidate axes -> sections -> arcs -> tracked regions (no refinement)."""
    axes = axis_candidates(mesh, prm) if axes is None else axes
    found = []
    for ax in axes:
        u = G.unit(ax["axis"])
        h = mesh.V @ u
        ext = float(h.max() - h.min())
        step = prm.step_mm or float(np.clip(ext / 90.0, 1.5, 8.0))
        levels, per = section_arcs(mesh, u, step, prm, t)
        tracks = join_tracks(track(levels, per, step))
        regs = pair_tracks(tracks, levels, t, prm, step)
        for r in regs:
            r.update({"u": u, "step": step, "levels": levels, "axis_cand": ax})
        regs = merge_coaxial(regs)
        log(f"  axis {np.round(u, 3)} band {ax.get('band_frac', 0):.3f}: {sum(len(a) for a in per)} arcs, "
            f"{len(tracks)} tracks, {len(regs)} regions")
        found += regs
    return dedupe_regions(found)


def _axial_range(r):
    ks = [k for key in ("outer", "inner") if r.get(key) for k, _ in r[key]]
    return r["levels"][min(ks)], r["levels"][max(ks)]


def merge_coaxial(regs, max_gap_mm=None):
    """Regions of one slicing axis that lie on the same cylinder (same axis line and radius) and follow each
    other along the axis (a welded bulkhead, a hole or a pairing break can cut a roll) become one region."""
    regs = list(regs)
    changed = True
    while changed:
        changed = False
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                A, B = regs[i], regs[j]
                if A is None or B is None:
                    continue
                Ra, ca3, _, _ = _reg_key(A)
                Rb, cb3, _, _ = _reg_key(B)
                u = A["u"]
                dd = cb3 - ca3
                perp = np.linalg.norm(dd - (dd @ u) * u)
                if abs(Ra - Rb) > 0.05 * Ra + 1.5 or perp > 0.05 * Ra + 3:
                    continue
                a0, a1 = _axial_range(A)
                b0, b1 = _axial_range(B)
                gap = max(a0, b0) - min(a1, b1)
                if gap > (max_gap_mm or max(30.0, 0.5 * Ra)):
                    continue
                M = {**A}
                for key in ("outer", "inner"):
                    d = {}
                    for src in (A.get(key), B.get(key)):
                        for k, a in (src or []):
                            if k not in d or a["sweep"] > d[k]["sweep"]:
                                d[k] = a
                    M[key] = sorted(d.items(), key=lambda x: x[0]) or None
                regs[i], regs[j] = M, None
                changed = True
        regs = [r for r in regs if r is not None]
    return regs


def _reg_key(r):
    tr = r["outer"] or r["inner"]
    R, c, l0, l1 = _track_summary(tr, r["levels"])
    e1, e2 = G.frame(r["u"])
    return R, _lift(c, 0.5 * (l0 + l1), r["u"], e1, e2), (l1 - l0), len(tr)


def dedupe_regions(regs):
    """The same roll can be found from several near-parallel candidate axes: keep the longest track."""
    keyed = sorted(regs, key=lambda r: -(_reg_key(r)[3] * (2 if (r["outer"] and r["inner"]) else 1)))
    out = []
    for r in keyed:
        R, c3, L, n = _reg_key(r)
        dup = False
        for o in out:
            Ro, co, Lo, no = _reg_key(o)
            if abs(r["u"] @ o["u"]) > math.cos(math.radians(12)) and abs(R - Ro) < 0.15 * Ro + 2:
                dd = c3 - co
                perp = np.linalg.norm(dd - (dd @ o["u"]) * o["u"])
                # axial overlap of the two regions along o's axis
                def rng(x):
                    tr = x["outer"] or x["inner"]
                    lv = x["levels"]
                    e1_, e2_ = G.frame(x["u"])
                    pts = [_lift(a["c"], lv[k], x["u"], e1_, e2_) @ o["u"] for k, a in tr]
                    return min(pts), max(pts)
                r0, r1 = rng(r)
                q0, q1 = rng(o)
                ov = min(r1, q1) - max(r0, q0)
                if perp < 0.15 * Ro + 3 and ov > 0.3 * min(r1 - r0, q1 - q0):
                    dup = True
                    break
        if not dup:
            out.append(r)
    return out


def _match_region(regs, R_ref, line_p0, line_d, u, e1, e2):
    best = None
    for r in regs:
        tr = r["outer"] or r["inner"]
        R, c, l0, l1 = _track_summary(tr, r["levels"])
        c3 = _lift(c, 0.5 * (l0 + l1), u, e1, e2)
        dd = c3 - line_p0
        perp = np.linalg.norm(dd - (dd @ line_d) * line_d)
        Rm = R if r["outer"] is None else _track_summary(r["outer"], r["levels"])[0]
        sc = perp / (0.1 * R_ref + 3) + abs(Rm - R_ref) / (0.2 * R_ref + 2) - 0.01 * len(tr)
        if perp < 0.3 * R_ref + 5 and abs(Rm - R_ref) < 0.3 * R_ref + 2 and (best is None or sc < best[0]):
            best = (sc, r)
    return None if best is None else best[1]


def refine_region(mesh, reg, prm: RollParams, t: float, log=print):
    """Re-slice perpendicular to the axis estimated from the station centres until it stops moving."""
    u = reg["u"]
    e1, e2 = G.frame(u)
    p0, d, S, C3 = region_axis(reg, reg["levels"], u, e1, e2)
    R_ref = float(np.median([(s.get("outer") or s.get("inner"))["R"] for s in S]))
    hist = []
    for it in range(prm.refine_iters):
        ang = math.degrees(math.acos(min(1.0, abs(float(d @ u)))))
        hist.append(round(ang, 4))
        if ang < 0.05:
            break
        u = d
        e1, e2 = G.frame(u)
        step = reg["step"]
        levels, per = section_arcs(mesh, u, step, prm, t)
        regs = pair_tracks(join_tracks(track(levels, per, step)), levels, t, prm, step)
        for r in regs:
            r.update({"u": u, "step": step, "levels": levels})
        nr = _match_region(regs, R_ref, p0, d, u, e1, e2)
        if nr is None:
            log("   refine: region lost (axis did not converge)")
            reg["unconverged"] = True
            break
        reg = {**reg, **nr}
        p0, d, S, C3 = region_axis(reg, reg["levels"], u, e1, e2)
    reg["axis_refine_deg"] = hist
    if hist and hist[-1] >= 0.5 and len(hist) == prm.refine_iters:
        reg["unconverged"] = True
    return reg


def _boot(fn, n, B, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = np.sort(rng.integers(0, n, n))
        try:
            v = fn(idx)
        except Exception:  # noqa: BLE001
            continue
        if v is not None and np.all(np.isfinite(v)):
            vals.append(v)
    return np.array(vals)


def _linfit(x, y):
    A = np.c_[x - x.mean(), np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _local_radius(P, c, st, sw, lo, hi):
    ph = G.wrap_from(G.angle_of(P - c), st) / max(sw, 1e-9)
    m = (ph >= lo) & (ph <= hi)
    if m.sum() < 8:
        return float("nan")
    _, R, _ = G.fit_circle(P[m])
    return R


def _flat_profile(a, step_deg=2.0, win_mm=6.0):
    """Radial deviation of one skin's arc points from the station circle, binned by angle; flat spots
    sit inside the circle (a chord is closer to the centre than the arc it replaces)."""
    P, c, R, st, sw = a["pts"], a["c"], a["R"], a["start"], a["sweep"]
    ph = G.wrap_from(G.angle_of(P - c), st)
    dev = np.linalg.norm(P - c, axis=1) - R
    nb = max(3, int(math.degrees(sw) / step_deg))
    edges = np.linspace(0, sw, nb + 1)
    idx = np.clip(np.searchsorted(edges, ph, side="right") - 1, 0, nb - 1)
    prof = np.full(nb, np.nan)
    for b in range(nb):
        m = idx == b
        if m.sum():
            prof[b] = np.median(dev[m])
    return np.degrees(0.5 * (edges[:-1] + edges[1:])), prof


def _union_skins(s, a):
    """Run-out angles from the union of both skins (as the CAD sweep is defined): where one skin is cut short by
    a scan hole the other skin still shows where the roll ends. Radius / points stay those of the main skin."""
    if not ("outer" in s and "inner" in s):
        return a
    c = a["c"]
    lo = hi = None
    for x in (s["outer"], s["inner"]):
        p0 = x["c"] + x["R"] * np.array([math.cos(x["start"]), math.sin(x["start"])])
        a0 = (math.atan2(p0[1] - c[1], p0[0] - c[0]) - a["start"] + math.pi) % (2 * math.pi) - math.pi
        a1 = a0 + x["sweep"]
        if lo is None or a0 < lo[0]:
            lo = (a0, x["ends"][0])
        if hi is None or a1 > hi[0]:
            hi = (a1, x["ends"][1])
    if lo[0] >= -1e-9 and hi[0] <= a["sweep"] + 1e-9:
        return a
    # only where the main skin stopped at an edge / hole: a main-skin end at a tangent point or a sharp
    # feature is a geometric end and is kept (the other skin's end search can overshoot into a bend)
    lo_v = min(lo[0], 0.0) if a["ends"][0] == "edge" else 0.0
    hi_v = max(hi[0], a["sweep"]) if a["ends"][1] == "edge" else a["sweep"]
    if lo_v == 0.0 and hi_v == a["sweep"]:
        return a
    b = dict(a)
    b["start"] = a["start"] + lo_v
    b["sweep"] = hi_v - lo_v
    b["ends"] = [lo[1] if lo_v < 0 else a["ends"][0], hi[1] if hi_v > a["sweep"] else a["ends"][1]]
    b["flat_len"] = a.get("flat_len")
    return b


def measure_region(reg, prm: RollParams, t_d1: float | None, t_src: str = "D1", _axis=None) -> dict[str, Any]:
    """Two passes: the axis is first the TLS line through the full-arc station centres; when a common
    angular window exists, it is re-fitted through the common-window centres (the same part of the profile
    at every station), which a multi-radius profile or a stepped outline cannot bias."""
    if _axis is None:
        first = measure_region(reg, prm, t_d1, t_src, _axis=False)
        cc = first.pop("_common_centres")
        ok = np.isfinite(cc).all(1)
        if ok.sum() >= max(4, 0.5 * len(cc)):
            p0c, dc, _ = G.fit_line3(cc[ok])
            res = measure_region(reg, prm, t_d1, t_src, _axis=(p0c, dc, cc))
            res.pop("_common_centres")
            res["axis"]["basis"] = "common-window station centres"
            res["axis"]["first_pass_full_arc_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(
                np.dot(dc, first["axis"]["direction"])))))))
            return res
        first["axis"]["basis"] = "full-arc station centres"
        return first
    u = reg["u"]
    e1, e2 = G.frame(u)
    levels = reg["levels"]
    p0, d, S, C3 = region_axis(reg, levels, u, e1, e2)
    if _axis:
        p0, d, cc = _axis
        if d @ u < 0:
            d = -d
        C3 = np.where(np.isfinite(cc).all(1)[:, None], cc, C3)
    n = len(S)
    lv = np.array([s["level"] for s in S])
    # per station radii
    R_in = np.full(n, np.nan)
    R_out = np.full(n, np.nan)
    t_st = np.full(n, np.nan)
    for i, s in enumerate(S):
        if "concentric" in s:
            R_in[i], R_out[i] = s["concentric"]["R_in"], s["concentric"]["R_out"]
            t_st[i] = R_out[i] - R_in[i]
        else:
            if "inner" in s:
                R_in[i] = s["inner"]["R"]
            if "outer" in s:
                R_out[i] = s["outer"]["R"]
    both = np.isfinite(t_st)
    t_pair = float(np.median(t_st[both])) if both.sum() >= 3 else None
    t_use = t_pair if t_pair is not None else t_d1
    t_used_src = "two-skin stations" if t_pair is not None else t_src
    t_nan = t_use if t_use else np.nan     # no thickness: the unscanned skin's radius is not measurable
    R_in_full = np.where(np.isfinite(R_in), R_in, R_out - t_nan)
    R_out_full = np.where(np.isfinite(R_out), R_out, R_in + t_nan)
    meas_skin = "both" if both.sum() >= 3 else ("inner" if np.isfinite(R_in).sum() >= np.isfinite(R_out).sum() else "outer")
    # station frame: fixed reference direction around the refined axis
    f1, f2 = G.frame(d)
    sk_main = "inner" if np.isfinite(R_in).sum() >= np.isfinite(R_out).sum() else "outer"

    def arc_of(s):
        return s.get(sk_main) or s.get("outer") or s.get("inner")

    sweeps, chord, ang0, ang1, rms_st, cov_ang = [], [], [], [], [], []
    extent, flat_end = [], []
    ro_pts = {0: [], 1: []}
    ro_type = {0: [], 1: []}
    sweeps_skin = {"inner": [], "outer": []}
    flat_rows = []
    zone = {"start": [], "mid": [], "end": []}
    for i, s in enumerate(S):
        for sk in ("inner", "outer"):
            if sk in s:
                sweeps_skin[sk].append(math.degrees(s[sk]["sweep"]))
        a = _union_skins(s, arc_of(s))
        c3 = _lift(a["c"], s["level"], u, e1, e2)
        pA = a["c"] + a["R"] * np.array([math.cos(a["start"]), math.sin(a["start"])])
        pB = a["c"] + a["R"] * np.array([math.cos(a["start"] + a["sweep"]), math.sin(a["start"] + a["sweep"])])
        PA, PB = _lift(pA, s["level"], u, e1, e2), _lift(pB, s["level"], u, e1, e2)
        sweeps.append(math.degrees(a["sweep"]))
        chord.append(float(np.linalg.norm(PA - PB)))
        for j, Pp in enumerate((PA, PB)):
            ro_pts[j].append(Pp)
            ro_type[j].append(a["ends"][j])
        # angular positions of the run-outs about the refined axis (for twist)
        for lst, Pp in ((ang0, PA), (ang1, PB)):
            v = Pp - p0
            v = v - (v @ d) * d
            lst.append(math.atan2(v @ f2, v @ f1))
        rms_st.append(a["rms"])
        fl = a.get("flat_len") or [None, None]
        ext = math.degrees(a["sweep"])
        for e_, f_ in zip(a["ends"], fl):
            if e_ == "edge" and f_:
                ext += math.degrees(math.atan2(f_, a["R"]))
        extent.append(ext)
        flat_end.append([f_ if (e_ == "edge" and f_ is not None) else np.nan for e_, f_ in zip(a["ends"], fl)])
        # angular coverage of the arc: 1-degree bins (or the mesh-sampling angle, if coarser) holding points
        ph = G.wrap_from(G.angle_of(a["cover_pts"] - a["c"]), a["start"])
        bw = max(math.radians(1.0), 2.5 * prm.ds_mm / a["R"])
        nb = max(1, int(a["sweep"] / bw))
        h = np.bincount(np.clip((ph / a["sweep"] * nb).astype(int), 0, nb - 1), minlength=nb)
        cov_ang.append(float(np.mean(h > 0)))
        angs, prof = _flat_profile(a)
        flat_rows.append(prof)
        P = a["cover_pts"]
        zone["start"].append(_local_radius(P, a["c"], a["start"], a["sweep"], 0.0, 0.2))
        zone["mid"].append(_local_radius(P, a["c"], a["start"], a["sweep"], 0.2, 0.8))
        zone["end"].append(_local_radius(P, a["c"], a["start"], a["sweep"], 0.8, 1.0))
    sweeps, chord, rms_st = map(np.array, (sweeps, chord, rms_st))
    # common angular window about the refined axis: the same part of the profile at every station, so a
    # multi-radius profile or a stepped outline does not masquerade as taper
    def phis(s, sk):
        a = s[sk]
        P3 = a["cover_pts"][:, :1] * e1 + a["cover_pts"][:, 1:2] * e2 + s["level"] * u - p0
        return np.arctan2(P3 @ f2, P3 @ f1)
    ref = None
    lims = []
    for s in S:
        a = arc_of(s)
        sk = sk_main if sk_main in s else ("outer" if "outer" in s else "inner")
        ph = phis(s, sk)
        if ref is None:
            ref = float(np.angle(np.mean(np.exp(1j * ph))))
        ph = np.mod(ph - ref + math.pi, 2 * math.pi) - math.pi
        lims.append((float(ph.min()), float(ph.max())))
    lims = np.array(lims)
    win_lo, win_hi = float(np.percentile(lims[:, 0], 80)), float(np.percentile(lims[:, 1], 20))
    Rc = {"inner": np.full(n, np.nan), "outer": np.full(n, np.nan)}
    Cc = {"inner": np.full((n, 3), np.nan), "outer": np.full((n, 3), np.nan)}
    if win_hi - win_lo >= math.radians(prm.min_sweep_deg):
        for i, s in enumerate(S):
            for sk in ("inner", "outer"):
                if sk not in s:
                    continue
                ph = np.mod(phis(s, sk) - ref + math.pi, 2 * math.pi) - math.pi
                msk = (ph >= win_lo) & (ph <= win_hi)
                if msk.sum() >= 10 and np.ptp(ph[msk]) > 0.8 * (win_hi - win_lo):
                    c2, Rc[sk][i], _ = G.fit_circle(s[sk]["cover_pts"][msk])
                    Cc[sk][i] = _lift(c2, s["level"], u, e1, e2)
    Rc_in = np.where(np.isfinite(Rc["inner"]), Rc["inner"], Rc["outer"] - t_nan)
    if not np.isfinite(Rc_in).any() and np.isfinite(Rc["outer"]).any():
        Rc_in = Rc["outer"]          # taper only needs one skin
    common = {"window_deg": [math.degrees(win_lo), math.degrees(win_hi)], "width_deg": math.degrees(win_hi - win_lo),
              "R_inner": None, "stations": int(np.isfinite(Rc_in).sum())}
    ang0, ang1 = np.unwrap(np.array(ang0)), np.unwrap(np.array(ang1))
    # axis straightness: centre deviations from the TLS line (rms, max, bow sagitta from a quadratic)
    sproj = (C3 - p0) @ d
    resid = (C3 - p0) - np.outer(sproj, d)
    dev = np.linalg.norm(resid, axis=1)
    bow = None
    if n >= 6:
        q = np.polyfit(sproj - sproj.mean(), resid, 2)
        Ls = sproj.max() - sproj.min()
        bow = float(np.linalg.norm(q[0]) * (Ls / 2) ** 2)
    L = float(sproj.max() - sproj.min())
    B = prm.n_boot

    def m_se(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return None
        bs = _boot(lambda idx: np.mean(x[idx]), len(x), B)
        return {"mean": float(np.mean(x)), "u_boot": float(np.std(bs)) if len(bs) else None,
                "sd_along": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x)), "n": int(len(x))}

    def slope_unc(x, y, scale=1000.0):
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 4:
            return None
        sl, _ = _linfit(x, y)
        bs = _boot(lambda idx: _linfit(x[idx], y[idx])[0] if np.ptp(x[idx]) > 0 else None, len(x), B)
        return {"value": sl * scale, "u_boot": float(np.std(bs)) * scale if len(bs) else None}

    # axis direction uncertainty (bootstrap of the TLS line through the station centres)
    def ax_fit(idx):
        _, dd, _ = G.fit_line3(C3[idx])
        return dd if dd @ d > 0 else -dd
    bd = _boot(ax_fit, n, B)
    ax_unc = float(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(bd @ d, -1, 1)) ** 2)))) if len(bd) else None
    mid = p0 + d * (0.5 * (sproj.max() + sproj.min()))

    def _line_fit(X, types):
        c, dd, rms = G.fit_line3(X)
        if dd @ d < 0:
            dd = -dd
        sp = (X - c) @ dd
        bs = _boot(lambda idx: G.fit_line3(X[idx])[0], len(X), B)
        pu = None
        if len(bs):
            dv = bs - c
            dv = dv - np.outer(dv @ dd, dd)
            pu = float(np.sqrt(np.mean(np.sum(dv ** 2, 1))))
        return {"start": (c + dd * sp.min()).tolist(), "end": (c + dd * sp.max()).tolist(), "rms_mm": rms,
                "length_mm": float(sp.max() - sp.min()), "n_stations_used": int(len(X)),
                "angle_to_axis_deg": float(math.degrees(math.acos(min(1.0, abs(dd @ d))))),
                "type_majority": max(set(types), key=types.count),
                "tangent_fraction": float(np.mean([x == "tangent" for x in types])), "u_pos_mm": pu}

    ro_out = {0: set(), 1: set()}
    seg_slopes: list = []

    def runout_line(j):
        """Run-out line(s) at one end of the roll: stations are cut into contiguous segments where the
        run-out position (angle about the axis) is continuous and the end type (tangent vs not) is the same;
        one line per segment (>= 3 stations). Primary = the longest tangent segment, else the longest."""
        psi = ang0 if j == 0 else ang1
        types_all = ro_type[j]
        X_all = np.array(ro_pts[j])
        Rref = float(np.nanmedian(R_in_full)) if np.isfinite(R_in_full).any() else float(np.nanmedian(R_out_full))
        jump = max(math.radians(5.0), 15.0 / max(Rref, 1.0))
        # drop isolated outlier stations (a hole or a local break): far from both neighbours
        keep = np.ones(n, bool)
        for i in range(1, n - 1):
            if abs(psi[i] - psi[i - 1]) > jump and abs(psi[i] - psi[i + 1]) > jump and abs(psi[i + 1] - psi[i - 1]) < jump:
                keep[i] = False
        idx = np.nonzero(keep)[0]
        segs, cur = [], [int(idx[0])]
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if abs(psi[b_] - psi[a_]) > jump:
                segs.append(cur)
                cur = []
            cur.append(int(b_))
        segs.append(cur)
        out = []
        min_seg = max(3, int(math.ceil(0.08 * n)))
        for sg in segs:
            if len(sg) < min_seg:
                for i in sg:
                    ro_out[j].add(int(i))       # too short to be a run-out line: count its stations as outliers
            if len(sg) >= min_seg:
                sg = np.array(sg)
                # robust: drop stations whose run-out angle is off the segment's linear trend (an arc cut short
                # by a hole, a local break); 3 x MAD, at least 1 deg / 3 mm
                inl = np.ones(len(sg), bool)
                for _ in range(3):
                    if inl.sum() < 3:
                        break
                    A = np.c_[lv[sg][inl] - lv[sg][inl].mean(), np.ones(inl.sum())]
                    co = np.linalg.lstsq(A, psi[sg][inl], rcond=None)[0]
                    r = psi[sg] - (co[0] * (lv[sg] - lv[sg][inl].mean()) + co[1])
                    mad = 1.4826 * np.median(np.abs(r[inl] - np.median(r[inl])))
                    inl = np.abs(r) <= max(3 * mad, math.radians(1.0), 3.0 / max(Rref, 1.0))
                sgi = sg[inl] if inl.sum() >= 3 else sg
                for i in sg[~inl]:
                    ro_out[j].add(int(i))
                f = _line_fit(X_all[sgi], [types_all[i] for i in sgi])
                f["stations"] = [float(lv[sgi[0]]), float(lv[sgi[-1]])]
                f["angle_about_axis_deg"] = float(np.degrees(np.mean(psi[sgi])))
                f["n_outlier_stations"] = int(len(sg) - len(sgi))
                if len(sgi) >= 4:
                    sl = _linfit(lv[sgi], np.degrees(psi[sgi]))[0] * 1000
                    bs = _boot(lambda q: _linfit(lv[sgi][q], np.degrees(psi[sgi][q]))[0] * 1000
                               if np.ptp(lv[sgi][q]) > 0 else None, len(sgi), B)
                    seg_slopes.append((sl, float(np.std(bs)) if len(bs) else np.inf, len(sgi)))
                out.append(f)
        if not out:
            f = _line_fit(X_all, types_all)
            f["stations"] = [float(lv[0]), float(lv[-1])]
            f["angle_about_axis_deg"] = float(np.degrees(np.mean(psi)))
            out = [f]
        tang = [f for f in out if f["type_majority"] == "tangent"]
        prim = max(tang or out, key=lambda f: f["n_stations_used"])
        return prim | {"n_segments": len(out), "segments": out}

    # flat spots: angular bins where >= 3 consecutive stations sit inside the circle by > threshold
    nbmax = max(len(r) for r in flat_rows)
    Fm = np.full((n, nbmax), np.nan)
    for i, r in enumerate(flat_rows):
        # resample each station's profile to a common normalised-angle grid
        x = np.linspace(0, 1, len(r))
        xx = np.linspace(0, 1, nbmax)
        ok = np.isfinite(r)
        if ok.sum() >= 2:
            Fm[i] = np.interp(xx, x[ok], r[ok])
    thr = max(prm.flat_depth_mm, 5.0 * float(np.median(rms_st)))
    inside = -Fm > thr
    flats = []
    for b in range(nbmax):
        col = inside[:, b]
        for i0, i1 in _runs(col):
            if i1 - i0 + 1 >= 3:
                flats.append((b, i0, i1, float(np.nanmax(-Fm[i0:i1 + 1, b]))))
    flat_summary = []
    if flats:
        # group into zones (adjacent bins / overlapping station ranges)
        flats.sort()
        zones = []
        for f in flats:
            for z in zones:
                if abs(f[0] - z["b1"]) <= 1 and not (f[2] < z["i0"] or f[1] > z["i1"]):
                    z["b1"] = max(z["b1"], f[0])
                    z["i0"], z["i1"] = min(z["i0"], f[1]), max(z["i1"], f[2])
                    z["depth"] = max(z["depth"], f[3])
                    break
            else:
                zones.append({"b0": f[0], "b1": f[0], "i0": f[1], "i1": f[2], "depth": f[3]})
        sw_m = float(np.median(sweeps))
        for z in zones:
            flat_summary.append({"angle_from_start_deg": [round(z["b0"] / nbmax * sw_m, 1), round((z["b1"] + 1) / nbmax * sw_m, 1)],
                                 "stations": [float(lv[z["i0"]]), float(lv[z["i1"]])],
                                 "length_mm": float(lv[z["i1"]] - lv[z["i0"]]), "max_depth_mm": z["depth"]})
    runouts = {"start": runout_line(0), "end": runout_line(1)}
    clean = [i for i in range(n) if i not in ro_out[0] and i not in ro_out[1]]
    if seg_slopes:
        w = np.array([1.0 / max(u_, 1e-6) ** 2 for _, u_, _ in seg_slopes])
        v = np.array([x for x, _, _ in seg_slopes])
        twist = {"value": float((w * v).sum() / w.sum()), "u_boot": float(1 / math.sqrt(w.sum())),
                 "basis": f"run-out line segments ({len(seg_slopes)}), inverse-variance mean of d(psi)/ds",
                 "per_segment": [{"value": x, "u_boot": u_, "n": k_} for x, u_, k_ in seg_slopes]}
    else:
        twist = None
    out = {
        "skins": meas_skin,
        "measured_skin_for_angles": sk_main,
        "n_stations": n,
        "length_mm": L,
        "station_step_mm": float(reg["step"]),
        "axis": {"point_mid": mid.tolist(), "direction": d.tolist(), "u_dir_deg": ax_unc,
                 "refine_history_deg": reg.get("axis_refine_deg")},
        "thickness": {"t_two_skin_mm": t_pair, "t_d1_mm": t_d1, "t_used_mm": t_use, "t_used_source": t_used_src,
                      "two_skin_stations": int(both.sum())},
        "R_inner": (m_se(R_in_full) or {"mean": None, "u_boot": None}) | {"source": "measured" if np.isfinite(R_in).sum() >= n / 2 else f"outer - t ({t_used_src})"},
        "R_outer": (m_se(R_out_full) or {"mean": None, "u_boot": None}) | {"source": "measured" if np.isfinite(R_out).sum() >= n / 2 else f"inner + t ({t_used_src})"},
        "R_inner_measured_stations": int(np.isfinite(R_in).sum()),
        "R_outer_measured_stations": int(np.isfinite(R_out).sum()),
        "sweep_deg": m_se(sweeps),
        "sweep_clean_deg": m_se(sweeps[clean]) | {"median": float(np.median(sweeps[clean])),
                                                  "note": "stations whose both run-outs are on their line (outliers = arcs cut by holes/breaks)"}
        if len(clean) >= 3 else None,
        "sweep_tangent_both_deg": m_se([sweeps[i] for i in range(n) if ro_type[0][i] == "tangent" and ro_type[1][i] == "tangent"]),
        "extent_to_edges_deg": m_se(extent) | {"note": "sweep + the angle subtended by straight (unrolled) tips that run "
                                                        "from an arc end to the sheet edge; = sweep where no such tip"},
        "flat_end_length_mm": {k: m_se(np.array(flat_end, float)[:, j]) for j, k in enumerate(("start", "end"))},
        "sweep_by_skin_deg": {k: (m_se(v) if v else None) for k, v in sweeps_skin.items()},
        "chord_runout_mm": m_se(chord),
        "chord_from_fit_mm": float(2 * np.nanmean(R_in_full if sk_main == "inner" else R_out_full)
                                   * math.sin(math.radians(min(float(np.mean(sweeps)), 359.9)) / 2)),

        "R_inner_common_window": common | {"R_inner": m_se(Rc_in) if np.isfinite(Rc_in).sum() >= 3 else None},
        "taper_mm_per_m": slope_unc(lv, Rc_in if np.isfinite(Rc_in).sum() >= max(4, 0.5 * n) else R_in_full),
        "taper_basis": "common angular window" if np.isfinite(Rc_in).sum() >= max(4, 0.5 * n) else "full arc per station",
        "twist_deg_per_m": twist,
        "runout_lines": runouts,
        "sweep_gradient_deg_per_m": slope_unc(lv, sweeps),
        "straightness": {"centre_rms_mm": float(np.sqrt(np.mean(dev ** 2))), "centre_max_mm": float(dev.max()),
                         "bow_sagitta_mm": bow},
        "fit_rms_mm": {"median": float(np.median(rms_st)), "p90": float(np.percentile(rms_st, 90))},
        "coverage": {"arc_fraction": m_se(cov_ang),
                     "length_fraction": float(n / (1 + round((lv.max() - lv.min()) / reg["step"]))) if n > 1 else 1.0},
        "zone_radius": {k: m_se(v) for k, v in zone.items()},
        "flat_spots": flat_summary,
        "flat_threshold_mm": thr,
        "per_station": [{"level": float(lv[i]), "R_in": _f(R_in_full[i]), "R_out": _f(R_out_full[i]),
                         "R_in_common": _f(Rc_in[i]),
                         "R_in_measured": bool(np.isfinite(R_in[i])), "R_out_measured": bool(np.isfinite(R_out[i])),
                         "sweep_deg": float(sweeps[i]), "chord_mm": float(chord[i]), "fit_rms_mm": float(rms_st[i]),
                         "centre_dev_mm": float(dev[i]), "arc_coverage": cov_ang[i],
                         "ends": [ro_type[0][i], ro_type[1][i]], "s_axis": float(sproj[i] - 0.5 * (sproj.max() + sproj.min())),
                         "psi_deg": [float(np.degrees(ang0[i])), float(np.degrees(ang0[i]) + sweeps[i])]} for i in range(n)],
        "_common_centres": np.where(np.isfinite(Cc["inner"]).all(1)[:, None], Cc["inner"], Cc["outer"]),
        "frame_note": "psi = angle about the axis in geom.frame(direction); s_axis = along the axis from point_mid",
    }
    return out


def _f(x):
    return None if not np.isfinite(x) else float(x)


def region_plot_data(reg):
    """Arrays the PNG needs (one section, station radii, run-out points, axis)."""
    u = reg["u"]
    e1, e2 = G.frame(u)
    S = _stations(reg, reg["levels"])
    return {"u": u, "e1": e1, "e2": e2, "stations": S}
