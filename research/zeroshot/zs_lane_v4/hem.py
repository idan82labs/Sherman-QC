"""zs_lane_v4 hem generator: 180-degree folds (closed or open hems), which neither Z3 nor Z2 can emit.

Why a separate generator (paper sec_theory, th:cor-floors iv and th:lem-axis-cond): a hem's two flange normals
are within 180 - beta of antiparallel, so on the projective Gauss sphere they make ONE peak (no axis pair for Z3),
Z3 rejects section corners above 165 deg and Z2 edges above 175 deg; and the axis from two nearly antiparallel
normals is ill-conditioned (1 / sin beta). A hem is instead observable directly from its own geometry, as two
parallel skins with a crown:
  * the crown: at the free end of the stack, the surface normals sweep ~180 deg around the hem axis, from +n through
    the outward direction o to -n (a half cylinder), whereas a cut edge is a flat wall facing o;
  * the stack: just behind the crown the material is two layers thick (~2t closed, 2t + gap open), whereas a
    single sheet's edge is t thick.

Method (per dominant plate normal n of the Gauss map, the frozen zs_lane_v2.z3s.normal_peaks):
  1. side-facing vertices (|vn . n| < cos 60 deg) and their in-plane outward direction o_v;
  2. dominant outward directions o (5 deg histogram on the circle normal to n, NMS 20 deg); axis a = n x o;
  3. straight edges: the side-facing vertices of one direction clustered in the section plane (q = p.o, h = p.n,
     1 mm grid, 8-connected) and split along a at gaps > 3 mm; edges >= ``min_len`` long are tested;
  4. per 1 mm station along the edge (all vertices, not only side-facing): crown q_max; sweep = the angle of the
     crown-zone normals in the (o, n) plane must occupy all ``sweep_bins`` bins of [-75, 75] deg with at least
     ``bin_min`` vertices each (a uniform half-cylinder, not a flat wall with two corners); stack = h-extent of the
     skin vertices (|vn . n| > 0.8) in the slab [q_max - (2t + 4), q_max - (t + 0.5)] must be in
     [``stack_min_t`` t, ``stack_max_t`` t + 2.5 mm] with skin on both faces;
  5. a hem line = the longest run of hem stations (gaps <= 2 stations bridged) >= ``min_len``; placed on the crown
     apex (q_max, the h-centre of the crown) from the first to the last hem station -- the gold's D0 tangent-band
     construction puts the hem line on the crown.
Deterministic; no labels, CAD, identities or file names. Thresholds are in degrees, mm and multiples of t.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class HemParams:
    side_cos: float = 0.5           # side-facing: |vn . n| < cos 60 deg
    max_peaks: int = 12             # plate normals tried (largest Gauss-map peaks)
    dir_bin_deg: float = 5.0
    dir_nms_deg: float = 20.0
    dir_win_deg: float = 15.0
    min_dir_count: int = 30
    qh_cell: float = 1.0
    seg_gap: float = 3.0
    min_len: float = 8.0
    sweep_bins: int = 6
    bin_min: int = 2
    crown_max_share: float = 0.35   # a round crown spreads its normals evenly over the arc (1/6 per bin); a flat
                                    # cut wall puts most of them in the middle bins
    skin_cos: float = 0.978         # skin vertices: normal within 12 deg of +-n (two layers parallel to the plate)
    skin_frac: float = 0.6          # most of the slab behind the crown is skin (not a tilted flange)
    skin_iqr_t: float = 0.35        # each outer skin is one flat layer (IQR of its height <= 0.35 t)
    layer_t: float = 0.5            # an outer skin layer: skin vertices within 0.5 t of the extreme height
    stack_min_t: float = 1.6
    stack_max_t: float = 4.0
    run_gap: int = 2
    step_ratio: float = 0.75        # the flange end: separation drops below 0.75 x the stack ...
    step_bins: int = 3              # ... and stays one flat layer for 3 depth bins (6 mm)
    grow_min: float = 20.0          # stations are also tested this far beyond the side-facing edge cluster ...
    grow_max: float = 100.0         # ... and at most this far (bounded work)
    max_edges: int = 400

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(a - (a @ n) * n)
    return e1, np.cross(n, e1)


def _runs(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - p > bridge + 1:
            out.append((s, p + 1))
            s = k
        p = k
    out.append((s, p + 1))
    return out


def detect(mesh, t: float, prm: HemParams | None = None, peaks: list[dict] | None = None,
           tree: cKDTree | None = None) -> list[dict]:
    """Hem lines [{'start_s','end_s','score','src':['hem'],'incl':0.0,'hem':{...}}] in the raw scan frame."""
    from research.zeroshot.zs_lane_v2 import z3s
    prm = prm or HemParams()
    V = np.asarray(mesh.V, float)
    VN = np.asarray(mesh.vn, float)
    if peaks is None:
        peaks = z3s.normal_peaks(mesh, z3s.Params())
    peaks = sorted(peaks, key=lambda p: -p["area"])[: prm.max_peaks]
    tree = tree or cKDTree(V)
    t = float(t)
    out: list[dict] = []
    n_edges = 0
    for pk in peaks:
        n = _unit(np.asarray(pk["normal"], float))
        dn = VN @ n
        side = np.abs(dn) < prm.side_cos
        if side.sum() < prm.min_dir_count:
            continue
        o_v = VN[side] - np.outer(dn[side], n)
        o_v /= np.maximum(np.linalg.norm(o_v, axis=1), 1e-12)[:, None]
        e1, e2 = _frame(n)
        th = np.degrees(np.arctan2(o_v @ e2, o_v @ e1))
        nb = int(round(360 / prm.dir_bin_deg))
        hist = np.bincount(((th + 180.0) / prm.dir_bin_deg).astype(int) % nb, minlength=nb).astype(float)
        sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
        used: list[float] = []
        for b in np.argsort(-sm):
            if sm[b] < prm.min_dir_count:
                break
            ang = -180.0 + (b + 0.5) * prm.dir_bin_deg
            if any(abs((ang - u + 180) % 360 - 180) < prm.dir_nms_deg for u in used):
                continue
            used.append(ang)
            o = _unit(math.cos(math.radians(ang)) * e1 + math.sin(math.radians(ang)) * e2)
            win = (o_v @ o) > math.cos(math.radians(prm.dir_win_deg))
            if win.sum() < prm.min_dir_count:
                continue
            o = _unit(o_v[win].mean(0) - (o_v[win].mean(0) @ n) * n)
            a = np.cross(n, o)
            P = V[side][win]
            s, q, h = P @ a, P @ o, P @ n
            q0, h0 = q.min(), h.min()
            gi = ((q - q0) / prm.qh_cell).astype(int)
            gj = ((h - h0) / prm.qh_cell).astype(int)
            grid = np.zeros((gi.max() + 1, gj.max() + 1), np.int32)
            np.add.at(grid, (gi, gj), 1)
            lab, nlab = ndimage.label(grid >= 2, structure=np.ones((3, 3)))
            pl = lab[gi, gj]
            for c in range(1, nlab + 1):
                m = pl == c
                if m.sum() < 8:
                    continue
                ss = np.sort(s[m])
                cuts = np.nonzero(np.diff(ss) > prm.seg_gap)[0]
                bounds = np.split(ss, cuts + 1)
                for seg in bounds:
                    if seg[-1] - seg[0] < prm.min_len:
                        continue
                    n_edges += 1
                    if n_edges > prm.max_edges:
                        break
                    sel = m & (s >= seg[0]) & (s <= seg[-1])
                    hit = _test_edge(V, VN, tree, n, o, a, float(seg[0]), float(seg[-1]),
                                     float(np.median(q[sel])), float(np.median(h[sel])), t, prm)
                    if hit is not None:
                        out.append(hit)
    return _dedupe(out)


def _test_edge(V, VN, tree, n, o, a, s0, s1, qc, hc, t, prm: HemParams) -> dict | None:
    L = s1 - s0
    E = float(min(prm.grow_max, max(prm.grow_min, L)))      # also test beyond the edge cluster: a crown can be
    e0, e1 = s0 - E, s1 + E                                   # longer than its side-facing cluster
    ctr = a * (0.5 * (e0 + e1)) + o * qc + n * hc
    idx = np.asarray(tree.query_ball_point(ctr, math.sqrt((0.5 * (e1 - e0) + 2.0) ** 2 + 12.0 ** 2)), dtype=np.int64)
    if len(idx) < 20:
        return None
    P, N = V[idx], VN[idx]
    s, q, h = P @ a, P @ o, P @ n
    keep = (np.abs(q - qc) < 10.0) & (np.abs(h - hc) < 8.0) & (s >= e0) & (s < e1 + 1.0)
    s, q, h, N = s[keep], q[keep], h[keep], N[keep]
    if len(s) < 20:
        return None
    phi = np.degrees(np.arctan2(N @ n, N @ o))
    dn = N @ n
    nst = int(math.floor(e1 - e0)) + 1
    k = np.clip(((s - e0) / 1.0).astype(int), 0, nst - 1)
    ok = np.zeros(nst, bool)
    qmax = np.full(nst, np.nan)
    hcr = np.full(nst, np.nan)
    stack = np.full(nst, np.nan)
    edges = np.linspace(-75.0, 75.0, prm.sweep_bins + 1)
    for i in range(nst):
        m = k == i
        if m.sum() < 10:
            continue
        qi, hi_, ph, di = q[m], h[m], phi[m], dn[m]
        qm = float(np.percentile(qi, 99.5))
        zone = qi > qm - max(1.0, 1.5 * t)
        cnt, _ = np.histogram(ph[zone], bins=edges)
        slab_all = (qi < qm - (t + 0.5)) & (qi > qm - (2 * t + 4.0))
        slab = slab_all & (np.abs(di) > prm.skin_cos)
        if slab.sum() < 6 or slab.sum() < prm.skin_frac * max(slab_all.sum(), 1):
            continue
        layers = _outer_layers(hi_[slab], di[slab], prm.layer_t * t)
        if layers is None:
            continue
        ht_o, hb_o = layers
        if (np.percentile(ht_o, 75) - np.percentile(ht_o, 25) > prm.skin_iqr_t * t
                or np.percentile(hb_o, 75) - np.percentile(hb_o, 25) > prm.skin_iqr_t * t):
            continue
        S = abs(float(np.median(ht_o) - np.median(hb_o)))
        near = qi > qm - 0.3
        qmax[i] = qm
        hcr[i] = float(np.median(hi_[near])) if near.any() else np.nan
        stack[i] = S
        tot = max(int(cnt.sum()), 1)
        round_crown = bool((cnt >= prm.bin_min).all() and cnt.max() <= prm.crown_max_share * tot)
        ok[i] = bool(round_crown and prm.stack_min_t * t <= S <= prm.stack_max_t * t + 2.5)
    runs = _runs(ok, prm.run_gap)
    # the run that overlaps the tested edge cluster most
    c0, c1 = int(round(s0 - e0)), int(round(s1 - e0)) + 1
    runs = [r for r in runs if min(r[1], c1) - max(r[0], c0) > 0]
    if not runs:
        return None
    r0, r1 = max(runs, key=lambda r: (min(r[1], c1) - max(r[0], c0), r[1] - r[0]))
    if r1 - r0 < prm.min_len:
        return None
    seg = slice(r0, r1)
    st = np.arange(r0, r1)
    good = st[ok[seg] & np.isfinite(qmax[seg]) & np.isfinite(hcr[seg])]
    if len(good) < 3:
        return None
    # the crown line: a TLS line through the per-station crown apexes (corrects an axis tilted by a plate normal
    # that is a few degrees off the hem's own skins)
    Pc = np.array([a * (e0 + i) + o * qmax[i] + n * hcr[i] for i in good])
    cm = Pc.mean(0)
    _, _, vt = np.linalg.svd(Pc - cm)
    d = vt[0] if vt[0] @ a >= 0 else -vt[0]
    tt = (Pc - cm) @ d
    p0, p1 = cm + tt.min() * d, cm + tt.max() * d
    frac = float(ok[seg].mean())
    S_med = float(np.nanmedian(stack[seg]))
    far = _depth_profile(V, VN, tree, n, o, d, p0, p1, S_med, t, prm)
    step = _flange_end(far, S_med, prm)
    if step is None:
        return None
    return {"start_s": p0.tolist(), "end_s": p1.tolist(), "start": p0.tolist(), "end": p1.tolist(),
            "score": frac, "src": ["hem"], "incl": 0.0,
            "hem": {"len": float(np.linalg.norm(p1 - p0)), "stack_mm": round(float(np.nanmedian(stack[seg])), 3),
                    "stack_t": round(float(np.nanmedian(stack[seg])) / t, 3), "frac": round(frac, 3),
                    "edge_len": round(L, 2), "far": far, "flange_end_mm": step[0], "t_local": step[1],
                    "stack_over_t_local": round(S_med / max(step[1], 1e-9), 3), "n": [round(float(x), 5) for x in n],
                    "o": [round(float(x), 5) for x in o]}}


def _depth_profile(V, VN, tree, n, o, d, p0, p1, S, t, prm: HemParams) -> list:
    """Separation of the two outermost skin layers vs depth behind the crown (2 mm bins, 2..30 mm), over the middle
    80 % of the crown line: [[depth, separation or None, n_points], ...]. A hem is two layers only as far as its
    flange reaches; a single sheet (or a scan whose global t is wrong) keeps one constant separation."""
    L = float(np.linalg.norm(p1 - p0))
    mid = 0.5 * (p0 + p1)
    a = d
    oo = o - (o @ a) * a
    oo /= max(np.linalg.norm(oo), 1e-12)
    nn = np.cross(a, oo)
    idx = np.asarray(tree.query_ball_point(mid - oo * 16.0, math.sqrt((0.4 * L) ** 2 + 16.0 ** 2 + 8.0 ** 2)),
                     dtype=np.int64)
    out = []
    if len(idx) < 20:
        return out
    Q = V[idx] - mid
    Nn = VN[idx]
    s_, q_, h_ = Q @ a, Q @ oo, Q @ nn
    dn = Nn @ nn
    m0 = (np.abs(s_) <= 0.4 * L) & (np.abs(dn) > prm.skin_cos) & (np.abs(h_) < S + 3.0)
    for dlo in range(2, 30, 2):
        m = m0 & (q_ <= -dlo) & (q_ > -(dlo + 2))
        layers = _outer_layers(h_[m], dn[m], prm.layer_t * t) if m.sum() >= 6 else None
        if layers is None:
            out.append([dlo, None, int(m.sum())])
            continue
        out.append([dlo, round(abs(float(np.median(layers[0]) - np.median(layers[1]))), 3), int(m.sum())])
    return out


def _outer_layers(h: np.ndarray, dn: np.ndarray, lay: float):
    """Heights of the two OUTERMOST skin layers (top, bottom) of a stack of skin vertices (|vn . n| large), or None.
    An open hem also shows the two inner skins facing its gap, so the layers are taken at the extreme heights, and
    they must face away from each other (opposite normal signs: the outward skins of one stack)."""
    if len(h) < 6:
        return None
    hi, lo = float(np.percentile(h, 99)), float(np.percentile(h, 1))
    top, bot = h >= hi - lay, h <= lo + lay
    if top.sum() < 3 or bot.sum() < 3 or hi - lo < lay:
        return None
    st, sb = float(np.mean(np.sign(dn[top]))), float(np.mean(np.sign(dn[bot])))
    if not (st * sb < 0 and abs(st) > 0.6 and abs(sb) > 0.6):
        return None
    return h[top], h[bot]


def _flange_end(far: list, S: float, prm: HemParams) -> tuple[float, float] | None:
    """(depth of the step, local single-sheet thickness) when the double layer behind the crown ends in a step down to
    one layer of thickness t_local with S >= stack_min_t x t_local, within the profiled depth; else None. A single
    sheet (or a scan whose global thickness estimate is wrong) shows one constant separation and no step."""
    vals = [(d, v) for d, v, _ in far if v is not None]
    for k, (d, v) in enumerate(vals):
        if v < prm.step_ratio * S:
            after = [w for _, w in vals[k:k + prm.step_bins]]
            if len(after) < prm.step_bins:
                return None
            tl = float(np.median(after))
            if max(after) - min(after) > 0.25 * tl + 0.1:          # one flat single layer after the step
                return None
            if S >= prm.stack_min_t * tl:
                return float(d), round(tl, 3)
            return None
    return None


def _same_crown(a: dict, b: dict) -> bool:
    """One crown found twice (e.g. from two plate normals a few degrees apart): axes within 30 deg, the shorter
    line within 3 mm of the longer one's axis and overlapping it by more than half."""
    from research.zeroshot.zs_lane_v3.fixes import pair_geom
    ang, off, ov = pair_geom(a["start_s"], a["end_s"], b["start_s"], b["end_s"])
    return ang < 30.0 and off < 3.0 and ov > 0.5


def _dedupe(lines: list[dict]) -> list[dict]:
    lines = sorted(lines, key=lambda l: (-l["hem"]["len"] * l["score"], -l["score"]))
    out: list[dict] = []
    for l in lines:
        if any(_same_crown(l, o) for o in out):
            continue
        out.append(l)
    return out
