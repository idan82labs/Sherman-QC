"""zs_lane_v4 extent hypotheses (runtime side; label-free).

Half of the pinned lanes' misses are extent errors: a candidate on the right fold with the wrong span (SPAN, the
largest FN/FP cause). Most of them are cases where pooling had the right extent and threw it away, or where the
right line is the union of two tracked pieces, or a piece of one over-long track (tools/span_analysis.py). v4 keeps
those extents as competing hypotheses and lets the selector choose:

* ``z2ext``  the extent of a Z2 line that pooling attached to a Z3 candidate (pool rule 4 deg / 2.5 mm / 50 %),
             on the candidate's snapped axis (pooling keeps only an agreement flag);
* ``union``  the union of the candidate's extent and that Z2 extent (v3's EXT extent);
* ``join``   the union of two consecutive collinear candidates (zs_lane_v2 evidence pairs) along the longer one's
             snapped axis, with the pair's plate-continuity evidence as features; chains of pairs whose plates
             are connected on both flanks give one chain hypothesis;
* ``split``  the pieces of a candidate between its interior interruptions (zs_lane_v2 split runs), >= 3 mm.

Every hypothesis carries its parent's full feature row plus hypothesis features (kind, length ratio, support of
the new extent from the parent's profile, how many other candidates it would swallow, pair evidence).
A hypothesis is placed by translating its raw segment with the parent's D1 snap displacement (the snap only
translates a line). Selection (select.select_h) is greedy by probability: a line is not taken when it duplicates an
already taken line (4 deg / 2.5 mm / 50 % of the shorter), so hypotheses of one fold compete and one wins.
"""
from __future__ import annotations

import math

import numpy as np

from research.zeroshot.zs_lane_v4 import profiles as PR

KINDS = ["base", "z2ext", "union", "join", "split"]
ALT_FEATS = ["h_kind_" + k for k in KINDS] + [
    "h_len_rel", "h_log_len", "h_touch", "h_touch_gap", "h_sw", "h_fa", "h_fb", "h_swallow", "h_swallow_z3",
    "h_gap", "h_ca", "h_cb", "h_gap_bend", "h_gap_fa", "h_gap_fb", "h_n_parts", "h_z2_cover", "h_piece_frac",
    "h_cut_tab"]
DUP = (4.0, 2.5, 0.5)
MIN_PIECE = 3.0


def _u(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _snap_d(c: dict) -> np.ndarray:
    return np.asarray(c["start_s"], float) - np.asarray(c["start"], float)


def _geom(a0, a1, b0, b1):
    from research.zeroshot.zs_lane_v3.fixes import pair_geom
    return pair_geom(a0, a1, b0, b1)


def _dup(a0, a1, b0, b1) -> bool:
    ang, off, ov = _geom(a0, a1, b0, b1)
    return ang < DUP[0] and off < DUP[1] and ov > DUP[2]


def _support(prof: dict | None, lo: float, hi: float) -> dict:
    """Profile support of [lo, hi] (mm along the parent's raw axis)."""
    if prof is None or hi - lo < 0.5:
        return {}
    return {"h_touch": PR.frac(prof, "touch", lo, hi), "h_touch_gap": PR.longest_gap(prof, "touch", lo, hi),
            "h_sw": PR.frac(prof, "sw", lo, hi), "h_fa": PR.frac(prof, "fa", lo, hi), "h_fb": PR.frac(prof, "fb", lo, hi)}


def build(rec: dict) -> list[dict]:
    """Alternative-extent hypotheses of a v4 record (base candidates are not repeated)."""
    C = rec.get("candidates") or []
    if not C:
        return []
    ev = rec.get("evidence") or {}
    prof = rec.get("profiles") or {}
    z2 = rec.get("z2_lines") or []
    out: list[dict] = []

    def add(kind, parents, a_raw, u, lo, hi, d, extra):
        if hi - lo < MIN_PIECE:
            return
        s = (a_raw + lo * u + d).tolist()
        e = (a_raw + hi * u + d).tolist()
        out.append({"kind": kind, "parents": parents, "start": (a_raw + lo * u).tolist(),
                    "end": (a_raw + hi * u).tolist(), "start_s": s, "end_s": e, "lo": lo, "hi": hi, **extra})

    for i, c in enumerate(C):
        p = prof.get(str(i))
        a = np.asarray(c["start"], float)
        b = np.asarray(c["end"], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        d = _snap_d(c)
        # (A) Z2 extents
        for s0, s1, k in (p or {}).get("z2_ext") or []:
            cov = z2[k].get("cover") if k < len(z2) else None
            if abs(s0) > 1.0 or abs(s1 - L) > 1.0:
                add("z2ext", [i], a, u, s0, s1, d, {"z2_cover": cov, **_support(p, s0, s1)})
            lo, hi = min(0.0, s0), max(L, s1)
            if (lo < -1.0 or hi > L + 1.0) and (abs(lo - s0) > 1.0 or abs(hi - s1) > 1.0):
                add("union", [i], a, u, lo, hi, d, {"z2_cover": cov, **_support(p, lo, hi)})
        # (C) split pieces
        rr = (ev.get("splits") or {}).get(str(i)) or []
        if rr:
            cuts = sorted((r["g0"], r["g1"], r) for r in rr)
            f = 0.0
            pieces = []
            for g0, g1, r in cuts:
                if g0 > f:
                    pieces.append((f, g0))
                f = max(f, g1)
            if f < L:
                pieces.append((f, L))
            pieces = [pc for pc in pieces if pc[1] - pc[0] >= MIN_PIECE]
            if len(pieces) >= 2:
                for lo, hi in pieces:
                    tab = any(_tabgap(r) for g0, g1, r in cuts if abs(g1 - lo) < 0.6 or abs(g0 - hi) < 0.6)
                    add("split", [i], a, u, lo, hi, d, {"piece_frac": (hi - lo) / L, "cut_tab": float(tab),
                                                        **_support(p, lo, hi)})
    # (B) joins (pairs and connected chains)
    pairs = [q for q in ev.get("pairs") or [] if q.get("has_flanks")]
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for q in pairs:
        i, j = int(q["i"]), int(q["j"])
        if i >= len(C) or j >= len(C):
            continue
        h = _union([i, j], C)
        if h is None:
            continue
        lo, hi, a, u, d, k0 = h
        add("join", [i, j], a, u, lo, hi, d, {"gap": q.get("gap"), "ca": float(bool(q.get("ca"))),
                                               "cb": float(bool(q.get("cb"))), "gap_bend": q.get("gap_bend"),
                                               "gap_fa": q.get("gap_fa"), "gap_fb": q.get("gap_fb"),
                                               **_support(prof.get(str(k0)), lo, hi)})
        if q.get("ca") and q.get("cb"):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    for g in groups.values():
        mem = sorted(set(g))
        if len(mem) >= 3:
            h = _union(mem, C)
            if h is not None:
                lo, hi, a, u, d, k0 = h
                add("join", mem, a, u, lo, hi, d, {"ca": 1.0, "cb": 1.0, **_support(prof.get(str(k0)), lo, hi)})
    return _dedupe_alts(out, C)


def _tabgap(r: dict) -> bool:
    for f in ("a", "b"):
        if not r[f"c{f}"] and r.get(f"sides_{f}", 0) == 2 and r.get(f"raw_{f}", 1.0) <= 0.34:
            return True
    return False


def _union(mem: list[int], C: list[dict]):
    """Union of members' raw segments along the longest member's raw axis; its snap displacement."""
    lens = [float(np.linalg.norm(np.subtract(C[k]["end"], C[k]["start"]))) for k in mem]
    k0 = mem[int(np.argmax(lens))]
    a = np.asarray(C[k0]["start"], float)
    u = _u(np.asarray(C[k0]["end"], float) - a)
    ss = []
    for k in mem:
        for key in ("start", "end"):
            ss.append(float((np.asarray(C[k][key], float) - a) @ u))
    if not ss:
        return None
    return min(ss), max(ss), a, u, _snap_d(C[k0]), k0


def _dedupe_alts(alts: list[dict], C: list[dict]) -> list[dict]:
    """Drop an alternative identical (within 1 mm at both ends) to a base candidate or an earlier alternative."""
    keep: list[dict] = []
    segs = [(np.asarray(c["start_s"], float), np.asarray(c["end_s"], float)) for c in C]
    for h in alts:
        hs, he = np.asarray(h["start_s"], float), np.asarray(h["end_s"], float)

        def same(a, b):
            return (max(np.linalg.norm(hs - a), np.linalg.norm(he - b)) < 1.0
                    or max(np.linalg.norm(hs - b), np.linalg.norm(he - a)) < 1.0)
        if any(same(a, b) for a, b in segs) or any(same(np.asarray(k["start_s"]), np.asarray(k["end_s"])) for k in keep):
            continue
        keep.append(h)
    return keep


def features(rec: dict, alts: list[dict], base_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(base hypothesis columns, alternative rows = parent's base row + hypothesis columns)."""
    C = rec.get("candidates") or []
    n = len(C)
    col = {k: i for i, k in enumerate(ALT_FEATS)}
    Hb = np.full((n, len(ALT_FEATS)), np.nan)
    if n:
        Hb[:, col["h_kind_base"]] = 1.0
        for k in KINDS[1:]:
            Hb[:, col["h_kind_" + k]] = 0.0
        Hb[:, col["h_len_rel"]] = 1.0
        Lb = np.array([float(np.linalg.norm(np.subtract(c["end"], c["start"]))) for c in C])
        Hb[:, col["h_log_len"]] = np.log(Lb + 1)
    rows = np.zeros((len(alts), base_rows.shape[1] if len(base_rows.shape) > 1 else 0))
    Ha = np.full((len(alts), len(ALT_FEATS)), np.nan)
    segs = [(c["start_s"], c["end_s"], bool(c.get("has_z3"))) for c in C]
    for r, h in enumerate(alts):
        par = h["parents"]
        lens = [float(np.linalg.norm(np.subtract(C[k]["end"], C[k]["start"]))) for k in par]
        k0 = par[int(np.argmax(lens))]
        rows[r] = base_rows[k0]
        for k in KINDS:
            Ha[r, col["h_kind_" + k]] = float(h["kind"] == k)
        Lh = h["hi"] - h["lo"]
        Ha[r, col["h_len_rel"]] = Lh / max(max(lens), 1e-9)
        Ha[r, col["h_log_len"]] = math.log(Lh + 1)
        for key in ("h_touch", "h_touch_gap", "h_sw", "h_fa", "h_fb"):
            if h.get(key) is not None:
                Ha[r, col[key]] = h[key]
        sw = swz = 0
        for m, (a, b, z3) in enumerate(segs):
            if m in par:
                continue
            if _dup(h["start_s"], h["end_s"], a, b):
                sw += 1
                swz += int(z3)
        Ha[r, col["h_swallow"]] = sw
        Ha[r, col["h_swallow_z3"]] = swz
        for key, name in (("gap", "h_gap"), ("ca", "h_ca"), ("cb", "h_cb"), ("gap_bend", "h_gap_bend"),
                          ("gap_fa", "h_gap_fa"), ("gap_fb", "h_gap_fb"), ("z2_cover", "h_z2_cover"),
                          ("piece_frac", "h_piece_frac"), ("cut_tab", "h_cut_tab")):
            if h.get(key) is not None:
                Ha[r, col[name]] = float(h[key])
        Ha[r, col["h_n_parts"]] = len(par)
    return Hb, np.hstack([rows, Ha]) if len(alts) else np.zeros((0, base_rows.shape[1] + len(ALT_FEATS)))
