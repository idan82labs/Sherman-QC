"""zs_lane_v2 span post-processing: decide split vs join of the selected lines on plate-continuity evidence.

Gold convention (GOLD_SET_V2): a fold split by windows or reliefs between the *same two plates* is one line;
separate tabs are separate lines. The decision uses the mesh evidence computed in core.evidence_stage
(evidence.py), never a gap length alone:

* SPLIT a selected line at an interior run of stations where the fold is interrupted (no bend-zone material or
  a flank missing) when, across that run, the plate of flank A or of flank B is NOT one connected piece
  (separate tabs on one base). Runs where both plates continue (a window / relief) are kept inside the line.
* JOIN two consecutive collinear lines (both selected, or one selected and the other above ``join_low``) when
  both flank plates are one connected piece across the gap between them (a window or relief between the same
  two plates). Optional 'notch' mode: also join when one plate is connected, the other flank has the same
  height on both sides, and the two pieces cover at least ``notch_cov`` of the joined span.

All coordinates of the output lines are the D1-snapped placements of the pieces ("start_s"/"end_s"); the
evidence itself was computed on the raw placement ("start"/"end"), so the lane and the DEV evaluation run the
same arithmetic on the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PostParams:
    split: bool = False
    split_min_gap: float = 1.0     # mm: minimum interior interruption that can split
    min_piece: float = 3.0         # mm: split pieces shorter than this are dropped
    join: bool = False
    join_mode: str = "both"        # 'both' | 'notch'
    join_low: float = 1.0          # probability floor for a non-selected piece to take part in a join (1 = off)
    notch_cov: float = 0.8
    notch_flank_ratio: float = 0.6
    split_max_cover: float = 0.34  # the disconnected plate must be absent over most of the run (raw coverage)
    sweep_min: float = 0.0         # reject a selected Z3 candidate whose station normal-sweep fraction is below
    dedupe_join: bool = False      # a joined line that duplicates another output line (<= 4 deg, <= 2.5 mm,
                                   # > 50 % of the shorter) absorbs it (the shorter one is dropped)

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


def features(X: np.ndarray, SW: np.ndarray, feats: str) -> np.ndarray:
    """Selector feature matrix: the v1 pooled features (21 Z3 + 5 pool flags) and, with 'sweep', the v2
    station-profile summaries (evidence.SWEEP_FEATS)."""
    parts = [np.asarray(X, float)]
    if "sweep" in feats.split(","):
        parts.append(np.asarray(SW, float))
    n = len(parts[0])
    return np.hstack(parts) if n else np.zeros((0, sum(p.shape[1] for p in parts)))


def sweep_matrix(sweep: list | None, n: int) -> np.ndarray:
    from research.zeroshot.zs_lane_v2.evidence import SWEEP_FEATS
    sw = sweep if sweep is not None else [None] * n
    if not n:
        return np.zeros((0, len(SWEEP_FEATS)))
    return np.array([[np.nan] * len(SWEEP_FEATS) if v is None else v for v in sw], float).reshape(n, len(SWEEP_FEATS))


def _seg(c: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(c.get("start_s", c["start"]), float), np.asarray(c.get("end_s", c["end"]), float)


def _raw_len(c: dict) -> float:
    return float(np.linalg.norm(np.subtract(c["end"], c["start"])))


def _sub(c: dict, f0: float, f1: float) -> tuple[np.ndarray, np.ndarray]:
    a, b = _seg(c)
    return a + f0 * (b - a), a + f1 * (b - a)


def _tab_gap(r: dict, prm: PostParams) -> bool:
    """An interruption of the bend is a gap between separate tabs when, for flank A or flank B, the plate is
    present on both sides of the run, absent inside it (raw coverage <= split_max_cover) and not one connected
    piece around it. Runs where both plates continue (a window / relief) never split."""
    for f in ("a", "b"):
        if (not r[f"c{f}"] and r.get(f"sides_{f}", 0) == 2
                and r.get(f"raw_{f}", 1.0) <= prm.split_max_cover):
            return True
    return False


def postprocess(cands: list[dict], prob: np.ndarray, thr: float, ev: dict | None, prm: PostParams) -> list[dict]:
    """Selected lines after split / join. Each output: start, end (snapped placement), score, src, incl,
    parts (candidate indices), op ('keep' | 'split' | 'join')."""
    n = len(cands)
    prob = np.asarray(prob, float)
    sel = [i for i in range(n) if prob[i] >= thr]
    if prm.sweep_min > 0 and ev and ev.get("sweep") is not None:
        sw = ev["sweep"]
        sel = [i for i in sel if sw[i] is None or sw[i][0] >= prm.sweep_min]
    low = set(i for i in range(n) if prm.join and prm.join_low < thr and prm.join_low <= prob[i] < thr)
    # pieces: (cand index, f0, f1) as fractions of the candidate's length
    pieces: dict[int, list[list[float]]] = {}
    for i in list(sel) + sorted(low):
        pieces[i] = [[0.0, 1.0]]
    splits = (ev or {}).get("splits", {}) if prm.split else {}
    for i in sel:
        rr = splits.get(str(i)) or []
        L = _raw_len(cands[i])
        if L <= 0:
            continue
        cuts = [(r["g0"] / L, r["g1"] / L) for r in rr
                if (r["g1"] - r["g0"]) >= prm.split_min_gap and _tab_gap(r, prm)]
        if not cuts:
            continue
        segs, f = [], 0.0
        for c0, c1 in sorted(cuts):
            if c0 > f:
                segs.append([f, c0])
            f = max(f, c1)
        if f < 1.0:
            segs.append([f, 1.0])
        segs = [s for s in segs if (s[1] - s[0]) * L >= prm.min_piece]
        pieces[i] = segs
    # join on consecutive collinear pairs
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    joined_any = set()
    if prm.join and ev:
        for p in ev.get("pairs", []):
            i, j = int(p["i"]), int(p["j"])
            if i not in pieces or j not in pieces or not pieces[i] or not pieces[j]:
                continue
            if i in low and j in low:
                continue
            if not p.get("has_flanks"):
                continue
            ok = bool(p.get("ca")) and bool(p.get("cb"))
            if not ok and prm.join_mode == "notch" and (p.get("ca") or p.get("cb")):
                cov = (p["len_i"] + p["len_j"]) / max(p["len_i"] + p["len_j"] + max(p["gap"], 0.0), 1e-9)
                side = 0 if not p.get("ca") else 1          # the disconnected flank
                fi, fj = p["flank_i"][side], p["flank_j"][side]
                ratio = min(fi, fj) / max(fi, fj, 1e-9) if fi == fi and fj == fj else 0.0
                ok = cov >= prm.notch_cov and ratio >= prm.notch_flank_ratio
            if not ok:
                continue
            # pieces adjacent to the gap: i's piece at its end (s = L_i), j's piece nearest to i's end
            pi = (i, int(np.argmax([s[1] for s in pieces[i]])))
            if pieces[i][pi[1]][1] < 0.999:
                continue           # i was split right at that end
            ie = np.asarray(cands[i]["end"], float)
            dj0 = np.linalg.norm(np.asarray(cands[j]["start"], float) - ie)
            dj1 = np.linalg.norm(np.asarray(cands[j]["end"], float) - ie)
            if dj0 <= dj1:
                k = int(np.argmin([s[0] for s in pieces[j]]))
                if pieces[j][k][0] > 0.001:
                    continue
            else:
                k = int(np.argmax([s[1] for s in pieces[j]]))
                if pieces[j][k][1] < 0.999:
                    continue
            pj = (j, k)
            ra, rb = find(pi), find(pj)
            if ra != rb:
                parent[ra] = rb
            joined_any.add(pi)
            joined_any.add(pj)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, segs in pieces.items():
        for k in range(len(segs)):
            groups.setdefault(find((i, k)), []).append((i, k))
    out = []
    for root, mem in groups.items():
        if all(m[0] in low for m in mem):
            continue
        if len(mem) == 1:
            i, k = mem[0]
            if i in low:
                continue
            f0, f1 = pieces[i][k]
            a, b = _sub(cands[i], f0, f1)
            op = "keep" if (f0 == 0.0 and f1 == 1.0) else "split"
            out.append({"start_s": a.tolist(), "end_s": b.tolist(), "score": float(prob[i]),
                        "src": cands[i].get("src", []), "incl": cands[i].get("incl"), "parts": [int(i)], "op": op,
                        "_i": int(i)})
            continue
        segs = [(_sub(cands[i], *pieces[i][k]), i) for i, k in mem]
        lens = [float(np.linalg.norm(b - a)) for (a, b), _ in segs]
        (a0, b0), i0 = segs[int(np.argmax(lens))]
        d = (b0 - a0) / max(np.linalg.norm(b0 - a0), 1e-9)
        pts = np.array([p for (a, b), _ in segs for p in (a, b)])
        s = (pts - a0) @ d
        out.append({"start_s": (a0 + s.min() * d).tolist(), "end_s": (a0 + s.max() * d).tolist(),
                    "score": float(max(prob[i] for _, i in segs)),
                    "src": sorted(set(x for _, i in segs for x in cands[i].get("src", []))),
                    "incl": cands[i0].get("incl"), "parts": sorted(set(int(i) for _, i in segs)), "op": "join",
                    "_i": int(i0)})
    if prm.dedupe_join and any(o["op"] == "join" for o in out):
        out = _dedupe_joined(out)
    out.sort(key=lambda r: (min(r["parts"]), r["start_s"]))
    return out


def _pair_geom(a: dict, b: dict) -> tuple[float, float, float]:
    """(angle deg, transverse offset mm, overlap fraction of the shorter) of two output lines."""
    pa0, pa1 = np.asarray(a["start_s"], float), np.asarray(a["end_s"], float)
    pb0, pb1 = np.asarray(b["start_s"], float), np.asarray(b["end_s"], float)
    da, db = pa1 - pa0, pb1 - pb0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    da, db = da / max(la, 1e-9), db / max(lb, 1e-9)
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(da @ db))))))
    ma, mb = 0.5 * (pa0 + pa1), 0.5 * (pb0 + pb1)
    dm = mb - ma
    off = max(float(np.linalg.norm(dm - da * (dm @ da))), float(np.linalg.norm(dm - db * (dm @ db))))
    ia = sorted([float((pa0 - ma) @ da), float((pa1 - ma) @ da)])
    ib = sorted([float((pb0 - ma) @ da), float((pb1 - ma) @ da)])
    ov = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
    return ang, off, ov / max(min(la, lb), 1e-9)


def _dedupe_joined(out: list[dict]) -> list[dict]:
    drop = set()
    joined = [k for k, o in enumerate(out) if o["op"] == "join"]
    for k in sorted(joined, key=lambda k: -float(np.linalg.norm(np.subtract(out[k]["end_s"], out[k]["start_s"])))):
        if k in drop:
            continue
        Lk = float(np.linalg.norm(np.subtract(out[k]["end_s"], out[k]["start_s"])))
        for m, o in enumerate(out):
            if m == k or m in drop:
                continue
            ang, off, ov = _pair_geom(out[k], o)
            Lm = float(np.linalg.norm(np.subtract(o["end_s"], o["start_s"])))
            if ang < 4.0 and off < 2.5 and ov > 0.5 and Lm <= Lk:
                drop.add(m)
    return [o for m, o in enumerate(out) if m not in drop]
