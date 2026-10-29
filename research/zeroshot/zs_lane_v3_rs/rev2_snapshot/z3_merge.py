"""zs_lane_v3 revision 2: bounded Z3 segment merging (exact; same output as the frozen v2 ``z3s.merge_segments``).

Why. ``zs_lane_v2/z3s.py::merge_segments`` (frozen, imported read-only) chains collinear segments of one axis
family with a restart loop: after every merge it re-sorts and re-tests every pair from the start. Each pair test
computes a per-point transverse offset median in Python. On a scan with one axis family and thousands of parallel
segments (a rolled shell: every pair has the same direction, included angle and bisector, so the cheap tests prune
nothing) this is O(merges x n^2) Python-level pair tests. On the N1 rolled shell 41018000 it took 99 s of a 130 s
uncontended profile, so the lane overran the runner's 90 s zs limit.

What this module changes. Only the amount of work, never the result:
  * a pair (a, b) is skipped only when it is **provably rejected**:
      - it was already tested with the original code (``a`` and ``b`` unchanged since: an item is immutable until it
        is merged, and a merged item gets a new slot), or
      - a vectorised prefilter puts it on the reject side of one of the original tests (family, axis parallelism,
        included angle, bisector, transverse-offset median, gap length) **by a safety margin** (1e-9 on cosines,
        1e-6 mm on lengths) that is many orders of magnitude larger than the floating-point difference between the
        vectorised and the scalar arithmetic. Pairs inside the margin are left to the original test;
      - the offset-median prefilter uses a count bound: median(off) > thr holds whenever at least floor(n/2) + 1 of
        the n offsets exceed thr (both middle order statistics then exceed thr);
  * every remaining pair is evaluated with the **original scalar code, in the original loop order** (sorted by
    descending length, row-major over (i, j > i)), and the first accepted pair is merged, exactly as the frozen loop.
So the merged candidates equal the frozen function's output; ``detect`` below is the frozen ``z3s.detect`` with
only the merge call replaced. The equality is checked on every DEV scan (v3 rev2 DEV dump bit-identical to the v3
dump) and by ``tests/test_prod_zs_lane_v3.py`` on synthetic segment sets.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from research.zeroshot.zs_lane_v2 import z3s as Z

COS_EPS = 1e-9        # margin on cosine / dot-product tests
LEN_EPS = 1e-6        # margin (mm) on length tests
COS3 = math.cos(math.radians(3.0))


def _pair_exact(a_it: list, b_it: list, prm: Z.Params, mesh: Z.Mesh, tree: cKDTree, t: float,
                max_gap_mm: float) -> tuple[dict, dict] | None:
    """The frozen merge_segments pair test and merge, verbatim. Returns (merged segment, merged line) or None."""
    a = a_it[1]
    pa0, pa1 = np.asarray(a["start"]), np.asarray(a["end"])
    d = Z._unit3(pa1 - pa0)
    b = b_it[1]
    if a["family"] != b["family"]:
        return None
    db_ = Z._unit3(np.asarray(b["end"]) - np.asarray(b["start"]))
    if abs(d @ db_) < math.cos(math.radians(3.0)):
        return None
    if abs(a["incl"] - b["incl"]) > 7.0 or np.dot(a["bis"], b["bis"]) < 0.9:
        return None
    Pb = b_it[0]["P"] - pa0
    off = np.linalg.norm(Pb - np.outer(Pb @ d, d), axis=1)
    if np.median(off) > 1.2 + 0.3 * t:
        return None
    sa = sorted([0.0, float((pa1 - pa0) @ d)])
    sb = sorted([float((np.asarray(b["start"]) - pa0) @ d), float((np.asarray(b["end"]) - pa0) @ d)])
    gap0, gap1 = (sa[1], sb[0]) if sb[0] >= sa[0] else (sb[1], sa[0])
    gap = gap1 - gap0
    if gap > max_gap_mm:
        return None
    if gap > 2.5 * prm.step_mm:
        a_max = min(80.0, 1.2 * max(3.0, min(a["flank_max"], b["flank_max"])))
        ok_a, ok_b = Z.same_plates(mesh, tree, pa0, d, np.asarray(a["da"]), np.asarray(a["db"]),
                                   gap0, gap1, t, a_max,
                                   flank_len=(min(a["flank_a"], b["flank_a"]), min(a["flank_b"], b["flank_b"])))
        if not (ok_a and ok_b):
            return None
    merged = Z._concat(a_it[0], b_it[0])
    ln = Z.segment_line(merged, prm)
    if ln is None:
        return None
    return merged, ln


class _Pool:
    """Item slots with the per-item arrays the prefilter needs, and the asymmetric 'surely rejected' matrix
    REJ[a_slot, b_slot] (pair tested with a in the role of the longer item ``a``)."""

    def __init__(self, items: list[list], t: float, max_gap_mm: float):
        self.t, self.max_gap = t, max_gap_mm
        self.thr = 1.2 + 0.3 * t
        cap = max(8, 2 * len(items))
        self.items: list[list | None] = []
        self.fam = np.zeros(cap, np.int64)
        self.incl = np.zeros(cap)
        self.bis = np.zeros((cap, 3))
        self.p0 = np.zeros((cap, 3))
        self.p1 = np.zeros((cap, 3))
        self.d = np.zeros((cap, 3))
        self.L = np.zeros(cap)              # (p1 - p0) . d
        self.ok = np.zeros(cap, bool)       # prefilter usable (non-degenerate direction)
        self.alive = np.zeros(cap, bool)
        self.REJ = np.zeros((cap, cap), bool)
        for it in items:
            self._add(it)
        self._prefilter_all()

    def _grow(self) -> None:
        cap = 2 * len(self.fam)
        def g(x):
            y = np.zeros((cap,) + x.shape[1:], x.dtype)
            y[: len(x)] = x
            return y
        self.fam, self.incl, self.bis, self.p0, self.p1, self.d = map(
            g, (self.fam, self.incl, self.bis, self.p0, self.p1, self.d))
        self.L, self.ok, self.alive = map(g, (self.L, self.ok, self.alive))
        R = np.zeros((cap, cap), bool)
        n = self.REJ.shape[0]
        R[:n, :n] = self.REJ
        self.REJ = R

    def _add(self, it: list) -> int:
        k = len(self.items)
        if k >= len(self.fam):
            self._grow()
        self.items.append(it)
        ln = it[1]
        s, e = np.asarray(ln["start"], float), np.asarray(ln["end"], float)
        v = e - s
        nv = float(np.sqrt(v @ v))
        self.fam[k] = int(ln["family"])
        self.incl[k] = float(ln["incl"])
        self.bis[k] = np.asarray(ln["bis"], float)
        self.p0[k] = s
        self.p1[k] = e
        self.ok[k] = nv > 1e-9
        self.d[k] = v / nv if self.ok[k] else 0.0
        self.L[k] = float(v @ self.d[k])
        self.alive[k] = True
        return k

    # -- prefilter ---------------------------------------------------------------------------------------------
    def _cheap(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Surely-rejected by the family / parallelism / included-angle / bisector tests, for rows A x cols B."""
        rej = self.fam[A][:, None] != self.fam[B][None, :]
        c = np.abs(self.d[A] @ self.d[B].T)
        rej |= c < COS3 - COS_EPS
        rej |= np.abs(self.incl[A][:, None] - self.incl[B][None, :]) > 7.0      # same float op as the original
        rej |= (self.bis[A] @ self.bis[B].T) < 0.9 - COS_EPS
        okm = self.ok[A][:, None] & self.ok[B][None, :]
        fam_rej = self.fam[A][:, None] != self.fam[B][None, :]
        return np.where(okm, rej, fam_rej)

    def _geom_row(self, a: int, B: np.ndarray) -> np.ndarray:
        """Surely-rejected by the offset-median or gap test, for a (role a) against items B (role b)."""
        out = np.zeros(len(B), bool)
        if not self.ok[a] or len(B) == 0:
            return out
        p0, d = self.p0[a], self.d[a]
        P = [self.items[b][0]["P"] for b in B]
        n = np.array([len(x) for x in P])
        Q = np.concatenate(P) - p0
        off = np.linalg.norm(Q - np.outer(Q @ d, d), axis=1)
        g = np.repeat(np.arange(len(B)), n)
        big = np.bincount(g, weights=(off > self.thr + LEN_EPS), minlength=len(B))
        out |= big >= (n // 2 + 1)
        # gap test (only where the branch choice sb0 >= sa0 is not within the margin)
        sa0, sa1 = min(0.0, self.L[a]), max(0.0, self.L[a])
        st = (self.p0[B] - p0) @ d
        en = (self.p1[B] - p0) @ d
        sb0, sb1 = np.minimum(st, en), np.maximum(st, en)
        gap = np.where(sb0 >= sa0, sb0 - sa1, sa0 - sb1)
        clear = np.abs(sb0 - sa0) > LEN_EPS
        out |= clear & (gap > self.max_gap + 1e-3)
        return out

    def _prefilter_all(self) -> None:
        idx = np.nonzero(self.alive)[0]
        if len(idx) == 0:
            return
        C = self._cheap(idx, idx)
        for r, a in enumerate(idx):
            cand = idx[~C[r]]
            cand = cand[cand != a]
            row = np.ones(len(idx), bool)
            if len(cand):
                g = self._geom_row(a, cand)
                pos = np.searchsorted(idx, cand)
                row[pos] = g
            row[r] = True
            self.REJ[a, idx] = row | C[r]

    def prefilter_new(self, k: int) -> None:
        idx = np.nonzero(self.alive)[0]
        others = idx[idx != k]
        self.REJ[k, :] = True
        self.REJ[:, k] = True
        if len(others) == 0:
            return
        # k in role a
        c = self._cheap(np.array([k]), others)[0]
        cand = others[~c]
        row = np.ones(len(others), bool)
        if len(cand):
            row[np.searchsorted(others, cand)] = self._geom_row(k, cand)
        self.REJ[k, others] = row | c
        # k in role b
        c = self._cheap(others, np.array([k]))[:, 0]
        col = np.ones(len(others), bool)
        for r in np.nonzero(~c)[0]:
            col[r] = self._geom_row(int(others[r]), np.array([k]))[0]
        self.REJ[others, k] = col | c


def merge_segments(segs: list[dict[str, Any]], prm: Z.Params, mesh: Z.Mesh, tree: cKDTree, t: float,
                   max_gap_mm: float = 80.0) -> list[dict[str, Any]]:
    """Same output as ``z3s.merge_segments`` (module docstring), with bounded work."""
    items = []
    for sg in segs:
        ln = Z.segment_line(sg, prm)
        if ln is not None:
            items.append([sg, ln])
    if len(items) < 2:
        return [it[1] for it in items]
    pool = _Pool(items, t, max_gap_mm)
    order = list(range(len(items)))           # slots, in the frozen loop's list order
    while True:
        # the frozen loop: stable sort by descending length, then row-major (i, j > i)
        order.sort(key=lambda s: -pool.items[s][1]["length"])
        o = np.asarray(order)
        U = ~pool.REJ[np.ix_(o, o)]
        U = np.triu(U, k=1)
        merged_any = False
        for i, j in zip(*np.nonzero(U)):
            sa, sb = int(o[i]), int(o[j])
            res = _pair_exact(pool.items[sa], pool.items[sb], prm, mesh, tree, t, max_gap_mm)
            if res is None:
                pool.REJ[sa, sb] = True
                continue
            pool.alive[sa] = pool.alive[sb] = False
            k = pool._add([res[0], res[1]])
            pool.prefilter_new(k)
            # the frozen loop writes the merged item at position i and pops position j
            order = [s for s in order if s != sb]
            order[order.index(sa)] = k
            merged_any = True
            break
        if not merged_any:
            break
    return [pool.items[s][1] for s in order]


def detect(mesh: Z.Mesh, prm: Z.Params | None = None, *, thickness: dict | None = None,
           families: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The frozen ``z3s.detect``, verbatim except that segment merging uses :func:`merge_segments` above."""
    prm = prm or Z.Params()
    tm = {}
    t0 = time.time()
    th = thickness or Z.estimate_thickness(mesh)
    t = float(np.clip(th["t_mm"], 0.5, 6.0))
    tm["thickness"] = time.time() - t0
    t1 = time.time()
    fams = families if families is not None else Z.axis_families(mesh, prm)
    fallback = False
    if families is None and not fams and prm.fallback_frac_only:
        from dataclasses import replace as _replace
        fams = Z.axis_families(mesh, _replace(prm, peak_min_area_mm2=0.0, peak_min_area_rel=0.0))
        fallback = True
    tm["families"] = time.time() - t1
    t2 = time.time()
    tree = cKDTree(mesh.V)
    raw = []
    for fi, f in enumerate(fams):
        u = np.asarray(f["axis"], float)
        levels, chains = Z.slice_family(mesh, u, prm.step_mm)
        per_level = [Z.slice_corners(ch, prm, t) for ch in chains]
        tracks = Z.track_corners(levels, per_level, prm)
        segs = Z.tracks_to_segments(tracks, levels, u, prm, t, fi)
        raw.extend(merge_segments(segs, prm, mesh, tree, t))
    tm["slicing"] = time.time() - t2
    t3 = time.time()
    for c in raw:
        Z.refine_span(tree, c, prm.step_mm, t)
        c.update(Z.strip_evidence(mesh, tree, c, t))
        c["quality"] = c["n_support"] * c["support_frac"] * (0.5 + c["frac_outer"] + c["frac_both"])
    tm["strip"] = time.time() - t3
    merged = Z.merge_candidates(raw)
    tm["total"] = time.time() - t0
    return {"thickness": th, "t_mm": t, "area_total": float(mesh.fa.sum()), "family_fallback": fallback,
            "families": [{"axis": np.asarray(f["axis"]).tolist(), "score": f.get("score"), "weight": f.get("weight"),
                          "arc_area": f.get("arc_area")} for f in fams],
            "raw_count": len(raw), "candidates": merged, "timing_s": tm, "params": asdict(prm)}
