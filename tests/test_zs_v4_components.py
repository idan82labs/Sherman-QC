"""zs_lane_v4 component tests on synthetic inputs only (no scans, gold, cache or engine output are read).

Covers: the hem generator on synthetic closed / open hems and plain cut edges; the safe EXT and the hypothesis guard
on a synthetic row of tabs (the gb_68cd06 failure of v3); the fast matcher vs the frozen matcher; the LOFO fold rule.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")


# ----------------------------------------------------------------------------- synthetic extruded profiles
def _area(P):
    P = np.asarray(P)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def _profile_hem(t=1.0, W=40.0, F=8.0, gap=0.0, n_arc=24):
    R, r, zc = t + gap / 2.0, gap / 2.0, t + gap / 2.0
    P = [(-W, 0.0)]
    for k in range(n_arc + 1):
        a = -math.pi / 2 + math.pi * k / n_arc
        P.append((R * math.cos(a), zc + R * math.sin(a)))
    P += [(-F, 2 * t + gap), (-F, t + gap)]
    if r > 1e-6:
        for k in range(n_arc + 1):
            a = math.pi / 2 - math.pi * k / n_arc
            P.append((r * math.cos(a), zc + r * math.sin(a)))
    P.append((-W, t))
    return np.array(P[::-1] if _area(P) < 0 else P)


def _profile_plain(t=1.0, W=40.0):
    return np.array([(-W, 0.0), (0.0, 0.0), (0.0, t), (-W, t)])


def _extrude(P, L=30.0, ds=0.25):
    P = np.asarray(P, float)
    pts = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / ds)))
        pts += [a + (b - a) * k / n for k in range(n)]
    Q = np.array(pts)
    m = len(Q)
    ys = np.linspace(0, L, max(2, int(L / ds) + 1))
    V = np.array([(q[0], y, q[1]) for y in ys for q in Q])
    F = []
    for j in range(len(ys) - 1):
        for i in range(m):
            a, b = j * m + i, j * m + (i + 1) % m
            c, d = (j + 1) * m + i, (j + 1) * m + (i + 1) % m
            F += [(a, b, d), (a, d, c)]
    F = np.array(F)
    cr = np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]])
    e = Q[1] - Q[0]
    out2 = np.array([e[1], -e[0]])
    if cr[0] * out2[0] + cr[2] * out2[1] < 0:
        F = F[:, ::-1]
    return V, F


def _mesh(P, seed=0):
    from research.zeroshot.zs_lane_v2 import z3s
    V, F = _extrude(P)
    V = V + np.random.default_rng(seed).normal(0, 0.02, V.shape)
    return z3s.mesh_from_arrays(V, F)


@pytest.mark.parametrize("t,gap", [(1.0, 0.0), (1.5, 2.0)])
def test_zs_v4_hem_detects_synthetic_closed_and_open_hems_on_the_crown(t, gap):
    from research.zeroshot.zs_lane_v4 import hem
    lines = hem.detect(_mesh(_profile_hem(t=t, gap=gap, F=10.0)), t)
    assert len(lines) == 1
    ln = lines[0]
    a, b = np.asarray(ln["start_s"]), np.asarray(ln["end_s"])
    R = t + gap / 2.0
    assert abs(a[0] - R) < 0.15 and abs(b[0] - R) < 0.15            # on the crown apex (x = outer radius)
    assert abs(a[2] - (t + gap / 2.0)) < 0.2                          # at the crown centre height
    assert abs(abs(b[1] - a[1]) - 30.0) < 1.5                         # the full extruded length
    assert ln["src"] == ["hem"] and 1.6 <= ln["hem"]["stack_t"] <= 4.0
    assert abs(ln["hem"]["t_local"] - t) < 0.1                         # the single sheet behind the flange end


@pytest.mark.parametrize("t,t_given", [(1.0, 1.0), (2.0, 2.0), (2.0, 0.6)])
def test_zs_v4_hem_rejects_a_plain_cut_edge_even_with_a_wrong_thickness(t, t_given):
    """A single sheet has no flange-end step behind its edge, so a too-small global t cannot fake a stack."""
    from research.zeroshot.zs_lane_v4 import hem
    assert hem.detect(_mesh(_profile_plain(t=t)), t_given) == []


# ----------------------------------------------------------------------------- a synthetic row of tabs
def _tab_record(n_tabs=5, tab=20.0, gap=6.0):
    """Collinear tab candidates along x, a 2-tab merged Z3 candidate, and one long Z2 line over the whole row that
    pooling attached to the merged candidate (the gb_68cd06 configuration)."""
    cands = []
    x = 0.0
    for k in range(n_tabs):
        cands.append({"start": [x, 0, 0], "end": [x + tab, 0, 0], "src": ["z3"], "has_z3": True, "incl": 90.0})
        x += tab + gap
    total = x - gap
    merged = {"start": [0.0, 0.0, 0.0], "end": [2 * tab + gap, 0, 0], "src": ["z2", "z3"], "has_z3": True,
              "incl": 90.0}
    cands.append(merged)
    for c in cands:
        c["start_s"], c["end_s"] = list(c["start"]), list(c["end"])
    z2 = [{"start": [0.0, 0.0, 0.0], "end": [total, 0.0, 0.0], "incl": 90.0}]
    rec = {"candidates": cands, "z2_lines": z2, "evidence": {"sweep": [[1.0] * 6] * len(cands), "pairs": [],
                                                            "splits": {}}, "profiles": {}, "meta": {"t_mm": 1.0}}
    return rec, n_tabs


def test_zs_v4_safe_ext_does_not_swallow_a_row_of_tabs_but_v3_ext_does():
    from research.zeroshot.zs_lane_v4 import select as SL
    rec, n = _tab_record()
    prob = np.full(len(rec["candidates"]), 0.9)
    v3 = SL.select(rec, prob, SL.SelParams(sweep_min=0.0, join=False, ext="v3"))
    safe = SL.select(rec, prob, SL.SelParams(sweep_min=0.0, join=False, ext="safe"))
    assert len(v3) < n                                   # v3: the extended line absorbs the tabs
    assert len(safe) == n + 1                            # safe: every tab line is kept, nothing extended
    assert all(l.get("fix") != "EXT" for l in safe)


def test_zs_v4_hypothesis_guard_keeps_separate_tabs_against_a_long_z2_extent():
    from research.zeroshot.zs_lane_v4 import select as SL
    rec, n = _tab_record()
    cands = rec["candidates"]
    m = len(cands) - 1
    alt = {"kind": "z2ext", "parents": [m], "start_s": rec["z2_lines"][0]["start"], "end_s": rec["z2_lines"][0]["end"],
           "start": rec["z2_lines"][0]["start"], "end": rec["z2_lines"][0]["end"], "lo": 0.0, "hi": 124.0}
    prob = np.full(len(cands), 0.8)
    pa = np.array([0.99])                               # the long hypothesis scores highest
    guarded = SL.select_h(rec, [alt], prob, pa, SL.SelParams(sweep_min=0.0, hyp=True, alt_guard=True))
    unguarded = SL.select_h(rec, [alt], prob, pa, SL.SelParams(sweep_min=0.0, hyp=True, alt_guard=False))
    assert not any(l["op"] == "z2ext" for l in guarded)
    assert len(unguarded) == 1 and unguarded[0]["op"] == "z2ext"


# ----------------------------------------------------------------------------- matcher and folds
def test_zs_v4_fast_scorer_equals_the_frozen_matcher_on_random_lines():
    from types import SimpleNamespace
    from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
    from research.zeroshot.zs_lane_v4.evalkit import FastScorer
    rng = np.random.default_rng(7)
    for trial in range(60):
        ng, npred = int(rng.integers(1, 7)), int(rng.integers(0, 8))
        G = []
        for k in range(ng):
            a = rng.uniform(-50, 50, 3)
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            G.append({"bend_id": f"B{k}", "line_start": a.tolist(), "line_end": (a + rng.uniform(5, 80) * d).tolist()})
        P = []
        for k in range(npred):
            g = G[int(rng.integers(0, ng))]
            ga, gb = np.asarray(g["line_start"]), np.asarray(g["line_end"])
            jit = rng.normal(0, [2.0, 2.0, 2.0])
            s0, s1 = rng.uniform(-0.3, 0.3), rng.uniform(0.7, 1.3)
            P.append({"start": (ga + s0 * (gb - ga) + jit).tolist(), "end": (ga + s1 * (gb - ga) + jit).tolist()})
        it = SimpleNamespace(case_id="x", gold=G, zero_bend=False, family="f")
        tp, fp, fn, _ = FastScorer([it]).counts("x", P)
        ref = match_lines(G, [{"bend_id": f"P{i}", "line_start": p["start"], "line_end": p["end"]}
                              for i, p in enumerate(P)], Thresholds())
        rtp = len(ref["matches"])
        assert (tp, fp, fn) == (rtp, len(P) - rtp, ng - rtp), trial


def test_zs_v4_outer_folds_keep_families_together_and_are_deterministic():
    from research.zeroshot.zs_lane_v4 import lofo
    pool = [{"family5": f"{k % 23:05d}", "case_id": f"s{k}"} for k in range(100)]
    f1, f2 = lofo.outer_folds(pool), lofo.outer_folds(list(reversed(pool)))
    assert f1 == f2
    assert set(f1.values()) == set(range(lofo.K_OUTER))
    inner = lofo.inner_folds([f"{k:05d}" for k in range(23)])
    assert set(inner.values()) == set(range(lofo.K_INNER))
