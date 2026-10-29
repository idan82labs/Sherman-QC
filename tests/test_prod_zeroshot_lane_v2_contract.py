"""zs_lane_v2: synthetic contract checks (no scan data, no gold).

Covers: plate connectivity across a gap (a window between the same two plates vs two separate tabs), the per-station
normal sweep across the bisector (a formed arc vs a sharp seam), span post-processing (join on two connected
plates, no join when one plate is disconnected, the optional notch mode, split at an interruption with a
disconnected plate, no split across a window), the vendored Z3 defaults equal to v1, the scale-relative peak floor,
and the shared selector feature matrix.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.zeroshot.zs_lane_v2 import evidence as E, postproc as PP, z3s  # noqa: E402


def _plate(x0, x1, y0, y1, z=0.0, step=0.5, normal=(0, 0, 1)):
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    return P, np.tile(np.asarray(normal, float), (len(P), 1))


def _ctx(parts, t=1.0):
    V = np.vstack([p for p, _ in parts])
    N = np.vstack([n for _, n in parts])
    return E.MeshCtx(V, N, np.zeros((0, 3), int), t, spacing=0.5)


def test_plate_connected_window_vs_tabs():
    # line along x at y=0; flank plate in +y (normal z). Window: plate continues above the hole.
    win = [_plate(0, 40, 4, 30)]
    hole = (win[0][0][:, 0] > 15) & (win[0][0][:, 0] < 25) & (win[0][0][:, 1] < 20)
    win = [(win[0][0][~hole], win[0][1][~hole])]
    ctx = _ctx(win)
    p0, d, f = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert E.plate_connected(ctx, p0, d, f, 15.0, 25.0, a_hi=32.0)["connected"]
    # two separate tabs with a 3 mm slot between them
    tabs = [_plate(0, 18.5, 4, 30), _plate(21.5, 40, 4, 30)]
    ctx2 = _ctx(tabs)
    assert not E.plate_connected(ctx2, p0, d, f, 18.5, 21.5, a_hi=32.0)["connected"]


def _bend(theta_deg, R, length, step=0.25, sharp=False):
    """Outer skin of a bend along x: flank A in z=0 towards -y (normal +z), an arc of radius R centred at
    (y=0, z=-R) whose normal turns from +z by theta, then flank B. sharp=True: no arc (a seam / sharp edge)."""
    th = math.radians(theta_deg)
    R = 0.0 if sharp else R
    nb = np.array([0, math.sin(th), math.cos(th)])
    db = np.array([0, math.cos(th), -math.sin(th)])
    pts, nrm = [], []
    for x in np.arange(0, length + 1e-9, step):
        c = np.array([x, 0.0, -R])
        for a in np.arange(0.0, 15, step):
            pts.append([x, -a, 0.0]); nrm.append([0, 0, 1.0])
            pts.append((c + R * nb + db * a).tolist()); nrm.append(nb.tolist())
        if not sharp:
            for phi in np.linspace(0, th, 30):
                n = np.array([0, math.sin(phi), math.cos(phi)])
                pts.append((c + R * n).tolist()); nrm.append(n.tolist())
    apex = np.array([0.0, 0.0, -R]) + R * np.array([0, math.sin(th / 2), math.cos(th / 2)])
    line = {"start": apex.tolist(), "end": (apex + [length, 0, 0]).tolist(), "da": [0, -1.0, 0], "db": db.tolist()}
    return np.asarray(pts), np.asarray(nrm), line


def test_station_sweep_arc_vs_seam():
    P, N, c = _bend(60.0, 3.0, 20.0)
    prof = E.station_profile(E.MeshCtx(P, N, np.zeros((0, 3), int), 1.0, spacing=0.25), c)
    assert prof["sweep"].mean() > 0.8 and abs(prof["theta"] - 60.0) < 1e-6
    Ps, Ns, cs = _bend(60.0, 3.0, 20.0, sharp=True)
    prof2 = E.station_profile(E.MeshCtx(Ps, Ns, np.zeros((0, 3), int), 1.0, spacing=0.25), cs)
    assert prof2["sweep"].mean() < 0.2


def _c(s, e, **kw):
    c = {"start": list(map(float, s)), "end": list(map(float, e)), "incl": 90.0, "src": ["z3"],
         "da": [0, 1.0, 0], "db": [0, 0, 1.0], "flank_a": 20.0, "flank_b": 20.0}
    c["start_s"], c["end_s"] = c["start"], c["end"]
    c.update(kw)
    return c


def _pair(ca, cb, gap=20.0, li=40.0, lj=40.0, fi=(20, 20), fj=(20, 20)):
    return {"i": 0, "j": 1, "g0": li, "g1": li + gap, "gap": gap, "len_i": li, "len_j": lj, "has_flanks": True,
            "ca": ca, "cb": cb, "flank_i": list(fi), "flank_j": list(fj)}


def test_join_needs_both_plates():
    C = [_c([0, 0, 0], [40, 0, 0]), _c([60, 0, 0], [100, 0, 0])]
    prob = np.array([0.9, 0.8])
    out = PP.postprocess(C, prob, 0.5, {"pairs": [_pair(True, True)]}, PP.PostParams(join=True))
    assert len(out) == 1 and out[0]["op"] == "join"
    assert np.allclose(out[0]["start_s"], [0, 0, 0]) and np.allclose(out[0]["end_s"], [100, 0, 0])
    out = PP.postprocess(C, prob, 0.5, {"pairs": [_pair(True, False)]}, PP.PostParams(join=True))
    assert len(out) == 2 and all(o["op"] == "keep" for o in out)
    out = PP.postprocess(C, prob, 0.5, {"pairs": [_pair(True, True)]}, PP.PostParams())
    assert len(out) == 2
    # notch mode: one plate connected, equal flank heights, coverage 80/100 >= 0.8
    out = PP.postprocess(C, prob, 0.5, {"pairs": [_pair(False, True)]},
                         PP.PostParams(join=True, join_mode="notch", notch_cov=0.8))
    assert len(out) == 1
    out = PP.postprocess(C, prob, 0.5, {"pairs": [_pair(False, True, fi=(5, 20))]},
                         PP.PostParams(join=True, join_mode="notch", notch_cov=0.8))
    assert len(out) == 2
    # a low-probability piece joins only when allowed and connected to a selected one
    prob2 = np.array([0.9, 0.3])
    assert len(PP.postprocess(C, prob2, 0.5, {"pairs": [_pair(True, True)]}, PP.PostParams(join=True))) == 1
    out = PP.postprocess(C, prob2, 0.5, {"pairs": [_pair(True, True)]}, PP.PostParams(join=True, join_low=0.2))
    assert len(out) == 1 and np.allclose(out[0]["end_s"], [100, 0, 0])


def test_split_on_disconnected_plate_only():
    C = [_c([0, 0, 0], [100, 0, 0])]
    run = {"g0": 40.0, "g1": 50.0, "ca": True, "cb": False, "sides_a": 2, "sides_b": 2, "raw_a": 1.0, "raw_b": 0.0}
    out = PP.postprocess(C, np.array([0.9]), 0.5, {"splits": {"0": [run]}}, PP.PostParams(split=True))
    assert len(out) == 2 and {o["op"] for o in out} == {"split"}
    assert np.allclose(out[0]["end_s"], [40, 0, 0]) and np.allclose(out[1]["start_s"], [50, 0, 0])
    win = dict(run, cb=True)
    out = PP.postprocess(C, np.array([0.9]), 0.5, {"splits": {"0": [win]}}, PP.PostParams(split=True))
    assert len(out) == 1 and out[0]["op"] == "keep"
    present = dict(run, raw_b=0.9)          # plate material inside the run: not a slot between tabs
    assert len(PP.postprocess(C, np.array([0.9]), 0.5, {"splits": {"0": [present]}}, PP.PostParams(split=True))) == 1
    oneside = dict(run, sides_b=1)          # plate missing on one side: no verdict, no split
    assert len(PP.postprocess(C, np.array([0.9]), 0.5, {"splits": {"0": [oneside]}}, PP.PostParams(split=True))) == 1
    small = dict(run, g0=40.0, g1=40.5)
    out = PP.postprocess(C, np.array([0.9]), 0.5, {"splits": {"0": [small]}}, PP.PostParams(split=True, split_min_gap=1.0))
    assert len(out) == 1


def test_vendored_z3_defaults_equal_v1():
    from research.zeroshot.z3_slicing import slicer
    a, b = asdict(slicer.Params()), asdict(z3s.Params())
    assert {k: v for k, v in b.items() if k in a} == a
    assert b["peak_min_area_rel"] == 0.0


def test_scale_relative_floor_only_lowers_small_parts():
    prm = z3s.Params(peak_min_area_rel=0.004)
    for total, expect in ((100000.0, max(0.0005 * 100000.0, 20.0)), (2000.0, max(0.0005 * 2000.0, 8.0))):
        floor = min(prm.peak_min_area_mm2, prm.peak_min_area_rel * total)
        assert max(prm.peak_min_area_frac * total, floor) == expect


def test_feature_matrix_shapes():
    X = np.zeros((3, 26))
    SW = np.ones((3, len(E.SWEEP_FEATS)))
    assert PP.features(X, SW, "base").shape == (3, 26)
    assert PP.features(X, SW, "base,sweep").shape == (3, 26 + len(E.SWEEP_FEATS))
    M = PP.sweep_matrix([None, [1] * len(E.SWEEP_FEATS)], 2)
    assert np.isnan(M[0]).all() and (M[1] == 1).all()
