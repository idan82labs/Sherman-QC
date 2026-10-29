"""Z3 axis-family slicing: synthetic contract checks (no scan data, no custody-relevant input)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.zeroshot.z3_slicing import slicer  # noqa: E402
from research.zeroshot.z3_slicing.custody import CustodyError, check_path  # noqa: E402


def _profile(segments, t, ri, step=0.25):
    """Mid-surface 2-D path from [("L", length), ("B", signed_bend_deg), ...]; returns outer/inner skins."""
    p = np.zeros(2)
    ang = 0.0
    mids, nrms = [], []
    rm = ri + t / 2
    for kind, val in segments:
        if kind == "L":
            n = max(2, int(val / step))
            for s in np.linspace(0, val, n, endpoint=False):
                d = np.array([math.cos(ang), math.sin(ang)])
                mids.append(p + s * d)
                nrms.append(np.array([-d[1], d[0]]))
            p = p + val * np.array([math.cos(ang), math.sin(ang)])
        else:
            b = math.radians(val)
            sgn = 1 if b > 0 else -1
            d = np.array([math.cos(ang), math.sin(ang)])
            c = p + sgn * rm * np.array([-d[1], d[0]])
            n = max(4, int(abs(b) * rm / step))
            for k in range(n):
                a = ang + b * k / n
                q = c - sgn * rm * np.array([-math.sin(a), math.cos(a)])
                mids.append(q)
                nrms.append(np.array([-math.sin(a), math.cos(a)]))
            ang += b
            p = c - sgn * rm * np.array([-math.sin(ang), math.cos(ang)])
    mids.append(p)
    d = np.array([math.cos(ang), math.sin(ang)])
    nrms.append(np.array([-d[1], d[0]]))
    M, N = np.array(mids), np.array(nrms)
    return M + N * t / 2, M - N * t / 2


def _extrude(loop2d, width, step=0.5):
    """Closed 2-D loop (CCW) extruded along z -> triangle mesh with outward winding."""
    zs = np.arange(0, width + 1e-9, step)
    n = len(loop2d)
    V = np.array([[x, y, z] for z in zs for (x, y) in loop2d], float)
    F = []
    for k in range(len(zs) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            c, d = a + n, b + n
            F += [[a, b, d], [a, d, c]]
    F = np.array(F)
    m = slicer.mesh_from_arrays(V, F)
    # outward check: normals should point away from the loop centroid on average -> flip if not
    cen = V.mean(0)
    fc = V[F].mean(1)
    if np.mean(np.einsum("ij,ij->i", m.fn, fc - cen)) < 0:
        m = slicer.mesh_from_arrays(V, F[:, ::-1])
    return m


def _channel(t=2.0, ri=2.0, width=80.0):
    up, lo = _profile([("L", 30), ("B", 90), ("L", 60), ("B", 90), ("L", 30)], t, ri)
    loop = np.vstack([up, lo[::-1]])
    # make CCW
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    if area < 0:
        loop = loop[::-1]
    return _extrude(loop, width), up, lo


def test_thickness_estimate_on_synthetic_sheet():
    m, _, _ = _channel(t=2.0)
    th = slicer.estimate_thickness(m, n_samples=4000)
    assert abs(th["t_mm"] - 2.0) < 0.1


def test_axis_family_is_bend_axis():
    m, _, _ = _channel()
    fams = slicer.axis_families(m, slicer.Params())
    assert fams, "no axis family"
    assert abs(abs(np.asarray(fams[0]["axis"]) @ np.array([0, 0, 1.0])) - 1) < 1e-3


def test_channel_two_bends_outer_apex_and_span():
    t, ri, W = 2.0, 2.0, 80.0
    m, up, lo = _channel(t, ri, W)
    res = slicer.detect(m, slicer.Params())
    C = [c for c in res["candidates"] if c["length"] > 20]
    assert len(C) == 2, [(c["length"], c["incl"]) for c in res["candidates"]]
    # outer skin = the convex side: for this channel the bends turn left, outer skin is "lo"
    ro = ri + t
    # expected outer apex 2-D points: bend centres offset by ro along the bisector
    exp = [np.array([30 + 0.0, 0.0]), np.array([30.0, 60.0])]
    for c in C:
        a, b = np.asarray(c["start"]), np.asarray(c["end"])
        assert abs(abs((b - a)[2]) - W) < 3.0
        assert abs(c["incl"] - 90.0) < 1.0
        mid = 0.5 * (a + b)
        # distance of the apex from the nearest outer-skin vertex must be tiny, from the inner skin ~t
        d_out = np.min(np.linalg.norm(lo - mid[:2], axis=1))
        d_in = np.min(np.linalg.norm(up - mid[:2], axis=1))
        assert d_out < 0.35 and d_in > 0.8 * t, (d_out, d_in)
    del exp, ro


def test_custody_refuses_sealed_and_holdout(tmp_path):
    for name in ("x_TESTB_SEALED/scan.ply", "HOLDOUT10/a.ply", "N015_scan.ply", "38173000_01 (V).ply"):
        with pytest.raises(CustodyError):
            check_path(tmp_path / name)


def test_joggle_gives_two_opposite_bends():
    t, ri, W = 1.5, 1.5, 60.0
    up, lo = _profile([("L", 30), ("B", 45), ("L", 2.5), ("B", -45), ("L", 30)], t, ri)
    loop = np.vstack([up, lo[::-1]])
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    if area < 0:
        loop = loop[::-1]
    m = _extrude(loop, W)
    res = slicer.detect(m, slicer.Params())
    C = [c for c in res["candidates"] if c["length"] > 0.6 * W and abs(c["incl"] - 135.0) < 6.0]
    assert len(C) >= 2, [(round(c["length"]), round(c["incl"])) for c in res["candidates"]]
    mids = [0.5 * (np.asarray(c["start"]) + np.asarray(c["end"]))[:2] for c in C]
    assert max(np.linalg.norm(a - b) for a in mids for b in mids) > 2.0   # two distinct fold lines
