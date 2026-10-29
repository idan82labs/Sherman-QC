"""Independent 2D check of a hook chain's middle panel length (OML sharp to OML sharp) on sections of an E3
registered scan (datum/CAD frame). usage: hook_section.py REGISTERED_PLY Y_STATIONS(comma)
Groups (fixed boxes for the 23553000 B7-B6-B5 chain, x-z sections): wall x in [45.4,48], z in [-29,-23];
hook panel: points with z in [-18,-13] (between the curls); lip: x in [35.5,38.5] and z in [-9.5,-7.0].
Diagnostic only (report); never an engine input."""
import sys

import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from domains.production.engine.inspect.scan_io import read_scan  # noqa: E402

pts = read_scan(sys.argv[1]).points


def split_skins(P):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    off = (P - c) @ n
    lo, hi = np.percentile(off, 10), np.percentile(off, 90)
    mid = 0.5 * (lo + hi)
    return P[off < mid], P[off >= mid], hi - lo


def line_fit(P):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def intersect(c1, d1, c2, d2):
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


for y0 in [float(s) for s in sys.argv[2].split(",")]:
    S = pts[np.abs(pts[:, 1] - y0) < 0.4][:, [0, 2]]
    wall = S[(S[:, 0] > 45.4) & (S[:, 0] < 48) & (S[:, 1] > -29) & (S[:, 1] < -23)]
    hook = S[(S[:, 1] > -18) & (S[:, 1] < -13) & (S[:, 0] > 34) & (S[:, 0] < 46)]
    lip = S[(S[:, 0] > 35.5) & (S[:, 0] < 38.5) & (S[:, 1] > -9.5) & (S[:, 1] < -7.0)]
    if min(len(wall), len(hook), len(lip)) < 20:
        print(f"y={y0}: too few points wall/hook/lip {len(wall)}/{len(hook)}/{len(lip)}")
        continue
    wa, wb, wt = split_skins(wall)
    ha, hb, ht = split_skins(hook)
    la, lb, lt = split_skins(lip)
    W = [line_fit(wa), line_fit(wb)]
    H = [line_fit(ha), line_fit(hb)]
    L = [line_fit(la), line_fit(lb)]
    w_oml = max(W, key=lambda q: q[0][0])  # outer side of the B6 corner: the wall's +x skin
    # the hook panel's skin on the outside of the B6 corner: the one lower at the panel's centre x
    h_oml = min(H, key=lambda q: q[0][1] + (q[1][1] / q[1][0]) * (40.0 - q[0][0]))
    w_iml = W[1] if w_oml is W[0] else W[0]
    h_iml = H[1] if h_oml is H[0] else H[0]
    out = []
    for name, lo in (("lipA", L[0]), ("lipB", L[1])):
        s5 = intersect(h_oml[0], h_oml[1], lo[0], lo[1])
        out.append((name, s5))
    s6 = intersect(w_oml[0], w_oml[1], h_oml[0], h_oml[1])
    s6i = intersect(w_iml[0], w_iml[1], h_iml[0], h_iml[1])
    ang6 = np.degrees(np.arccos(abs(float(w_oml[1] @ h_oml[1]))))
    msg = " ".join(f"{nm}: B5 sharp {np.round(s, 2)} len {np.linalg.norm(s6 - s):.2f}" for nm, s in out)
    print(f"y={y0}: n {len(wall)}/{len(hook)}/{len(lip)} t {wt:.2f}/{ht:.2f}/{lt:.2f} B6 OML sharp {np.round(s6, 2)} "
          f"(IML {np.round(s6i, 2)}) wall-hook angle {ang6:.1f} | {msg} (nominal 18.18)")
