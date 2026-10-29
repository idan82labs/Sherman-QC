"""Part frame for sc_08686dc1d5d4: e1 along gold M01/M02 direction (in plate), e2 along M03/M04, e3 plate normal."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from a1_common import load_all, pca_frame, unit, gold_lines


def part_frame(xyz, nrm):
    c, vt = pca_frame(xyz)
    a = nrm @ vt[2]
    m = np.abs(a) > 0.98
    p3 = (xyz - c) @ vt[2]
    # robust plane on plate-parallel points near the dominant level (upper slab, not the lips)
    sel = m & (p3 > -10)
    P = xyz[sel]
    for _ in range(3):
        cc = P.mean(0)
        _, _, v = np.linalg.svd(P[::3] - cc, full_matrices=False)
        n = v[2]
        d = (P - cc) @ n
        keep = np.abs(d - np.median(d)) < 3 * max(1e-3, np.median(np.abs(d - np.median(d))) * 1.4826)
        P = P[keep]
    n = unit(n)
    if n @ vt[2] < 0:
        n = -n
    e1 = unit(vt[0] - (vt[0] @ n) * n)
    e2 = np.cross(n, e1)
    return cc, np.stack([e1, e2, n])


if __name__ == "__main__":
    xyz, nrm, faces = load_all()
    o, R = part_frame(xyz, nrm)
    print("origin", o.round(3)); print("R", R.round(5))
    np.savez(__file__.replace("a1_frame.py", "a1_frame.npz"), o=o, R=R)
    p = (xyz - o) @ R.T
    print("extent", p.min(0).round(2), p.max(0).round(2))
    g = gold_lines()
    for k, (s, e) in g.items():
        print(k, ((s - o) @ R.T).round(2), ((e - o) @ R.T).round(2))
