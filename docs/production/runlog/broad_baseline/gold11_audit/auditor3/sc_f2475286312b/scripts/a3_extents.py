"""Physical fold extents from flange-B occupancy just beyond the bend arc (both flanges needed),
1 mm bins along each fold axis."""
import json
import numpy as np
from a3common import load_mesh, OUT

V, N, F = load_mesh()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
X = (V - o) @ R.T
A, B, H = X[:, 0], X[:, 1], X[:, 2]
# boxes: (axis column, flange-A box, flange-B box) each box = dict(col: (lo, hi)) on the two other coords
FOLDS = {
    "LIP_+b": (0, {1: (104, 109), 2: (-1.2, 1.2)}, {1: (111.5, 114.5), 2: (4, 9)}),
    "LIP_-b": (0, {1: (-107, -102), 2: (-1.2, 1.2)}, {1: (-112.5, -109.5), 2: (4, 9)}),
    "TAB_+b": (0, {1: (86, 92), 2: (-1.2, 1.2)}, {1: (80.5, 84.0), 2: (-9, -4)}),
    "TAB_-b": (0, {1: (-99, -94), 2: (-1.2, 1.2)}, {1: (-92.5, -89.0), 2: (-9, -4)}),
    "WALL": (1, {0: (102, 108), 2: (-1.2, 1.2)}, {0: (111.5, 114.5), 2: (-12, -5)}),
    "SHORT_LIP": (1, {0: (-133, -128), 2: (-1.2, 1.2)}, {0: (-141.5, -138.0), 2: (4, 9)}),
    "EAR_+b": (2, {0: (111.5, 114.5), 1: (102, 108)}, {0: (118, 123), 1: (112, 115)}),
    "EAR_-b": (2, {0: (111.5, 114.5), 1: (-106, -100)}, {0: (118, 123), 1: (-112.5, -109.5)}),
}
res = {}
for name, (col, boxA, boxB) in FOLDS.items():
    def occ(box):
        m = np.ones(len(X), bool)
        for c, (lo, hi) in box.items():
            m &= (X[:, c] > lo) & (X[:, c] < hi)
        return X[m, col]
    ta, tb = occ(boxA), occ(boxB)
    lo = np.floor(min(ta.min() if len(ta) else 0, tb.min() if len(tb) else 0)) - 1
    hi = np.ceil(max(ta.max() if len(ta) else 0, tb.max() if len(tb) else 0)) + 1
    edges = np.arange(lo, hi + 1, 1.0)
    ha, _ = np.histogram(ta, edges)
    hb, _ = np.histogram(tb, edges)
    both = (ha >= 2) & (hb >= 2)
    idx = np.nonzero(both)[0]
    if not len(idx):
        res[name] = None
        print(name, "none")
        continue
    # runs
    runs = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            runs.append((float(edges[s]), float(edges[p + 1])))
            s = i
        p = i
    runs.append((float(edges[s]), float(edges[p + 1])))
    total = sum(r[1] - r[0] for r in runs)
    res[name] = {"axis": "abh"[col], "runs": runs, "span": [runs[0][0], runs[-1][1]], "span_len": runs[-1][1] - runs[0][0],
                 "occupied_len": total}
    print(name, "axis", "abh"[col], "span [%.0f, %.0f] len %.0f, occupied %.0f mm, runs:" % (runs[0][0], runs[-1][1], runs[-1][1] - runs[0][0], total),
          [(round(r[0]), round(r[1])) for r in runs])
(OUT / "a3_extents.json").write_text(json.dumps(res, indent=1))
