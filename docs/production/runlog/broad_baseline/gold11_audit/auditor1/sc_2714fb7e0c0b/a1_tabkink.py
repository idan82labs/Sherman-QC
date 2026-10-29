"""Quantify the 30-deg kink at the end of each window tab (CAD C5/C7 analogue) in the scan.
In N slabs through the tab (N = -15 .. -26), fit a line to the straight tab portion (Z in [-20, 9])
and to the kinked portion (Z in [15, 19]) using all tab points (both skins -> mid-surface), and report
the included kink angle and the kink vertex. Read-only on inputs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import a1common as A

pts, nrm, _ = A.load_scan()
g = A.load_gold()
ex, ez, en = A.part_frame(g)
c = pts.mean(0)
M = np.c_[ex, ez, en].T
P = (pts - c) @ M.T


def fitline(q):
    m = q.mean(0)
    _, _, vt = np.linalg.svd(q - m)
    return m, vt[0]


out = {}
for bid, x0 in (("B09", -82.0), ("B10", 92.3)):
    rows = []
    for nl in np.arange(-15.0, -26.5, -1.0):
        sl = (np.abs(P[:, 2] - nl) < 0.4) & (np.abs(P[:, 0] - x0) < 7)
        q = P[sl][:, :2]
        s1 = q[(q[:, 1] > -20) & (q[:, 1] < 9)]
        s2 = q[(q[:, 1] > 15) & (q[:, 1] < 19)]
        if len(s1) < 20 or len(s2) < 8:
            rows.append({"N": float(nl), "n1": int(len(s1)), "n2": int(len(s2))}); continue
        m1, d1 = fitline(s1)
        m2, d2 = fitline(s2)
        ang = float(np.degrees(np.arccos(min(1.0, abs(d1 @ d2)))))
        Mx = np.array([d1, -d2]).T
        t = np.linalg.solve(Mx, m2 - m1)
        vtx = m1 + t[0] * d1
        rows.append({"N": float(nl), "kink_deg": ang, "vertex_XZ": vtx.tolist(), "n1": int(len(s1)), "n2": int(len(s2))})
    ok = [r for r in rows if "kink_deg" in r]
    v = np.array([[r["vertex_XZ"][0], r["vertex_XZ"][1], r["N"]] for r in ok])
    summ = {"kink_deg_median": float(np.median([r["kink_deg"] for r in ok])),
            "kink_deg_range": [float(min(r["kink_deg"] for r in ok)), float(max(r["kink_deg"] for r in ok))],
            "vertex_part_first_last": [v[0].tolist(), v[-1].tolist()],
            "vertex_raw_first_last": [(c + v[0] @ M).tolist(), (c + v[-1] @ M).tolist()],
            "N_span_measured": [float(v[:, 2].max()), float(v[:, 2].min())], "rows": rows}
    out[bid + "_tab_kink"] = summ
    print(bid, {k: summ[k] for k in summ if k != "rows"})
(A.OUT / "a1_tabkink.json").write_text(json.dumps(out, indent=1))
