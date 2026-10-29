#!/usr/bin/env python3
"""Residual of each arm's skin points from its far-field skin planes vs distance from the outer virtual
sharp (aggregated over all fold stations). A straight plate gives ~0 residual right up to the bend
tangent; bowing or a hidden second fold shows a growing residual / a step. Read-only."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_sections as A

V, N = A.load(); tree = cKDTree(V)
rep = json.loads((HERE / "audit_sections.json").read_text())
want = sys.argv[2:] if len(sys.argv) > 2 else ["M01", "M02", "M03", "M04", "M05"]
out = {}
for r in rep["lines"]:
    bid = r["bend_id"]
    if bid not in want:
        continue
    f = r["frame"]; d = np.asarray(f["d"])
    aP = np.asarray(f["aP"]); aQ = np.asarray(f["aQ"])
    st = [x for x in r["stations"] if x.get("present") and x.get("loc_planes")]
    acc = {"P": [], "Q": []}
    for s in st[::2]:
        X0 = np.asarray(s["ovs_local_xyz"]); lp = s["loc_planes"]
        ii = np.array(tree.query_ball_point(X0, 70.0)); Q = V[ii]
        Q = Q[np.abs((Q - X0) @ d) < 0.5]
        for arm, a, ko, ki in (("P", aP, "Pout", "Pin"), ("Q", aQ, "Qout", "Qin")):
            no, co = np.asarray(lp[ko][0]), np.asarray(lp[ko][1])
            dist_o = (Q - co) @ no            # + toward concave (inner side)
            along = (Q - X0) @ a
            # keep points on this arm: projected along the arm and within 3.5 mm of its outer plane band
            sel = (along > 0) & (along < 60) & (dist_o > -3) & (dist_o < 5)
            # exclude the other arm's points: they lie far from this plane only beyond the corner; use a cone test
            acc[arm].append(np.c_[along[sel], dist_o[sel]])
    res = {}
    for arm in ("P", "Q"):
        Mx = np.vstack(acc[arm]) if acc[arm] else np.zeros((0, 2))
        rows = []
        for lo in np.arange(0, 56, 2.0):
            b = Mx[(Mx[:, 0] >= lo) & (Mx[:, 0] < lo + 2)]
            if len(b) < 10:
                rows.append((lo, None, None, len(b)))
                continue
            # outer skin = lowest cluster (dist_o near its 5th percentile), inner = highest cluster
            o = b[:, 1][b[:, 1] < np.percentile(b[:, 1], 5) + 0.6]
            i_ = b[:, 1][b[:, 1] > np.percentile(b[:, 1], 95) - 0.6]
            rows.append((float(lo), float(np.median(o)), float(np.median(i_)), int(len(b))))
        res[arm] = rows
    out[bid] = res
    print(f"== {bid} bend {r['bend_angle_deg']:.1f}  (residual of OUTER-skin points from far-field outer plane, mm; inner in brackets)")
    for arm, name in (("P", r["plates"][0]), ("Q", r["plates"][1])):
        print(f"  {name:5s}", " ".join(f"{lo:.0f}:{o:+.2f}[{i:+.2f}]" if o is not None else f"{lo:.0f}:--" for lo, o, i, n in res[arm]))
(HERE / "residual_profile.json").write_text(json.dumps(out, indent=1))
