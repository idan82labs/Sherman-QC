#!/usr/bin/env python3
"""Local surface-angle profile across each bend (outer skin), to separate one discrete fold from
bowing / a hidden second fold. Read-only. Prints the outer-skin tangent angle vs arc-length."""
import sys
sys.dont_write_bytecode = True
import json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_sections as A

V, N = A.load(); tree = cKDTree(V)
rep = json.loads((HERE / "audit_sections.json").read_text())
out = {}
for r in rep["lines"]:
    bid = r["bend_id"]; f = r["frame"]
    d = np.asarray(f["d"]); g = np.asarray(f["g"]); h = np.asarray(f["h"])
    aP = np.asarray(f["aP"]); aQ = np.asarray(f["aQ"])
    st = [x for x in r["stations"] if x.get("present") and "ovs_local_xyz" in x]
    prof = []
    for s in st[len(st)//5: 4*len(st)//5: max(1, len(st)//12)]:
        X0 = np.asarray(s["ovs_local_xyz"])
        ii = np.array(tree.query_ball_point(X0, 80.0))
        Q = V[ii]; NQ = N[ii]
        m = np.abs((Q - X0) @ d) < 1.0
        Q, NQ = Q[m], NQ[m]
        # outer skin points: normals pointing to the convex side (n.g < 0 after orienting with the outer planes)
        nPo = -np.asarray(f["nP_in"]); nQo = -np.asarray(f["nQ_in"])
        # classify each point to P side / Q side by its in-plane coordinate
        xp = (Q - X0) @ aP; xq = (Q - X0) @ aQ
        # angle of the surface normal in the section, measured from the P-plate normal toward the Q-plate normal
        n2 = np.c_[NQ @ g, NQ @ h]
        ang = np.degrees(np.arctan2(NQ @ h, NQ @ g))
        prof.append((Q, NQ, X0))
    # summarise: for points on the OUTER skin (closest skin to the convex side), bin by signed distance along the surface
    rows = []
    for Q, NQ, X0 in prof:
        y = (Q - X0) @ h   # + toward P (base)
        x = (Q - X0) @ g   # + toward concave
        nn = np.c_[NQ @ h, NQ @ g]
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        # orient normals to point toward -g (convex side) where ambiguous: outer skin normal has n.g<0 for outward meshes
        rows.append((y, x, nn))
    # tangent angle profile on each arm as a function of distance along the arm from the corner
    res = {}
    for arm, sgn in (("P(base)", 1), ("Q(flange)", -1)):
        bins = np.arange(0, 46, 2.0)
        ang_b = []
        for lo in bins:
            vals = []
            for y, x, nn in rows:
                sel = (sgn * y >= lo) & (sgn * y < lo + 2) & (np.abs(nn[:, 1]) > 0.05)
                if sel.sum() < 4:
                    continue
                # local direction of the surface trace: fit a line to points in this bin (outer-most skin only)
                yy, xx = y[sel], x[sel]
                # outer skin = lower envelope in x for this arm
                lowx = xx < np.percentile(xx, 50) + 0.3
                if lowx.sum() < 3:
                    continue
                A_ = np.c_[yy[lowx], np.ones(lowx.sum())]
                sl, _ = np.linalg.lstsq(A_, xx[lowx], rcond=None)[0]
                vals.append(math.degrees(math.atan(sl)) * sgn)
            ang_b.append((float(lo), float(np.median(vals)) if vals else None, len(vals)))
        res[arm] = ang_b
    out[bid] = res
    print("==", bid, "bend", round(r["bend_angle_deg"], 1))
    for arm in res:
        print("  ", arm, " ".join(f"{lo:.0f}:{a:+.1f}" if a is not None else f"{lo:.0f}:--" for lo, a, n in res[arm]))
(HERE / "angle_profile.json").write_text(json.dumps(out, indent=1))
