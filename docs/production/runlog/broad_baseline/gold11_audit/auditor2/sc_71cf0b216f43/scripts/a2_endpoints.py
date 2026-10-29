"""Auditor 2: gold endpoint diagnostics (plan distance to the inner vertical box corner, height vs the bend height)."""
import json, numpy as np
from scipy.spatial import cKDTree
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
tree = cKDTree(P)
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
# inner skins of the walls (from the normal histograms): e1-min inner -147.6, e1-max inner ~162.4, e2-min inner -97.75, e2-max inner ~105.3
inner_e1 = {-1: -147.6, +1: 162.4}; inner_e2 = {-1: -97.75, +1: 105.3}
bend_h = {'top': 19.5, 'bottom': -19.9}   # approx inner-corner heights of top/bottom bends
which = {'M01': 'bottom', 'M02': 'bottom', 'M03': 'top', 'M04': 'top', 'M05': 'top', 'M06': 'bottom', 'M07': 'top', 'M08': 'bottom'}
rows = []
for k, (A, B) in GL.items():
    for tag, E in (('start', A), ('end', B)):
        d, i = tree.query(E)
        s1 = -1 if E[0] < 0 else 1; s2 = -1 if E[1] < 0 else 1
        plan_d = float(np.hypot(E[0]-inner_e1[s1], E[1]-inner_e2[s2]))
        rows.append(dict(line=k, end=tag, e1=round(float(E[0]), 1), e2=round(float(E[1]), 1), e3=round(float(E[2]), 1),
                         plan_dist_to_inner_vertical_corner_mm=round(plan_d, 1), height_minus_bend_mm=round(float(E[2]-bend_h[which[k]]), 1),
                         nearest_scan_mm=round(float(d), 2), nearest_scan_normal=[round(float(x), 2) for x in Nn[i]]))
for r in rows: print(r)
json.dump(rows, open(OUT + '/a2_endpoints.json', 'w'), indent=1)
