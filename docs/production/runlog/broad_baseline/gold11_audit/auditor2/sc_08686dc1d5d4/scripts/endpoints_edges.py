import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
from scipy.spatial import cKDTree
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy"); pts = np.load(W / "pts.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
tree = cKDTree(P)
# display cloud (200k raw-vertex subset used by the labeller)
import open3d as o3d
disp = np.asarray(o3d.io.read_point_cloud("/Users/idantal/ShermanQC/FullProject/output/bend_label_02_10839000/aligned_scan.ply").points)
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy"); B = np.vstack([e1, e2, e3])
dispP = ((disp - c) @ B.T) @ R.T
dtree = cKDTree(dispP)
# is display a subset of raw vertices?
rt = cKDTree(pts); dd, _ = rt.query(disp[:2000]); print("display->raw nearest max %.2e (subset check)" % dd.max())
names = {"M01.start/M03.start": G["M01"][0], "M01.end/M04.start": G["M01"][1], "M02.start/M03.end": G["M02"][0], "M02.end/M04.end": G["M02"][1]}
for nm, p in names.items():
    d, i = tree.query(p); dd2, j = dtree.query(p)
    q = P[i]
    # neighbourhood normal at the nearest point
    nb = tree.query_ball_point(q, 1.2)
    X = P[nb] - P[nb].mean(0); w, v = np.linalg.eigh(X.T @ X); n = v[:, 0]
    print(f"{nm}: gold {p.round(2)} | nearest dense pt {q.round(2)} d={d:.2f} | nearest display pt d={dd2:.2f} | offset vec gold-nearest {np.round(p-q,2)} | local normal {np.round(n,2)}")
# plate free edge beyond the walls: y of the outermost plate points vs fold line, for x bins
for side, s, yw in (("M01 side (y+)", 1, 152.86), ("M02 side (y-)", -1, -158.66)):
    for x0 in (-170, -150, -125, -110, 110, 125, 150, 170):
        m = (np.abs(P[:, 0] - x0) < 1.0) & (P[:, 2] < 2) & (s * P[:, 1] > 100)
        if m.sum() == 0: continue
        yedge = (P[m, 1].max() if s > 0 else P[m, 1].min())
        print(f"   {side} x={x0}: plate free-edge y={yedge:.2f} ; fold mid-surface line y={yw:.2f} ; edge - fold = {s*(yedge-yw):+.2f} mm")
