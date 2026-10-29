"""Relabel sc_71cf0b216f43: part frame + profile sections from the prep ds cloud (read-only).

Part frame: z = up (bottom ring -> top flanges), x = long-wall direction, y = short direction.
Writes section plots of the long-side top flange (wall fold + joggle) and the short-side lips.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREP = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/prep/sc_71cf0b216f43")
OUT = Path(__file__).resolve().parent
ds = np.load(PREP / "points_ds.npz")
P, N, L = ds["points"], ds["normals"], ds["labels"]
patches = {p["id"]: p for p in json.loads((PREP / "patches.json").read_text())["patches"]}

def fitn(ids):
    pts = P[np.isin(L, ids)]
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    return vt[2], c

n1, c1 = fitn([1])
up = -n1 if n1 @ np.array(patches[5]["normal"]) < 0 else n1   # up = direction of top faces' outward normal
up = up / np.linalg.norm(up)
n3, c3 = fitn([3])
ex = np.cross(n3, up); ex /= np.linalg.norm(ex)
if ex[int(np.argmax(np.abs(ex)))] < 0:
    ex = -ex
ey = np.cross(up, ex)
origin = P.mean(0)
R = np.stack([ex, ey, up])
Q = (P - origin) @ R.T
frame = {"origin": origin.tolist(), "ex": ex.tolist(), "ey": ey.tolist(), "ez": up.tolist()}
(OUT / "part_frame.json").write_text(json.dumps(frame, indent=2))
print("frame", json.dumps(frame))
for pid in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 23):
    q = Q[L == pid]
    print(pid, "n", len(q), "x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f] zmed %.2f" % (
        q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), q[:, 2].min(), q[:, 2].max(), np.median(q[:, 2])))
