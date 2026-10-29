"""POST-HOC diagnostic: top view + perpendicular sections at engine nominal bend lines (CAD frame)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.eval.plyio import read_ply_vertices
ed = Path(sys.argv[1]); out = sys.argv[2]; labels = sys.argv[3].split(",") if len(sys.argv) > 3 else None
P = read_ply_vertices(ed / "registered_scan.ply").vertices
res = json.loads((ed / "result.json").read_text())
bends = [b for b in res["bends"] if b["display"].get("nominal_polyline_mm")]
if labels: bends = [b for b in bends if b["label"] in labels]
def fitline(q):
    c = q.mean(0); u, s, vt = np.linalg.svd(q - c); d = vt[0]
    for _ in range(3):
        r = np.abs((q - c) @ np.array([-d[1], d[0]])); keep = r < max(0.3, 2.5 * np.median(r))
        c = q[keep].mean(0); u, s, vt = np.linalg.svd(q[keep] - c); d = vt[0]
    return c, d, float(np.sqrt(np.mean(((q[keep] - c) @ np.array([-d[1], d[0]])) ** 2))), int(keep.sum())
fig, axs = plt.subplots(1, 1 + len(bends), figsize=(6 * (1 + len(bends)), 6))
ax = axs[0]; # top view along datum normal approx: PCA smallest axis
c0 = P.mean(0); _, _, vt = np.linalg.svd(P[::20] - c0)
e1, e2 = vt[0], vt[1]
ax.scatter((P - c0) @ e1, (P - c0) @ e2, s=0.05, c=(P - c0) @ vt[2], cmap="viridis")
for b in res["bends"]:
    for key, col in (("nominal_polyline_mm", "r"), ("measured_polyline_mm", "b")):
        L = b["display"].get(key)
        if L: L = np.asarray(L); ax.plot((L - c0) @ e1, (L - c0) @ e2, col + "-", lw=2)
    a = np.asarray(b["display"]["label_anchor_mm"]); ax.text((a - c0) @ e1, (a - c0) @ e2, b["label"] + ":" + b["state"][:4], fontsize=8)
ax.set_aspect("equal"); ax.set_title("top (red nominal, blue measured)")
info = {}
for k, b in enumerate(bends):
    L = np.asarray(b["display"]["nominal_polyline_mm"]); q0, q1 = L[0], L[-1]
    d = (q1 - q0) / np.linalg.norm(q1 - q0); mid = 0.5 * (q0 + q1); half = 0.5 * np.linalg.norm(q1 - q0)
    sa = np.asarray(b["display"]["side_a_anchor_mm"]) - mid; sa -= (sa @ d) * d; ua = sa / np.linalg.norm(sa)
    va = np.cross(d, ua)
    rel = P - mid; ax_ = rel @ d
    m = (np.abs(ax_) < 0.6 * half) & (np.linalg.norm(rel - np.outer(ax_, d), axis=1) < 60)
    q = np.c_[rel[m] @ ua, rel[m] @ va]
    a2 = axs[k + 1]; a2.scatter(q[:, 0], q[:, 1], s=0.2, c="k")
    sb = np.asarray(b["display"]["side_b_anchor_mm"]) - mid; sb2 = np.array([sb @ ua, sb @ va])
    a2.plot([0, 30], [0, 0], "r-"); a2.plot([0, sb2[0] * 30 / np.linalg.norm(sb2)], [0, sb2[1] * 30 / np.linalg.norm(sb2)], "r-")
    # fit flanges: side A points u>5, side B: points closer to direction sb2 beyond 5mm
    ub = sb2 / np.linalg.norm(sb2)
    A = q[(q @ np.array([1, 0]) > 6) & (q @ np.array([1, 0]) < 45) & (np.abs(q[:, 1]) < 15)]
    B = q[(q @ ub > 6) & (q @ ub < 45) & (np.abs(q @ np.array([-ub[1], ub[0]])) < 15)]
    s = ""
    if len(A) > 30 and len(B) > 30:
        ca, da, ra, na = fitline(A); cb, db, rb, nb = fitline(B)
        if da @ np.array([1, 0]) < 0: da = -da
        if db @ ub < 0: db = -db
        inc = np.degrees(np.arccos(np.clip(da @ db, -1, 1)))
        s = f"incl~{inc:.2f} rmsA={ra:.2f} rmsB={rb:.2f} nA={na} nB={nb}"
        info[b["label"]] = round(float(inc), 2)
    ang = [c for c in b["characteristics"] if c["kind"] == "INCLUDED_ANGLE"][0]
    a2.set_title(f"{b['label']} {b['state']} eng={ang.get('actual')} nom={ang.get('expected')} {ang.get('decision')}\n{s}", fontsize=8)
    a2.set_aspect("equal"); a2.set_xlim(-50, 50); a2.set_ylim(-50, 50); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out, dpi=70); print(info)
