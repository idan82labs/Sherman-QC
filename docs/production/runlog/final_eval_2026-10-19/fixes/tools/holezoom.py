import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.eval.plyio import read_ply_vertices
ed = Path(sys.argv[1]); cp = json.loads(Path(sys.argv[2]).read_text()); out = sys.argv[3]; pid = sys.argv[4] if len(sys.argv) > 4 else cp["datum_panel_id"]
src = sys.argv[5] if len(sys.argv) > 5 else None
P = read_ply_vertices(src).vertices if src else read_ply_vertices(ed / "registered_scan.ply").vertices
if src:
    T = np.asarray(json.loads((ed / "result.json").read_text())["inspection_validity"]["registration"]["raw_to_frame"])
    P = P @ T[:3, :3].T + T[:3, 3]
p = {q["panel_id"]: q for q in cp["panels"]}[pid]
pl = p["skin0_plane"]; n = np.asarray(pl.get("normal") or pl["n"]); o = np.asarray(pl["point_mm"])
t = p["thickness_mm"]
h = (P - o) @ n
holes = [np.asarray(x) for x in p["holes_mm"]][:4]
fig, axs = plt.subplots(2, len(holes), figsize=(4 * len(holes), 8))
axs = np.atleast_2d(axs)
for k, hl in enumerate(holes):
    c = hl.mean(0); r = np.linalg.norm(hl - c, axis=1).max()
    e1 = np.cross(n, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 0.1 else np.cross(n, [0, 1, 0]) / np.linalg.norm(np.cross(n, [0, 1, 0]))
    e2 = np.cross(n, e1)
    rel = P - c
    m = (np.abs(rel @ e1) < 3 * r + 5) & (np.abs(rel @ e2) < 3 * r + 5) & (np.abs(rel @ n) < 3 * t + 5)
    q = np.c_[rel[m] @ e1, rel[m] @ e2, rel[m] @ n]
    ax = axs[0, k]; ax.scatter(q[:, 0], q[:, 1], s=0.5, c=q[:, 2], cmap="coolwarm"); ax.plot((hl - c) @ e1, (hl - c) @ e2, "g-")
    ax.set_aspect("equal"); ax.set_title(f"hole {k} r~{r:.1f} top (color=height)")
    ax = axs[1, k]; ax.scatter(q[:, 0], q[:, 2], s=0.5, c="k"); ax.axhline(0, c="g"); ax.axhline(-t, c="g"); ax.set_title("side: height vs e1")
plt.tight_layout(); plt.savefig(out, dpi=60)
