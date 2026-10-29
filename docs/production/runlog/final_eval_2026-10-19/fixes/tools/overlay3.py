"""POST-HOC: CAD final stage (grey) vs registered scan (red) in 3 views, for several engine dirs."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.eval.plyio import read_ply_vertices
cdir = Path(sys.argv[1]); out = sys.argv[2]; runs = sys.argv[3:]
cad = np.load(cdir / json.loads((cdir / "compiled_part.json").read_text())["stages"][-1]["sample_cloud"]["rel_path"])["points"][::8]
fig, axs = plt.subplots(len(runs), 3, figsize=(15, 5 * len(runs)))
axs = np.atleast_2d(axs)
for r, ed in enumerate(runs):
    ed = Path(ed); res = json.loads((ed / "result.json").read_text())
    T = res["inspection_validity"]["registration"]["raw_to_frame"]
    if T is None or not (ed / "registered_scan.ply").exists():
        continue
    P = read_ply_vertices(ed / "registered_scan.ply").vertices[::10]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = axs[r, k]; ax.scatter(cad[:, i], cad[:, j], s=0.1, c="0.6"); ax.scatter(P[:, i], P[:, j], s=0.1, c="r", alpha=0.4)
        ax.set_aspect("equal"); ax.set_title(f"{ed.parent.name} {res['inspection_validity']['state']} axes {i}{j}", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=55)
