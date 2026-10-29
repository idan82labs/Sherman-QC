"""scan and CAD final stage, each in its own PCA frame, 3 views. usage: views.py SCAN CDIR OUT"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.inspect.scan_io import read_scan
scan = read_scan(sys.argv[1]).points[::15]
cdir = Path(sys.argv[2])
cp = json.loads((cdir / "compiled_part.json").read_text())
cad = np.load(cdir / cp["stages"][-1]["sample_cloud"]["rel_path"])["points"][::4]
def pca(P):
    m = P.mean(0); u, s, vt = np.linalg.svd(P - m, full_matrices=False); return (P - m) @ vt.T
fig, axs = plt.subplots(2, 3, figsize=(18, 11))
for r, (nm, P) in enumerate((("scan", pca(scan)), ("cad", pca(cad)))):
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = axs[r, k]; ax.scatter(P[:, i], P[:, j], s=0.05, c="k"); ax.set_aspect("equal"); ax.set_title(f"{nm} pca {i}{j} n={len(P)}")
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=60)
