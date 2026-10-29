import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.inspect.scan_io import read_scan
S = read_scan(sys.argv[1]).points[::12]
cdir = Path(sys.argv[2]); cp = json.loads((cdir / "compiled_part.json").read_text())
C = np.load(cdir / cp["stages"][-1]["sample_cloud"]["rel_path"])["points"][::6]
z = np.load(sys.argv[3]); q = C @ z["R"].T + z["t"]
d, _ = cKDTree(S).query(q); ds, _ = cKDTree(q).query(S)
m = S.mean(0); u, sv, vt = np.linalg.svd(S - m, full_matrices=False)
Sp, Qp = (S - m) @ vt.T, (q - m) @ vt.T
fig, axs = plt.subplots(2, 3, figsize=(18, 11))
for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
    ax = axs[0, k]; sc = ax.scatter(Sp[:, i], Sp[:, j], s=0.05, c=np.minimum(ds, 10), cmap="jet"); ax.set_aspect("equal"); ax.set_title(f"scan coloured by dist to CAD {i}{j}")
    ax = axs[1, k]; ax.scatter(Qp[:, i], Qp[:, j], s=0.05, c=np.minimum(d, 10), cmap="jet"); ax.set_aspect("equal"); ax.set_title(f"CAD coloured by dist to scan {i}{j}")
plt.tight_layout(); plt.savefig(sys.argv[4], dpi=60)
