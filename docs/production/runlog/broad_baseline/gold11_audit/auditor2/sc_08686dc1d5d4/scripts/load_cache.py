import sys, json, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/research/broad_baseline")
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
import bb_common as bb
from pathlib import Path
from scipy.spatial import cKDTree
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
scan = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_1.ply")
pts = bb.load_scan_vertices(scan)
np.save(W / "pts.npy", pts)
print("n", len(pts), "bbox", pts.min(0).round(1), pts.max(0).round(1))
gold = json.load(open("/Users/idantal/ShermanQC/FullProject/output/bend_label_02_10839000/bend_instance_labels.json"))
tree = cKDTree(pts)
for b in gold["manual_bends"]:
    for k in ("line_start", "line_end"):
        p = np.array(b[k]); d, i = tree.query(p)
        print(b["bend_id"], k, np.round(p, 3), "nearest d=%.6f" % d, "idx", i)
c = pts.mean(0)
u, s, vt = np.linalg.svd(pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)] - c, full_matrices=False)
print("centroid", c.round(2), "PCA sv", s.round(1))
print("e1", vt[0].round(4), "e2", vt[1].round(4), "e3", vt[2].round(4))
np.save(W / "pca.npy", np.vstack([c, vt]))
