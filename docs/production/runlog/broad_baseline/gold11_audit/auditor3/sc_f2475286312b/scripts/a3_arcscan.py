"""Exhaustive check for formed folds: cluster mesh vertices whose normal is ~45 deg between two
principal directions (the middle of a 90-deg bend arc), plus any vertices whose normal is far from
all six principal directions, and report each cluster's extent in the plate frame."""
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from a3common import load_mesh, OUT

V, N, F = load_mesh()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
X = (V - o) @ R.T
Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
NL = Nn @ R.T  # normals in (a,b,h)
absn = np.abs(NL)
mx = absn.max(1)
# arc-ish: max principal component < cos(20deg) -> normal at least 20 deg away from every principal axis
arc = mx < np.cos(np.radians(20))
print("arc-ish vertices", int(arc.sum()), "of", len(V))
P = X[arc]
tree = cKDTree(P)
pairs = tree.query_pairs(1.5, output_type="ndarray")
A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
nc, lab = connected_components(A, directed=False)
sizes = np.bincount(lab)
order = np.argsort(sizes)[::-1]
rows = []
for c in order:
    if sizes[c] < 25:
        break
    Q = P[lab == c]
    NQ = NL[arc][lab == c]
    ext = Q.max(0) - Q.min(0)
    # dominant axis of the cluster = fold axis
    ax = "abh"[int(np.argmax(ext))]
    mean_n = NQ.mean(0)
    rows.append({"size": int(sizes[c]), "axis": ax, "min": Q.min(0).round(1).tolist(), "max": Q.max(0).round(1).tolist(),
                 "extent": ext.round(1).tolist(), "mean_normal_abh": mean_n.round(2).tolist()})
for r in rows:
    print(r)
(OUT / "a3_arcscan.json").write_text(json.dumps(rows, indent=1))
