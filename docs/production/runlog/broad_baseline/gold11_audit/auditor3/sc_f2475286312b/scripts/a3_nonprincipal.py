"""Look for planar patches whose normal is NOT along a principal plate-frame axis (would reveal
tilted/partial bends such as 15-deg tabs). Planarity from 2.5 mm neighbourhood PCA."""
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
rng = np.random.default_rng(1)
idx = rng.choice(len(X), 150000, replace=False)
tree = cKDTree(X)
S = X[idx]
nb = tree.query_ball_point(S, 2.5)
keep = []
nrm = []
for i, ids in enumerate(nb):
    if len(ids) < 12:
        continue
    Q = X[ids] - X[ids].mean(0)
    w, v = np.linalg.eigh(Q.T @ Q / len(ids))
    if w[0] / max(w[1], 1e-12) > 0.02:  # not planar (or two skins mixed)
        continue
    nv = v[:, 0]
    ang = np.degrees(np.arccos(np.clip(np.abs(nv).max(), 0, 1)))
    if ang > 6:
        keep.append(i)
        nrm.append(nv * np.sign(nv[np.argmax(np.abs(nv))]))
keep = np.array(keep)
print("planar non-principal samples:", len(keep))
if len(keep):
    P = S[keep]
    t2 = cKDTree(P)
    pairs = t2.query_pairs(3.0, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    nc, lab = connected_components(A, directed=False)
    sz = np.bincount(lab)
    for c in np.argsort(sz)[::-1][:15]:
        if sz[c] < 15:
            break
        Q = P[lab == c]
        NN = np.array(nrm)[lab == c]
        mn = NN.mean(0)
        mn /= np.linalg.norm(mn)
        tilt = np.degrees(np.arccos(np.clip(np.abs(mn).max(), 0, 1)))
        print(int(sz[c]), "min", Q.min(0).round(1), "max", Q.max(0).round(1), "mean normal(a,b,h)", mn.round(3), "tilt from nearest axis %.1f deg" % tilt)
