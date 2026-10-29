import numpy as np, trimesh, json, sys
p = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply"
m = trimesh.load(p, process=False)
V = np.asarray(m.vertices, dtype=np.float64)
F = np.asarray(m.faces, dtype=np.int64)
N = None
try:
    N = np.asarray(m.vertex_normals, dtype=np.float64)
except Exception as e:
    print("no normals", e)
out = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/"
np.save(out+"V.npy", V); np.save(out+"F.npy", F)
if N is not None: np.save(out+"N.npy", N)
print(V.shape, F.shape, V.min(0), V.max(0))
# PCA
c = V.mean(0)
U,S,Wt = np.linalg.svd(V-c, full_matrices=False)
print("centroid", c, "sing", S/np.sqrt(len(V)))
print("axes", Wt)
