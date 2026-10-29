import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
V, F, N = load_mesh()
print('V', V.shape, 'F', F.shape, 'normals', None if N is None else N.shape)
c, R = pca_frame(V)
P = (V - c) @ R.T
print('centroid', c.round(3)); print('PC axes', R.round(4))
print('extent in PC frame', P.min(0).round(2), P.max(0).round(2))
G = load_gold()
for k, (s, e) in G.items():
    ps = (s - c) @ R.T; pe = (e - c) @ R.T
    print(k, 'start', ps.round(2), 'end', pe.round(2), 'dir', unit(pe - ps).round(4))
# normal clusters
if N is not None:
    Np = N @ R.T
    # quantize directions
    from collections import Counter
    q = np.round(Np * 4) / 4
    cnt = Counter(map(tuple, q))
    for k, v in cnt.most_common(14):
        print('normal~', k, v)
