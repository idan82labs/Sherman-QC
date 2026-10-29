"""Segment the mesh into near-planar face regions (region growing on face adjacency)."""
import sys, json, math
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np
from collections import deque
from common import *

m = load_mesh()
fn = m.face_normals
fa = m.face_adjacency
area = m.area_faces
cent = m.triangles_center
nbrs = [[] for _ in range(len(m.faces))]
for a, b in fa:
    nbrs[a].append(b); nbrs[b].append(a)

# region growing: seed-normal similarity (<4 deg to running mean) and plane distance (<0.3 mm)
lab = -np.ones(len(m.faces), int)
order = np.argsort(-area)
regions = []
cos_t = math.cos(math.radians(4.0))
for s in order:
    if lab[s] >= 0:
        continue
    rid = len(regions)
    q = deque([s]); lab[s] = rid; members = [s]
    nsum = fn[s] * area[s]; n0 = unit(nsum); p0 = cent[s]
    while q:
        f = q.popleft()
        for g in nbrs[f]:
            if lab[g] >= 0:
                continue
            if fn[g] @ n0 < cos_t:
                continue
            if abs((cent[g] - p0) @ n0) > 0.35:
                continue
            lab[g] = rid; q.append(g); members.append(g)
            nsum += fn[g] * area[g]
            if len(members) % 50 == 0:
                n0 = unit(nsum)
                p0 = (cent[members] * area[members, None]).sum(0) / area[members].sum()
    regions.append(members)

big = []
for rid, mem in enumerate(regions):
    A = area[mem].sum()
    if A < 15:
        continue
    mem = np.array(mem)
    n = unit((fn[mem] * area[mem, None]).sum(0))
    c = (cent[mem] * area[mem, None]).sum(0) / A
    d = (cent[mem] - c) @ n
    big.append(dict(rid=rid, area=round(float(A), 1), n=n.round(4).tolist(), c=c.round(2).tolist(),
                    rms=round(float(np.sqrt((d ** 2).mean())), 3), nf=len(mem)))
big.sort(key=lambda r: -r['area'])
for r in big[:40]:
    print(r)
np.save(OUT + '/scripts/face_region_labels.npy', lab)
json.dump(big, open(OUT + '/scripts/plates.json', 'w'), indent=1)
