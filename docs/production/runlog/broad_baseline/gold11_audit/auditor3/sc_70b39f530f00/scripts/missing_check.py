"""Missing-bend check: every vertex should lie on one of the known plates' slabs or on the three fold zones.
Vertices outside (a) the arm/web/A'/frame slabs (mid-plane +- t/2 + 0.35 mm) and (b) the three fold tubes
are clustered and reported; a formed feature (extra flange, tab, extruded hole, emboss) would show up here."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
from scipy.spatial import cKDTree
from common import *

m = load_mesh()
V = m.vertices
PL = json.load(open(OUT + '/scripts/plate_planes.json'))
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
MC = json.load(open(OUT + '/matcher_check.json'))
pairs_skin = {'A': ('A_r1', 'A_r2'), 'W': ('W_r6', 'W_r4'), 'Ap': ('Ap_r0', 'Ap_r5'), 'F': ('F_r241', 'F_r82')}
t = 2.815
on_plate = np.zeros(len(V), bool)
for k, (p, q) in pairs_skin.items():
    n1, c1 = np.array(PL[p]['n']), np.array(PL[p]['c'])
    n2, c2 = np.array(PL[q]['n']), np.array(PL[q]['c'])
    # mid-plane between the two skins
    n = n1 if n1 @ n2 < 0 else -n1
    cm = (c1 + c2) / 2
    dist = np.abs((V - cm) @ n1)
    on_plate |= dist <= t / 2 + 0.35
# fold tubes: within (R_o + 0.5) of each fold's arc centre line ~ approximate by distance to outer apex line <= 9 mm
in_fold = np.zeros(len(V), bool)
for g in ('M01', 'M02', 'M03'):
    a, b = (np.array(x) for x in MC['aud3_lines'][f'{g}_outer'])
    v = unit(b - a)
    rel = V - a
    tt = rel @ v
    perp = np.linalg.norm(rel - np.outer(tt, v), axis=1)
    L = np.linalg.norm(b - a)
    in_fold |= (perp <= 8.0) & (tt >= -3) & (tt <= L + 3)
rest = ~(on_plate | in_fold)
print('vertices', len(V), 'on plate slabs', on_plate.sum(), 'in fold tubes only', (in_fold & ~on_plate).sum(), 'unexplained', rest.sum())
if rest.sum():
    P = V[rest]
    tree = cKDTree(P)
    # simple connected components with 1.0 mm linkage
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    parent = np.arange(len(P))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(len(P))])
    labs, counts = np.unique(roots, return_counts=True)
    order = np.argsort(-counts)
    for k in order[:15]:
        sel = roots == labs[k]
        Q = P[sel]
        cq = Q.mean(0)
        ext = Q.max(0) - Q.min(0)
        print(f'cluster n={counts[k]:5d} centre(s,u,w)=({(cq-c0)@d:7.1f},{(cq-c0)@u:7.1f},{(cq-c0)@w:7.1f}) bbox={np.round(ext,1)}')
