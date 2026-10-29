"""Exact mesh cross-sections in ONE common frame (perpendicular to the common bend axis),
at several stations along the axis; gold lines plotted where they cross each plane."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

m = load_mesh()
gold = load_gold()
nA = unit([-0.6973, 0.7003, 0.1527])
nW = unit([-0.6633, 0.3026, 0.6844])
nF = unit([-0.0459, -0.5122, 0.8576])
d1 = unit(np.cross(nA, nW)); d3 = unit(np.cross(nA, nF))
d = unit(d1 + d3)
print('axis bend1/2', d1, 'axis bend3', d3, 'angle between', math.degrees(math.acos(min(1, abs(d1 @ d3)))))
for k, (a, b) in gold.items():
    g = unit(b - a)
    print(k, 'gold dir angle to axis (deg)', round(math.degrees(math.acos(min(1, abs(g @ d)))), 2), 'len', round(np.linalg.norm(b - a), 2))
# in-plane basis: u along plate A (in-plane, perp to d), w = plate A normal
w = unit(nA - d * (nA @ d))
u = unit(np.cross(w, d))
V = m.vertices
c0 = V.mean(0)
s_all = (V - c0) @ d
print('extent along axis', s_all.min(), s_all.max())
# gold endpoints along the axis
for k, (a, b) in gold.items():
    print(k, 'axis span', round(float((a - c0) @ d), 2), round(float((b - c0) @ d), 2),
          'uv start', np.round([(a - c0) @ u, (a - c0) @ w], 2), 'uv end', np.round([(b - c0) @ u, (b - c0) @ w], 2))
json.dump(dict(d=d.tolist(), u=u.tolist(), w=w.tolist(), c0=c0.tolist()), open(OUT + '/scripts/common_frame.json', 'w'))
