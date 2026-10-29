"""Fine (0.25 mm) scan of each fold's ends: where does any bent material (outer or inner arc with
normals in the middle of the plate-to-plate range) still exist near the apex?"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import trimesh
from common import *

m = load_mesh(); gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
PL = json.load(open(OUT + '/scripts/plate_planes.json'))
PHI = {k: v['phi_deg'] for k, v in PL.items()}
fn = m.face_normals
BENDS = {'bend1_A_W': ('A_r1', 'W_r6', 'A_r2', 'W_r4', (7.56, 0.25), (8.48, -2.32)),
         'bend2_W_Ap': ('W_r4', 'Ap_r5', 'W_r6', 'Ap_r0', (-6.66, -13.45), (-7.6, -10.93)),
         'bend3_Ap_F': ('Ap_r5', 'F_r241', 'Ap_r0', 'F_r82', (-19.75, -12.62), (-18.03, -10.5))}
ranges = {'bend1_A_W': [(-24.0, -20.0), (-8.0, -3.0)], 'bend2_W_Ap': [(-32.5, -28.0), (-11.0, -9.0)],
          'bend3_Ap_F': [(-38.5, -33.5), (-11.0, -9.0)]}


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

for b, (o1, o2, i1, i2, apo, api) in BENDS.items():
    print('=====', b)
    for lo, hi in ranges[b]:
        for s in np.arange(lo, hi + 1e-6, 0.25):
            o = c0 + d * s
            segs, fidx = trimesh.intersections.mesh_plane(m, plane_normal=d, plane_origin=o, return_faces=True)
            if len(segs) == 0:
                print(f's={s:6.2f} empty'); continue
            P0 = np.c_[(segs[:, 0] - o) @ u, (segs[:, 0] - o) @ w]
            P1 = np.c_[(segs[:, 1] - o) @ u, (segs[:, 1] - o) @ w]
            mid = (P0 + P1) / 2; L = np.linalg.norm(P1 - P0, axis=1)
            N = fn[fidx]; phi = np.degrees(np.arctan2(N @ w, N @ u))
            out = []
            for (p1, p2, ap) in ((o1, o2, apo), (i1, i2, api)):
                a1, a2 = PHI[p1], PHI[p2]
                span = angdiff(a2, a1)
                f = angdiff(phi, a1) / span
                near = np.linalg.norm(mid - np.array(ap), axis=1) < 3.5
                arc = near & (f > 0.2) & (f < 0.8)
                out.append((arc.sum(), L[arc].sum()))
            gl = {}
            for gid, (ga, gb) in gold.items():
                ta, tb = (ga - c0) @ d, (gb - c0) @ d
                gl[gid] = 0 <= (s - ta) / (tb - ta) <= 1
            print(f's={s:6.2f} outer mid-arc segs={out[0][0]:3d} len={out[0][1]:5.2f} | inner mid-arc segs={out[1][0]:3d} len={out[1][1]:5.2f} | gold-in-span ' + ' '.join(k for k, v in gl.items() if v))
