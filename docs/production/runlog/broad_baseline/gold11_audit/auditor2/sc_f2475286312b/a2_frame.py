"""Plate-based part frame for sc_f2475286312b. Writes a2_frame.json (origin, axes)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
from a2_common import OUT, load_ply, load_gold, unit, robust_plane, pca_frame

pts, nrm = load_ply()
c0, vt = pca_frame(pts)
e3 = vt[2]
# plate candidates: normals within 10 deg of e3, then iterate
m = np.abs(nrm @ e3) > np.cos(np.radians(12))
c, n, rms = robust_plane(pts[m])
for _ in range(3):
    d = (pts - c) @ n
    m = (np.abs(nrm @ n) > np.cos(np.radians(5))) & (np.abs(d) < 3.0)
    c, n, rms = robust_plane(pts[m])
G = load_gold()
a, b = G['M01']
u = unit((b - a) - ((b - a) @ n) * n)
v = np.cross(n, u)
# orient n so that the tall wall (M07/M08 side) is at +n
a8, b8 = G['M08']
if ((a8 + b8) / 2 - c) @ n < 0:
    n = -n
    v = np.cross(n, u)
R = np.stack([u, v, n])
q = (pts - c) @ R.T
print('plate rms %.3f  n_plate_pts %d' % (rms, m.sum()))
print('n', n.round(5), 'u', u.round(5), 'v', v.round(5))
print('extent', q.min(0).round(1), q.max(0).round(1))
# plate two skins: histogram of w for points with normal ~n
w = q[:, 2]
mm = np.abs(nrm @ n) > np.cos(np.radians(5))
h, e = np.histogram(w[mm & (np.abs(w) < 4)], bins=80, range=(-4, 4))
pk = np.argsort(h)[-6:]
print('w histogram peaks', [(round((e[i] + e[i + 1]) / 2, 2), int(h[i])) for i in sorted(pk)])
# shift origin to the mid-plane between the two skins
sk = w[mm & (np.abs(w) < 4)]
lo = np.median(sk[sk < np.median(sk) - 0.0]) if len(sk) else 0
json.dump({'origin': c.tolist(), 'R_rows_u_v_n': R.tolist(), 'plate_rms_mm': rms,
           'note': 'n points to the side the M07/M08 wall rises to; u along gold M01 projected in plate'},
          open(OUT / 'a2_frame.json', 'w'), indent=1)
for k, (a, b) in G.items():
    qa, qb = (a - c) @ R.T, (b - c) @ R.T
    d = unit(b - a) @ R.T
    print(k, 'start', qa.round(2), 'end', qb.round(2), 'dir', d.round(4))
