"""Quick probes of feature extents in the plate frame (auditor 2)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
from a2_common import OUT, load_ply

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
pts, nrm = load_ply()
q = (pts - c) @ R.T
nq = nrm @ R.T
np.save(OUT / 'q_cache.npy', q.astype(np.float32))
np.save(OUT / 'nq_cache.npy', nq.astype(np.float32))

def runs(mask_u, coord, step=1.0, lo=-140, hi=145):
    bins = np.arange(lo, hi + step, step)
    h, _ = np.histogram(coord[mask_u], bins=bins)
    occ = h > 3
    out = []; start = None
    for i, o in enumerate(occ):
        if o and start is None: start = bins[i]
        if not o and start is not None: out.append((round(start, 1), round(bins[i], 1))); start = None
    if start is not None: out.append((round(start, 1), round(bins[-1], 1)))
    return out

# M01 lip: points with v<-110.5 and w<-4 (lip below plate)
m = (q[:, 1] < -110) & (q[:, 2] < -4) & (q[:, 2] > -14)
print('M01 lip (v<-110, -14<w<-4) u-runs:', runs(m, q[:, 0]))
m = (q[:, 1] > 110) & (q[:, 2] < -4) & (q[:, 2] > -14)
print('M02 lip (v>110, -14<w<-4) u-runs:', runs(m, q[:, 0]))
m = (q[:, 0] < -124) & (q[:, 2] < -4) & (q[:, 2] > -14)
print('left lip (u<-124, -14<w<-4) v-runs:', runs(m, q[:, 1]))
m = (q[:, 0] > 124) & (q[:, 0] < 132) & (q[:, 2] > 8) & (q[:, 2] < 55)
print('wall (124<u<132, 8<w<55) v-runs:', runs(m, q[:, 1]))
m = (q[:, 1] < -78) & (q[:, 1] > -84) & (q[:, 2] > 4)
print('bottom tab (-84<v<-78, w>4) u-runs:', runs(m, q[:, 0]))
m = (q[:, 1] > 89) & (q[:, 1] < 95) & (q[:, 2] > 4)
print('top tab (89<v<95, w>4) u-runs:', runs(m, q[:, 0]))
m = (q[:, 1] < -110) & (q[:, 0] > 129) & (q[:, 2] > 4)
print('-v ear (v<-110,u>129,w>4) u-runs:', runs(m, q[:, 0]), 'w-runs:', runs(m, q[:, 2], lo=-15, hi=70))
m = (q[:, 1] > 110) & (q[:, 0] > 129) & (q[:, 2] > 4)
print('+v ear (v>110,u>129,w>4) u-runs:', runs(m, q[:, 0]), 'w-runs:', runs(m, q[:, 2], lo=-15, hi=70))
# wall top: max w per v bin for wall points
m = (q[:, 0] > 118) & (q[:, 0] < 135) & (q[:, 2] > 50)
vv = q[m, 1]; ww = q[m, 2]; uu = q[m, 0]
for v0 in range(-115, 116, 5):
    s = np.abs(vv - v0) < 2.5
    if s.any():
        print(f'  wall-top v={v0:4d}: max w {ww[s].max():6.2f}  u-range [{uu[s].min():.1f},{uu[s].max():.1f}] n={s.sum()}')
