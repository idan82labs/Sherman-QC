"""Close-up of the wall top edge near v~0 (small feature) and M08 gold (auditor 2)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_common import OUT, load_gold

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
q = np.load(OUT / 'q_cache.npy').astype(float)
G = {k: ((a - c) @ R.T, (b - c) @ R.T) for k, (a, b) in load_gold().items()}
a8, b8 = G['M08']
m = (q[:, 0] > 118) & (q[:, 0] < 140) & (q[:, 2] > 48) & (np.abs(q[:, 1]) < 30)
fig, axs = plt.subplots(1, 3, figsize=(20, 7))
# v-w view (looking along u) of wall top
ax = axs[0]
ax.scatter(q[m, 1], q[m, 2], s=0.5, c=q[m, 0], cmap='viridis')
ax.plot([a8[1], b8[1]], [a8[2], b8[2]], '-', color='#1a9c3a', lw=2, label='gold M08')
ax.set_xlim(-30, 30); ax.set_ylim(48, 68); ax.set_aspect('equal'); ax.grid(alpha=.3)
ax.set_xlabel('v'); ax.set_ylabel('w'); ax.set_title('wall top, looking along u (colour=u)'); ax.legend()
# u-w sections at v=-3, v=-1, v=+2
for ax, v0 in zip(axs[1:], (-2.0, 10.0)):
    s = m & (np.abs(q[:, 1] - v0) < 0.5)
    ax.plot(q[s, 0], q[s, 2], '.', ms=2, color='k')
    t = (v0 - a8[1]) / (b8[1] - a8[1]); p = a8 + t * (b8 - a8)
    ax.plot(p[0], p[2], '*', ms=14, color='#1a9c3a', label='gold M08')
    ax.add_patch(plt.Circle((p[0], p[2]), 4, fill=False, ls=':', color='#1a9c3a'))
    ax.set_xlim(115, 140); ax.set_ylim(45, 68); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_xlabel('u'); ax.set_ylabel('w'); ax.set_title(f'u-w section v={v0} (+-0.5)'); ax.legend()
fig.tight_layout(); fig.savefig(OUT / 'a2_walltop_M08.png', dpi=100)
# stats of the feature
s = m & (q[:, 2] > 61.8)
print('points above w=61.8 near wall top:', s.sum(), 'v-range', q[s,1].min().round(2), q[s,1].max().round(2), 'u-range', q[s,0].min().round(2), q[s,0].max().round(2), 'w max', q[s,2].max().round(2))
