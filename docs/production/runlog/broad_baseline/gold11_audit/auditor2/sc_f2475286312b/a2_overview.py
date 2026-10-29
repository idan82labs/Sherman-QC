"""Top-view height map + v-w sections in the plate frame, gold overlaid (auditor 2)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_common import OUT, load_ply, load_gold

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
pts, nrm = load_ply()
q = (pts - c) @ R.T
nq = nrm @ R.T
G = {k: ((a - c) @ R.T, (b - c) @ R.T) for k, (a, b) in load_gold().items()}
rng = np.random.default_rng(0)
sel = rng.choice(len(q), 250000, replace=False)
fig, ax = plt.subplots(figsize=(15, 12))
sc = ax.scatter(q[sel, 0], q[sel, 1], c=np.clip(q[sel, 2], -14, 20), s=0.3, cmap='coolwarm', vmin=-14, vmax=20)
plt.colorbar(sc, ax=ax, label='w = height above plate mid-plane (mm), +w = wall side')
for k, (a, b) in G.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#1a9c3a', lw=1.6)
    t = 0.12 if k in ('M03', 'M05') else (0.88 if k in ('M04', 'M06') else 0.5)
    p = a + t * (b - a)
    ax.text(p[0], p[1] + 2, f"{k} (w={p[2]:.1f})", color='#0b5d1e', fontsize=9, fontweight='bold')
ax.set_aspect('equal'); ax.grid(alpha=.3)
ax.set_xlabel('u (mm, along M01/M02)'); ax.set_ylabel('v (mm)')
ax.set_title('sc_f2475286312b auditor2 plate-frame top view (colour = w); green = gold M01-M08')
fig.tight_layout(); fig.savefig(OUT / 'a2_topview_height.png', dpi=110)
plt.close(fig)

# v-w sections at several u stations
stations = [-120, -60, -20, 0, 20, 40, 60, 100]
fig, axs = plt.subplots(len(stations), 1, figsize=(16, 3.0 * len(stations)))
for ax, u0 in zip(axs, stations):
    m = np.abs(q[:, 0] - u0) < 0.6
    ax.plot(q[m, 1], q[m, 2], '.', ms=1.2, color='k')
    for k, (a, b) in G.items():
        if k in ('M07', 'M08'):
            continue
        t = (u0 - a[0]) / (b[0] - a[0])
        p = a + t * (b - a)
        inside = 0 <= t <= 1
        ax.plot(p[1], p[2], 'o', color='#1a9c3a' if inside else '#9fd8ad', ms=6)
        ax.text(p[1] + 1, p[2] + 1, k, color='#0b5d1e', fontsize=8)
    ax.set_title(f'section u = {u0} mm (slab +-0.6 mm)', fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_ylim(-18, 22); ax.set_xlim(-118, 118)
fig.tight_layout(); fig.savefig(OUT / 'a2_vw_sections.png', dpi=100)
plt.close(fig)

# u-w sections at several v stations (wall side)
stations = [-100, -50, 0, 50, 100]
fig, axs = plt.subplots(len(stations), 1, figsize=(16, 3.4 * len(stations)))
for ax, v0 in zip(axs, stations):
    m = np.abs(q[:, 1] - v0) < 0.6
    ax.plot(q[m, 0], q[m, 2], '.', ms=1.2, color='k')
    for k, (a, b) in G.items():
        if k not in ('M07', 'M08'):
            continue
        t = (v0 - a[1]) / (b[1] - a[1])
        p = a + t * (b - a)
        ax.plot(p[0], p[2], 'o', color='#1a9c3a', ms=6)
        ax.text(p[0] + 1, p[2] + 1, k, color='#0b5d1e', fontsize=8)
    ax.set_title(f'section v = {v0} mm (slab +-0.6 mm)', fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / 'a2_uw_sections.png', dpi=100)
print('done')
