"""Perpendicular close-up sections for every gold line at t = 0.1 .. 0.9 (slab +-0.4 mm normal to the gold
direction), +-10 mm window around the gold point, with the auditor's apex / outer & inner virtual sharp
(from the dense PC1 track, nearest station) and the 4 mm matcher circle."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V, F, N = load_mesh()
c, R = pca_frame(V)
T = json.load(open(OUT + '/a1_apex_track.json'))
G = load_gold()
FOLD_OF = {'M01': 'L_TAB', 'M02': 'L_FLANGE', 'M03': 'L_FLANGE', 'M04': 'R_TAB', 'M05': 'R_FLANGE', 'M06': 'R_FLANGE'}
ts = [0.1, 0.3, 0.5, 0.7, 0.9]
fig, axes = plt.subplots(6, 5, figsize=(22, 27))
for i, k in enumerate(['M01', 'M02', 'M03', 'M04', 'M05', 'M06']):
    s, e = G[k]; d = unit(e - s)
    # in-plane basis: PC2 and PC3 axes orthogonalised against d (so the plot reads like the PC2-PC3 profile)
    u = unit(R[1] - (R[1] @ d) * d); w = np.cross(d, u)
    if w @ R[2] < 0: w = -w
    rows = {r['x0']: r for r in T['folds'][FOLD_OF[k]]}
    for j, t in enumerate(ts):
        ax = axes[i, j]
        g = s + t * (e - s)
        rel = V - g
        m = (np.abs(rel @ d) < 0.4) & (np.abs(rel @ u) < 14) & (np.abs(rel @ w) < 14)
        Q = np.c_[rel[m] @ u, rel[m] @ w]
        ax.scatter(Q[:, 0], Q[:, 1], s=1.0, c='k')
        ax.plot(0, 0, marker='*', ms=14, color='limegreen', mec='k', zorder=5)
        ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=':', color='g'))
        # auditor apex from nearest PC1 station (convert PC2/PC3 coords to this basis)
        x_pc1 = (g - c) @ R[0]
        key = min(rows, key=lambda x: abs(x - x_pc1))
        r = rows[key]
        txt = f'{k} t={t:.1f} PC1={x_pc1:.0f}'
        if r.get('bend') and abs(key - x_pc1) < 1.5:
            for name, mk, col in [('apex', 'D', 'deepskyblue'), ('Xo', 'x', 'tab:blue'), ('Xi', 'x', 'tab:purple')]:
                p2 = np.array(r[name]); p3 = c + np.array([key, p2[0], p2[1]]) @ R
                q = p3 - g
                ax.plot(q @ u, q @ w, marker=mk, ms=8, color=col, mec='k' if mk == 'D' else None, zorder=6)
            ap = c + np.array([key, *r['apex']]) @ R - g
            ap2 = np.array([ap @ u, ap @ w])
            txt += f'\nd(gold, outer apex)={np.linalg.norm(ap2):.2f} mm'
        else:
            dmin = np.linalg.norm(Q, axis=1).min() if len(Q) else np.nan
            txt += f'\nNO FOLD at this station; gold->scan {dmin:.2f} mm'
        ax.set_title(txt, fontsize=9)
        ax.set_xlim(-14, 14); ax.set_ylim(-14, 14); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_xlabel('~PC2 (mm)', fontsize=7); ax.set_ylabel('~PC3 (mm)', fontsize=7)
fig.suptitle('sc_07062eee17aa auditor-1: perpendicular sections at each gold line (green star = gold, dotted = 4 mm matcher circle,\n'
             'cyan diamond = measured outer apex, blue x = outer virtual sharp, purple x = inner virtual sharp)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(OUT + '/a1_closeups_all_gold.png', dpi=62)
print('ok')
