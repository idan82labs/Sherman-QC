import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
T = json.load(open(OUT + '/a1_apex_track.json')); EXT = json.load(open(OUT + '/a1_fold_extents.json'))
GOLD_FOR = {'L_FLANGE': ['M02', 'M03'], 'R_FLANGE': ['M05', 'M06'], 'L_TAB': ['M01'], 'R_TAB': ['M04']}
cols = {'M01': 'tab:green', 'M02': 'tab:green', 'M03': 'tab:red', 'M04': 'tab:green', 'M05': 'tab:green', 'M06': 'tab:red'}
fig, axes = plt.subplots(4, 2, figsize=(18, 16))
for i, (fold, rows) in enumerate(T['folds'].items()):
    for j, comp in enumerate(['d_apex', 'along_b']):
        ax = axes[i, j]
        for a, b in EXT[fold]:
            ax.axvspan(a, b, color='0.85', zorder=0)
        for k in GOLD_FOR[fold]:
            xs = [r['x0'] for r in rows if r.get('bend') and r['gold'][k]['in_span']]
            ys = [r['gold'][k][comp] for r in rows if r.get('bend') and r['gold'][k]['in_span']]
            ax.plot(xs, ys, '.-', ms=3, color=cols[k], label=f'{k} {comp}')
            # gold span markers
            xsp = [r['x0'] for r in rows if r['gold'][k]['in_span']]
            ax.plot([min(xsp), max(xsp)], [-5.5 if k in ('M03', 'M06') else -4.5] * 2, '|-', color=cols[k], lw=3)
        if comp == 'd_apex':
            ax.axhline(4, color='r', ls=':', lw=1); ax.set_ylim(-6, 17)
        else:
            ax.axhline(0, color='k', lw=0.8)
            xo = [r['x0'] for r in rows if r.get('bend')]; ao = [r['apex_to_Xo'] for r in rows if r.get('bend')]
            ax.plot(xo, ao, '-', color='tab:blue', lw=1, label='outer virtual sharp (along b)')
            ax.set_ylim(-12, 4)
        ax.set_title(f'{fold}: gold {comp} vs PC1 (grey = fold present; bars = gold span)', fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right'); ax.set_xlim(-170, 165)
axes[-1, 0].set_xlabel('PC1 (mm)'); axes[-1, 1].set_xlabel('PC1 (mm)')
plt.tight_layout(); plt.savefig(OUT + '/a1_gold_vs_apex_along_length.png', dpi=70)
# apex position drift for right/left flange (PC2, PC3 of apex) to check segment A offset
fig, axes = plt.subplots(2, 2, figsize=(16, 6))
for i, fold in enumerate(['L_FLANGE', 'R_FLANGE']):
    rows = [r for r in T['folds'][fold] if r.get('bend')]
    x = np.array([r['x0'] for r in rows]); a = np.array([r['apex'] for r in rows]); xo = np.array([r['Xo'] for r in rows])
    ang = np.array([r['fold_angle_deg'] for r in rows])
    axes[0, i].plot(x, a[:, 0], '.', ms=3, label='apex PC2'); axes[0, i].plot(x, xo[:, 0], '.', ms=2, label='outer sharp PC2')
    for k in GOLD_FOR[fold]:
        g = np.array([r['gold'][k]['g'] for r in rows]); axes[0, i].plot(x, g[:, 0], '-', lw=1, label=f'gold {k} PC2')
    axes[0, i].set_title(f'{fold}: PC2 of apex / gold'); axes[0, i].legend(fontsize=7); axes[0, i].grid(alpha=.3)
    axes[1, i].plot(x, a[:, 1], '.', ms=3, label='apex PC3'); 
    for k in GOLD_FOR[fold][:1]:
        g = np.array([r['gold'][k]['g'] for r in rows]); axes[1, i].plot(x, g[:, 1], '-', lw=1, label=f'gold {k} PC3')
    ax2 = axes[1, i].twinx(); ax2.plot(x, 180 - ang, 'r.', ms=2); ax2.set_ylabel('bend angle (deg, red)')
    axes[1, i].set_title(f'{fold}: PC3 of apex / gold; red = bend angle'); axes[1, i].legend(fontsize=7); axes[1, i].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/a1_flange_apex_drift.png', dpi=70)
print('ok')
