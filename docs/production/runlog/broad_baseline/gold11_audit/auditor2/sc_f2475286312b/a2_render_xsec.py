"""Auditor-2 perpendicular cross-sections for every gold line of sc_f2475286312b.

Each panel: raw scan points in a +-0.5 mm slab (plate frame u,v,w; see a2_frame.json),
gold intersection (green star, dotted 4 mm matcher circle), and where the fold exists:
outer apex (cyan diamond), outer virtual sharp X (blue x), inner virtual sharp Xi
(purple x), arc centre C (brown +) from a2_folds.json.  Read-only on inputs.
"""
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
folds = json.load(open(OUT / 'a2_folds.json'))
AX = {'u': 0, 'v': 1, 'w': 2}


def fold_row(fname, s):
    rows = folds[fname]['rows']
    best = min(rows, key=lambda r: abs(r['s'] - s))
    return best if abs(best['s'] - s) < 1.01 else None


def panel(ax, axis, s, ij, gids, fname, centre, half=14, title=''):
    a = AX[axis]; i, j = ij
    m = (np.abs(q[:, a] - s) < 0.5) & (np.abs(q[:, i] - centre[0]) < half * 1.6) & (np.abs(q[:, j] - centre[1]) < half * 1.6)
    ax.plot(q[m, i], q[m, j], '.', ms=1.3, color='0.25', zorder=1)
    r = fold_row(fname, s) if fname else None
    txt = []
    if r and r.get('exists'):
        X = r['X']; ax.plot(*X, 'x', color='#1f77b4', ms=9, mew=2, zorder=4, label='outer virtual sharp')
        if r.get('Xi'): ax.plot(*r['Xi'], 'x', color='#9467bd', ms=8, mew=2, zorder=4, label='inner virtual sharp')
        if r.get('C'): ax.plot(*r['C'], '+', color='#8c564b', ms=11, mew=2, zorder=4, label='arc centre')
        if r.get('apex'): ax.plot(*r['apex'], 'D', color='#17becf', mec='k', ms=7, zorder=5, label='outer apex')
        txt.append(f"fold: bend {180 - r['phi']:.1f} deg, Ro {r.get('R_outer') or float('nan'):.2f}")
    elif fname:
        txt.append('fold absent at this station')
    for g in gids:
        A, B = G[g]
        t = (s - A[a]) / (B[a] - A[a])
        if not (0 <= t <= 1):
            txt.append(f'{g}: outside gold span'); continue
        p = A + t * (B - A)
        ax.plot(p[i], p[j], '*', color='#1a9c3a', mec='k', ms=15, zorder=6)
        ax.add_patch(plt.Circle((p[i], p[j]), 4.0, fill=False, ls=':', color='#1a9c3a', lw=1.3))
        ax.annotate(g, (p[i], p[j]), xytext=(5, 5), textcoords='offset points', color='#0b5d1e', fontsize=9, fontweight='bold')
        if r and r.get('exists') and r.get(g):
            gg = r[g]
            txt.append(f"{g}: {gg['dist']:.2f} mm from apex (along bisector {gg['along']:+.2f}, across {gg['across']:+.2f}); to X {gg['d_X']:.2f}, Xi {gg['d_Xi']:.2f}, C {gg['d_C']:.2f}")
        else:
            d = np.min(np.hypot(q[m, i] - p[i], q[m, j] - p[j])) if m.any() else float('nan')
            txt.append(f'{g}: nearest scan pt in slab {d:.2f} mm')
    ax.set_xlim(centre[0] - half, centre[0] + half); ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    lab = 'uvw'
    ax.set_xlabel(lab[i] + ' (mm)'); ax.set_ylabel(lab[j] + ' (mm)')
    ax.set_title(title + f'  {axis}={s:g}', fontsize=9)
    ax.text(0.01, 0.01, '\n'.join(txt), transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(fc='white', ec='0.7', alpha=0.85))


def fig_for(name, specs, ncol=3):
    n = len(specs); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 6.0 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, sp in zip(axs, specs):
        panel(ax, **sp)
    for ax in axs[n:]:
        ax.axis('off')
    h, l = axs[0].get_legend_handles_labels()
    if h:
        uniq = dict(zip(l, h))
        fig.legend(uniq.values(), uniq.keys(), loc='upper right', fontsize=8)
    fig.suptitle(f'sc_f2475286312b auditor2 cross-sections: {name}  (plate frame; slab +-0.5 mm; dotted = 4 mm matcher radius)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f'a2_xsec_{name}.png', dpi=90)
    plt.close(fig)


# M01 / M02 lips: stations along u incl. beyond the lip end (u=83)
for g, fname, cv in (('M01', 'F_M01_lip', -110.0), ('M02', 'F_M02_lip', 110.0)):
    specs = [dict(axis='u', s=s, ij=(1, 2), gids=[g], fname=fname, centre=(cv, -2.0), half=9,
                  title=f'{g} vs lip fold') for s in (-120, -60, 0, 60, 80)]
    specs.append(dict(axis='u', s=110, ij=(1, 2), gids=[g], fname=fname, centre=(cv, -2.0), half=9,
                      title=f'{g} overrun (no lip)'))
    fig_for(g, specs)
# M07 wall bend
specs = [dict(axis='v', s=s, ij=(0, 2), gids=['M07'], fname='F_wall_M07', centre=(124.0, 2.0), half=9,
              title='M07 vs wall fold') for s in (-100, -50, 0, 50, 100, 108)]
fig_for('M07', specs)
# M03/M04 bottom tab (v~-81) and M05/M06 top tab (v~+92): tab region and flat region
for pair, fname, cv in ((('M03', 'M04'), 'F_tab_bottom', -80.0), (('M05', 'M06'), 'F_tab_top', 86.0)):
    specs = [dict(axis='u', s=s, ij=(1, 2), gids=list(pair), fname=fname, centre=(cv, 3.0), half=15,
                  title=f'{pair[0]}/{pair[1]}') for s in (-110, -40, -10, 20, 60, 110)]
    fig_for('_'.join(pair), specs)
# M08 wall top
specs = [dict(axis='v', s=s, ij=(0, 2), gids=['M08'], fname=None, centre=(128.0, 57.0), half=9,
              title='M08 vs wall top edge') for s in (-100, -50, -2, 10, 50, 100)]
fig_for('M08', specs)
print('ok')
