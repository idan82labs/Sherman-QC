"""Auditor 2: perpendicular close-up sections per gold line (5 stations) + along-length offset plot."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
exec(open('a2_measure_defs.py').read())
M = json.load(open(OUT + '/a2_station_measurements.json'))
for k, fd in FOLDS.items():
    d = M[k]; st = [r for r in d['stations'] if r.get('fit_ok') and 'd_apex' in r]
    ss = np.array([r['s'] for r in st])
    picks = [st[int(np.argmin(np.abs(np.array([r['gold_t'] for r in st]) - t)))] for t in (0.03, 0.25, 0.5, 0.75, 0.97)]
    fig = plt.figure(figsize=(26, 10.5))
    for i, r in enumerate(picks):
        ax = fig.add_subplot(2, 5, i+1)
        Q, Qn, nax = section(fd, r['s'], half=0.6, win=24)
        inw = np.array([-fd['wall_sign'], 0.0]); inl = np.array([0.0, -fd['leg_sign']])
        VSo = np.array(r['VSo'])
        u = (Q - VSo) @ inw; v = (Q - VSo) @ inl
        outer = (Qn @ -inw > 0.3) | (Qn @ -inl > 0.3)
        ax.scatter(u[outer], v[outer], s=1.2, c='0.25', label='outer-facing skin')
        ax.scatter(u[~outer], v[~outer], s=1.2, c='tab:orange', label='inner-facing skin')
        au, av = r['apex_uv']; R = r['R_outer_est']
        ax.plot(0, 0, 'x', color='tab:blue', ms=10, mew=2, label='outer virtual sharp')
        tw = r['thick_wall'] or 1.45; tl = r['thick_leg'] or 1.45
        ax.plot(tw, tl, 'x', color='tab:purple', ms=10, mew=2, label='inner virtual sharp')
        ax.plot(au, av, 'D', color='cyan', mec='k', ms=8, label='outer apex')
        ax.plot([0, -1.5], [R, R], color='m', lw=2); ax.plot([R, R], [0, -1.5], color='m', lw=2, label='outer tangent lines')
        ax.add_patch(plt.Circle((au, av), 4.0, fill=False, ls=':', color='g'))
        gu, gv = r['gold_u'], r['gold_v']
        ax.plot(gu, gv, '*', color='lime', mec='k', ms=18, label=f'gold {k}')
        lim = max(8.0, gv + 3, gu + 3)
        ax.set_xlim(-3, lim); ax.set_ylim(-3, lim); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_xlabel('u = depth inward from WALL outer plane (mm)'); ax.set_ylabel('v = depth inward from FACE/FLANGE outer plane (mm)')
        ax.set_title(f"{k} t={r['gold_t']:.2f} (axis s={r['s']:.0f})\nd_apex={r['d_apex']:.2f}  gold(u,v)=({gu:.2f},{gv:.2f})  R_o~{R:.2f}", fontsize=10)
        if i == 0: ax.legend(fontsize=7, loc='upper right')
    ax = fig.add_subplot(2, 1, 2)
    ax.plot([r['gold_t'] for r in st], [r['d_apex'] for r in st], 'k-o', ms=3, label='gold -> outer apex (mm)')
    ax.plot([r['gold_t'] for r in st], [r['gold_u'] for r in st], 'b--', label='gold u (depth from wall outer plane)')
    ax.plot([r['gold_t'] for r in st], [r['gold_v'] for r in st], 'r--', label='gold v (depth from face/flange outer plane)')
    ax.plot([r['gold_t'] for r in st], [r['d_scan'] for r in st], 'g:', label='gold -> nearest scan point')
    ax.axhline(4, color='g', ls=':', lw=1); ax.text(0.0, 4.15, '4 mm matcher limit', color='g', fontsize=8)
    ax.set_xlabel('gold line parameter t'); ax.set_ylabel('mm'); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4)
    ax.set_title(f"{k}: {fd['name']} -- along-length placement (stations every 5 mm)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT + f'/a2_xsec_{k}.png', dpi=55); plt.close()
print('ok')
