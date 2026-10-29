"""Verdict figure: per gold line, mid-span exact section (+-6 mm) with outer/inner apex, bench apex line
crossing, gold point and 4 mm circle; bottom row: gold offset to outer/inner apex vs axis station with
fold extent and gold span."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

m = load_mesh(); gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
S = json.load(open(OUT + '/scripts/stations.json'))
bys = {r['s']: r for r in S['stations'] if not r['empty']}
GO = json.load(open(OUT + '/scripts/gold_offsets.json'))
bench = {b['bend_id']: b for b in json.load(open(OUT + '/bench_lines/lines.json'))['bends']}
pairs = {'M01': ('bend3_Ap_F', 'B3', (-36.6, -9.6)), 'M02': ('bend2_W_Ap', 'B2', (-30.8, -9.8)), 'M03': ('bend1_A_W', 'B1', (-22.25, -4.6))}
fig, axs = plt.subplots(2, 3, figsize=(19, 12), gridspec_kw={'height_ratios': [1.25, 1]})
for j, (gid, (b, bid, ext)) in enumerate(pairs.items()):
    a, bb = gold[gid]
    ta, tb = (a - c0) @ d, (bb - c0) @ d
    s = round(((ta + tb) / 2) * 2) / 2
    o = c0 + d * s
    ax = axs[0, j]
    segs, _ = section2d(m, o, d, u, w)
    for q in segs:
        ax.plot(q[:, 0], q[:, 1], '-', color='0.15', lw=1.3)
    rec = bys[s]['bends'][b]
    ao = np.array(rec['outer']['apex_fit']); ai = np.array(rec['inner']['apex_fit'])
    ax.plot(*ao, 'D', mfc='none', mec='#1f5fbf', ms=10, mew=2, label='outer apex (auditor3 fit)')
    ax.plot(*ai, 's', mfc='none', mec='#8a2be2', ms=10, mew=2, label='inner apex (auditor3 fit)')
    ax.plot(*rec['outer']['X'], 'x', color='#1f5fbf', ms=10, mew=2, label='outer virtual sharp')
    ax.plot(*rec['inner']['X'], 'x', color='#8a2be2', ms=10, mew=2, label='inner virtual sharp')
    # bench apex crossing
    L = bench[bid]['label_line']
    p0, p1 = np.array(L['start']), np.array(L['end'])
    t = (s - (p0 - c0) @ d) / ((p1 - p0) @ d)
    pb = p0 + (p1 - p0) * t
    ax.plot((pb - o) @ u, (pb - o) @ w, 'o', color='#ff7f0e', ms=7, label=f'bench_labeling apex {bid}')
    t = (s - ta) / (tb - ta)
    pg = a + (bb - a) * t
    gq = np.array([(pg - o) @ u, (pg - o) @ w])
    ax.plot(*gq, '*', color='#1a9c3a', ms=18, mec='k', label=f'human gold {gid}')
    ax.add_patch(plt.Circle(gq, 4.0, fill=False, ls=':', color='#1a9c3a'))
    c = (ao + ai) / 2
    ax.set_xlim(c[0] - 6.5, c[0] + 6.5); ax.set_ylim(c[1] - 6.5, c[1] + 6.5)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower left')
    dvo = np.linalg.norm(gq - ao); dvi = np.linalg.norm(gq - ai)
    ax.set_title(f'{gid} ({b}) exact section at s={s} (mid-span)\n|gold-outer apex|={dvo:.2f} mm  |gold-inner apex|={dvi:.2f} mm  t=2.82 mm', fontsize=9)
    ax2 = axs[1, j]
    rows = GO[gid]
    ss = [r['s'] for r in rows]
    ax2.plot(ss, [r['d_outer_apex'] for r in rows], '-o', ms=3, color='#1f5fbf', label='|gold - outer apex|')
    ax2.plot([r['s'] for r in rows if 'd_inner_apex' in r], [r['d_inner_apex'] for r in rows if 'd_inner_apex' in r], '-s', ms=3, color='#8a2be2', label='|gold - inner apex|')
    ax2.plot(ss, [abs(r['across']) for r in rows], '-', color='0.5', label='|across-bisector component|')
    ax2.axhline(4.0, color='r', ls=':', label='matcher 4 mm')
    ax2.axvspan(*ext, color='#cfe8cf', alpha=0.6, label='physical fold extent')
    ax2.axvspan(min(ta, tb), max(ta, tb), ymin=0.93, ymax=1.0, color='#1a9c3a', alpha=0.9, label='gold span')
    ax2.set_ylim(0, 5); ax2.set_xlim(-40, -2); ax2.grid(alpha=0.3)
    ax2.set_xlabel('station s along common bend axis (mm)'); ax2.set_ylabel('mm')
    ax2.legend(fontsize=7, loc='center left')
    ax2.set_title(f'{gid}: offsets along the fold (exact sections every 0.5 mm)', fontsize=9)
fig.suptitle('sc_70b39f530f00 (49024000_A) auditor3: all three human lines sit on the INNER (concave) skin apex, ~2.5-2.6 mm from the outer apex, parallel and full-length', fontsize=11)
fig.tight_layout()
fig.savefig(OUT + '/verdict_summary.png', dpi=75)
print('ok')
