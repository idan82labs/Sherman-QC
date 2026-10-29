"""Auditor-3 evidence renders for sc_152f1b1aacb2 (part frame: e1 along walls, e2 across, e3 up from base outer skin)."""
import numpy as np, sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
from a3c_sect import *
d = np.load(SCR/'a3c'/'pf.npz'); R, o, X = d['R'], d['o'], d['X']
GOLD = {bid: (to_part(s, R, o), to_part(e, R, o)) for bid, s, e in load_gold()}
LM = json.load(open(SCR/'a3c'/'lines_match.json')); AL = {L['bend_id']: (np.array(L['part_start']), np.array(L['part_end'])) for L in LM['apex_lines']}
C = json.load(open(SCR/'a3c'/'corners2.json'))
MISSING = ['A_tab_neg_e2_A', 'A_tab_neg_e2_B', 'A_lip_pos_e1', 'A_lip_neg_e1']
rng = np.random.default_rng(1)

# ---------------- Figure A: overview with verdicts
fig, ax = plt.subplots(2, 2, figsize=(22, 15), dpi=80)
S = X[rng.choice(len(X), 160000, replace=False)]
a0 = ax[0, 0]; oo = np.argsort(S[:, 2])
a0.scatter(S[oo, 0], S[oo, 1], c=S[oo, 2], s=0.25, cmap='Greys', vmin=-15, vmax=50, rasterized=True)
VER = {'M01': 'REAL_ERROR (extent)', 'M02': 'REAL_ERROR (extent)', 'M07': 'REAL_ERROR (extent)', 'M03': 'NOT_A_BEND', 'M04': 'NOT_A_BEND', 'M05': 'NOT_A_BEND', 'M06': 'NOT_A_BEND'}
def draw(axh, i, j, filt=lambda a, b: True, lab=True):
    for bid, (a, b) in AL.items():
        if not filt(a, b): continue
        col = 'red' if bid in MISSING else 'tab:blue'
        axh.plot([a[i], b[i]], [a[j], b[j]], color=col, lw=3.5 if bid in MISSING else 2.2, ls='-', alpha=.85, zorder=4)
        if lab and bid in MISSING:
            axh.text(0.5 * (a[i] + b[i]), 0.5 * (a[j] + b[j]) + 2.5, 'MISSING\n' + bid[2:], color='red', fontsize=8, ha='center', zorder=6)
    for bid, (a, b) in GOLD.items():
        if not filt(a, b): continue
        axh.plot([a[i], b[i]], [a[j], b[j]], color='limegreen', lw=1.6, ls='--', zorder=5)
        axh.plot([a[i], b[i]], [a[j], b[j]], 'o', color='green', ms=4, zorder=5)
        if lab:
            axh.text(0.5 * (a[i] + b[i]) + 1, 0.5 * (a[j] + b[j]) - 3.5, bid, color='darkgreen', fontsize=10, weight='bold', zorder=6)
draw(a0, 0, 1)
a0.set_aspect('equal'); a0.grid(alpha=.3); a0.set_xlabel('e1 (mm)'); a0.set_ylabel('e2 (mm)')
a0.set_title('TOP (e1-e2). green dashed = human gold M01-M07; blue = auditor outer-apex line of a visible fold;\nred = visible fold with NO gold line', fontsize=10)
for k, (axh, cond, ttl) in enumerate([(ax[0, 1], X[:, 1] > 75, '+e2 wall elevation (e2>75): M01 bottom, M07 top, M05/M06 vertical'),
                                       (ax[1, 0], X[:, 1] < -75, '-e2 wall elevation (e2<-75): M02 bottom, M03/M04 vertical, tabs A/B (red) unlabeled')]):
    W = X[cond]; W = W[rng.choice(len(W), min(90000, len(W)), replace=False)]
    oo = np.argsort(-np.abs(W[:, 1]))
    axh.scatter(W[oo, 0], W[oo, 2], c=np.abs(W[oo, 1]), s=0.3, cmap='Greys_r', vmin=60, vmax=110, rasterized=True)
    sgn = 1 if k == 0 else -1
    draw(axh, 0, 2, filt=lambda a, b, sgn=sgn: (a[1] * sgn > 60) and (b[1] * sgn > 60))
    axh.set_aspect('equal'); axh.grid(alpha=.3); axh.set_xlabel('e1'); axh.set_ylabel('e3'); axh.set_title(ttl, fontsize=10)
    axh.set_ylim(-6, 50)
# lips: e1-e3 section strip at e2 in [30,42]
a3 = ax[1, 1]
W = X[(X[:, 1] > 30) & (X[:, 1] < 42)]; W = W[rng.choice(len(W), min(60000, len(W)), replace=False)]
a3.scatter(W[:, 0], W[:, 2], s=0.3, c='0.4', rasterized=True)
draw(a3, 0, 2, filt=lambda a, b: min(a[1], b[1]) < 42 and max(a[1], b[1]) > 30 and abs(a[2] - b[2]) < 2 and abs(a[0]) > 90)
a3.set_xlim(-112, 112); a3.set_ylim(-4, 14); a3.grid(alpha=.3); a3.set_xlabel('e1'); a3.set_ylabel('e3 (stretched x4)')
a3.set_aspect(4.0)
a3.set_title('e1-e3 strip |e2-36|<6: base flat across e1=+-65 (CAD long-side bends NOT formed);\n10 mm 90-deg end lips at e1=+104/-107 (red) have no gold', fontsize=10)
fig.suptitle('sc_152f1b1aacb2 (40500000 SCAN2) auditor 3: human gold vs visible folds  -  gold on folds: M01, M02, M07 (all over-extended); M03-M06 vertical wall-end edges (no fold); 4 folds unlabeled', fontsize=12)
fig.tight_layout(); fig.savefig(OUT/'a3c_A_overview_verdicts.png'); plt.close(fig)

# ---------------- Figure B: perpendicular sections for M01, M02, M07
def row_for(key, s):
    for r in C[key]:
        if abs(r['s'] - s) < 1e-6: return r
    return None
sl0 = Slicer(0)
specs = [('M01', 'M01', [(-60, 'on fold'), (45, 'on fold'), (-40, 'window: ear, no fold'), (75, 'OVERRUN: base free edge')], 1),
         ('M02', 'M02', [(-60, 'on fold'), (45, 'on fold'), (15, 'window: ear, no fold'), (-80, 'OVERRUN: base free edge')], -1),
         ('M07', 'M07', [(-15, 'on fold'), (10, 'on fold'), (-45, 'OVERRUN: plain wall top'), (40, 'OVERRUN: plain wall top')], 1)]
fig, axs = plt.subplots(3, 4, figsize=(22, 17), dpi=80)
for rowi, (gid, key, sts, sgn) in enumerate(specs):
    a, b = GOLD[gid]
    for coli, (s, lab) in enumerate(sts):
        axh = axs[rowi, coli]
        P = sl0.section(s); t = (s - a[0]) / (b[0] - a[0]); g = a + t * (b - a)
        # local 2D: (y,z) with y = sgn*e2
        Q = np.c_[sgn * P[:, 1], P[:, 2]]; gq = np.array([sgn * g[1], g[2]])
        half = 6 if 'on fold' in lab else 9
        m = (np.abs(Q[:, 0] - gq[0]) < half + 3) & (np.abs(Q[:, 1] - gq[1]) < half + 3)
        axh.plot(Q[m, 0], Q[m, 1], '.', color='0.25', ms=2.2)
        axh.add_patch(Circle(gq, 4.0, fill=False, ls=':', color='green'))
        axh.plot(*gq, '*', color='limegreen', mec='k', ms=20, zorder=6, label=f'gold {gid}')
        r = row_for(key, s); txt = f'{gid} @ e1={s}: {lab}'
        if r is not None and r.get('present') and 'apex' in r:
            so, ap = np.array(r['sharp_o']), np.array(r['apex'])
            axh.plot(*so, 'x', color='tab:blue', ms=12, mew=2.5, label='outer virtual sharp')
            axh.plot(*ap, 'D', color='cyan', mec='k', ms=9, label='outer apex (arc midline)', zorder=5)
            if r.get('sharp_i') is not None: axh.plot(*r['sharp_i'], 'x', color='tab:purple', ms=10, mew=2, label='inner virtual sharp')
            if r.get('centre') is not None: axh.plot(*r['centre'], '+', color='brown', ms=14, mew=2, label='arc centre')
            gg = np.array(r['gold_' + gid])
            txt += f"\n|gold-apex| {r['off_apex_' + gid]:.2f} mm (along bisector {r['off_bis_' + gid]:+.2f}, + = into bend)\n|gold-outer sharp| {r['off_sharpO_' + gid]:.2f}  R_o~{r['Ro']:.2f}  angle {r['angle']:.1f}"
        else:
            dmin = np.min(np.linalg.norm(Q - gq, axis=1)) if len(Q) else np.nan
            txt += f'\nno fold in this section; gold->nearest scan {dmin:.2f} mm'
        axh.set_xlim(gq[0] - half, gq[0] + half); axh.set_ylim(gq[1] - half, gq[1] + half); axh.set_aspect('equal'); axh.grid(alpha=.3)
        axh.set_title(txt, fontsize=9); axh.set_xlabel(('+' if sgn > 0 else '-') + 'e2 (mm)'); axh.set_ylabel('e3 (mm)')
        if coli == 0 and rowi == 0: axh.legend(fontsize=8, loc='lower left')
fig.suptitle('Perpendicular mesh sections (plane normal = e1 = bend axis). Dotted circle = 4 mm matcher radius around gold.', fontsize=12)
fig.tight_layout(); fig.savefig(OUT/'a3c_B_xsec_M01_M02_M07.png'); plt.close(fig)

# ---------------- Figure C: vertical lines M03-M06
sl2 = Slicer(2)
fig, axs = plt.subplots(2, 4, figsize=(22, 14), dpi=80, gridspec_kw=dict(height_ratios=[1, 1.6]))
for coli, gid in enumerate(['M03', 'M04', 'M05', 'M06']):
    a, b = GOLD[gid]; gx, gy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    axh = axs[0, coli]
    for z, col in [(10, 'k'), (20, 'tab:orange'), (30, 'tab:purple')]:
        P = sl2.section(z); m = (np.abs(P[:, 0] - gx) < 12) & (np.abs(P[:, 1] - gy) < 8)
        axh.plot(P[m, 0], P[m, 1], '.', color=col, ms=2.5, label=f'section e3={z}')
    axh.plot(gx, gy, '*', color='limegreen', mec='k', ms=20, zorder=6, label=f'gold {gid} (vertical, seen end-on)')
    axh.add_patch(Circle((gx, gy), 4.0, fill=False, ls=':', color='green'))
    axh.set_xlim(gx - 12, gx + 12); axh.set_ylim(gy - 6, gy + 6); axh.set_aspect('equal'); axh.grid(alpha=.3)
    axh.set_xlabel('e1'); axh.set_ylabel('e2'); axh.legend(fontsize=7, loc='lower left')
    axh.set_title(f'{gid}: horizontal sections - wall is a straight 1 mm plate,\nno change of direction (no fold about a vertical axis)', fontsize=9)
    axh = axs[1, coli]
    sgn = 1 if gy > 0 else -1
    W = X[(np.abs(X[:, 1] - gy) < 4) & (np.abs(X[:, 0] - gx) < 14)]
    axh.scatter(W[:, 0], W[:, 2], s=0.6, c=sgn * W[:, 1], cmap='viridis', rasterized=True)
    axh.plot([a[0], b[0]], [a[2], b[2]], '--', color='limegreen', lw=3, label=f'gold {gid}', zorder=5)
    for bid, (p, q) in AL.items():
        if abs(p[1] - gy) < 5 and min(p[0], q[0]) < gx + 14 and max(p[0], q[0]) > gx - 14:
            axh.plot([p[0], q[0]], [p[2], q[2]], color='red' if bid in MISSING else 'tab:blue', lw=3, label=bid)
    axh.set_xlim(gx - 14, gx + 14); axh.set_ylim(-3, 47); axh.set_aspect('equal'); axh.grid(alpha=.3)
    axh.set_xlabel('e1'); axh.set_ylabel('e3'); axh.legend(fontsize=7, loc='center')
    endtxt = {'M03': 'line lies on wall inner face, 7 mm inboard of the wall end (-65.2), under tab A',
              'M04': 'line traces the wall\'s vertical cut end edge (e1 62.7)', 'M05': 'line traces the wall\'s vertical cut end edge (e1 62.5)',
              'M06': 'line traces the wall\'s vertical cut end edge (e1 -65.4)'}[gid]
    axh.set_title(f'{gid} wall-face close-up (points within 4 mm of the wall)\n{endtxt}', fontsize=9)
fig.suptitle('Vertical gold lines M03-M06: no formed fold anywhere along them (blue/red = real folds nearby, perpendicular to the gold)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT/'a3c_C_vertical_M03_M06.png'); plt.close(fig)

# ---------------- Figure D: missing bends
sl1 = Slicer(1)
fig, axs = plt.subplots(1, 4, figsize=(22, 6.5), dpi=80)
items = [('A_tab_neg_e2_A', 'TABS', 0, -58.0, -1), ('A_tab_neg_e2_B', 'TABS', 0, 56.0, -1), ('A_lip_pos_e1', 'LIPpos', 1, 36.0, 1), ('A_lip_neg_e1', 'LIPneg', 1, 36.0, -1)]
for axh, (bid, key, axis, s, sgn) in zip(axs, items):
    r = row_for(key, s)
    if axis == 0:
        P = sl0.section(s); Q = np.c_[sgn * P[:, 1], P[:, 2]]; xl = 'e2 x %d' % sgn
    else:
        P = sl1.section(s); Q = np.c_[sgn * P[:, 0], P[:, 2]]; xl = 'e1 x %d' % sgn
    ap = np.array(r['apex']); so = np.array(r['sharp_o'])
    m = (np.abs(Q[:, 0] - ap[0]) < 14) & (np.abs(Q[:, 1] - ap[1]) < 14)
    axh.plot(Q[m, 0], Q[m, 1], '.', color='0.25', ms=2.2)
    axh.plot(*so, 'x', color='tab:blue', ms=12, mew=2.5); axh.plot(*ap, 'D', color='red', mec='k', ms=10)
    # nearest gold line distance (3D) from apex point
    ap3 = {'TABS': np.array([s, -ap[0], ap[1]]), 'LIPpos': np.array([ap[0], s, ap[1]]), 'LIPneg': np.array([-ap[0], s, ap[1]])}[key]
    best = min(((np.linalg.norm(np.cross(q - p, p - ap3)) / np.linalg.norm(q - p), gid) for gid, (p, q) in GOLD.items()))
    L = [x for x in LM['apex_lines'] if x['bend_id'] == bid][0]
    axh.set_xlim(ap[0] - 12, ap[0] + 6); axh.set_ylim(ap[1] - 12, ap[1] + 6); axh.set_aspect('equal'); axh.grid(alpha=.3)
    axh.set_xlabel(xl); axh.set_ylabel('e3')
    axh.set_title(f'MISSING {bid[2:]} @ {"e1" if axis == 0 else "e2"}={s}\nangle {r["angle"]:.1f} deg, fold length {L["length_mm"]:.1f} mm, R_o~{r["Ro"]:.2f}\nnearest gold line to apex: {best[1]} at {best[0]:.1f} mm (perpendicular/unrelated)', fontsize=9)
fig.suptitle('Visible formed 90-deg folds with no gold line (red diamond = outer apex, blue x = outer virtual sharp)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT/'a3c_D_missing_bends.png'); plt.close(fig)
print('figs ok')
