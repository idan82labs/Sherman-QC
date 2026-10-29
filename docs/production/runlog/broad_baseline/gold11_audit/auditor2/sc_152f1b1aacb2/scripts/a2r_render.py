#!/usr/bin/env python3
"""Auditor 2 (sc_152f1b1aacb2): light matplotlib renders of the audit evidence (read-only inputs).
argv: scratch dir with pf.npz, a2r_sections.json, a2r_truth_match.json, output dir."""
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

S, SEC, TM, OUT = sys.argv[1:5]
d = np.load(S + '/pf.npz')
P, GP, ids, R, c = d['P'], d['GP'], [str(x) for x in d['ids']], d['R'], d['c']
sec = json.load(open(SEC))
tm = json.load(open(TM))
eng = json.load(open('/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/runs/best_f1/sc_152f1b1aacb2/predictions_best_f1.json'))
E = {r['bend_id']: (np.array([r['line_start'], r['line_end']]) - c) @ R.T for r in eng['owned_bend_regions']}
T = {t['truth_id']: np.array(t['apex_pf']) for t in tm['truth_lines']}
G = {i: GP[k] for k, i in enumerate(ids)}
INV = {
    'F_A_base|wall_e2neg': lambda s, q: np.array([s, q[0] - 92.3, q[1] - 7.3]),
    'F_B_base|wall_e2pos': lambda s, q: np.array([s, 92.2 - q[0], q[1] - 7.3]),
    'F_C_wall_e2neg|top_flange': lambda s, q: np.array([s, q[0] - 92.3, 36.0 - q[1]]),
    'F_D_wall_e2pos|top_tabs': lambda s, q: np.array([s, 92.2 - q[0], 36.0 - q[1]]),
    'F_E1_base|end_tab_e1neg': lambda s, q: np.array([q[0] - 105.2, s, q[1] - 7.3]),
    'F_E2_base|end_tab_e1pos': lambda s, q: np.array([107.7 - q[0], s, q[1] - 7.3]),
}


def line_at(L, ax, s):
    a, b = L
    if abs(b[ax] - a[ax]) < 1e-9:
        return None
    t = (s - a[ax]) / (b[ax] - a[ax])
    if t < -1e-6 or t > 1 + 1e-6:
        return None
    return a + t * (b - a)


def st_row(fold, s):
    for r in sec['stations'][fold]:
        if abs(r['s'] - s) < 1e-6:
            return r
    return None


def section_panel(ax, fold, axis, s, gid, eng_id, tid, view, half=0.5, title=''):
    # view: indices of the two part-frame coords to plot (h, v)
    h, v = view
    m = np.abs(P[:, axis] - s) < half
    r = st_row(fold, s)
    gp = line_at(G[gid], axis, s) if gid else None
    centre = INV[fold](s, np.array(r['Xo'])) if (r and 'Xo' in r) else (gp if gp is not None else None)
    if centre is None:
        centre = np.array([s, 0, 0])
    W = 5.5
    mm = m & (np.abs(P[:, h] - centre[h]) < W) & (np.abs(P[:, v] - centre[v]) < W)
    ax.scatter(P[mm, h], P[mm, v], s=4, c='0.45', lw=0, label='scan pts (slab %.1f mm)' % (2 * half))
    txt = []
    if r and r.get('fold'):
        Xo = INV[fold](s, np.array(r['Xo'])); Xi = INV[fold](s, np.array(r['Xi'])); Ap = INV[fold](s, np.array(r['apex']))
        ax.plot(Xo[h], Xo[v], 'kx', ms=9, mew=2, label='X_o outer virtual sharp')
        ax.plot(Xi[h], Xi[v], 'b+', ms=10, mew=2, label='X_i inner virtual sharp')
        ax.plot(Ap[h], Ap[v], 'o', mfc='none', mec='g', ms=9, mew=2, label='outer apex (arc midpoint)')
        ax.add_patch(Circle((Ap[h], Ap[v]), 4.0, fill=False, ls='--', ec='g', lw=1))
        txt.append('fold: incl %.1f deg, Ro~%.1f' % (r['incl_deg'], r['R_outer_est']))
        if gp is not None:
            dv = gp - Ap
            txt.append('%s - apex: %.2f mm' % (gid, np.linalg.norm(dv[[h, v]])))
            txt.append('%s - X_o: %.2f mm' % (gid, np.linalg.norm((gp - Xo)[[h, v]])))
    else:
        txt.append('NO FOLD at this station')
    if gp is not None:
        ax.plot(gp[h], gp[v], '*', c='r', ms=15, mec='k', mew=0.6, label='gold %s' % gid)
    if eng_id and eng_id in E:
        ep = line_at(E[eng_id], axis, s)
        if ep is not None:
            ax.plot(ep[h], ep[v], 'D', c='orange', ms=7, mec='k', mew=0.5, label='best_f1 engine')
    if tid and tid in T:
        tp = line_at(T[tid], axis, s)
        if tp is not None:
            ax.plot(tp[h], tp[v], 's', mfc='none', mec='m', ms=11, mew=1.5, label='auditor apex line')
    ax.set_xlim(centre[h] - W, centre[h] + W); ax.set_ylim(centre[v] - W, centre[v] + W)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    names = ['e1', 'e2', 'e3']
    ax.set_xlabel(names[h] + ' (mm)'); ax.set_ylabel(names[v] + ' (mm)')
    ax.set_title(title + '\n' + '; '.join(txt), fontsize=9)


# ---------------------------------------------------------------- 1) cross-sections of the labelled folds
cfg = [
    ('M01', 'F_A_base|wall_e2neg', 'F1C_63882b208eadf52a', 'F_A', [-46.0, 13.0, 60.0, -75.0, 85.0]),
    ('M02', 'F_B_base|wall_e2pos', 'F1C_ec17d0d585495dd2', 'F_B', [-46.0, 13.0, 61.0, -74.0, 80.0]),
    ('M07', 'F_C_wall_e2neg|top_flange', 'F1C_38dde97c8c7b81c4', 'F_C', [-10.0, 5.0, 24.0, -40.0, 45.0]),
]
fig, axs = plt.subplots(3, 5, figsize=(26, 16))
for i, (gid, fold, eid, tid, sts) in enumerate(cfg):
    for j, s in enumerate(sts):
        lab = 'fold station' if j < 3 else 'OVERRUN station (outside fold)'
        section_panel(axs[i, j], fold, 0, s, gid, eid, tid, (1, 2), half=0.5 if gid != 'M07' else 1.0,
                      title='%s  e1=%.0f  (%s)' % (gid, s, lab))
axs[0, 0].legend(fontsize=7, loc='upper right')
fig.suptitle('sc_152f1b1aacb2 auditor 2: perpendicular sections of M01/M02/M07 (green dashed = 4 mm matcher radius around the outer apex)', fontsize=13)
plt.tight_layout(); plt.savefig(OUT + '/a2r_xsec_M01_M02_M07.png', dpi=72); plt.close()

# ---------------------------------------------------------------- 2) span strips
fig, ax = plt.subplots(figsize=(18, 6.5))
rows = [('F_A base|wall e2<0  vs M01', 'F_A_base|wall_e2neg', 'M01', 'F1C_63882b208eadf52a', 'F_A', 0),
        ('F_B base|wall e2>0  vs M02', 'F_B_base|wall_e2pos', 'M02', 'F1C_ec17d0d585495dd2', 'F_B', 0),
        ('F_C wall e2<0|top flange  vs M07', 'F_C_wall_e2neg|top_flange', 'M07', 'F1C_38dde97c8c7b81c4', 'F_C', 0),
        ('F_D wall e2>0|top tabs  (no gold)', 'F_D_wall_e2pos|top_tabs', None, 'CADF_CAD_B1', None, 0),
        ('F_E1 base|end tab e1<0, along e2  (no gold)', 'F_E1_base|end_tab_e1neg', None, 'F1C_d22eaa92dfac6111', 'F_E1', 1),
        ('F_E2 base|end tab e1>0, along e2  (no gold)', 'F_E2_base|end_tab_e1pos', None, 'F1C_8f6fc7c64898124c', 'F_E2', 1)]
for k, (lab, fold, gid, eid, tid, axn) in enumerate(rows):
    y = -k
    st = [r['s'] for r in sec['stations'][fold] if r.get('fold')]
    ax.scatter(st, [y] * len(st), marker='|', s=180, c='g', lw=2)
    if gid:
        g = G[gid][:, axn]; ax.plot(sorted(g), [y + 0.25] * 2, '-', c='r', lw=4)
        ax.text(max(g) + 1, y + 0.25, gid, color='r', va='center', fontsize=10)
    if eid in E:
        e = E[eid][:, axn]; ax.plot(sorted(e), [y - 0.25] * 2, '--', c='orange', lw=3)
    ax.text(-128, y, lab, ha='right', va='center', fontsize=10)
for t in tm['truth_lines']:
    pass
ax.plot([], [], '|', c='g', ms=15, mew=2, label='scan: fold present at 1 mm station (both plates + arc)')
ax.plot([], [], '-', c='r', lw=4, label='Gold11 human line span')
ax.plot([], [], '--', c='orange', lw=3, label='best_f1 engine line span')
ax.set_yticks([]); ax.set_xlim(-130, 115); ax.grid(axis='x', alpha=.3)
ax.set_xlabel('position along the fold axis (mm, part frame: e1 for F_A..F_D, e2 for F_E1/F_E2)')
ax.legend(loc='lower right', fontsize=9)
ax.set_title('sc_152f1b1aacb2: where each fold physically exists vs where the gold / engine lines run', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + '/a2r_span_strips.png', dpi=80); plt.close()

# ---------------------------------------------------------------- 3) vertical gold lines M03-M06
fig, axs = plt.subplots(2, 4, figsize=(24, 11))
for j, gid in enumerate(['M05', 'M06', 'M04', 'M03']):
    g = G[gid]
    side = 1 if g[:, 1].mean() > 0 else -1
    ge1 = g[:, 0].mean()
    for i, lev in enumerate((15.0, 33.0)):
        ax = axs[i, j]
        m = (np.abs(P[:, 2] - lev) < 0.5) & (np.abs(P[:, 0] - ge1) < 9) & (np.abs(P[:, 1] - side * 91.8) < 4)
        ax.scatter(P[m, 0], P[m, 1], s=5, c='0.4', lw=0)
        t = (lev - g[0][2]) / (g[1][2] - g[0][2]); gp = g[0] + t * (g[1] - g[0])
        ax.plot(gp[0], gp[1], '*', c='r', ms=16, mec='k', mew=.6)
        vr = [x for x in tm['vertical_gold'][gid]['rows'] if abs(x['e3'] - lev) < 1e-6]
        info = ''
        if vr:
            vr = vr[0]
            info = 'wall free end at e1=%.2f; gold %.2f mm %s; end-cap width %.2f mm' % (
                vr['wall_end_e1'], abs(vr['gold_minus_wall_end_e1']),
                'inboard of end' if (vr['gold_inside_wall_by_mm'] or 0) > 0.3 else 'from end', vr['end_cap_e2_extent_mm'] or -1)
        ax.set_xlim(ge1 - 9, ge1 + 9); ax.set_ylim(side * 91.8 - 4, side * 91.8 + 4)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1'); ax.set_ylabel('e2')
        ax.set_title('%s (vertical gold, seen end-on) horizontal slab e3=%.0f\n%s' % (gid, lev, info), fontsize=9)
fig.suptitle('M03-M06: horizontal sections through the vertical gold lines. Wall = 1 mm strip; no perpendicular flange or kink anywhere near the gold point', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + '/a2r_vertical_M03_M06.png', dpi=72); plt.close()

# ---------------------------------------------------------------- 4) missing formed bends
fig, axs = plt.subplots(1, 4, figsize=(24, 6.5))
miss = [('F_D_wall_e2pos|top_tabs', 0, -56.0, (1, 2), 'F_D1 wall e2>0 top tab (left)'),
        ('F_D_wall_e2pos|top_tabs', 0, 58.0, (1, 2), 'F_D2 wall e2>0 top tab (right)'),
        ('F_E1_base|end_tab_e1neg', 1, -36.0, (0, 2), 'F_E1 base end tab at e1~-105'),
        ('F_E2_base|end_tab_e1pos', 1, -37.0, (0, 2), 'F_E2 base end tab at e1~+107')]
for ax, (fold, axn, s, view, lab) in zip(axs, miss):
    section_panel(ax, fold, axn, s, None, None, None, view, half=1.0 if fold.startswith('F_D') else 0.5,
                  title='%s  station %.0f  -- no gold line' % (lab, s))
axs[0].legend(fontsize=7)
fig.suptitle('Formed bends present in the scan with NO Gold11 line (all ~90 deg, R_o ~2 mm)', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + '/a2r_missing_bends.png', dpi=72); plt.close()

# ---------------------------------------------------------------- 5) overview with all three line sets
rng = np.random.default_rng(0)
sub = P[rng.choice(len(P), 60000, replace=False)]
fig, axs = plt.subplots(1, 3, figsize=(27, 9))
views = [((0, 1), 'top view (e1,e2)'), ((0, 2), 'wall e2<0 face-on (e1,e3), pts e2<-80', ), ((0, 2), 'wall e2>0 face-on (e1,e3), pts e2>80')]
for k, (ax, (vw, name)) in enumerate(zip(axs, views)):
    h, v = vw
    if k == 0:
        Q = sub
    elif k == 1:
        Q = P[P[:, 1] < -80][::3]
    else:
        Q = P[P[:, 1] > 80][::3]
    ax.scatter(Q[:, h], Q[:, v], s=0.3, c='0.7', lw=0)
    for nm, L in T.items():
        if k == 1 and L[:, 1].mean() > -80: continue
        if k == 2 and L[:, 1].mean() < 80: continue
        ax.plot(L[:, h], L[:, v], '-', c='g', lw=5, alpha=.6)
        ax.text(L[:, h].mean(), L[:, v].mean() - (4 if k else 6), nm, color='g', fontsize=9, ha='center')
    for nm, L in G.items():
        if k == 1 and L[:, 1].mean() > -80: continue
        if k == 2 and L[:, 1].mean() < 80: continue
        ax.plot(L[:, h], L[:, v], '-', c='r', lw=1.8)
        ax.text(L[:, h].mean() + 1, L[:, v].mean() + 2, nm, color='r', fontsize=10, fontweight='bold')
    for nm, L in E.items():
        if k == 1 and L[:, 1].mean() > -80: continue
        if k == 2 and L[:, 1].mean() < 80: continue
        ax.plot(L[:, h], L[:, v], '--', c='orange', lw=1.2)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(name)
axs[0].plot([], [], '-', c='g', lw=5, alpha=.6, label='auditor-2 scan-measured fold apex lines (7)')
axs[0].plot([], [], '-', c='r', lw=1.8, label='Gold11 human lines M01-M07')
axs[0].plot([], [], '--', c='orange', lw=1.2, label='best_f1 engine lines (6)')
axs[0].legend(loc='center', fontsize=10)
fig.suptitle('sc_152f1b1aacb2 (40500000 SCAN2, intermediate stage: end walls not yet bent) - fold inventory vs gold vs engine', fontsize=13)
plt.tight_layout(); plt.savefig(OUT + '/a2r_overview.png', dpi=70); plt.close()
print('ok')
