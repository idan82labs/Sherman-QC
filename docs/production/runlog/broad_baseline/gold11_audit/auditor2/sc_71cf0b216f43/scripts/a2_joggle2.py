"""Auditor 2: joggle bend geometry (web angle, the two bend line positions) + edge/hem and corner-tab checks."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
out = {}
fig, axs = plt.subplots(2, 4, figsize=(24, 9))
for r, (side, y0, sgn) in enumerate([('e2min', -76.3, +1), ('e2max', 84.1, -1)]):
    rows = []
    for s in np.arange(-140, 161, 10.0):
        m = (np.abs(P[:, 0]-s) < 1.0) & (np.abs(P[:, 1]-y0) < 6) & (P[:, 2] < -16) & (P[:, 2] > -24)
        Q = P[m]; Qn = Nn[m]
        # outer (downward-facing) skin
        lo = Q[Qn[:, 2] < -0.2]; lon = Qn[Qn[:, 2] < -0.2]
        if len(lo) < 30: continue
        # web points: normal tilted more than 15 deg from -e3
        tilt = np.degrees(np.arccos(np.clip(-lon[:, 2], -1, 1)))
        web = tilt > 15
        if web.sum() < 5: continue
        # web angle = 90th percentile tilt (flat middle of web)
        wa = float(np.percentile(tilt[web], 85))
        # levels each side
        a = lo[(lo[:, 1]-y0)*sgn < -2.5][:, 2]; b = lo[(lo[:, 1]-y0)*sgn > 2.5][:, 2]
        # bend lines: where tilt crosses half of web angle on each side of the web
        wy = lo[web][:, 1]
        rows.append(dict(e1=float(s), web_angle_deg=wa, web_e2_min=float(wy.min()), web_e2_max=float(wy.max()), level_wall_side=float(np.median(a)) if len(a) else None, level_far_side=float(np.median(b)) if len(b) else None))
    out[side] = rows
    print(side)
    for q in rows[::3]: print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()})
    wa = np.array([q['web_angle_deg'] for q in rows]); print('   web angle median', round(float(np.median(wa)), 1), 'range', round(float(wa.min()), 1), round(float(wa.max()), 1))
    for i, s in enumerate([-100, 0, 100]):
        ax = axs[r, i]
        m = (np.abs(P[:, 0]-s) < 0.5) & (np.abs(P[:, 1]-y0) < 8) & (P[:, 2] < -15) & (P[:, 2] > -24)
        col = np.where(Nn[m, 2] < -0.2, 'k', 'tab:orange')
        ax.scatter(P[m, 1], P[m, 2], s=2, c=col); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'{side} flange joggle, cut e1={s} (black=down-facing skin)', fontsize=9)
    # edge / hem check: window edge of the top face and flange inner edge
    ax = axs[r, 3]
    yy = -64.5 if side == 'e2min' else 71.0
    m = (np.abs(P[:, 0]-0) < 0.5) & (np.abs(P[:, 1]-yy) < 7)
    ax.scatter(P[m, 1], P[m, 2], s=1.5, c='k'); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_title(f'{side}: top-face window edge & flange edge near e2={yy} (cut e1=0) - hem check', fontsize=9)
plt.tight_layout(); plt.savefig(OUT + '/a2_joggles_and_edges.png', dpi=60); plt.close()
json.dump(out, open(OUT + '/a2_joggle_bends.json', 'w'), indent=1)
