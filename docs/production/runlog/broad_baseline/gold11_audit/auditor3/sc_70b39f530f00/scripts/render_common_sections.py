"""Exact cross-sections in the common frame at stations s along the bend axis."""
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
cols = {'M01': '#1a9c3a', 'M02': '#0060c0', 'M03': '#c03000'}
stations = [-45, -40, -38, -35, -30, -25, -20, -15, -10, -5, 0, 5]
fig, axs = plt.subplots(3, 4, figsize=(22, 16))
for ax, s in zip(axs.ravel(), stations):
    o = c0 + d * s
    segs, _ = section2d(m, o, d, u, w)
    for q in segs:
        ax.plot(q[:, 0], q[:, 1], '-', color='0.25', lw=0.8)
    for k, (a, b) in gold.items():
        ta, tb = (a - c0) @ d, (b - c0) @ d
        lo, hi = min(ta, tb), max(ta, tb)
        t = (s - ta) / (tb - ta)
        p = a + (b - a) * t
        inside = lo <= s <= hi
        ax.plot((p - o) @ u, (p - o) @ w, '*' if inside else 'o', ms=14 if inside else 7,
                mfc=cols[k] if inside else 'none', mec=cols[k], label=k + ('' if inside else ' (extrap.)'))
        if inside:
            cc = plt.Circle(((p - o) @ u, (p - o) @ w), 4.0, fill=False, ls=':', color=cols[k])
            ax.add_patch(cc)
    ax.set_title(f's = {s} mm along common axis')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlim(-45, 30); ax.set_ylim(-40, 25)
    ax.legend(fontsize=7, loc='lower left')
fig.suptitle('sc_70b39f530f00 auditor3: exact mesh sections perpendicular to the common bend axis (u = in plane of arm A, w = arm-A normal); star = gold line inside its span, ring = gold extrapolated; dotted = 4 mm')
fig.tight_layout()
fig.savefig(OUT + '/common_sections_overview.png', dpi=80)
print('ok')
