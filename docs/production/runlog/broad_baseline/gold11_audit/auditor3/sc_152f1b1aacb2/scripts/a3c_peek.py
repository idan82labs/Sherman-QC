"""Quick-look sections (calibration of presence tests)."""
import numpy as np, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
from a3c_sect import *
d = np.load(SCR/'a3c'/'pf.npz'); R, o = d['R'], d['o']
GOLD = {bid: (to_part(s, R, o), to_part(e, R, o)) for bid, s, e in load_gold()}
sl = Slicer(0)
specs = [('M01 +e2 base-wall', s, (1, 2), (70, 100), (-4, 16)) for s in (-60, -40, -10, 15, 45, 70)] + \
        [('M07 +e2 wall-top', s, (1, 2), (74, 96), (30, 48)) for s in (-40, -24, -10, 0, 10, 19, 30)] + \
        [('-e2 wall-top', s, (1, 2), (-96, -74), (30, 48)) for s in (-60, -50, 0, 55)]
fig, axes = plt.subplots(3, 7, figsize=(28, 12), dpi=80); axes = axes.ravel()
for ax, (nm, s, (i, j), xl, yl) in zip(axes, specs):
    P = sl.section(s)
    ax.plot(P[:, i], P[:, j], 'k.', ms=1.5)
    for bid, (a, b) in GOLD.items():
        if abs(b[0] - a[0]) > 1 and min(a[0], b[0]) <= s <= max(a[0], b[0]):
            t = (s - a[0]) / (b[0] - a[0]); g = a + t * (b - a)
            if xl[0] <= g[i] <= xl[1] and yl[0] <= g[j] <= yl[1]:
                ax.plot(g[i], g[j], 'g*', ms=14); ax.text(g[i], g[j] + 0.8, bid, color='g')
        if abs(b[0] - a[0]) < 1 and abs(a[0] - s) < 3:
            ax.plot([a[i], b[i]], [a[j], b[j]], 'g--', lw=1)
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_title(f'{nm} @ e1={s}', fontsize=9); ax.set_xlabel('e2'); ax.set_ylabel('e3')
for ax in axes[len(specs):]: ax.axis('off')
fig.tight_layout(); fig.savefig(SCR/'a3c'/'peek.png'); print('ok')
