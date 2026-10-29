"""Close-up exact sections around each fold, common frame, with outer/inner apex, gold point, 4 mm circle."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from common import *

m = load_mesh(); gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
S = json.load(open(OUT + '/scripts/stations.json'))
bys = {r['s']: r for r in S['stations'] if not r['empty']}
pairs = {'M01': 'bend3_Ap_F', 'M02': 'bend2_W_Ap', 'M03': 'bend1_A_W'}
centers = {'M01': (-18.9, -11.6), 'M02': (-7.1, -12.2), 'M03': (8.0, -1.0)}
gcol = '#1a9c3a'
which = sys.argv[1]
st = [float(x) for x in sys.argv[2].split(',')]
half = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
b = pairs[which]
ncol = 4
nrow = int(math.ceil(len(st) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.2 * nrow))
axs = np.atleast_1d(axs).ravel()
for ax, s in zip(axs, st):
    o = c0 + d * s
    segs, _ = section2d(m, o, d, u, w)
    for q in segs:
        ax.plot(q[:, 0], q[:, 1], '-', color='0.2', lw=1.0)
    r = bys.get(round(s * 2) / 2)
    txt = []
    if r is not None:
        rec = r['bends'][b]
        for skin, col, mk in (('outer', '#1f5fbf', 'D'), ('inner', '#8a2be2', 's')):
            rr = rec[skin]
            X = rr['X']
            ax.plot(*X, 'x', color=col, ms=10, mew=2)
            if 'apex_fit' in rr and rec['present_' + skin]:
                ax.plot(*rr['apex_fit'], mk, color=col, ms=8, mfc='none', mew=2)
        txt.append(f"fold present outer={rec['present_outer']} inner={rec['present_inner']}")
    for gid, (a, bb) in gold.items():
        ta, tb = (a - c0) @ d, (bb - c0) @ d
        t = (s - ta) / (tb - ta)
        p = a + (bb - a) * t
        qx, qy = (p - o) @ u, (p - o) @ w
        inside = 0 <= t <= 1
        if gid == which:
            ax.plot(qx, qy, '*', ms=16 if inside else 10, mfc=gcol if inside else 'none', mec=gcol, mew=1.5)
            ax.add_patch(plt.Circle((qx, qy), 4.0, fill=False, ls=':', color=gcol))
            txt.append(f"gold {gid} t={t:.2f} ({'inside span' if inside else 'EXTRAPOLATED'})")
    cx, cy = centers[which]
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(f'{which} / {b}  s={s}\n' + '; '.join(txt), fontsize=8)
for ax in axs[len(st):]:
    ax.axis('off')
fig.suptitle(f'sc_70b39f530f00 auditor3 {which}: exact mesh sections (common frame). star=gold; blue D=outer apex, blue x=outer virtual sharp; purple s=inner apex, purple x=inner virtual sharp; dotted=4 mm', fontsize=10)
fig.tight_layout()
fn = OUT + f'/closeup_{which}_{sys.argv[4] if len(sys.argv) > 4 else "a"}.png'
fig.savefig(fn, dpi=75)
print(fn)
