"""Fine fold extents: a fold is present at PC1=x if leg-skin points exist 1.5..4 mm along the leg from the
corner (just past the arc) AND plate points exist 3..8 mm from the corner, in a +-0.25 mm slab."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
V, F, N = load_mesh()
c, R = pca_frame(V)
P = (V - c) @ R.T
FOLDS = {
  'L_FLANGE': dict(y0=-112.0, side=+1, leg=+1),
  'R_FLANGE': dict(y0=111.5, side=-1, leg=+1),
  'L_TAB':    dict(y0=-82.0, side=-1, leg=-1),
  'R_TAB':    dict(y0=92.5, side=+1, leg=-1),
}
order = np.argsort(P[:, 0]); Ps = P[order]
xs = np.arange(-168, 163, 0.5)
res = {}
for f, cfg in FOLDS.items():
    pres = []
    for x in xs:
        i0, i1 = np.searchsorted(Ps[:, 0], [x - 0.25, x + 0.25]); Q = Ps[i0:i1]
        dy = (Q[:, 1] - cfg['y0']); h = (Q[:, 2] + 0.75) * cfg['leg']
        leg = (np.abs(dy) < 3.5) & (h > 3.0) & (h < 6.0)
        plate = (dy * cfg['side'] > 3) & (dy * cfg['side'] < 8) & (np.abs(Q[:, 2] + 0.75) < 2.0)
        pres.append(leg.sum() >= 4 and plate.sum() >= 4)
    pres = np.array(pres)
    runs = []; cur = None
    for x, p in zip(xs, pres):
        if p and cur is None: cur = [x, x]
        elif p: cur[1] = x
        elif cur is not None: runs.append(cur); cur = None
    if cur is not None: runs.append(cur)
    res[f] = runs
    print(f, 'fold present runs (PC1, 0.5 mm step):', [[float(a), float(b)] for a, b in runs], ' total', sum(b - a for a, b in runs))
# part extent along PC1 at plate level near each fold
for f, cfg in FOLDS.items():
    m = (np.abs(P[:, 1] - cfg['y0']) < 15) & (np.abs(P[:, 2] + 0.75) < 2)
    print(f, 'plate PC1 extent near fold', P[m, 0].min().round(2), P[m, 0].max().round(2))
json.dump(res, open(OUT + '/a1_fold_extents.json', 'w'), indent=1)
