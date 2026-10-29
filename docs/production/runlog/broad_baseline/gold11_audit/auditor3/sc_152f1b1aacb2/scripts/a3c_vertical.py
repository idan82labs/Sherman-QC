"""Vertical gold lines M03-M06: horizontal sections (normal e3) around each line; wall-end position; any fold?
Also: base flatness across e1=+-65 (where the CAD has the two unformed long-side bends)."""
import numpy as np, sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
from a3c_sect import *
d = np.load(SCR/'a3c'/'pf.npz'); R, o = d['R'], d['o']; X = d['X']
GOLD = {bid: (to_part(s, R, o), to_part(e, R, o)) for bid, s, e in load_gold()}
sl = Slicer(2)
out = {}
for gid in ['M03', 'M04', 'M05', 'M06']:
    a, b = GOLD[gid]; gx, gy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    side = 1 if gy > 0 else -1
    rows = []
    for z in [1.5, 3, 5, 8, 12, 15, 20, 25, 30, 35, 38, 40, 41.5, 43]:
        P = sl.section(z)
        W = P[(np.abs(P[:, 1] - gy) < 6) & (np.abs(P[:, 0] - gx) < 25)]
        if len(W) < 5:
            rows.append(dict(z=z, n=int(len(W)))); continue
        # wall run along e1 near the gold: material e1 extent within 25 mm of gold, end position nearest gold
        e1s = np.sort(W[:, 0])
        gaps = np.where(np.diff(e1s) > 1.2)[0]
        segs = []; st = 0
        for gi in list(gaps) + [len(e1s) - 1]:
            segs.append((float(e1s[st]), float(e1s[gi]))); st = gi + 1
        # direction of material in this slice: PCA of points within 3 mm of gold e1
        L = W[np.abs(W[:, 0] - gx) < 4]
        dirn = None
        if len(L) > 6:
            c, u = fit_line2(L[:, :2]); dirn = float(np.degrees(np.arctan2(abs(u[1]), abs(u[0]))))
        # gold point at this height
        t = (z - a[2]) / (b[2] - a[2]); g = a + t * (b - a)
        dmin = float(np.min(np.linalg.norm(W[:, :2] - g[:2], axis=1)))
        # skins at gold e1 (+-1 mm)
        S = W[np.abs(W[:, 0] - g[0]) < 1.0][:, 1]
        rows.append(dict(z=z, n=int(len(W)), segs=[(round(s0, 2), round(s1, 2)) for s0, s1 in segs], dir_deg_from_e1=dirn,
                         gold_e1=round(float(g[0]), 2), gold_e2=round(float(g[1]), 2), gold_to_scan=round(dmin, 2),
                         skins_e2_at_gold=[round(float(S.min()), 2), round(float(S.max()), 2)] if len(S) else None))
    out[gid] = dict(mid_e1=gx, mid_e2=gy, rows=rows)
    print('\n==', gid, 'gold e1 %.2f e2 %.2f  e3 %.1f..%.1f' % (gx, gy, a[2], b[2]))
    for r in rows: print('  ', r)
# base flatness across e1=+-65: fit planes to base strips and compare normals
def plane(m):
    P = X[m]; c, n, s = fit_plane(P); n = n if n[2] > 0 else -n; return c, n, len(P)
res = {}
for nm, lo, hi in [('centre', -55, 55), ('pos_out', 70, 100), ('neg_out', -103, -70)]:
    m = (X[:, 0] > lo) & (X[:, 0] < hi) & (np.abs(X[:, 1]) < 75) & (X[:, 2] > -1.2) & (X[:, 2] < -0.3)  # bottom skin only
    res[nm] = plane(m)
for nm in ['pos_out', 'neg_out']:
    ang = np.degrees(np.arccos(np.clip(res[nm][1] @ res['centre'][1], -1, 1)))
    print('base bottom-skin plane %s vs centre: angle %.2f deg (n=%d)' % (nm, ang, res[nm][2]))
# profile of bottom skin e3 along e1 for |e2|<60
m = (np.abs(X[:, 1]) < 60) & (X[:, 2] > -1.3) & (X[:, 2] < -0.2)
P = X[m]; bins = np.arange(-110, 108, 5)
prof = [(float(b0), round(float(np.median(P[(P[:, 0] >= b0) & (P[:, 0] < b0 + 5), 2])), 3)) for b0 in bins if ((P[:, 0] >= b0) & (P[:, 0] < b0 + 5)).sum() > 20]
print('bottom-skin e3 median per 5 mm e1 bin:', prof)
json.dump(dict(vertical=out, base_profile=prof), open(SCR/'a3c'/'vertical.json', 'w'), indent=1, default=float)
