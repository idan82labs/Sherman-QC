"""Auditor 2: measure the long-side bottom-flange joggles along e1 (position, rise, and slope angle of the joggle web)."""
import numpy as np, json
from a2_frame import *
V, N, F = load_scan()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
res = {}
for side, (lo, hi, sgn) in {'e2min': (-95, -58, +1), 'e2max': (66, 102, -1)}.items():
    rows = []
    for s in np.arange(-150, 170, 10.0):
        m = (np.abs(P[:, 0]-s) < 1.0) & (P[:, 1] > lo) & (P[:, 1] < hi) & (P[:, 2] < -15)
        Q = P[m]; Qn = Nn[m]
        if len(Q) < 50: rows.append(dict(e1=s, n=int(len(Q)))); continue
        # lower (outer) skin: normal pointing -e3
        low = Q[Qn[:, 2] < -0.9]
        if len(low) < 20: rows.append(dict(e1=s, n=int(len(Q)), note='no outer skin')); continue
        # bin along e2 and take median e3 of outer skin
        bins = np.arange(lo, hi+0.5, 1.0)
        idx = np.digitize(low[:, 1], bins)
        prof = [(bins[i-1]+0.5, float(np.median(low[idx == i, 2]))) for i in range(1, len(bins)) if (idx == i).sum() >= 3]
        prof = np.array(prof)
        # outer levels: near wall side vs inner side
        if sgn > 0:
            near = prof[prof[:, 0] < lo+12]; far = prof[prof[:, 0] > hi-10]
        else:
            near = prof[prof[:, 0] > hi-12]; far = prof[prof[:, 0] < lo+10]
        zn = float(np.median(near[:, 1])) if len(near) else None
        zf = float(np.median(far[:, 1])) if len(far) else None
        # joggle web: points with normals tilted (|n_e2| > 0.3) on the outer skin side in the step
        web = Q[(np.abs(Qn[:, 1]) > 0.35) & (Q[:, 2] < -17) & (Q[:, 2] > -23)]
        if sgn > 0: web = web[(web[:, 1] > lo+5) & (web[:, 1] < hi-3)]
        else: web = web[(web[:, 1] < hi-5) & (web[:, 1] > lo+3)]
        wpos = float(np.median(web[:, 1])) if len(web) else None
        # web slope from normals
        wn = Qn[(np.abs(Qn[:, 1]) > 0.35) & (Q[:, 2] < -17) & (Q[:, 2] > -23)]
        rows.append(dict(e1=float(s), n=int(len(Q)), z_outer_near_wall=zn, z_outer_far=zf, rise=(zf-zn) if (zn is not None and zf is not None) else None, web_e2=wpos, n_web=int(len(web))))
    res[side] = rows
    print(side)
    for r in rows: print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(res, open(OUT + '/a2_joggle_measure.json', 'w'), indent=1)
