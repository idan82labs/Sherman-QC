"""Auditor 2: physical extent of each fold (outer-arc presence along the fold axis) + gold/fold angle + coverage."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
exec(open('a2_measure_defs.py').read())
M = json.load(open(OUT + '/a2_station_measurements.json'))
res = {}
fig, axs = plt.subplots(8, 1, figsize=(16, 26))
for i, (k, fd) in enumerate(FOLDS.items()):
    ax_i, yi = fd['axis'], fd['y']
    nw = np.zeros(3); nw[yi] = fd['wall_sign']; nl = np.zeros(3); nl[2] = fd['leg_sign']
    bis = (nw + nl) / np.sqrt(2)
    # corner location from the median fitted virtual sharp of central stations
    st = [r for r in M[k]['stations'] if r.get('fit_ok')]
    vs = np.median(np.array([r['VSo'] for r in st]), 0)
    rel_y = P[:, yi] - vs[0]; rel_z = P[:, 2] - vs[1]
    # outer arc points: normal close to bisector (within 30 deg), within 3.5 mm of the virtual sharp
    near = np.hypot(rel_y, rel_z) < 3.5
    arcm = near & (Nn @ bis > np.cos(np.radians(30)))
    s = P[arcm, ax_i]
    bins = np.arange(-160, 175, 2.0)
    h, _ = np.histogram(s, bins)
    present = h >= 6
    # longest run of presence allowing gaps up to 6 mm (3 bins)
    idx = np.where(present)[0]
    runs = []; start = idx[0]; prev = idx[0]
    for j in idx[1:]:
        if j - prev > 3: runs.append((start, prev)); start = j
        prev = j
    runs.append((start, prev))
    runs = [(bins[a], bins[b+1]) for a, b in runs]
    main = max(runs, key=lambda r: r[1]-r[0])
    A, B = GL[k]
    g_lo, g_hi = sorted([A[ax_i], B[ax_i]])
    ov = max(0.0, min(main[1], g_hi) - max(main[0], g_lo))
    fold_len = main[1]-main[0]
    # apex points along the fold (from station fits) -> 3D fold axis direction
    pts = []
    for r in st:
        if not (main[0] <= r['s'] <= main[1]): continue
        au, av = r['apex_uv']; inw = -fd['wall_sign']; inl = -fd['leg_sign']
        y = r['VSo'][0] + inw*au; z = r['VSo'][1] + inl*av
        p = np.zeros(3); p[ax_i] = r['s']; p[yi] = y; p[2] = z; pts.append(p)
    pts = np.array(pts); cc = pts.mean(0); _, _, vt = np.linalg.svd(pts - cc); fdir = vt[0]
    gdir = (B - A) / np.linalg.norm(B - A)
    ang = np.degrees(np.arccos(min(1.0, abs(fdir @ gdir))))
    within = [r for r in st if 'd_apex' in r and main[0] <= r['s'] <= main[1]]
    frac4 = float(np.mean([r['d_apex'] <= 4.0 for r in within])) if within else 0.0
    # fraction of the FOLD length where gold is present and within 4 mm (stations every 5 mm over fold)
    fold_st = [r for r in st if main[0] <= r['s'] <= main[1]]
    cov4 = float(np.mean([('d_apex' in r and r['d_apex'] <= 4.0) for r in fold_st]))
    res[k] = dict(fold=fd['name'], fold_extent_axis=[float(main[0]), float(main[1])], fold_len_mm=float(fold_len), all_runs=[[float(a), float(b)] for a, b in runs],
                  gold_extent_axis=[float(g_lo), float(g_hi)], gold_len_mm=float(np.linalg.norm(B-A)),
                  overlap_mm=float(ov), coverage_of_fold_pct=float(100*ov/fold_len), gold_outside_fold_mm=float((g_hi-g_lo)-ov),
                  angle_gold_vs_fold_apex_deg=float(ang), fold_apex_rms_mm=float(np.sqrt(np.mean(np.sum(((pts-cc) - np.outer((pts-cc)@fdir, fdir))**2, 1)))),
                  frac_fold_len_gold_within_4mm=cov4,
                  d_apex_median=float(np.median([r['d_apex'] for r in within])), d_apex_min=float(min(r['d_apex'] for r in within)), d_apex_max=float(max(r['d_apex'] for r in within)))
    ax = axs[i]; ax.bar(bins[:-1], h, width=2, align='edge', color='0.4'); ax.axvspan(g_lo, g_hi, color='lime', alpha=.25, label='gold extent')
    ax.axvspan(main[0], main[1], ymin=0.9, ymax=1.0, color='r', alpha=.6, label='fold extent (outer arc present)')
    ax.set_title(f"{k}: {fd['name']} -- outer-arc point count per 2 mm along fold axis", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + '/a2_fold_extents.png', dpi=55); plt.close()
json.dump(res, open(OUT + '/a2_fold_extent_coverage.json', 'w'), indent=1)
for k, r in res.items():
    print(k, json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in r.items() if kk != 'fold'}))
