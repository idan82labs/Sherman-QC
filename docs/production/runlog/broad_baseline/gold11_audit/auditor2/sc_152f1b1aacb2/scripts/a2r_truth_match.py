#!/usr/bin/env python3
"""Auditor 2 (sc_152f1b1aacb2): build scan-measured apex lines for every formed fold (from a2r_sections.json),
score the Gold11 human lines and the broad-baseline best_f1 lines against them with the FROZEN matcher
(scripts/evaluate_f1_against_manual_gold.match_lines, default Thresholds), and analyse the four vertical
gold lines (M03-M06) with horizontal slabs.  Read-only except for the output json given on argv."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

ROOT = Path('/Users/idantal/ShermanQC/FullProject')
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics  # frozen matcher

S = sys.argv[1]
SEC = sys.argv[2]
OUT = sys.argv[3]
d = np.load(S + '/pf.npz')
P, NP, GP, ids, R, c = d['P'], d['NP'], d['GP'], [str(x) for x in d['ids']], d['R'], d['c']
sec = json.load(open(SEC))
to_raw = lambda X: np.asarray(X) @ R + c
to_pf = lambda X: (np.asarray(X) - c) @ R.T

INV = {
    'F_A_base|wall_e2neg': lambda s, q: [s, q[0] - 92.3, q[1] - 7.3],
    'F_B_base|wall_e2pos': lambda s, q: [s, 92.2 - q[0], q[1] - 7.3],
    'F_C_wall_e2neg|top_flange': lambda s, q: [s, q[0] - 92.3, 36.0 - q[1]],
    'F_D_wall_e2pos|top_tabs': lambda s, q: [s, 92.2 - q[0], 36.0 - q[1]],
    'F_E1_base|end_tab_e1neg': lambda s, q: [q[0] - 105.2, s, q[1] - 7.3],
    'F_E2_base|end_tab_e1pos': lambda s, q: [107.7 - q[0], s, q[1] - 7.3],
}
AX = {'F_A_base|wall_e2neg': 0, 'F_B_base|wall_e2pos': 0, 'F_C_wall_e2neg|top_flange': 0,
      'F_D_wall_e2pos|top_tabs': 0, 'F_E1_base|end_tab_e1neg': 1, 'F_E2_base|end_tab_e1pos': 1}


def fit_line(pts, s_lo, s_hi, ax):
    m = pts.mean(0)
    u = np.linalg.svd(pts - m)[2][0]
    if u[ax] < 0:
        u = -u
    a = m + u * (s_lo - m[ax]) / u[ax]
    b = m + u * (s_hi - m[ax]) / u[ax]
    resid = np.linalg.norm(np.cross(pts - m, u), axis=1)
    return a, b, float(np.median(resid)), float(resid.max())


truth = []
for name, rows in sec['stations'].items():
    fs = [r for r in rows if r.get('fold')]
    segs = sec['summary'][name]['segments']
    groups = []
    if name.startswith('F_D'):          # two separate tabs -> two lines
        for lo, hi in segs:
            groups.append(([r for r in fs if lo <= r['s'] <= hi], lo, hi))
    else:                               # one fold (windows/reliefs do not split it: owner convention)
        groups.append((fs, segs[0][0], segs[-1][1]))
    for k, (g, lo, hi) in enumerate(groups):
        apex = np.array([INV[name](r['s'], r['apex']) for r in g])
        xo = np.array([INV[name](r['s'], r['Xo']) for r in g])
        a, b, rmed, rmax = fit_line(apex, lo - 0.5, hi + 0.5, AX[name])
        ao, bo, _, _ = fit_line(xo, lo - 0.5, hi + 0.5, AX[name])
        tid = name.split('_')[0] + '_' + name.split('_')[1] + (f'{k + 1}' if len(groups) > 1 else '')
        present = sum(1 for r in g)
        truth.append(dict(truth_id=tid, fold=name, s_range=[lo - 0.5, hi + 0.5], fold_present_stations=present,
                          span_mm=float(np.linalg.norm(b - a)), apex_fit_resid_med_mm=rmed, apex_fit_resid_max_mm=rmax,
                          incl_deg_median=float(np.median([r['incl_deg'] for r in g])),
                          R_outer_median=float(np.median([r['R_outer_est'] for r in g])),
                          apex_pf=[a.tolist(), b.tolist()], sharp_pf=[ao.tolist(), bo.tolist()],
                          line_start=to_raw(a).tolist(), line_end=to_raw(b).tolist(),
                          sharp_raw=[to_raw(ao).tolist(), to_raw(bo).tolist()]))

cat = json.load(open(ROOT / 'data/bend_corpus/annotations/bend_line_gold_catalog_v1.json'))
case = [x for x in cat['cases'] if x['case_id'] == 'corpus/08_40500000'][0]
gold = [dict(bend_id=b['bend_id'], line_start=b['line_start'], line_end=b['line_end']) for b in case['bends']]
eng = json.load(open(ROOT / 'docs/production/runlog/broad_baseline/runs/best_f1/sc_152f1b1aacb2/predictions_best_f1.json'))
engine = [dict(bend_id=r['bend_id'], line_start=r['line_start'], line_end=r['line_end']) for r in eng['owned_bend_regions']]
truth_rec = [dict(bend_id=t['truth_id'], line_start=t['line_start'], line_end=t['line_end']) for t in truth]
truth_sharp = [dict(bend_id=t['truth_id'] + '_sharp', line_start=t['sharp_raw'][0], line_end=t['sharp_raw'][1]) for t in truth]

TH = Thresholds()
res = dict(thresholds=TH.__dict__ if hasattr(TH, '__dict__') else str(TH))
res['truth_lines'] = truth
res['gold_vs_truth_apex'] = match_lines(gold, truth_rec, TH)          # truth playing "prediction"
res['gold_vs_truth_sharp'] = match_lines(gold, truth_sharp, TH)
res['truth_apex_vs_engine'] = match_lines(truth_rec, engine, TH)      # truth playing "gold"
res['gold_vs_engine'] = match_lines(gold, engine, TH)
# full pair metrics gold x truth (to show WHY a pair fails)
pm = {}
for g in gold:
    for t in truth_rec:
        m = pair_metrics(np.array([g['line_start'], g['line_end']]), np.array([t['line_start'], t['line_end']]))
        if m['axis_angle_deg'] < 10 and m['transverse_offset_mm'] < 15:
            pm[f"{g['bend_id']}~{t['bend_id']}"] = {k: round(v, 4) for k, v in m.items()}
res['gold_truth_pair_metrics_nearby'] = pm

# coverage / overextension of gold lines over their fold (along the fold axis, part frame)
cov = {}
for gid, tid in (('M01', 'F_A'), ('M02', 'F_B'), ('M07', 'F_C')):
    g = GP[ids.index(gid)]
    t = [x for x in truth if x['truth_id'] == tid][0]
    glo, ghi = sorted([g[0][0], g[1][0]])
    flo, fhi = t['s_range']
    ov = max(0.0, min(ghi, fhi) - max(glo, flo))
    cov[gid] = dict(fold=tid, gold_e1=[glo, ghi], fold_e1=[flo, fhi], fold_span=fhi - flo, gold_span=ghi - glo,
                    fold_coverage_pct=100 * ov / (fhi - flo), gold_on_fold_pct=100 * ov / (ghi - glo),
                    overrun_before_mm=max(0.0, flo - glo), overrun_after_mm=max(0.0, ghi - fhi))
res['gold_coverage'] = cov

# vertical gold lines: horizontal slabs
vert = {}
for gid in ('M03', 'M04', 'M05', 'M06'):
    g = GP[ids.index(gid)]
    side = 1 if g[:, 1].mean() > 0 else -1
    rows = []
    for lev in (-3.0, 5.0, 15.0, 25.0, 33.0):
        t = (lev - g[0][2]) / (g[1][2] - g[0][2])
        gp = g[0] + t * (g[1] - g[0])
        m = (np.abs(P[:, 2] - lev) < 0.5) & (np.abs(P[:, 0] - gp[0]) < 14) & (np.abs(P[:, 1] - side * 91.8) < 2.5)
        Q = P[m]
        wall_end = float(Q[:, 0].max()) if gp[0] > 0 else float(Q[:, 0].min())
        # e2 extent of points within 0.8 mm of the wall end (end cap ~ thickness; a return flange would be wider)
        cap = Q[np.abs(Q[:, 0] - wall_end) < 0.8]
        # nearest scan point to the gold point
        dn = float(np.min(np.linalg.norm(P[np.abs(P[:, 2] - lev) < 3] - gp, axis=1)))
        # wall skins near the gold e1
        near = Q[np.abs(Q[:, 0] - gp[0]) < 1.0]
        rows.append(dict(e3=lev, gold_e1=float(gp[0]), gold_e2=float(gp[1]), wall_end_e1=wall_end,
                         gold_minus_wall_end_e1=float(gp[0] - wall_end),
                         gold_inside_wall_by_mm=float(abs(wall_end) - abs(gp[0])) if np.sign(wall_end) == np.sign(gp[0]) else None,
                         end_cap_e2_extent_mm=float(np.ptp(cap[:, 1])) if len(cap) > 3 else None,
                         wall_skins_e2_at_gold=[float(np.percentile(near[:, 1], 5)), float(np.percentile(near[:, 1], 95))] if len(near) > 5 else None,
                         nearest_scan_mm=dn))
    # kink test along e1 around the gold position (vertical fold would change the wall's in-plane direction)
    W = P[(np.abs(P[:, 1] - side * 91.8) < 1.3) & (P[:, 2] > 5) & (P[:, 2] < 30) & (np.abs(NP[:, 1]) > 0.9)]
    ge1 = g[:, 0].mean()
    sgn = 1 if ge1 > 0 else -1
    def slope(lo, hi):
        Z = W[(W[:, 0] > lo) & (W[:, 0] < hi)]
        if len(Z) < 50:
            return None
        A = np.c_[Z[:, 0], np.ones(len(Z))]
        # fit e2 = a*e1 + b separately on each skin, average slopes
        med = np.median(Z[:, 1])
        sl = []
        for sk in (Z[Z[:, 1] < med], Z[Z[:, 1] >= med]):
            if len(sk) > 20:
                sl.append(np.linalg.lstsq(np.c_[sk[:, 0], np.ones(len(sk))], sk[:, 1], rcond=None)[0][0])
        return float(np.degrees(np.arctan(np.mean(sl)))) if sl else None
    inner_side = (ge1 - sgn * 12, ge1 - sgn * 1) if True else None
    lo1, hi1 = sorted(inner_side)
    s_in = slope(lo1, hi1)
    lo2, hi2 = sorted((ge1 + sgn * 1, ge1 + sgn * 12))
    s_out = slope(lo2, hi2)
    vert[gid] = dict(rows=rows, wall_dir_deg_inboard=s_in, wall_dir_deg_outboard_of_gold=s_out,
                     kink_deg=(abs(s_in - s_out) if (s_in is not None and s_out is not None) else None),
                     gold_len_mm=float(np.linalg.norm(g[1] - g[0])), gold_e3=[float(g[0][2]), float(g[1][2])])
res['vertical_gold'] = vert
json.dump(res, open(OUT, 'w'), indent=1, default=float)

print('TRUTH LINES')
for t in truth:
    print(f"  {t['truth_id']:5s} {t['fold']:28s} span {t['span_mm']:6.1f} mm  s {t['s_range']}  incl {t['incl_deg_median']:.1f}  Ro {t['R_outer_median']:.2f}  resid med/max {t['apex_fit_resid_med_mm']:.2f}/{t['apex_fit_resid_max_mm']:.2f}")
for k in ('gold_vs_truth_apex', 'gold_vs_truth_sharp', 'truth_apex_vs_engine', 'gold_vs_engine'):
    r = res[k]
    print(k, 'TP', r['true_positive'], 'FP', r['false_positive'], 'FN', r['false_negative'],
          'matches', [(m['gold_id'], m['prediction_id'], round(m['transverse_offset_mm'], 2), round(m['gold_span_overlap'], 2), round(m['prediction_span_overlap'], 2)) for m in r['matches']],
          'unmatched gold', r['unmatched_gold_ids'], 'unmatched pred', r['unmatched_prediction_ids'])
print('PAIR METRICS (nearby)')
for k, v in pm.items():
    print('  ', k, v)
print('COVERAGE', json.dumps(cov, indent=0))
for gid, v in vert.items():
    print('VERT', gid, 'kink_deg', v['kink_deg'], 'dir in/out', v['wall_dir_deg_inboard'], v['wall_dir_deg_outboard_of_gold'])
    for r in v['rows']:
        print('    ', {k: (round(x, 2) if isinstance(x, float) else ([round(y, 2) for y in x] if isinstance(x, list) else x)) for k, x in r.items()})
