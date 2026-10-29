"""Gold vs apex reference lines with the frozen matcher metrics (auditor 2, read-only).

Reference lines:
  A2_*  : this auditor's physical-extent apex lines (a2_folds.json stations where the fold
          exists; 3-D line fitted through the outer-apex points; ends at the first/last
          existing station +-1 mm)
  BL_*  : research.bench_labeling 'lines' apex output (bl_lines_apex/lines.json)
Writes a2_match.json.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
sys.path.insert(0, '/Users/idantal/ShermanQC/FullProject')
import json
import numpy as np
from a2_common import OUT, load_gold, unit
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics, is_match

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
to_raw = lambda p: c + np.asarray(p) @ R
to_frame = lambda p: (np.asarray(p) - c) @ R.T
G = load_gold()
folds = json.load(open(OUT / 'a2_folds.json'))

ref = {}
for name, F in folds.items():
    d = F['def_']; ax = d['ax']; i, j = d['ij']
    P = []
    for r in F['rows']:
        if r.get('exists') and r.get('apex'):
            p = np.zeros(3); p[ax] = r['s']; p[i], p[j] = r['apex']
            P.append(p)
    if len(P) < 3:
        continue
    P = np.array(P)
    m = P.mean(0); _, _, vt = np.linalg.svd(P - m); dd = vt[0]
    if dd[ax] < 0: dd = -dd
    s = (P - m) @ dd
    resid = np.linalg.norm((P - m) - np.outer(s, dd), axis=1)
    runs = F['exist_runs']
    lo_s = min(r0 for r0, r1 in runs if r1 - r0 >= 6) - 1.0
    hi_s = max(r1 for r0, r1 in runs if r1 - r0 >= 6) + 1.0
    t0 = (lo_s - m[ax]) / dd[ax]; t1 = (hi_s - m[ax]) / dd[ax]
    a, b = m + t0 * dd, m + t1 * dd
    ref['A2_' + name] = dict(start=to_raw(a).tolist(), end=to_raw(b).tolist(), length=float(np.linalg.norm(b - a)),
                            fit_resid_max=float(resid.max()), fit_resid_med=float(np.median(resid)),
                            start_frame=a.round(2).tolist(), end_frame=b.round(2).tolist())
bl = json.load(open(OUT / 'bl_lines_apex/lines.json'))
names = ['M01_lip', 'M02_lip', 'wall_M07', 'left_lip', 'tab_bottom_hinge', 'tab_top_hinge',
         'ear_minus_v', 'ear_plus_v', 'tab_bottom_endkink', 'tab_top_endkink']
for b_, nm in zip(bl['bends'], names):
    if b_['status'] != 'ok':
        continue
    L = b_['label_line']
    ref['BL_' + b_['bend_id'] + '_' + nm] = dict(start=L['start'], end=L['end'], length=b_['length_mm'],
                                                  start_frame=to_frame(L['start']).round(2).tolist(),
                                                  end_frame=to_frame(L['end']).round(2).tolist())
for k, v in ref.items():
    print(f"{k:34s} len {v['length']:6.1f}  {v['start_frame']} -> {v['end_frame']}" + (f"  fit resid max {v['fit_resid_max']:.2f}" if 'fit_resid_max' in v else ''))

th = Thresholds()
out = {'pairs': {}, 'matching': {}}
print('\nper gold line: best reference by transverse offset (angle/offset/gold_ov/pred_ov)')
for gid, (ga, gb) in G.items():
    rows = []
    for k, v in ref.items():
        mt = pair_metrics(np.array([ga, gb]), np.array([v['start'], v['end']]))
        rows.append((k, mt))
    rows.sort(key=lambda x: (x[1]['axis_angle_deg'] > 10, x[1]['transverse_offset_mm']))
    out['pairs'][gid] = {k: {kk: round(vv, 3) for kk, vv in mt.items()} for k, mt in rows}
    for k, mt in rows[:3]:
        print(f"  {gid} vs {k:34s} ang {mt['axis_angle_deg']:5.2f}  off {mt['transverse_offset_mm']:6.2f}  gold_ov {mt['gold_span_overlap']:.2f}  pred_ov {mt['prediction_span_overlap']:.2f}  match={is_match(mt, th)}")
# Hungarian with each reference family
for fam in ('A2_', 'BL_'):
    preds = [dict(bend_id=k, line_start=v['start'], line_end=v['end']) for k, v in ref.items() if k.startswith(fam)]
    gold = [dict(bend_id=k, line_start=a.tolist(), line_end=b.tolist()) for k, (a, b) in G.items()]
    res = match_lines(gold, preds, th)
    out['matching'][fam] = res
    print(f"\nHungarian vs {fam}: TP {res['true_positive']} FP {res['false_positive']} FN {res['false_negative']}; matches {[(m['gold_id'], m['prediction_id']) for m in res['matches']]}")
    print('  unmatched gold', res['unmatched_gold_ids'])
    print('  unmatched ref ', res['unmatched_prediction_ids'])
out['reference_lines'] = ref
json.dump(out, open(OUT / 'a2_match.json', 'w'), indent=1)
