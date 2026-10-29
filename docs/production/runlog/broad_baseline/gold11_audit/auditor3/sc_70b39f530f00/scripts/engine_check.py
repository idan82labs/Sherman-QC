"""Where do the engine (best_f1, runtime) lines sit relative to the measured folds? Read-only."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
sys.path.insert(0, '/Users/idantal/ShermanQC/FullProject')
sys.path.insert(0, '/Users/idantal/ShermanQC/FullProject/research/broad_baseline')
import numpy as np, json, math
from common import *
import bb_common as bb
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics

gold = load_gold()
MC = json.load(open(OUT + '/matcher_check.json'))
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
apex = {g: np.array(MC['aud3_lines'][f'{g}_outer']) for g in ('M01', 'M02', 'M03')}
RB = '/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/runs'
for lane, path in (('best_f1', RB + '/best_f1/sc_70b39f530f00/predictions_best_f1.json'),
                   ('runtime', RB + '/runtime/sc_70b39f530f00/predictions_runtime.json')):
    try:
        doc = json.load(open(path))
    except FileNotFoundError:
        print(lane, 'missing'); continue
    rows = doc.get('owned_bend_regions') or doc.get('bends') or []
    print('=====', lane, len(rows), 'lines')
    preds = []
    for r in rows:
        e = bb.endpoints(r)
        if e is None:
            continue
        a, b = np.asarray(e[0], float), np.asarray(e[1], float)
        preds.append({'bend_id': str(r.get('bend_id')), 'line_start': a.tolist(), 'line_end': b.tolist()})
        mid = (a + b) / 2
        q = [(mid - c0) @ u, (mid - c0) @ w]
        sa, sb = (a - c0) @ d, (b - c0) @ d
        ang = math.degrees(math.acos(min(1, abs(unit(b - a) @ d))))
        best = None
        for g in ('M01', 'M02', 'M03'):
            pm = pair_metrics(apex[g], np.array([a, b]))
            pg = pair_metrics(np.array(gold[g]), np.array([a, b]))
            if best is None or pm['transverse_offset_mm'] < best[1]['transverse_offset_mm']:
                best = (g, pm, pg)
        g, pm, pg = best
        print(f"{r.get('bend_id')}: mid(u,w)=({q[0]:.2f},{q[1]:.2f}) s[{min(sa,sb):.1f},{max(sa,sb):.1f}] angle-to-axis={ang:.2f}  "
              f"nearest fold={g}: vs OUTER-APEX off={pm['transverse_offset_mm']:.2f} ang={pm['axis_angle_deg']:.2f} ovG={pm['gold_span_overlap']:.2f} ovP={pm['prediction_span_overlap']:.2f} | "
              f"vs HUMAN off={pg['transverse_offset_mm']:.2f} ang={pg['axis_angle_deg']:.2f} ovG={pg['gold_span_overlap']:.2f} ovP={pg['prediction_span_overlap']:.2f}")
    gold_recs = [{'bend_id': k, 'line_start': list(v[0]), 'line_end': list(v[1])} for k, v in gold.items()]
    apex_recs = [{'bend_id': 'APEX_' + k, 'line_start': list(v[0]), 'line_end': list(v[1])} for k, v in apex.items()]
    for name, G in (('human gold', gold_recs), ('aud3 outer-apex reference', apex_recs)):
        mr = match_lines(G, preds, Thresholds())
        print(f'  match vs {name}: TP={mr["true_positive"]} FP={mr["false_positive"]} FN={mr["false_negative"]} unmatched gold={mr["unmatched_gold_ids"]} unmatched pred={mr["unmatched_prediction_ids"]}')
