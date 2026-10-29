"""Frozen-matcher metrics: human gold vs (a) bench_labeling outer-apex lines, (b) auditor-3 station-fit
outer-apex and inner-apex lines clipped to the physical fold extent."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
sys.path.insert(0, '/Users/idantal/ShermanQC/FullProject')
import numpy as np, json, math
from common import *
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics

gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
S = json.load(open(OUT + '/scripts/stations.json'))
rows = [r for r in S['stations'] if not r['empty']]
bench = json.load(open(OUT + '/bench_lines/lines.json'))
B = {b['bend_id']: b for b in bench['bends']}
bench_map = {'M03': 'B1', 'M02': 'B2', 'M01': 'B3'}
pairs = {'M01': 'bend3_Ap_F', 'M02': 'bend2_W_Ap', 'M03': 'bend1_A_W'}
# physical fold extents along the common axis (from exact close-up sections: first/last station with any arc)
fold_ext = {'bend1_A_W': (-22.5, -5.5), 'bend2_W_Ap': (-30.5, -10.0), 'bend3_Ap_F': (-36.5, -10.0)}


def to3d(s, q):
    return c0 + d * s + u * q[0] + w * q[1]


def fit_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    v = vt[0]
    if v @ d < 0:
        v = -v
    return c, v


lines_out = {}
for gid, b in pairs.items():
    for skin in ('outer', 'inner'):
        P = []
        for r in rows:
            rec = r['bends'][b]
            if rec['present_' + skin] and 'apex_fit' in rec[skin] and rec[skin].get('fit_rms', 0) < 0.06:
                P.append(to3d(r['s'], rec[skin]['apex_fit']))
        P = np.array(P)
        c, v = fit_line(P)
        res = np.linalg.norm((P - c) - np.outer((P - c) @ v, v), axis=1)
        s0, s1 = fold_ext[b]
        # points on fitted line at axis coords s0, s1
        def at(sv):
            t = (sv - (c - c0) @ d) / (v @ d)
            return c + v * t
        lines_out[(gid, skin)] = (at(s0), at(s1))
        print(f'{gid} {b} {skin}-apex line: n={len(P)} fit resid rms={np.sqrt((res**2).mean()):.3f} max={res.max():.3f} '
              f'angle to common axis={math.degrees(math.acos(min(1, abs(v @ d)))):.2f} deg')

thr = Thresholds()
print('\n--- pair_metrics (frozen) human gold vs reference lines ---')
report = {}
for gid, (a, bb) in gold.items():
    G = np.array([a, bb])
    refs = {
        'bench_apex(outer)': np.array([B[bench_map[gid]]['label_line']['start'], B[bench_map[gid]]['label_line']['end']]),
        'bench_sharp(outer)': np.array([B[bench_map[gid]]['sharp']['start'], B[bench_map[gid]]['sharp']['end']]),
        'aud3_outer_apex_full_fold': np.array(lines_out[(gid, 'outer')]),
        'aud3_inner_apex_full_fold': np.array(lines_out[(gid, 'inner')]),
    }
    report[gid] = {}
    for name, R in refs.items():
        pm = pair_metrics(G, R)
        ok = (pm['axis_angle_deg'] <= thr.angle_deg and pm['transverse_offset_mm'] <= thr.transverse_offset_mm
              and pm['gold_span_overlap'] >= thr.gold_span_overlap and pm['prediction_span_overlap'] >= thr.prediction_span_overlap)
        report[gid][name] = {k: round(v, 3) for k, v in pm.items()} | {'match': ok}
        print(f"{gid} vs {name:28s} angle={pm['axis_angle_deg']:.2f} offset={pm['transverse_offset_mm']:.2f} "
              f"goldOv={pm['gold_span_overlap']:.2f} predOv={pm['prediction_span_overlap']:.2f} "
              f"Lg={pm['gold_length_mm']:.1f} Lp={pm['prediction_length_mm']:.1f} MATCH={ok}")

# Hungarian: human gold (3) vs bench apex lines (3)
gold_recs = [{'bend_id': k, 'line_start': list(v[0]), 'line_end': list(v[1])} for k, v in gold.items()]
pred_recs = [{'bend_id': b['bend_id'], 'line_start': b['label_line']['start'], 'line_end': b['label_line']['end']} for b in bench['bends']]
mres = match_lines(gold_recs, pred_recs, thr)
print('\nHungarian human-gold vs bench apex:', {k: mres[k] for k in ('true_positive', 'false_positive', 'false_negative', 'exact')})
for mm in mres['matches']:
    print('  ', mm['gold_id'], '->', mm['prediction_id'], 'offset', mm['transverse_offset_mm'], 'angle', mm['axis_angle_deg'])
aud_recs = [{'bend_id': f'A3_{gid}', 'line_start': list(lines_out[(gid, 'outer')][0]), 'line_end': list(lines_out[(gid, 'outer')][1])} for gid in pairs]
mres2 = match_lines(gold_recs, aud_recs, thr)
print('Hungarian human-gold vs auditor3 outer-apex full-fold lines:', {k: mres2[k] for k in ('true_positive', 'false_positive', 'false_negative', 'exact')})
json.dump(dict(report=report, hungarian_bench=mres, hungarian_aud3=mres2,
               aud3_lines={f'{g}_{s}': [list(map(float, p)) for p in v] for (g, s), v in lines_out.items()},
               fold_extent_common_axis_mm=fold_ext),
          open(OUT + '/matcher_check.json', 'w'), indent=1, default=float)
