import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
sys.path.insert(0, ROOT)
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
T = json.load(open(OUT + '/a1_apex_track.json')); EXT = json.load(open(OUT + '/a1_fold_extents.json'))
L = json.load(open(OUT + '/a1_apex_lines_raw.json'))
c = np.array(T['frame']['centroid']); R = np.array(T['frame']['R'])
G = load_gold()
FOLD_OF = {'M01': 'L_TAB', 'M02': 'L_FLANGE', 'M03': 'L_FLANGE', 'M04': 'R_TAB', 'M05': 'R_FLANGE', 'M06': 'R_FLANGE'}
off = G['M03'][0] - G['M02'][0]
out = dict(scan_id='sc_07062eee17aa', gold='corpus/06_38172000 (Gold11 CG65CD575B1C)', frame='raw scan == reference_mesh_raw (identity)',
           method='PCA frame of raw vertices; PC1 slabs every 2 mm; outer apex = extreme outer-skin point along fitted outward bisector; '
                  'virtual sharps = intersections of fitted skin lines; fold presence = leg skin 3-6 mm past corner + plate 3-8 mm from corner (0.5 mm steps)',
           m03_minus_m02=off.tolist(), m06_minus_m05=(G['M06'][0] - G['M05'][0]).tolist(), offset_len=float(np.linalg.norm(off)),
           pc3_plate_normal=R[2].tolist(), folds={}, lines={})
for f, runs in EXT.items():
    lo = min(a for a, b in runs); hi = max(b for a, b in runs)
    out['folds'][f] = dict(pc1_runs=runs, span_pc1=[lo, hi], span_mm=hi - lo, apex_line_raw=[L[f]['line_start'], L[f]['line_end']],
                           apex_line_residual_med_mm=L[f]['resid_med'])
th = Thresholds()
for k, (s, e) in G.items():
    f = FOLD_OF[k]; rows = T['folds'][f]
    ps = (s - c) @ R.T; pe = (e - c) @ R.T
    g_lo, g_hi = sorted([ps[0], pe[0]])
    lo, hi = out['folds'][f]['span_pc1']
    ov = max(0.0, min(g_hi, hi) - max(g_lo, lo))
    vals = [r['gold'][k] for r in rows if r.get('bend') and r['gold'][k]['in_span']]
    da = np.array([v['d_apex'] for v in vals]); ab = np.array([v['along_b'] for v in vals]); ac = np.array([v['across'] for v in vals])
    dxo = np.array([v['d_Xo'] for v in vals]); dxi = np.array([v['d_Xi'] for v in vals])
    nofold = [r for r in rows if (not r.get('bend')) and r['gold'][k]['in_span']]
    within4 = float((da <= 4).mean())
    cov = ov / (hi - lo) if within4 > 0.5 else 0.0
    pm = pair_metrics(np.array([s, e]), np.array([L[f]['line_start'], L[f]['line_end']]))
    out['lines'][k] = dict(fold=f, gold_pc1=[float(g_lo), float(g_hi)], gold_len_mm=float(np.linalg.norm(e - s)),
        n_in_fold_stations=len(vals), n_in_span_no_fold_stations=len(nofold),
        d_outer_apex_mm=dict(median=float(np.median(da)), p10=float(np.percentile(da, 10)), p90=float(np.percentile(da, 90)), max=float(da.max())),
        along_outward_bisector_mm_median=float(np.median(ab)), across_mm_median=float(np.median(ac)),
        d_outer_virtual_sharp_median=float(np.median(dxo)), d_inner_virtual_sharp_median=float(np.median(dxi)),
        frac_in_fold_stations_within_4mm=within4, fold_coverage_pct=100 * cov,
        frac_of_gold_on_fold_pct=100 * ov / (g_hi - g_lo) if within4 > 0.5 else 0.0,
        frozen_matcher_vs_auditor_apex_line=dict(pm, match=bool(is_match(pm, th))))
    o = out['lines'][k]
    print(k, f, f"gold PC1 [{g_lo:.1f},{g_hi:.1f}] len {o['gold_len_mm']:.1f} | d_apex med {o['d_outer_apex_mm']['median']:.2f} p90 {o['d_outer_apex_mm']['p90']:.2f} max {o['d_outer_apex_mm']['max']:.2f}"
          f" | along_b {o['along_outward_bisector_mm_median']:+.2f} across {o['across_mm_median']:+.2f} | dXo {o['d_outer_virtual_sharp_median']:.2f} dXi {o['d_inner_virtual_sharp_median']:.2f}"
          f" | cov {o['fold_coverage_pct']:.1f}% gold-on-fold {o['frac_of_gold_on_fold_pct']:.1f}% | no-fold stations in span {len(nofold)}"
          f" | matcher transv {pm['transverse_offset_mm']:.2f} gov {pm['gold_span_overlap']:.2f} pov {pm['prediction_span_overlap']:.2f} match {is_match(pm, th)}")
json.dump(out, open(OUT + '/a1_gold_line_metrics.json', 'w'), indent=1)
for f, d in out['folds'].items(): print(f, d['pc1_runs'], 'span', d['span_mm'])
