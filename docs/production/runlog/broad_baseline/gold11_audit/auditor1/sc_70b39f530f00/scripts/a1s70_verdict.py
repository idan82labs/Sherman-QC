import sys, json; sys.path.insert(0,'.')
from a1s70_load import OUT
R=json.load(open(OUT+'a1_bendfit_bins.json')); E=json.load(open(OUT+'a1_extent_lines.json'))
M=json.load(open(OUT+'a1_matcher_checks.json')); BC=json.load(open(OUT+'a1_bench_crosscheck.json')); SD=json.load(open(OUT+'a1_gold_surface_distance.json'))
lines={}
for k in ('M01','M02','M03'):
    r=R[k]; e=E[k]
    lines[k]=dict(verdict='CONVENTION_OFFSET',locus='inner (concave) skin arc midline',
        deflection_deg=round(r['deflection_deg'],1),
        gold_to_outer_apex_mm=dict(per_1mm_bins_median=round(r['d_apex_bins']['median'],2),bins_min=round(r['d_apex_bins']['min'],2),bins_max=round(r['d_apex_bins']['max'],2)),
        gold_to_inner_arc_midline_mm=round(r['d_imid_bins']['median'],2),
        gold_to_outer_virtual_sharp_mm=round(r['gold_to_outer_sharp'],2),gold_to_inner_virtual_sharp_mm=round(r['gold_to_inner_sharp'],2),
        gold_to_scan_surface_median_mm=round(SD[k]['median'],2),
        axis_angle_vs_fold_deg=0.45,
        fold_extent_s_along_apex=[round(e['outer_apex_line']['s_min'],2),round(e['outer_apex_line']['s_max'],2)],
        fold_length_mm=round(e['fold_len_apex'],2),gold_s=[round(v,2) for v in e['gold_s']],gold_length_mm=round(e['gold_len'],2),
        coverage_of_fold_pct=round(e['coverage_of_fold_pct'],1),gold_inside_fold_pct=round(e['gold_inside_fold_pct'],1),
        overshoot_mm=dict(start=round(e['overshoot_start_mm'],2),end=round(e['overshoot_end_mm'],2)),
        frozen_matcher=dict(vs_ideal_outer_apex_line=dict(transverse_mm=round(M['gold_vs_reference_loci'][k]['apex']['transverse_offset_mm'],2),match=M['gold_vs_reference_loci'][k]['apex']['match']),
                            vs_bench_labeling_outer_apex=dict(transverse_mm=round(BC['outer_'+k]['label_line']['vs_gold']['transverse_offset_mm'],2),match=BC['outer_'+k]['label_line']['vs_gold']['match']),
                            vs_bench_labeling_inner_apex=dict(transverse_mm=round(BC['inner_'+k]['label_line']['vs_gold']['transverse_offset_mm'],2),match=BC['inner_'+k]['label_line']['vs_gold']['match']),
                            vs_outer_virtual_sharp=dict(transverse_mm=round(M['gold_vs_reference_loci'][k]['outer_sharp']['transverse_offset_mm'],2),match=M['gold_vs_reference_loci'][k]['outer_sharp']['match'])))
out=dict(scan_id='sc_70b39f530f00',family='49024000',gold='bend_line_gold_catalog_v1.json#corpus/11_49024000 (M01-M03, human_two_point)',
    auditor='auditor1',date='2026-10-17',frame='reference_mesh_raw == scan raw (identity)',
    part='3 parallel bends: strip -(M03, 38.9 deg)- joggle web -(M02, 38.6 deg)- mid segment -(M01, 79.2 deg)- motor-mount frame; t=2.82 mm',
    lines=lines,missing_bends=[],
    notes=['gold is ON the scanned inner surface, not in free air',
           'earlier provisional tag SLANTED_OR_INCONSISTENT_2.6-11.8mm for M01 is a slab-fit artifact (50% station latched onto M02 arc); exact sections give 2.59-2.69 mm at 2/25/50/75%',
           'baseline misses on this scan are prediction-side: best_f1 M02 FN = CADF_CAD_B4 skewed 12.7 deg; runtime M02/M03 FN = whole-part 120-129 mm lines'],
    reference_apex_lines='a1_reference_locus_lines_scan_raw.json (outer apex over measured fold extent); bench/lines_outer/lines.json (bench_labeling, outer skin)')
json.dump(out,open(OUT+'a1_verdict.json','w'),indent=1)
print(json.dumps(out['lines'],indent=1)[:3000])
