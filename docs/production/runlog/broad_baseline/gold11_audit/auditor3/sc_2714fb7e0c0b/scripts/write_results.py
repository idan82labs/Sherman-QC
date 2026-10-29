import json, numpy as np
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
E=json.load(open(W+'extent.json')); I=json.load(open(W+'ideal_apex_lines_raw.json'))
F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
loc=lambda p:((np.array(p)-o)@R.T).round(2).tolist()
lines={
 'B01':dict(verdict='OK',offset_mm=0.30,offset_range_mm=[0.19,0.40],matcher_transverse_vs_apex_mm=0.21,angle_vs_apex_deg=0.17,fold_len_mm=218.0,gold_len_mm=220.1,coverage_pct=100.0,gold_on_fold_pct=99.1,
            note='main-plate side flange (13.5 mm), 92 deg, R_o~2.6; gold on outer apex full length; ends 2 mm past fold end at step'),
 'B02':dict(verdict='OK',offset_mm=0.14,offset_range_mm=[0.11,0.51],matcher_transverse_vs_apex_mm=0.13,angle_vs_apex_deg=0.17,fold_len_mm=218.0,gold_len_mm=219.5,coverage_pct=100.0,gold_on_fold_pct=99.3,
            note='main-plate side flange, 91.5 deg; on outer apex'),
 'B03':dict(verdict='OK',offset_mm=0.20,offset_range_mm=[0.06,0.47],matcher_transverse_vs_apex_mm=0.11,angle_vs_apex_deg=0.20,fold_len_mm=221.0,gold_len_mm=225.0,coverage_pct=100.0,gold_on_fold_pct=98.2,
            note='step lower bend (joggle bend 1), 91.4 deg; on outer apex; ~2 mm overrun each end'),
 'B04':dict(verdict='OK',offset_mm=0.61,offset_range_mm=[0.24,0.92],matcher_transverse_vs_apex_mm=0.59,angle_vs_apex_deg=0.23,fold_len_mm=221.0,gold_len_mm=224.9,coverage_pct=100.0,gold_on_fold_pct=98.3,
            note='shelf return flange (15 mm), 92 deg; on outer arc within 1 mm of apex'),
 'B05':dict(verdict='CONVENTION_OFFSET',offset_mm=2.49,offset_range_mm=[2.43,2.54],matcher_transverse_vs_apex_mm=2.48,angle_vs_apex_deg=0.07,fold_len_mm=221.0,gold_len_mm=225.5,coverage_pct=100.0,gold_on_fold_pct=98.0,
            note='step upper bend (joggle bend 2), 91 deg, R_o~2.7, t=1.5. Gold is +2.43..+2.54 mm along the bisector toward the CONCAVE side, lateral <0.22 mm: sits on the bend axis (arc centre), ~1.0 mm off the inner skin in free air; 1.55 mm from inner virtual sharp, 3.7 mm from outer virtual sharp. Parallel (0.07 deg), full length, constant: a bend-axis locus convention, not the outer virtual sharp. Within frozen 4 mm.'),
 'B06':dict(verdict='NOT_A_BEND',offset_mm=None,coverage_pct=0.0,gold_len_mm=15.0,
            note='vertical 15 mm line in the open corner-relief notch between the shelf return flange (B04) and the shelf left side flange; both flanges end separately (end faces visible, ~2x2 mm notch), no fold. Line is 90 deg to the real small fold it presumably meant (shelf left side flange, missing). 1.0-1.6 mm from nearest surface (in air).'),
 'B07':dict(verdict='CONVENTION_OFFSET',offset_mm=3.10,offset_range_mm=[2.63,3.62],endpoint_perp_to_apex_line_mm=[2.66,3.92],matcher_transverse_vs_apex_mm=3.06,angle_vs_apex_deg=0.69,fold_len_mm=202.0,gold_len_mm=225.5,coverage_pct=100.0,gold_on_fold_pct=89.6,
            note='main-plate end flange (13.5 mm, opposite the step), 92 deg. Gold +2.63..+2.88 mm along bisector toward CONCAVE side (bend-axis depth, same family as B05) but skewed 0.69 deg so lateral drift 0.0->2.2 mm; ends 2.7/3.9 mm from apex. Gold spans full part width 225.5 mm while fold is 202 mm (flange ends at corner reliefs, X=-101.6..100.4): overruns ~11.5 and ~12 mm into reliefs/side flanges. Imprecise but within frozen matcher (3.06 mm, 0.69 deg, 90% overlap).'),
 'B08':dict(verdict='NOT_A_BEND',offset_mm=None,coverage_pct=0.0,gold_len_mm=15.3,
            note='mirror of B06 at the right shelf corner: open corner notch between B04 return flange end (X~110) and right side-flange end (Y~116.8); no fold; 90 deg to the missing right side-flange fold.'),
 'B09':dict(verdict='OK',offset_mm=1.05,offset_range_mm=[0.95,1.11],matcher_transverse_vs_apex_mm=1.05,angle_vs_apex_deg=0.08,fold_len_mm=78.8,gold_len_mm=75.3,coverage_pct=95.7,gold_on_fold_pct=100.0,
            note='formed tab main fold (drawing 78.62), 92 deg; gold on outer arc surface ~1 mm lateral of apex (0.1-0.2 mm from scan); fold Y=-65.3..13.5, gold -63.8..11.6'),
 'B10':dict(verdict='OK',offset_mm=0.86,offset_range_mm=[0.78,0.97],matcher_transverse_vs_apex_mm=0.87,angle_vs_apex_deg=0.17,fold_len_mm=78.8,gold_len_mm=75.8,coverage_pct=96.2,gold_on_fold_pct=100.0,
            note='second formed tab main fold, 92 deg; on outer arc ~0.9 mm lateral of apex'),
}
missing=[
 dict(id='SF_L',confidence='definite',desc='shelf LEFT side flange (drawing Section D-D "15 BOTH SIDES", PEM holes at 16.5/38.5): fold along Y at X=-112.4, Z~30 (shelf level), 91.5 deg, ~44 mm long (Y 73.0..117.0)',apex_raw=I['A_SF_L'],apex_local=[loc(p) for p in I['A_SF_L']],len_mm=44.0),
 dict(id='SF_R',confidence='definite',desc='shelf RIGHT side flange (mirror), fold along Y at X=111.6, 92.7 deg, ~44 mm (Y 73.0..117.0)',apex_raw=I['A_SF_R'],apex_local=[loc(p) for p in I['A_SF_R']],len_mm=44.0),
 dict(id='TAB_ENDS',confidence='minor/convention-dependent',desc='each tab (B09,B10) has a ~7 mm end segment angled ~35 deg (drawing Section C-C 7.40 @ 30 deg) folded 90 deg from the plate along its own ~5 mm fold line, plus a ~35 deg vertical-axis kink in the tab wall (~12 mm tall). Drawing/sender count treat each tab as ONE bend; only counts if every small formed fold is labelled.'),
]
res=dict(scan_id='sc_2714fb7e0c0b',gold11_id='CGF41F49066A',catalog_case='shaked_xv/C06',gold_tier='codex_manual_multiview_drawing_backed (NOT human)',
 scan_sha256_verified='2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba',frame='scan_raw identity',
 method='auditor3 independent: numpy slabs (+-0.5/0.6 mm) perpendicular to each gold line at 19 stations; legs from vertex normals, two skins split, outer/inner virtual sharps, outer apex = first surface hit along bisector from outer sharp; apex line = PCA of apex points; fold extent = 1 mm march along apex line requiring apex points + both legs. Drawing 38172000_01.pdf (both sheets) read for bend inventory.',
 part='ISAC EtherCAT PS tray, AL5052 1.5 mm: main plate + 2 side flanges (B01,B02) + end flange (B07) + Z-step (B03,B05) + shelf with return flange (B04) and 2 side flanges (missing) + 2 formed tabs (B09,B10) = 10 bends, matching sender count 10',
 lines=lines,missing_bends=missing,
 ideal_apex_engine_vs_this_gold_frozen_matcher=dict(tp=8,fp=2,fn=2,precision=0.8,recall=0.8,f1=0.8,unmatched_gold=['B06','B08'],unmatched_ideal=['SF_L','SF_R']),
 ideal_apex_lines_raw=I, local_frame_origin_axes=F.tolist(),
 renders=['a3_overview_top_and_YZ.png','a3_sections_XZ.png','a3_xsections_gold_vs_apex.png','a3_B06_corner_hslices.png','a3_B08_corner_hslices.png','a3_shelf_sideflange_XZ_sections.png','a3_tabs_hslices.png','a3_tabs_XZ_sections.png','a3_tabend_hslices_zoom.png','a3_tabend_perp_sections.png','a3_stepwall_ends_hslices.png','a3_B07_ends_hslices.png'])
json.dump(res,open(OUT+'auditor3_sc_2714fb7e0c0b_measurements.json','w'),indent=1,default=float)
print('written')
