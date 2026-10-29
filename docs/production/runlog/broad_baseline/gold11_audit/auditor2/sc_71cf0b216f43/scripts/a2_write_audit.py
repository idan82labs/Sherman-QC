"""Auditor 2: assemble the audit record for sc_71cf0b216f43 (no gold/human files are modified)."""
import json, numpy as np
from a2_frame import *
V, N, F = load_scan()
c, U = refined_frame(V, N)
raw = lambda p: (c + U @ np.asarray(p, float)).round(2).tolist()
jog = json.load(open(OUT + '/a2_joggle_bends.json'))
def jline(side, e2_lo, e2_hi):
    rows = [r for r in jog[side] if -140 <= r['e1'] <= 150]
    lv = np.median([(r['level_wall_side'] + r['level_far_side'])/2 for r in rows])
    return lv
missing = []
for side, (b1, b2), e1r in [('e2min', (-77.0, -75.9), (-147, 160)), ('e2max', (85.0, 83.8), (-148, 160))]:
    rows = [r for r in jog[side] if -140 <= r['e1'] <= 150]
    lw = float(np.median([r['level_wall_side'] for r in rows])); lf = float(np.median([r['level_far_side'] for r in rows]))
    ang = float(np.median([r['web_angle_deg'] for r in rows]))
    for nm, e2v, z in (('lower (wall-side) joggle bend', b1, lw + 0.3), ('upper (far-side) joggle bend', b2, lf - 0.3)):
        missing.append(dict(side=side, bend=nm, frame_refined=dict(e1=list(e1r), e2=e2v, e3_outer_skin=round(z, 2)),
                            raw_start=raw([e1r[0], e2v, z]), raw_end=raw([e1r[1], e2v, z]), angle_deg_approx=round(ang, 1),
                            joggle_rise_mm=round(lf - lw, 2)))
ext = json.load(open(OUT + '/a2_fold_extent_coverage.json'))
verdicts = {
 'M01': ('CONVENTION_OFFSET', 'parallel (0.19 deg), full-length, constant 4.0-4.2 mm (4.9 at corner ends) on the concave side: sits on the inner (top) skin of the short bottom flange, u~4.8 mm from the wall outer plane, ~2.2 mm past the inner tangent line. Not toward the outer virtual sharp (5.1 mm away). Just over the 4 mm matcher limit.'),
 'M02': ('CONVENTION_OFFSET', 'parallel (0.16 deg), full-length, constant 4.6-4.8 mm (5.3 at ends) on the concave/flange side: u~5.5 mm from the wall outer plane inside the flange thickness (inner skin unscanned there), ~2.9 mm past the inner tangent. Same inside-of-bend placement as M01; larger than any canonical locus (<=2.6 mm) so it is a click-placement convention, not a mold-line one. Borderline.'),
 'M03': ('OK', 'inner-skin locus: on the wall inner skin 0.3-1.2 mm below the inner tangent; 2.3-3.2 mm from the outer apex, parallel (0.31 deg); within the 4 mm limit over its whole length.'),
 'M04': ('OK', 'inside the bend arc (inner side), 1.6-2.0 mm from the outer apex, parallel (0.28 deg), full length.'),
 'M05': ('REAL_ERROR', 'slanted 2.37 deg: runs down the inner face of the e2-min wall from 6.4 mm to 18.4 mm below the top-face plane (d_apex 5.5-17.9, median 11.6); never within 4 mm. Endpoints sit on the inner vertical box-corner edges (1.8 mm in plan) at heights 14.7 and 3.1 vs top bend ~20: wrong pick depth.'),
 'M06': ('REAL_ERROR', 'slanted 1.64 deg: starts 10.7 mm up the e2-min wall at the +e1 corner (endpoint e3=-11.0), converges to the bottom bend at the -e1 corner (d_apex 10.1 -> 1.9, median 5.9); within 4 mm on only 28% of the fold.'),
 'M07': ('REAL_ERROR', 'slanted 1.63 deg: near the top bend at the +e1 corner (3.2 mm), descends to 11.7 mm below it at the -e1 corner (endpoint e3=8.9); within 4 mm on only 13% of the fold; median 6.8 mm.'),
 'M08': ('REAL_ERROR', 'parallel within 0.59 deg but displaced 6.7-9.6 mm up the inner face of the e2-max wall from the bottom bend (d_apex 6.3-8.8, median 7.5), in air 0.2-1.3 mm in front of the unscanned inner skin; offset is away from both virtual sharps, so not a locus convention. Right bend identity, wrong height.'),
}
lines = []
for k, (v, why) in verdicts.items():
    e = ext[k]
    lines.append(dict(gold_id=k, fold=e['fold'], verdict=v, offset_mm_median=round(e['d_apex_median'], 2), offset_mm_range=[round(e['d_apex_min'], 2), round(e['d_apex_max'], 2)],
                      coverage_of_fold_pct=round(e['coverage_of_fold_pct'], 1), fold_len_mm=e['fold_len_mm'], gold_len_mm=round(e['gold_len_mm'], 1),
                      frac_fold_within_4mm=round(e['frac_fold_len_gold_within_4mm'], 2), angle_vs_fold_deg=round(e['angle_gold_vs_fold_apex_deg'], 2), reason=why))
audit = dict(auditor='auditor2', scan_id='sc_71cf0b216f43', gold11_id='CG84C8FA12D2', part='10839000', catalog_case='corpus/01_10839000',
             method='independent: own loader + plane-fitted part frame; per-5mm perpendicular sections fit outer/inner skins, virtual sharps, outer apex; gold expressed as (u,v) depth from wall / face-flange outer planes. Light numpy only (no engine/bench runs, no lock needed).',
             part_geometry='rectangular box frame ~314x208x42 mm, t~1.4-1.5 mm, R_outer~2.4-2.8 mm: continuous top frame face (window) + 4 walls; C-section on all sides; short sides have ~12 mm bottom flanges; long sides have ~35 mm bottom flanges each with a full-length ~1.9 mm joggle ~22 mm in from the wall; open relief seams at the 4 vertical corners, no tabs, no hems.',
             lines=lines, missing_bends=missing,
             counts=dict(gold=8, OK=2, CONVENTION_OFFSET=2, REAL_ERROR=4, missing=4, physical_formed_bends_owner_convention=12))
json.dump(audit, open(OUT + '/a2_audit_sc_71cf0b216f43.json', 'w'), indent=1)
for m in missing: print(m)
print('written')
