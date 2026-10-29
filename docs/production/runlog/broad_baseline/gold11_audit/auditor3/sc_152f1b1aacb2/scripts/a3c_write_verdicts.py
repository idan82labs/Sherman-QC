"""Consolidate auditor-3 measurements + verdicts for sc_152f1b1aacb2 into one JSON (data file, read-only on gold)."""
import json, shutil, numpy as np, sys
sys.path.insert(0, __file__.rsplit('/',1)[0])
from a3c_common import *
S = SCR/'a3c'; D = OUT/'data'; D.mkdir(exist_ok=True)
for f in ['corners2.json', 'corners2_summary.json', 'lines_match.json', 'vertical.json', 'cad_bends.json']:
    shutil.copy(S/f, D/f)
d = np.load(S/'pf.npz'); R, o = d['R'], d['o']
LM = json.load(open(S/'lines_match.json')); SUMM = json.load(open(S/'corners2_summary.json'))
AL = {L['bend_id']: L for L in LM['apex_lines']}; PM = LM['pair_metrics']
G = {bid: (to_part(s, R, o), to_part(e, R, o)) for bid, s, e in load_gold()}
def span_e1(bid): a, b = G[bid]; return sorted([float(a[0]), float(b[0])])
def fold_e1(aid): L = AL[aid]; return sorted([L['part_start'][0], L['part_end'][0]])
lines = []
for gid, aid, key in [('M01', 'A_base_wall_pos_e2', 'M01'), ('M02', 'A_base_wall_neg_e2', 'M02'), ('M07', 'A_top_flange_pos_e2', 'M07')]:
    gs, fs = span_e1(gid), fold_e1(aid)
    ov = max(0.0, min(gs[1], fs[1]) - max(gs[0], fs[0]))
    gsum = SUMM[key]['gold']
    lines.append(dict(gold_id=gid, verdict='REAL_ERROR', error_kind='extent_overrun', on_fold_locus='outer apex' if gid != 'M07' else 'between outer apex and outer virtual sharp',
        offset_from_outer_apex_mm_median=round(gsum['off_apex_med'], 2), offset_p90=round(gsum['off_apex_p90'], 2), offset_max=round(gsum['off_apex_max'], 2),
        offset_along_bisector_mm=round(gsum['along_bis_med'], 2), dist_to_outer_virtual_sharp_mm=round(gsum['to_outer_sharp_med'], 2),
        fold_e1=[round(x, 2) for x in fs], gold_e1=[round(x, 2) for x in gs], fold_coverage_pct=round(100 * ov / (fs[1] - fs[0]), 1),
        gold_on_fold_pct=round(100 * ov / (gs[1] - gs[0]), 1), overrun_mm=[round(fs[0] - gs[0], 1), round(gs[1] - fs[1], 1)],
        frozen_matcher_vs_auditor_apex_line=PM[gid]))
for gid, txt in [('M03', 'vertical line on inner face of -e2 wall at e1=-58.2, 7 mm inboard of the wall end; straight plate at every height; crosses tab-A fold perpendicularly'),
                 ('M04', 'traces vertical cut end edge of -e2 wall (e1=62.7), base to tab B; no fold'),
                 ('M05', 'traces vertical cut end edge of +e2 wall (e1=62.5); no fold'),
                 ('M06', 'traces vertical cut end edge of +e2 wall (e1=-65.4); no fold')]:
    lines.append(dict(gold_id=gid, verdict='NOT_A_BEND', fold_coverage_pct=0.0, note=txt, frozen_matcher_vs_auditor_apex_line=PM[gid]))
missing = [dict(auditor_id=k, line_start_raw=AL[k]['line_start'], line_end_raw=AL[k]['line_end'], part_start=AL[k]['part_start'], part_end=AL[k]['part_end'],
                length_mm=round(AL[k]['length_mm'], 1), angle_deg=round(SUMM['TABS' if 'tab' in k else ('LIPpos' if 'pos' in k else 'LIPneg')]['angle'], 1))
           for k in ['A_tab_neg_e2_A', 'A_tab_neg_e2_B', 'A_lip_pos_e1', 'A_lip_neg_e1']]
out = dict(scan='sc_152f1b1aacb2', part='40500000', file='40500000_0_SCAN2 1.ply', gold='corpus/08_40500000 (human_two_point, 7 lines)', auditor='auditor 3 of 3 (independent)',
    part_frame=dict(R_rows_e1_e2_e3=R.tolist(), origin=o.tolist(), note='part = raw @ R.T - o; e3 up from base outer skin; walls at e2=+90.5/-92.6'),
    method='mesh-plane sections every 0.5 mm perpendicular to each fold axis; outer-skin envelope lines -> outer virtual sharp; outer apex = closest outer-skin point to it; fold present only where wall/flange material leaves the arc and no flat ear continues',
    lines=lines, missing_bends=missing,
    frozen_matcher_human_gold_vs_auditor_apex_lines={k: v for k, v in LM['match'].items() if k != 'matches'},
    cad_crosscheck='STEP 40500000_0 1.stp: 9 bends (Ri 0.8/Ro 1.8, t 1.0): 2x127.3 window-wall bends (=M01/M02 folds), 2x181.3 long-side bends NOT formed in this scan (base flat within 0.15 deg across e1=+-65), 1x49.0 central flange (=M07 fold), 2x13.8 coaxial tabs, 2x21.0 long-side top flanges (= the 10 mm end lips). No vertical-axis bend exists.',
    figures=[str(OUT/f) for f in ['a3c_A_overview_verdicts.png', 'a3c_B_xsec_M01_M02_M07.png', 'a3c_C_vertical_M03_M06.png', 'a3c_D_missing_bends.png']])
json.dump(out, open(OUT/'a3c_verdicts_sc_152f1b1aacb2.json', 'w'), indent=1)
for l in lines: print({k: v for k, v in l.items() if k != 'frozen_matcher_vs_auditor_apex_line'})
