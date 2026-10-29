import sys, json; sys.path.insert(0,'.'); sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject')
from a1s70_load import *
import numpy as np
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
TH=Thresholds()
R=json.load(open(OUT+'a1_bendfit_bins.json')); E=json.load(open(OUT+'a1_extent_lines.json'))
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
def to3(p2,s): return p2[0]*e1+p2[1]*e2+s*a
ref={}
for k in GOLD:
    ex=E[k]['outer_apex_line']; s0,s1=ex['s_min'],ex['s_max']
    ref[k]={'apex':np.array([to3(R[k]['apex_mid'],s0),to3(R[k]['apex_mid'],s1)]),
            'inner_mid':np.array([to3(R[k]['inner_mid'],s0),to3(R[k]['inner_mid'],s1)]),
            'outer_sharp':np.array([to3(R[k]['outer_sharp'],s0),to3(R[k]['outer_sharp'],s1)]),
            'inner_sharp':np.array([to3(R[k]['inner_sharp'],s0),to3(R[k]['inner_sharp'],s1)])}
out={'gold_vs_reference_loci':{}, 'predictions':{}}
print('gold vs reference locus lines (reference extent = measured fold extent along apex):')
for k,(p,q) in GOLD.items():
    g=np.array([p,q]); out['gold_vs_reference_loci'][k]={}
    for loc,L in ref[k].items():
        pm=pair_metrics(g,L); ok=is_match(pm,TH)
        out['gold_vs_reference_loci'][k][loc]=dict(pm,match=ok)
        print('  %s vs %-11s angle %.2f  transverse %.2f  gold_ovl %.2f  pred_ovl %.2f  match %s'%(k,loc,pm['axis_angle_deg'],pm['transverse_offset_mm'],pm['gold_span_overlap'],pm['prediction_span_overlap'],ok))
base='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_repro/runs/sc_70b39f530f00/'
for lane,f in [('best_f1','best_f1/predictions_best_f1.json'),('runtime','runtime/predictions_runtime.json')]:
    d=json.load(open(base+f)); out['predictions'][lane]={}
    for r in d['owned_bend_regions']:
        P=np.array([r['line_start'],r['line_end']]); mid=P.mean(0); m2=np.array([mid@e1,mid@e2])
        dirn=unit(P[1]-P[0]); angA=np.degrees(np.arccos(min(1,abs(dirn@a))))
        row={'len':float(np.linalg.norm(P[1]-P[0])),'angle_to_axis_deg':float(angA),'s_range':sorted([float(P[0]@a),float(P[1]@a)])}
        best=None
        for k in GOLD:
            dA=float(np.linalg.norm(m2-np.array(R[k]['apex_mid']))); dI=float(np.linalg.norm(m2-np.array(R[k]['inner_mid'])))
            pmg=pair_metrics(np.array(GOLD[k]),P); pma=pair_metrics(ref[k]['apex'],P)
            row[k]=dict(mid2d_to_apex=dA,mid2d_to_inner_mid=dI,vs_gold=dict(pmg,match=is_match(pmg,TH)),vs_apex_ref=dict(pma,match=is_match(pma,TH)))
        out['predictions'][lane][r['bend_id']]=row
        print(lane,r['bend_id'],'len %.1f angle-to-axis %.1f s %s'%(row['len'],angA,np.round(row['s_range'],1)))
        for k in GOLD:
            v=row[k]
            print('     %s: mid->apex %.2f mid->inner %.2f | vs gold: ang %.1f tr %.2f ovl %.2f/%.2f match %s | vs apex-ref: tr %.2f ovl %.2f/%.2f match %s'%(k,v['mid2d_to_apex'],v['mid2d_to_inner_mid'],v['vs_gold']['axis_angle_deg'],v['vs_gold']['transverse_offset_mm'],v['vs_gold']['gold_span_overlap'],v['vs_gold']['prediction_span_overlap'],v['vs_gold']['match'],v['vs_apex_ref']['transverse_offset_mm'],v['vs_apex_ref']['gold_span_overlap'],v['vs_apex_ref']['prediction_span_overlap'],v['vs_apex_ref']['match']))
json.dump(out,open(OUT+'a1_matcher_checks.json','w'),indent=1,default=float)
json.dump({k:{loc:L.tolist() for loc,L in v.items()} for k,v in ref.items()},open(OUT+'a1_reference_locus_lines_scan_raw.json','w'),indent=1)
