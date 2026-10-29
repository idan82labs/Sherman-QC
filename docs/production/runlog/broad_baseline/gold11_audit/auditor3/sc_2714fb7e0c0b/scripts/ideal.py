import sys, json, numpy as np
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject')
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics, is_match
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
E=json.load(open(W+'extent.json'))
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:np.array([b['line_start'],b['line_end']]) for b in G['bends']}
ideal={}
for k in ['B01','B02','B03','B04','B05','B07','SF_L','SF_R']:
    ideal['A_'+k]=np.array([E[k]['apex_fold_start_xyz'],E[k]['apex_fold_end_xyz']])
for k in ['B09','B10']:
    c=np.array(E[k]['apex_line_point']); d=np.array(E[k]['apex_line_dir'])
    cl=(c-o)@R.T; dl=d@R.T
    pts=[]
    for y in (-65.3,13.5):
        t=(y-cl[1])/dl[1]; pts.append(c+t*d)
    ideal['A_'+k]=np.array(pts)
for k,v in ideal.items(): print(k,'len %.1f'%np.linalg.norm(v[1]-v[0]), 'local',((v-o)@R.T).round(1).tolist())
print()
th=Thresholds()
for gk,gv in gold.items():
    best=None
    for ik,iv in ideal.items():
        m=pair_metrics(gv,iv)
        if best is None or (m['axis_angle_deg']<10 and m['transverse_offset_mm']<best[1]['transverse_offset_mm']): 
            if m['axis_angle_deg']<10 or best is None: best=(ik,m)
    ik,m=best
    print(f"{gk} vs {ik}: angle {m['axis_angle_deg']:.2f} transverse {m['transverse_offset_mm']:.2f} gold_ov {m['gold_span_overlap']:.2f} pred_ov {m['prediction_span_overlap']:.2f} match={is_match(m,th)}")
# full Hungarian
gl=[{'bend_id':k,'line_start':v[0].tolist(),'line_end':v[1].tolist()} for k,v in gold.items()]
pl=[{'bend_id':k,'line_start':v[0].tolist(),'line_end':v[1].tolist()} for k,v in ideal.items()]
import inspect
print(inspect.signature(match_lines))
r=match_lines(gl,pl,th)
print({k:(v if not isinstance(v,list) else len(v)) for k,v in r.items()})
for k in r:
    if isinstance(r[k],list) and r[k] and isinstance(r[k][0],dict):
        print(k, [ (x.get('gold_bend_id') or x.get('gold_id') or x.get('bend_id'), x.get('prediction_bend_id') or x.get('prediction_id')) for x in r[k]][:12])
# also: ideal set without SF lines, to see effect; and corrected gold (B06/B08 replaced by SF) vs ideal
json.dump({k:v.tolist() for k,v in ideal.items()},open(W+'ideal_apex_lines_raw.json','w'),indent=1)
