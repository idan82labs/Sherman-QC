import sys, json; sys.path.insert(0,'.'); sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject')
from a1s70_load import *
import numpy as np
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
TH=Thresholds()
B=OUT+'bench/'
names={0:'M03',1:'M02',2:'M01'}
out={}
for sk in ('outer','inner'):
    L=json.load(open(B+'lines_%s/lines.json'%sk))
    for i,b in enumerate(L['bends']):
        k=names[i]
        g=np.array(GOLD[k])
        row={'status':b.get('status'),'skin':b.get('skin'),'bend_angle_deg':b.get('bend_angle_deg'),'length_mm':b.get('length_mm'),
             'apex_offset_mm':b.get('apex_offset_mm'),'expected_apex_offset_mm':b.get('expected_apex_offset_mm'),'coverage':b.get('coverage'),
             'label_locus':b.get('label_locus'),'confidence':b.get('confidence')}
        for loc in ('label_line','sharp'):
            seg=b.get(loc)
            if not seg: continue
            P=np.array([seg['start'],seg['end']])
            pm=pair_metrics(g,P); row[loc]=dict(line=P.tolist(),vs_gold=dict(pm,match=is_match(pm,TH)))
        out['%s_%s'%(sk,k)]=row
        lv=row.get('label_line',{}).get('vs_gold',{}); sv=row.get('sharp',{}).get('vs_gold',{})
        print('%-5s %s status=%s skin=%s bend=%.1f len=%.1f apex_off=%s | gold vs label_line: tr %.2f ang %.2f ovl %.2f/%.2f match %s | gold vs sharp: tr %.2f match %s'%(
            sk,k,row['status'],row['skin'],row['bend_angle_deg'],row['length_mm'],None if row['apex_offset_mm'] is None else round(row['apex_offset_mm'],2),
            lv.get('transverse_offset_mm',np.nan),lv.get('axis_angle_deg',np.nan),lv.get('gold_span_overlap',np.nan),lv.get('prediction_span_overlap',np.nan),lv.get('match'),
            sv.get('transverse_offset_mm',np.nan),sv.get('match')))
json.dump(out,open(OUT+'a1_bench_crosscheck.json','w'),indent=1,default=float)
