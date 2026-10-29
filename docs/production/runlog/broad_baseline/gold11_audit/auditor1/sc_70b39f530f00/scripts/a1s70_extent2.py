import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np
R=json.load(open(OUT+'a1_bendfit_bins.json'))
a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
m=load(); FN=m.face_normals; FC=m.triangles_center
e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
S=FC@a; X=np.stack([FC@e1,FC@e2],1); NA=FN@a
TH=np.degrees(np.arctan2(FN@e2,FN@e1))
def wrap(d): return (d+180)%360-180
out={}
for k in ['M01','M02','M03']:
    r=R[k]; T1,T2=r['outer_normal_angles']; d=T2-T1
    apex=np.array(r['apex_mid']); imid=np.array(r['inner_mid'])
    p,q=GOLD[k]; ss,se=sorted([p@a,q@a])
    fo=wrap(TH-T1)/d; fi=wrap(TH-(T1+180))/d
    res={}
    for nm,f,P0 in [('outer_apex_line',fo,apex),('inner_midline',fi,imid)]:
        sel=(np.abs(NA)<0.3)&(np.abs(f-0.5)<0.12)&(np.linalg.norm(X-P0,axis=1)<1.2)
        sv=np.sort(S[sel])
        # robust ends: first/last s where local density is sustained (gaps < 1.5 mm)
        lo,hi=sv[0],sv[-1]
        gaps=np.diff(sv); big=np.where(gaps>1.5)[0]
        res[nm]=dict(s_min=float(lo),s_max=float(hi),n=int(sel.sum()),gaps_over_1p5mm=[[float(sv[i]),float(sv[i+1])] for i in big])
    fo_ext=res['outer_apex_line']; L=fo_ext['s_max']-fo_ext['s_min']
    ov=max(0,min(se,fo_ext['s_max'])-max(ss,fo_ext['s_min']))
    res['gold_s']=[float(ss),float(se)]; res['gold_len']=float(se-ss)
    res['fold_len_apex']=float(L); res['coverage_of_fold_pct']=float(100*ov/L); res['gold_inside_fold_pct']=float(100*ov/(se-ss))
    res['overshoot_start_mm']=float(fo_ext['s_min']-ss); res['overshoot_end_mm']=float(se-fo_ext['s_max'])
    fi_ext=res['inner_midline']; Li=fi_ext['s_max']-fi_ext['s_min']; ovi=max(0,min(se,fi_ext['s_max'])-max(ss,fi_ext['s_min']))
    res['coverage_of_inner_midline_pct']=float(100*ovi/Li); res['overshoot_vs_inner_start_mm']=float(fi_ext['s_min']-ss); res['overshoot_vs_inner_end_mm']=float(se-fi_ext['s_max'])
    out[k]=res
    print(k,json.dumps(res))
json.dump(out,open(OUT+'a1_extent_lines.json','w'),indent=1)
