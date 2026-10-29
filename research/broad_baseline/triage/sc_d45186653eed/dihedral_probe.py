"""Read-only triage probe for sc_d45186653eed: local dihedral along gold/pred lines."""
import json, sys, numpy as np
from scipy.spatial import cKDTree
R='/Users/idantal/ShermanQC/FullProject/'
z=np.load(R+'docs/production/runlog/benchmark_v2/label_prep/sc_d45186653eed/points_ds.npz')
pts,nrm,lab=z['points'],z['normals'],z['labels']
t=cKDTree(pts)
L={}
for b in json.load(open(R+'docs/production/runlog/benchmark_v2/label_lines/sc_d45186653eed/lines.json'))['bends']:
    L['G:'+b['bend_id']]=(np.array(b['label_line']['start']),np.array(b['label_line']['end']))
for lane,f in [('bf','best_f1/sc_d45186653eed/predictions_best_f1.json'),('rt','runtime/sc_d45186653eed/predictions_runtime.json')]:
    for l in json.load(open(R+'docs/production/runlog/broad_baseline/runs/'+f))['owned_bend_regions']:
        L[lane+':'+l['bend_id']]=(np.array(l['line_start']),np.array(l['line_end']))
def two_means(N):
    # unoriented normals: use outer-product PCA, then split
    c1=N[0]; 
    for _ in range(20):
        s=np.abs(N@c1)
        far=N[np.argmin(s)]
        a=np.abs(N@c1)>=np.abs(N@far)
        if a.all() or (~a).all(): return 0.0
        A=N[a]*np.sign(N[a]@c1)[:,None]; B=N[~a]*np.sign(N[~a]@far)[:,None]
        c1=A.mean(0);c1/=np.linalg.norm(c1); c2=B.mean(0); c2/=np.linalg.norm(c2)
        far=c2
    if min(a.sum(),(~a).sum())<5: return 0.0
    return float(np.degrees(np.arccos(min(1,abs(c1@c2)))))
def probe(s,e,n=12,r=5.0):
    d=(e-s)/np.linalg.norm(e-s); out=[]
    for f in np.linspace(0.04,0.96,n):
        q=s+f*(e-s); idx=t.query_ball_point(q,r)
        if len(idx)<15: out.append('  -- '); continue
        P=pts[idx]; ax=np.abs((P-q)@d); idx=np.array(idx)[ax<2.5]
        if len(idx)<15: out.append('  -- '); continue
        out.append('%5.0f'%two_means(nrm[idx]))
    return ' '.join(out)
for k in (sys.argv[1:] or L):
    s,e=L[k]; print('%-26s len %6.1f | %s'%(k,np.linalg.norm(e-s),probe(s,e)))
