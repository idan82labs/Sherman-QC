"""Dense cross-sections perpendicular to one named line (gold:HBx or lane:ID), stations given as fractions; all nearby lines marked."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787')
from common import *
P,N=load_points(); G=gold()
L=dict(('gold:'+k,v) for k,v in G.items())
for n,tag in [('predictions_best_f1','best'),('predictions_cad_gated','cadg'),('predictions_f1_zero_shot_decomposition','zs')]:
    for k,v in preds(n).items(): L[tag+':'+k]=v
cols={'gold':'g','best':'r','cadg':'m','zs':'b'}
name=sys.argv[1]; fr=[float(x) for x in sys.argv[2].split(',')]; W=float(sys.argv[3]) if len(sys.argv)>3 else 35
s,e=L[name]; d=(e-s)/np.linalg.norm(e-s)
a=np.array([0,0,1.]) if abs(d[2])<.9 else np.array([1.,0,0]); e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1)
nc=min(6,len(fr)); nr=int(np.ceil(len(fr)/nc))
fig,ax=plt.subplots(nr,nc,figsize=(5*nc,5*nr),squeeze=False)
for a_,f in zip(ax.flat,fr):
    c=s+f*(e-s); q=P-c; m=np.abs(q@d)<0.75
    u=q[m]@e1; v=q[m]@e2; r=(np.abs(u)<W)&(np.abs(v)<W)
    a_.scatter(u[r],v[r],s=1.5,c='0.3')
    for k,(ps,pe) in L.items():
        pd=pe-ps
        if abs(pd@d)/np.linalg.norm(pd)<np.cos(np.radians(10)): continue
        tt=((c-ps)@d)/(pd@d); x=ps+tt*pd-c
        if abs(x@e1)<W and abs(x@e2)<W:
            ins=-0.001<=tt<=1.001
            a_.plot(x@e1,x@e2,'o' if k.startswith('gold') else 'x',mfc='none',ms=12 if ins else 6,mew=2.5 if ins else 1,color=cols[k.split(':')[0]])
            a_.annotate(k.split(':')[1][-6:]+('' if ins else '(ext)'),(x@e1,x@e2),fontsize=7,color=cols[k.split(':')[0]],xytext=(4,4),textcoords='offset points')
    a_.set_aspect('equal'); a_.set_xlim(-W,W); a_.set_ylim(-W,W); a_.grid(alpha=.3); a_.set_title(f'{name} @ {f:.2f} ({f*np.linalg.norm(e-s):.0f} mm)',fontsize=9)
for a_ in ax.flat[len(fr):]: a_.axis('off')
plt.suptitle(f'sc_a0305a6cf787 sections perpendicular to {name}; o=gold, x=pred (big=inside span, small=extended); green gold, red best_f1, magenta cad_gated, blue zero-shot decomposition')
plt.tight_layout(); plt.savefig(OUT+f'xsec_dense_{name.replace(":","_")}.png',dpi=50); plt.close(); print('ok')
