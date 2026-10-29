"""Cross-sections perpendicular to each gold line (stations at 10/30/50/70/90% of gold span), with gold and every lane's nearby lines marked."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787')
from common import *
P,N=load_points(); G=gold()
L={}
for n,tag in [('predictions_best_f1','best'),('predictions_cad_gated','cadg'),('predictions_f1_zero_shot_decomposition','zs')]:
    for k,v in preds(n).items(): L[tag+':'+k]=v
cols={'best':'r','cadg':'m','zs':'b'}
def frame(d):
    a=np.array([0,0,1.]) if abs(d[2])<.9 else np.array([1.,0,0])
    e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1); return e1,e2
for g,(s,e) in G.items():
    d=(e-s)/np.linalg.norm(e-s); e1,e2=frame(d)
    fig,ax=plt.subplots(1,5,figsize=(30,6.5))
    for a,f in zip(ax,[.1,.3,.5,.7,.9]):
        c=s+f*(e-s)
        q=P-c; t=q@d; m=(np.abs(t)<1.0)
        u=q[m]@e1; v=q[m]@e2; r=(np.abs(u)<35)&(np.abs(v)<35)
        a.scatter(u[r],v[r],s=2,c='0.3')
        a.plot(0,0,'o',ms=14,mfc='none',mec='g',mew=3,label=g+' gold')
        for k,(ps,pe) in L.items():
            pd=pe-ps
            if abs(pd@d)/np.linalg.norm(pd)<np.cos(np.radians(10)): continue
            # intersection of pred line with station plane
            tt=((c-ps)@d)/(pd@d); x=ps+tt*pd-c
            if 0<=tt<=1 and abs(x@e1)<35 and abs(x@e2)<35:
                a.plot(x@e1,x@e2,'x',ms=14,mew=3,color=cols[k.split(':')[0]],label=k)
            elif abs(x@e1)<35 and abs(x@e2)<35:
                a.plot(x@e1,x@e2,'+',ms=10,mew=1.5,color=cols[k.split(':')[0]],label=k+' (extended, outside span)')
        a.set_aspect('equal'); a.set_xlim(-35,35); a.set_ylim(-35,35); a.grid(alpha=.3)
        a.set_title(f'{g} station {int(f*100)}%  (1mm slab, n={r.sum()})'); a.legend(fontsize=7,loc='lower left')
    plt.suptitle(f'sc_a0305a6cf787 cross-section perpendicular to gold {g} (len {np.linalg.norm(e-s):.1f} mm); axes mm in plane perp to gold axis')
    plt.tight_layout(); plt.savefig(OUT+f'xsec_gold_{g}.png',dpi=55); plt.close()
print('ok')
