"""Per-station mid-thickness virtual sharp of the fold near each gold line (2-cluster normal split, 2D line fit per flange,
intersection), and distance of gold / every nearby pred line to it in the section plane. Stations without two flanges -> NO_FOLD."""
import sys, numpy as np
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787')
from common import *
P,N=load_points(); G=gold()
L={}
for n,tag in [('predictions_best_f1','best'),('predictions_cad_gated','cadg'),('predictions_f1_zero_shot_decomposition','zs')]:
    for k,v in preds(n).items(): L[tag+':'+k]=v
def fit2(u,v,n2):
    a=np.angle(np.exp(2j*np.arctan2(n2[:,1],n2[:,0])))  # doubled angle
    # 2-means on unit circle of doubled angle
    c=np.array([a.min(),a.max()])
    for _ in range(20):
        dd=np.abs(np.angle(np.exp(1j*(a[:,None]-c[None]))))
        lab=dd.argmin(1)
        for k in range(2):
            if (lab==k).any(): c[k]=np.angle(np.exp(1j*a[lab==k]).mean())
    sep=np.degrees(abs(np.angle(np.exp(1j*(c[0]-c[1])))))/2
    lines=[]
    for k in range(2):
        m=lab==k
        if m.sum()<30: return None,sep
        X=np.c_[u[m],v[m]]; nh=np.array([np.cos(c[k]/2),np.sin(c[k]/2)])  # cluster normal
        o=X@nh; keep=np.abs(o-np.median(o))<1.5   # drop edge-cap points that share the normal but not the plane
        X=X[keep]; mu=X.mean(0); lines.append((mu,np.array([-nh[1],nh[0]])))
    (p1,d1),(p2,d2)=lines
    A=np.c_[d1,-d2]
    if abs(np.linalg.det(A))<0.2: return None,sep
    t=np.linalg.solve(A,p2-p1); return p1+t[0]*d1,sep
for g,(s,e) in G.items():
    ln=np.linalg.norm(e-s); d=(e-s)/ln
    a=np.array([0,0,1.]) if abs(d[2])<.9 else np.array([1.,0,0]); e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1)
    print(f'== {g} (len {ln:.1f})')
    for f in np.linspace(0.05,0.95,10):
        c=s+f*(e-s); q=P-c; m=(np.abs(q@d)<0.75)
        u=q[m]@e1; v=q[m]@e2; r=(u*u+v*v<15**2)
        n2=np.c_[N[m][r]@e1,N[m][r]@e2]
        vs,sep=fit2(u[r],v[r],n2)
        if vs is None or sep<20 or np.linalg.norm(vs)>15:
            print(f'  {f:.2f}  NO_FOLD (normal sep {sep:.0f} deg)'); continue
        out=[f'gold {np.linalg.norm(vs):4.1f}']
        for k,(ps,pe) in L.items():
            pd=pe-ps
            if abs(pd@d)/np.linalg.norm(pd)<np.cos(np.radians(10)): continue
            tt=((c-ps)@d)/(pd@d); x=ps+tt*pd-c; x2=np.array([x@e1,x@e2])
            dist=np.linalg.norm(x2-vs)
            if dist<15: out.append(f'{k}{"" if 0<=tt<=1 else "(ext)"} {dist:4.1f}')
        print(f'  {f:.2f}  fold angle-between-normals {sep:5.1f}  dist to mid-thickness virtual sharp: '+' | '.join(out))
