"""Sweep along each gold/pred line: at 2 mm stations, measure whether the sheet is actually bent within 10 mm of the line
(spread of surface-normal angle about the line axis) and the distance from the line to the nearest material."""
import sys, numpy as np
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787')
from common import *
from scipy.spatial import cKDTree
P,N=load_points(); G=gold(); T=cKDTree(P)
L=dict(('gold:'+k,v) for k,v in G.items())
for n,tag in [('predictions_best_f1','best'),('predictions_cad_gated','cadg'),('predictions_f1_zero_shot_decomposition','zs')]:
    for k,v in preds(n).items(): L[tag+':'+k]=v
def frame(d):
    a=np.array([0,0,1.]) if abs(d[2])<.9 else np.array([1.,0,0])
    e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1); return e1,e2
only=sys.argv[1:] 
for k,(s,e) in L.items():
    if only and not any(o in k for o in only): continue
    ln=np.linalg.norm(e-s); d=(e-s)/ln; e1,e2=frame(d)
    row=[]
    for t in np.arange(-0.15*ln,1.15*ln+1e-6,max(2.0,ln/60)):
        c=s+t*d; q=P-c; tt=q@d; u=q@e1; v=q@e2
        m=(np.abs(tt)<1.0)&(u*u+v*v<12**2)
        if m.sum()<20: row.append((t/ln,'.',np.nan)); continue
        n2=np.c_[N[m]@e1,N[m]@e2]; ang=np.degrees(np.arctan2(n2[:,1],n2[:,0]))%180
        # circular spread on 180-periodic angles
        z=np.exp(2j*np.radians(ang)); R=abs(z.mean())
        dist=T.query(c)[0]
        sym='B' if R<0.75 else '-'
        row.append((t/ln,sym,dist))
    print(f'{k:32s} len {ln:6.1f}  ', ''.join(r[1] for r in row))
    print(' '*46, ''.join(('|' if abs(r[0])<0.02 or abs(r[0]-1)<0.02 else ' ') for r in row))
    print(' '*46, ''.join(str(min(9,int(r[2]))) if r[2]==r[2] else '.' for r in row), ' <- dist to nearest material (mm, capped 9)')
