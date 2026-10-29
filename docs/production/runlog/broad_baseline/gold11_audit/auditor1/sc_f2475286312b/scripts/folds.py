"""Auditor-1 fold apex extraction for sc_f2475286312b in a part frame (u along M01..M06, v along M07, w = base normal)."""
from common import *
import json
from scipy.spatial import cKDTree
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T
s2=np.sqrt(2)
# fold definitions: axis index (0=u,1=v,2=w), 2D coord indices, corner guess (2D), plate dirs (from corner), extent range along axis to probe
FOLDS={
 'F1_upflange_v-54':  dict(ax=0, cc=(1,2), c0=(-54.0,-66.0), d1=(-1,0), d2=(0,1),  rng=(-490,-205)),
 'F2_upflange_v-278': dict(ax=0, cc=(1,2), c0=(-278.0,-66.0), d1=(1,0), d2=(0,1),  rng=(-490,-205)),
 'F5_slotflange_v-85':dict(ax=0, cc=(1,2), c0=(-85.0,-66.0), d1=(1,0), d2=(0,-1),  rng=(-490,-205)),
 'F6_slotflange_v-258':dict(ax=0, cc=(1,2), c0=(-258.0,-66.0), d1=(-1,0), d2=(0,-1), rng=(-490,-205)),
 'F3_leftflange_u-477':dict(ax=1, cc=(0,2), c0=(-476.5,-66.0), d1=(1,0), d2=(0,1),  rng=(-285,-45)),
 'F4_endwall_u-224':  dict(ax=1, cc=(0,2), c0=(-224.0,-66.0), d1=(-1,0), d2=(0,-1), rng=(-285,-45)),
 'F7_ear_v-54':       dict(ax=2, cc=(0,1), c0=(-224.0,-54.0), d1=(0,-1), d2=(1,0),  rng=(-135,-60)),
 'F8_ear_v-278':      dict(ax=2, cc=(0,1), c0=(-224.0,-278.0), d1=(0,1), d2=(1,0),  rng=(-135,-60)),
}
def skin_fit(Q, c0, d, nrm, s_lo=5.0, s_hi=14.0):
    rel=Q-c0; s=rel@d; t=rel@nrm
    m=(s>s_lo)&(s<s_hi)&(np.abs(t)<4)
    if m.sum()<12: return None
    tt=t[m]; ss=s[m]
    lo,hi=np.percentile(tt,5),np.percentile(tt,95)
    if hi-lo<0.6: return None  # single skin only
    mid=(lo+hi)/2
    out=tt>mid; inn=~out
    if out.sum()<5 or inn.sum()<5: return None
    # fit t = a + b s for outer and inner skins (allow slight tilt)
    def fit(si,ti):
        A=np.vstack([np.ones_like(si),si]).T; coef,*_=np.linalg.lstsq(A,ti,rcond=None); res=ti-A@coef
        k=np.abs(res)<0.35
        if k.sum()>=5: coef,*_=np.linalg.lstsq(A[k],ti[k],rcond=None)
        return coef
    co=fit(ss[out],tt[out]); ci=fit(ss[inn],tt[inn])
    return dict(outer=co, inner=ci, thick=float(co[0]-ci[0]), n=int(m.sum()))
def line_from(c0,d,nrm,coef):
    # points p = c0 + s d + (a + b s) nrm  -> point at s=0 and direction
    p0=np.array(c0)+coef[0]*np.array(nrm); dd=np.array(d)+coef[1]*np.array(nrm); return p0, dd/np.linalg.norm(dd)
def intersect(p1,t1,p2,t2):
    M=np.array([t1,-t2]).T
    if abs(np.linalg.det(M))<1e-6: return None
    s,_=np.linalg.solve(M,p2-p1); return p1+s*t1
def station(fd, x0, half=0.4, win=18.0):
    ax=fd['ax']; i,j=fd['cc']; c0=np.array(fd['c0'],float)
    m=(np.abs(P[:,ax]-x0)<half)
    Q=P[m][:,[i,j]]
    Q=Q[(np.abs(Q[:,0]-c0[0])<win)&(np.abs(Q[:,1]-c0[1])<win)]
    if len(Q)<30: return None
    d1=np.array(fd['d1'],float); d2=np.array(fd['d2'],float)
    o=-(d1+d2)/np.linalg.norm(d1+d2)   # outward (convex) bisector
    # plate normals pointing to convex side
    n1=np.array([-d1[1],d1[0]]); n1=n1 if n1@o>0 else -n1
    n2=np.array([-d2[1],d2[0]]); n2=n2 if n2@o>0 else -n2
    f1=skin_fit(Q,c0,d1,n1); f2=skin_fit(Q,c0,d2,n2)
    if f1 is None or f2 is None: return dict(ok=False, npts=len(Q), why='plate1' if f1 is None else 'plate2')
    p1,t1=line_from(c0,d1,n1,f1['outer']); p2,t2=line_from(c0,d2,n2,f2['outer'])
    q1,u1=line_from(c0,d1,n1,f1['inner']); q2,u2=line_from(c0,d2,n2,f2['inner'])
    Xo=intersect(p1,t1,p2,t2); Xi=intersect(q1,u1,q2,u2)
    if Xo is None: return dict(ok=False,npts=len(Q),why='parallel')
    # local bisector from fitted outer skins
    ob=-(t1+t2); ob=ob/np.linalg.norm(ob)
    rel=Q-Xo; along=rel@ob; perp=rel@np.array([-ob[1],ob[0]])
    k=(np.abs(perp)<0.45)&(along<0)&(along>-8)
    if k.sum()<1: return dict(ok=False,npts=len(Q),why='no_apex_pts')
    # closest material along the bisector toward concave side = outer apex
    idx=np.where(k)[0]; order=idx[np.argsort(-along[idx])][:3]
    apex=Q[order].mean(0)
    # corner presence: points near the apex within 1 mm (material at the arc)
    dmin=np.min(np.linalg.norm(Q-apex,axis=1))
    angle=np.degrees(np.arccos(np.clip(-(t1@t2),-1,1)))  # bend angle (90 for L)
    return dict(ok=True, npts=len(Q), apex=apex, Xo=Xo, Xi=Xi, ob=ob, apex_to_Xo=float(np.linalg.norm(apex-Xo)),
                thick1=f1['thick'], thick2=f2['thick'], bend_angle=float(180-angle) if angle>90 else float(angle), raw_angle=float(angle))
def to3d(fd, x0, p2):
    v=np.zeros(3); v[fd['ax']]=x0; i,j=fd['cc']; v[i]=p2[0]; v[j]=p2[1]; return v
if __name__=='__main__':
    res={}
    for name,fd in FOLDS.items():
        lo,hi=fd['rng']; xs=np.arange(lo,hi+0.01,2.0)
        rows=[]
        for x0 in xs:
            st=station(fd,x0)
            if st is None or not st['ok']: rows.append(dict(x=float(x0),ok=False,why=None if st is None else st.get('why'))); continue
            rows.append(dict(x=float(x0),ok=True,apex=st['apex'].tolist(),Xo=st['Xo'].tolist(),Xi=None if st['Xi'] is None else st['Xi'].tolist(),
                             apex_to_Xo=st['apex_to_Xo'],thick=(st['thick1'],st['thick2']),angle=st['raw_angle'],ob=st['ob'].tolist()))
        res[name]=rows
        okx=[r['x'] for r in rows if r['ok']]
        print(name,'ok stations',len(okx),'of',len(rows), 'x-range of ok:', (min(okx),max(okx)) if okx else None)
    json.dump(res,open('fold_stations.json','w'))
