from common import *
from scipy.spatial import cKDTree
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
# bend spec: station axis index, (p,q) axis indices, approx outer corner (p,q), concave signs (sp,sq)
SPEC={
 "HB1":dict(s=0,p=1,q=2,corner=(117.7,-12.6),sp=-1,sq=+1),
 "HB2":dict(s=0,p=1,q=2,corner=(117.7, 30.0),sp=-1,sq=-1),
 "HB3":dict(s=1,p=0,q=2,corner=(155.2,-12.6),sp=-1,sq=+1),
 "HB4":dict(s=1,p=0,q=2,corner=(-158.1,-12.4),sp=+1,sq=+1),
 "HB5":dict(s=1,p=0,q=2,corner=(-158.1, 28.6),sp=+1,sq=-1),
 "HB6":dict(s=1,p=0,q=2,corner=(155.2, 28.7),sp=-1,sq=-1),
}
def two_skins(u,w):
    """u: coordinate along flange, w: coordinate along flange normal. split into 2 skins by w gap; return list of (a,b,n) for w=a+b*u sorted by mean w"""
    if len(w)<20: return None
    ws=np.sort(w); 
    # 2-means 1D
    c1,c2=np.percentile(w,15),np.percentile(w,85)
    for _ in range(20):
        lab=np.abs(w-c1)>np.abs(w-c2)
        if lab.all() or (~lab).all(): break
        c1n,c2n=w[~lab].mean(),w[lab].mean()
        if abs(c1n-c1)<1e-6 and abs(c2n-c2)<1e-6: break
        c1,c2=c1n,c2n
    out=[]
    for sel in (~lab,lab):
        if sel.sum()<8: out.append(None); continue
        A=np.c_[np.ones(sel.sum()),u[sel]]
        coef,*_=np.linalg.lstsq(A,w[sel],rcond=None)
        r=w[sel]-A@coef
        # one robust refit
        k=np.abs(r)<max(0.25,2.5*np.std(r))
        coef,*_=np.linalg.lstsq(A[k],w[sel][k],rcond=None)
        out.append((coef[0],coef[1],int(k.sum()),float(np.std(w[sel][k]-A[k]@coef))))
    return out
def station(bid,s0,half=0.6,win=(4.5,22.0),verbose=False):
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]; cp,cq=sp["corner"]; spg,sqg=sp["sp"],sp["sq"]
    m=np.abs(VF[:,si]-s0)<half
    P=VF[m][:,[pi,qi]]
    res=dict(s=s0,n=int(m.sum()))
    # horizontal flange (normal q): extends from corner in spg*p direction
    du=(P[:,0]-cp)*spg; dq=(P[:,1]-cq)*sqg
    hsel=(du>win[0])&(du<win[1])&(np.abs(P[:,1]-cq)<4.0)
    vsel=(dq>win[0])&(dq<win[1])&(np.abs(P[:,0]-cp)<4.0)
    res["nh"]=int(hsel.sum()); res["nv"]=int(vsel.sum())
    H=two_skins(P[hsel,0],P[hsel,1]) if hsel.sum()>=20 else None
    Vv=two_skins(P[vsel,1],P[vsel,0]) if vsel.sum()>=20 else None
    if H is None or Vv is None or None in H or None in Vv:
        res["ok"]=False; return res,P
    # outer skin: horizontal flange -> the skin with w extreme in -sqg direction (away from concave)
    Hs=sorted(H,key=lambda t:t[0]*sqg)  # first = smallest along concave => outer
    Vs=sorted(Vv,key=lambda t:t[0]*spg)
    Ho,Hi=Hs[0],Hs[1]; Vo,Vi=Vs[0],Vs[1]
    # check: skins approx at same p for intercept: q=a+b p ; p=c+d q
    def inter(h,v):
        a,b=h[0],h[1]; c,d=v[0],v[1]
        # q=a+b p ; p=c+d q -> p=c+d(a+b p) -> p(1-db)=c+da
        p=(c+d*a)/(1-d*b); q=a+b*p; return np.array([p,q])
    Xo=inter(Ho,Vo); Xi=inter(Hi,Vi)
    # flange unit directions pointing away from corner
    th=np.array([1.0,Ho[1]]); th/=np.linalg.norm(th); th*=np.sign(th[0]*spg)
    tv=np.array([Vo[1],1.0]); tv/=np.linalg.norm(tv); tv*=np.sign(tv[1]*sqg)
    bis=th+tv; bis/=np.linalg.norm(bis)
    ang=np.degrees(np.arccos(np.clip(th@tv,-1,1)))  # included angle between flanges
    bend_angle=180-ang
    perp=np.array([-bis[1],bis[0]])
    d=P-Xo; t=d@bis; c=d@perp
    near=(np.abs(c)<0.35)&(t>-2)&(t<6)
    res.update(ok=True,Xo=Xo,Xi=Xi,bis=bis,perp=perp,bend_angle=bend_angle,
               thick_h=abs(Hi[0]-Ho[0])*np.cos(np.arctan(Ho[1])),thick_v=abs(Vi[0]-Vo[0])*np.cos(np.arctan(Vo[1])),
               skin_sd=(Ho[3],Hi[3],Vo[3],Vi[3]))
    if near.sum()<3:
        res["apex_ok"]=False; return res,P
    tt=np.sort(t[near])
    t_o=np.median(tt[:max(1,len(tt)//10)]) if len(tt)>=10 else tt[0]
    inner=tt[(tt>t_o+0.4)&(tt<t_o+3.5)]
    t_i=np.median(inner[-max(1,len(inner)//10):]) if len(inner)>0 else np.nan
    res.update(apex_ok=True,apex=Xo+t_o*bis,t_apex=t_o,t_inner=t_i,n_near=int(near.sum()))
    # arc radius estimate: distance from outer sharp to apex = R(sqrt2-1)/... general: d = R(1/sin(phi/2)-1) where phi included angle
    half_inc=np.radians(ang/2)
    res["R_outer_est"]=t_o/(1/np.sin(half_inc)-1) if t_o>0 else np.nan
    return res,P
def gold_at(bid,s0):
    s,e=gold[bid]; S_,E_=tof(s),tof(e); si=SPEC[bid]["s"]
    f=(s0-S_[si])/(E_[si]-S_[si]); g=S_+f*(E_-S_)
    return g,f
