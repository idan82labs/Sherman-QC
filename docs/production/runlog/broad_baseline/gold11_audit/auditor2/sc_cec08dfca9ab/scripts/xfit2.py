from common import *
# (p,q) coords; corner approx; concave signs; windows (horizontal flange window, vertical flange window) in mm from the outer corner
SPEC={
 "HB1":dict(s=0,p=1,q=2,corner=(117.7,-12.6),sp=-1,sq=+1,wh=(5,25),wv=(5,25)),
 "HB2":dict(s=0,p=1,q=2,corner=(117.7, 30.0),sp=-1,sq=-1,wh=(4.5,15),wv=(5,25)),
 "HB3":dict(s=1,p=0,q=2,corner=(155.2,-12.6),sp=-1,sq=+1,wh=(5,25),wv=(5,25)),
 "HB4":dict(s=1,p=0,q=2,corner=(-158.1,-12.4),sp=+1,sq=+1,wh=(5,25),wv=(5,25)),
 "HB5":dict(s=1,p=0,q=2,corner=(-158.2, 28.6),sp=+1,sq=-1,wh=(4.0,9.5),wv=(5,16)),
 "HB6":dict(s=1,p=0,q=2,corner=(155.1, 28.6),sp=-1,sq=-1,wh=(4.0,9.5),wv=(5,16)),
}
TRUE_EXT={"HB1":(-155.0,151.5),"HB2":(-155.0,151.5),"HB3":(-85.0,114.5),"HB4":(-85.5,114.0),"HB5":(-52.0,81.0),"HB6":(-52.0,81.0)}
def skins(u,w,expect_gap=(0.7,2.2)):
    # two skins via 1D 2-means on residual after a common line fit
    if len(w)<20: return None
    A=np.c_[np.ones(len(u)),u]
    c,*_=np.linalg.lstsq(A,w,rcond=None); r=w-A@c
    lab=r>0
    for _ in range(30):
        if lab.all() or (~lab).all(): return None
        m1,m2=r[~lab].mean(),r[lab].mean(); nl=np.abs(r-m1)>np.abs(r-m2)
        if (nl==lab).all(): break
        lab=nl
    out=[]
    for sel in (~lab,lab):
        if sel.sum()<6: return None
        Ai=A[sel]; wi=w[sel]
        ci,*_=np.linalg.lstsq(Ai,wi,rcond=None); ri=wi-Ai@ci
        k=np.abs(ri)<max(0.2,2.5*np.std(ri)); ci,*_=np.linalg.lstsq(Ai[k],wi[k],rcond=None)
        out.append((ci[0],ci[1],int(k.sum()),float(np.std(wi[k]-Ai[k]@ci))))
    return out
def circle_fit(P):
    A=np.c_[2*P[:,0],2*P[:,1],np.ones(len(P))]; b=(P**2).sum(1)
    c,*_=np.linalg.lstsq(A,b,rcond=None); cx,cy=c[0],c[1]; R=np.sqrt(c[2]+cx**2+cy**2)
    return np.array([cx,cy]),R
def station(bid,s0,half=0.6):
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]; cp,cq=sp["corner"]; spg,sqg=sp["sp"],sp["sq"]
    m=np.abs(VF[:,si]-s0)<half
    P=VF[m][:,[pi,qi]]
    res=dict(s=s0,ok=False)
    du=(P[:,0]-cp)*spg; dq=(P[:,1]-cq)*sqg
    hsel=(du>sp["wh"][0])&(du<sp["wh"][1])&(np.abs(P[:,1]-cq)<3.5)
    vsel=(dq>sp["wv"][0])&(dq<sp["wv"][1])&(np.abs(P[:,0]-cp)<3.5)
    H=skins(P[hsel,0],P[hsel,1]); Vv=skins(P[vsel,1],P[vsel,0])
    res["nh"],res["nv"]=int(hsel.sum()),int(vsel.sum())
    if H is None or Vv is None: return res,P
    Hs=sorted(H,key=lambda t:t[0]*sqg); Vs=sorted(Vv,key=lambda t:t[0]*spg)
    Ho,Hi=Hs; Vo,Vi=Vs
    def inter(h,v):
        a,b=h[0],h[1]; c,d=v[0],v[1]; p=(c+d*a)/(1-d*b); return np.array([p,a+b*p])
    Xo=inter(Ho,Vo); Xi=inter(Hi,Vi)
    th=np.array([1.0,Ho[1]]); th/=np.linalg.norm(th); th*=np.sign(th[0]*spg)
    tv=np.array([Vo[1],1.0]); tv/=np.linalg.norm(tv); tv*=np.sign(tv[1]*sqg)
    bis=th+tv; bis/=np.linalg.norm(bis); perp=np.array([-bis[1],bis[0]])
    inc=np.degrees(np.arccos(np.clip(th@tv,-1,1)))
    d=P-Xo; t=d@bis; c=d@perp
    # outer arc points: within the corner zone, outer side: distance to outer skin lines small and not on flats
    # select points with t in [0, 3.5] and |c| < 3.0 that are within 0.5 mm of the 'outer envelope'
    zone=(t>-0.5)&(t<4.0)&(np.abs(c)<3.2)
    res.update(ok=True,Xo=Xo,Xi=Xi,bis=bis,perp=perp,bend_angle=180-inc,thick=(abs(Hi[0]-Ho[0]),abs(Vi[0]-Vo[0])),skin_sd=(Ho[3],Hi[3],Vo[3],Vi[3]))
    near=zone&(np.abs(c)<0.35)
    if near.sum()<3: return res,P
    tt=np.sort(t[near]); t_o=np.median(tt[:max(1,min(5,len(tt)//5))])
    inner=tt[(tt>t_o+0.5)&(tt<t_o+3.0)]
    t_i=np.median(inner[-max(1,min(5,len(inner)//5)):]) if len(inner) else np.nan
    # circle fit to outer arc: points in zone whose t <= t_o + (outer envelope) ; approximate: for |c|<2.4, take min t per c-bin
    bins=np.linspace(-2.6,2.6,14); arc=[]
    for b0,b1 in zip(bins[:-1],bins[1:]):
        k=zone&(c>=b0)&(c<b1)
        if k.sum()>=2:
            j=np.where(k)[0][np.argmin(t[k])]; arc.append(P[j])
    arc=np.array(arc)
    C,Rf=(circle_fit(arc) if len(arc)>=5 else (None,np.nan))
    res.update(apex_ok=True,apex=Xo+t_o*bis,inner_apex=(Xo+t_i*bis) if np.isfinite(t_i) else None,t_o=t_o,t_i=t_i,arc_C=C,arc_R=Rf)
    return res,P
def gold_at(bid,s0):
    s,e=gold[bid]; S_,E_=tof(s),tof(e); si=SPEC[bid]["s"]
    f=(s0-S_[si])/(E_[si]-S_[si]); return S_+f*(E_-S_),f
