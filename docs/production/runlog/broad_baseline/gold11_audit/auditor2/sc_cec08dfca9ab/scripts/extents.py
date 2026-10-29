from xfit import *
import json
st=json.load(open("stations.json"))
res={}
for bid in ["HB1","HB2","HB3","HB4","HB5","HB6"]:
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]
    rows=[r for r in st[bid] if r.get("d_apex") is not None and 80<r["angle"]<100]
    ap=np.array([r["apex"] for r in rows]); ss=np.array([r["s"] for r in rows])
    # robust central apex position (p,q) as a function of s: linear fit on central part
    A=np.c_[np.ones(len(ss)),ss]
    cp,*_=np.linalg.lstsq(A,ap[:,0],rcond=None); cq,*_=np.linalg.lstsq(A,ap[:,1],rcond=None)
    # concave bisector approx
    bis=np.array([sp["sp"],sp["sq"]])/np.sqrt(2)
    # scan along s at 0.5 mm: count points within 1.2 mm of apex line and on the arc (arc = within 3 mm of apex, and 'between' flanges)
    s_all=VF[:,si]; lo=ss.min()-60; hi=ss.max()+60
    out=[]
    for s0 in np.arange(lo,hi,0.5):
        m=np.abs(s_all-s0)<0.25
        if not m.any(): out.append((s0,0,0)); continue
        P=VF[m][:,[pi,qi]]
        apx=np.array([cp[0]+cp[1]*s0,cq[0]+cq[1]*s0])
        d=P-apx
        n_apex=int((np.linalg.norm(d,axis=1)<1.0).sum())   # points at the outer apex
        # flange presence near the corner: points 4-10 mm from the apex along each flange direction, within 2 mm of the flange plane
        fa=((d[:,0]*sp["sp"]>4)&(d[:,0]*sp["sp"]<10)&(np.abs(d[:,1])<2.5)).sum()
        fb=((d[:,1]*sp["sq"]>4)&(d[:,1]*sp["sq"]<10)&(np.abs(d[:,0])<2.5)).sum()
        out.append((s0,n_apex,int(fa),int(fb)))
    # contiguous extent where n_apex>0 and both flanges present
    ok=[o[0] for o in out if len(o)==4 and o[1]>0 and o[2]>0 and o[3]>0]
    s,e=gold[bid]; S_,E_=tof(s),tof(e)
    glo,ghi=sorted([S_[si],E_[si]])
    # segments
    segs=[]; 
    for v in ok:
        if segs and v-segs[-1][1]<=2.6: segs[-1][1]=v
        else: segs.append([v,v])
    res[bid]=dict(gold=[glo,ghi],bend_segments=segs,apex_line=[list(cp),list(cq)])
    print(bid,"gold s-range [%.1f, %.1f] len %.1f"%(glo,ghi,ghi-glo),"bend segments:",[(round(a,1),round(b,1)) for a,b in segs])
json.dump(res,open("extents.json","w"),indent=1)
