from xfit import *
import json
NF=N@R.T
NF/=np.linalg.norm(NF,axis=1,keepdims=True)+1e-12
ext=json.load(open("extents.json"))
res={}
for bid in ["HB1","HB2","HB3","HB4","HB5","HB6"]:
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]
    cp,cq=ext[bid]["apex_line"]
    outb=-np.array([sp["sp"],sp["sq"]],float)/np.sqrt(2)   # outward bisector (convex side)
    s_all=VF[:,si]
    glo,ghi=ext[bid]["gold"]
    rows=[]
    for s0 in np.arange(min(glo,-170)-5,max(ghi,125)+5,0.5):
        m=np.abs(s_all-s0)<0.25
        if not m.any(): rows.append((s0,0)); continue
        P=VF[m][:,[pi,qi]]; Nn=NF[m][:,[pi,qi]]
        apx=np.array([cp[0]+cp[1]*s0,cq[0]+cq[1]*s0])
        d=np.linalg.norm(P-apx,axis=1)
        nn=Nn/ (np.linalg.norm(Nn,axis=1,keepdims=True)+1e-12)
        cosang=np.abs(nn@outb)   # abs: normals may be flipped
        k=int(((d<1.2)&(cosang>np.cos(np.radians(25)))).sum())
        rows.append((s0,k))
    ok=[r[0] for r in rows if r[1]>0]
    segs=[]
    for v in ok:
        if segs and v-segs[-1][1]<=3.1: segs[-1][1]=v
        else: segs.append([v,v])
    res[bid]=dict(gold=[glo,ghi],arc_segments=segs)
    tot=sum(b-a for a,b in segs)
    print(bid,"gold [%.1f, %.1f]"%(glo,ghi),"arc segments (normal-on-bisector test):",[(round(a,1),round(b,1)) for a,b in segs],"total %.1f"%tot)
json.dump(res,open("extents2.json","w"),indent=1)
