import numpy as np
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/"
P = np.load(S+"P.npy")
def runs(mask, xs):
    out=[]; start=None
    for x,m in zip(xs,mask):
        if m and start is None: start=x
        if not m and start is not None: out.append((start,prev)); start=None
        prev=x
    if start is not None: out.append((start,prev))
    return out
xs = np.arange(-112, 113, 0.5)
for side,sgn in (("LEFT(e2<0)",-1),("RIGHT(e2>0)",1)):
    e2abs = sgn*P[:,1]
    # wall at e2abs in [88,95]
    wallmid = (e2abs>88)&(e2abs<95)&(P[:,2]>8)&(P[:,2]<28)
    wallbot = (e2abs>88)&(e2abs<95)&(P[:,2]>-4.0)&(P[:,2]<0)
    walltop = (e2abs>88)&(e2abs<95)&(P[:,2]>30)&(P[:,2]<34)
    beyond_base = (e2abs>94)&(P[:,2]<-5)   # flat ears beyond wall line at base level
    topflange = (e2abs>75)&(e2abs<89)&(P[:,2]>34)
    for name,m in (("wall mid-height e3 8..28",wallmid),("wall bottom e3 -4..0",wallbot),("wall top e3 30..34",walltop),("ears beyond e2|>94 at base",beyond_base),("top flange e3>34 inboard",topflange)):
        h,_=np.histogram(P[m,0], bins=np.append(xs-0.25, xs[-1]+0.25))
        r=runs(h>=3, xs)
        print(side, name, [(round(a,1),round(b,1)) for a,b in r if b-a>=1.0])
# base plate e1 extent at |e2|<40
bp = (np.abs(P[:,1])<40)&(P[:,2]<-5)
print("base plate e1 extent (|e2|<40):", P[bp,0].min(), P[bp,0].max())
bp2 = (P[:,2]<-5)
print("base plate e2 extent:", P[bp2,1].min(), P[bp2,1].max())
# end tabs: points with e3>-4 and |e2|<80 and |e1|>95
et = (P[:,2]>-4.5)&(np.abs(P[:,1])<80)&(np.abs(P[:,0])>95)
for sgn in (-1,1):
    m = et & (sgn*P[:,0]>0)
    if m.sum():
        print("end tab", sgn, "n", m.sum(), "e1", P[m,0].min(), P[m,0].max(), "e2", P[m,1].min(), P[m,1].max(), "e3", P[m,2].min(), P[m,2].max())
