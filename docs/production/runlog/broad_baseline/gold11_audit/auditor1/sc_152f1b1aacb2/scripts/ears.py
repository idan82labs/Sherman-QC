import numpy as np
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
def plane_fit(X):
    c0=X.mean(0); U,S_,Wt=np.linalg.svd(X-c0); n=Wt[2]; 
    if n[2]<0: n=-n
    return c0,n
# base plate reference: upper skin of base near walls (e3>-6.8), |e2| in [60,85]
for sgn,name in ((-1,"left ears (e2<-94)"),(1,"right ears (e2>94)")):
    for lo,hi in ((-29,0),(26,55)):
        ear=(sgn*P[:,1]>94.5)&(sgn*P[:,1]<101)&(P[:,0]>lo+2)&(P[:,0]<hi-2)&(P[:,2]<-4)
        X=P[ear]
        # split skins by median e3
        med=np.median(X[:,2]); up=X[X[:,2]>med]
        base=(sgn*P[:,1]>70)&(sgn*P[:,1]<86)&(P[:,0]>lo+2)&(P[:,0]<hi-2)&(P[:,2]<-4)
        B=P[base]; bup=B[B[:,2]>np.median(B[:,2])]
        c1,n1=plane_fit(up); c2,n2=plane_fit(bup)
        ang=np.degrees(np.arccos(np.clip(n1@n2,-1,1)))
        # height difference of ear upper skin vs extrapolated base plane at ear centroid
        dz=(c1-c2)@n2
        print(name, (lo,hi), "n ear", len(up), "angle ear vs base %.2f deg"%ang, "offset %.2f mm"%dz, "ear e3 range %.2f..%.2f"%(X[:,2].min(),X[:,2].max()))
