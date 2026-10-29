import numpy as np
from scipy.spatial import cKDTree
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
N=np.load(S+"N.npy"); Np=N@R.T
Np/=np.linalg.norm(Np,axis=1,keepdims=True)+1e-12
rng=np.random.default_rng(2); sel=rng.choice(len(P),120000,replace=False)
X=P[sel]; NX=Np[sel]
tree=cKDTree(P)
flat=np.zeros(len(X),bool)
for i in range(len(X)):
    idx=tree.query_ball_point(X[i],1.5)
    nn=Np[idx]
    # keep same-skin neighbours (normal within 60 deg of centre normal)
    d=nn@NX[i]; nn=nn[d>0.5]
    if len(nn)<8: continue
    m=nn.mean(0); m/=np.linalg.norm(m)
    dev=np.degrees(np.arccos(np.clip(nn@m,-1,1)))
    flat[i]= np.percentile(dev,90)<6
al=np.abs(NX).max(1)
obl=flat&(al<np.cos(np.radians(8)))
print("sampled",len(X),"flat",flat.sum(),"flat & >8deg off every axis",obl.sum())
if obl.sum():
    from sklearn.cluster import DBSCAN
    Z=X[obl]; lab=DBSCAN(eps=2.5,min_samples=6).fit_predict(Z)
    for l in sorted(set(lab)):
        m=lab==l
        tag="noise" if l<0 else "cluster %d"%l
        if l<0: print(tag,m.sum()); continue
        mn=NX[obl][m].mean(0); mn/=np.linalg.norm(mn)
        print(tag,"n",m.sum(),"e1 %.1f..%.1f e2 %.1f..%.1f e3 %.1f..%.1f"%(Z[m,0].min(),Z[m,0].max(),Z[m,1].min(),Z[m,1].max(),Z[m,2].min(),Z[m,2].max()),"normal",np.round(mn,2))
