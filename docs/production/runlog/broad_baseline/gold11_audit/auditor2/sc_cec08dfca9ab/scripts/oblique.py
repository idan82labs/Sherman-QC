from common import *
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sps
NF=N@R.T; NF/=np.linalg.norm(NF,axis=1,keepdims=True)+1e-12
mx=np.abs(NF).max(1)
obl=mx<np.cos(np.radians(25))   # normal >25 deg away from every axis
idx=np.where(obl)[0]
print("oblique vertices",len(idx),"of",len(VF))
# cluster spatially via mesh edges among oblique vertices
E=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
mp=-np.ones(len(VF),int); mp[idx]=np.arange(len(idx))
e=E[(mp[E[:,0]]>=0)&(mp[E[:,1]]>=0)]
g=sps.coo_matrix((np.ones(len(e)),(mp[e[:,0]],mp[e[:,1]])),shape=(len(idx),len(idx)))
nc,lab=connected_components(g,directed=False)
sizes=np.bincount(lab)
order=np.argsort(-sizes)
print("clusters",nc)
for c in order[:40]:
    P=VF[idx[lab==c]]
    if len(P)<30: break
    ext=P.max(0)-P.min(0)
    nm=NF[idx[lab==c]].mean(0)
    print("n %6d  X %7.1f..%7.1f  Y %7.1f..%7.1f  Z %6.1f..%6.1f  ext %s  mean normal %s"%(len(P),P[:,0].min(),P[:,0].max(),P[:,1].min(),P[:,1].max(),P[:,2].min(),P[:,2].max(),np.round(ext,1),np.round(nm,2)))
