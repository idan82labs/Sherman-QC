from common import *
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
E=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
def check(side, wall_out, inward, lip_top, ys):
    out=[]
    for y0 in ys:
        win=(np.abs(Y-y0)<1.0)&((X-wall_out)*inward>-1.0)&((X-wall_out)*inward<12)&(Z>lip_top-12)&(Z<lip_top+2.5)
        idx=np.where(win)[0]
        if len(idx)==0: out.append((y0,None)); continue
        mp=-np.ones(len(VF),int); mp[idx]=np.arange(len(idx))
        e=E[(mp[E[:,0]]>=0)&(mp[E[:,1]]>=0)]
        g=sps.coo_matrix((np.ones(len(e)),(mp[e[:,0]],mp[e[:,1]])),shape=(len(idx),len(idx)))
        nc,lab=connected_components(g,directed=False)
        dx=(X[idx]-wall_out)*inward; z=Z[idx]; y=Y[idx]
        wall=(np.abs(dx)<0.5)&(z>lip_top-10)&(z<lip_top-5)&(np.abs(y-y0)<0.5)
        lip=(dx>6)&(dx<10)&(z>lip_top-2.0)&(z<lip_top+1.5)&(np.abs(y-y0)<0.5)
        conn=bool(set(lab[wall])&set(lab[lip])) if wall.any() and lip.any() else False
        out.append((y0,conn,int(wall.sum()),int(lip.sum())))
    return out
for side,wo,inw,lt in (("LEFT",-158.2,1,28.55),("RIGHT",155.1,-1,28.5)):
    ys=np.arange(-90,118,1.0)
    r=check(side,wo,inw,lt,ys)
    conn=[a[0] for a in r if a[1]]
    segs=[]
    for v in conn:
        if segs and v-segs[-1][1]<=1.5: segs[-1][1]=v
        else: segs.append([v,v])
    print(side,"wall-lip connected segments:",segs)
    for a in r:
        if a[0] in (-56,-54,-53,-52,-51,-50,-49,76,78,79,80,81,82,83,84,86,90,95,100,105,110,113,115): print("   ",a)
