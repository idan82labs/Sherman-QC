import json, sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
jobs=json.load(open(sys.argv[1])); outdir=sys.argv[2]; cache={}
for j in jobs:
    if j['scan'] not in cache:
        cache.clear(); m=o3d.io.read_triangle_mesh(j['scan']); V=np.asarray(m.vertices).copy(); cache[j['scan']]=(V,cKDTree(V))
    V,tree=cache[j['scan']]
    L=json.load(open(j['lines_json']))
    for b in L['bends']:
        if b['bend_id'] not in j['bend_ids']: continue
        s=np.array(b['sharp']['start']); e=np.array(b['sharp']['end']); d=(e-s)/np.linalg.norm(e-s); Lm=np.linalg.norm(e-s)
        A=np.array(b['side_dirs']['plane_a']); B=np.array(b['side_dirs']['plane_b'])
        bi=np.array(b['bisector']); p=np.cross(d,bi)
        fig,axs=plt.subplots(1,4,figsize=(24,6))
        for ax,fr in zip(axs[:3],(0.25,0.5,0.75)):
            c=s+fr*Lm*d; idx=tree.query_ball_point(c,20); P=V[idx]-c
            k=np.abs(P@d)<0.4; Q=P[k]
            ax.scatter(Q@p,Q@bi,s=2,c='k'); ax.plot([0],[0],'rx'); 
            for v,col in ((A,'b'),(B,'g')): ax.plot([0,15*v@p],[0,15*v@bi],col+'--')
            ap=b['apex_offset_mm']; ax.plot([0],[ap],'ro')
            ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f"{j['name']} {b['bend_id']} t={fr*Lm:.1f}/{Lm:.1f} (x=perp, y=bisector into material)")
        # face-on: points within 8 mm, coloured by bisector height
        c=s+0.5*Lm*d; idx=tree.query_ball_point(c,Lm/2+15); P=V[idx]-c
        n=np.cross(d,A); n/=np.linalg.norm(n)   # plane_a normal-ish
        ax=axs[3]; k=np.abs(P@n)<6; ax.scatter(P[k]@d,P[k]@A,s=1,c=P[k]@n,cmap='coolwarm'); ax.plot([-Lm/2,Lm/2],[0,0],'k-')
        ax.set_aspect('equal'); ax.set_title('face-on in plane_a (x=line dir, y=side a dir), colour=normal offset')
        fig.savefig(f"{outdir}/{j['name']}_{b['bend_id']}.png",dpi=60); plt.close(fig)
        print('ok',j['name'],b['bend_id'])
