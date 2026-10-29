import json, sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
jobs={j['name']:j for j in json.load(open(sys.argv[1]))}; H=json.load(open(sys.argv[2])); outdir=sys.argv[3]
cache={}
for name,h in H.items():
    j=jobs[name]
    if j['scan'] not in cache:
        cache.clear(); m=o3d.io.read_triangle_mesh(j['scan']); V=np.asarray(m.vertices).copy(); cache[j['scan']]=(V,cKDTree(V))
    V,tree=cache[j['scan']]
    p0=np.array(h['p0']); a=np.array(h['axis']); e=np.array(h['e_out']); f=np.array(h['f']); L=np.linalg.norm(np.array(h['p1'])-p0)
    fig,axs=plt.subplots(1,3,figsize=(18,6))
    for ax,fr in zip(axs,(0.25,0.5,0.75)):
        c=p0+fr*L*a; idx=tree.query_ball_point(c,14); P=V[idx]-p0
        k=np.abs(P@a-fr*L)<0.3; Q=P[k]
        ax.scatter(Q@e,Q@f,s=2,c='k'); ax.set_aspect('equal'); ax.grid(True,alpha=.3)
        ax.set_title(f'{name} t={fr*L:.1f} (e=out, f)')
    fig.savefig(f'{outdir}/{name}.png',dpi=70); plt.close(fig)
    print(name,'p0',np.round(p0,2).tolist(),'a',np.round(a,4).tolist(),'e',np.round(e,4).tolist(),'f',np.round(f,4).tolist(),'L',round(L,1))
