import sys, json; sys.path.insert(0,'.')
from a1s70_load import *
import numpy as np, matplotlib, trimesh
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open(OUT+'a1_bendfit_bins.json'))
m=load()
STAT=(0.02,0.25,0.5,0.75,0.98)
fig,axs=plt.subplots(3,5,figsize=(30,18))
summary={}
for r,(k,(p,q)) in enumerate(GOLD.items()):
    d=unit(q-p); L=np.linalg.norm(q-p)
    bis2=np.array(R[k]['bisector']) if 'bisector' in R[k] else None
    summary[k]=[]
    for c,t in enumerate(STAT):
        P0=p+t*(q-p)
        segs=trimesh.intersections.mesh_plane(m,d,P0)   # (n,2,3)
        # local 2D frame: u = outer-bisector direction in 3D (convex side), w = d x u
        a=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_s70_axis.npy')
        e1=unit(np.cross(a,[0,0,1])); e2=np.cross(a,e1)
        th=np.radians(np.mean(R[k]['outer_normal_angles']))
        u3=np.cos(th)*e1+np.sin(th)*e2; u3=unit(u3-d*(u3@d)); w3=np.cross(d,u3)
        A=segs[:,0,:]-P0; B=segs[:,1,:]-P0
        ax=axs[r,c]
        for s0,s1 in zip(A,B):
            ax.plot([s0@w3,s1@w3],[s0@u3,s1@u3],'k-',lw=1.0)
        # nearest point of section polyline to gold point (0,0)
        pts=np.concatenate([A,B]); dd=np.linalg.norm(pts,axis=1)
        # find outer apex in this exact section: section point maximizing (x . u3) within 6 mm of gold
        near=dd<7
        PP=pts[near]; 
        if len(PP):
            iapex=np.argmax(PP@u3); apex=PP[iapex]
            # inner skin midpoint: points with small |w| on the concave side: choose point near the line w=apex.w with smallest u among skin points around gold
            # (approx) the nearest section point to gold along u direction
            ax.plot(apex@w3,apex@u3,'D',color='cyan',mec='k',ms=8,label='outer apex (max along bisector)')
            dist_apex=np.linalg.norm(apex); along=apex@u3; across=apex@w3
        else:
            dist_apex=np.nan; along=np.nan; across=np.nan
        dsurf=dd.min() if len(dd) else np.nan
        ax.plot(0,0,'*',color='#1a9c3a',ms=16,mec='k',label='gold')
        ax.add_patch(plt.Circle((0,0),4,fill=False,ls=':',color='r'))
        ax.set_xlim(-10,10); ax.set_ylim(-8,8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title('%s @ %d%% (s=%.1f mm of %.1f)\ngold->outer apex %.2f mm (along bisector %.2f, across %.2f)\ngold->nearest section pt %.2f mm'%(k,round(t*100),t*L,L,dist_apex,along,across,dsurf),fontsize=9)
        if c==0: ax.legend(fontsize=7,loc='lower left')
        summary[k].append(dict(t=t,d_apex=float(dist_apex),along=float(along),across=float(across),d_surface=float(dsurf)))
        ax.set_xlabel('across (mm)'); ax.set_ylabel('along outer bisector (mm), + = convex/outer side')
fig.suptitle('sc_70b39f530f00 auditor1: EXACT mesh-plane cross-sections perpendicular to each gold line (gold at origin; up = convex/outer side; red dotted = 4 mm)',fontsize=13)
plt.tight_layout(); plt.savefig(OUT+'a1_xsec_exact_5stations.png',dpi=65)
print(json.dumps(summary,indent=0))
