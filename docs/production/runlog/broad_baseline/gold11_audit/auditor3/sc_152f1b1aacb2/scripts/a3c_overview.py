import numpy as np, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from a3c_common import *
V,F=load_mesh(); R,o=part_frame(V); X=to_part(V,R,o)
G=load_gold()
GL={bid:(to_part(s,R,o),to_part(e,R,o)) for bid,s,e in G}
rng=np.random.default_rng(0)
sub=X[rng.choice(len(X),200000,replace=False)]
fig,axes=plt.subplots(2,2,figsize=(18,15),dpi=90)
# top view colored by e3
ax=axes[0,0]
o_=np.argsort(sub[:,2])
sc=ax.scatter(sub[o_,0],sub[o_,1],c=sub[o_,2],s=0.3,cmap='viridis',vmin=-1,vmax=45,rasterized=True)
plt.colorbar(sc,ax=ax,label='e3 (mm above base outer)')
for bid,(a,b) in GL.items():
    ax.plot([a[0],b[0]],[a[1],b[1]],'r-',lw=2); ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+2,bid,color='r',fontsize=10,weight='bold')
ax.set_aspect('equal'); ax.set_title('top e1-e2, colour=e3'); ax.grid(alpha=.3)
# side views
for ax,(i,j,nm,cut) in zip([axes[0,1],axes[1,0],axes[1,1]],[(1,2,'end view e2-e3 (all)',None),(0,2,'wall e2>+80, e1-e3',('gt',80)),(0,2,'wall e2<-80, e1-e3',('lt',-80))]):
    S=sub
    if cut:
        S=sub[sub[:,1]>cut[1]] if cut[0]=='gt' else sub[sub[:,1]<cut[1]]
    k=3-i-j
    o_=np.argsort(S[:,k])
    ax.scatter(S[o_,i],S[o_,j],c=S[o_,k],s=0.3,cmap='coolwarm',rasterized=True)
    for bid,(a,b) in GL.items():
        if cut and ((cut[0]=='gt' and a[1]<0) or (cut[0]=='lt' and a[1]>0)): continue
        ax.plot([a[i],b[i]],[a[j],b[j]],'r-',lw=2); ax.text((a[i]+b[i])/2,(a[j]+b[j])/2,bid,color='r',fontsize=10,weight='bold')
    ax.set_aspect('equal'); ax.set_title(nm); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/'a3c_overview_partframe.png'); print('ok')
