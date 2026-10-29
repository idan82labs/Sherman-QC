import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage/triage_f247.py').read().split('# top & two')[0])
m=(np.abs(Q[:,0])<60)&(np.abs(Q[:,1])<60)
A=np.c_[Q[m,0],Q[m,1],np.ones(m.sum())]; co,*_=np.linalg.lstsq(A,Q[m,2],rcond=None)
h=Q[:,2]-(co[0]*Q[:,0]+co[1]*Q[:,1]+co[2]); print('plane',co)
np.save(S+'h.npy',h)
fig,ax=plt.subplots(figsize=(14,12))
o=np.argsort(h)
sc=ax.scatter(Q[o,0],Q[o,1],s=0.5,c=np.clip(h[o],-20,20),cmap='coolwarm')
plt.colorbar(sc,label='height above plate plane (mm)')
for k,(s0,e0) in G.items():
    s1,e1=T(s0),T(e0); ax.plot([s1[0],e1[0]],[s1[1],e1[1]],'g-',lw=2,alpha=.6); ax.text(*(0.5*(s1+e1))[:2]+[0,2],k,color='g',fontsize=12)
for k,(s0,e0) in B.items():
    s1,e1=T(s0),T(e0); ax.plot([s1[0],e1[0]],[s1[1],e1[1]],'k--',lw=1.2); ax.text(*(0.3*s1+0.7*e1)[:2]+[0,-4],k[-4:],color='k',fontsize=10)
ax.set_aspect('equal'); ax.set_title('sc_f2475286312b top view, colour = height vs central plate plane; green=gold, black dashed=best_f1')
plt.tight_layout(); plt.savefig(OUT+'triage_heightmap.png',dpi=60)
