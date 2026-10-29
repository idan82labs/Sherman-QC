import sys
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
fig,axs=plt.subplots(2,2,figsize=(18,14))
# top view u-v
ax=axs[0,0]; sel=np.random.default_rng(0).choice(len(P),150000,replace=False)
sc=ax.scatter(P[sel,0],P[sel,1],c=P[sel,2],s=0.2,cmap='viridis'); plt.colorbar(sc,ax=ax,label='w (along base normal)')
for k,(s,e) in gp.items():
    ax.plot([s[0],e[0]],[s[1],e[1]],'r-',lw=1.2); ax.text((s[0]+e[0])/2,(s[1]+e[1])/2,k,color='r',fontsize=9)
ax.set_aspect('equal'); ax.set_title('top view (u along M01..M06 axis, v along M07 axis), colour=w'); ax.set_xlabel('u'); ax.set_ylabel('v')
# side view v-w all points (looking along u)
ax=axs[0,1]; ax.scatter(P[sel,1],P[sel,2],s=0.1,c='k',alpha=0.3)
for k,(s,e) in gp.items():
    if k in('M07','M08'):
        ax.plot([s[1],e[1]],[s[2],e[2]],'r-',lw=1)
    else:
        ax.plot(s[1],s[2],'r*',ms=8); ax.text(s[1],s[2]+1.5,k,color='r',fontsize=9)
ax.set_aspect('equal'); ax.set_title('looking along u (profile across M01-M06), all points'); ax.set_xlabel('v'); ax.set_ylabel('w')
# front view u-w (looking along v)
ax=axs[1,0]; ax.scatter(P[sel,0],P[sel,2],s=0.1,c='k',alpha=0.3)
for k,(s,e) in gp.items():
    if k in('M07','M08'):
        ax.plot(s[0],s[2],'r*',ms=8); ax.text(s[0],s[2]+1.5,k,color='r',fontsize=9)
    else:
        ax.plot([s[0],e[0]],[s[2],e[2]],'r-',lw=0.8)
ax.set_aspect('equal'); ax.set_title('looking along v (profile across M07/M08), all points'); ax.set_xlabel('u'); ax.set_ylabel('w')
ax=axs[1,1]; ax.hist(P[:,2],bins=400); ax.set_yscale('log'); ax.set_title('w histogram')
plt.tight_layout(); plt.savefig(OUT+'a1_frame_views.png',dpi=90)
