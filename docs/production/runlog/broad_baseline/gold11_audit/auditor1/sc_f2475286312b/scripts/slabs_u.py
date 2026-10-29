from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
us=[-480,-460,-420,-380,-350,-330,-300,-270,-240,-228]
fig,axs=plt.subplots(len(us),1,figsize=(20,4.2*len(us)))
for ax,u0 in zip(axs,us):
    m=np.abs(P[:,0]-u0)<0.75
    ax.scatter(P[m,1],P[m,2],s=1,c='k')
    for k,(s,e) in gp.items():
        if k in('M07','M08'): continue
        lo,hi=sorted([s[0],e[0]])
        inside = lo<=u0<=hi
        ax.plot(s[1],s[2],'*',color='r' if inside else 'orange',ms=10); ax.text(s[1]+0.5,s[2]+1,k,color='r',fontsize=9)
    ax.set_aspect('equal'); ax.set_title(f'slab |u-({u0})|<0.75 mm, profile v-w'); ax.set_ylim(-132,-48); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT+'a1_slabs_u_profiles.png',dpi=70)
