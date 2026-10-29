from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
vs=[-276,-265,-250,-200,-150,-95,-70,-56]
fig,axs=plt.subplots(len(vs),2,figsize=(22,4.0*len(vs)),gridspec_kw={'width_ratios':[1,3.2]})
for row,v0 in zip(axs,vs):
    m=np.abs(P[:,1]-v0)<0.75
    for ax,(ulo,uhi) in zip(row,[(-490,-460),(-300,-205)]):
        mm=m&(P[:,0]>ulo)&(P[:,0]<uhi)
        ax.scatter(P[mm,0],P[mm,2],s=1.5,c='k')
        for k,(s,e) in gp.items():
            if k not in('M07','M08'):
                # these run along u: show where the line is at this v? only if v near
                if abs(s[1]-v0)<20:
                    ax.plot([max(s[0],ulo),min(e[0],uhi)],[s[2],e[2]],'r-',lw=0.8); 
                continue
            lo,hi=sorted([s[1],e[1]])
            col='r' if lo<=v0<=hi else 'orange'
            ax.plot(s[0],s[2],'*',color=col,ms=10); ax.text(s[0]+0.5,s[2]+1,k,color=col,fontsize=9)
        ax.set_aspect('equal'); ax.set_xlim(ulo,uhi); ax.set_ylim(-132,-48); ax.grid(alpha=0.3)
        ax.set_title(f'slab |v-({v0})|<0.75, profile u-w, u in [{ulo},{uhi}]')
plt.tight_layout(); plt.savefig(OUT+'a1_slabs_v_profiles.png',dpi=65)
