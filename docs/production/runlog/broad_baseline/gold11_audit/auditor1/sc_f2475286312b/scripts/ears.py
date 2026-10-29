from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T; u,v,w=P.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
fig,axs=plt.subplots(2,4,figsize=(26,13))
# (u,v) slabs at several w through the wall/ears, near v=-54 corner and v=-278 corner
for col,w0 in enumerate([-60,-75,-95,-120]):
    for row,(vlo,vhi) in enumerate([(-70,-48),(-284,-262)]):
        ax=axs[row,col]
        m=(np.abs(w-w0)<0.5)&(u>-245)&(v>vlo)&(v<vhi)
        ax.scatter(u[m],v[m],s=3,c='k')
        for k,(s,e) in gp.items():
            # where does gold line cross this w plane? for lines along u/v they are at fixed w
            if abs(s[2]-w0)<6 or abs(e[2]-w0)<6:
                ax.plot([s[0],e[0]],[s[1],e[1]],'r-',lw=1); ax.text(np.clip((s[0]+e[0])/2,-244,-206),np.clip((s[1]+e[1])/2,vlo,vhi),k,color='r')
        ax.set_aspect('equal'); ax.set_xlim(-245,-205); ax.set_ylim(vlo,vhi); ax.grid(alpha=0.3)
        ax.set_title(f'plan slab |w-({w0})|<0.5, u-v (right end, corner v~{(vlo+vhi)/2:.0f})')
plt.tight_layout(); plt.savefig(OUT+'a1_ears_plan_slabs.png',dpi=65)
