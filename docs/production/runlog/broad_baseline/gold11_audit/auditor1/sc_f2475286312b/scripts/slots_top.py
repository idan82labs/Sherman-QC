from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy')
P=V@R.T; u,v,w=P.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
fig,axs=plt.subplots(2,1,figsize=(22,13))
for ax,(vlo,vhi) in zip(axs,[(-115,-50),(-282,-232)]):
    m=(v>vlo)&(v<vhi)
    sc=ax.scatter(u[m],v[m],c=w[m],s=0.6,cmap='turbo',vmin=-82,vmax=-52); plt.colorbar(sc,ax=ax,label='w')
    for k,(s,e) in gp.items():
        if k in('M07','M08'):
            ax.plot([s[0],e[0]],[s[1],e[1]],'m-',lw=1.5); ax.text(s[0],np.clip(s[1],vlo,vhi),k,color='m')
            continue
        if vlo<s[1]<vhi:
            ax.plot([s[0],e[0]],[s[1],e[1]],'m-',lw=1.5); ax.text(s[0]+3,s[1]+0.8,k+f' (w={s[2]:.1f})',color='m',fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(-492,-205); ax.set_ylim(vlo,vhi); ax.grid(alpha=0.3)
    ax.set_title(f'plan (u-v) colour = w; gold in magenta; v in [{vlo},{vhi}]')
plt.tight_layout(); plt.savefig(OUT+'a1_slots_plan.png',dpi=70)
