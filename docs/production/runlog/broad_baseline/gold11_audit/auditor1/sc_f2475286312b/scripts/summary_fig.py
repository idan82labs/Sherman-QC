import json
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V,F,N=load_scan(); g,_=load_gold(); R=np.load('R.npy'); P=V@R.T
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
L=json.load(open('apex_lines.json'))
VERD={'M01':('CONVENTION_OFFSET*','orange'),'M02':('CONVENTION_OFFSET*','orange'),'M07':('CONVENTION_OFFSET*','orange'),
      'M03':('REAL_ERROR','red'),'M04':('REAL_ERROR','red'),'M05':('REAL_ERROR','red'),'M06':('REAL_ERROR','red'),'M08':('NOT_A_BEND','magenta')}
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
sel=np.random.default_rng(1).choice(len(P),120000,replace=False)
fig,axs=plt.subplots(1,2,figsize=(24,10),gridspec_kw={'width_ratios':[1.35,1]})
for ax,(i,j,lab) in zip(axs,[(0,1,'plan u-v (looking along base normal)'),(0,2,'side u-w (looking along v)')]):
    ax.scatter(P[sel,i],P[sel,j],s=0.15,c='0.75')
    for name,l in L.items():
        s=np.array(l['start_part']); e=np.array(l['end_part'])
        if np.hypot(*(e-s)[[i,j]])<3:
            ax.plot(s[i],s[j],'o',ms=14,mfc='none',mec='c',mew=2.5)
        else:
            ax.plot([s[i],e[i]],[s[j],e[j]],'-',color='c',lw=5,alpha=0.6)
        ax.text((s[i]+e[i])/2,(s[j]+e[j])/2-(4 if j==1 else 2.5),name.split('_')[0],color='teal',fontsize=11,weight='bold')
    for k,(s,e) in gp.items():
        vt,col=VERD[k]
        ax.plot([s[i],e[i]],[s[j],e[j]],'-',color=col,lw=1.6)
        ax.text(s[i]+(3 if i==0 else 0),s[j]+(1.5 if j==1 else 1.0),f'{k} {vt}',color=col,fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.set_title(lab); ax.set_xlabel('u'); ax.set_ylabel('v' if j==1 else 'w')
axs[0].set_ylim(-290,-45); axs[1].set_ylim(-135,-48)
fig.suptitle('sc_f2475286312b auditor1 summary: cyan = scan-derived outer-apex lines of the 8 formed folds (F1..F8; F7/F8 ears are vertical -> circles in plan); gold coloured by verdict\n'
             '(*CONVENTION_OFFSET here = systematic 4.6-5.1 mm concave free-air locus copied from F1 region lines, NOT a virtual-sharp/inner-skin convention)',fontsize=11)
plt.tight_layout(); plt.savefig(OUT+'a1_summary_folds_vs_gold.png',dpi=70)
