import sys, numpy as np, json, math
sys.path.insert(0,'/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714')
import xs
from xs import Tube, fit_section, offsets
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
W=xs.W
ideal=json.load(open(W+'ideal_apex_lines_raw.json'))
F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
panels=[('B01',0.5),('B02',0.25),('B03',0.5),('B04',0.5),('B05',0.5),('B07',0.1),('B07',0.5),('B07',0.9),('B09',0.5),('B10',0.5),('B06',0.5),('B08',0.5)]
fig,axs=plt.subplots(3,4,figsize=(28,21))
for ax,(bid,f) in zip(axs.flat,panels):
    A,B=xs.GOLD[bid]; T=Tube(A,B,rw=30,ext=5)
    uv,nuv,na=T.slab(f*T.L,0.5)
    m=np.linalg.norm(uv,axis=1)<14
    ax.scatter(uv[m,0],uv[m,1],s=3,c='0.35')
    ax.plot(0,0,'*',ms=20,mfc='lime',mec='k',label=f'gold {bid}')
    # ideal apex line intersection with this slab plane
    key='A_'+bid
    if key in ideal:
        P0,P1=np.array(ideal[key]); d=P1-P0; g3=A+(B-A)*f
        t=((g3-P0)@T.a)/(d@T.a); pi=P0+t*d-g3
        ai=np.array([pi@T.e1,pi@T.e2])
        ax.plot(ai[0],ai[1],'D',ms=11,mfc='cyan',mec='k',label='measured outer apex (auditor3)')
        ax.add_patch(plt.Circle(ai,4,fill=False,ls=':',color='r',lw=1.5))
        dd=np.linalg.norm(ai); ttl=f'{bid} @ {int(f*100)}%: gold->apex {dd:.2f} mm'
    else:
        ttl=f'{bid} @ {int(f*100)}%: corner notch (no fold)'
    r=fit_section(uv,nuv,na)
    if r is not None:
        ax.plot(*r['Xo'],'x',ms=13,mew=3,color='tab:blue',label='outer virtual sharp')
        if r['Xi'] is not None: ax.plot(*r['Xi'],'x',ms=13,mew=3,color='tab:purple',label='inner virtual sharp')
    ax.set_xlim(-9,9); ax.set_ylim(-9,9); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_title(ttl+'  (slab +-0.5 mm; red dotted = 4 mm around apex)',fontsize=11)
    ax.legend(loc='lower left',fontsize=8)
plt.suptitle('auditor3 sc_2714fb7e0c0b: perpendicular cross-sections, gold vs measured outer-skin apex',fontsize=16)
plt.tight_layout(); plt.savefig(OUT+'a3_xsections_gold_vs_apex.png',dpi=55); plt.close()
print('ok')
