import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
st=[-55,-48,-46,-42,-37,-32,-28,-26,-20,0,30,60]
fig,axs=plt.subplots(4,6,figsize=(30,20))
info={}
for r,(side,sgn) in enumerate((("e1 min end",-1),("e1 max end",1))):
    info[side]=[]
    for j,s in enumerate(st):
        ax=axs[2*r + j//6, j%6]
        m=(np.abs(P[:,1]-s)<0.5)&(sgn*P[:,0]>90)
        Q=P[m]
        ax.scatter(Q[:,0],Q[:,2],s=2,c='k')
        up = Q[Q[:,2]>-4.5]
        h = float(up[:,2].max()) if len(up) else None
        info[side].append(dict(e2=s, tab_top_e3=h, n_up=int(len(up))))
        ax.set_title(f"{side}: slice e2={s}  " + (f"TAB up to e3={h:.1f}" if h is not None else "no tab"),fontsize=11)
        cx = sgn*104
        ax.set_xlim(cx-12,cx+12); ax.set_ylim(-12,8); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('e1'); ax.set_ylabel('e3')
fig.suptitle("Short-end tabs: e1-e3 cross-sections (slab |e2-s|<0.5). No gold line exists near e1=+-105.",fontsize=14)
plt.tight_layout(); plt.savefig(OUT+"a1_xsec_end_tabs.png",dpi=55); plt.close()
for k,v in info.items():
    for x in v: print(k,x)
# tab extents in e2 and geometry
for sgn in (-1,1):
    m=(P[:,2]>-4.5)&(np.abs(P[:,1])<80)&(sgn*P[:,0]>95)
    Q=P[m]
    print("tab",sgn,"e1 %.2f..%.2f e2 %.2f..%.2f e3 %.2f..%.2f"%(Q[:,0].min(),Q[:,0].max(),Q[:,1].min(),Q[:,1].max(),Q[:,2].min(),Q[:,2].max()))
    # skins: e1 of outer/inner at mid height
    mm=(Q[:,2]>-2)&(Q[:,2]<1)
    e1s=np.sort(Q[mm,0]); print("   e1 at mid-height pct 5/50/95:", np.percentile(e1s,[5,50,95]))
