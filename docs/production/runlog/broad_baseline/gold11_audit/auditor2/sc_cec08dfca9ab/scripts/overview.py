from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
rng=np.random.default_rng(0)
idx=rng.choice(len(VF),200000,replace=False)
P=VF[idx]
fig,axs=plt.subplots(1,3,figsize=(24,8))
views=[(0,1,"top X-Y (color=Z)"),(0,2,"front X-Z (color=Y)"),(1,2,"side Y-Z (color=X)")]
for ax,(i,j,t) in zip(axs,views):
    k=3-i-j
    sc=ax.scatter(P[:,i],P[:,j],c=P[:,k],s=0.2,cmap="viridis")
    plt.colorbar(sc,ax=ax,shrink=0.6)
    for bid,(s,e) in gold.items():
        S_,E_=tof(s),tof(e)
        ax.plot([S_[i],E_[i]],[S_[j],E_[j]],"r-",lw=1.5)
        m=(S_+E_)/2; ax.text(m[i],m[j],bid,color="red",fontsize=10,weight="bold")
    ax.set_aspect("equal"); ax.set_title(t); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+"a2_overview_frame.png",dpi=90)
