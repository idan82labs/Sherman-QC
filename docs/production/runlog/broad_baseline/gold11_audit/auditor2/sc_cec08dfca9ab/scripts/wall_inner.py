from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
fig,axs=plt.subplots(2,3,figsize=(22,10))
# Y-Z maps of the side walls (inner and outer skins), coloured by X, to look for ribs/beads/extra folds
for k,(name,lo,hi) in enumerate((("RIGHT wall inner skin X 152.6..154.4",152.6,154.4),("RIGHT wall outer skin X 154.4..156",154.4,156.0),("LEFT wall inner skin X -157.6..-155.6",-157.6,-155.6))):
    ax=axs[0,k]; m=(X>lo)&(X<hi)&(Z>-14)&(Z<31)
    sc=ax.scatter(Y[m],Z[m],c=X[m],s=0.3,cmap="viridis"); plt.colorbar(sc,ax=ax,shrink=0.6,label="X")
    ax.set_title(name); ax.set_xlabel("Y"); ax.set_ylabel("Z"); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-95,120)
# cross-sections X-Z at Y = 92, 100, 104 near right wall
for k,y0 in enumerate((92.0,100.0,104.0)):
    ax=axs[1,k]; m=(np.abs(Y-y0)<0.4)&(X>140)&(X<158)
    ax.scatter(X[m],Z[m],s=1.2,c="k"); ax.set_title("X-Z section Y=%.0f +-0.4 (right wall, back corner region)"%y0)
    ax.set_xlim(140,158); ax.set_ylim(-14,31); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("X"); ax.set_ylabel("Z")
plt.tight_layout(); plt.savefig(OUT+"a2_sidewall_inner_check.png",dpi=65); plt.close(); print("ok")
