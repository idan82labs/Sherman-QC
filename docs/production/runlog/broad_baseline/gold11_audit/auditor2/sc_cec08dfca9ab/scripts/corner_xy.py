from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
# horizontal (X-Y) slices at several heights through the 4 vertical wall corners/ends
regions=[("left-back corner",(-165,-140),(100,125)),("right-back corner",(137,162),(100,125)),
         ("left wall front end",(-165,-140),(-97,-72)),("right wall front end",(137,162),(-97,-72))]
zs=[-5.0,5.0,15.0,24.0]
fig,axs=plt.subplots(len(regions),len(zs),figsize=(4.2*len(zs),4.2*len(regions)))
for i,(name,xr,yr) in enumerate(regions):
    for j,z0 in enumerate(zs):
        ax=axs[i,j]
        m=(np.abs(Z-z0)<0.5)&(X>xr[0])&(X<xr[1])&(Y>yr[0])&(Y<yr[1])
        ax.scatter(X[m],Y[m],s=1.0,c="k")
        for bid,(s,e) in gold.items():
            S_,E_=tof(s),tof(e)
            ax.plot([S_[0],E_[0]],[S_[1],E_[1]],"-",color="#1a9c3a",lw=1)
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title("%s  Z=%.0f+-0.5"%(name,z0),fontsize=9); ax.set_xlabel("X"); ax.set_ylabel("Y")
plt.tight_layout(); plt.savefig(OUT+"a2_vertical_corners_xy.png",dpi=70); plt.close()
print("ok")
