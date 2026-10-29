from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
fig,axs=plt.subplots(2,3,figsize=(24,15))
regions=[("left-back corner, Z>20",(-162,-110),(60,122),Z>20),("right-back corner, Z>20",(105,160),(60,122),Z>20),
         ("left-front lip start, Z>20",(-162,-130),(-100,-30),Z>20),("right-front lip start, Z>20",(130,160),(-100,-30),Z>20),
         ("left wall end region, all Z (side view Y-Z, X<-150)",None,None,None),("right wall end region (Y-Z, X>148)",None,None,None)]
for ax,(t,xr,yr,zm) in zip(axs.ravel(),regions):
    if xr is not None:
        m=zm&(X>xr[0])&(X<xr[1])&(Y>yr[0])&(Y<yr[1])
        sc=ax.scatter(X[m],Y[m],c=Z[m],s=0.5,cmap="turbo",vmin=24,vmax=31)
        plt.colorbar(sc,ax=ax,shrink=0.7,label="Z")
        for bid,(s,e) in gold.items():
            S_,E_=tof(s),tof(e); ax.plot([S_[0],E_[0]],[S_[1],E_[1]],"-",color="#1a9c3a",lw=2)
            ax.text((S_[0]+E_[0])/2,(S_[1]+E_[1])/2,bid,color="#1a9c3a")
            for P_ in (S_,E_):
                if xr[0]<P_[0]<xr[1] and yr[0]<P_[1]<yr[1]: ax.plot(P_[0],P_[1],"o",color="#1a9c3a",ms=8,mec="k")
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_xlabel("X"); ax.set_ylabel("Y")
    else:
        side=-1 if "left" in t else 1
        m=(X*side>148)
        sc=ax.scatter(Y[m],Z[m],c=X[m]*side,s=0.3,cmap="viridis")
        plt.colorbar(sc,ax=ax,shrink=0.7,label="|X|")
        for bid in (("HB4","HB5") if side<0 else ("HB3","HB6")):
            s,e=gold[bid]; S_,E_=tof(s),tof(e); ax.plot([S_[1],E_[1]],[S_[2],E_[2]],"-",color="#1a9c3a",lw=2); ax.text(S_[1],S_[2]+1,bid+" start",color="#1a9c3a")
        ax.set_xlim(-170,125); ax.set_ylim(-16,34); ax.set_xlabel("Y"); ax.set_ylabel("Z")
    ax.set_title(t); ax.set_aspect("equal"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+"a2_corners_and_extents.png",dpi=70); plt.close()
