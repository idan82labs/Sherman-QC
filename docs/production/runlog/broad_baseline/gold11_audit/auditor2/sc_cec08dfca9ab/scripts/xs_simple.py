from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, sys
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
# generic slice plotter: axis (0=X,1=Y), stations, window in other coords
def run(axis, stations, xlim, ylim, fname, half=0.5, golds=()):
    other=[i for i in (0,1,2) if i!=axis]
    n=len(stations); cols=min(n,4); rows=(n+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(6*cols,5.2*rows),squeeze=False)
    for k,s0 in enumerate(stations):
        ax=axs[k//cols,k%cols]
        m=np.abs(VF[:,axis]-s0)<half
        P=VF[m][:,other]
        sel=(P[:,0]>xlim[0])&(P[:,0]<xlim[1])&(P[:,1]>ylim[0])&(P[:,1]<ylim[1])
        ax.scatter(P[sel,0],P[sel,1],s=1.0,c="k")
        for bid in golds:
            s,e=gold[bid]; S_,E_=tof(s),tof(e)
            f=(s0-S_[axis])/(E_[axis]-S_[axis])
            if -0.001<=f<=1.001:
                g=S_+f*(E_-S_); ax.plot(g[other[0]],g[other[1]],"*",color="#1a9c3a",ms=14,mec="k"); ax.text(g[other[0]]+0.5,g[other[1]]+0.5,bid,color="#1a9c3a")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title("%s=%.1f  slab +-%.1f"%("XYZ"[axis],s0,half)); ax.set_xlabel("XYZ"[other[0]]); ax.set_ylabel("XYZ"[other[1]])
    plt.tight_layout(); plt.savefig(OUT+fname,dpi=70); plt.close()
if __name__=="__main__":
    axis=int(sys.argv[1]); st=[float(x) for x in sys.argv[2].split(",")]
    xl=[float(x) for x in sys.argv[3].split(",")]; yl=[float(x) for x in sys.argv[4].split(",")]
    golds=sys.argv[6].split(",") if len(sys.argv)>6 else ()
    run(axis,st,xl,yl,sys.argv[5],golds=golds)
