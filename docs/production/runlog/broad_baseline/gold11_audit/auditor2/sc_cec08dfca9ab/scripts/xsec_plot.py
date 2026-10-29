from xfit import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, sys
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
def plot_bend(bid, stations, fname, wide=45, close=9):
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]
    names="XYZ"
    fig,axs=plt.subplots(2,len(stations),figsize=(6*len(stations),12))
    for k,s0 in enumerate(stations):
        r,P=station(bid,s0)
        g,f=gold_at(bid,s0); gp=np.array([g[pi],g[qi]])
        for row,W in ((0,wide),(1,close)):
            ax=axs[row,k]
            c=gp if row==0 else (r["apex"] if r.get("apex_ok") else gp)
            m=(np.abs(P[:,0]-c[0])<W)&(np.abs(P[:,1]-c[1])<W)
            ax.scatter(P[m,0],P[m,1],s=1.5 if row else 0.6,c="k")
            ax.plot(*gp,"*",color="#1a9c3a",ms=16,mec="k",label="gold %s"%bid)
            circ=plt.Circle(gp,4.0,fill=False,ls=":",color="#1a9c3a"); ax.add_patch(circ)
            if r.get("ok"):
                ax.plot(*r["Xo"],"x",color="tab:blue",ms=10,mew=2,label="outer sharp")
                ax.plot(*r["Xi"],"x",color="tab:purple",ms=10,mew=2,label="inner sharp")
                if r.get("apex_ok"):
                    ax.plot(*r["apex"],"D",color="cyan",mec="k",ms=8,label="outer apex (A2 fit)")
                    L=np.array([r["Xo"]-3*r["bis"],r["Xo"]+9*r["bis"]]); ax.plot(L[:,0],L[:,1],"--",color="gray",lw=0.8)
            ax.set_xlim(c[0]-W,c[0]+W); ax.set_ylim(c[1]-W,c[1]+W); ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_xlabel(names[pi]); ax.set_ylabel(names[qi])
            t="%s @ %s=%.0f (f=%.2f) slab +-0.6"%(bid,names[si],s0,f)
            if r.get("apex_ok"):
                dg=gp-r["apex"]; t+="\n gold-apex %.2f mm (along bis %.2f, across %.2f), bend %.0f deg"%(np.linalg.norm(dg),dg@r["bis"],dg@r["perp"],r["bend_angle"])
            elif not r.get("ok"): t+="\n NO BEND FIT (nh=%d nv=%d)"%(r["nh"],r["nv"])
            ax.set_title(t,fontsize=10)
            if row==1 and k==0: ax.legend(fontsize=8,loc="lower left")
    plt.tight_layout(); plt.savefig(OUT+fname,dpi=75); plt.close()
if __name__=="__main__":
    bid=sys.argv[1]; st=[float(x) for x in sys.argv[2].split(",")]; fn=sys.argv[3]
    plot_bend(bid,st,fn)
