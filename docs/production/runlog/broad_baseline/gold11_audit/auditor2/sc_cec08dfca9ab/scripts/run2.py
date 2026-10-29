from xfit2 import *
import json
out={}
for bid in ["HB1","HB2","HB3","HB4","HB5","HB6"]:
    sp=SPEC[bid]; si,pi,qi=sp["s"],sp["p"],sp["q"]
    lo,hi=TRUE_EXT[bid]
    rows=[]
    for s0 in np.arange(lo+3,hi-3+0.01,(hi-lo-6)/40):
        r,P=station(bid,s0); g,f=gold_at(bid,s0); gp=np.array([g[pi],g[qi]])
        if not (r.get("ok") and r.get("apex_ok")): rows.append(dict(s=float(s0),ok=False)); continue
        dg=gp-r["apex"]
        row=dict(s=float(s0),f=float(f),ok=True,angle=float(r["bend_angle"]),apex2d=[float(x) for x in r["apex"]],
                 d_apex=float(np.linalg.norm(dg)),along=float(dg@r["bis"]),across=float(dg@r["perp"]),
                 d_outer_sharp=float(np.linalg.norm(gp-r["Xo"])),d_inner_sharp=float(np.linalg.norm(gp-r["Xi"])),
                 d_inner_apex=(float(np.linalg.norm(gp-r["inner_apex"])) if r["inner_apex"] is not None else None),
                 d_arc_centre=(float(np.linalg.norm(gp-r["arc_C"])) if r["arc_C"] is not None else None),
                 arc_R=float(r["arc_R"]),t_o=float(r["t_o"]),t_i=float(r["t_i"]),thick=[float(x) for x in r["thick"]],
                 gold_along_from_centre=(float((gp-r["arc_C"])@r["bis"]) if r["arc_C"] is not None else None))
        # 3D apex point
        A3=np.zeros(3); A3[si]=s0; A3[pi],A3[qi]=r["apex"]; row["apex3d_frame"]=A3.tolist()
        rows.append(row)
    out[bid]=rows
    ok=[x for x in rows if x["ok"]]
    def med(k): 
        v=[x[k] for x in ok if x.get(k) is not None and np.isfinite(x[k])]; return (np.median(v),np.min(v),np.max(v)) if v else (np.nan,)*3
    print("==",bid,"stations ok %d/%d"%(len(ok),len(rows)))
    for k in ["d_apex","along","across","d_outer_sharp","d_inner_sharp","d_inner_apex","d_arc_centre","arc_R","angle","t_o","t_i","gold_along_from_centre"]:
        print("   %-24s median %6.2f  min %6.2f  max %6.2f"%((k,)+tuple(med(k))))
json.dump(out,open("stations2.json","w"))
