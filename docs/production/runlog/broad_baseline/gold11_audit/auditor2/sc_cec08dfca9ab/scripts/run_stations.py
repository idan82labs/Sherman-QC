from xfit import *
import json,sys
out={}
for bid in ["HB1","HB2","HB3","HB4","HB5","HB6"]:
    s,e=gold[bid]; S_,E_=tof(s),tof(e); si=SPEC[bid]["s"]; pi,qi=SPEC[bid]["p"],SPEC[bid]["q"]
    lo,hi=min(S_[si],E_[si]),max(S_[si],E_[si])
    rows=[]
    for s0 in np.arange(np.floor(lo)-12,np.ceil(hi)+12,2.0):
        r,P=station(bid,s0)
        g,f=gold_at(bid,s0)
        row=dict(s=float(s0),f=float(f),n=r["n"],nh=r["nh"],nv=r["nv"],ok=r.get("ok",False))
        if r.get("ok") and r.get("apex_ok"):
            gp=np.array([g[pi],g[qi]])
            dg=gp-r["apex"]
            row.update(angle=float(r["bend_angle"]),apex=[float(x) for x in r["apex"]],
                       d_apex=float(np.linalg.norm(dg)),along_bis=float(dg@r["bis"]),across=float(dg@r["perp"]),
                       d_outer_sharp=float(np.linalg.norm(gp-r["Xo"])),d_inner_sharp=float(np.linalg.norm(gp-r["Xi"])),
                       t_apex=float(r["t_apex"]),t_inner=float(r["t_inner"]),R_o=float(r["R_outer_est"]),
                       thick=[float(r["thick_h"]),float(r["thick_v"])])
        rows.append(row)
    out[bid]=rows
    print("=====",bid,"gold s-range %.1f..%.1f"%(lo,hi))
    for row in rows:
        if row.get("d_apex") is not None:
            print("  s %7.1f f %5.2f nh %4d nv %4d ang %5.1f d_apex %5.2f along %5.2f across %5.2f dOS %5.2f dIS %5.2f tA %4.2f tI %4.2f Ro %4.2f th %.2f/%.2f"%(
                row["s"],row["f"],row["nh"],row["nv"],row["angle"],row["d_apex"],row["along_bis"],row["across"],row["d_outer_sharp"],row["d_inner_sharp"],row["t_apex"],row["t_inner"],row["R_o"],row["thick"][0],row["thick"][1]))
        else:
            print("  s %7.1f f %5.2f nh %4d nv %4d  NO-BEND-FIT ok=%s"%(row["s"],row["f"],row["nh"],row["nv"],row["ok"]))
json.dump(out,open("stations.json","w"))
