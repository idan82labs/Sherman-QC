from xfit2 import *
import json, sys
sys.path.insert(0,"/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
st=json.load(open("stations2.json"))
def to_raw(Pf): return np.asarray(Pf)@R+O
out={}
for bid in ["HB1","HB2","HB3","HB4","HB5","HB6"]:
    si=SPEC[bid]["s"]
    pts=np.array([r["apex3d_frame"] for r in st[bid] if r.get("ok")])
    c=pts.mean(0); u,s,vt=np.linalg.svd(pts-c); d=vt[0]
    if d[si]<0: d=-d
    lo,hi=TRUE_EXT[bid]
    p0=c+d*(lo-c[si])/d[si]; p1=c+d*(hi-c[si])/d[si]
    resid=np.linalg.norm((pts-c)-np.outer((pts-c)@d,d),axis=1)
    A0,A1=to_raw(p0),to_raw(p1)
    gs,ge=gold[bid]
    m=pair_metrics(np.array([gs,ge]),np.array([A0,A1]))
    gdir=unit(ge-gs); adir=unit(A1-A0)
    out[bid]=dict(apex_line_raw=[A0.round(3).tolist(),A1.round(3).tolist()],apex_fit_rms_mm=float(np.sqrt((resid**2).mean())),
                  true_extent_frame=[lo,hi],true_len_mm=float(np.linalg.norm(A1-A0)),gold_len_mm=float(np.linalg.norm(ge-gs)),
                  matcher_vs_gold=dict((k,round(float(v),3)) for k,v in m.items()),match=bool(is_match(m,Thresholds())))
    print(bid,"apex rms %.2f"%out[bid]["apex_fit_rms_mm"],"true len %.1f gold len %.1f"%(out[bid]["true_len_mm"],out[bid]["gold_len_mm"]),
          "| angle %.2f transverse %.2f gold_ovl %.2f pred_ovl %.2f -> match=%s"%(m["axis_angle_deg"],m["transverse_offset_mm"],m["gold_span_overlap"],m["prediction_span_overlap"],out[bid]["match"]))
json.dump(out,open("apex_lines.json","w"),indent=1)
# frame-free inter-line distances: gold vs apex
def line_dist(a0,a1,b0,b1):
    d=unit(a1-a0); mb=(b0+b1)/2; v=mb-a0; return float(np.linalg.norm(v-(v@d)*d))
pairs=[("HB1","HB2"),("HB3","HB4"),("HB5","HB6"),("HB3","HB6"),("HB4","HB5")]
print("pair  gold_dist  apex_dist  diff")
for a,b in pairs:
    gd=line_dist(*gold[a],*gold[b]); A=out[a]["apex_line_raw"]; B=out[b]["apex_line_raw"]
    ad=line_dist(np.array(A[0]),np.array(A[1]),np.array(B[0]),np.array(B[1]))
    print(a,b,"%.2f  %.2f  %+.2f"%(gd,ad,gd-ad))
