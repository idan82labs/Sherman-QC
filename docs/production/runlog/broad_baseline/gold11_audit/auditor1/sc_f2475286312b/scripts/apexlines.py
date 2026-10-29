import sys, json
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject')
from folds import *
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
res=json.load(open('fold_stations.json'))
EXT={'F1_upflange_v-54':(-485.5,-267.0),'F2_upflange_v-278':(-485.0,-267.0),'F5_slotflange_v-85':(-403.0,-320.5),'F6_slotflange_v-258':(-403.0,-320.5),
     'F3_leftflange_u-477':(-267.0,-65.5),'F4_endwall_u-224':(-276.5,-55.5),'F7_ear_v-54':(-111.5,-69.0),'F8_ear_v-278':(-112.5,-69.0)}
lines={}
for name,rows in res.items():
    fd=FOLDS[name]; lo,hi=EXT[name]
    pts=[]
    for r in rows:
        if not r['ok'] or not (lo<=r['x']<=hi): continue
        if not (80<r['angle']<100): continue
        pts.append(to3d(fd,r['x'],np.array(r['apex'])))
    pts=np.array(pts)
    # robust line fit in part frame
    c=pts.mean(0); _,_,vt=np.linalg.svd(pts-c); d=vt[0]
    res_perp=np.linalg.norm((pts-c)-np.outer((pts-c)@d,d),axis=1)
    keep=res_perp<max(0.6,3*np.median(res_perp))
    c=pts[keep].mean(0); _,_,vt=np.linalg.svd(pts[keep]-c); d=vt[0]
    if d[fd['ax']]<0: d=-d
    # endpoints at the fold extent along axis
    t_lo=(lo-c[fd['ax']])/d[fd['ax']]; t_hi=(hi-c[fd['ax']])/d[fd['ax']]
    e0=c+t_lo*d; e1=c+t_hi*d
    rp=np.linalg.norm((pts[keep]-c)-np.outer((pts[keep]-c)@d,d),axis=1)
    raw0=R.T@e0; raw1=R.T@e1
    lines[name]=dict(start_part=e0.tolist(),end_part=e1.tolist(),start_raw=raw0.tolist(),end_raw=raw1.tolist(),length=float(np.linalg.norm(e1-e0)),
                     n_stations=int(keep.sum()),rms_resid=float(np.sqrt(np.mean(rp**2))),max_resid=float(rp.max()))
    print(f"{name:22s} len={lines[name]['length']:.1f} n={keep.sum()} rms={lines[name]['rms_resid']:.2f} max={lines[name]['max_resid']:.2f}  part start {np.round(e0,2)} end {np.round(e1,2)}")
json.dump(lines,open('apex_lines.json','w'),indent=1)
# matcher metrics gold vs apex lines
th=Thresholds()
print()
print('gold  fold                  angle  transv  gold_ov pred_ov match')
summary={}
for k,(s,e,_) in g.items():
    G=np.array([s,e])
    best=None
    for name,L in lines.items():
        Pp=np.array([L['start_raw'],L['end_raw']])
        m=pair_metrics(G,Pp)
        if m['axis_angle_deg']<10 and m['transverse_offset_mm']<25:
            print(f"{k}  {name:22s} {m['axis_angle_deg']:5.2f} {m['transverse_offset_mm']:7.2f} {m['gold_span_overlap']:7.2f} {m['prediction_span_overlap']:7.2f} {is_match(m,th)}")
        if best is None or m['transverse_offset_mm']<best[1]['transverse_offset_mm'] and m['axis_angle_deg']<10:
            best=(name,m)
    summary[k]=best
