import numpy as np, json, sys
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
sys.path.insert(0,"/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, Thresholds
TH=Thresholds()

def fitline(x,y):
    A=np.vstack([x,np.ones_like(x)]).T; return np.linalg.lstsq(A,y,rcond=None)[0]

def corner_apex(u,v,leg_h,leg_v,outer_h_low,outer_v_high):
    """generic L corner in (u,v): leg_h = mask of horizontal leg points (v ~ const), leg_v = mask of vertical leg (u ~ const).
    outer_h_low: True if the outer skin of the horizontal leg is the lower one (smaller v).
    outer_v_high: True if the outer skin of the vertical leg is the one with larger u."""
    if leg_h.sum()<8 or leg_v.sum()<8: return None
    vh=v[leg_h]; vm=np.median(vh)
    ho=leg_h.copy(); ho[leg_h]=(vh<vm) if outer_h_low else (vh>vm)
    uv=u[leg_v]; um=np.median(uv)
    vo=leg_v.copy(); vo[leg_v]=(uv>um) if outer_v_high else (uv<um)
    if ho.sum()<4 or vo.sum()<4: return None
    kh=fitline(u[ho],v[ho]); kv=fitline(v[vo],u[vo])
    k1,b1=kh; k2,b2=kv
    vv=(k1*b2+b1)/(1-k1*k2); uu=k2*vv+b2; X=np.array([uu,vv])
    # leg directions away from the corner, toward the material
    hu = np.median(u[leg_h]) - uu; dh=np.array([np.sign(hu), np.sign(hu)*k1]); dh/=np.linalg.norm(dh)
    vvv = np.median(v[leg_v]) - vv; dv=np.array([np.sign(vvv)*k2, np.sign(vvv)]); dv/=np.linalg.norm(dv)
    bis=dh+dv; bis/=np.linalg.norm(bis)
    ang=180-np.degrees(np.arccos(np.clip(dh@dv,-1,1)))
    rel=np.vstack([u,v]).T-X; along=rel@bis; perp=rel@np.array([-bis[1],bis[0]])
    near=(np.abs(perp)<0.35)&(along>-0.5)&(along<5)
    if not near.sum(): return None
    j=np.argmin(np.where(near,along,np.inf))
    return dict(apex=np.array([u[j],v[j]]),sharp=X,bis=bis,angle=ang)

def sweep(kind, s_values):
    pts=[]; angs=[]
    for s in s_values:
        if kind in ("B1","B2"):
            sgn = -1 if kind=="B1" else 1
            m=(np.abs(P[:,0]-s)<0.5)&(sgn*P[:,1]>70)&(P[:,2]<15)
            Q=P[m]; u=sgn*Q[:,1]; v=Q[:,2]
            leg_h=(u>76)&(u<86)&(v<-4); leg_v=(u>89)&(u<95)&(v>-1.5)&(v<12)
            r=corner_apex(u,v,leg_h,leg_v,True,True)
            if r is None: continue
            # reject window stations: require a real vertical leg next to the corner (points within v in [-4,-1] near u~92)
            if ((u>89)&(u<94)&(v>-4.5)&(v<-1)).sum()<6: continue
            if abs(r['angle']-90)>8: continue
            X3=np.array([s, sgn*r['apex'][0], r['apex'][1]])
        elif kind in ("B3","B4"):
            sgn = -1 if kind=="B3" else 1
            m=(np.abs(P[:,0]-s)<0.5)&(sgn*P[:,1]>70)&(P[:,2]>20)
            Q=P[m]; u=sgn*Q[:,1]; v=Q[:,2]   # wall at u~92, flange toward smaller u at top
            leg_h=(u>79)&(u<88)&(v>33); leg_v=(u>89)&(u<95)&(v>20)&(v<32)
            r=corner_apex(u,v,leg_h,leg_v,False,True)
            if r is None or abs(r['angle']-90)>8: continue
            X3=np.array([s, sgn*r['apex'][0], r['apex'][1]])
        elif kind in ("B5","B6"):
            sgn = -1 if kind=="B5" else 1
            m=(np.abs(P[:,1]-s)<0.5)&(sgn*P[:,0]>85)
            Q=P[m]; u=sgn*Q[:,0]; v=Q[:,2]
            ut = 104.5 if sgn<0 else 107.0
            leg_h=(u>ut-15)&(u<ut-5)&(v<-4); leg_v=(u>ut-2.5)&(u<ut+2.5)&(v>-3.5)&(v<2)
            r=corner_apex(u,v,leg_h,leg_v,True,True)
            if r is None or abs(r['angle']-90)>8: continue
            X3=np.array([sgn*r['apex'][0], s, r['apex'][1]])
        pts.append(X3); angs.append(r['angle'])
    return np.array(pts), np.array(angs)

folds = {
 "B1_left_base_wall":  ("B1", np.arange(-63.0, 66.01, 1.0)),
 "B2_right_base_wall": ("B2", np.arange(-63.0, 66.51, 1.0)),
 "B3_left_wall_top_tab": ("B3", np.arange(-24.5, 26.51, 1.0)),
 "B4_right_wall_top_tabs": ("B4", np.concatenate([np.arange(-63.5,-47.0,1.0), np.arange(51.0,66.51,1.0)])),
 "B5_end_tab_e1min": ("B5", np.arange(-47.5,-25.5,1.0)),
 "B6_end_tab_e1max": ("B6", np.arange(-48.0,-26.0,1.0)),
}
lines={}
for name,(kind,sv) in folds.items():
    X,A=sweep(kind,sv)
    if len(X)<3: print(name,"too few",len(X)); continue
    cen=X.mean(0); U,Sg,Wt=np.linalg.svd(X-cen); d=Wt[0]
    if np.argmax(np.abs(d))!=np.argmax(np.abs(d)) : pass
    t=(X-cen)@d
    resid=np.linalg.norm((X-cen)-np.outer(t,d),axis=1)
    a=cen+t.min()*d; b=cen+t.max()*d
    lines[name]=dict(start_part=a.tolist(), end_part=b.tolist(), n_stations=int(len(X)), length=float(t.max()-t.min()), resid_rms=float(np.sqrt((resid**2).mean())), angle_median=float(np.median(A)), station_coord_range=[float(X[:,0 if kind not in ('B5','B6') else 1].min()), float(X[:,0 if kind not in ('B5','B6') else 1].max())])
    print(name, "n=%d len=%.1f rms=%.2f ang=%.1f"%(len(X),t.max()-t.min(),np.sqrt((resid**2).mean()),np.median(A)), "start",np.round(a,2),"end",np.round(b,2))
# also split B4 into two
X,A=sweep("B4",np.arange(-63.5,-47.0,1.0)); 
for nm,(lo,hi) in {"B4a_right_top_tab_e1neg":(-63.5,-47.0),"B4b_right_top_tab_e1pos":(51.0,66.6)}.items():
    X,A=sweep("B4",np.arange(lo,hi,1.0)); cen=X.mean(0); U,Sg,Wt=np.linalg.svd(X-cen); d=Wt[0]; t=(X-cen)@d
    a=cen+t.min()*d; b=cen+t.max()*d
    lines[nm]=dict(start_part=a.tolist(), end_part=b.tolist(), n_stations=int(len(X)), length=float(t.max()-t.min()), angle_median=float(np.median(A)))
    print(nm,"n=%d len=%.1f"%(len(X),t.max()-t.min()),np.round(a,2),np.round(b,2))
# raw coords
for k,v in lines.items():
    v['line_start_raw']=to_raw(np.array(v['start_part'])).tolist(); v['line_end_raw']=to_raw(np.array(v['end_part'])).tolist()
json.dump(lines, open(OUT+"a1_measured_apex_lines.json","w"), indent=1)
# matcher metrics: every gold vs every measured fold
print()
print("gold vs measured apex lines (frozen matcher pair_metrics, raw frame):")
for gid in ["M01","M02","M03","M04","M05","M06","M07"]:
    graw=np.array([to_raw(G[gid][0]),to_raw(G[gid][1])])
    best=None
    for k,v in lines.items():
        praw=np.array([v['line_start_raw'],v['line_end_raw']])
        pm=pair_metrics(graw,praw); ok=is_match(pm,TH)
        if pm['axis_angle_deg']<10 and pm['transverse_offset_mm']<20:
            print(" ",gid,"vs",k,"ang %.2f off %.2f gold_ov %.2f pred_ov %.2f -> %s"%(pm['axis_angle_deg'],pm['transverse_offset_mm'],pm['gold_span_overlap'],pm['prediction_span_overlap'],"MATCH" if ok else "no"))
