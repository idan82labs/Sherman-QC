from common import *
import json
from scipy.ndimage import gaussian_filter1d
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
Rinv=R  # frame->raw: P = F@R + O
def to_raw(Pf): return np.asarray(Pf)@R+O
def bends_at(x0, ylo, yhi, zlo, zhi, upper=True, dy=0.2):
    m=(np.abs(X-x0)<0.5)&(Y>ylo)&(Y<yhi)&(Z>zlo)&(Z<zhi)
    y=Y[m]; z=Z[m]
    ys=np.arange(ylo,yhi,dy); env=np.full(len(ys),np.nan)
    for i,yy in enumerate(ys):
        k=np.abs(y-yy)<dy
        if k.sum(): env[i]=z[k].max() if upper else z[k].min()
    ok=np.isfinite(env)
    if ok.sum()<len(ys)*0.6: return None
    env=np.interp(ys,ys[ok],env[ok])
    s=gaussian_filter1d(env,3)
    d2=np.gradient(np.gradient(s,ys),ys)
    i1=np.argmax(d2); i2=np.argmin(d2)
    sl=np.gradient(s,ys); imax=np.argmax(np.abs(sl))
    return dict(y_pos=float(ys[i1]),z_pos=float(s[i1]),y_neg=float(ys[i2]),z_neg=float(s[i2]),max_slope_deg=float(np.degrees(np.arctan(abs(sl[imax])))))
out={}
for name,ylo,yhi,zlo,zhi,xr in (("back_lip_joggle",88,102,24,31.5,(-150,150)),("front_joggle",-156,-141,-14,-8,(-150,150))):
    rows=[]
    for x0 in np.arange(xr[0],xr[1]+1,10.0):
        r=bends_at(x0,ylo,yhi,zlo,zhi,True)
        if r: r["x"]=float(x0); rows.append(r)
    A=np.array([[r["x"],r["y_pos"],r["z_pos"]] for r in rows]); B=np.array([[r["x"],r["y_neg"],r["z_neg"]] for r in rows])
    res={}
    for tag,Pts in (("concave_up_bend(d2>0)",A),("convex_up_bend(d2<0)",B)):
        c=Pts.mean(0); u,s,vt=np.linalg.svd(Pts-c); d=vt[0]
        t=(Pts-c)@d; p0=c+d*t.min(); p1=c+d*t.max()
        res[tag]=dict(frame_mean=c.round(2).tolist(),y_sd=float(Pts[:,1].std()),raw_start=to_raw(p0).round(2).tolist(),raw_end=to_raw(p1).round(2).tolist())
    res["max_slope_deg_median"]=float(np.median([r["max_slope_deg"] for r in rows]))
    out[name]=res
    print(name, json.dumps(res,indent=1))
json.dump(out,open("joggle_lines.json","w"),indent=1)
