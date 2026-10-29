from common import *
import json
X,Y,Z=VF[:,0],VF[:,1],VF[:,2]
def profile_env(x0, ylo, yhi, zlo, zhi, upper=True, half=0.5, dy=0.25):
    m=(np.abs(X-x0)<half)&(Y>ylo)&(Y<yhi)&(Z>zlo)&(Z<zhi)
    y=Y[m]; z=Z[m]
    ys=np.arange(ylo,yhi,dy); env=np.full(len(ys),np.nan)
    for i,yy in enumerate(ys):
        k=np.abs(y-yy)<dy
        if k.sum(): env[i]=z[k].max() if upper else z[k].min()
    return ys,env
def analyze(name, ylo, yhi, zlo, zhi, xs):
    rows=[]
    for x0 in xs:
        for upper in (True,False):
            ys,env=profile_env(x0,ylo,yhi,zlo,zhi,upper)
            ok=np.isfinite(env)
            if ok.sum()<20: continue
            ys2,env2=ys[ok],env[ok]
            # smooth & slope
            k=np.ones(5)/5; es=np.convolve(env2,k,mode="same")
            sl=np.gradient(es,ys2)
            i=np.argmax(np.abs(sl[3:-3]))+3
            # flat levels: medians away from step
            yc=ys2[i]
            left=env2[(ys2<yc-4)]; right=env2[(ys2>yc+4)]
            # bends: where |slope| crosses half max on each side
            half_=np.abs(sl[i])/2
            j0=i
            while j0>0 and abs(sl[j0])>half_: j0-=1
            j1=i
            while j1<len(sl)-1 and abs(sl[j1])>half_: j1+=1
            rows.append(dict(x=float(x0),skin="upper" if upper else "lower",y_step=float(yc),max_slope_deg=float(np.degrees(np.arctan(abs(sl[i])))),
                             z_left=float(np.median(left)) if len(left) else None,z_right=float(np.median(right)) if len(right) else None,
                             y_bend_a=float(ys2[j0]),y_bend_b=float(ys2[j1])))
    print("==",name)
    for r in rows:
        print("  x %6.1f %-5s step@Y %7.2f  maxslope %5.1f deg  zL %s zR %s  half-slope span Y %.2f..%.2f"%(r["x"],r["skin"],r["y_step"],r["max_slope_deg"],
              "%.2f"%r["z_left"] if r["z_left"] is not None else "-", "%.2f"%r["z_right"] if r["z_right"] is not None else "-", r["y_bend_a"],r["y_bend_b"]))
    return rows
xs=[-150,-140,-100,-60,-20,0,20,60,100,140,148]
back=analyze("BACK LIP JOGGLE",86,104,24,31.5,xs)
front=analyze("FRONT JOGGLE",-160,-138,-14,-8,xs)
json.dump(dict(back=back,front=front),open("joggles.json","w"),indent=1)
# extent of joggles along X: presence of step
for name,ylo,yhi,zlo,zhi in (("back",86,104,24,31.5),("front",-160,-138,-14,-8)):
    present=[]
    for x0 in np.arange(-160,158,2.0):
        ys,env=profile_env(x0,ylo,yhi,zlo,zhi,True)
        ok=np.isfinite(env)
        if ok.sum()<20: continue
        e=env[ok]; yy=ys[ok]
        lo=np.median(e[yy<(ylo+yhi)/2-3]) if (yy<(ylo+yhi)/2-3).sum()>3 else np.nan
        hi=np.median(e[yy>(ylo+yhi)/2+3]) if (yy>(ylo+yhi)/2+3).sum()>3 else np.nan
        if np.isfinite(lo) and np.isfinite(hi) and abs(hi-lo)>0.8: present.append(x0)
    print(name,"joggle step present at X from",min(present) if present else None,"to",max(present) if present else None,"n",len(present))
