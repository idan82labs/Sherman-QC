import numpy as np
exec(open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage/triage_f247.py').read().split('# top & two')[0])
h=np.load(S+'h.npy'); co=np.array([9.05856392e-02,1.77835004e-03,-4.85222931e+00])
def feature(name,axis,u0,side,stations,legsign=None):
    # axis: in-plane coordinate index along the line (0 or 1); u = other in-plane coord
    a=axis; b=1-axis; pts=[]
    for s in stations:
        m=(np.abs(Q[:,a]-s)<1.5)&(np.abs(Q[:,b]-u0)<14)
        q=Q[m]; hh=h[m]
        # local plate level: points on plate side
        pm=(np.sign(q[:,b]-u0)==side)&(np.abs(q[:,b]-u0)>4)&(np.abs(q[:,b]-u0)<14)
        if pm.sum()<10: continue
        hp=np.median(hh[pm])
        leg=(np.abs(hh-hp)>3)&(np.abs(hh-hp)<8)&(np.abs(q[:,b]-u0)<6)
        if legsign is not None: leg&=np.sign(hh-hp)==legsign
        if leg.sum()<5: continue
        uc=np.median(q[leg,b])
        p=np.zeros(3); p[a]=s; p[b]=uc
        p[2]=hp+(co[0]*p[0]+co[1]*p[1]+co[2])
        pts.append(p)
    pts=np.array(pts)
    if len(pts)<3: print(name,'no corner'); return None
    c0=pts.mean(0); _,_,vt=np.linalg.svd(pts-c0); d=vt[0]
    t=(pts-c0)@d; res=np.linalg.norm((pts-c0)-np.outer(t,d),axis=1)
    print(f'{name}: n={len(pts)} extent {t.min():.0f}..{t.max():.0f} (len {t.max()-t.min():.0f}) rms_res {np.sqrt((res**2).mean()):.2f}')
    return c0,d,t.min(),t.max()
def off(line,s1,e1):
    c0,d,_,_=line; u=e1-s1; ang=np.degrees(np.arccos(abs(u@d)/np.linalg.norm(u)))
    ts=np.linspace(0,1,21); P_=s1+np.outer(ts,u); t=(P_-c0)@d
    inside=(t>=line[2])&(t<=line[3])
    r=np.linalg.norm((P_-c0)-np.outer(t,d),axis=1)
    rr=r[inside] if inside.any() else r
    return ang,rr.mean(),inside.mean()
F={}
F['lip+PC2 (M01)']=feature('lip +PC2 y~112',0,112,-1,np.arange(-100,150,5))
F['lip-PC2 (M02)']=feature('lip -PC2 y~-111',0,-111,+1,np.arange(-100,150,5))
F['tab top (M03)']=feature('tab top y~82',0,82,+1,np.arange(-60,120,3))
F['tab bot (M05)']=feature('tab bottom y~-92',0,-92,-1,np.arange(-60,120,3))
F['wall -PC1 (M07)']=feature('wall x~-110',1,-110,+1,np.arange(-105,105,5))
F['lip +PC1']=feature('lip x~143',1,143,-1,np.arange(-105,105,5))
for fn,line in F.items():
    if line is None: continue
    print('==',fn)
    for nm,D in [('gold',G),('best',B),('ph',C)]:
        for k,(s0,e0) in D.items():
            ang,o,ins=off(line,T(s0),T(e0))
            if ang<8 and o<25: print(f'   {nm:5s} {k:22s} ang {ang:5.2f} off_mean {o:6.2f} mm  frac_inside_true_extent {ins:.2f}')
