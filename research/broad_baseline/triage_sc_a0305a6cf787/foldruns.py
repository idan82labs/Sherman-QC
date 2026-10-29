"""Physical fold extent along each gold / pred line (2 mm steps, extended 20 mm beyond each end): F = two flanges >6 mm long meet within 6 mm of the line."""
import sys,re,numpy as np
src=open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787/vsharp.py').read().split('for g,(s,e) in G.items():')[0]
exec(src)
for k,v in G.items(): L['gold:'+k]=v
for name,(s,e) in sorted(L.items()):
    ln=np.linalg.norm(e-s); d=(e-s)/ln
    a=np.array([0,0,1.]); e1=np.cross(d,a); e1/=np.linalg.norm(e1); e2=np.cross(d,e1)
    out=[]; ts=np.arange(-20,ln+20,2.0); dist=[]
    for t in ts:
        c=s+t*d; q=P-c; m=np.abs(q@d)<0.75; u=q[m]@e1; v=q[m]@e2; r=u*u+v*v<12**2
        if r.sum()<30: out.append('.'); continue
        n2=np.c_[N[m][r]@e1,N[m][r]@e2]; vs,sep=fit2(u[r],v[r],n2); X=np.c_[u[r],v[r]]
        ok = vs is not None and np.linalg.norm(vs)<6
        if ok:
            dv=X-vs; far=np.linalg.norm(dv,axis=1)>6; dirs=np.degrees(np.arctan2(dv[far,1],dv[far,0]))%360
            h,_=np.histogram(dirs,bins=12,range=(0,360)); ok=len(dirs)>=20 and np.sort(h)[-2]>5
        out.append('F' if ok else '-'); 
        if ok: dist.append(np.linalg.norm(vs))
    s_=''.join(out); i0=10; i1=10+int(ln//2)
    runs=[(m.start()*2-20,m.end()*2-20) for m in re.finditer('F{3,}',s_)]
    tot=sum(min(b,ln)-max(a_,0) for a_,b in runs if min(b,ln)>max(a_,0))
    print(f'{name:34s} len {ln:6.1f}  fold runs(mm from start) {runs}  fold inside span {tot:.0f} mm ({tot/ln:.0%})  median line->fold {np.median(dist) if dist else float("nan"):.1f} mm')
