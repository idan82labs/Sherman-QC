import sys, numpy as np, json, math
sys.path.insert(0,'/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714')
from xs import *
from scipy.spatial import cKDTree
tree=cKDTree(V)
OUTJ=W+'extent.json'
only=sys.argv[1:] if len(sys.argv)>1 else list(GOLD)

def to3(T,s,p2):
    return T.A+T.a*s+T.e1*p2[0]+T.e2*p2[1]

def station_fits(T, fracs, ref=None):
    rows=[]
    for f in fracs:
        uv,nuv,na=T.slab(f*T.L,0.6)
        r=fit_section(uv,nuv,na,ref_dirs=ref)
        if r is None or r['apex'] is None: continue
        o=offsets(r,np.zeros(2))
        if o['d_apex']>8: continue
        rows.append((f,r,o))
    return rows

res={}
for bid in only:
    A,B=GOLD[bid]
    T=Tube(A,B,rw=35,ext=45)
    # ref from mid
    ref=None
    for f in [0.5,0.45,0.55,0.4,0.6,0.3,0.7]:
        uv,nuv,na=T.slab(f*T.L)
        r=fit_section(uv,nuv,na)
        if r is not None: ref=[math.degrees(math.atan2(l['n'][1],l['n'][0]))%180 for l in r['legs']]; break
    fr=np.round(np.arange(0.05,0.951,0.05),3)
    rows=station_fits(T,fr,ref)
    if len(rows)<3:
        print(bid,'too few fits',len(rows)); continue
    AP=np.array([to3(T,f*T.L,r['apex']) for f,r,o in rows])
    c=AP.mean(0); U,S,Vt=np.linalg.svd(AP-c); dA=Vt[0]
    if dA@T.a<0: dA=-dA
    resid=np.linalg.norm((AP-c)-np.outer((AP-c)@dA,dA),axis=1)
    ang=math.degrees(math.acos(min(1,abs(dA@T.a))))
    # gold endpoints & midpoint relative to apex line
    def rel(p):
        d=p-c; s=d@dA; perp=d-s*dA; return s,np.linalg.norm(perp)
    sA,pA=rel(A); sB,pB=rel(B); sM,pM=rel((A+B)/2)
    # mid-station geometry for leg presence
    fmid,rmid,omid=min(rows,key=lambda x:abs(x[0]-0.5))
    legs=rmid['legs']; b2=rmid['b']; sap=rmid['s_apex']
    n3=[l['n'][0]*T.e1+l['n'][1]*T.e2 for l in legs]
    d3=[l['d'][0]*T.e1+l['d'][1]*T.e2 for l in legs]
    cs=[l['concave_sign'] for l in legs]
    b3=b2[0]*T.e1+b2[1]*T.e2
    # extent scan along apex line
    _,f1,f2=basis(dA, hint=T.e1)
    dV=V-c; sV=dV@dA
    lo=min(sA,sB)-40; hi=max(sA,sB)+40
    m=(sV>lo-1)&(sV<hi+1)
    perp=dV[m]-np.outer(sV[m],dA); rr=np.linalg.norm(perp,axis=1); m2=rr<16
    P=dV[m][m2]; sP=sV[m][m2]
    pres=[]
    grid=np.arange(lo,hi+0.01,1.0)
    for s0 in grid:
        q=P[np.abs(sP-s0)<0.5]
        if len(q)==0: pres.append(False); continue
        apex0=s0*dA  # apex line point (relative to c)
        rel3=q-apex0; rel3=rel3-np.outer(rel3@dA,dA)
        nap=(np.linalg.norm(rel3,axis=1)<1.2).sum()
        Xo=sap*b3
        ok=nap>=3
        for k in range(2):
            nk=n3[k]*cs[k]   # toward concave
            lam=(rel3-Xo)@d3[k]; mu=(rel3-Xo)@nk
            cnt=((lam>4)&(lam<10)&(mu>-0.7)&(mu<2.3)).sum()
            ok = ok and cnt>=5
        pres.append(bool(ok))
    pres=np.array(pres)
    # runs
    runs=[]; i=0
    while i<len(pres):
        if pres[i]:
            j=i
            while j+1<len(pres) and pres[j+1]: j+=1
            runs.append((grid[i],grid[j])); i=j+1
        else: i+=1
    # merge gaps <=3mm
    merged=[]
    for r0 in runs:
        if merged and r0[0]-merged[-1][1]<=3.5: merged[-1]=(merged[-1][0],r0[1])
        else: merged.append(r0)
    big=[r0 for r0 in merged if r0[1]-r0[0]>=3]
    g0,g1=sorted([sA,sB])
    if big:
        f0=min(r0[0] for r0 in big); f1_=max(r0[1] for r0 in big)
        # restrict to the run(s) overlapping the gold span (+-20)
        ov=[r0 for r0 in big if r0[1]>=g0-20 and r0[0]<=g1+20]
        f0=min(r0[0] for r0 in ov) if ov else f0; f1_=max(r0[1] for r0 in ov) if ov else f1_
        inter=max(0,min(g1,f1_)-max(g0,f0))
        cov=inter/(f1_-f0) if f1_>f0 else float('nan'); onfold=inter/(g1-g0)
    else:
        f0=f1_=float('nan'); cov=onfold=float('nan'); ov=[]
    # along-offset stats
    alongs=[o['along_concave'] for f,r,o in rows]; across=[o['across'] for f,r,o in rows]; dap=[o['d_apex'] for f,r,o in rows]
    info=dict(bid=bid,L=T.L,n_fits=len(rows),apex_line_resid_max=float(resid.max()),angle_gold_vs_apexline_deg=ang,
              gold_start_perp_mm=pA,gold_end_perp_mm=pB,gold_mid_perp_mm=pM,gold_span=[g0,g1],fold_span=[f0,f1_],
              fold_runs=[list(map(float,r0)) for r0 in merged],coverage_of_fold=cov,gold_fraction_on_fold=onfold,
              along_concave_med=float(np.median(alongs)),along_min=float(min(alongs)),along_max=float(max(alongs)),
              across_med=float(np.median(across)),across_min=float(min(across)),across_max=float(max(across)),
              d_apex_med=float(np.median(dap)),d_apex_max=float(max(dap)),
              bend_angle_med=float(np.median([r['bend_angle'] for f,r,o in rows])),
              apex_line_point=c.tolist(),apex_line_dir=dA.tolist(),
              apex_fold_start_xyz=(c+dA*f0).tolist() if big else None, apex_fold_end_xyz=(c+dA*f1_).tolist() if big else None)
    res[bid]=info
    print(f"{bid}: L={T.L:.1f} fits={len(rows)} ang(gold,apexline)={ang:.2f}deg resid_max={resid.max():.2f} perp(start/mid/end)={pA:.2f}/{pM:.2f}/{pB:.2f} | gold span [{g0:.1f},{g1:.1f}] fold span [{f0:.1f},{f1_:.1f}] runs={[(round(a,1),round(b,1)) for a,b in merged]} | coverage={cov:.3f} gold_on_fold={onfold:.3f} | along med {np.median(alongs):+.2f} [{min(alongs):+.2f},{max(alongs):+.2f}] across med {np.median(across):+.2f} [{min(across):+.2f},{max(across):+.2f}] d_apex med {np.median(dap):.2f} max {max(dap):.2f} angle {info['bend_angle_med']:.1f}")
old={}
try: old=json.load(open(OUTJ))
except Exception: pass
old.update(res); json.dump(old,open(OUTJ,'w'),indent=1,default=float)
