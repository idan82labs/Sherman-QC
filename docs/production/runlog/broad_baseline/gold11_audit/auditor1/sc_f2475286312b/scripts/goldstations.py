import json
from folds import *
from scipy.spatial import cKDTree
tree=cKDTree(P)
ASSOC={'M01':'F1_upflange_v-54','M02':'F2_upflange_v-278','M03':'F5_slotflange_v-85','M04':'F5_slotflange_v-85',
       'M05':'F6_slotflange_v-258','M06':'F6_slotflange_v-258','M07':'F4_endwall_u-224','M08':None}
EXT={'F1_upflange_v-54':(-485.5,-267.0),'F2_upflange_v-278':(-485.0,-267.0),'F5_slotflange_v-85':(-403.0,-320.5),'F6_slotflange_v-258':(-403.0,-320.5),
     'F4_endwall_u-224':(-276.5,-55.5)}
def circle_fit(pts):
    x,y=pts[:,0],pts[:,1]; A=np.c_[2*x,2*y,np.ones(len(x))]; b=x**2+y**2
    c,*_=np.linalg.lstsq(A,b,rcond=None); r=np.sqrt(c[2]+c[0]**2+c[1]**2); return np.array([c[0],c[1]]),r
out={}
for k,(s,e,_) in g.items():
    gs=R@s; ge=R@e
    L=np.linalg.norm(ge-gs); n=int(L//5)+1
    rows=[]
    for t in np.linspace(0,1,n):
        p=gs+t*(ge-gs); praw=R.T@p
        dscan=float(tree.query(p)[0])
        row=dict(t=float(t),pos=p.tolist(),d_scan=dscan)
        fname=ASSOC[k]
        if fname:
            fd=FOLDS[fname]; ax=fd['ax']; x0=p[ax]
            lo,hi=EXT[fname]
            row['in_fold_extent']=bool(lo<=x0<=hi)
            st=station(fd,x0,half=0.4) if lo<=x0<=hi else None
            if st is not None and st['ok']:
                i,j=fd['cc']; g2=np.array([p[i],p[j]]); apex=st['apex']; ob=st['ob']
                dlt=g2-apex
                row['apex2d']=apex.tolist(); row['gold2d']=g2.tolist()
                row['d_apex']=float(np.linalg.norm(dlt))
                row['concave_along']=float(-(dlt@ob))      # + = toward concave side
                row['across']=float(dlt@np.array([-ob[1],ob[0]]))
                # arc centre from outer-skin arc points near apex
                m=(np.abs(P[:,ax]-x0)<0.4); Q=P[m][:,[i,j]]
                near=Q[np.linalg.norm(Q-apex,axis=1)<2.2]
                # outer arc: points whose distance to Xo is small-ish (outer skin) -> choose points with (Q-Xo).ob > -2.5 and on the outer skin: nearest to Xo per angle
                rel=near-st['Xo']; 
                if len(near)>=6:
                    C,r=circle_fit(near[np.argsort(np.linalg.norm(rel,axis=1))][:max(6,len(near)//2)])
                    row['arc_R']=float(r); row['d_arc_centre']=float(np.linalg.norm(g2-C)); row['centre_along']=float(-((C-apex)@ob))
                row['Xo_along']=float(-((st['Xo']-apex)@ob)); row['Xi_along']=None if st['Xi'] is None else float(-((st['Xi']-apex)@ob))
                row['angle']=st['raw_angle']
        rows.append(row)
    out[k]=rows
    # summary
    ds=np.array([r['d_scan'] for r in rows])
    ina=[r for r in rows if 'd_apex' in r]
    msg=f"{k}: len={L:.1f} stations={len(rows)} d_scan med={np.median(ds):.2f} frac<1mm={np.mean(ds<1):.2f}"
    if ina:
        da=np.array([r['d_apex'] for r in ina]); ca=np.array([r['concave_along'] for r in ina]); ac=np.array([r['across'] for r in ina])
        msg+=f" | on-fold stations={len(ina)} d_apex med={np.median(da):.2f} [{da.min():.2f}..{da.max():.2f}] concave_along med={np.median(ca):.2f} [{ca.min():.2f}..{ca.max():.2f}] across med={np.median(ac):.2f} [{ac.min():.2f}..{ac.max():.2f}]"
        cr=[r for r in ina if 'arc_R' in r]
        if cr:
            msg+=f" | arcR med={np.median([r['arc_R'] for r in cr]):.2f} centre_along med={np.median([r['centre_along'] for r in cr]):.2f} Xo_along med={np.median([r['Xo_along'] for r in ina]):.2f} Xi_along med={np.median([r['Xi_along'] for r in ina if r['Xi_along'] is not None]):.2f}"
        inext=[r for r in rows if r.get('in_fold_extent')]
        msg+=f" | gold stations inside fold extent={len(inext)}/{len(rows)}"
    print(msg)
json.dump(out,open('gold_stations.json','w'))
