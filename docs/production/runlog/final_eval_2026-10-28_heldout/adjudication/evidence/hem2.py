"""Refine hem crown axis on the raw scan and build the D0 tangent-band bend spec (two plane_from_points seeds on the
outer hem arc at +-45 deg about the crown, extent = crown extent). Writes spec json + section plots with seeds."""
import json, sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
jobs=json.load(open(sys.argv[1])); outdir=sys.argv[2]; res={}; cache={}
def unit(v): v=np.asarray(v,float); return v/np.linalg.norm(v)
for j in jobs:
    if j['scan'] not in cache:
        cache.clear(); m=o3d.io.read_triangle_mesh(j['scan']); V=np.asarray(m.vertices).copy(); cache[j['scan']]=(V,cKDTree(V))
    V,tree=cache[j['scan']]
    ds=np.load(j['prep']+'/points_ds.npz')['points']; dtree=cKDTree(ds)
    nrm=unit(j['n']); p0=np.array(j['p0'],float); p1=np.array(j['p1'],float); body=np.array(j['body'],float)
    a=p1-p0; a-=nrm*(a@nrm); L=np.linalg.norm(a); a/=L
    tmin=j.get('stack_min')
    def scan_stations(p0,a,L):
        o=np.cross(a,nrm); o=o if (p0-body)@o>0 else -o
        idx=tree.query_ball_point(p0+a*L/2, L/2+12); P=V[idx]-p0; t=P@a; po=P@o; pf=P@nrm
        st=[]
        for ti in np.arange(-3, L+3.01, 1.0):
            k=(np.abs(t-ti)<0.3)&(np.abs(po)<10)&(np.abs(pf)<10)
            if k.sum()<5: continue
            om=np.percentile(po[k],99.5); near=k&(po>om-0.15); fc=np.median(pf[near])
            dp=j.get('depth',2.5); leg=k&(po>om-dp-1)&(po<om-dp)
            S=(np.percentile(pf[leg],98)-np.percentile(pf[leg],2)) if leg.sum()>=4 else -1
            st.append((ti,om,fc,S,p0+ti*a+om*o+fc*nrm))
        return o,st
    for it in range(6):
        o,st=scan_stations(p0,a,L)
        oms=np.array([x[1] for x in st]); med=np.median([x[1] for x in st if x[3]>tmin])
        sel=[x for x in st if x[3]>tmin and abs(x[1]-med)<(1.5 if it<2 else 0.6)]
        C=np.array([x[4] for x in sel]); cm=C.mean(0); _,_,vt=np.linalg.svd(C-cm); a2=vt[0]; a2-=nrm*(a2@nrm); a2=unit(a2)
        if a2@a<0: a2=-a2
        a=a2; tt=(C-cm)@a; p0=cm+(tt.min())*a; L=tt.max()-tt.min()
    o=np.cross(a,nrm); o=o if (p0-body)@o>0 else -o
    # final per-station crown on the refined axis
    idx=tree.query_ball_point(p0+a*L/2, L/2+12); P=V[idx]-p0; t=P@a; po=P@o; pf=P@nrm
    st=[]
    for ti in np.arange(-2, L+2.01, 1.0):
        k=(np.abs(t-ti)<0.3)&(np.abs(po)<10)&(np.abs(pf)<10)
        if k.sum()<5: st.append([ti,None,None,None]); continue
        om=np.percentile(po[k],99.5); near=k&(po>om-0.15); fc=np.median(pf[near])
        leg=k&(po>om-j.get('depth',2.5)-1)&(po<om-j.get('depth',2.5))
        S=(np.percentile(pf[leg],98)-np.percentile(pf[leg],2)) if leg.sum()>=4 else None
        st.append([round(float(ti),1),round(float(om),3),round(float(fc),3),None if S is None else round(float(S),3)])
    good=[s for s in st if s[1] is not None and abs(s[1])<0.6 and s[3] is not None and s[3]>tmin]
    t0=min(s[0] for s in good); t1=max(s[0] for s in good)
    om=np.median([s[1] for s in good]); fc=np.median([s[2] for s in good])
    Ss=[s[3] for s in good if s[3]]; S=float(np.median(Ss)) if Ss else 2*j.get('R',1.0)
    R=j.get('R',S/2)
    ctr=lambda ti: p0+ti*a+(om-R)*o+fc*nrm
    seeds=[]
    tm=(t0+t1)/2
    for sg in (+1,-1):
        pt=ctr(tm)+R*(np.cos(np.pi/4)*o+sg*np.sin(np.pi/4)*nrm)
        # snap onto the scan surface: nearest raw vertex in the half-plane
        d,i=tree.query(pt); sp=V[i]; dd,_=dtree.query(sp)
        seeds.append({'xyz':[round(float(x),3) for x in sp],'raw_dist':round(float(d),3),'ds_dist':round(float(dd),3)})
    apex0=p0+t0*a+om*o+fc*nrm; apex1=p0+t1*a+om*o+fc*nrm
    rr=float(j.get('ball', max(1.0,0.8*R)))
    spec={"patches":[{"plane_from_points":{"seed_xyz":seeds[0]['xyz'],"radius_mm":rr,"grow":True}},
                     {"plane_from_points":{"seed_xyz":seeds[1]['xyz'],"radius_mm":rr,"grow":True}}],
          "extent_xyz":[[round(float(x),3) for x in apex0],[round(float(x),3) for x in apex1]],
          "note":f"[adj] 180 deg hem ({j['name']}): sides are the two 45 deg tangent bands of the outer hem arc so the apex lands on the arc crown (D0 gb_44b168 B5 convention); reported angle is not the hem angle (real ~180 deg); outer stack {S:.2f} mm, R {R:.2f} mm, crown {t1-t0:.1f} mm"}
    res[j['name']]={'spec':spec,'axis':a.tolist(),'o':o.tolist(),'n':nrm.tolist(),'crown_line':[apex0.tolist(),apex1.tolist()],
                    'crown_len':float(t1-t0),'stack':S,'R':R,'stations':st,'seeds':seeds}
    print(j['name'],'L',round(t1-t0,1),'stack',round(S,2),'R',round(R,2),'seeds',seeds,'n_good',len(good),'/',len(st))
    fig,axs=plt.subplots(1,3,figsize=(18,6))
    for ax,fr in zip(axs,(0.2,0.5,0.8)):
        ti=t0+fr*(t1-t0); k=(np.abs(t-ti)<0.3)&(np.abs(po)<12)&(np.abs(pf)<12)
        ax.scatter(po[k],pf[k],s=2,c='k')
        c=ctr(ti)-p0; ax.plot(c@o,c@nrm,'r+')
        for sd in seeds:
            q=np.array(sd['xyz'])-p0; ax.plot(q@o,q@nrm,'go')
        ax.plot(om,fc,'bx'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f"{j['name']} t={ti:.1f} (x=out, y=plate normal)")
    fig.savefig(f"{outdir}/{j['name']}_v2.png",dpi=70); plt.close(fig)
json.dump(res,open(f'{outdir}/hem2_out.json','w'),indent=1)
