import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())

def top_apex(Q2, sgn):
    """top fold: wall (vertical at |e2|~92) to inward flange (horizontal at e3~37..38).
    mirror u = -sgn*e2 so the flange extends toward larger u (inward); wall at u~-92."""
    u = -sgn*Q2[:,0]; v = Q2[:,1]
    fm = (u>-88)&(u<-79)&(v>33)          # flange
    wm = (u>-94)&(u<-89)&(v>20)&(v<33)   # wall below the fold
    if fm.sum()<10 or wm.sum()<10: return None
    vf=v[fm]; vmid=np.median(vf)
    fo = fm.copy(); fo[fm]= vf>vmid      # outer skin of top flange = upper surface
    fi = fm.copy(); fi[fm]= vf<=vmid
    uw=u[wm]; umid=np.median(uw)
    wo = wm.copy(); wo[wm]= uw<umid      # outer skin of wall = outer side (more negative u)
    wi = wm.copy(); wi[wm]= uw>=umid
    def fit(x,y):
        A=np.vstack([x,np.ones_like(x)]).T; return np.linalg.lstsq(A,y,rcond=None)[0]
    kf_o=fit(u[fo],v[fo]); kw_o=fit(v[wo],u[wo]); kf_i=fit(u[fi],v[fi]); kw_i=fit(v[wi],u[wi])
    def inter(kf,kw):
        k1,b1=kf; k2,b2=kw
        vv=(k1*b2+b1)/(1-k1*k2); uu=k2*vv+b2; return np.array([uu,vv])
    Xo=inter(kf_o,kw_o); Xi=inter(kf_i,kw_i)
    df=np.array([1.0,kf_o[0]]); df/=np.linalg.norm(df)        # flange goes +u
    dw=np.array([-kw_o[0],-1.0]); dw/=np.linalg.norm(dw)      # wall goes down (-v)
    bis=df+dw; bis/=np.linalg.norm(bis)
    ang=180-np.degrees(np.arccos(np.clip(df@dw,-1,1)))
    rel=np.vstack([u,v]).T-Xo; along=rel@bis; perp=rel@np.array([-bis[1],bis[0]])
    near=(np.abs(perp)<0.35)&(along>-0.5)&(along<6)
    apex=None
    if near.sum():
        j=np.argmin(np.where(near,along,np.inf)); apex=np.array([u[j],v[j]])
    unm=lambda p: None if p is None else np.array([-sgn*p[0],p[1]])
    return dict(Xo=unm(Xo),Xi=unm(Xi),apex=unm(apex),bis=np.array([-sgn*bis[0],bis[1]]),bend_angle=float(ang),
                flange_width=float(u[fm].max()-Xo[0]))

stations=[-66,-63,-60,-55,-50,-47,-40,-30,-24,-20,-10,0,10,20,25,30,40,50,52,55,58,62,65,67]
res={}
for side,sgn,gid in (("left",-1,"M07"),("right",1,None)):
    fig,axs=plt.subplots(4,6,figsize=(30,20)); axs=axs.ravel(); rows=[]
    for i,s in enumerate(stations):
        ax=axs[i]
        m=(np.abs(P[:,0]-s)<0.5)&(sgn*P[:,1]>70)&(P[:,2]>20)
        Q=P[m][:,1:3]
        ax.scatter(Q[:,0],Q[:,1],s=1.5,c='k')
        row=dict(e1=s)
        info=top_apex(Q,sgn) if len(Q)>30 else None
        if info is not None and info['apex'] is not None:
            ax.plot(*info['Xo'],'bx',ms=12,mew=2); ax.plot(*info['Xi'],'mx',ms=12,mew=2); ax.plot(*info['apex'],'cD',ms=9,mec='k')
            row.update(fold=True,bend_angle=round(info['bend_angle'],1),apex=info['apex'].tolist(),flange_width=round(info['flange_width'],1))
        else: row.update(fold=False)
        if gid:
            g,inside=gold_at(G[gid],0,s)
            if g is not None:
                ax.plot(g[1],g[2],'*',ms=18,mfc='lime' if inside else 'none',mec='g',mew=2)
                row.update(gold_inside=bool(inside))
                if row['fold'] and inside:
                    d=np.array([g[1],g[2]])-info['apex']; row.update(d_apex=float(np.linalg.norm(d)),along_bis=float(d@info['bis']),d_sharp=float(np.linalg.norm(np.array([g[1],g[2]])-info['Xo'])))
                elif inside:
                    # distance from gold to nearest scan point in slice
                    dd=np.sqrt(((Q-np.array([g[1],g[2]]))**2).sum(1)); row.update(d_nearest_pt=float(dd.min()) if len(dd) else None)
        cx=sgn*90
        ax.set_xlim(cx-11,cx+11); ax.set_ylim(26,44); ax.set_aspect('equal'); ax.grid(alpha=.3)
        t=f"{side} top e1={s}: "+("FOLD ang %.1f, flange w %.1f"%(row['bend_angle'],row['flange_width']) if row['fold'] else "no fold (free top edge)")
        if 'd_apex' in row: t+=f"\n{gid} gold-apex {row['d_apex']:.2f} (bis {row['along_bis']:+.2f})"
        elif 'd_nearest_pt' in row: t+=f"\n{gid} gold IN span, no fold; nearest pt {row['d_nearest_pt']:.2f}"
        ax.set_title(t,fontsize=10); rows.append(row)
    fig.suptitle(f"{side} wall top-edge cross-sections (e2-e3, slab |e1-s|<0.5). "+(f"Gold {gid} spans e1 {G[gid][0,0]:.1f}..{G[gid][1,0]:.1f}" if gid else "NO gold line on this wall top"),fontsize=14)
    plt.tight_layout(); plt.savefig(OUT+f"a1_xsec_top_{side}.png",dpi=55); plt.close()
    res[side]=rows
    for r in rows: print(side, {k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items() if k not in ('apex',)})
json.dump(res,open(OUT+"a1_top_sections.json","w"),indent=1)
