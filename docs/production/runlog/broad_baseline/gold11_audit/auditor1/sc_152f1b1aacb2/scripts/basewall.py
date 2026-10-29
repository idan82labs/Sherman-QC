import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())

def fold_apex_section(Q2, sgn, which):
    """Q2: Nx2 (u=e2, v=e3) points in slab. sgn: -1 left wall (wall at e2~-92), +1 right.
    which='base': base-wall fold at bottom. Returns dict with outer sharp, apex, inner sharp etc."""
    u = sgn*Q2[:,0]; v = Q2[:,1]   # mirror so wall is at u~+92, base extends to smaller u
    # base plate region: u in [78, 86], v < -5
    bm = (u>76)&(u<86)&(v<-4)
    if bm.sum()<10: return None
    vb = v[bm]; ub=u[bm]
    vmid = np.median(vb)
    lo = bm.copy(); lo[bm] = vb < vmid   # lower skin (outer) of base
    hi = bm.copy(); hi[bm] = vb >= vmid
    # wall region: v in [-1, 12], u in [86, 96]
    wm = (u>86)&(u<97)&(v>-1.5)&(v<12)
    if wm.sum()<10: return None
    uw = u[wm]; umid=np.median(uw)
    wo = wm.copy(); wo[wm] = uw > umid   # outer skin of wall = larger mirrored u
    wi = wm.copy(); wi[wm] = uw <= umid
    def fitline(x,y):
        A=np.vstack([x,np.ones_like(x)]).T; k,b=np.linalg.lstsq(A,y,rcond=None)[0]; return k,b
    kb_o = fitline(u[lo], v[lo])    # v = k*u + b (outer base)
    kb_i = fitline(u[hi], v[hi])
    kw_o = fitline(v[wo], u[wo])    # u = k*v + b (outer wall)
    kw_i = fitline(v[wi], u[wi])
    def inter(kb, kw):
        # v = kb0*u+kb1 ; u = kw0*v + kw1
        k1,b1=kb; k2,b2=kw
        vv = (k1*b2 + b1)/(1-k1*k2); uu = k2*vv + b2
        return np.array([uu,vv])
    Xo = inter(kb_o, kw_o); Xi = inter(kb_i, kw_i)
    # directions of the two legs from the sharp, pointing along the material
    db = np.array([-1.0, -kb_o[0]]); db/=np.linalg.norm(db)       # base goes toward smaller u
    dw = np.array([kw_o[0], 1.0]); dw/=np.linalg.norm(dw)         # wall goes up (+v)
    bis = db+dw; bis/=np.linalg.norm(bis)                          # into the inside
    ang = np.degrees(np.arccos(np.clip(np.dot(db,dw),-1,1)))
    bend_angle = 180-ang
    # apex: points near bisector ray from Xo, within 6 mm
    rel = np.vstack([u,v]).T - Xo
    along = rel@bis; perp = rel@np.array([-bis[1],bis[0]])
    near = (np.abs(perp)<0.35)&(along>-0.5)&(along<6)
    apex=None
    if near.sum():
        j = np.argmin(np.where(near, along, np.inf)); apex = np.array([u[j],v[j]])
    tb = abs((kb_i[1]-kb_o[1]))/np.sqrt(1+kb_o[0]**2)
    tw = abs((kw_o[1]-kw_i[1]))/np.sqrt(1+kw_o[0]**2)
    unm = lambda p: None if p is None else np.array([sgn*p[0], p[1]])
    return dict(Xo=unm(Xo), Xi=unm(Xi), apex=unm(apex), bis=np.array([sgn*bis[0],bis[1]]), bend_angle=bend_angle, t_base=tb, t_wall=tw)

stations = [-100,-90,-80,-70,-66,-64,-62,-58,-45,-35,-15,10,20,40,58,62,64,66,70,80,90,98]
results = {}
for side, sgn, gid in (("left",-1,"M01"),("right",1,"M02")):
    fig, axs = plt.subplots(4,6, figsize=(30,20)); axs=axs.ravel()
    rows=[]
    for i,s in enumerate(stations):
        ax=axs[i]
        m = (np.abs(P[:,0]-s)<0.5)&(sgn*P[:,1]>70)&(P[:,2]<15)
        Q=P[m][:,1:3]
        ax.scatter(Q[:,0],Q[:,1],s=1.5,c='k')
        g,inside = gold_at(G[gid],0,s)
        info = fold_apex_section(Q, sgn, 'base') if len(Q)>30 else None
        row = dict(e1=s, gold_inside_span=bool(inside), n=int(len(Q)))
        if info is not None and info['apex'] is not None:
            ax.plot(*info['Xo'],'bx',ms=12,mew=2,label='outer virtual sharp')
            ax.plot(*info['Xi'],'mx',ms=12,mew=2,label='inner virtual sharp')
            ax.plot(*info['apex'],'cD',ms=9,mec='k',label='outer apex')
            row.update(apex=info['apex'].tolist(), Xo=info['Xo'].tolist(), bend_angle=round(float(info['bend_angle']),1), t_base=round(float(info['t_base']),2), t_wall=round(float(info['t_wall']),2))
        if g is not None:
            ax.plot(g[1],g[2],'*',ms=18,mfc='lime' if inside else 'none',mec='g',mew=2,label=gid+(' (in span)' if inside else ' (extrapolated, OUT of span)'))
            if info is not None and info['apex'] is not None and inside:
                d = np.array([g[1],g[2]])-info['apex']
                row.update(gold_minus_apex=d.tolist(), d_apex=float(np.linalg.norm(d)), d_sharp=float(np.linalg.norm(np.array([g[1],g[2]])-info['Xo'])), along_bis=float(d@info['bis']))
        cx = sgn*91.5
        ax.set_xlim(cx-10,cx+10); ax.set_ylim(-14,6); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ttl = f"{gid} e1={s}"
        if 'd_apex' in row: ttl += f"\n gold-apex {row['d_apex']:.2f} mm (along bis {row['along_bis']:+.2f}), ang {row.get('bend_angle')}"
        elif 'bend_angle' in row: ttl += f"\n fold ang {row['bend_angle']}"
        else: ttl += "\n no fold"
        ax.set_title(ttl, fontsize=10)
        if i==0: ax.legend(fontsize=7, loc='lower left')
        rows.append(row)
    fig.suptitle(f"{side} base-wall fold cross-sections (e2-e3 plane, slab |e1-s|<0.5). Gold {gid} spans e1 {G[gid][0,0]:.1f}..{G[gid][1,0]:.1f}", fontsize=14)
    plt.tight_layout(); plt.savefig(OUT+f"a1_xsec_basewall_{side}_{gid}.png", dpi=60); plt.close()
    results[gid]=rows
    for r in rows:
        print(gid, r['e1'], 'inspan' if r['gold_inside_span'] else 'OUT', r.get('bend_angle'), 'd_apex=%s'%(None if 'd_apex' not in r else round(r['d_apex'],2)), 'bis=%s'%(None if 'along_bis' not in r else round(r['along_bis'],2)), 'd_sharp=%s'%(None if 'd_sharp' not in r else round(r['d_sharp'],2)), 't', r.get('t_base'), r.get('t_wall'))
json.dump(results, open(OUT+"a1_basewall_sections.json","w"), indent=1, default=float)
