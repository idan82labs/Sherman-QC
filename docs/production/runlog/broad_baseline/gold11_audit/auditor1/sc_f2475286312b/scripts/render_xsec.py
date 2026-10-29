import json
from folds import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_f2475286312b/'
AX=['u','v','w']
def circ(ax,c,r,**kw):
    t=np.linspace(0,2*np.pi,120); ax.plot(c[0]+r*np.cos(t),c[1]+r*np.sin(t),**kw)
def panel(ax, fd_name, x0, golds, half=0.4, win=12.0, title=''):
    """fd_name: fold used for slab orientation (or dict for raw slab); golds: list of (id, line part-frame endpoints)"""
    fd=FOLDS[fd_name]; axi=fd['ax']; i,j=fd['cc']
    m=np.abs(P[:,axi]-x0)<half; Q=P[m][:,[i,j]]
    st=station(fd,x0,half=half)
    gp2=[]
    for gid,(gs,ge) in golds:
        if abs(ge[axi]-gs[axi])<1e-6: continue
        t=(x0-gs[axi])/(ge[axi]-gs[axi]); inside=0<=t<=1
        p=gs+t*(ge-gs); gp2.append((gid,np.array([p[i],p[j]]),inside))
    # window centre
    pts=[g2 for _,g2,_ in gp2]
    if st is not None and st.get('ok'): pts.append(st['apex'])
    cen=np.mean(pts,axis=0) if pts else np.array(fd['c0'])
    k=(np.abs(Q[:,0]-cen[0])<win)&(np.abs(Q[:,1]-cen[1])<win)
    ax.scatter(Q[k,0],Q[k,1],s=4,c='0.25',zorder=1)
    txt=[]
    if st is not None and st.get('ok'):
        a=st['apex']; ax.plot(*a,'D',color='c',mec='k',ms=8,zorder=5,label='outer apex (scan)')
        circ(ax,a,4.0,color='c',ls=':',lw=1.2)
        ax.plot(*st['Xo'],'x',color='tab:blue',ms=9,mew=2,label='outer virtual sharp')
        if st['Xi'] is not None: ax.plot(*st['Xi'],'x',color='tab:purple',ms=9,mew=2,label='inner virtual sharp')
        ob=st['ob']; ax.plot([a[0]-ob[0]*win,a[0]+ob[0]*win],[a[1]-ob[1]*win,a[1]+ob[1]*win],':',color='tab:orange',lw=0.8)
    for gid,g2,inside in gp2:
        ax.plot(*g2,'*',color='#1a9c3a' if inside else 'orange',mec='k',ms=15,zorder=6)
        ax.annotate(gid+('' if inside else ' (beyond gold span)'),g2,xytext=(5,5),textcoords='offset points',color='#1a9c3a',fontsize=10,weight='bold')
        if st is not None and st.get('ok'):
            dl=g2-st['apex']; ob=st['ob']
            txt.append(f"{gid}: |g-apex|={np.linalg.norm(dl):.2f}  concave-along={-(dl@ob):+.2f}  across={dl@np.array([-ob[1],ob[0]]):+.2f} mm")
    if st is None or not st.get('ok'): txt.append('no fold corner at this station (fit failed: '+str(None if st is None else st.get('why'))+')')
    ax.set_xlim(cen[0]-win,cen[0]+win); ax.set_ylim(cen[1]-win,cen[1]+win); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel(AX[i]); ax.set_ylabel(AX[j])
    ax.set_title(title+f"\nslab |{AX[axi]}-({x0:.1f})|<{half}  "+"\n".join(txt),fontsize=8.5)
gp={k:(R@s,R@e) for k,(s,e,_) in g.items()}
def fig_for(name, fold, stations, gids, win=12.0, ncol=3):
    n=len(stations); nrow=int(np.ceil(n/ncol))
    fig,axs=plt.subplots(nrow,ncol,figsize=(6.3*ncol,6.6*nrow)); axs=np.atleast_1d(axs).ravel()
    for ax,(x0,lab) in zip(axs,stations):
        panel(ax,fold,x0,[(k,gp[k]) for k in gids],win=win,title=f'{name} {lab}')
    for ax in axs[n:]: ax.axis('off')
    axs[0].legend(loc='lower left',fontsize=7)
    fig.suptitle(f'sc_f2475286312b auditor1: {name} perpendicular cross-sections (grey=scan slab, green star=gold, cyan diamond=outer apex with 4 mm matcher circle)',fontsize=10)
    plt.tight_layout(); plt.savefig(OUT+f'a1_xsec_{name}.png',dpi=72); plt.close(fig)
fig_for('M01','F1_upflange_v-54',[(-470,'25% fold'),(-376,'50% fold'),(-290,'90% fold'),(-268.5,'fold end'),(-250,'gold overrun'),(-215,'gold overrun (past wall)')],['M01'])
fig_for('M02','F2_upflange_v-278',[(-470,'25% fold'),(-376,'50% fold'),(-290,'90% fold'),(-268.5,'fold end'),(-250,'gold overrun'),(-215,'gold overrun (past wall)')],['M02'])
fig_for('M07','F4_endwall_u-224',[(-262,'10% fold'),(-210,'30%'),(-166,'50%'),(-110,'75%'),(-64,'96%'),(-58,'fold end region')],['M07'])
fig_for('M03_M04','F5_slotflange_v-85',[(-450,'outside slot (no fold)'),(-380,'slot flange 25%'),(-360,'slot flange 50%'),(-335,'slot flange 80%'),(-290,'outside slot'),(-240,'outside slot')],['M03','M04'],win=13)
fig_for('M05_M06','F6_slotflange_v-258',[(-450,'outside slot (no fold)'),(-380,'slot flange 25%'),(-360,'slot flange 50%'),(-335,'slot flange 80%'),(-290,'outside slot'),(-240,'outside slot')],['M05','M06'],win=13)
