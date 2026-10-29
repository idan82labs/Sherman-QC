"""Read-only visual triage for sc_6e5d5a2f60f5: labelled views + cross-sections of gold / predicted lines."""
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/idantal/ShermanQC/FullProject/'
OUT=R+'docs/production/runlog/broad_baseline/overlays/sc_6e5d5a2f60f5/triage/'
P=np.load(R+'docs/production/runlog/benchmark_v2/label_prep/sc_6e5d5a2f60f5/points_ds.npz')['points']
gold=json.load(open(R+'data/bend_corpus/annotations/ai_bend_lines_v1/f1840bb889ab736f1c9efa55.json'))['bends']
pred=json.load(open(R+'docs/production/runlog/broad_baseline/runs/best_f1/sc_6e5d5a2f60f5/predictions_best_f1.json'))['owned_bend_regions']
L={('G:'+g['bend_id']):(np.array(g['line_start']),np.array(g['line_end'])) for g in gold}
for p in pred: L['P:'+p['bend_id']]=(np.array(p['line_start']),np.array(p['line_end']))
nfloor=np.array([-0.2258,0.2477,-0.9422])
c=P.mean(0); X=P-c
# top-ish view basis
right=np.array([-0.7987,-0.6007,0.0335]); up=np.cross(nfloor,right); up/=np.linalg.norm(up)
def views():
    fig,axs=plt.subplots(1,2,figsize=(22,11))
    for ax,(a,b,nm) in zip(axs,[(right,up,'top (floor normal)'),(right,nfloor,'front (edge-on to floor)')]):
        ax.scatter(X@a,X@b,s=0.2,c='0.7')
        for k,(s,e) in L.items():
            s2=s-c;e2=e-c
            col='g' if k.startswith('G') else 'r'
            ax.plot([s2@a,e2@a],[s2@b,e2@b],col,lw=3 if col=='g' else 1.5,ls='-' if col=='g' else '--')
            m=(s2+e2)/2; ax.text(m@a,m@b,k,color=col,fontsize=9)
        ax.set_aspect('equal'); ax.set_title(nm)
    fig.savefig(OUT+'labelled_views.png',dpi=90,bbox_inches='tight')
def xsec(name,stations=(0.15,0.5,0.85),half=1.0,rad=22):
    s,e=L[name]; u=(e-s)/np.linalg.norm(e-s); ln=np.linalg.norm(e-s)
    n=nfloor-(nfloor@u)*u; 
    if np.linalg.norm(n)<0.3: n=np.cross(u,right)
    n/=np.linalg.norm(n); v=np.cross(u,n)
    fig,axs=plt.subplots(1,len(stations),figsize=(6*len(stations),6))
    for ax,t in zip(axs,stations):
        q=s+t*(e-s); D=P-q; a=D@u
        m=(np.abs(a)<half)&(np.abs(D@v)<rad)&(np.abs(D@n)<rad)
        ax.scatter(D[m]@v,D[m]@n,s=3,c='k')
        for k,(s2,e2) in L.items():
            u2=(e2-s2); den=u2@u
            if abs(den)<1e-6: continue
            tt=((q-s2)@u)/den
            if -0.05<=tt<=1.05:
                pp=s2+tt*u2-q
                if abs(pp@v)<rad and abs(pp@n)<rad:
                    ax.plot(pp@v,pp@n,'o' if k.startswith('G') else 'x',ms=10,mfc='none',color='g' if k.startswith('G') else 'r',mew=2)
                    ax.text(pp@v+0.5,pp@n+0.5,k,fontsize=8,color='g' if k.startswith('G') else 'r')
        ax.set_aspect('equal'); ax.set_xlim(-rad,rad); ax.set_ylim(-rad,rad); ax.grid(alpha=.3)
        ax.set_title(f'{name} t={t:.2f} (len {ln:.0f}mm)')
    fig.savefig(OUT+'xsec_'+name.replace(':','_')+'.png',dpi=80,bbox_inches='tight'); plt.close(fig)
views()
for k in L: xsec(k)
# also sections ALONG the line (profile of points near line) - distance from line to nearest points
from scipy.spatial import cKDTree
T=cKDTree(P)
for k,(s,e) in L.items():
    ts=np.linspace(0,1,21); pts=s[None]+ts[:,None]*(e-s)[None]
    d,_=T.query(pts); cnt=[len(T.query_ball_point(p,2.0)) for p in pts]
    print(k, 'len %.1f'%np.linalg.norm(e-s), 'nn_dist med %.2f max %.2f'%(np.median(d),d.max()), 'pts<2mm min',min(cnt))
