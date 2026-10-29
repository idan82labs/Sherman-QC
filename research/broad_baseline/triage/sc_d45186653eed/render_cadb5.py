"""Read-only: cross-sections perpendicular to CAD_B5 (spurious 131 deg) and B9 for sc_d45186653eed."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/idantal/ShermanQC/FullProject/'
OUT=R+'docs/production/runlog/broad_baseline/overlays/sc_d45186653eed/triage/'
z=np.load(R+'docs/production/runlog/benchmark_v2/label_prep/sc_d45186653eed/points_ds.npz'); pts=z['points']; lab=z['labels']
G={b['bend_id']:b for b in json.load(open(R+'docs/production/runlog/benchmark_v2/label_lines/sc_d45186653eed/lines.json'))['bends']}
P={l['bend_id']:l for l in json.load(open(R+'docs/production/runlog/broad_baseline/runs/runtime/sc_d45186653eed/predictions_runtime.json'))['owned_bend_regions']}
Pb={l['bend_id']:l for l in json.load(open(R+'docs/production/runlog/broad_baseline/runs/best_f1/sc_d45186653eed/predictions_best_f1.json'))['owned_bend_regions']}
others={'G'+k:(np.array(b['label_line']['start']),np.array(b['label_line']['end'])) for k,b in G.items()}
others.update({'rt_'+k:(np.array(l['line_start']),np.array(l['line_end'])) for k,l in P.items()})
others.update({'bf_'+k:(np.array(l['line_start']),np.array(l['line_end'])) for k,l in Pb.items()})
def xsec(ax,s,e,f,title,half=1.0,win=25):
    d=(e-s)/np.linalg.norm(e-s); q=s+f*(e-s)
    a=np.cross(d,[0,0,1.0]); a/=np.linalg.norm(a); b=np.cross(d,a)
    X=pts-q; t=X@d; m=(np.abs(t)<half)&(np.abs(X@a)<win)&(np.abs(X@b)<win)
    ax.scatter(X[m]@a,X[m]@b,s=4,c=lab[m],cmap='tab20')
    for k,(s2,e2) in others.items():
        d2=e2-s2
        den=d2@d
        if abs(den)<1e-6: continue
        tt=((q-s2)@d)/den
        if -0.05<=tt<=1.05:
            p=s2+tt*d2-q
            if abs(p@a)<win and abs(p@b)<win:
                ax.plot(p@a,p@b,'o' if k[0]=='G' else 'x',ms=10,mfc='none',mew=2,color='g' if k[0]=='G' else ('r' if k.startswith('bf') else 'b')); ax.text(p@a+.5,p@b+.5,k,fontsize=7)
    ax.plot(0,0,'k+',ms=20); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(title,fontsize=9)
fig,axs=plt.subplots(2,3,figsize=(24,15))
s,e=np.array(P['CAD_B5']['line_start']),np.array(P['CAD_B5']['line_end'])
for ax,f in zip(axs[0],[0.2,0.5,0.8]): xsec(ax,s,e,f,'xsec perp to rt CAD_B5 at %.0f%% (+ = CAD_B5)'%(f*100))
s,e=np.array(G['B9']['label_line']['start']),np.array(G['B9']['label_line']['end'])
for ax,f in zip(axs[1],[0.2,0.5,0.8]): xsec(ax,s,e,f,'xsec perp to gold B9 at %.0f%% (+ = B9)'%(f*100))
fig.savefig(OUT+'cadb5_b9_xsec.png',dpi=65,bbox_inches='tight')
# 3D-ish view of the end corner
from mpl_toolkits.mplot3d import Axes3D
c=np.array(P['CAD_B5']['line_start']); m=np.linalg.norm(pts-c,axis=1)<45
fig=plt.figure(figsize=(18,9))
for i,(el,az) in enumerate([(20,30),(20,150)]):
    ax=fig.add_subplot(1,2,i+1,projection='3d'); ax.scatter(*pts[m].T,s=0.5,c=lab[m],cmap='tab20',alpha=.4)
    for k,(s2,e2) in others.items():
        if min(np.linalg.norm(s2-c),np.linalg.norm(e2-c))<60:
            col='g' if k[0]=='G' else ('r' if k.startswith('bf') else 'b'); ax.plot(*np.stack([s2,e2]).T,'-' if k[0]=='G' else '--',color=col,lw=3 if k[0]=='G' else 1.5); ax.text(*((s2+e2)/2),k,fontsize=7,color=col)
    ax.view_init(el,az); ax.set_box_aspect(np.ptp(pts[m],0))
fig.savefig(OUT+'end_corner_3d.png',dpi=70,bbox_inches='tight')
