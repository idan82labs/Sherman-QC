"""One-page triage figure: representative sections with gold (green o), lane lines (x) and the measured mid-thickness fold (black star)."""
import sys,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
src=open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787/vsharp.py').read().split('for g,(s,e) in G.items():')[0]
exec(src)
for k,v in G.items(): L['gold:'+k]=v
cols={'gold':'g','best':'r','cadg':'m','zs':'b'}
panels=[('gold:HB1',0.7,'HB1 = TP (gold 3.3 mm inner side, CAD_B2 0.2 mm)'),
        ('gold:HB2',0.4,'HB2 FN: best_f1 has no line (zero-shot OB1 would match)'),
        ('gold:HB3',0.2,'HB3 FN: gold 4.6 mm down plate face; F1C/CAD_B1 on fold'),
        ('gold:HB3',0.7,'HB3 @70%: no fold - plain plate edge (gold+preds overlong)'),
        ('gold:HB4',0.8,'HB4 FN: best_f1 nothing; gold 5-6 mm off fold'),
        ('gold:HB5',0.5,'HB5 FN: gold 8 mm up tab face; CAD_B5 3 mm from fold'),
        ('best:CADF_CAD_B5',0.15,'UNLABELED tab-1 fold on CAD_B5 axis (no gold)'),
        ('best:CADF_CAD_B5',0.5,'CAD_B5 @50%: plain edge, no fold (merged 2 tabs)')]
fig,ax=plt.subplots(2,4,figsize=(24,12.5)); W=20
for a_,(name,f,title) in zip(ax.flat,panels):
    s,e=L[name]; d=(e-s)/np.linalg.norm(e-s); aa=np.array([0,0,1.]); e1=np.cross(d,aa); e1/=np.linalg.norm(e1); e2=np.cross(d,e1)
    c=s+f*(e-s); q=P-c; m=np.abs(q@d)<0.75; u=q[m]@e1; v=q[m]@e2; r=(np.abs(u)<W)&(np.abs(v)<W)
    a_.scatter(u[r],v[r],s=1.5,c='0.35')
    rr=u*u+v*v<12**2; vs,sep=fit2(u[rr],v[rr],np.c_[N[m][rr]@e1,N[m][rr]@e2])
    if vs is not None and np.linalg.norm(vs)<12 and 'no fold' not in title and 'plain edge' not in title: a_.plot(*vs,'k*',ms=16,label='fold (mid-thickness virtual sharp)')
    for k,(ps,pe) in L.items():
        pd=pe-ps
        if abs(pd@d)/np.linalg.norm(pd)<np.cos(np.radians(10)): continue
        tt=((c-ps)@d)/(pd@d); x=ps+tt*pd-c
        if abs(x@e1)<W and abs(x@e2)<W and -0.05<=tt<=1.05:
            g=k.split(':')[0]; a_.plot(x@e1,x@e2,'o' if g=='gold' else 'x',mfc='none',ms=13,mew=2.5,color=cols[g],label=k)
    a_.set_aspect('equal'); a_.set_xlim(-W,W); a_.set_ylim(-W,W); a_.grid(alpha=.3); a_.set_title(title,fontsize=10); a_.legend(fontsize=7,loc='lower left')
plt.suptitle('sc_a0305a6cf787 (47959001_01PREL_scan1) triage: sections perpendicular to the named line; green=gold, red=best_f1, magenta=cad_gated, blue=zero-shot decomposition (not in best_f1)')
plt.tight_layout(); plt.savefig(OUT+'triage_summary.png',dpi=55); print('ok')
