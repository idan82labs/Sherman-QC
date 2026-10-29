import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage/triage_f247.py').read().split('# top & two')[0])
h=np.load(S+'h.npy')
fig,ax=plt.subplots(2,2,figsize=(14,12))
for r,(hz) in enumerate([20,40]):
  for cidx,ys in enumerate([-1,1]):
    a=ax[r,cidx]; m=(np.abs(h-hz)<1.0)&(Q[:,0]<-95)&(ys*Q[:,1]>95)
    a.scatter(Q[m,0],Q[m,1],s=3,c='k')
    for D,col in [(G,'g'),(B,'r')]:
        for k,(s0,e0) in D.items():
            s1,e1=T(s0),T(e0); a.plot([s1[0],e1[0]],[s1[1],e1[1]],'-',color=col,lw=1.5); 
    a.set_xlim(-130,-95); a.set_ylim(*(sorted([ys*95,ys*120]))); a.set_aspect('equal'); a.set_title(f'slab h={hz}+-1mm, {"-" if ys<0 else "+"}PC2 end (wall/ear corner)'); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+'triage_zoom_ears.png',dpi=60)
