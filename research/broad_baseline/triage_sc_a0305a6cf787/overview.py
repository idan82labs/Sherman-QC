"""Overview + close-ups in the part PCA frame, points coloured by |normal . PC3| (flat plate vs bent flanges), all lane lines labelled."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/Users/idantal/ShermanQC/FullProject/research/broad_baseline/triage_sc_a0305a6cf787')
from common import *
P,N=load_points(); G=gold()
c0=P.mean(0); _,_,V=np.linalg.svd(P[::20]-c0,full_matrices=False)
Q=(P-c0)@V.T; NQ=N@V.T
L=dict(('gold:'+k,v) for k,v in G.items())
for n,tag in [('predictions_best_f1','best'),('predictions_cad_gated','cadg'),('predictions_f1_zero_shot_decomposition','zs')]:
    for k,v in preds(n).items(): L[tag+':'+k]=v
cols={'gold':'g','best':'r','cadg':'m','zs':'b'}; lw={'gold':4,'best':2,'cadg':1.2,'zs':1.5}; ls={'gold':'-','best':'--','cadg':':','zs':'-.'}
T=lambda p:(p-c0)@V.T
def draw(a,i,j,box=None,title=''):
    m=np.ones(len(Q),bool)
    if box is not None:
        for ax_,(lo,hi) in box.items(): m&=(Q[:,ax_]>lo)&(Q[:,ax_]<hi)
    o=np.argsort(Q[m,3-i-j])
    a.scatter(Q[m,i][o],Q[m,j][o],s=0.6 if box is None else 2,c=np.abs(NQ[m,2])[o],cmap='coolwarm_r',vmin=0,vmax=1)
    for k,(s,e) in L.items():
        g=k.split(':')[0]; s1,e1=T(s),T(e)
        a.plot([s1[i],e1[i]],[s1[j],e1[j]],ls[g],color=cols[g],lw=lw[g])
        a.annotate(k.split(':')[1][-6:],tuple((0.5*(s1+e1))[[i,j]]),color=cols[g],fontsize=8)
    if box is not None:
        a.set_xlim(*box.get(i,(Q[m,i].min(),Q[m,i].max()))); a.set_ylim(*box.get(j,(Q[m,j].min(),Q[m,j].max())))
    a.set_aspect('equal'); a.set_title(title,fontsize=10); a.grid(alpha=.25)
fig,ax=plt.subplots(2,2,figsize=(20,16))
draw(ax[0,0],0,1,title='top PC1-PC2 (colour: blue=normal along PC3 i.e. main plate, red=bent/edge)')
draw(ax[0,1],0,2,title='front PC1-PC3')
draw(ax[1,0],1,2,title='side PC2-PC3')
# close-up near the two tabs along CAD_B5 edge
s,e=T(L['best:CADF_CAD_B5'][0]),T(L['best:CADF_CAD_B5'][1])
lo=np.minimum(s,e)-15; hi=np.maximum(s,e)+15
draw(ax[1,1],0,2,box={0:(lo[0],hi[0]),2:(lo[2],hi[2]),1:(lo[1]-5,hi[1]+5)},title='close-up: band around CAD_B5 (front view)')
plt.suptitle('sc_a0305a6cf787 47959001_01PREL_scan1: green=gold, red dashed=best_f1, magenta dotted=cad_gated, blue dash-dot=zero-shot decomposition')
plt.tight_layout(); plt.savefig(OUT+'overview_normals.png',dpi=55); plt.close()
np.savez(OUT+'../../../../../../research/broad_baseline/triage_sc_a0305a6cf787/.pca.npz' if False else '/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a030_pca.npz',c0=c0,V=V)
print('ok')
