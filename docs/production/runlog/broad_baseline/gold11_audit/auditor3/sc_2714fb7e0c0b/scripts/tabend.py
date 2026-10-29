import numpy as np, json, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
# zoomed horizontal slices of tab ends
fig,axs=plt.subplots(2,5,figsize=(30,13))
for i,(bid,xc) in enumerate((('B09',-82.0),('B10',92.0))):
    for j,z0 in enumerate([-11.6,-12.6,-13.6,-15.0,-20.0]):
        ax=axs[i,j]
        m=(np.abs(L[:,2]-z0)<0.25)&(np.abs(L[:,0]-xc)<10)&(L[:,1]>2)&(L[:,1]<32)
        col=np.arctan2(NL[m,1],NL[m,0])
        ax.scatter(L[m,0],L[m,1],c=col,cmap='hsv',s=6,vmin=-np.pi,vmax=np.pi)
        ax.set_title(f'{bid} tab end: Z={z0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-10,xc+10); ax.set_ylim(2,32)
plt.tight_layout(); plt.savefig(OUT+'a3_tabend_hslices_zoom.png',dpi=60); plt.close()
# section perpendicular to the angled segment of the B09 tab: estimate angled wall direction from slice Z=-20
res={}
for bid,xc,sgn in (('B09',-82.0,-1),('B10',92.0,+1)):
    m=(np.abs(L[:,2]+20)<0.5)&(np.abs(L[:,0]-xc)<8)&(L[:,1]>14.5)&(L[:,1]<19.5)
    P=L[m][:,:2]; c=P.mean(0); U,S,Vt=np.linalg.svd(P-c); d=Vt[0]
    if d[1]<0: d=-d
    ang=math.degrees(math.atan2(abs(d[0]),d[1]))
    res[bid]=(c,d,ang); print(bid,'angled wall centre',c.round(2),'dir',d.round(3),'angle from Y %.1f deg'%ang, 'n',m.sum())
fig,axs=plt.subplots(2,4,figsize=(26,12))
for i,bid in enumerate(('B09','B10')):
    c,d,ang=res[bid]
    nrm=np.array([d[0],d[1],0.0]); # section plane normal = along the angled wall
    perp=np.array([-d[1],d[0],0.0])
    for j,off in enumerate([-2.5,0.0,2.0,3.5]):
        ax=axs[i,j]
        p0=np.array([c[0],c[1],0])+nrm*off
        s=(L-p0)@nrm
        m=(np.abs(s)<0.3)&(np.abs((L-p0)@perp)<12)&(L[:,2]<-5)&(L[:,2]>-30)
        ax.scatter((L[m]-p0)@perp,L[m,2],s=4,c='k')
        ax.set_title(f'{bid}: section perpendicular to angled tab-end wall, offset {off} mm along it'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(-12,12); ax.set_ylim(-30,-6)
plt.tight_layout(); plt.savefig(OUT+'a3_tabend_perp_sections.png',dpi=60); plt.close()
print('ok')
