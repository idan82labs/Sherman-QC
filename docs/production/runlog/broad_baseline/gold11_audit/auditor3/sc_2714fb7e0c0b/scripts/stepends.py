import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
fig,axs=plt.subplots(2,5,figsize=(30,12))
for i,xc in enumerate((-110.0,109.0)):
    for j,z0 in enumerate([-8,0,10,20,27]):
        ax=axs[i,j]
        m=(np.abs(L[:,2]-z0)<0.3)&(np.abs(L[:,0]-xc)<12)&(L[:,1]>55)&(L[:,1]<90)
        col=np.arctan2(NL[m,1],NL[m,0])
        ax.scatter(L[m,0],L[m,1],c=col,cmap='hsv',s=5,vmin=-np.pi,vmax=np.pi)
        ax.set_title(f'step-wall end X~{xc}: horizontal slice Z={z0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-12,xc+12); ax.set_ylim(55,90)
plt.tight_layout(); plt.savefig(OUT+'a3_stepwall_ends_hslices.png',dpi=60); plt.close()
print('ok')
