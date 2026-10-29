import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:((np.array(b['line_start'])-o)@R.T,(np.array(b['line_end'])-o)@R.T) for b in G['bends']}
fig,axs=plt.subplots(2,3,figsize=(27,13))
for i,xc in enumerate((-104.0,104.0)):
    for j,z0 in enumerate([-11.0,-7.0,-2.0]):
        ax=axs[i,j]
        m=(np.abs(L[:,2]-z0)<0.3)&(np.abs(L[:,0]-xc)<14)&(L[:,1]>-152)&(L[:,1]<-128)
        col=np.arctan2(NL[m,1],NL[m,0])
        ax.scatter(L[m,0],L[m,1],c=col,cmap='hsv',s=5,vmin=-np.pi,vmax=np.pi)
        a,b=gold['B07']; ax.plot([a[0],b[0]],[a[1],b[1]],'g-',lw=2,alpha=.7,label='gold B07 (XY proj.)')
        ax.axvline(-101.6 if xc<0 else 100.4,color='b',ls='--',label='auditor3 fold end')
        ax.set_title(f'B07 end X~{xc}: horizontal slice Z={z0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-14,xc+14); ax.set_ylim(-152,-128); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT+'a3_B07_ends_hslices.png',dpi=55); plt.close()
print('ok')
