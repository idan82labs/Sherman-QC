import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:((np.array(b['line_start'])-o)@R.T,(np.array(b['line_end'])-o)@R.T) for b in G['bends']}
fig,axs=plt.subplots(2,4,figsize=(26,16))
for i,(bid,xc) in enumerate((('B09',-78.0),('B10',88.0))):
    for j,z0 in enumerate([-12.3,-14.5,-19,-25]):
        ax=axs[i,j]
        m=(np.abs(L[:,2]-z0)<0.3)&(np.abs(L[:,0]-xc)<16)&(L[:,1]>-75)&(L[:,1]<40)
        col=np.arctan2(NL[m,1],NL[m,0])
        ax.scatter(L[m,0],L[m,1],c=col,cmap='hsv',s=2,vmin=-np.pi,vmax=np.pi)
        a,b=gold[bid]; ax.plot([a[0],b[0]],[a[1],b[1]],'g-',lw=2,alpha=.6)
        ax.set_title(f'{bid}: horizontal slice Z={z0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-16,xc+16); ax.set_ylim(-75,40)
plt.tight_layout(); plt.savefig(OUT+'a3_tabs_hslices.png',dpi=60); plt.close()
fig,axs=plt.subplots(2,6,figsize=(30,11))
for i,(bid,xc) in enumerate((('B09',-80.0),('B10',90.0))):
    for j,y0 in enumerate([-66,0,12,16,20,24]):
        ax=axs[i,j]
        m=(np.abs(L[:,1]-y0)<0.3)&(np.abs(L[:,0]-xc)<14)&(L[:,2]<0)
        ax.scatter(L[m,0],L[m,2],s=3,c='k')
        a,b=gold[bid]; d=b-a; t=(y0-a[1])/d[1]
        if 0<=t<=1: p=a+t*d; ax.plot(p[0],p[2],'g*',ms=14)
        ax.set_title(f'{bid}: X-Z section Y={y0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-14,xc+14); ax.set_ylim(-30,-5)
plt.tight_layout(); plt.savefig(OUT+'a3_tabs_XZ_sections.png',dpi=60); plt.close()
print('ok')
