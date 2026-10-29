import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T; NL=N@R.T
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:((np.array(b['line_start'])-o)@R.T,(np.array(b['line_end'])-o)@R.T) for b in G['bends']}
zs=[30.0,31.5,34,38,42,44.5]
for bid,xc in (('B06',-112.0),('B08',111.5)):
    fig,axs=plt.subplots(2,3,figsize=(18,12))
    for ax,z0 in zip(axs.flat,zs):
        m=(np.abs(L[:,2]-z0)<0.3)&(np.abs(L[:,0]-xc)<11)&(np.abs(L[:,1]-117)<11)
        col=np.arctan2(NL[m,1],NL[m,0])
        ax.scatter(L[m,0],L[m,1],c=col,cmap='hsv',s=4,vmin=-np.pi,vmax=np.pi)
        a,b=gold[bid]; d=b-a; t=(z0-a[2])/d[2]
        if -0.05<=t<=1.05:
            p=a+t*d; ax.plot(p[0],p[1],'k*',ms=16,mfc='lime'); ax.add_patch(plt.Circle((p[0],p[1]),4,fill=False,ls=':',color='g'))
        # B04 gold at this z? (B04 is horizontal at z~31)
        ax.set_title(f'{bid} corner: horizontal slice Z={z0} (+-0.3mm)  colour=normal azimuth')
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_xlim(xc-11,xc+11); ax.set_ylim(106,128)
    plt.tight_layout(); plt.savefig(OUT+f'a3_{bid}_corner_hslices.png',dpi=70); plt.close()
# vertical sections through the shelf side flanges near the corner and mid (X-Z at fixed Y) and (Y-Z at fixed X) 
fig,axs=plt.subplots(2,4,figsize=(24,12))
for j,y0 in enumerate([80,100,112,117.5]):
    for i,(xc,lab) in enumerate(((-112.0,'left(B06 side)'),(111.5,'right(B08 side)'))):
        ax=axs[i,j]
        m=(np.abs(L[:,1]-y0)<0.3)&(np.abs(L[:,0]-xc)<12)&(L[:,2]>22)
        ax.scatter(L[m,0],L[m,2],s=4,c='k')
        for bid in ('B06','B08'):
            a,b=gold[bid]
            ax.plot([a[0],b[0]],[a[2],b[2]],'g-',lw=1); 
        ax.set_title(f'{lab} X-Z section at Y={y0}'); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(xc-12,xc+12); ax.set_ylim(24,48)
plt.tight_layout(); plt.savefig(OUT+'a3_shelf_sideflange_XZ_sections.png',dpi=70); plt.close()
print('ok')
