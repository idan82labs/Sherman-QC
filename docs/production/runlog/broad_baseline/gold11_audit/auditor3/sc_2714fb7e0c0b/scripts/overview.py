import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_2714fb7e0c0b/'
V=np.load(W+'V.npy'); F=np.load(W+'frame.npy'); o=F[0]; R=F[1:]
L=(V-o)@R.T
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
gold={b['bend_id']:((np.array(b['line_start'])-o)@R.T,(np.array(b['line_end'])-o)@R.T) for b in G['bends']}
rng=np.random.default_rng(1); s=rng.choice(len(L),300000,replace=False)
fig,axs=plt.subplots(1,2,figsize=(22,11))
ax=axs[0]; sc=ax.scatter(L[s,0],L[s,1],c=L[s,2],s=0.15,cmap='turbo'); plt.colorbar(sc,ax=ax,label='Z (plate normal) mm')
for k,(a,b) in gold.items():
    ax.plot([a[0],b[0]],[a[1],b[1]],'k-',lw=2); m=(a+b)/2; ax.text(m[0]+2,m[1]+2,k,fontsize=11,bbox=dict(fc='w',alpha=0.8))
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('X (width, along B03)'); ax.set_ylabel('Y (length, along B01)'); ax.set_title('auditor3 sc_2714fb7e0c0b top view (X-Y), colour=Z; gold lines black')
ax=axs[1]
m=np.abs(L[:,0]-0)<0.5
ax.scatter(L[m,1],L[m,2],s=0.5,c='k'); ax.set_title('section X=0 (Y-Z profile, like drawing section B-B)')
for k,(a,b) in gold.items():
    d=b-a
    if abs(d[0])>1e-6:
        t=(0-a[0])/d[0]
        if 0<=t<=1:
            p=a+t*d; ax.plot(p[1],p[2],'g*',ms=14); ax.text(p[1]+1,p[2]+2,k,color='g',fontsize=12)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('Y'); ax.set_ylabel('Z')
plt.tight_layout(); plt.savefig(OUT+'a3_overview_top_and_YZ.png',dpi=70)
# sections across X at shelf and main plate
fig,axs=plt.subplots(3,1,figsize=(22,14))
for ax,y0 in zip(axs,[95,-40,105]):
    m=np.abs(L[:,1]-y0)<0.5
    ax.scatter(L[m,0],L[m,2],s=0.5,c='k'); ax.set_title(f'section Y={y0} (X-Z profile)')
    for k,(a,b) in gold.items():
        d=b-a
        if abs(d[1])>1e-6:
            t=(y0-a[1])/d[1]
            if 0<=t<=1:
                p=a+t*d; ax.plot(p[0],p[2],'g*',ms=14); ax.text(p[0]+1,p[2]+2,k,color='g',fontsize=12)
    ax.set_aspect('equal'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+'a3_sections_XZ.png',dpi=70)
print('ok')
