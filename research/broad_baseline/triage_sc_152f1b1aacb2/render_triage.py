"""Read-only triage renders for sc_152f1b1aacb2 (Gold11 CG64C412CC73). Outputs to docs/.../triage/sc_152f1b1aacb2."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='docs/production/runlog/broad_baseline/triage/sc_152f1b1aacb2/'
Q=np.load(OUT+'points_ds.npz')['points']
g=json.load(open('output/bend_label_08_40500000/bend_instance_labels.json'))['manual_bends']
G={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in g}
pr=json.load(open('docs/production/runlog/broad_baseline/runs/best_f1/sc_152f1b1aacb2/predictions_best_f1.json'))['owned_bend_regions']
Pd={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in pr}
e1=np.array([0.88,0.146,-0.453]); e1/=np.linalg.norm(e1)
e3=np.array([-0.456,0.546,-0.703]); e3-= (e3@e1)*e1; e3/=np.linalg.norm(e3)
e2=np.cross(e3,e1)
R=np.stack([e1,e2,e3]); c=Q.mean(0)
L=lambda X:(np.atleast_2d(X)-c)@R.T
q=L(Q)
lines={**{('G',k):v for k,v in G.items()},**{('P',k):v for k,v in Pd.items()}}
for k,(a,b) in lines.items(): print(k, L(a).round(1), L(b).round(1))
matched={'F1C_63882b208eadf52a','F1C_ec17d0d585495dd2','F1C_38dde97c8c7b81c4'}
def draw(ax,i,j,mask=None,lim=None):
    qq=q if mask is None else q[mask]
    ax.scatter(qq[:,i],qq[:,j],s=0.2,c='0.5',alpha=0.4,rasterized=True)
    for (t,k),(a,b) in lines.items():
        A,B=L(a)[0],L(b)[0]
        col='g' if t=='G' else ('r' if k in matched else 'm')
        ax.plot([A[i],B[i]],[A[j],B[j]],col,lw=3 if t=='G' else 1.5,ls='-' if t=='G' else '--')
        m=(A+B)/2; ax.text(m[i],m[j],k[-6:] if t=='P' else k,color=col,fontsize=7)
    ax.set_aspect('equal')
    if lim: ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
fig,axs=plt.subplots(2,2,figsize=(16,14))
draw(axs[0,0],0,1); axs[0,0].set_title('top (e1 along walls, e2 across)')
draw(axs[0,1],1,2); axs[0,1].set_title('end view e2-e3 (all points)')
m=np.abs(q[:,0])<15; draw(axs[1,0],1,2,m); axs[1,0].set_title('section |e1|<15 mm (mid wall)')
m=q[:,0]>q[:,0].max()-25; draw(axs[1,1],1,2,m); axs[1,1].set_title('section near e1 max end (last 25 mm)')
plt.tight_layout(); plt.savefig(OUT+'views_partframe.png',dpi=90)
