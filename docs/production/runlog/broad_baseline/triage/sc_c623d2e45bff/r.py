import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/'
d=np.load(P+'benchmark_v2/label_prep/sc_c623d2e45bff/points_ds.npz'); X=d['points']; lab=d['labels']
g=json.load(open(P+'benchmark_v2/label_lines/sc_c623d2e45bff/lines.json'))
gold={b['bend_id']:(np.array(b['label_line']['start']),np.array(b['label_line']['end'])) for b in g['bends']}
def preds(f,pre):
    dd=json.load(open(P+'broad_baseline/'+f)); return {pre+l['bend_id']:(np.array(l['line_start']),np.array(l['line_end'])) for l in dd['owned_bend_regions']}
rt=preds('runs/runtime/sc_c623d2e45bff/predictions_runtime.json','rt:')
bf=preds('runs/best_f1/sc_c623d2e45bff/predictions_best_f1.json','bf:')
e1=np.array([0.8532727765818074,0.5126603611243663,0.09541971953536366]); e3=np.array([-0.179642,0.458746,-0.870219]); e3-=e1*(e1@e3); e3/=np.linalg.norm(e3); e2=np.cross(e3,e1)
c=X.mean(0); T=lambda p:(np.atleast_2d(p)-c)@np.stack([e1,e2,e3]).T
Y=T(X)
np.save('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/c623/Y.npy',Y)
for k,(a,b) in {**gold,**rt,**bf}.items():
    A,B=T(a)[0],T(b)[0]; print(k, np.round(A,1), np.round(B,1), 'len',round(np.linalg.norm(B-A),1))
views=[(0,1,'e1-e2 (plan)'),(0,2,'e1-e3 (elev)'),(1,2,'e2-e3 (end)')]
def draw(ax,i,j,mask=None,lw=1):
    m=np.ones(len(Y),bool) if mask is None else mask
    ax.scatter(Y[m,i],Y[m,j],s=0.2,c='0.7')
    for k,(a,b) in gold.items():
        A,B=T(a)[0],T(b)[0]; ax.plot([A[i],B[i]],[A[j],B[j]],'g-',lw=4,alpha=.6); ax.text(A[i],A[j],k,color='g')
    for dct,col in ((rt,'b'),(bf,'r')):
        for k,(a,b) in dct.items():
            A,B=T(a)[0],T(b)[0]; ax.plot([A[i],B[i]],[A[j],B[j]],col+'-',lw=lw); ax.text(B[i],B[j],k.split(':')[1][-5:],color=col,fontsize=6)
    ax.set_aspect('equal')
fig,axs=plt.subplots(1,3,figsize=(24,8))
for ax,(i,j,t) in zip(axs,views): draw(ax,i,j); ax.set_title(t)
plt.savefig('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/c623/all.png',dpi=80)
