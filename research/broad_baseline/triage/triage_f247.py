import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/f247/'
OUT='/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/overlays/sc_f2475286312b/'
R='/Users/idantal/ShermanQC/FullProject/'
d=np.load(S+'pts.npz'); P=d['P']; N=d['N']
cat=json.load(open(R+'data/bend_corpus/annotations/bend_line_gold_catalog_v1.json'))
def walk(o):
    if isinstance(o,dict):
        if o.get('case_id')=='corpus/07_38172000': return o
        for v in o.values():
            r=walk(v)
            if r: return r
    elif isinstance(o,list):
        for v in o:
            r=walk(v)
            if r: return r
G={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in walk(cat)['bends']}
pr=json.load(open(R+'docs/production/runlog/broad_baseline/runs/best_f1/sc_f2475286312b/predictions_best_f1.json'))
B={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in pr['owned_bend_regions']}
pp=json.load(open(R+'docs/production/runlog/broad_baseline/runs/runtime/sc_f2475286312b/predictions_runtime_with_placeholders.json'))
C={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in pp['owned_bend_regions']}
c=P.mean(0); U,s,Vt=np.linalg.svd(P-c,full_matrices=False); V=Vt
T=lambda x:(x-c)@V.T
Q=T(P)
np.save(S+'frame.npy',np.vstack([c,V]))
# top & two side views with labels
fig,ax=plt.subplots(1,3,figsize=(27,9))
for a,(i,j) in zip(ax,[(0,1),(0,2),(1,2)]):
    a.scatter(Q[::3,i],Q[::3,j],s=0.2,c=Q[::3,3-i-j],cmap='viridis',alpha=0.4)
    for k,(s0,e0) in G.items():
        s1,e1=T(s0),T(e0); a.plot([s1[i],e1[i]],[s1[j],e1[j]],'g-',lw=3,alpha=.7); a.text(*(0.5*(s1+e1))[[i,j]],k,color='g',fontsize=11)
    for k,(s0,e0) in B.items():
        s1,e1=T(s0),T(e0); a.plot([s1[i],e1[i]],[s1[j],e1[j]],'r-',lw=1.2); a.text(*(0.3*s1+0.7*e1)[[i,j]],k[-4:],color='r',fontsize=9)
    for k,(s0,e0) in C.items():
        s1,e1=T(s0),T(e0); a.plot([s1[i],e1[i]],[s1[j],e1[j]],'b--',lw=1); a.text(*(0.7*s1+0.3*e1)[[i,j]],k,color='b',fontsize=8)
    a.set_aspect('equal'); a.set_title(f'PC{i+1}-PC{j+1}')
plt.tight_layout(); plt.savefig(OUT+'triage_views_labeled.png',dpi=70)
# print gold & preds in PCA frame
for nm,D in [('gold',G),('best',B),('ph',C)]:
    for k,(s0,e0) in D.items():
        s1,e1=T(s0),T(e0); dd=e1-s1; print(nm,k,np.round(s1,1),np.round(e1,1),'len',round(np.linalg.norm(dd),1),'dir',np.round(dd/np.linalg.norm(dd),3))
