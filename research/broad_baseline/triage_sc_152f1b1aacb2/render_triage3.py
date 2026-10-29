import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('research/broad_baseline/triage_sc_152f1b1aacb2/render_triage.py').read().split('fig,axs')[0])
fig,axs=plt.subplots(2,4,figsize=(22,10))
specs=[('M04 wall2 e1 -70..-48',(-70,-48),1),('M03 wall2 e1 48..66',(48,66),1),('M05 wall1 e1 -70..-48',(-70,-48),-1),('M06 wall1 e1 48..66',(48,66),-1)]
for col,(t,(lo,hi),s) in enumerate(specs):
    m=(q[:,0]>lo)&(q[:,0]<hi)&(q[:,1]*s>60)
    sc=axs[0,col].scatter(q[m,1],q[m,2],s=0.5,c=q[m,0],cmap='viridis'); plt.colorbar(sc,ax=axs[0,col],label='e1')
    axs[0,col].set_aspect('equal'); axs[0,col].set_title(t+' : e2-e3 section, colour=e1')
    sc=axs[1,col].scatter(q[m,0],q[m,1],s=0.5,c=q[m,2],cmap='plasma'); plt.colorbar(sc,ax=axs[1,col],label='e3')
    axs[1,col].set_aspect('equal'); axs[1,col].set_title(t+' : top e1-e2, colour=e3')
    for k in ['M03','M04','M05','M06']:
        A,B=L(G[k][0])[0],L(G[k][1])[0]
        axs[1,col].plot([A[0]],[A[1]],'gx',ms=12,mew=3); axs[0,col].plot([A[1],B[1]],[A[2],B[2]],'g-',lw=1)
    axs[1,col].set_xlim(lo,hi)
plt.tight_layout(); plt.savefig(OUT+'corner_sections.png',dpi=75)
for lo,hi,s in [(-70,-48,1),(48,66,1),(-70,-48,-1),(48,66,-1)]:
    m=(q[:,0]>lo)&(q[:,0]<hi)&(q[:,1]*s>60)&(q[:,1]*s<89)&(q[:,2]>-3)
    print(lo,hi,s,'inner-side pts above base:',m.sum(), 'e3 range',np.percentile(q[m,2],[1,50,99]).round(1) if m.sum() else None, 'e1 range', np.percentile(q[m,0],[1,99]).round(1) if m.sum() else None)
