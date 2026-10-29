import numpy as np, open3d as o3d, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('research/broad_baseline/triage_sc_152f1b1aacb2/render_triage.py').read().split('fig,axs')[0])
P=np.asarray(o3d.io.read_point_cloud('/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply').points)
p=L(P)
fig,axs=plt.subplots(1,4,figsize=(20,6))
for ax,k in zip(axs,['M03','M04','M05','M06']):
    A=L(G[k][0])[0]
    m=(np.abs(p[:,0]-A[0])<12)&(np.abs(p[:,1]-A[1])<12)&(p[:,2]>5)&(p[:,2]<30)
    ax.scatter(p[m,0],p[m,1],s=0.3,c='k'); ax.plot(A[0],A[1],'gx',ms=15,mew=3)
    ax.set_aspect('equal'); ax.set_title(f'{k}: full-res top view, mid-height e3 5..30 (n={m.sum()})')
    # local normals stats
    print(k,'pts',m.sum(),'e2 spread p1/p99',np.percentile(p[m,1],[1,99]).round(2),'e1 range',np.percentile(p[m,0],[0.5,99.5]).round(2), 'gold e1',A[0].round(1))
plt.tight_layout(); plt.savefig(OUT+'vertical_gold_fullres_top.png',dpi=80)
