import numpy as np, open3d as o3d
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
pc=pc.voxel_down_sample(0.7)
X=np.asarray(pc.points)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=3.0))
N=np.asarray(pc.normals)
# planarity via covariance eigen ratios
tree=o3d.geometry.KDTreeFlann(pc)
k=len(X); planar=np.zeros(k,bool); curv=np.zeros(k)
for i in range(k):
    _,idx,_=tree.search_radius_vector_3d(X[i],3.0)
    if len(idx)<12: continue
    Y=X[idx]-X[idx].mean(0); ev=np.linalg.eigvalsh(Y.T@Y/len(idx))
    curv[i]=ev[0]/ev.sum()
planar = curv<0.004
ax=np.abs(N)  # alignment to e1,e2,e3
maxal=ax.max(1)
obl = planar & (maxal<np.cos(np.radians(10)))
print("downsampled", k, "planar", planar.sum(), "planar-oblique(>10deg from all axes)", obl.sum())
if obl.sum():
    Z=X[obl]
    # cluster oblique planar points
    from sklearn.cluster import DBSCAN
    lab=DBSCAN(eps=2.0,min_samples=8).fit_predict(Z)
    for l in sorted(set(lab)):
        if l<0: continue
        m=lab==l
        print("cluster",l,"n",m.sum(),"e1 %.1f..%.1f e2 %.1f..%.1f e3 %.1f..%.1f"%(Z[m,0].min(),Z[m,0].max(),Z[m,1].min(),Z[m,1].max(),Z[m,2].min(),Z[m,2].max()), "mean normal", np.round(N[obl][m].mean(0),2))
    print("noise", (lab<0).sum())
