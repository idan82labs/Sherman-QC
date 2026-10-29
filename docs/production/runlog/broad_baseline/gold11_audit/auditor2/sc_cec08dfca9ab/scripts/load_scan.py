import numpy as np, open3d as o3d, json, time
t=time.time()
p="/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply"
m=o3d.io.read_triangle_mesh(p)
V=np.asarray(m.vertices); F=np.asarray(m.triangles); N=np.asarray(m.vertex_normals) if m.has_vertex_normals() else None
print(V.shape, F.shape, None if N is None else N.shape, time.time()-t)
np.savez_compressed("scan.npz", V=V.astype(np.float64), F=F.astype(np.int32), N=(N if N is not None else np.zeros((0,3))).astype(np.float32))
print("bbox", V.min(0), V.max(0))
# display cloud
d=o3d.io.read_point_cloud("/Users/idantal/ShermanQC/FullProject/output/bend_label_03_10839000/aligned_scan.ply")
D=np.asarray(d.points); print("display", D.shape, D.min(0), D.max(0))
np.save("display.npy", D)
