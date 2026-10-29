import numpy as np, trimesh
PLY="/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply"
OUT="/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_70b39f530f00/"
def load():
    return trimesh.load(PLY, process=False)
GOLD={
 "M01":(np.array([-214.487,-126.078,283.985]),np.array([-234.12,-143.164,272.978])),
 "M02":(np.array([-212.201,-122.472,274.308]),np.array([-227.344,-135.651,265.818])),
 "M03":(np.array([-224.555,-118.025,256.23]),np.array([-211.013,-106.239,263.823])),
}
def unit(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)
