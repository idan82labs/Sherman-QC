"""Read-only triage helpers for sc_a0305a6cf787 (47959001_01PREL_scan1.stl). Outputs only under docs/production/runlog/broad_baseline/triage/sc_a0305a6cf787/."""
import json, os, numpy as np
R='/Users/idantal/ShermanQC/FullProject/'
SCAN='/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/47959001_01PREL_scan1.stl'
OUT=R+'docs/production/runlog/broad_baseline/triage/sc_a0305a6cf787/'
NPZ=OUT+'points_ds.npz'
RUN=R+'docs/production/runlog/broad_baseline/runs/best_f1/sc_a0305a6cf787/'

def load_points():
    if os.path.exists(NPZ):
        d=np.load(NPZ); return d['P'],d['N']
    import trimesh
    m=trimesh.load(SCAN,force='mesh')
    P,fi=trimesh.sample.sample_surface_even(m,400000,seed=0) if hasattr(trimesh.sample,'sample_surface_even') else trimesh.sample.sample_surface(m,400000,seed=0)
    N=m.face_normals[fi]
    np.savez_compressed(NPZ,P=np.asarray(P,np.float32),N=np.asarray(N,np.float32))
    return np.asarray(P),np.asarray(N)

def gold():
    d=json.load(open(R+'output/bend_label_09_47959001/bend_instance_labels.json'))
    return {b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in d['manual_bends']}

def preds(name):
    d=json.load(open(RUN+name+'.json'))
    return {b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in d['owned_bend_regions']}
