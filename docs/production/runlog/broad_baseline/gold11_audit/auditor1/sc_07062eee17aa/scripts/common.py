"""Auditor-1 helpers for sc_07062eee17aa (read-only; loads the Gold11 raw scan + gold lines)."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = "/Users/idantal/ShermanQC/FullProject"
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply"
GOLD = ROOT + "/data/bend_corpus/annotations/human_bend_lines_v1/a962166c3019434c10628722.json"
OUT = ROOT + "/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_07062eee17aa"


def load_mesh():
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(SCAN)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    N = np.asarray(m.vertex_normals, dtype=np.float64) if m.has_vertex_normals() else None
    return V, F, N


def load_gold():
    d = json.load(open(GOLD))
    out = {}
    for b in d["bends"]:
        out[b["bend_id"]] = (np.array(b["line_start"], float), np.array(b["line_end"], float))
    return out


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def pca_frame(V):
    c = V.mean(0)
    u, s, vt = np.linalg.svd(V - c, full_matrices=False)
    R = vt  # rows = PC1, PC2, PC3
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return c, R
