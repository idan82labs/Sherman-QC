"""Auditor-3 helpers for sc_152f1b1aacb2 (40500000 SCAN2) — read-only on scan + gold."""
import json, numpy as np
from pathlib import Path
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply"
GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/8370b5e44945adfba2f2fb34.json"
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_152f1b1aacb2")
SCR = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")

def load_mesh():
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(SCAN)
    V = np.asarray(m.vertices, float); F = np.asarray(m.triangles, int)
    return V, F

def load_gold():
    g = json.load(open(GOLD))
    return [(b["bend_id"], np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in g["bends"]]

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 0 else v

def fit_plane(P):
    c = P.mean(0); _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2], s

def mesh_plane_section(V, F, p0, n):
    """Intersect triangle mesh with plane (p0, n). Returns array of segments (k,2,3)."""
    d = (V - p0) @ n
    dF = d[F]
    s = np.sign(dF)
    cross = (s.max(1) > 0) & (s.min(1) < 0)
    segs = []
    for tri, dd in zip(F[cross], dF[cross]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = dd[a], dd[b]
            if (da > 0) != (db > 0) and da != db:
                t = da / (da - db)
                pts.append(V[tri[a]] + t * (V[tri[b]] - V[tri[a]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs).reshape(-1, 2, 3)

# ---- part frame (from RANSAC planes on the raw vertices; see a3c_frame.py)
def part_frame(V):
    nb = unit(np.array([0.4532, -0.5437, 0.7064]))
    nw = unit(np.array([0.1382, 0.8273, 0.5444]))
    e3 = -nb
    e2 = unit(nw - (nw @ e3) * e3)
    e1 = np.cross(e2, e3)
    R = np.stack([e1, e2, e3])  # rows
    # origin: base outer plane centroid-ish
    X = (V - 0) @ R.T
    base = np.abs(X[:, 2] - np.median(X[:, 2])) < 3
    o = np.array([np.median(X[base, 0]), np.median(X[base, 1]), np.median(X[base, 2])])
    return R, o

def to_part(P, R, o):
    return np.asarray(P) @ R.T - o
