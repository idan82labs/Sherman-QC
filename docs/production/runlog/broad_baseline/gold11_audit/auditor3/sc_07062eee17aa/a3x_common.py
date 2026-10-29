"""Auditor-3 helpers for sc_07062eee17aa (read-only analysis; writes only renders/json in this folder)."""
import sys
sys.dont_write_bytecode = True
import json, math, os
from pathlib import Path
import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_07062eee17aa"
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply")
GOLD = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/a962166c3019434c10628722.json"
CACHE = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_07062/pts.npy")


def load_points():
    if CACHE.is_file():
        return np.load(CACHE)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(SCAN))
    p = np.asarray(pc.points, dtype=np.float64)
    if not len(p):
        m = o3d.io.read_triangle_mesh(str(SCAN))
        p = np.asarray(m.vertices, dtype=np.float64)
    p = p[np.isfinite(p).all(1)]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, p)
    return p


def load_gold():
    g = json.load(open(GOLD))
    out = []
    for b in g["bends"]:
        out.append((b["bend_id"], np.array(b["line_start"], float), np.array(b["line_end"], float)))
    return out


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perp_basis(d, ref=None):
    """u, w orthonormal, perpendicular to d. If ref given, u is ref projected off d."""
    d = unit(d)
    if ref is None:
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = unit(np.cross(d, a))
    else:
        u = unit(ref - d * (ref @ d))
    w = np.cross(d, u)
    return u, w
