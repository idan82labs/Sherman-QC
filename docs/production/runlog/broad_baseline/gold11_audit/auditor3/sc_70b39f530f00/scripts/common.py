"""Auditor 3 helpers for sc_70b39f530f00 (49024000_A.ply). Read-only on all inputs."""
import json, math
import numpy as np
import trimesh

SCAN = '/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply'
GOLD = '/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json'
OUT = '/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_70b39f530f00'


def load_mesh():
    return trimesh.load(SCAN, process=False)


def load_gold():
    d = json.load(open(GOLD))
    for c in d['cases']:
        if c['case_id'] == 'corpus/11_49024000':
            return {b['bend_id']: (np.array(b['line_start'], float), np.array(b['line_end'], float)) for b in c['bends']}
    raise KeyError


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 0 else v


def basis_for(d, up_hint=None):
    d = unit(d)
    a = np.array([0, 0, 1.0]) if up_hint is None else unit(up_hint)
    a = a - d * (a @ d)
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0, 0]); a = a - d * (a @ d)
    w = unit(a)
    u = np.cross(w, d)
    return u, w


def section2d(mesh, origin, normal, u, w):
    """Exact plane cut. Returns list of (N,2) polylines in (u,w) coords about origin and the 3D segs."""
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return [], None
    segs = []
    for ent in sec.entities:
        P = sec.vertices[ent.points]
        q = np.c_[(P - origin) @ u, (P - origin) @ w]
        segs.append(q)
    return segs, sec
