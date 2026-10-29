"""Auditor-2 helpers for sc_70b39f530f00 (49024000_A.ply). Read-only on all inputs.

Loads the raw scan mesh (identity frame to the gold, residual 0 per the frame record) and the
three human gold lines straight from the label job (byte-identical to the catalog mirror).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply")
LABELS = Path("/Users/idantal/ShermanQC/FullProject/output/bend_label_11_49024000/bend_instance_labels.json")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_70b39f530f00")


def load_mesh():
    m = trimesh.load(str(SCAN), process=False)
    return m


def load_gold():
    d = json.loads(LABELS.read_text())
    out = {}
    for b in d["manual_bends"]:
        out[b["bend_id"]] = (np.asarray(b["line_start"], float), np.asarray(b["line_end"], float))
    return out


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, a))
    w = np.cross(d, u)
    return u, w


def section(mesh, origin, normal):
    """Exact plane section of the mesh -> list of (N,3) polylines (closed loops or open paths)."""
    path = mesh.section(plane_origin=origin, plane_normal=normal)
    if path is None:
        return []
    out = []
    for ent in path.entities:
        pts = path.vertices[ent.points]
        out.append(np.asarray(pts, float))
    return out
