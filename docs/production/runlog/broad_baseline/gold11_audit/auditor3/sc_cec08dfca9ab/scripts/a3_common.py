"""Auditor 3 helpers for sc_cec08dfca9ab (10839000 scan 2). Read-only on all inputs.

Loads the raw scan once (cached as .npz in the session scratchpad), the gold lines from the
authoritative catalog (frame = raw scan, identity), and provides a part frame.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply")
SCAN_SHA = "cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_cec08dfca9ab"
CACHE = Path(os.environ.get("A3_CACHE", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_cec08_scan.npz"))


def load_points() -> np.ndarray:
    if CACHE.is_file():
        return np.load(CACHE)["pts"]
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(SCAN))
    pts = np.asarray(pc.points, dtype=np.float64)
    if not len(pts):
        m = o3d.io.read_triangle_mesh(str(SCAN))
        pts = np.asarray(m.vertices, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, pts=pts)
    return pts


def load_gold() -> list[tuple[str, np.ndarray, np.ndarray]]:
    cat = json.loads(CATALOG.read_text())
    case = next(c for c in cat["cases"] if c["case_id"] == "corpus/03_10839000")
    assert case["scan_sha256"] == SCAN_SHA
    return [(b["bend_id"], np.asarray(b["line_start"], float), np.asarray(b["line_end"], float)) for b in case["bends"]]


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, a))
    w = np.cross(d, u)
    return u, w
