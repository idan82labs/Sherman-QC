"""Auditor-3 helpers for sc_f2475286312b (read-only on the scan and gold).

Loads the raw Geomagic PLY (vertices, normals, faces) exactly as stored (identity frame,
the gold is in reference_mesh_raw == scan raw, residual 0), caches it as npz in the
session scratchpad, and offers mesh/plane section slicing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_2 1.ply")
SCAN_SHA = "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63"
GOLD_CATALOG = Path("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json")
CACHE = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_sc_f2475286312b_mesh.npz")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_f2475286312b")


def load_mesh():
    if CACHE.is_file():
        z = np.load(CACHE)
        return z["V"], z["N"], z["F"]
    with open(SCAN, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
        txt = header.decode("ascii", "replace")
        nv = int([l for l in txt.splitlines() if l.startswith("element vertex")][0].split()[-1])
        nf = int([l for l in txt.splitlines() if l.startswith("element face")][0].split()[-1])
        assert "property float nx" in txt and "binary_little_endian" in txt
        vb = np.frombuffer(fh.read(nv * 24), dtype="<f4").reshape(nv, 6)
        fb = np.frombuffer(fh.read(nf * 13), dtype=np.uint8).reshape(nf, 13)
    assert (fb[:, 0] == 3).all()
    F = fb[:, 1:].copy().view("<i4").reshape(nf, 3)
    V = vb[:, :3].astype(np.float64)
    N = vb[:, 3:].astype(np.float64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, V=V, N=N, F=F)
    return V, N, F


def load_gold():
    d = json.loads(GOLD_CATALOG.read_text())
    c = [x for x in d["cases"] if x["case_id"] == "corpus/07_38172000"][0]
    assert c["scan_sha256"] == SCAN_SHA
    return {b["bend_id"]: (np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in c["bends"]}


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def slice_mesh(V, F, origin, normal, tri_centroid=None, radius=None):
    """Intersect the mesh with plane (origin, normal). Returns array of segments (k,2,3).
    Optional pre-filter: only triangles whose centroid lies within `radius` of origin."""
    normal = unit(normal)
    idx = np.arange(len(F))
    if radius is not None and tri_centroid is not None:
        idx = idx[np.linalg.norm(tri_centroid - origin, axis=1) < radius]
    tri = F[idx]
    d = (V[tri] - origin) @ normal  # (k,3)
    s = np.sign(d)
    s[s == 0] = 1e-9
    cross = ~((d > 0).all(1) | (d < 0).all(1))
    tri = tri[cross]
    d = d[cross]
    P = V[tri]
    segs = []
    for (a, b) in ((0, 1), (1, 2), (2, 0)):
        m = (d[:, a] > 0) != (d[:, b] > 0)
        t = np.where(m, d[:, a] / np.where(m, d[:, a] - d[:, b], 1.0), np.nan)
        pts = P[:, a] + t[:, None] * (P[:, b] - P[:, a])
        segs.append((m, pts))
    out = []
    m0, p0 = segs[0]
    m1, p1 = segs[1]
    m2, p2 = segs[2]
    for i in range(len(tri)):
        pp = [p for m, p in ((m0[i], p0[i]), (m1[i], p1[i]), (m2[i], p2[i])) if m]
        if len(pp) >= 2:
            out.append(pp[:2])
    return np.array(out) if out else np.zeros((0, 2, 3))


def perp_basis(d, up=None):
    d = unit(d)
    if up is None:
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = unit(np.cross(d, a))
    else:
        u = unit(up - (up @ d) * d)
    w = np.cross(d, u)
    return u, w
