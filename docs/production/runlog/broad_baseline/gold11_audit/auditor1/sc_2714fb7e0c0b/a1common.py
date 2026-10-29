"""Auditor-1 helpers for sc_2714fb7e0c0b (38172000_01_V.ply). Read-only on all inputs.

Loads the raw scan (binary PLY, Geomagic export: float xyz + float normals) without
Open3D, the frozen gold (catalog shaked_xv/C06, scan_raw == file coordinates), and
defines a part-aligned frame:
  ex = along the step bends (B03 direction)
  ez = along the main-panel long edges (B01 direction, orthogonalised)
  en = ex x ez (panel normal)
Nothing outside gold11_audit/auditor1/sc_2714fb7e0c0b is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_2714fb7e0c0b"
SHA = "2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba"


def load_scan(with_faces=False):
    raw = SCAN.read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    while raw[end:end + 1] in (b"\r", b"\n"):
        end += 1
    header = raw[:end].decode("ascii", "replace")
    nv = int(header.split("element vertex ")[1].split()[0])
    nf = int(header.split("element face ")[1].split()[0])
    assert "property float nx" in header and "binary_little_endian" in header
    v = np.frombuffer(raw, dtype="<f4", count=nv * 6, offset=end).reshape(nv, 6).astype(np.float64)
    pts, nrm = v[:, :3], v[:, 3:]
    faces = None
    if with_faces:
        off = end + nv * 24
        rec = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        f = np.frombuffer(raw, dtype=rec, count=nf, offset=off)
        assert (f["n"] == 3).all()
        faces = f["i"].astype(np.int64)
    return pts, nrm, faces


def load_gold():
    cat = json.loads(CATALOG.read_text())
    row = [c for c in cat["cases"] if c["scan_sha256"] == SHA]
    assert len(row) == 1 and row[0]["case_id"] == "shaked_xv/C06"
    return {b["bend_id"]: (np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in row[0]["bends"]}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def part_frame(gold):
    ex = unit(gold["B03"][1] - gold["B03"][0])
    ez0 = unit(gold["B01"][1] - gold["B01"][0])
    ez = unit(ez0 - (ez0 @ ex) * ex)
    en = np.cross(ex, ez)
    return ex, ez, en


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, a))
    w = np.cross(d, u)
    return u, w
