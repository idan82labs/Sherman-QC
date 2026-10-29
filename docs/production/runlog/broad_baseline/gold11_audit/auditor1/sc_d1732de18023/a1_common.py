"""Auditor-1 helpers for sc_d1732de18023 (read-only on all inputs)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply")
GOLD = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_d1732de18023"
CACHE = Path(os.environ.get("A1_CACHE", "/private/tmp/claude-501/a1_sc_d1732de18023_cache.npz"))


def load_ply(path=SCAN):
    with open(path, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
        txt = header.decode("ascii", "replace")
        nv = int([l for l in txt.splitlines() if l.startswith("element vertex")][0].split()[-1])
        nf = int([l for l in txt.splitlines() if l.startswith("element face")][0].split()[-1])
        vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
        v = np.frombuffer(fh.read(nv * vdt.itemsize), dtype=vdt)
        fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        rest = fh.read()
        f = np.frombuffer(rest[: nf * fdt.itemsize], dtype=fdt)
        assert np.all(f["n"] == 3)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    nrm = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64)
    return xyz, nrm, f["i"].astype(np.int64)


def load_all():
    if CACHE.is_file():
        z = np.load(CACHE)
        return z["xyz"], z["nrm"], z["faces"]
    xyz, nrm, faces = load_ply()
    np.savez(CACHE, xyz=xyz, nrm=nrm, faces=faces)
    return xyz, nrm, faces


def gold_lines():
    cat = json.loads(GOLD.read_text())
    row = [c for c in cat["cases"] if c["case_id"] == "corpus/05_37361005"][0]
    out = {}
    for b in row["bends"]:
        out[b["bend_id"]] = (np.array(b["line_start"], float), np.array(b["line_end"], float))
    return out


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(d, a))
    w = np.cross(d, u)
    return u, w


def pca_frame(xyz):
    c = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz[::20] - c, full_matrices=False)
    return c, vt  # rows: PC1, PC2, PC3
