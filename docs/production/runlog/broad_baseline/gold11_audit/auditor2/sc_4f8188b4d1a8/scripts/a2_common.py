"""Auditor-2 helpers for sc_4f8188b4d1a8 (read-only on gold + scan)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_X.ply"
SCAN_SHA = "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_4f8188b4d1a8"
CACHE = Path(os.environ.get("A2_CACHE", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/sc_4f8188b4d1a8_cache.npz"))


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
        v = np.frombuffer(fh.read(nv * 24), dtype="<f4").reshape(nv, 6).astype(np.float64)
        fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        f = np.frombuffer(fh.read(nf * fdt.itemsize), dtype=fdt)
        assert np.all(f["n"] == 3)
        faces = f["i"].astype(np.int64)
    return v[:, :3].copy(), v[:, 3:].copy(), faces


def get_scan():
    if CACHE.is_file():
        z = np.load(CACHE)
        return z["xyz"], z["nrm"], z["faces"]
    xyz, nrm, faces = load_ply()
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(n, 1e-12)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, xyz=xyz, nrm=nrm, faces=faces)
    return xyz, nrm, faces


def gold_lines():
    cat = json.loads(CATALOG.read_text())
    rows = [r for r in cat["cases"] if r["case_id"] == "shaked_xv/C05"]
    assert len(rows) == 1 and rows[0]["scan_sha256"] == SCAN_SHA
    out = {}
    for b in rows[0]["bends"]:
        out[b["bend_id"]] = (np.asarray(b["line_start"], float), np.asarray(b["line_end"], float))
    return out


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def perp_basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(d, a))
    e2 = np.cross(d, e1)
    return e1, e2
