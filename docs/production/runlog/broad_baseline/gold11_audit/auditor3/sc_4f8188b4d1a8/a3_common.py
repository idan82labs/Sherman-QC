"""Auditor-3 helpers for the Gold11 sc_4f8188b4d1a8 (38172000_01_X) locus audit.

Read-only: loads the Gold11 development scan (SHA-bound) and the gold lines, caches
vertex arrays in the session scratchpad. Writes nothing into data/ or annotations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
SCAN = ROOT / "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_X.ply"
SCAN_SHA = "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
CASE_ID = "shaked_xv/C05"
OUT = ROOT / "docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_4f8188b4d1a8"
CACHE = Path(os.environ.get("A3_CACHE", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3"))


def _read_ply(path: Path):
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline()
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        nv = nf = 0
        props = []
        for h in header:
            if h.startswith("element vertex"):
                nv = int(h.split()[-1])
            elif h.startswith("element face"):
                nf = int(h.split()[-1])
            elif h.startswith("property float") and nf == 0:
                props.append(h.split()[-1])
        assert props == ["x", "y", "z", "nx", "ny", "nz"], props
        v = np.frombuffer(fh.read(nv * 24), dtype="<f4").reshape(nv, 6).astype(np.float64)
        raw = np.frombuffer(fh.read(nf * 13), dtype=np.uint8)
        assert raw.size == nf * 13
        counts = raw[0::13]
        assert np.all(counts == 3)
        idx = np.frombuffer(raw.reshape(nf, 13)[:, 1:].copy().tobytes(), dtype="<i4").reshape(nf, 3)
    return v[:, :3], v[:, 3:], idx


def load_scan():
    CACHE.mkdir(parents=True, exist_ok=True)
    cp, cn, cf = CACHE / "pts.npy", CACHE / "nrm.npy", CACHE / "faces.npy"
    if cp.is_file() and cn.is_file() and cf.is_file():
        return np.load(cp), np.load(cn), np.load(cf)
    sha = hashlib.sha256(SCAN.read_bytes()).hexdigest()
    assert sha == SCAN_SHA, sha
    p, n, f = _read_ply(SCAN)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nn > 0, nn, 1)
    np.save(cp, p)
    np.save(cn, n)
    np.save(cf, f)
    return p, n, f


def load_gold():
    cat = json.loads(CATALOG.read_text())
    rows = [r for r in cat["cases"] if r["case_id"] == CASE_ID]
    assert len(rows) == 1
    row = rows[0]
    assert row["scan_sha256"] == SCAN_SHA
    out = []
    for b in row["bends"]:
        s, e = np.asarray(b["line_start"], float), np.asarray(b["line_end"], float)
        out.append({"id": b["bend_id"], "s": s, "e": e})
    return out, row


def basis(d):
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return d, e1, e2
