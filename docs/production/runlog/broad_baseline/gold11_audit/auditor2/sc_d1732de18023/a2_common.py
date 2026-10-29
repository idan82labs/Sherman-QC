"""Auditor-2 helpers for sc_d1732de18023 (37361005_01.ply).  Independent of the
orchestrator's render_gold_xsections.py.  Read-only on all inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply")
GOLD = Path("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/"
            "human_bend_lines_v1/05df0a708054946a1d77f7ee.json")
OUT = Path(__file__).resolve().parent


def load_ply(path: Path = SCAN, faces: bool = False):
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
        props = [l.split()[-1] for l in txt.splitlines() if l.startswith("property float")]
        vdt = np.dtype([(p, "<f4") for p in props])
        v = np.frombuffer(fh.read(nv * vdt.itemsize), dtype=vdt)
        pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64) if "nx" in props else None
        tri = None
        if faces:
            fdt = np.dtype([("n", "u1"), ("v", "<i4", (3,))])
            f = np.frombuffer(fh.read(nf * fdt.itemsize), dtype=fdt)
            assert np.all(f["n"] == 3)
            tri = f["v"].astype(np.int64)
    return pts, nrm, tri


def load_gold():
    d = json.loads(GOLD.read_text())
    return {b["bend_id"]: (np.array(b["line_start"], float), np.array(b["line_end"], float))
            for b in d["bends"]}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_plane(p):
    c = p.mean(0)
    _, s, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[2], s


def robust_plane(p, iters=4, keep=2.5):
    c, n, _ = fit_plane(p)
    for _ in range(iters):
        r = (p - c) @ n
        mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
        m = np.abs(r - np.median(r)) < keep * mad + 0.05
        c, n, _ = fit_plane(p[m])
    r = (p - c) @ n
    return c, n, float(np.sqrt(np.mean(r[np.abs(r) < 1.0] ** 2))) if np.any(np.abs(r) < 1.0) else float("nan")
