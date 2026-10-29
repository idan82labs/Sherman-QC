"""Auditor-2 helpers for sc_f2475286312b (38172000_01_SCAN_2 1.ply, Gold11 CGEBA9A869AC).
Independent of the orchestrator's render_gold_xsections.py.  Read-only on all inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_2 1.ply")
GOLD = Path("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/"
            "human_bend_lines_v1/a6a0b1c3f8052cb3a45412e8.json")
LABEL_JOB = Path("/Users/idantal/ShermanQC/FullProject/output/bend_label_07_38172000/"
                 "bend_instance_labels.json")
OUT = Path(__file__).resolve().parent


def load_ply(path: Path = SCAN):
    with open(path, "rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
        txt = header.decode("ascii", "replace")
        nv = int([l for l in txt.splitlines() if l.startswith("element vertex")][0].split()[-1])
        props = []
        for l in txt.splitlines():
            if l.startswith("element face"):
                break
            if l.startswith("property float"):
                props.append(l.split()[-1])
        vdt = np.dtype([(p, "<f4") for p in props])
        v = np.frombuffer(fh.read(nv * vdt.itemsize), dtype=vdt)
        pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64) if "nx" in props else None
    return pts, nrm


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


def pca_frame(pts):
    c0 = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
    return c0, vt
