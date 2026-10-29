"""Auditor-2 helpers for sc_07062eee17aa (read-only; writes only into this directory)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply")
GOLD = Path("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/a962166c3019434c10628722.json")
OUT = Path(__file__).resolve().parent
CACHE = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/sc_07062eee17aa_scan_cache.npz")


def load_scan():
    if CACHE.is_file():
        z = np.load(CACHE)
        return z["v"], z["n"], z["f"]
    with open(SCAN, "rb") as fh:
        head = fh.read(4096)
        k = head.index(b"end_header")
        k = head.index(b"\n", k) + 1  # header lines may end in CRLF (Geomagic export)
        header = head[:k]
        fh.seek(k)
        lines = header.decode("ascii", "replace").splitlines()
        nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        nf = int([l for l in lines if l.startswith("element face")][0].split()[-1])
        vert = np.frombuffer(fh.read(nv * 24), dtype="<f4").reshape(nv, 6).astype(np.float64)
        fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        faces = np.frombuffer(fh.read(nf * 13), dtype=fdt)
        assert (faces["n"] == 3).all()
        f = faces["i"].astype(np.int64)
    v, n = vert[:, :3].copy(), vert[:, 3:].copy()
    np.savez_compressed(CACHE, v=v, n=n, f=f)
    return v, n, f


def load_gold():
    g = json.loads(GOLD.read_text())
    out = {}
    for b in g["bends"]:
        out[b["bend_id"]] = (np.asarray(b["line_start"], float), np.asarray(b["line_end"], float))
    return out


def unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


def frame_from_gold(gold):
    """Common axis d (mean gold direction) + perpendicular basis (u, w)."""
    ds = []
    for s, e in gold.values():
        d = unit(e - s)
        if ds and np.dot(d, ds[0]) < 0:
            d = -d
        ds.append(d)
    d = unit(np.mean(ds, axis=0))
    return d
