#!/usr/bin/env python3
"""Auditor 3: list the cylindrical faces of the 10839000_01 CAD (STEP AP214) with axis, radius and
extent along the axis, to know which bends the finished part has (read-only, text parse)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import re
from pathlib import Path

import numpy as np

STEP = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp")
OUT = Path(__file__).resolve().parent

txt = STEP.read_text(errors="replace")
data = txt.split("DATA;", 1)[1]
ents = {}
for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|ENDSEC)", data, re.S):
    ents[int(m.group(1))] = " ".join(m.group(2).split())


def refs(s):
    return [int(x) for x in re.findall(r"#(\d+)", s)]


def nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.(?:[eE][-+]?\d+)?", s)]


def point(i):
    s = ents[i]
    assert s.startswith("CARTESIAN_POINT"), s
    return np.array(nums(s.split(",", 1)[1])[:3])


def direction(i):
    s = ents[i]
    assert s.startswith("DIRECTION"), s
    return np.array(nums(s.split(",", 1)[1])[:3])


def placement(i):
    s = ents[i]
    r = refs(s)
    return point(r[0]), direction(r[1]), (direction(r[2]) if len(r) > 2 else None)


def face_vertices(face_id):
    s = ents[face_id]
    pts = []
    stack = refs(s)
    seen = set()
    while stack:
        k = stack.pop()
        if k in seen or k not in ents:
            continue
        seen.add(k)
        e = ents[k]
        if e.startswith("CARTESIAN_POINT"):
            pts.append(point(k))
            continue
        if e.startswith(("FACE_BOUND", "FACE_OUTER_BOUND", "EDGE_LOOP", "ORIENTED_EDGE", "EDGE_CURVE", "VERTEX_POINT")):
            rr = refs(e)
            if e.startswith("EDGE_CURVE"):
                rr = rr[:2]  # only the two vertices, not the curve geometry
            stack.extend(rr)
    return np.array(pts)


rows = []
for fid, s in ents.items():
    if not s.startswith("ADVANCED_FACE"):
        continue
    r = refs(s)
    surf = r[-1]
    se = ents.get(surf, "")
    if not se.startswith("CYLINDRICAL_SURFACE"):
        continue
    pl = refs(se)[0]
    loc, ax, _ = placement(pl)
    rad = nums(se.split(",")[-1])[-1]
    V = face_vertices(fid)
    t = (V - loc) @ ax if len(V) else np.zeros(0)
    radial = V - loc - np.outer(t, ax) if len(V) else np.zeros((0, 3))
    rows.append({"face": fid, "radius": round(rad, 3), "axis": np.round(ax, 4).tolist(), "loc": np.round(loc, 3).tolist(),
                 "t_min": float(t.min()) if len(t) else None, "t_max": float(t.max()) if len(t) else None,
                 "len": float(t.max() - t.min()) if len(t) else None, "n_vert": int(len(V)),
                 "centre_on_axis": np.round(loc + ax * (0.5 * (t.min() + t.max())), 2).tolist() if len(t) else None})

rows.sort(key=lambda r: (-(r["len"] or 0), r["radius"]))
for r in rows:
    print(r)
(OUT / "cad_cylinders.json").write_text(json.dumps(rows, indent=1))

# overall CAD vertex bbox
allp = np.array([point(i) for i, s in ents.items() if s.startswith("CARTESIAN_POINT") and len(nums(s.split(',', 1)[1])) >= 3])
print("CAD point bbox min", allp.min(0).round(2), "max", allp.max(0).round(2))
