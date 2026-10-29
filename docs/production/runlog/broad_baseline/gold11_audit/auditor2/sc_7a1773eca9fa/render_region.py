#!/usr/bin/env python3
"""Auditor 2 - shaded close-up renders of a scan region (mesh triangles, Lambert shading) with gold lines.

usage: render_region.py NPZ CX,CY,CZ RADIUS OUT.png [view dirs as 'x,y,z;x,y,z;...'] [title]
Triangles whose three vertices lie within RADIUS of the centre are drawn with a painter's sort.
Gold lines (green) and optional extra lines from extra_lines.json (red) are overlaid.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xsec_audit import GOLD  # noqa: E402

npz = sys.argv[1]
C = np.array([float(v) for v in sys.argv[2].split(",")])
R = float(sys.argv[3])
outp = sys.argv[4]
views = [np.array([float(v) for v in s.split(",")]) for s in (sys.argv[5].split(";") if len(sys.argv) > 5 else
                                                               ["1,0,0", "0,1,0", "0,0,1", "1,1,1"])]
title = sys.argv[6] if len(sys.argv) > 6 else ""
extra = json.loads(Path(sys.argv[7]).read_text()) if len(sys.argv) > 7 else {}
d = np.load(npz)
P, N, F = d["P"], d["N"], d["F"]
inside = np.linalg.norm(P - C, axis=1) <= R
fm = inside[F].all(axis=1)
T = F[fm]
V = P[T]                      # (k,3,3)
fn = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
n = len(views)
cols = min(n, 2)
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows), squeeze=False)
for k, vd in enumerate(views):
    ax = axs[k // cols][k % cols]
    v = vd / np.linalg.norm(vd)          # direction from the object toward the camera
    up0 = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([0, 1.0, 0])
    ex = np.cross(up0, v)
    ex /= np.linalg.norm(ex)
    ey = np.cross(v, ex)
    depth = (V.mean(1) - C) @ v
    order = np.argsort(depth)            # far first
    light = np.clip(np.abs(fn @ v) * 0.85 + 0.15, 0, 1)
    facing = fn @ v
    cols_ = np.stack([light * np.where(facing >= 0, 0.75, 0.95), light * np.where(facing >= 0, 0.75, 0.65),
                      light * np.where(facing >= 0, 0.8, 0.45)], 1)
    polys = np.stack([(V - C) @ ex, (V - C) @ ey], -1)
    pc = PolyCollection(polys[order], facecolors=cols_[order], edgecolors="none")
    ax.add_collection(pc)
    for bid, (a, b) in GOLD.items():
        a, b = np.array(a), np.array(b)
        seg = np.stack([a, b])
        if np.min(np.linalg.norm(seg - C, axis=1)) > R * 1.5 and np.linalg.norm((a + b) / 2 - C) > R * 1.2:
            continue
        xy = np.stack([(seg - C) @ ex, (seg - C) @ ey], 1)
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#1a9c3a", lw=2.2, zorder=10)
        ax.text(*xy.mean(0), bid, color="#1a9c3a", fontsize=11, fontweight="bold", zorder=11)
    for name, (a, b) in extra.items():
        seg = np.stack([np.array(a), np.array(b)])
        xy = np.stack([(seg - C) @ ex, (seg - C) @ ey], 1)
        ax.plot(xy[:, 0], xy[:, 1], "--", color="#d62728", lw=1.6, zorder=12)
        ax.text(*xy[1], name, color="#d62728", fontsize=9, zorder=13)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.set_title(f"view from {np.round(v, 2).tolist()} (grey = faces toward camera, tan = back faces)", fontsize=9)
    ax.grid(alpha=0.2)
for k in range(n, rows * cols):
    axs[k // cols][k % cols].axis("off")
fig.suptitle(f"{title} centre {np.round(C, 1).tolist()} r={R} mm, {len(T)} triangles", fontsize=11)
fig.tight_layout()
fig.savefig(outp, dpi=95)
print(outp)
