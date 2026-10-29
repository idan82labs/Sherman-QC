#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4 (10839000 scan_1): global plate frame + height map (read-only).

Fits the main plate (mid-plane of both skins), expresses every scan point in a plate frame
(u, v in-plane, h = signed height above the plate mid-plane) and renders a top view coloured
by h so that flanges, joggles, steps and lips are visible. Gold lines (green) and engine lines
(dashed) are overlaid. Writes only into this directory.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/Users/idantal/ShermanQC/FullProject")
OUT = Path(__file__).resolve().parent
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_1.ply")
GOLD = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/72bd51c4885d493d805f1108.json"
RUN = ROOT / "docs/production/runlog/broad_baseline/gold11_repro/runs/sc_08686dc1d5d4/best_f1"


def load_points():
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(SCAN))
    p = np.asarray(pc.points, float)
    return p[np.isfinite(p).all(1)]


def gold_lines():
    g = json.loads(GOLD.read_text())
    return [(b["bend_id"], np.array(b["line_start"], float), np.array(b["line_end"], float)) for b in g["bends"]]


def engine_lines():
    out = []
    for fname, tag in (("predictions_best_f1.json", "best_f1"), ("predictions_cad_raw_ungated.json", "cad_raw"),
                       ("predictions_f1_zero_shot_decomposition.json", "zs_decomp")):
        p = RUN / fname
        if not p.is_file():
            continue
        for r in json.loads(p.read_text()).get("owned_bend_regions") or []:
            if r.get("line_start") and r.get("line_end"):
                out.append((tag, r["bend_id"], np.array(r["line_start"], float), np.array(r["line_end"], float)))
    return out


def plate_frame(P):
    """Mid-plane of the dominant plate (both skins) and in-plane axes."""
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(len(P), 120000), replace=False)]
    best = None
    for _ in range(400):
        i = rng.choice(len(sub), 3, replace=False)
        a, b, c = sub[i]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        d = np.abs((sub - a) @ n)
        cnt = int((d < 1.2).sum())
        if best is None or cnt > best[0]:
            best = (cnt, a, n)
    _, a, n = best
    for thr in (2.0, 1.5):
        m = np.abs((P - a) @ n) < thr
        Q = P[m]
        cen = Q.mean(0)
        _, _, vt = np.linalg.svd(Q[rng.choice(len(Q), min(len(Q), 200000), replace=False)] - cen, full_matrices=False)
        n = vt[2]
        a = cen
    e1, e2 = vt[0], vt[1]
    if np.cross(e1, e2) @ n < 0:
        e2 = -e2
    return a, e1, e2, n, m


def main():
    P = load_points()
    o, e1, e2, n, platemask = plate_frame(P)
    h = (P - o) @ n
    # orient n so that the flanges (largest |h| mass) are at negative h (flanges hang "down")
    far = np.abs(h) > 8
    if far.sum() and np.median(h[far]) > 0:
        n = -n
        e2 = -e2
        h = -h
    u = (P - o) @ e1
    v = (P - o) @ e2
    frame = {"origin": o.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "n": n.tolist(),
             "n_points": int(len(P)), "plate_points_within_1p5mm": int(platemask.sum())}
    # skin offsets of the plate: histogram of h near 0
    hh = h[np.abs(h) < 3]
    hist, edges = np.histogram(hh, bins=120, range=(-3, 3))
    frame["h_hist_peaks"] = [float(0.5 * (edges[i] + edges[i + 1])) for i in np.argsort(hist)[-6:]]
    np.save(OUT / "_plate_frame.npy", np.vstack([o, e1, e2, n]))
    (OUT / "plate_frame.json").write_text(json.dumps(frame, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = gold_lines()
    E = engine_lines()

    def to2(x):
        x = np.atleast_2d(x) - o
        return np.c_[x @ e1, x @ e2]

    rng = np.random.default_rng(1)
    sel = rng.choice(len(P), min(len(P), 400000), replace=False)
    fig, ax = plt.subplots(figsize=(16, 14), dpi=100)
    hs = h[sel]
    flat = np.abs(hs) <= 4
    sc = ax.scatter(u[sel][flat], v[sel][flat], c=np.clip(hs[flat], -2.5, 2.5), cmap="coolwarm", s=0.3, linewidths=0,
                    rasterized=True, vmin=-2.5, vmax=2.5)
    ax.scatter(u[sel][~flat], v[sel][~flat], c=np.clip(hs[~flat], -40, 40), cmap="viridis", s=0.3, linewidths=0,
               rasterized=True, vmin=-40, vmax=10)
    plt.colorbar(sc, ax=ax, fraction=0.03, label="h above plate mid-plane (mm), |h|<=4")
    for bid, s0, s1 in G:
        e = to2(np.array([s0, s1]))
        ax.plot(e[:, 0], e[:, 1], color="#1a9c3a", lw=2.2)
        m = e.mean(0)
        ax.text(m[0], m[1], bid, color="white", fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="#1a9c3a", ec="none"))
        ax.plot(e[0, 0], e[0, 1], "o", color="#1a9c3a", ms=5)
    cols = {"best_f1": "#d62728", "cad_raw": "#ff7f0e", "zs_decomp": "#9467bd"}
    for tag, pid, s0, s1 in E:
        e = to2(np.array([s0, s1]))
        ax.plot(e[:, 0], e[:, 1], color=cols[tag], lw=1.0, ls="--")
        ax.text(e[1, 0], e[1, 1], f"{tag}:{pid[:10]}", color=cols[tag], fontsize=6)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("sc_08686dc1d5d4 top view in plate frame (u=e1, v=e2); colour = height above plate mid-plane; "
                 "flanges (|h|>4) viridis; gold green, engine dashed", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "a3_heightmap_top.png")
    plt.close(fig)

    # height histogram along h for all points (shows flange depth levels)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=90)
    ax.hist(h, bins=400, range=(-45, 10), log=True)
    ax.set_xlabel("h (mm)")
    fig.tight_layout()
    fig.savefig(OUT / "a3_h_hist.png")
    plt.close(fig)

    # gold lines in plate frame
    rows = []
    for bid, s0, s1 in G:
        e = to2(np.array([s0, s1]))
        hz = ((np.array([s0, s1]) - o) @ n).tolist()
        d = s1 - s0
        rows.append({"bend_id": bid, "uv_start": e[0].tolist(), "uv_end": e[1].tolist(), "h_start_end": hz,
                     "len": float(np.linalg.norm(d)),
                     "angle_to_n_deg": float(math.degrees(math.acos(abs(d @ n) / np.linalg.norm(d))))})
    for tag, pid, s0, s1 in E:
        e = to2(np.array([s0, s1]))
        rows.append({"engine": tag, "bend_id": pid, "uv_start": e[0].tolist(), "uv_end": e[1].tolist(),
                     "h_start_end": ((np.array([s0, s1]) - o) @ n).tolist(), "len": float(np.linalg.norm(s1 - s0))})
    (OUT / "lines_in_plate_frame.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(frame, indent=1))
    for r in rows:
        print(r)
    # extents of the plate / flanges
    print("u range", np.percentile(u, [0.1, 99.9]), "v range", np.percentile(v, [0.1, 99.9]))
    for lo, hi in ((-50, -30), (-30, -10), (-10, -4), (-4, -2), (2, 4), (4, 10), (10, 50)):
        m = (h > lo) & (h <= hi)
        if m.sum():
            print(f"h in ({lo},{hi}]: n={m.sum()} u[{u[m].min():.1f},{u[m].max():.1f}] v[{v[m].min():.1f},{v[m].max():.1f}]")


if __name__ == "__main__":
    main()
