"""Post-selection placement for Z4/Z5 lines (scan-only, no fitted parameters; all thresholds fixed a priori).

1. Span repair: walk stations every 2 mm along the line's axis, from 60 mm (or half the length) before the
   start to the same after the end. A station is "fold" when the full-density cloud has points within
   max(0.75, 3 x spacing) of it AND the 3-25 mm ring of the cross-section slab shows exactly two flange arms.
   The new span is the contiguous fold run (gaps <= 10 mm bridged) that contains the most of the original
   span; it replaces the old span only when it covers >= 50 % of the old one and is >= 10 mm long.
2. Gated outer-apex snap (D1, research/zeroshot/d1_two_skin.two_skin.snap_line_to_outer_apex), shift gate
   t / sin(theta/2) + 1 mm, as recommended by D1_REPORT.

  .venv/bin/python -B -m research.zeroshot.z4_selector.postproc --in-name iter1 --out-name iter1_pp [--no-snap]
Reads z4/<in-name>/eval_lines/<scan>.json, writes z4/<out-name>/eval_lines + summary.json (scored OOF:
the inputs are already out-of-fold selections and nothing here is fitted).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.candidates import load_points
from research.zeroshot.z4_selector.custody import OUT, check_dev_scan, dev_ids, dev_inventory

STEP = 2.0
RING = (3.0, 25.0)
GAP = 10.0


def _basis(u):
    h = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, h); e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def two_arms(theta: np.ndarray) -> bool:
    bins = ((theta + math.pi) / (2 * math.pi) * 36).astype(int) % 36
    hist = np.bincount(bins, minlength=36)
    occ = hist >= max(2, 0.02 * len(theta))
    if occ.all() or not occ.any():
        return False
    start = int(np.argmin(occ))
    runs, cur = [], 0
    for k in range(36):
        b = (start + k) % 36
        if occ[b]:
            cur += hist[b]
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    big = [r for r in runs if r >= 0.08 * len(theta)]
    return len(big) == 2


def repair_span(tree, coarse, ctree, spacing, a, b, extend_only=False):
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    e1, e2 = _basis(u)
    ext = min(60.0, 0.5 * L)
    ts = np.arange(-ext, L + ext + 1e-9, STEP)
    cs = a[None] + ts[:, None] * u[None]
    near_r = max(0.75, 3 * spacing)
    near = np.array([len(x) for x in tree.query_ball_point(cs, near_r)]) >= 3
    ok = np.zeros(len(ts), bool)
    balls = ctree.query_ball_point(cs, math.hypot(RING[1], 1.5))
    for i, idx in enumerate(balls):
        if not near[i] or len(idx) < 8:
            continue
        d = coarse[idx] - cs[i]
        d = d[np.abs(d @ u) <= 1.5]
        uv = np.stack([d @ e1, d @ e2], 1)
        r = np.linalg.norm(uv, axis=1)
        ring = (r >= RING[0]) & (r <= RING[1])
        if ring.sum() >= 6:
            ok[i] = two_arms(np.arctan2(uv[ring, 1], uv[ring, 0]))
    # runs with gap bridging
    runs, s, last = [], None, None
    for i in np.flatnonzero(ok):
        if s is None:
            s = last = i
        elif (i - last) * STEP <= GAP:
            last = i
        else:
            runs.append((s, last)); s = last = i
    if s is not None:
        runs.append((s, last))
    best, cov = None, 0.0
    for s, e in runs:
        lo, hi = ts[s] - STEP / 2, ts[e] + STEP / 2
        c = max(0.0, min(hi, L) - max(lo, 0.0)) / L
        if c > cov:
            best, cov = (lo, hi), c
    if best is None or cov < 0.5 or best[1] - best[0] < 10.0:
        return a, b, {"repaired": False, "cov": round(cov, 3)}
    if extend_only:
        best = (min(best[0], 0.0), max(best[1], L))
        if best == (0.0, L):
            return a, b, {"repaired": False, "cov": round(cov, 3)}
    return a + best[0] * u, a + best[1] * u, {"repaired": True, "cov": round(cov, 3),
                                              "new_len": round(best[1] - best[0], 1), "old_len": round(L, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--no-span", action="store_true")
    ap.add_argument("--extend-only", action="store_true")
    a = ap.parse_args(argv)
    from scipy.spatial import cKDTree

    inv = dev_inventory()
    src = OUT / a.in_name / "eval_lines"
    dst = OUT / a.out_name / "eval_lines"
    dst.mkdir(parents=True, exist_ok=True)
    recs, stats = [], {"repaired": 0, "snapped": 0, "lines": 0}
    for sid in dev_ids():
        d = json.loads((src / f"{sid}.json").read_text())
        lines = d.get("owned_bend_regions") or []
        out = []
        if lines:
            scan = check_dev_scan(sid, inv)
            pts = load_points(scan)
            tree = cKDTree(pts)
            rng = np.random.default_rng(0)
            coarse = pts if len(pts) <= 200_000 else pts[rng.choice(len(pts), 200_000, replace=False)]
            ctree = cKDTree(coarse)
            probe = pts[rng.choice(len(pts), min(5000, len(pts)), replace=False)]
            spacing = float(np.median(tree.query(probe, k=2)[0][:, 1]))
            mesh = t = None
            if not a.no_snap:
                from research.zeroshot.d1_two_skin import two_skin as TS

                try:
                    mesh = TS.load_mesh(str(scan))
                    t = TS.estimate_thickness(mesh).t_mm
                except ValueError:  # point-cloud scan (no faces): D1 needs a mesh, keep the line
                    mesh = t = None
            for l in lines:
                p0, p1 = np.asarray(l["line_start"], float), np.asarray(l["line_end"], float)
                info = {}
                if not a.no_span:
                    p0, p1, info = repair_span(tree, coarse, ctree, spacing, p0, p1, a.extend_only)
                    stats["repaired"] += int(info["repaired"])
                if mesh is not None and t:
                    prof = TS.bend_profile(mesh.vertices, mesh.vertex_normals, p0, p1, t)
                    theta = prof.included_angle_deg or 90.0
                    gate = t / max(0.05, math.sin(math.radians(theta) / 2)) + 1.0
                    p0, p1, si = TS.snap_line_to_outer_apex(p0, p1, prof, max_shift_mm=gate)
                    stats["snapped"] += int(si.get("moved", False))
                stats["lines"] += 1
                out.append({**l, "line_start": p0.tolist(), "line_end": p1.tolist(), "post": info})
        recs.append(D.score_lines(sid, inv, out))
        (dst / f"{sid}.json").write_text(json.dumps({**d, "lane": f"z4_{a.out_name}", "owned_bend_regions": out}))
    res = {"in": a.in_name, "snap": not a.no_snap, "span": not a.no_span, "extend_only": a.extend_only, "stats": stats, "oof": D.aggregate(recs)}
    (OUT / a.out_name / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
