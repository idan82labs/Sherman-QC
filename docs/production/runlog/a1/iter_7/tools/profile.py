"""Cross-section profile of a bend's two sides on a physical scan (diagnostic only, A1 iter 7).

usage: profile.py COMPILED_DIR RECIPE SCAN LABEL[,LABEL...] [--far 60] [--zbins 3]

In-process E3 up to the chosen pose (as dbg_angle.py), then per bend and side: the skin levels (distance to that
side's region-wide OML plane, outward positive) binned along s (distance from the region OML sharp line, in-plane,
away from the bend), in 0.5 mm bins up to the side's fit depth (+ a few mm), for each axial third. Then the local
slope of the OML skin (deg, relative to the region plane; positive = surface rises outward with s) over
consecutive windows, and what included angle each window pair gives.
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect import metrology as M
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply, unit
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

cdir, recipe, scan, labels = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(",")
args = sys.argv[5:]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d  # noqa: E731
FAR = opt("--far", 60.0)
ZB = int(opt("--zbins", 3))
cfg = InspectConfig()
inputs = load_inputs(cdir, recipe, scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points),
            min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel)
vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None,
                     median_spacing(reg_src) if reg_src is not pts else spacing, mirrored=False)
scored = list(proper[:cfg.articulated_hypotheses])
for h in proper[cfg.articulated_hypotheses:]:
    if any(E._matching_symmetry(h.fit.T @ np.linalg.inv(g.fit.T), part, cfg, rms_max=cfg.near_twin_rms_mm) for g in scored):
        scored.append(h)
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted([h for h in proper if h.art is not None], key=lambda h: -h.total)
h = E._ambiguity(proper, part, cfg, evidence=lambda h: E._hole_evidence(h, pts, vox, part, cfg, spacing))[3]
full = apply(h.fit.T, pts)
idx = M.PointIndex(full, vox)
art = h.art
lab = {b.label: b.index for b in part.bends}


def line_fit(s, y):
    if len(s) < 20 or np.ptp(s) < 0.8:
        return None
    A = np.c_[s, np.ones(len(s))]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ sol
    keep = np.abs(r) < max(3.0 * 1.4826 * np.median(np.abs(r)), 0.01)
    sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return float(sol[0]), float(sol[1]), int(keep.sum())


for name in labels:
    j = lab[name]
    bend = part.bends[j]
    parent, child = art.panels[bend.parent], art.panels[bend.child]
    bm = M.measure_bend_geometry(idx, part, art, j, cfg)
    print(f"=== {name} P{bend.parent}->P{bend.child} nominal sweep {math.degrees(bend.sweep):.2f} Ri {bend.ri:.3f}"
          f" ro {bend.ro:.3f} ok={bm.ok} reason={bm.reason}")
    if not bm.ok:
        continue
    t = float(parent.t or part.t)
    pp, pc = bm.plane_parent, bm.plane_child
    a, lp = bm.line_dir, bm.line_point
    sweep = bm.sweep_deg
    d_t = bend.ro * math.tan(math.radians(min(abs(sweep), 170.0)) / 2.0)
    print(f"  flat_shift p/c {bm.flat_shift[0]:.2f}/{bm.flat_shift[1]:.2f}")
    print(f"  region angle {180 - math.degrees(math.acos(max(-1, min(1, float(pp.n @ pc.n))))):.3f}"
          f"  station median {np.median(bm.angles[bm.valid]):.3f}  d_t {d_t:.2f}  t {t:.3f}"
          f"  depth p/c {bm.depth_parent + d_t + 0.5:.1f}/{bm.depth_child + d_t + 0.5:.1f}  visible {bm.visible[0]:.1f}..{bm.visible[1]:.1f}")
    rel = full - lp
    z = rel @ a
    lo, hi = bm.visible
    zedges = np.linspace(lo, hi, ZB + 1)
    slopes = {}
    for side, pl, w, depth in (("parent", pp, bm.w_parent, bm.depth_parent + d_t + 0.5),
                               ("child", pc, bm.w_child, bm.depth_child + d_t + 0.5)):
        s = rel @ w
        dd = pl.distance(full)
        top = min(depth + 4.0, FAR)
        sel = (np.abs(dd + 0.5 * t) < 0.5 * t + 0.8) & (s > -1.0) & (s < top) & (z > lo) & (z < hi)
        print(f"  -- {side}: points {int(sel.sum())}  (s bins of 0.5 mm; outer-skin level / inner-skin level+t, mm, per axial third)")
        FINE = "--fine" in args
        bins = np.arange(-1.0, (min(top, 12.0) if FINE else top) + 1e-9, 0.25 if FINE else (0.5 if top < 25 else 2.0))
        for zi in range(ZB):
            inz = sel & (z >= zedges[zi]) & (z < zedges[zi + 1])
            row = []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = inz & (s >= b0) & (s < b1)
                o = dd[m & (dd > -0.5 * t)]
                ii = dd[m & (dd <= -0.5 * t)]
                row.append(f"{0.5 * (b0 + b1):5.1f}:{(np.median(o) if len(o) > 5 else float('nan')):+.3f}/"
                           f"{(np.median(ii) + t if len(ii) > 5 else float('nan')):+.3f}")
            print(f"     z[{zedges[zi]:7.1f},{zedges[zi + 1]:7.1f}] " + " ".join(row))
        # local slope of the OML skin over windows
        win = []
        start = d_t + 0.5
        step = 2.0 if depth < 25 else 6.0
        while start + step <= depth + 1e-6:
            win.append((start, start + step))
            start += step
        win.append((d_t + 0.5, depth))
        out = []
        for (w0, w1) in win:
            m = sel & (s >= w0) & (s < w1) & (np.abs(dd) < 0.25)
            f = line_fit(s[m], dd[m])
            out.append((w0, w1, None if f is None else math.degrees(math.atan(f[0])), 0 if f is None else f[2]))
        slopes[side] = out
        print("     OML slope vs region plane (deg): " + "  ".join(
            f"[{w0:.1f},{w1:.1f}]={'nan' if sl is None else f'{sl:+.3f}'}(n{n})" for w0, w1, sl, n in out))
    print("  region angle corrected by window slopes: included = region + slope_parent + slope_child")
    ra = 180 - math.degrees(math.acos(max(-1, min(1, float(pp.n @ pc.n)))))
    for wp in slopes["parent"]:
        for wc in slopes["child"]:
            if wp[2] is None or wc[2] is None:
                continue
            if (wp, wc) and (wp is slopes["parent"][-1] or wc is slopes["child"][-1] or wp[0] == wc[0]):
                print(f"     parent[{wp[0]:.1f},{wp[1]:.1f}] child[{wc[0]:.1f},{wc[1]:.1f}] -> {ra + wp[2] + wc[2]:.3f}")
