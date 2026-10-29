"""Angles with and without the flat-start search (A1 iter 7 diagnostic). Cross-section profile of a bend's two sides on a physical scan (diagnostic only, A1 iter 7).

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

cdir, recipe, scan = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SETS = [a.split("=", 1) for a in sys.argv[4:] if "=" in a]
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



from dataclasses import replace
cfg_off = replace(cfg, flat_start_max_shift_mm=0.0)
cfg = replace(cfg, **{k: float(v) for k, v in SETS})
for b in part.bends:
    j = b.index
    if art.panels[b.child].state != "FITTED":
        print(f"{b.label:4s} child {art.panels[b.child].state}")
        continue
    bm0 = M.measure_bend_geometry(idx, part, art, j, cfg_off)
    bm1 = M.measure_bend_geometry(idx, part, art, j, cfg)
    if not (bm0.ok and bm1.ok):
        print(f"{b.label:4s} ok {bm0.ok}/{bm1.ok} {bm0.reason}/{bm1.reason}")
        continue
    a0 = M.angle_measurement(bm0, cfg_off); a1 = M.angle_measurement(bm1, cfg)
    nom = 180.0 - math.degrees(b.sweep)
    print(f"{b.label:4s} nom {nom:7.2f}  off {a0.value:8.3f} (U {a0.u:.3f})  on {a1.value:8.3f} (U {a1.u:.3f})  "
          f"d {a1.value - a0.value:+.3f}  shift p/c {bm1.flat_shift[0]:.2f}/{bm1.flat_shift[1]:.2f}  "
          f"dev off {a0.value - nom:+.3f} on {a1.value - nom:+.3f}")
