"""flat-start trace (A1 iter 7 diagnostic). Cross-section profile of a bend's two sides on a physical scan (diagnostic only, A1 iter 7).

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



import domains.production.engine.inspect.metrology as MM
orig = MM._flat_start
def traced(s, d, z, start0, depth, t, zlo, zhi, cfg):
    g, gap, step = cfg.flat_start_test_mm, cfg.flat_start_gap_mm, cfg.flat_start_step_mm
    band = 0.45 * t
    sel = (s >= start0) & (s <= depth) & (z >= zlo) & (z <= zhi)
    ss, dd, zz = s[sel], d[sel], z[sel]
    need = cfg.min_station_points * cfg.stations_min
    skins = [(m, lv) for m, lv in ((np.abs(dd) < band, 0.0), (np.abs(dd + t) < band, -t)) if int(m.sum()) >= need]
    print(f"    start0 {start0:.2f} depth {depth:.2f} skins {[(int(m.sum()), lv) for m, lv in skins]}")
    if skins:
        keep, level = max(skins, key=lambda k: int(k[0].sum()))
        ss, dd, zz = ss[keep], dd[keep] - level, zz[keep]
        n_chunks = max(1, int(round((zhi - zlo) / cfg.flat_start_chunk_mm)))
        chunk = np.clip(((zz - zlo) / max(zhi - zlo, 1e-9) * n_chunks).astype(np.int64), 0, n_chunks - 1)
        s0 = start0
        while s0 <= depth - 3.0:
            r0 = s0 + g + gap; L = min(cfg.flat_start_ref_mm, depth - r0)
            out = []
            for k in range(n_chunks):
                gi = chunk == k
                sg, dg = ss[gi], dd[gi]
                ref = (sg >= r0) & (sg < r0 + L)
                if int(ref.sum()) < 20 or float(np.ptp(sg[ref])) < 0.6 * L:
                    out.append("skip"); continue
                line = MM._robust_line(sg[ref], dg[ref])
                if line is None: out.append("none"); continue
                sag = MM._sagitta(sg[ref][line[2]], dg[ref][line[2]])
                tst = (sg >= s0) & (sg < s0 + g)
                r = dg[tst] - (line[0] + line[1] * sg[tst])
                r = r[(r > -band) & (r < 0.15)]
                out.append(f"n{int(ref.sum())}/{len(r)} sag{sag:+.3f} med{(np.median(r) if len(r) else float('nan')):+.3f}")
            print(f"    s0 {s0:.2f}: " + " | ".join(out))
            s0 += step
    res = orig(s, d, z, start0, depth, t, zlo, zhi, cfg)
    print(f"    -> {res:.2f}")
    return res
MM._flat_start = traced
for name in labels:
    j = lab[name]
    print("===", name)
    bm = M.measure_bend_geometry(idx, part, art, j, cfg)
