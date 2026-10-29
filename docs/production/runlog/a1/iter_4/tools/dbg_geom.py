"""Why measure_bend_geometry refuses a bend: replays its selection on the chosen hypothesis (in-process E3).
usage: dbg_geom.py COMPILED RECIPE SCAN LABEL[,LABEL...]"""
import hashlib, math, sys
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
cfg = InspectConfig()
inputs = load_inputs(cdir, recipe, scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None, median_spacing(reg_src) if reg_src is not pts else spacing, mirrored=False)
scored = list(proper[:cfg.articulated_hypotheses])
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted([h for h in proper if h.art is not None], key=lambda h: -h.total)
h = E._ambiguity(proper, part, cfg, evidence=lambda h: E._hole_evidence(h, pts, vox, part, cfg, spacing))[3]
full = apply(h.fit.T, pts)
idx = M.PointIndex(full, vox)
art = h.art
lab = {b.label: b.index for b in part.bends}
for name in labels:
    j = lab[name]; bend = part.bends[j]; bf = art.bends[j]
    parent, child = art.panels[bend.parent], art.panels[bend.child]
    frame = bf.frame
    t = float(parent.t or part.t)
    skin = bend.oml_skin
    pp, pc = M._oml_plane(parent, skin), M._oml_plane(child, skin)
    lp, ld = M._intersect(pp, pc)
    a = unit(ld if float(ld @ frame.a) > 0 else -ld)
    sweep = math.degrees(math.acos(max(-1.0, min(1.0, float(pp.n @ pc.n)))))
    ro = bend.ro
    d_t = ro * math.tan(math.radians(min(sweep, 170.0)) / 2.0)
    rel = full - lp; z = rel @ a
    z_frame0 = float((lp - frame.o) @ frame.a)
    zlo_nom, zhi_nom = frame.z0 - z_frame0, frame.z1 - z_frame0
    w_p = M._in_plane_dir(pp, a, -frame.u); w_c = M._in_plane_dir(pc, a, frame.line_dir(math.radians(bf.theta_deg or 0.0)))
    s_p, s_c = rel @ w_p, rel @ w_c
    dp, dc = pp.distance(full), pc.distance(full)
    slab = 0.5 * t + 0.6
    depth_p = M._flat_depth(part, bend, bend.parent, d_t, cfg.local_fit_depth_mm, cfg)
    depth_c = M._flat_depth(part, bend, bend.child, d_t, cfg.local_fit_depth_mm, cfg)
    inz = (z > zlo_nom - 1) & (z < zhi_nom + 1)
    near_p = (np.abs(dp + 0.5 * t) < slab) & (s_p > d_t + 0.5) & (s_p < depth_p) & inz
    near_c = (np.abs(dc + 0.5 * t) < slab) & (s_c > d_t + 0.5) & (s_c < depth_c) & inz
    print(f"=== {name} P{bend.parent}->P{bend.child} sweep {sweep:.2f} d_t {d_t:.2f} depth_p {depth_p:.2f} depth_c {depth_c:.2f} z_nom {zlo_nom:.1f}..{zhi_nom:.1f} t {t:.3f}")
    print(f"   parent {parent.state} vox {len(parent.voxels)} skins {parent.skins_seen}; child {child.state} vox {len(child.voxels)} skins {child.skins_seen}")
    print(f"   near_p {int(near_p.sum())} near_c {int(near_c.sum())} need {cfg.min_station_points * cfg.stations_min}")
    print(f"   in z & strip (any dist): parent {int(((s_p > d_t + 0.5) & (s_p < depth_p) & inz).sum())} child {int(((s_c > d_t + 0.5) & (s_c < depth_c) & inz).sum())}")
    for nm, d_, s_, dep in (("parent", dp, s_p, depth_p), ("child", dc, s_c, depth_c)):
        strip = (s_ > d_t + 0.5) & (s_ < dep) & inz
        if strip.sum():
            hist, edges = np.histogram(d_[strip], bins=np.arange(-3 * t, 2 * t + 0.01, 0.25))
            print(f"   {nm} dist-to-OML hist (bins of 0.25 from {-3*t:.2f}):", hist.tolist())
    bm = M.measure_bend_geometry(idx, part, art, j, cfg)
    print("   metrology", bm.ok, bm.reason, None if not bm.ok else (round(bm.sweep_deg, 2), int(bm.valid.sum()), len(bm.z)))
