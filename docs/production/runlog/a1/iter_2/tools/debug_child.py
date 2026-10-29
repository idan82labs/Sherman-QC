"""Run E3 registration + articulated fit on a probe case and print panel fits and bend geometry."""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import metrology as M
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.geom import apply

cdir = Path(sys.argv[1])
scan = next(cdir.glob("scan.*"))
cfg = InspectConfig()
inputs = load_inputs(cdir / "compiled", cdir / "recipe.json", scan, hashlib.sha256(scan.read_bytes()).hexdigest())
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
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None, median_spacing(reg_src), mirrored=False)
h = proper[0]
E._articulate(h, pts, vox, part, model, cfg, budget)
print("voxel", voxel, "slab", vox.slab, "spacing", spacing)
for p_ in h.art.panels:
    pv = part.panels[p_.index]
    print(f"panel {pv.panel_id} state {p_.state} skins {p_.skins_seen} voxels {len(p_.voxels)} region {len(p_.region)} n0 {None if p_.n0 is None else np.round(p_.n0, 3)} d0 {p_.d0}")
for b in part.bends:
    bf = h.art.bends[b.index]
    print(f"bend {b.label} theta {bf.theta_deg} rho {bf.rho_iml} score {bf.score} found {bf.found} flat {bf.flat} one_skin {bf.one_skin} detail {bf.detail}")
reg = apply(h.fit.T, pts)
idx = M.PointIndex(reg, vox)
for j, b in enumerate(part.bends):
    bm = M.measure_bend_geometry(idx, part, h.art, j, cfg)
    if not bm.ok:
        print("geom", b.label, "not ok", bm.reason); continue
    print(f"geom {b.label} sweep {bm.sweep_deg:.2f} angles {np.round(bm.angles[bm.valid], 2)} inner p/c {bm.inner_parent:.2f}/{bm.inner_child:.2f} support {bm.support_parent}/{bm.support_child} depth {bm.depth_parent:.1f}/{bm.depth_child:.1f}")
# station internals for one bend (argv[2])
if len(sys.argv) > 2:
    lab = sys.argv[2]
    j = next(b.index for b in part.bends if b.label == lab)
    bend = part.bends[j]; bf = h.art.bends[j]
    parent, child = h.art.panels[bend.parent], h.art.panels[bend.child]
    t = float(parent.t or part.t); skin = bend.oml_skin
    pp, pc = parent.skin_plane(skin), child.skin_plane(skin)
    print("oml skin", skin, "pp", np.round(pp.n, 3), round(pp.d, 3), "pc", np.round(pc.n, 3), round(pc.d, 3), "child skins seen", child.skins_seen)
    line = M._intersect(pp, pc); lp, ld = line
    a = M.unit(ld if float(ld @ bf.frame.a) > 0 else -ld)
    sweep = math.degrees(math.acos(max(-1, min(1, float(pp.n @ pc.n)))))
    d_t = bend.ro * math.tan(math.radians(min(sweep, 170)) / 2)
    w_c = M._in_plane_dir(pc, a, bf.frame.line_dir(math.radians(bf.theta_deg or 0)))
    depth_c = M._flat_depth(part, bend, bend.child, d_t, cfg.local_fit_depth_mm, cfg)
    rel = reg - lp; s_c = rel @ w_c; dc = pc.distance(reg); z = rel @ a
    near_c = (np.abs(dc + 0.5 * t) < 0.5 * t + 0.6) & (s_c > d_t + 0.5) & (s_c < depth_c)
    print(f"sweep(regions) {sweep:.2f} d_t {d_t:.2f} depth_c {depth_c:.2f} near_c {near_c.sum()}")
    q = reg[near_c]; dq = dc[near_c]; sq = s_c[near_c]
    hist, edges = np.histogram(dq, bins=np.arange(-t - 0.7, 0.71, 0.1))
    print("dc histogram (OML plane at 0, IML at -t):", list(zip(np.round(edges[:-1], 1).tolist(), hist.tolist())))
    hs, es = np.histogram(sq, bins=np.arange(0, depth_c + 0.5, 0.25))
    print("s_c histogram:", list(zip(np.round(es[:-1], 2).tolist(), hs.tolist())))
    fc = M._fit_side(q, pc, t, cfg)
    print("fit_side all:", None if fc[0] is None else (np.round(fc[0].n, 3), fc[1], fc[2], round(fc[3], 3)),
          "angle to parent", None if fc[0] is None else round(180 - math.degrees(math.acos(max(-1, min(1, float(fc[0].n @ pp.n))))), 2))
