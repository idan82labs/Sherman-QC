"""In-process E3 on one scan (COMPILED RECIPE SCAN) with per-bend diagnostics of the articulated fit, the wrong-way
search and the metrology. usage: dbg_scan6.py COMPILED_DIR RECIPE SCAN B1,B2 [--png PREFIX]"""
import hashlib, math, sys
from pathlib import Path
import numpy as np
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply, signed_sweep_deg
from domains.production.engine.inspect.metrology import PointIndex, measure_bend_geometry
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
cdir, recipe, scan = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
labels = set(sys.argv[4].split(","))
png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
cfg = InspectConfig()
orig = AR._reversed_reading
def patched(full_reg, vox, vox_reg, vox_nrm, owner, part_, bend, parent, frame, sweep_deg, cad_score, reach, cfg_):
    r = orig(full_reg, vox, vox_reg, vox_nrm, owner, part_, bend, parent, frame, sweep_deg, cad_score, reach, cfg_)
    if bend.label in labels:
        fm, mb = AR.mirrored_hinge_frame(part_, bend, parent, frame)
        lo = max(-cfg_.theta_min_deg, min(cfg_.flat_sweep_max_deg, cfg_.flat_fraction_max * sweep_deg)) + 1.0
        hi = min(sweep_deg + cfg_.theta_overbend_deg, 180.0 - cfg_.reversed_wrap_margin_deg - 1.0)
        widx, wz, wu, wv = AR._band_points(full_reg, fm, 2.0, radius=reach)
        free = owner[vox.inverse[widx]] < 0
        print(f"--- rev {bend.label} cad_score {cad_score:.0f} band {len(widx)} free {int(free.sum())} range {lo:.1f}..{hi:.1f} o' {np.round(fm.o,2)} rho_nom90 {fm.rho_iml_at(math.radians(90)):.2f}")
        widx, wz, wu, wv = widx[free], wz[free], wu[free], wv[free]
        inb = np.hypot(wu, wv) < fm.band
        th = np.radians(np.arange(lo, hi + 1e-9, 1.0))
        sc, rh, ni, no = AR._hough(wz[inb], wu[inb], wv[inb], fm, th, cfg_, fm.band)
        print("   coarse:", [(round(math.degrees(th[k]), 1), round(float(sc[k])), round(float(ni[k])), round(float(no[k])), round(float(rh[k]), 2)) for k in np.argsort(-sc)[:6]])
        sc2, rh2, ni2, no2 = AR._hough(wz, wu, wv, fm, th, cfg_, reach)
        print("   reach:", [(round(math.degrees(th[k]), 1), round(float(sc2[k])), round(float(ni2[k])), round(float(no2[k])), round(float(rh2[k]), 2)) for k in np.argsort(-sc2)[:6]])
        for tdeg in (80.0, 90.0, 100.0):
            t_ = math.radians(tdeg); c_, s_ = math.cos(t_), math.sin(t_)
            sv = wu * c_ + wv * s_; rr = -wu * s_ + wv * c_
            sel = (sv > 1.0) & (sv < reach)
            h_, e_ = np.histogram(rr[sel], bins=np.arange(-6.0, 4.0, 0.2))
            print(f"   rho hist {tdeg}: " + " ".join(f"{e_[i]:.1f}:{h_[i]}" for i in range(len(h_)) if h_[i] > 15))
        print("   adopted:", None if r is None else (round(math.degrees(r[1]), 2), round(r[3])))
    return r
AR._reversed_reading = patched
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
for h in proper[:cfg.articulated_hypotheses]:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted([h for h in proper if h.art is not None], key=lambda h: -h.total)
h = proper[0]
print(f"best total {h.total:.4f} expl {h.explained:.3f} states " + " ".join(f"{part.bends[j].label}:{d.state[:3]}" for j, d in enumerate(h.decodes)))
full = apply(h.fit.T, pts)
idx = PointIndex(full, vox)
art = h.art
for j, b in enumerate(part.bends):
    if b.label not in labels:
        continue
    bf = art.bends[j]
    par, ch = art.panels[b.parent], art.panels[b.child]
    print(f"=== {b.label} P{b.parent}->P{b.child} sweep {math.degrees(b.sweep):.1f} ri {b.ri:.2f} t {part.t} decode {h.decodes[j].state} sw {h.decodes[j].measured_sweep_deg}")
    print(f"   bf found={bf.found} reversed={bf.reversed} theta={bf.theta_deg} rho={bf.rho_iml} score={bf.score:.0f} pts={bf.child_line_points} cov={bf.axial_coverage:.2f} reason={bf.reason}")
    print(f"   detail { {k: round(v, 3) for k, v in bf.detail.items()} }")
    print(f"   parent {par.state} skins={par.skins_seen} vox={len(par.voxels)} ; child {ch.state} skins={ch.skins_seen} vox={len(ch.voxels)} region={len(ch.region)} reason={ch.reason}")
    if ch.n0 is not None and par.n0 is not None and bf.frame is not None:
        pp, pc = par.skin_plane(b.oml_skin), ch.skin_plane(b.oml_skin)
        print(f"   region sweep signed {signed_sweep_deg(pp.n, pc.n, bf.frame.a, cfg.reversed_wrap_margin_deg):.2f}")
        # child's expected normal under the mirrored kinematics at the found angle vs fitted
        if bf.reversed:
            mb = AR.mirrored_bend(part, b)
            A = par.A @ AR.fold_transform(mb, math.radians(-bf.theta_deg))
            n_pred = A[:3, :3] @ np.asarray(part.panels[b.child].n0)
            print(f"   child n0 fitted·kinematic = {float(n_pred @ ch.n0):.5f} ; fitted d0 {ch.d0:.3f}")
    if ch.state == "FITTED":
        bm = measure_bend_geometry(idx, part, art, j, cfg)
        print(f"   metrology ok={bm.ok} reason={bm.reason} sweep={bm.sweep_deg} rev={bm.reversed} stations={len(bm.z)} valid={int(bm.valid.sum()) if len(bm.valid) else 0}")
        if bm.ok:
            print(f"   angles {np.round(bm.angles[bm.valid], 2).tolist()} supp p/c {bm.support_parent}/{bm.support_child} inner p/c {bm.inner_parent:.2f}/{bm.inner_child:.2f} depth p/c {bm.depth_parent:.1f}/{bm.depth_child:.1f}")
    if png and bf.frame is not None:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fr = bf.frame
        reach = fr.band * cfg.hough_fine_reach
        widx, wz, wu, wv = AR._band_points(full, fr, 2.0, radius=2 * reach)
        own = art.owner[vox.inverse[widx]]
        cols = np.where(own == b.parent, 1, np.where(own == b.child, 2, np.where(own >= 0, 3, 0)))
        cmap = np.array(["0.6", "b", "r", "g"])
        fig, ax = plt.subplots(figsize=(8, 8))
        sub = np.arange(len(widx))[:: max(1, len(widx) // 60000)]
        ax.scatter(wu[sub], wv[sub], s=0.3, c=cmap[cols[sub]])
        if bf.theta_deg is not None:
            th = math.radians(bf.theta_deg); d = np.array([math.cos(th), math.sin(th)])
            ax.plot([0, 20 * d[0]], [0, 20 * d[1]], "m-", lw=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-15, 25); ax.set_ylim(-25, 15)
        ax.set_title(f"{b.label}: blue parent, red child, green other; theta {bf.theta_deg}")
        fig.savefig(f"{png}_{b.label}.png", dpi=80); plt.close(fig)
