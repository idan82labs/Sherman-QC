"""In-process E3 (registration -> articulated -> pose choice) on one scan, then per-bend diagnostics.

usage: dbg_scan.py COMPILED_DIR RECIPE SCAN [--bends B1,B2] [--png PREFIX] [--hyp K] [--cfg key=val ...]
Prints every articulated hypothesis (total, datum, explained, states), the chosen one, and for the listed bends:
BendFit, child PanelFit, metrology (measure_bend_geometry) and a hinge-band plot (u, v) with the line and regions.
"""
import hashlib, math, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply
from domains.production.engine.inspect.metrology import PointIndex, measure_bend_geometry
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

args = sys.argv[1:]
def opt(name, default=None):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
bends_req = (opt("--bends", "") or "").split(",")
png = opt("--png")
hk = opt("--hyp")
overrides = {}
while "--cfg" in args:
    i = args.index("--cfg"); k, v = args[i + 1].split("="); del args[i:i + 2]
    overrides[k] = type(getattr(InspectConfig(), k))(v)
cdir, recipe, scan = Path(args[0]), Path(args[1]), Path(args[2])
cfg = replace(InspectConfig(), **overrides) if overrides else InspectConfig()
inputs = load_inputs(cdir, recipe, scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel)
vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
notes = []
proper = E._register(reg_src, vox, part, model, cfg, budget, notes.append, median_spacing(reg_src) if reg_src is not pts else spacing, mirrored=False)
print("\n".join(notes))
scored = list(proper[:cfg.articulated_hypotheses])
for h in proper[cfg.articulated_hypotheses:]:
    if any(E._matching_symmetry(h.fit.T @ np.linalg.inv(g.fit.T), part, cfg, rms_max=cfg.near_twin_rms_mm) for g in scored):
        scored.append(h)
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted([h for h in proper if h.art is not None], key=lambda h: -h.total)
amb, amb_states, margin, chosen = E._ambiguity(proper, part, cfg, evidence=lambda h: E._hole_evidence(h, pts, vox, part, cfg, spacing), note=print)
T0 = proper[0].fit.T
for i, h in enumerate(proper):
    rel = h.fit.T @ np.linalg.inv(T0)
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(rel[:3, :3]) - 1) / 2))))
    sym = E._matching_symmetry(rel, part, cfg, rms_max=cfg.near_twin_rms_mm) if i else []
    print(f"hyp{i}{' *' if h is chosen else '  '} total {h.total:.4f} datum {h.fit.score:.4f} expl {h.explained:.4f}/{h.explained_tight:.4f} "
          f"rot {ang:.1f} t {np.linalg.norm(rel[:3, 3]):.1f} twin={bool(sym)} states " + " ".join(f"{part.bends[j].label}:{d.state[:3]}" for j, d in enumerate(h.decodes)))
print(f"ambiguous {amb} margin {margin:.4f}")
h = proper[int(hk)] if hk is not None else chosen
T = h.fit.T
full = apply(T, pts)
idx = PointIndex(full, vox)
art = h.art
lab = {b.label: b.index for b in part.bends}
for name in [b for b in bends_req if b]:
    j = lab[name]
    bend = part.bends[j]
    bf = art.bends[j]
    par, ch = art.panels[bend.parent], art.panels[bend.child]
    print(f"\n=== {name} P{bend.parent}->P{bend.child} sweep {math.degrees(bend.sweep):.1f} state {h.decodes[j].state}")
    print(f"  bendfit found={bf.found} flat={bf.flat} theta={bf.theta_deg} rho={bf.rho_iml} score={bf.score} pts={bf.child_line_points} cov={bf.axial_coverage:.2f} one_skin={bf.one_skin} reason={bf.reason} detail={ {k: round(v, 3) for k, v in bf.detail.items()} }")
    print(f"  parent {par.state} skins={par.skins_seen} vox={len(par.voxels)}; child {ch.state} reason={ch.reason} skins={ch.skins_seen} vox={len(ch.voxels)} region={len(ch.region)}")
    if bf.frame is not None and ch.n0 is not None and par.n0 is not None:
        sw = math.degrees(math.acos(max(-1, min(1, float(par.skin_plane(bend.oml_skin).n @ ch.skin_plane(bend.oml_skin).n)))))
        print(f"  region-wide sweep {sw:.2f}")
    if ch.state == "FITTED":
        bm = measure_bend_geometry(idx, part, art, j, cfg)
        print(f"  metrology ok={bm.ok} reason={bm.reason} sweep={bm.sweep_deg} stations={len(bm.z)} valid={int(bm.valid.sum()) if len(bm.valid) else 0} "
              f"angles={np.round(bm.angles[bm.valid], 2).tolist() if len(bm.valid) else []} depth_p={bm.depth_parent:.2f} depth_c={bm.depth_child:.2f}")
    if png and bf.frame is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr = bf.frame
        reach = fr.band * cfg.hough_fine_reach
        widx, wz, wu, wv = AR._band_points(full, fr, 2.0, radius=reach)
        fig, ax = plt.subplots(figsize=(9, 9))
        own = art.owner[vox.inverse[widx]]
        cols = np.where(own == bend.parent, 1, np.where(own == bend.child, 2, np.where(own >= 0, 3, 0)))
        sub = np.arange(len(widx))[:: max(1, len(widx) // 80000)]
        cmap = np.array(["0.6", "b", "r", "g"])
        ax.scatter(wu[sub], wv[sub], s=0.3, c=cmap[cols[sub]])
        if bf.theta_deg is not None and bf.rho_iml is not None:
            th = math.radians(bf.theta_deg)
            d = np.array([math.cos(th), math.sin(th)]); n = np.array([-math.sin(th), math.cos(th)])
            for off, st in ((bf.rho_iml, "m-"), (bf.rho_iml + fr.delta, "m--")):
                p0 = n * off
                ax.plot([p0[0], p0[0] + reach * d[0]], [p0[1], p0[1] + reach * d[1]], st, lw=1.0)
        ax.set_xlim(-reach * 0.5, reach); ax.set_ylim(-reach * 0.75, reach * 0.75); ax.set_aspect("equal")
        ax.set_title(f"{name} band: blue parent P{bend.parent}, red child P{bend.child}, green other, grey free; theta {bf.theta_deg}")
        out = f"{png}_{name}.png"
        fig.savefig(out, dpi=80); plt.close(fig)
        print("  wrote", out)
