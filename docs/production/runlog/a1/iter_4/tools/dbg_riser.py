"""Riser (short child) line-search anatomy for one bend on a real/probe scan: per-theta score with the engine's s_cut,
and the support distribution along s for selected thetas.  usage: dbg_riser.py COMPILED RECIPE SCAN LABEL [--hyp K]"""
import hashlib, math, sys
from pathlib import Path
import numpy as np
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
args = sys.argv[1:]
hk = 0
if "--hyp" in args:
    i = args.index("--hyp"); hk = int(args[i + 1]); del args[i:i + 2]
cdir, recipe, scan, label = Path(args[0]), Path(args[1]), Path(args[2]), args[3]
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
for h in proper[cfg.articulated_hypotheses:]:
    if any(E._matching_symmetry(h.fit.T @ np.linalg.inv(g.fit.T), part, cfg, rms_max=cfg.near_twin_rms_mm) for g in scored):
        scored.append(h)
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
proper = sorted([h for h in proper if h.art is not None], key=lambda h: -h.total)
h = proper[hk]
full = apply(h.fit.T, pts)
vox_reg = apply(h.fit.T, vox.points); vox_nrm = vox.normals @ h.fit.T[:3, :3].T
j = next(b.index for b in part.bends if b.label == label)
bend = part.bends[j]
parent = h.art.panels[bend.parent]
frame = AR.hinge_frame(bend, parent, cfg)
reach = frame.band * cfg.hough_fine_reach
widx, wz, wu, wv = AR._band_points(full, frame, 2.0, radius=reach)
grand = AR._short_child(part, bend, reach)
gu = AR._grand_offset(part, bend, grand, parent, frame) if grand is not None else None
print(f"{label}: grand {None if grand is None else part.bends[grand].label} grand_u {gu} band {frame.band:.2f} reach {reach:.2f} rho_iml {frame.rho_iml:.3f} delta {frame.delta:.3f}")
def cut(ths):
    if grand is None: return None
    lims = [AR._short_child_limit(part, bend, grand, parent, frame, float(t), False, cfg) for t in ths]
    return np.asarray([x[0] for x in lims]), lims[0][1], lims[0][2]
ths = np.radians(np.arange(-5, 61, 1.0))
sc, rh, ni, no = AR._hough(wz, wu, wv, frame, ths, cfg, reach, continuity=grand is not None, grand_u=gu, s_cut=cut(ths))
cuts = cut(ths)
for k in range(len(ths)):
    c, hw = AR.rho_window(frame, float(ths[k]), cfg.hinge_rho_slack_mm, cfg)
    print(f"  theta {math.degrees(ths[k]):5.1f} score {sc[k]:8.0f} rho {rh[k]:+.3f} off {rh[k]-c:+.3f} n_i {ni[k]:6.0f} n_o {no[k]:6.0f} s_cut {cuts[0][k] if cuts else float('nan'):.2f}")
# support distribution along s for some thetas (their best rho)
vn = None
for tdeg in (13.0, 15.0, 25.0, 31.0, 35.0, 40.0, 45.0, 50.0):
    k = int(np.argmin(np.abs(np.degrees(ths) - tdeg)))
    th = ths[k]; c_, s_ = math.cos(th), math.sin(th)
    sv = wu * c_ + wv * s_; rho = -wu * s_ + wv * c_
    on_i = np.abs(rho - rh[k]) < cfg.hough_window_mm; on_o = np.abs(rho - (rh[k] + frame.delta)) < cfg.hough_window_mm
    sel = (sv > 0.3) & (on_i | on_o)
    if cuts is not None:
        sel &= ~((wz > cuts[1]) & (wz < cuts[2]) & (sv >= cuts[0][k]))
    hist_i, edges = np.histogram(sv[sel & on_i], bins=np.arange(0, 8.01, 0.5))
    hist_o, _ = np.histogram(sv[sel & on_o], bins=edges)
    # normals of the supporting voxels vs the line normal
    n_line = frame.line_normal(th)
    vi = vox.inverse[widx[sel]]
    cosn = np.abs(vox_nrm[vi] @ n_line)
    print(f"  theta {tdeg:4.1f}: s-hist inner {hist_i.tolist()} outer {hist_o.tolist()} | normal within 10deg {np.mean(cosn > math.cos(math.radians(10))):.2f} within 20deg {np.mean(cosn > math.cos(math.radians(20))):.2f}")
if "--png" in sys.argv:
    out = sys.argv[sys.argv.index("--png") + 1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (zlo, zhi) in zip(axs, ((frame.z0, frame.z0 + (frame.z1 - frame.z0) / 3), (frame.z0 + (frame.z1 - frame.z0) / 3, frame.z0 + 2 * (frame.z1 - frame.z0) / 3), (frame.z0 + 2 * (frame.z1 - frame.z0) / 3, frame.z1))):
        sel = (wz > zlo) & (wz < zhi) & (wu > -3) & (wu < 7) & (wv > -3) & (wv < 4)
        sub = np.flatnonzero(sel)[:: max(1, int(sel.sum()) // 40000)]
        ax.scatter(wu[sub], wv[sub], s=0.3, c="k")
        for tdeg, col in ((13.0, "r"), (35.0, "g"), (45.0, "b")):
            k = int(np.argmin(np.abs(np.degrees(ths) - tdeg)))
            th = ths[k]
            d = np.array([math.cos(th), math.sin(th)]); n = np.array([-math.sin(th), math.cos(th)])
            for off, st in ((rh[k], "-"), (rh[k] + frame.delta, "--")):
                p0 = n * off
                ax.plot([p0[0] - 2 * d[0], p0[0] + 8 * d[0]], [p0[1] - 2 * d[1], p0[1] + 8 * d[1]], col + st, lw=0.8, label=f"{tdeg}" if st == "-" else None)
        ax.set_xlim(-3, 7); ax.set_ylim(-3, 4); ax.set_aspect("equal"); ax.grid(True, lw=0.3); ax.legend()
        ax.set_title(f"{label} z {zlo:.0f}..{zhi:.0f}")
    fig.tight_layout(); fig.savefig(out, dpi=80)
    print("wrote", out)
