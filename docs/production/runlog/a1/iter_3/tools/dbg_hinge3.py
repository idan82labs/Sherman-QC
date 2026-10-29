"""Debug the hinge search of one bend on a stage-probe scan (in-process E3 registration + articulated fit).

usage: dbg_hinge3.py COMPILED_DIR RECIPE SCAN LABEL [OUT.png] [--hyp K]
Prints every articulated hypothesis (total, datum score, explained, states), then for LABEL on the best
(or the K-th) hypothesis: the coarse Hough profile, the rho window, the chosen line and its detail.
"""
import hashlib
import math
import sys
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

cdir, recipe, scan, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
outpng = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else None
hk = int(sys.argv[sys.argv.index("--hyp") + 1]) if "--hyp" in sys.argv else 0
cfg = InspectConfig()
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
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None, median_spacing(reg_src), mirrored=False)
pose_p0 = scan.parent / "pose.npy"
Tt0 = np.linalg.inv(np.load(pose_p0)) if pose_p0.exists() else None
for i, hp in enumerate(proper):
    msg = f"proper{i}: datum {hp.fit.score:.4f}"
    if Tt0 is not None:
        rel0 = hp.fit.T @ np.linalg.inv(Tt0)
        msg += f" vs truth rot {math.degrees(math.acos(max(-1, min(1, (np.trace(rel0[:3, :3]) - 1) / 2)))):.2f} t {np.linalg.norm(rel0[:3, 3]):.2f}"
    print(msg)
scored = list(proper[:cfg.articulated_hypotheses])
for h in proper[cfg.articulated_hypotheses:]:
    if any(E._matching_symmetry(h.fit.T @ np.linalg.inv(g.fit.T), part, cfg, rms_max=cfg.near_twin_rms_mm) for g in scored):
        scored.append(h)
for h in scored:
    E._articulate(h, pts, vox, part, model, cfg, budget)
scored.sort(key=lambda h: -h.total)
T0 = scored[0].fit.T
for i, h in enumerate(scored):
    rel = h.fit.T @ np.linalg.inv(T0)
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(rel[:3, :3]) - 1) / 2))))
    print(f"hyp{i}: total {h.total:.4f} datum {h.fit.score:.4f} expl {h.explained:.4f}/{h.explained_tight:.4f} "
          f"rel_rot {ang:.1f} deg rel_t {np.linalg.norm(rel[:3, 3]):.1f}  states "
          + " ".join(f"{part.bends[j].label}:{d.state[:3]}" for j, d in enumerate(h.decodes)))
h = scored[hk]
j = next(b.index for b in part.bends if b.label == label)
bend = part.bends[j]
parent = h.art.panels[bend.parent]
bf = h.art.bends[j]
print(f"\n{label}: parent {part.panels[bend.parent].panel_id} {parent.state}; child {part.panels[bend.child].panel_id} "
      f"{h.art.panels[bend.child].state}; theta {bf.theta_deg} rho {bf.rho_iml} score {bf.score} detail {bf.detail}")
frame = AR.hinge_frame(bend, parent, cfg)
sweep = math.degrees(bend.sweep)
print(f"rho_iml {frame.rho_iml:.3f} delta {frame.delta:.3f} band {frame.band:.2f} sweep {sweep}")
grand = AR._short_child(part, bend, frame.band * cfg.hough_fine_reach)
print("short child grand:", None if grand is None else part.bends[grand].label)
for th in (0, 5, 10, 20, 30, sweep, sweep + 10):
    c, hw = AR.rho_window(frame, math.radians(th), cfg.hinge_rho_slack_mm, cfg)
    print(f"  window theta {th:6.1f}: centre {c:.3f} half {hw:.3f}")
full = apply(h.fit.T, pts)
reach = frame.band * cfg.hough_fine_reach
widx, wz, wu, wv = AR._band_points(full, frame, 2.0, radius=reach)
inb = np.hypot(wu, wv) < frame.band
th_c = np.radians(np.arange(cfg.theta_min_deg, sweep + cfg.theta_overbend_deg + 1e-9, cfg.theta_coarse_step_deg))
sc, rh, ni, no = AR._hough(wz[inb], wu[inb], wv[inb], frame, th_c, cfg, frame.band, continuity=grand is not None)
order = np.argsort(-sc)[:10]
for k in order:
    c, hw = AR.rho_window(frame, float(th_c[k]), cfg.hinge_rho_slack_mm, cfg)
    print(f"  coarse theta {math.degrees(th_c[k]):6.1f} score {sc[k]:8.1f} rho {rh[k]:.3f} (centre {c:.3f}, off {rh[k]-c:+.3f}) n_i {ni[k]:.0f} n_o {no[k]:.0f}")
if outpng:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    sel = (np.abs(wu) < reach) & (np.abs(wv) < reach)
    sub = np.flatnonzero(sel)[:: max(1, int(sel.sum()) // 60000)]
    ax.scatter(wu[sub], wv[sub], s=0.2, c="k")
    for th, col in ((0.0, "g"), (math.radians(bf.theta_deg or 0.0), "r")):
        c, hw = AR.rho_window(frame, th, cfg.hinge_rho_slack_mm, cfg)
        d = np.array([math.cos(th), math.sin(th)])
        n = np.array([-math.sin(th), math.cos(th)])
        for off, st in ((c, "-"), (c - hw, ":"), (c + hw, ":"), (c + frame.delta, "--")):
            p0 = n * off
            ax.plot([p0[0], p0[0] + reach * d[0]], [p0[1], p0[1] + reach * d[1]], col + st, lw=0.8)
    if bf.rho_iml is not None:
        th = math.radians(bf.theta_deg)
        d = np.array([math.cos(th), math.sin(th)])
        n = np.array([-math.sin(th), math.cos(th)])
        p0 = n * bf.rho_iml
        ax.plot([p0[0], p0[0] + reach * d[0]], [p0[1], p0[1] + reach * d[1]], "m-", lw=1.5)
    ax.set_xlim(-reach * 0.5, reach)
    ax.set_ylim(-reach * 0.75, reach * 0.75)
    ax.set_aspect("equal")
    ax.set_title(f"{label} hinge band (u, v); green: flat window, red: chosen-theta window, magenta: chosen line")
    fig.savefig(outpng, dpi=90)
    print("wrote", outpng)
# true pose check (stage probes write pose.npy next to scan.ply: raw = pose @ cad)
pose_p = scan.parent / "pose.npy"
if pose_p.exists():
    Ttrue = np.linalg.inv(np.load(pose_p))
    for i, hh in enumerate(scored):
        rel = hh.fit.T @ np.linalg.inv(Ttrue)
        ang = math.degrees(math.acos(max(-1, min(1, (np.trace(rel[:3, :3]) - 1) / 2))))
        print(f"hyp{i} vs truth: rot {ang:.3f} deg, t {np.linalg.norm(rel[:3, 3]):.3f} mm")
# growth diagnostics for LABEL on the chosen hypothesis
if "--grow" in sys.argv and bf.found:
    from domains.production.engine.inspect.cloud import components_of
    vox_reg = apply(h.fit.T, vox.points)
    vox_nrm = vox.normals @ h.fit.T[:3, :3].T
    owner = h.art.owner
    th = math.radians(bf.theta_deg)
    c_, s_ = math.cos(th), math.sin(th)
    bidx, zc, uc, vc = AR._band_points(full, frame, 2.0)
    sv = uc * c_ + vc * s_
    rho = -uc * s_ + vc * c_
    on = (sv > 0.3) & ((np.abs(rho - bf.rho_iml) < cfg.hough_window_mm) | (np.abs(rho - (bf.rho_iml + frame.delta)) < cfg.hough_window_mm))
    seeds = np.unique(vox.inverse[bidx[on]])
    print("seeds", len(seeds), "owned by parent", int((owner[seeds] == bend.parent).sum()), "owned by others",
          int(((owner[seeds] >= 0) & (owner[seeds] != bend.parent)).sum()), "free", int((owner[seeds] < 0).sum()))
    sv_seed = ((vox_reg[seeds] - frame.o) @ frame.line_dir(th))
    print("seed sv range", float(sv_seed.min()) if len(sv_seed) else None, float(sv_seed.max()) if len(sv_seed) else None)
    par = h.art.panels[bend.parent]
    pv = np.flatnonzero(owner == bend.parent)
    psv = (vox_reg[pv] - frame.o) @ frame.line_dir(0.0)
    print("parent owned voxels", len(pv), "their extent along the child direction (max sv)", float(psv.max()) if len(pv) else None)
    t = float(par.t)
    n_line = frame.line_normal(th)
    d_mid = float(n_line @ frame.o) + bf.rho_iml + 0.5 * frame.delta
    dist = np.abs(vox_reg @ n_line - d_mid)
    okn = np.abs(vox_nrm @ n_line) > math.cos(math.radians(cfg.grow_normal_tol_deg))
    mask = (dist < 0.5 * t + cfg.grow_plane_tol_mm) & okn & (owner != bend.parent)
    print("mask voxels", int(mask.sum()), "seeds in mask", int(mask[seeds].sum()), "seeds failing normal", int((~okn[seeds]).sum()), "seeds failing dist", int((dist[seeds] >= 0.5 * t + cfg.grow_plane_tol_mm).sum()))
    flat = bool(h.art.bends[j].flat)
    if flat:
        mask &= ((vox_reg - frame.o) @ frame.line_dir(th)) > 0.0
    reg = components_of(vox.graph, mask, seeds)
    print("flat", flat, "region", len(reg))
# panel-plane check vs truth for the child of LABEL (truth pose; the probe's stage S_k panel transform)
if "--panel" in sys.argv and Tt0 is not None:
    import json as _json
    k_stage = int(sys.argv[sys.argv.index("--panel") + 1])
    cpj = _json.loads((cdir / "compiled_part.json").read_text())
    st = cpj["stages"][k_stage]
    for pid_idx in (bend.parent, bend.child):
        pv = part.panels[pid_idx]
        Tk = np.asarray(st["panel_transforms"][pv.panel_id], dtype=np.float64)
        # stage pose of the panel in the CAD frame; the fitted pose maps CAD -> registered (~ truth)
        n_true = Tk[:3, :3] @ pv.n0
        p_true = Tk[:3, :3] @ pv.p0 + Tk[:3, 3]
        pf = h.art.panels[pid_idx]
        if pf.n0 is None:
            print(pv.panel_id, "no fit"); continue
        # registered frame = CAD frame when the registration is right (hyp vs truth printed above)
        ang = math.degrees(math.acos(max(-1, min(1, float(pf.n0 @ n_true)))))
        off = float(pf.n0 @ p_true - pf.d0)
        print(f"{pv.panel_id}: fitted skin0 vs truth: angle {ang:.3f} deg, offset of true skin0 point {off:+.3f} mm; skins_seen {pf.skins_seen} t {pf.t}")
# --profile LO HI: per-theta best line in [LO, HI] deg, narrow and wide windows, without and with the grandchild bound
if "--profile" in sys.argv:
    from dataclasses import replace as _replace
    lo_d = float(sys.argv[sys.argv.index("--profile") + 1]); hi_d = float(sys.argv[sys.argv.index("--profile") + 2])
    ths = np.radians(np.arange(lo_d, hi_d + 1e-9, 1.0))
    gu = AR._grand_offset(part, bend, grand, parent, frame) if grand is not None else None
    print("grand_u", gu)
    wide = _replace(cfg, hinge_rho_slack_mm=cfg.hinge_rho_slack_wide_mm)
    for name, cf, smax, gg in (("narrow/band", cfg, frame.band, None), ("narrow/band bounded", cfg, frame.band, gu),
                               ("narrow/reach", cfg, reach, None), ("wide/reach", wide, reach, None), ("wide/reach bounded", wide, reach, gu)):
        sel = slice(None) if smax == reach else inb
        sc_, rh_, ni_, no_ = AR._hough(wz[sel], wu[sel], wv[sel], frame, ths, cf, smax, continuity=grand is not None, grand_u=gg)
        print(" ", name)
        for k in range(len(ths)):
            c, hw = AR.rho_window(frame, float(ths[k]), cf.hinge_rho_slack_mm, cf)
            print(f"    theta {math.degrees(ths[k]):5.1f} score {sc_[k]:8.1f} off {rh_[k]-c:+.3f} (half {hw:.2f}) n_i {ni_[k]:.0f} n_o {no_[k]:.0f}")
