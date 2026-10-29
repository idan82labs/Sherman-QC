"""Trace _grow_child / _refine_skin_levels for one bend's child on a probe scan (monkeypatched, in-process)."""
import hashlib, math, sys
from pathlib import Path
import numpy as np
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import skin_peaks
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

cdir, recipe, scan, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
cfg = InspectConfig()
inputs = load_inputs(cdir, recipe, scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
target = next(b for b in part.bends if b.label == label)
orig_refine = AR._refine_skin_levels
orig_grow = AR._grow_child

def traced_refine(pts, nrm, d_mid, t, sig, bend, seen, cfg_):
    if bend.index == target.index:
        dd = pts @ nrm - d_mid
        dd2 = dd[np.abs(dd) < 0.5 * t + cfg_.grow_plane_tol_mm]
        h, e = np.histogram(dd2, bins=np.arange(-1.2, 1.21, 0.1))
        print("refine: d_mid", round(d_mid, 3), "sig", sig, "seen_in", seen, "n", len(dd2))
        print("   hist", " ".join(f"{x:.1f}:{c}" for x, c in zip(e[:-1], h)))
        print("   skin_peaks", skin_peaks(dd2, t) if len(dd2) > 30 else None)
    out = orig_refine(pts, nrm, d_mid, t, sig, bend, seen, cfg_)
    if bend.index == target.index:
        print("   ->", out)
    return out

def traced_grow(*args, **kw):
    bend = args[7]
    res = orig_grow(*args, **kw)
    if bend.index == target.index:
        print("grow result:", res.state, "skins_seen", res.skins_seen, "region", len(res.region), "voxels", len(res.voxels))
    return res

AR._refine_skin_levels = traced_refine
AR._grow_child = traced_grow
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
pose_p = scan.parent / "pose.npy"
Tt = np.linalg.inv(np.load(pose_p))
best = min(proper, key=lambda h: np.linalg.norm((h.fit.T @ np.linalg.inv(Tt))[:3, 3]))
print("articulating the hypothesis nearest the truth; datum", round(best.fit.score, 4))
E._articulate(best, pts, vox, part, model, cfg, budget)
# replay the growth loop of the target bend with per-iteration prints
from domains.production.engine.inspect.geom import apply, robust_plane, unit
from domains.production.engine.inspect.cloud import components_of
art = best.art
j = target.index
bf = art.bends[j]
parent = art.panels[target.parent]
frame = bf.frame
theta = math.radians(bf.theta_deg)
rho_i = bf.rho_iml
full_reg = apply(best.fit.T, pts)
vox_reg = apply(best.fit.T, vox.points)
vox_nrm = vox.normals @ best.fit.T[:3, :3].T
reach = frame.band * cfg.hough_fine_reach
widx, wz, wu, wv = AR._band_points(full_reg, frame, 2.0, radius=reach)
inband = np.hypot(wu, wv) < frame.band
bidx, zc, uc, vc = widx[inband], wz[inband], wu[inband], wv[inband]
c, s = math.cos(theta), math.sin(theta)
sv = uc * c + vc * s
rho = -uc * s + vc * c
line = (sv > 0.3) & ((np.abs(rho - rho_i) < cfg.hough_window_mm) | (np.abs(rho - (rho_i + frame.delta)) < cfg.hough_window_mm))
seeds = np.unique(vox.inverse[bidx[line]])
# owner as it was before this child was grown: remove this child's own voxels
owner = art.owner.copy()
owner[owner == target.child] = -1
t = float(parent.t)
n_line = frame.line_normal(theta)
sig = 1.0 if frame.delta > 0 else -1.0
d_mid = float(n_line @ frame.o) + rho_i + 0.5 * frame.delta
print("hough: theta", round(bf.theta_deg, 2), "rho_i", round(rho_i, 3), "delta", round(frame.delta, 3), "d_mid0", round(d_mid, 3), "sig", sig)
nrm = n_line.copy()
cos_n = math.cos(math.radians(cfg.grow_normal_tol_deg))
last = -1
for it in range(cfg.grow_iterations):
    dist = np.abs(vox_reg @ nrm - d_mid)
    okn = np.abs(vox_nrm @ nrm) > cos_n
    mask = (dist < 0.5 * t + cfg.grow_plane_tol_mm) & okn & (owner != parent.index)
    region = components_of(vox.graph, mask, seeds)
    if len(region) < 3 or len(region) == last:
        print("stop", len(region)); break
    last = len(region)
    sel = np.isin(vox.inverse, region)
    P = full_reg[sel]
    dd = P @ nrm - d_mid
    ins = np.abs(dd) < 0.5 * t + cfg.grow_plane_tol_mm
    P, dd = P[ins], dd[ins]
    up, lo = P[dd > 0], P[dd <= 0]
    h, e = np.histogram(dd, bins=np.arange(-1.2, 1.21, 0.2))
    print(f"it{it}: region {len(region)} up {len(up)} lo {len(lo)} med_up {np.median(dd[dd>0]) if len(up) else None} med_lo {np.median(dd[dd<=0]) if len(lo) else None}")
    print("    hist", " ".join(f"{x:.1f}:{cc}" for x, cc in zip(e[:-1], h)))
    planes = []
    for grp in (up, lo):
        if len(grp) >= 30:
            pl = robust_plane(grp, iterations=3)
            planes.append((pl if float(pl.n @ nrm) > 0 else pl.flipped(), len(grp)))
    nrm_new = unit(sum(pl.n * cnt for pl, cnt in planes))
    print("    normal change deg", round(math.degrees(math.acos(min(1, float(nrm_new @ nrm)))), 3))
    nrm = nrm_new
    if len(up) >= 30 and len(lo) >= 30:
        d_mid = 0.5 * (float(np.median(up @ nrm)) + float(np.median(lo @ nrm)))
    elif len(up) >= 30:
        d_mid = float(np.median(up @ nrm)) - 0.5 * t
    else:
        d_mid = float(np.median(lo @ nrm)) + 0.5 * t
    print("    d_mid ->", round(d_mid, 3))
# truth
Tk = None
