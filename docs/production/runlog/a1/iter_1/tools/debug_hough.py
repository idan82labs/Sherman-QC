"""Debug the hinge Hough of one bend: replicate the engine up to the articulated fit and print the theta profile."""
import json
import math
import sys
from pathlib import Path

import numpy as np

from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, gross_inliers, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import apply
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

ROOT = Path(sys.argv[1])
case = sys.argv[2]
label = sys.argv[3]
cdir_case = ROOT / "physical/E3" / case
res = json.loads((cdir_case / "engine/result.json").read_text())
cdir = ROOT / "compiled/dev" / res["sku"]
cfg = InspectConfig()
scanp = next(cdir_case.glob("input/scan.*"))
inputs = load_inputs(cdir, cdir_case / "input/recipe.json", scanp, res["scan"]["sha256"])
part = PartView(inputs.compiled, inputs.compiled_dir)
pts = read_scan(inputs.scan_path).points
pts = pts[gross_inliers(pts, cfg.extent_trim_percent, cfg.gross_outlier_extent_factor)]
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
h = proper[0]
E._articulate(h, pts, vox, part, model, cfg, budget)
art = h.art
j = next(i for i, b in enumerate(part.bends) if b.label == label)
bend = part.bends[j]
parent = art.panels[bend.parent]
print("bend", label, "parent", part.panels[bend.parent].panel_id, parent.state, "child", part.panels[bend.child].panel_id,
      "sweep", math.degrees(bend.sweep), "Ri", bend.ri, "t", bend.t, "found", art.bends[j].found, "theta", art.bends[j].theta_deg)
T = h.fit.T
full_reg = apply(T, pts)
frame = AR.hinge_frame(bend, parent, cfg)
print("frame band", frame.band, "rho_iml", frame.rho_iml, "delta", frame.delta)
reach = frame.band * cfg.hough_fine_reach
widx, wz, wu, wv = AR._band_points(full_reg, frame, 2.0, radius=reach)
inband = np.hypot(wu, wv) < frame.band
zc, uc, vc = wz[inband], wu[inband], wv[inband]
coarse = np.radians(np.arange(cfg.theta_min_deg, math.degrees(bend.sweep) + cfg.theta_overbend_deg + 1e-9, cfg.theta_coarse_step_deg))
sc, rh, ni, no = AR._hough(zc, uc, vc, frame, coarse, cfg, frame.band)
for th, s, r, a, b in zip(np.degrees(coarse), sc, rh, ni, no):
    if int(round(th)) % 5 == 0 or s == sc.max():
        print(f"  theta {th:6.1f} score {s:8.0f} rho {r:7.2f} n_i {a:7.0f} n_o {b:7.0f}")
# what does the scan look like in the section? histogram of points by angle about the hinge
ang = np.degrees(np.arctan2(vc, uc))
r = np.hypot(uc, vc)
for lo in range(-30, 140, 10):
    m = (ang >= lo) & (ang < lo + 10)
    print(f"   polar {lo:4d}..{lo+10:4d} deg: {m.sum():7d} pts, r med {np.median(r[m]) if m.any() else float('nan'):.2f}")
# unrestricted rho search at a few thetas (no slack window)
for thd in (16.7, 80, 85, 88, 90, 92, 95):
    th = math.radians(thd)
    c, s = math.cos(th), math.sin(th)
    sv = wu * c + wv * s
    rho = -wu * s + wv * c
    sel = (sv > 2.0) & (sv < reach)
    hist, edges = np.histogram(rho[sel], bins=np.arange(-30, 30, 0.2))
    top = np.argsort(hist)[-3:][::-1]
    print(f"  theta {thd:5.1f}: top rho bins " + ", ".join(f"{edges[k]:.1f}:{hist[k]}" for k in top), f"(nominal iml {frame.rho_iml:.2f}, oml {frame.rho_iml + frame.delta:.2f})")
