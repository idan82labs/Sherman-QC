"""Diagnostic: E3 registration internals on one scan + an injected independent (ICP) pose.
usage: regdiag.py SCAN CDIR RECIPE ICP.npz"""
import sys, math, hashlib
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import load_inputs, PartView
from domains.production.engine.inspect.scan_io import read_scan
from domains.production.engine.inspect.cloud import median_spacing, adaptive_voxel, voxelize, estimate_normals, neighbour_graph
from domains.production.engine.inspect.registration import DatumModel, extract_planes, refine_datum, coarse_hypotheses
from domains.production.engine.inspect.geom import apply
scan, cdir, rec = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sha = hashlib.sha256(Path(scan).read_bytes()).hexdigest()
inp = load_inputs(cdir, rec, scan, sha)
cfg = InspectConfig()
part = PartView(inp.compiled, inp.compiled_dir)
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); slab = cfg.slab_mode(part.t, voxel); vox.slab = slab
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
print("datum panel", part.panels[part.datum].panel_id if hasattr(part.panels[part.datum], "panel_id") else part.datum,
      "area", round(float(part.panels[part.datum].area), 1), "n0", np.round(model.n0, 3))
segs = extract_planes(vox, cfg, min_area=max(30.0, cfg.datum_min_area_fraction * float(part.panels[part.datum].area)), thickness=part.t)
z = np.load(sys.argv[4]); R, t = z["R"], z["t"]
Ticp = np.eye(4); Ticp[:3, :3] = R.T; Ticp[:3, 3] = -R.T @ t  # raw -> cad
n_cad_datum = model.n0
for i, s in enumerate(segs):
    n_c = R.T @ s.n
    print(f"seg {i} area {s.area:8.0f} n_in_cad {np.round(n_c,3)} |dot datum n| {abs(n_c @ n_cad_datum):.3f} rms {s.rms:.2f}")
budget = E._Budget(300.0)
log = []
note = lambda m: log.append(m)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
reg_spacing = median_spacing(reg_src) if reg_src is not pts else spacing
proper = E._register(reg_src, vox, part, model, cfg, budget, note, reg_spacing, mirrored=False)
print("\n".join(log))
for h in proper[:6]:
    E._articulate(h, pts, vox, part, model, cfg, budget)
    turn = math.degrees(E._turn(h.fit.T, Ticp)); dpos = np.linalg.norm(h.fit.T[:3, 3] - Ticp[:3, 3])
    print(f"engine hyp score {h.fit.score:.3f} explained {h.explained:.3f} total {h.total:.3f}  vs ICP turn {turn:.1f} deg  dT {dpos:.1f}")
fit = refine_datum(reg_src, Ticp, model, cfg, reg_spacing)
h = E._Hyp(fit=fit, mirrored=False)
E._articulate(h, pts, vox, part, model, cfg, budget)
print(f"ICP-seeded: score {fit.score:.3f} explained {h.explained:.3f} total {h.total:.3f} turn from ICP {math.degrees(E._turn(fit.T, Ticp)):.2f}")
print("decodes", [getattr(d, "state", d) for d in h.decodes])
