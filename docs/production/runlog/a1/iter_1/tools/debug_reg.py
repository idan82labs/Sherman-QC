"""Re-run the registration stage of E3 on an iter-0 physical input and print the in-plane fit details."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, gross_inliers, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan

ROOT = Path(sys.argv[1])
case = sys.argv[2]
cdir_case = ROOT / "physical/E3" / case
res = json.loads((cdir_case / "engine/result.json").read_text())
sku = res["sku"]
cdir = ROOT / "compiled/dev" / sku
cfg = InspectConfig()
inputs = load_inputs(cdir, cdir_case / "input/recipe.json", cdir_case / "input/scan.ply", res["scan"]["sha256"])
part = PartView(inputs.compiled, inputs.compiled_dir)
scan = read_scan(inputs.scan_path)
pts = scan.points
keep = gross_inliers(pts, cfg.extent_trim_percent, cfg.gross_outlier_extent_factor)
pts = pts[keep]
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel)
slab = cfg.slab_mode(part.t, voxel)
vox.slab = slab
normal_voxel = max(voxel, cfg.thin_normal_voxel_factor * part.t) if slab else None
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=normal_voxel)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
reg_spacing = median_spacing(reg_src) if reg_src is not pts else spacing
budget = E._Budget(170)
logs = []
proper = E._register(reg_src, vox, part, model, cfg, budget, logs.append, reg_spacing, mirrored=False)
print("\n".join(logs))
print("datum", part.panels[part.datum].panel_id, "t", part.t, "bends", part.n, "holes", len(model.holes),
      "free segs", len(model.free_segments()), "free segs >= trim", len(model.free_segments(min_length=2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm)))
for a, b, n in model.free_segments():
    print("   free", np.round(a, 1), np.round(b, 1), "len", round(float(np.linalg.norm(b - a)), 1), "nout", np.round(n, 3))
for k, h in enumerate(proper[:4]):
    f = h.fit
    ev, evec = np.linalg.eigh(f.information)
    print(f"hyp {k}: score {f.score:.3f} degenerate {f.degenerate} pose_sigma {E._pose_sigma(f):.3f} edge_bins {f.edge_bins}/{f.edge_bins_expected}"
          f" holes {f.holes_seen}/{f.holes_total} mism {f.hole_mismatch} edge_rms {f.edge_rms:.3f} cov {f.datum_coverage:.3f}")
    print("    info evals", np.round(ev, 2), "evecs(cols: rotL,tx,ty)\n", np.round(evec, 3))
    print("    trans cov", np.round(f.translation_cov, 4), "rot sigma rad", f.rotation_sigma_rad, "L", f.length_scale)
    print("    detail", {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in f.detail.items()})
for j, b in enumerate(part.bends):
    a, e = b.locus("oml_virtual_sharp")
    d = (e - a) / np.linalg.norm(e - a)
    d2 = np.array([d @ model.f1, d @ model.f2])
    print(f"  bend {b.label} parent {part.panels[b.parent].panel_id} child {part.panels[b.child].panel_id} axis2d {np.round(d2, 3)} len {np.linalg.norm(e - a):.1f} sweep {math.degrees(b.sweep):.1f}")
