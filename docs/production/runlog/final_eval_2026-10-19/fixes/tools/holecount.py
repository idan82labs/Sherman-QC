"""POST-HOC: hole_mismatch of the engine's constraint routine on the registered scan (CAD frame)."""
import sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel, _inplane_constraints
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.eval.plyio import read_ply_vertices
ed, cdir = Path(sys.argv[1]), Path(sys.argv[2])
cp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
part = PartView(cp, cdir); model = DatumModel.from_part(part); cfg = InspectConfig()
P = read_ply_vertices(ed / "registered_scan.ply").vertices
h = (P - model.p0) @ model.n0
cand = (h < 0.5) & (h > -model.t - 0.5)
uv = model.to2d(P[cand])
margin = cfg.wall_skin_margin_fraction * model.t
interior = (h[cand] < -margin) & (h[cand] > -model.t + margin)
segs = list(model.free_segments(min_length=2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm))
con = _inplane_constraints(uv, model, segs, cfg, 0.0, float(sys.argv[3]) if len(sys.argv) > 3 else 0.5, model.polygon.mean(axis=0), 10.0, wide=False, interior=interior)
print("holes", len(model.holes), "hole_mismatch", con["hole_mismatch"], "holes_seen", con["holes_seen"])
hp_ = (P - model.p0) @ model.n0
pr = (hp_ >= 1.0) | (hp_ <= -model.t - 1.0)
con = _inplane_constraints(uv, model, segs, cfg, 0.0, 0.5, model.polygon.mean(axis=0), 10.0, wide=False, interior=interior, proud=model.to2d(P[pr]))
print("with proud: hole_mismatch", con["hole_mismatch"], "hardware", con["holes_hardware"])
