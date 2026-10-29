"""List panels, their path bends' axes and their free edges/holes relative to those axes (diagnostic)."""
import math, sys
from pathlib import Path
import numpy as np
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel

cdir = Path(sys.argv[1])
comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
part = PartView(comp, cdir)
model = DatumModel.from_part(part)
print("datum", part.panels[part.datum].panel_id, "n0", np.round(model.n0, 3), "t", part.t)
for p in part.panels:
    path = part.path_bends(p.index)
    axes = [part.bends[j].axis_dir for j in path]
    ax2 = [np.round([a @ model.f1, a @ model.f2, a @ model.n0], 3) for a in axes]
    common = None
    if axes:
        a0 = axes[0]
        if all(abs(abs(float(a @ a0)) - 1) < 1e-3 for a in axes):
            common = a0
    print(f"panel {p.panel_id} depth {p.depth} area {p.area:.0f} path {[part.bends[j].label for j in path]} axes2d {ax2}"
          f" common_axis {'yes' if common is not None else 'no'} holes {len(p.model.holes_mm)}")
    for fe in p.model.free_edges:
        L = np.asarray(fe.polyline_mm)
        for a, b in zip(L[:-1], L[1:]):
            d = b - a
            ln = float(np.linalg.norm(d))
            if ln < 3:
                continue
            e = d / ln
            # in-panel normal: perpendicular to e within the panel plane
            n_in = np.cross(p.n0, e)
            info = ""
            if common is not None:
                c = float(abs(n_in @ common))
                info = f" |n.axis|={c:.3f}"
            print(f"     edge {fe.edge_id} usable={fe.registration_usable} len {ln:.1f} mid {np.round(0.5*(a+b),1)} dir {np.round(e,3)}{info}")
for b in part.bends:
    print(f"bend {b.label} seq {b.seq} parent {part.panels[b.parent].panel_id} child {part.panels[b.child].panel_id}"
          f" axis {np.round(b.axis_dir,3)} len {b.length:.1f} sweep {math.degrees(b.sweep):.1f} ri {b.ri} sides {{A: {part.panels[b.side_panel['A']].panel_id}, B: {part.panels[b.side_panel['B']].panel_id}}}"
          f" outA {np.round(b.side_out['A'],3)}")
