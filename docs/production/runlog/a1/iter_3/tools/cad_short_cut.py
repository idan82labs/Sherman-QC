"""CAD-only: for every short-child bend, where the riser's line support would stop (prototype s_cut) at flat and at
the nominal sweep, vs the line's start s_min and the flat-continuity span.
usage: cad_short_cut.py COMPILED_DIR ..."""
import math
import sys
from pathlib import Path

import numpy as np

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView

cfg = InspectConfig()
for d in map(Path, sys.argv[1:]):
    f = d / "compiled_part.json"
    if not f.exists():
        continue
    try:
        part = PartView(CompiledPartV1.model_validate_json(f.read_text()), d)
    except ValueError:
        continue
    A = {part.datum: np.eye(4)}
    for j in part.bfs:
        b = part.bends[j]
        A[b.child] = A[b.parent] @ AR.fold_transform(b, 0.0)
    for b in part.bends:
        band = b.ri + 2.0 * b.t + cfg.hinge_band_extra_mm
        g = AR._short_child(part, b, band * cfg.hough_fine_reach)
        if g is None:
            continue
        par = AR.PanelFit(b.parent, "FITTED", A[b.parent])
        fr = AR.hinge_frame(b, par, cfg)
        r_out = max(abs(fr.rho_iml), abs(fr.rho_iml + fr.delta))
        s_min = max(0.3, math.sqrt(2.0 * r_out * cfg.hough_window_mm) + cfg.hough_arc_clearance_mm)
        s0 = AR._short_child_limit(part, b, g, par, fr, 0.0, True, cfg)
        s1 = AR._short_child_limit(part, b, g, par, fr, b.sweep, False, cfg)
        gu = AR._grand_offset(part, b, g, par, fr)
        print(f"{d.name} {b.label}->{part.bends[g].label} sweep {math.degrees(b.sweep):5.1f} grand_u {gu if gu is None else round(gu, 2)} "
              f"s_min {s_min:.2f} cut@0 {s0[0]:.2f} cut@sweep {s1[0]:.2f} (flat continuity needs s up to {s_min + cfg.flat_continuity_mm:.2f}) "
              f"z {s0[1]:.1f}..{s0[2]:.1f} bend z {b.z0:.1f}..{b.z1:.1f}")
