"""CAD-only risk listing per A1 bend: sweep, length, nominal flange lengths, hinge band vs flange, nearest
parallel bend gap (joggles). No scan is read."""
import math, sys
from pathlib import Path
import numpy as np
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView

cfg = InspectConfig()
root = Path(sys.argv[1])
for cdir in sorted(root.iterdir()):
    if not (cdir / "compiled_part.json").exists():
        continue
    comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
    if comp.status != "COMPILED":
        continue
    part = PartView(comp, cdir)
    print(f"== {cdir.name} t {part.t} bends {part.n}")
    for b in part.bends:
        fl = {}
        for s in ("A", "B"):
            nom = part.nominals.get(f"{b.uid}/FLANGE_LENGTH/{s}")
            fl[s] = None if nom is None or nom.nominal is None else (round(float(nom.nominal), 1), nom.boundary_kind[:4] if nom.boundary_kind else None,
                                                                      nom.unavailable_reason)
            if nom is not None and nom.nominal is None:
                fl[s] = ("NA", nom.unavailable_reason)
        band = b.ri + 2 * b.t + cfg.hinge_band_extra_mm
        a0, a1 = b.locus("oml_virtual_sharp")
        gaps = []
        for o in part.bends:
            if o is b or abs(abs(float(o.axis_dir @ b.axis_dir)) - 1) > 1e-3:
                continue
            q0, q1 = o.locus("oml_virtual_sharp")
            m = 0.5 * (q0 + q1) - 0.5 * (a0 + a1)
            perp = m - (m @ b.axis_dir) * b.axis_dir
            gaps.append(float(np.linalg.norm(perp)))
        g = min(gaps) if gaps else None
        flags = []
        if math.degrees(b.sweep) < 35: flags.append("SMALL_SWEEP")
        if math.degrees(b.sweep) > 120: flags.append("LARGE_SWEEP")
        if g is not None and g < band + 3: flags.append(f"NEAR_PARALLEL_BEND:{g:.1f}")
        for s in ("A", "B"):
            v = fl[s]
            if isinstance(v, tuple) and isinstance(v[0], float) and v[0] < band:
                flags.append(f"SHORT_FLANGE_{s}:{v[0]}")
            if isinstance(v, tuple) and v[0] == "NA":
                flags.append(f"NO_NOMINAL_{s}:{v[1]}")
        if b.length < 15: flags.append(f"SHORT_BEND:{b.length:.1f}")
        print(f"  {b.label:>4} sweep {math.degrees(b.sweep):5.1f} len {b.length:6.1f} ri {b.ri} band {band:.1f} A {fl['A']} B {fl['B']} {' '.join(flags)}")
