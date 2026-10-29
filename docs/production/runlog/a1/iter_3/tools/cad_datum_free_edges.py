"""CAD-only: per compiled A1 part, the datum's free edges (lengths), holes, and stage material just outside each edge."""
import sys
from pathlib import Path
import numpy as np
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel
C = Path(sys.argv[1])
for d in sorted(C.iterdir()):
    f = d / "compiled_part.json"
    if not f.exists():
        continue
    cp = CompiledPartV1.model_validate_json(f.read_text())
    if cp.status != "COMPILED":
        continue
    part = PartView(cp, d)
    m = DatumModel.from_part(part)
    segs = m.free_segments()
    P = part.cloud("S0").points
    h = (P - m.p0) @ m.n0
    sl = (h > -m.t - 0.5) & (h < 0.5)
    uv = np.stack([(P[sl] - m.p0) @ m.f1, (P[sl] - m.p0) @ m.f2], axis=1)
    rows = []
    for a, b, n in segs:
        L = float(np.linalg.norm(b - a)); e = (b - a) / L
        rel = uv - a; al, of = rel @ e, rel @ n
        sel = (al > 0) & (al < L) & (of > 0.3) & (of < 6.0)
        rows.append(f"{L:.1f}{'*' if sel.any() else ''}")
    print(f"{d.name}: datum area {m.area:.0f} mm2, holes {len(m.holes)}, free edges (mm; * = S0 stage material within 6 mm outside): {' '.join(rows) or '-'}")
