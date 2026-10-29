import json, sys, math
from pathlib import Path
import numpy as np
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.registration import DatumModel
ROOT = Path(sys.argv[1])
for case in sys.argv[2:]:
    res = json.loads((ROOT / "physical/E3" / case / "engine/result.json").read_text())
    cdir = ROOT / "compiled/dev" / res["sku"]
    comp = CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes())
    part = PartView(comp, cdir)
    model = DatumModel.from_part(part)
    print(case, res["sku"])
    for b in part.bends:
        q0, q1 = b.locus("oml_virtual_sharp")
        dn = (q1 - q0) / np.linalg.norm(q1 - q0)
        out_a = b.side_out["A"]
        oa = out_a - (out_a @ dn) * dn
        oa /= np.linalg.norm(oa)
        m = np.array([oa @ model.f1, oa @ model.f2])
        mid2 = model.to2d(((q0 + q1) / 2)[None])[0]
        print(f"  {b.label} sideA={part.panels[b.side_panel['A']].panel_id} parent={part.panels[b.parent].panel_id} oa_inplane={np.round(m,3)} |m|={np.linalg.norm(m):.3f} oa.n0={oa@model.n0:.3f} mid2d={np.round(mid2,1)}")
