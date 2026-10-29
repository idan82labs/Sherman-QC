"""Print the SideProfile of a flange side vs the compiled nominal (no scan needed)."""
import sys
from pathlib import Path
import numpy as np
from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.part import PartView
from domains.production.engine.inspect.metrology import SideProfile
S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
sku = sys.argv[1]
cdir = S / "a1_iter0/phys/compiled/dev" / sku
part = PartView(CompiledPartV1.model_validate_json((cdir / "compiled_part.json").read_bytes()), cdir)
for spec in sys.argv[2:]:
    lab, side = spec.split(":")
    j = next(i for i, b in enumerate(part.bends) if b.label == lab)
    bend = part.bends[j]
    nom = part.nominals[f"{bend.uid}/FLANGE_LENGTH/{side}"]
    sp = SideProfile.build(part, j, side, nom.boundary_kind, nom.adjacent_bend_uid)
    print(lab, side, "nominal", nom.nominal, nom.boundary_kind, "len", bend.z1 - bend.z0, "profile", None if sp is None else sp.kind,
          "covers", None if sp is None or nom.nominal is None else sp.covers(float(nom.nominal)))
    if sp is not None:
        us = np.linspace(0, sp.length, 23)
        print("   ", [(round(float(u), 1), None if sp.at(float(u)) is None else round(sp.at(float(u)), 2)) for u in us])
