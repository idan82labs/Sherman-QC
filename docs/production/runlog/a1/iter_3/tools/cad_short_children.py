"""CAD-only: which bends of each compiled part have a short child (a parallel bend on the child within the fine reach).

usage: cad_short_children.py COMPILED_DIR [COMPILED_DIR ...]   (each holds compiled_part.json)
"""
import json
import sys
from pathlib import Path

from contracts.production.compiled_part import CompiledPartV1
from domains.production.engine.inspect.articulated import _short_child
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView

cfg = InspectConfig()
for d in map(Path, sys.argv[1:]):
    f = d / "compiled_part.json"
    if not f.exists():
        continue
    try:
        part = PartView(CompiledPartV1.model_validate_json(f.read_text()), d)
    except ValueError as e:
        print(d.name, "not compiled:", e); continue
    out = []
    for b in part.bends:
        band = b.ri + 2.0 * b.t + cfg.hinge_band_extra_mm
        g = _short_child(part, b, band * cfg.hough_fine_reach)
        if g is not None:
            out.append(f"{b.label}->{part.bends[g].label}")
    print(d.name, part.n, "bends; short-child bends:", out or "none")
