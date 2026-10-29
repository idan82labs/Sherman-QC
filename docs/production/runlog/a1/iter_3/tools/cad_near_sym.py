"""CAD-only: compiled A1 parts with a near symmetry (a ROTATION candidate between E1's 0.3 mm threshold and 1.0 mm)."""
import json, sys
from pathlib import Path
C = Path(sys.argv[1])
for d in sorted(C.iterdir()):
    f = d / "compiled_part.json"
    if not f.exists():
        continue
    cp = json.loads(f.read_text())
    if cp.get("status") != "COMPILED":
        continue
    near, cert = set(), set()
    for st in cp["stages"]:
        sym = st["symmetry"]
        for c in sym["candidates"]:
            if c["kind"] != "ROTATION":
                continue
            key = st["stage_model_id"]
            if c["rms_mm"] <= sym["threshold_rms_mm"]:
                cert.add(key)
            elif c["rms_mm"] <= 1.0:
                near.add(key)
    print(d.name, "certified:", ",".join(sorted(cert, key=lambda s: int(s[1:]))) or "-", "| near (0.3-1.0 mm):", ",".join(sorted(near, key=lambda s: int(s[1:]))) or "-")
