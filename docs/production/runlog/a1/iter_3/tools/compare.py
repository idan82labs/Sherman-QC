"""Compare per-characteristic results of two run roots (A = reference dir layout <root>/<case>/result.json or
<root>/physical/E3/<case>/engine/result.json)."""
import json, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2])
cases = sys.argv[3:]
def load(root, c):
    for p in (root / c / "result.json", root / "physical/E3" / c / "engine/result.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None
tot = 0
for c in cases:
    a, b = load(A, c), load(B, c)
    if a is None or b is None:
        print(c, "missing"); continue
    diffs = []
    ra = {(ch["characteristic_id"]): ch for r in a["bends"] for ch in r["characteristics"]}
    rb = {(ch["characteristic_id"]): ch for r in b["bends"] for ch in r["characteristics"]}
    lab = {r["bend_uid"]: r["label"] for r in a["bends"]}
    for k in sorted(set(ra) | set(rb)):
        x, y = ra.get(k), rb.get(k)
        if x is None or y is None:
            diffs.append(f"{k} only in one"); continue
        dx, dy = x.get("deviation"), y.get("deviation")
        ux, uy = x.get("uncertainty"), y.get("uncertainty")
        changed = x["state"] != y["state"] or x["decision"] != y["decision"] or \
            (dx is None) != (dy is None) or (dx is not None and abs(dx - dy) > 1e-4) or \
            (ux is not None and uy is not None and abs(ux - uy) > 1e-4)
        if changed:
            name = lab.get(k.split("/")[0], "?") + "/" + "/".join(k.split("/")[1:])
            diffs.append(f"{name}: {x['state'][:4]}/{x['decision'][:3]} {dx if dx is None else round(dx,3)}±{ux if ux is None else round(ux,3)}"
                         f" -> {y['state'][:4]}/{y['decision'][:3]} {dy if dy is None else round(dy,3)}±{uy if uy is None else round(uy,3)}")
    sa, sb = a["summary"].get("state"), b["summary"].get("state")
    print(f"== {c}: {len(diffs)} changed; summary {sa} -> {sb}")
    for d in diffs[:40]:
        print("    " + d)
    tot += len(diffs)
print("total changed", tot)
