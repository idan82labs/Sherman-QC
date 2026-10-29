"""char-level diff of two result.json trees: rescmp.py DIR_A DIR_B (DIR/<case>/engine/result.json)"""
import json, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2]); tol = 0.05
def L(p):
    r = json.loads(p.read_text()); o = {}
    for b in r["bends"]:
        for c in b["characteristics"]:
            o[c["characteristic_id"]] = (b["label"], c["kind"], c.get("side"), c["decision"], c.get("actual"), c.get("uncertainty"), [n for n in (c.get("evidence") or {}).get("notes") or [] if "WITHHELD" in n])
    return o, r["summary"]["state"], r["engine"]["engine_version"]
nc = nd = 0
for d in sorted(A.iterdir()):
    pa, pb = d / "engine/result.json", B / d.name / "engine/result.json"
    if not pa.exists() or not pb.exists(): print("missing", d.name, pa.exists(), pb.exists()); continue
    a, b = L(pa), L(pb); nc += 1
    diffs = [k for k in a[0] if a[0][k][3] != b[0].get(k, (0,0,0,None))[3] or (a[0][k][4] is not None and b[0].get(k, [None]*5)[4] is not None and abs(a[0][k][4] - b[0][k][4]) > tol)]
    if diffs or a[1] != b[1]:
        nd += 1; print(d.name, a[2], a[1], "->", b[2], b[1])
        for k in diffs:
            x, y = a[0][k], b[0].get(k)
            print(f"   {x[0]:>4} {x[1]:<18} {x[2] or '':2} {x[3]:<7} {x[4]} U{x[5]} -> {y[3]} {y[4]} U{y[5]} {y[6]}")
print("cases", nc, "changed", nd)
