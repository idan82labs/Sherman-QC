"""native decision diffs base (survey e3-1.12.0) vs candidate. usage: natdiff.py CAND_RUNS [variant]"""
import json, sys
from pathlib import Path
B = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3stab/runs"); C = Path(sys.argv[1])
v = sys.argv[2] if len(sys.argv) > 2 else "native"
def L(p):
    if not p.exists(): return None
    r = json.loads(p.read_text()); o = {}
    for b in r["bends"]:
        for c in b["characteristics"]:
            o[c["characteristic_id"]] = (b["label"], c["kind"], c.get("side"), c["decision"], c.get("actual"), c.get("uncertainty"), (c.get("evidence") or {}).get("notes") or [])
    return o, r["summary"]["state"]
n = nd = 0
for d in sorted(C.iterdir()):
    cp = d / v / "engine/result.json"
    if not cp.exists(): continue
    job = json.loads((d / "job.json").read_text())
    b = L(B / d.name / v / "engine/result.json"); c = L(cp)
    if not b: continue
    n += 1
    diffs = [(k, b[0][k], c[0].get(k)) for k in b[0] if c[0].get(k) is None or b[0][k][3] != c[0][k][3] or (b[0][k][4] is not None and c[0][k][4] is not None and abs(b[0][k][4] - c[0][k][4]) > 1e-4)]
    if diffs or b[1] != c[1]:
        nd += 1
        print(job["corpus"], job["id"], b[1], "->", c[1])
        for k, x, y in diffs:
            print(f"   {x[0]:>4} {x[1]:<18} {x[2] or '':2} {x[3]:<7} {x[4]} U{x[5]}  ->  {y[3] if y else None} {y[4] if y else None} U{y[5] if y else None} {[q for q in (y[6] if y else []) if 'BRANCH' in q or 'RESAMPLE' in q]}")
print("scans", n, "with diffs", nd)
