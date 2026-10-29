"""realization-level root-vs-base decision changes (all 7 realizations): realdiff.py MANIFEST BASE CAND [arm-filter]"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
man = json.load(open(sys.argv[1])); B, C = Path(sys.argv[2]), Path(sys.argv[3])
VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
out = []
for j in man["unique_jobs"]:
    for v in VARS:
        pb, pc = B / j["key"] / v / "engine/result.json", C / j["key"] / v / "engine/resample_check.json"
        if not pb.exists() or not pc.exists(): continue
        rb = json.loads(pb.read_text()); rc = json.loads(pc.read_text())
        if not rc.get("primary_raw"):
            continue
        prim = rc["primary_raw"]["chars"]
        for b in rb["bends"]:
            for c in b["characteristics"]:
                x = prim.get(c["characteristic_id"])
                if x and x[0] != c["decision"]:
                    out.append(dict(id=j["id"], v=v, label=b["label"], kind=c["kind"], side=c.get("side"), base=c["decision"], root=x[0],
                                    bval=c.get("actual"), bu=c.get("uncertainty"), rval=x[1], ru=x[2], exp=c.get("expected"), tol=c.get("tolerance", {}) and [c["tolerance"]["minus"], c["tolerance"]["plus"]]))
for o in out:
    print(f'{o["id"]} {o["v"]:<9} {o["label"]:>4} {o["kind"]:<18} {o["side"] or "":2} {o["base"]:>6} -> {o["root"]:<6} base {o["bval"]} U{o["bu"]}  root {o["rval"]} U{o["ru"]}  exp {o["exp"]} tol {o["tol"]}')
json.dump(out, open(sys.argv[4], "w"), indent=1)
