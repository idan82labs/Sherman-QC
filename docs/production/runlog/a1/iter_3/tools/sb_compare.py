"""Summarise springback probe sweeps (summary.json per probe) per part: states wrong, angle errors within U,
false angle FAIL/PASS, other FAILs. usage: sb_compare.py SBPROBES_ROOT TAG [TAG ...]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tags = sys.argv[2:]
res = {}
for tag in tags:
    for f in sorted((root / tag).glob("*/S*/summary.json")):
        d = json.loads(f.read_text())
        res.setdefault(tag, {})[(f.parent.parent.name, int(f.parent.name[1:]))] = d
common = sorted(set.intersection(*[set(v) for v in res.values()])) if res else []
print(f"common probes: {len(common)}")
for tag in tags:
    per = {}
    for k in common:
        d = res[tag][k]
        p = per.setdefault(k[0], {"probes": 0, "bends": 0, "wrong": 0, "probes_wrong": 0, "angles": 0, "in_u": 0,
                                  "false_fail": 0, "false_pass": 0, "other_fail": 0, "invalid": 0})
        p["probes"] += 1
        p["bends"] += len(d.get("states", "").split())
        w = len(d.get("wrong_states", [])) if "error" not in d else len(d.get("states", "").split())
        p["wrong"] += w
        p["probes_wrong"] += int(w > 0)
        ang = d.get("angle_errors", [])
        p["angles"] += len(ang)
        p["in_u"] += sum(1 for a in ang if a["inU"])
        p["false_fail"] += len(d.get("angle_false_fail", []))
        p["false_pass"] += len(d.get("angle_false_pass", []))
        p["other_fail"] += len(d.get("fails_other", []))
        p["invalid"] += int(d.get("validity") != "VALID")
    tot = {}
    print(f"== {tag}")
    for part, p in sorted(per.items()):
        print(f"  {part:16s} " + " ".join(f"{k}={v}" for k, v in p.items()))
        for k, v in p.items():
            tot[k] = tot.get(k, 0) + v
    print(f"  {'ALL':16s} " + " ".join(f"{k}={v}" for k, v in tot.items()))
for k in common:
    rows = [(t, res[t][k]) for t in tags]
    if any(d.get("wrong_states") or d.get("angle_false_fail") or d.get("angle_false_pass") or d.get("fails_other")
           for _, d in rows):
        print(f"{k[0]} S{k[1]}:")
        for t, d in rows:
            print(f"   {t}: wrong={[w['bend'] for w in d.get('wrong_states', [])]} ffail={d.get('angle_false_fail')} "
                  f"fpass={d.get('angle_false_pass')} other={d.get('fails_other')}")
