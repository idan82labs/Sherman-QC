"""Compare CAD-stage probe sweeps (summary.json per probe): baseline engine vs the iteration-3 engine.

usage: sweep_compare.py PROBES_ROOT TAG_A TAG_B [--json OUT]
Per part and overall: probes, bend states wrong (FORMED/NOT_YET_FORMED/UNKNOWN vs the stage truth),
characteristics FAIL on nominal geometry (every FAIL is false), REVIEW, MEASURED deviations beyond U,
UNKNOWN on formed bends, invalid runs, E3 time.
"""
import json
import sys
from pathlib import Path

root, ta, tb = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out_json = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None


def load(tag):
    res = {}
    for f in sorted((root / tag).glob("*/S*/summary.json")):
        d = json.loads(f.read_text())
        res[(f.parent.parent.name, int(f.parent.name[1:]))] = d
    return res


A, B = load(ta), load(tb)
keys = sorted(set(A) & set(B))
rows = {}
tot = {ta: {}, tb: {}}


def stats(d):
    states = d.get("states", "")
    nb = len(states.split()) if states else 0
    return {"probes": 1, "bends": nb, "wrong": len(d.get("wrong_states", [])) if "error" not in d else nb,
            "false_fail": len(d.get("fails", [])), "review": len(d.get("reviews", [])),
            "beyond_u": len(d.get("dev_beyond_U", [])), "unknown_formed": len(d.get("unknown_on_formed", [])),
            "invalid": int(d.get("validity") != "VALID"), "t_e3": float(d.get("t_e3_s", 0.0))}


per_part = {}
changes = []
for k in keys:
    sa, sb = stats(A[k]), stats(B[k])
    for tag, s in ((ta, sa), (tb, sb)):
        pp = per_part.setdefault(k[0], {ta: {}, tb: {}})[tag]
        for kk, v in s.items():
            pp[kk] = pp.get(kk, 0) + v
            tot[tag][kk] = tot[tag].get(kk, 0) + v
    if (sa["wrong"], sa["false_fail"], sa["beyond_u"], sa["unknown_formed"]) != (sb["wrong"], sb["false_fail"], sb["beyond_u"], sb["unknown_formed"]):
        changes.append((k, sa, sb, A[k].get("states"), B[k].get("states")))

cols = ["probes", "bends", "wrong", "false_fail", "review", "beyond_u", "unknown_formed", "invalid", "t_e3"]
print(f"{'part':16s} " + " ".join(f"{c[:10]:>10s}" for c in cols))
for part, d in sorted(per_part.items()):
    for tag in (ta, tb):
        s = d[tag]
        print(f"{part[:12] + ' ' + tag[:3]:16s} " + " ".join(f"{(round(s.get(c, 0), 1) if c == 't_e3' else s.get(c, 0)):>10}" for c in cols))
print("-" * 120)
for tag in (ta, tb):
    s = tot[tag]
    print(f"{'ALL ' + tag:16s} " + " ".join(f"{(round(s.get(c, 0), 1) if c == 't_e3' else s.get(c, 0)):>10}" for c in cols))
print("\nprobes whose state/FAIL/beyond-U/unknown counts changed:")
for k, sa, sb, sta, stb in changes:
    print(f"  {k[0]} S{k[1]}: wrong {sa['wrong']}->{sb['wrong']} fail {sa['false_fail']}->{sb['false_fail']} "
          f"beyondU {sa['beyond_u']}->{sb['beyond_u']} unkF {sa['unknown_formed']}->{sb['unknown_formed']}")
    if sa["wrong"] or sb["wrong"]:
        print(f"      {ta}: {sta}\n      {tb}: {stb}")
if out_json:
    out_json.write_text(json.dumps({"tags": [ta, tb], "per_part": per_part, "total": tot,
                                    "changed": [[f"{k[0]}:S{k[1]}", sa, sb] for k, sa, sb, _, _ in changes]}, indent=1))
