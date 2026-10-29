"""Dense set vs native-density reference (same engine). usage: dense_an.py ROOT [OUT_JSON]"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
recs = {}
for d in sorted((root / "runs").iterdir()):
    rp = d / "record.json"
    if rp.exists():
        recs[d.name] = json.loads(rp.read_text())


def chars(r):
    return {c[9]: (c[5], c[6], c[7], c[10] if len(c) > 10 else []) for c in r.get("chars", []) if len(c) > 9}


def log_lines(name, keys):
    p = root / "runs" / name / "engine" / "engine.log"
    if not p.exists():
        return []
    return [l[:400] for l in p.read_text().splitlines() if any(k in l for k in keys)]


out = {"native": {}, "dense": []}
for n in ("A_n1", "B_n1", "C_n1", "D_n1"):
    if n in recs:
        r = recs[n]
        out["native"][n] = {"wall_s": r["wall_s"], "rss_gib": round(r["peak_rss_bytes"] / 2**30, 2),
                            "summary": r.get("summary"),
                            "log": log_lines(n, ("pose well", "alternative poses", "resample check"))}
tot_fp = 0
for name, r in recs.items():
    if name.endswith("_n1"):
        continue
    ref = recs.get(name.split("_")[0] + "_n1")
    if ref is None:
        continue
    a, b = chars(ref), chars(r)
    ch, fp, tp, flips = [], 0, 0, 0
    for cid, (d0, v0, u0, _) in a.items():
        d1, v1, u1, notes = b.get(cid, (None, None, None, []))
        if d0 != d1:
            ch.append({"cid": cid, "native": [d0, v0, u0], "dense": [d1, v1, u1],
                       "notes": [x for x in notes if x.isupper()]})
            fp += d0 == "FAIL" and d1 == "PASS"
            tp += d1 == "PASS"
            flips += d0 in ("PASS", "FAIL") and d1 in ("PASS", "FAIL")
    tot_fp += fp
    lg = log_lines(name, ("dense input thinned", "pose well", "alternative poses", "resample check", "best datum"))
    out["dense"].append({"mesh": name, "wall_s": r["wall_s"], "rss_gib": round(r["peak_rss_bytes"] / 2**30, 2),
                         "state": r.get("state"), "summary": r.get("summary"), "native_summary": ref.get("summary"),
                         "changes": len(ch), "fail_to_pass": fp, "to_pass": tp, "decided_flips": flips,
                         "thinned": any("thinned" in l for l in lg),
                         "switched": any("replaces the chosen" in l for l in lg),
                         "alt_fired": [l for l in lg if "alternative poses" in l],
                         "log": lg, "change_list": ch})
out["fail_to_pass_total"] = tot_fp
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for d in out["dense"]:
    print(f"{d['mesh']:7s} wall {d['wall_s']:6.1f} rss {d['rss_gib']:4.2f} {d['summary']:26s} thin={int(d['thinned'])} "
          f"switch={int(d['switched'])} alt={len(d['alt_fired'])} changes {d['changes']:2d} F->P {d['fail_to_pass']} "
          f"->P {d['to_pass']} flips {d['decided_flips']}")
print("FAIL->PASS total", tot_fp)
for n, v in out["native"].items():
    print(n, v["wall_s"], v["summary"], [l[:160] for l in v["log"]])
