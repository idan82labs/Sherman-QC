"""Aggregate springback-probe summaries: per tag, probes, validity, wrong bend states, angle false FAIL/PASS,
angle errors within U, other FAILs; and per-probe differences between two tags.
usage: sbagg.py DIR_A [DIR_B]   (DIR = .../sbprobes/<tag>)"""
import json, sys
from pathlib import Path


def load(d):
    out = {}
    for p in sorted(Path(d).glob("*/S*/summary.json")):
        out[(p.parent.parent.name, p.parent.name)] = json.loads(p.read_text())
    return out


def agg(rows):
    n = len(rows)
    inval = sum(1 for r in rows.values() if r.get("validity") != "VALID")
    err = sum(1 for r in rows.values() if r.get("rc") not in (0, None) or "error" in r)
    wrong = sum(len(r.get("wrong_states") or []) for r in rows.values())
    pw = sum(1 for r in rows.values() if r.get("wrong_states"))
    ff = sum(len(r.get("angle_false_fail") or []) for r in rows.values())
    fp = sum(len(r.get("angle_false_pass") or []) for r in rows.values())
    ang = [a for r in rows.values() for a in (r.get("angle_errors") or [])]
    inu = sum(1 for a in ang if a["inU"])
    other = sum(len(r.get("fails_other") or []) for r in rows.values())
    bends = sum(len((r.get("states") or "").split()) for r in rows.values())
    t = [r.get("t_e3_s", 0.0) for r in rows.values()]
    return {"probes": n, "bend_states": bends, "not_valid": inval, "errors": err, "wrong_states": wrong,
            "probes_with_wrong": pw, "angle_false_fail": ff, "angle_false_pass": fp, "angles": len(ang),
            "angles_within_U": inu, "other_fails": other, "t_e3_max_s": max(t) if t else None}


A = load(sys.argv[1])
print("A", sys.argv[1], json.dumps(agg(A)))
if len(sys.argv) > 2:
    B = load(sys.argv[2])
    print("B", sys.argv[2], json.dumps(agg(B)))
    common = sorted(set(A) & set(B))
    print("common probes", len(common))
    print("A on common", json.dumps(agg({k: A[k] for k in common})))
    print("B on common", json.dumps(agg({k: B[k] for k in common})))
    for k in common:
        a, b = A[k], B[k]
        if a.get("states") != b.get("states") or a.get("validity") != b.get("validity") or \
                len(a.get("angle_false_fail") or []) != len(b.get("angle_false_fail") or []) or \
                len(a.get("angle_false_pass") or []) != len(b.get("angle_false_pass") or []) or \
                sorted(a.get("fails_other") or []) != sorted(b.get("fails_other") or []):
            print("DIFF", k, "\n   A:", a.get("validity"), a.get("states"), a.get("angle_false_fail"), a.get("fails_other"),
                  "\n   B:", b.get("validity"), b.get("states"), b.get("angle_false_fail"), b.get("fails_other"))
