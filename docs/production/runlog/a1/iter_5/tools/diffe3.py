"""Characteristic-level differences between two E3 run dirs (DIR/E3/<case>/engine/result.json).
usage: diffe3.py DIR_A DIR_B [THR_MM_OR_DEG=0.05]"""
import json, glob, os, sys
A, B = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
def summ(p):
    r = json.load(open(p))
    out = {}
    for b in r["bends"]:
        for ch in b["characteristics"]:
            out[(b["label"], ch["kind"][:5] + (ch.get("side") or ""))] = (b["state"], ch.get("decision"), None if ch.get("deviation") is None else round(ch["deviation"], 2), None if ch.get("uncertainty") is None else round(ch["uncertainty"], 2), ch.get("reason"))
    return r["inspection_validity"]["state"], (r.get("stage_decision") or {}).get("detail", "")[:60], out
n_cases = n_changed = 0
for d in sorted(glob.glob(f"{A}/E3/pc_*")):
    c = os.path.basename(d)
    pa = d + "/engine/result.json"; pb = f"{B}/E3/{c}/engine/result.json"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    n_cases += 1
    va, sa, a = summ(pa); vb, sb, b = summ(pb)
    rows = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x == y: continue
        if x is None or y is None or x[0] != y[0] or x[1] != y[1] or (x[2] is not None and y[2] is not None and abs(x[2] - y[2]) > thr) or ((x[2] is None) != (y[2] is None)):
            rows.append((k, x, y))
    if rows or va != vb:
        n_changed += 1
        print(f"{c} {va}->{vb}")
        for k, x, y in rows:
            print("    ", k, x, "->", y)
print(f"cases compared {n_cases}, with changes {n_changed}")
