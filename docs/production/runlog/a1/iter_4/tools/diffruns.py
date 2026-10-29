import json, glob, os, sys
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4"
A, B = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
def summ(p):
    r = json.load(open(p))
    out = {}
    for b in r["bends"]:
        for ch in b["characteristics"]:
            out[(b["label"], ch["kind"][:5] + (ch.get("side") or ""))] = (b["state"], ch.get("decision"), None if ch.get("deviation") is None else round(ch["deviation"], 2), None if ch.get("uncertainty") is None else round(ch["uncertainty"], 2))
    return r["inspection_validity"]["state"], out
for d in sorted(glob.glob(f"{S}/phys/physical_{A}/E3/pc_*")):
    c = os.path.basename(d)
    pa = d + "/engine/result.json"; pb = f"{S}/phys/physical_{B}/E3/{c}/engine/result.json"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    va, a = summ(pa); vb, b = summ(pb)
    rows = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x == y: continue
        if x is None or y is None or x[0] != y[0] or x[1] != y[1] or (x[2] is not None and y[2] is not None and abs(x[2] - y[2]) > thr) or ((x[2] is None) != (y[2] is None)):
            rows.append((k, x, y))
    if rows or va != vb:
        print(f"{c} {va}->{vb}")
        for k, x, y in rows:
            print("    ", k, x, "->", y)
