import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
def gates(d):
    return {g.get("gate_id") or g.get("id") or g.get("name"): g for g in d["gates"]}
ga, gb = gates(a), gates(b)
print("verdict", a["verdict"], "->", b["verdict"], "| regression", a.get("regression_verdict"), "->", b.get("regression_verdict"), "| failed", a.get("failed_gates"), "->", b.get("failed_gates"))
for k in ga:
    x, y = ga[k], gb.get(k, {})
    vx, vy = x.get("value"), y.get("value")
    flag = "" if x.get("verdict") == y.get("verdict") else "  <-- verdict changed"
    if isinstance(vx, float) and isinstance(vy, float) and abs(vx - vy) > 1e-9:
        flag += f"  (delta {vy - vx:+.6g})"
    print(f"{k:40s} {x.get('verdict'):>14s} {str(vx)[:12]:>12s} -> {str(y.get('verdict')):>14s} {str(vy)[:12]:>12s} n={y.get('n')}{flag}")
