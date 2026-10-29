"""Characteristic-level diff of two A1 physical runs (E3 result.json per case)."""
import json, sys
from pathlib import Path
A, B = Path(sys.argv[1]), Path(sys.argv[2])
tol = 0.05
n_case = n_chg = 0; out = []
for ca in sorted(A.iterdir()):
    ra, rb = ca / "engine" / "result.json", B / ca.name / "engine" / "result.json"
    if not ra.exists() or not rb.exists():
        out.append(f"{ca.name}: missing {ra.exists()} {rb.exists()}"); continue
    a, b = json.loads(ra.read_text()), json.loads(rb.read_text())
    n_case += 1
    if a["inspection_validity"]["state"] != b["inspection_validity"]["state"]:
        out.append(f"{ca.name}: validity {a['inspection_validity']['state']} -> {b['inspection_validity']['state']}")
    for x, y in zip(a["bends"], b["bends"]):
        if x["state"] != y["state"]:
            out.append(f"{ca.name} {x['label']}: state {x['state']} -> {y['state']}")
        for cx, cy in zip(x["characteristics"], y["characteristics"]):
            ch = cx["decision"] != cy["decision"] or cx["state"] != cy["state"] or (
                cx.get("actual") is not None and cy.get("actual") is not None and abs(cx["actual"] - cy["actual"]) > tol)
            if ch:
                n_chg += 1
                out.append(f"{ca.name} {x['label']} {cx['kind']}{('/' + cx['side']) if cx.get('side') else ''}: {cx['state']}/{cx['decision']}/{cx.get('actual')} -> {cy['state']}/{cy['decision']}/{cy.get('actual')}")
print(f"cases compared {n_case}; characteristic changes {n_chg}")
print("\n".join(out))
