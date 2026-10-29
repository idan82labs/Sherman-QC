"""Per-probe angle value comparison of two springback-probe sweeps. usage: sbcmp.py DIR_A DIR_B"""
import json, sys
from pathlib import Path
def load(d):
    return {(p.parent.parent.name, p.parent.name): json.loads(p.read_text()) for p in Path(d).glob("*/S*/summary.json")}
A, B = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(A) & set(B))
n = moved = 0; mx = 0.0
for k in common:
    ea = {(e.get("bend") or e.get("label")): e for e in (A[k].get("angle_errors") or [])}
    eb = {(e.get("bend") or e.get("label")): e for e in (B[k].get("angle_errors") or [])}
    for b in set(ea) & set(eb):
        n += 1
        va = ea[b].get("err", ea[b].get("error")); vb = eb[b].get("err", eb[b].get("error"))
        if va is None or vb is None:
            continue
        d = abs(float(va) - float(vb)); mx = max(mx, d)
        if d > 1e-6:
            moved += 1
print(f"common probes {len(common)}, angles compared {n}, changed {moved}, max |change| {mx:.6f} deg")
if common:
    k = common[0]; print("example keys:", list((A[k].get('angle_errors') or [{}])[0].keys()))
