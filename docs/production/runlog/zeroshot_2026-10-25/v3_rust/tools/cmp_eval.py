"""Line-for-line comparison of eval_lines.json: zs_lane_v3_rs outputs vs the frozen zs_lane_v3 revision-2 DEV CLI
outputs (v3/rev2/dev_cli/zs/<sid>/eval_lines.json). Per line: endpoints (max |d| mm), score (max |d|), incl, src, op,
parts, fix; per scan: same number and order of lines, status.
usage: .venv/bin/python -B docs/.../v3_rust/tools/cmp_eval.py RS_ROOT [--ref REF_ROOT] [--out JSON]"""
import argparse, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
ap = argparse.ArgumentParser(); ap.add_argument("rs"); ap.add_argument("--ref", default=str(REPO / "docs/production/runlog/zeroshot_2026-10-25/v3/rev2/dev_cli/zs"))
ap.add_argument("--out", default=""); a = ap.parse_args()
res = {"scans": 0, "identical": 0, "within_tol": 0, "diff": {}, "max_endpoint_mm": 0.0, "max_score": 0.0, "missing": []}
for d in sorted(Path(a.ref).iterdir()):
    r = d / "eval_lines.json"; m = Path(a.rs) / d.name / "eval_lines.json"
    if not r.is_file(): continue
    if not m.is_file(): res["missing"].append(d.name); continue
    A, B = json.loads(r.read_text()), json.loads(m.read_text())
    res["scans"] += 1
    la, lb = A["owned_bend_regions"], B["owned_bend_regions"]
    probs = []
    if A["status"] != B["status"]: probs.append(f"status {A['status']} {B['status']}")
    if len(la) != len(lb): probs.append(f"n_lines {len(la)} {len(lb)}")
    ep = sc = 0.0
    for x, y in zip(la, lb):
        ep = max(ep, max(abs(p - q) for k in ("line_start", "line_end") for p, q in zip(x[k], y[k])))
        sc = max(sc, abs(x["score"] - y["score"]))
        for k in ("incl", "src", "op", "parts", "fix"):
            if x.get(k) != y.get(k): probs.append(f"{x['bend_id']}.{k} {x.get(k)} {y.get(k)}")
    res["max_endpoint_mm"] = max(res["max_endpoint_mm"], ep); res["max_score"] = max(res["max_score"], sc)
    exact = not probs and ep == 0.0 and sc == 0.0
    if exact: res["identical"] += 1
    if not probs and ep <= 1e-6 and sc <= 1e-9: res["within_tol"] += 1
    elif probs or ep > 1e-6 or sc > 1e-9: res["diff"][d.name] = {"problems": probs[:8], "max_endpoint_mm": ep, "max_score": sc}
print(json.dumps({k: v for k, v in res.items() if k != "diff"}, indent=1))
for k, v in res["diff"].items(): print(k, v)
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
