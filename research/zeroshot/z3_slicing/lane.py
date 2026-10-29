"""Z3 standalone lane: scan -> eval_lines.json (sherman.broad_baseline.eval_lines.v1, raw frame).

usage: python -m research.zeroshot.z3_slicing.lane --scan SCAN.ply --model z3/model_z3.pkl --out eval_lines.json
       [--all-candidates CANDS.json]
Custody: the caller's harness guard decides which scans may be run; this module only refuses forbidden paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, slicer, zscore  # noqa: E402
from research.zeroshot.z3_slicing.selector import feats  # noqa: E402


def run_scan(scan: str, model_path: str) -> tuple[dict, dict]:
    p = custody.check_path(scan)
    t0 = time.time()
    mesh = slicer.load_mesh(str(p))
    res = slicer.detect(mesh)
    M = pickle.loads(Path(model_path).read_bytes())
    C = res["candidates"]
    pr = M["model"].predict(feats(res)) if C else []
    for c, q in zip(C, pr):
        c["score"] = float(q)
    sel = [c for c in C if c["score"] >= M["threshold"]]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    conv = {"rule": "z3 axis-family slicing + selector", "model": Path(model_path).name, "threshold": M["threshold"],
            "wall_s": round(time.time() - t0, 2), "t_mm": res["t_mm"]}
    return (zscore.to_payload(sel, lane="z3_axis_slicing", scan_path=str(p), scan_sha=sha, extra=conv),
            zscore.to_payload(C, lane="z3_axis_slicing_candidates", scan_path=str(p), scan_sha=sha, extra=conv))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-candidates", default="")
    a = ap.parse_args(argv)
    sel, allc = run_scan(a.scan, a.model)
    Path(a.out).write_text(json.dumps(sel, indent=1))
    if a.all_candidates:
        Path(a.all_candidates).write_text(json.dumps(allc, indent=1))
    print(f"{a.scan}: {sel['line_count']} lines ({allc['line_count']} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
