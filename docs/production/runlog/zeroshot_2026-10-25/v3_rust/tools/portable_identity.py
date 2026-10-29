"""Byte identity of two zs_rust runs (e.g. arm64 vs x86-64 under Rosetta, or 1 vs 16 threads), per scan:

* eval_lines.json without its volatile fields: everything except `created_at` and `conversion` (which carries wall
  times); `conversion.stages` is compared with every `wall` / `lane_wall_s` / timing entry removed;
* the stage dump (dump.json, when both runs wrote one) without `stages` (wall times): Z2 lines, candidates, the
  feature matrix X, evidence (with split runs), both GBM probabilities, the scan sha256;
* the exit code (10 = point cloud on both sides counts as identical).
Compared as serialized bytes (json.dumps of the parsed file with key order kept), i.e. bit-identical floats.
usage: .venv/bin/python -B docs/.../v3_rust/tools/portable_identity.py RUN_A/t1 RUN_B/t1 [--runs-a runs.json --runs-b
       runs.json --key t1] [--out OUT.json]"""
import argparse, json
from pathlib import Path

VOLATILE = {"wall", "lane_wall_s", "rust", "timing_s", "timers_s"}


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in VOLATILE}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def canon_eval(p: Path) -> str:
    d = json.loads(p.read_text())
    stages = strip((d.get("conversion") or {}).get("stages"))
    d = {k: v for k, v in d.items() if k not in ("created_at", "conversion")}
    d["scan"] = {"sha256": d.get("scan", {}).get("sha256")}
    d["stages"] = stages
    return json.dumps(d)


def canon_dump(p: Path) -> str:
    d = json.loads(p.read_text())
    d.pop("stages", None)
    return json.dumps(d)


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", default="")
a = ap.parse_args()
A, B = Path(a.a), Path(a.b)
res = {"a": str(A), "b": str(B), "scans": 0, "eval_identical": 0, "dump_compared": 0, "dump_identical": 0,
       "point_cloud_both": 0, "differ": {}, "missing": []}
sids = sorted({p.name for p in A.iterdir() if p.is_dir()} | {p.name for p in B.iterdir() if p.is_dir()})
for sid in sids:
    ea, eb = A / sid / "eval_lines.json", B / sid / "eval_lines.json"
    res["scans"] += 1
    if not ea.is_file() and not eb.is_file():
        res["point_cloud_both"] += 1  # no output on either side: the binary refused (exit 10) on both
        continue
    if not (ea.is_file() and eb.is_file()):
        res["missing"].append(sid)
        continue
    probs = []
    if canon_eval(ea) == canon_eval(eb):
        res["eval_identical"] += 1
    else:
        probs.append("eval_lines")
    da, db = A / sid / "dump.json", B / sid / "dump.json"
    if da.is_file() and db.is_file():
        res["dump_compared"] += 1
        if canon_dump(da) == canon_dump(db):
            res["dump_identical"] += 1
        else:
            probs.append("dump")
    if probs:
        res["differ"][sid] = probs
res["identical_all"] = not res["differ"] and not res["missing"]
print(json.dumps({k: v for k, v in res.items() if k != "differ"}), "differ:", list(res["differ"])[:10])
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
