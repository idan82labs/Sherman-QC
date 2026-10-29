#!/usr/bin/env python3
"""Broad-run serial driver: both lanes on every case of a broad case list.

Layout (as requested for the broad run)
  <runs-root>/<lane>/<scan_id>/predictions_*.json, run_record.json, logs/
  <work-root>/<lane>/<scan_id>/...        bulky intermediates (outside docs/)
  <runs-root>/progress.json               per lane/case status, wall, peak RSS
  <runs-root>/driver_manifest.json        case list sha, engine source tree sha at start/end

Strictly one heavy job at a time: run_lane.py is called sequentially; it holds the
broad-baseline engine flock, waits for total python RSS < --rss-limit-gib before
every heavy step, runs each step under /usr/bin/time -l and SIGKILLs a lane at
--lane-timeout (600 s) of compute.

Resume-safe: a lane/case whose run_record.json exists for the same scan sha256 and
CAD sha256 is skipped (status recorded as SKIPPED_FINISHED).

  .venv/bin/python -B research/broad_baseline/run_broad.py \
      --case-list docs/production/runlog/broad_baseline/case_list_broad_v2.json \
      --runs-root docs/production/runlog/broad_baseline/runs --lanes runtime,best_f1
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

ENGINE_TREES = ("backend", "domains/bend", "scripts", "config")
ENGINE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}


def engine_tree_sha() -> dict:
    digest = hashlib.sha256()
    count = 0
    for tree in ENGINE_TREES:
        root = bb.ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(bb.ROOT).as_posix()
            if not path.is_file() or path.suffix not in ENGINE_SUFFIXES or "__pycache__" in rel \
                    or "/tests/" in rel or "/node_modules/" in rel:
                continue
            digest.update(rel.encode("utf-8") + b"\0" + bb.sha256_file(path).encode("ascii") + b"\n")
            count += 1
    return {"trees": list(ENGINE_TREES), "suffixes": sorted(ENGINE_SUFFIXES), "files": count,
            "sha256": digest.hexdigest()}


def finished(out: Path, case: dict) -> bool:
    record_path = out / "run_record.json"
    if not record_path.is_file():
        return False
    try:
        record = bb.read_json(record_path)
    except (OSError, ValueError):
        return False
    if record.get("scan", {}).get("sha256") != case["scan_sha256"]:
        return False
    cad = record.get("cad") or {}
    return (cad.get("sha256") if cad else None) == case.get("cad_sha256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path,
                        default=Path("/private/tmp/claude-501/sherman_broad_baseline_work/broad_v2"))
    parser.add_argument("--lanes", default="runtime,best_f1")
    parser.add_argument("--only", action="append")
    parser.add_argument("--gold11-last", action="store_true", default=True)
    parser.add_argument("--step-timeout", type=float, default=600.0)
    parser.add_argument("--lane-timeout", type=float, default=600.0)
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    args = parser.parse_args(argv)
    case_list = bb.read_json(args.case_list)
    cases = list(case_list["cases"])
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only)]
    if args.gold11_last:
        cases.sort(key=lambda c: (c.get("tier") == "GOLD11", c["case_id"]))
    lanes = [lane.strip() for lane in args.lanes.split(",") if lane.strip()]
    args.runs_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.runs_root / "driver_manifest.json"
    manifest = bb.read_json(manifest_path) if manifest_path.is_file() else {"sessions": []}
    session = {"started_at": bb.utc_now(), "case_list": str(args.case_list),
               "case_list_file_sha256": bb.sha256_file(args.case_list),
               "case_list_sha256": case_list.get("case_list_sha256"), "lanes": lanes,
               "cases": len(cases), "engine_tree_at_start": engine_tree_sha(),
               "limits": {"step_timeout_s": args.step_timeout, "lane_timeout_s": args.lane_timeout,
                          "rss_limit_gib": args.rss_limit_gib, "OMP_NUM_THREADS": "1"}}
    manifest["sessions"].append(session)
    bb.write_json(manifest_path, manifest)
    progress_path = args.runs_root / "progress.json"
    progress = bb.read_json(progress_path) if progress_path.is_file() else {"runs": {}}
    for lane in lanes:
        for index, case in enumerate(cases, start=1):
            key = f"{lane}/{case['case_id']}"
            out = args.runs_root / lane / case["case_id"]
            if finished(out, case):
                print(f"[{lane} {index}/{len(cases)}] {case['case_id']} finished -> skip", flush=True)
                continue
            if any(flag.startswith("SCAN_SHA_MISMATCH") for flag in case.get("flags") or []):
                progress["runs"][key] = {"status": "SKIPPED_SCAN_SHA_MISMATCH"}
                continue
            cmd = [str(bb.APP_PYTHON), "-B", str(bb.HERE / "run_lane.py"), "--lane", lane,
                   "--scan", case["scan_path"], "--out", str(out),
                   "--work-dir", str(args.work_root / lane / case["case_id"]),
                   "--step-timeout", str(args.step_timeout), "--lane-timeout", str(args.lane_timeout),
                   "--rss-limit-gib", str(args.rss_limit_gib)]
            if case.get("cad_path"):
                cmd += ["--cad", case["cad_path"]]
            print(f"[{lane} {index}/{len(cases)}] {case['case_id']} {Path(case['scan_path']).name} "
                  f"cad={'yes' if case.get('cad_path') else 'no'}", flush=True)
            started = time.monotonic()
            done = subprocess.run(cmd, cwd=str(bb.ROOT), capture_output=True, text=True, check=False)
            record_path = out / "run_record.json"
            record = bb.read_json(record_path) if record_path.is_file() else {}
            progress["runs"][key] = {
                "status": record.get("status") or f"DRIVER_RC_{done.returncode}",
                "returncode": done.returncode, "wall_s": round(time.monotonic() - started, 1),
                "lane_wall_s": record.get("lane_wall_s"), "heavy_step_wall_s": record.get("heavy_step_wall_s"),
                "peak_rss_bytes": record.get("peak_rss_bytes"), "line_count": record.get("line_count"),
                "timed_out_steps": [s["name"] for s in record.get("steps", []) if s.get("timed_out")],
                "stderr_tail": done.stderr[-1500:] if done.returncode not in (0,) else None,
                "finished_at": bb.utc_now(),
            }
            progress["updated_at"] = bb.utc_now()
            bb.write_json(progress_path, progress)
            peak = (record.get("peak_rss_bytes") or 0) / 2**30
            print(f"    -> {progress['runs'][key]['status']} lines={record.get('line_count')} "
                  f"wall={progress['runs'][key]['wall_s']}s peak={peak:.2f}GiB", flush=True)
    session["finished_at"] = bb.utc_now()
    session["engine_tree_at_end"] = engine_tree_sha()
    session["engine_tree_unchanged"] = session["engine_tree_at_end"]["sha256"] == session["engine_tree_at_start"]["sha256"]
    bb.write_json(manifest_path, manifest)
    print(f"engine tree unchanged during session: {session['engine_tree_unchanged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
