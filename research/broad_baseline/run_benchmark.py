#!/usr/bin/env python3
"""Serial driver: run lanes over a build_case_list.py case list, then score them.

One heavy job at a time (run_lane.py is invoked sequentially and itself holds the
broad-baseline engine flock and the python-RSS gate). Resumable: a lane whose
run_record.json already exists is skipped with --skip-existing.

Layout
  <runs-root>/<case_id>/<lane>/predictions_*.json, run_record.json, logs/
  <work-root>/<case_id>/<lane>/...   bulky intermediates (keep out of docs/)
  <runs-root>/progress.json          per case/lane status, wall, peak RSS
  <runs-root>/score_<lane>[_<variant>].json   score.py case-list reports

Example
  .venv/bin/python -B research/broad_baseline/run_benchmark.py \
      --case-list docs/production/runlog/broad_baseline/case_list_consumed_v1.json \
      --runs-root docs/production/runlog/broad_baseline/runs \
      --work-root /private/tmp/claude-501/sherman_broad_baseline_work \
      --lanes runtime,best_f1 --skip-existing
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402

# lane -> prediction files to score (relative to the lane dir)
SCORED_FILES = {
    "runtime": ["predictions_runtime.json", "predictions_runtime_gated.json",
                "predictions_runtime_with_placeholders.json"],
    "best_f1": ["predictions_best_f1.json", "predictions_a0.json", "predictions_cad_gated.json",
                "predictions_f1_zero_shot_decomposition.json"],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/private/tmp/claude-501/sherman_broad_baseline_work"))
    parser.add_argument("--lanes", default="runtime,best_f1")
    parser.add_argument("--only", action="append")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--step-timeout", type=float, default=600.0)
    parser.add_argument("--lane-timeout", type=float, default=600.0)
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args(argv)
    lanes = [lane.strip() for lane in args.lanes.split(",") if lane.strip()]
    cases = bb.read_json(args.case_list)["cases"]
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only)]
    if args.max_cases:
        cases = cases[: args.max_cases]
    progress_path = args.runs_root / "progress.json"
    progress = bb.read_json(progress_path) if progress_path.is_file() else {"runs": {}}
    for lane in lanes:  # lane-major: all fast runtime runs give signal before the slow CP-SAT lane
        for index, case in enumerate(cases, start=1):
            key = f"{case['case_id']}/{lane}"
            out = args.runs_root / case["case_id"] / lane
            if args.skip_existing and (out / "run_record.json").is_file():
                continue
            if any(flag.startswith("SCAN_SHA_MISMATCH") for flag in case.get("flags") or []):
                progress["runs"][key] = {"status": "SKIPPED_SCAN_SHA_MISMATCH"}
                continue
            cmd = [str(bb.APP_PYTHON), "-B", str(bb.HERE / "run_lane.py"), "--lane", lane,
                   "--scan", case["scan_path"], "--out", str(out),
                   "--work-dir", str(args.work_root / case["case_id"] / lane),
                   "--step-timeout", str(args.step_timeout), "--lane-timeout", str(args.lane_timeout),
                   "--rss-limit-gib", str(args.rss_limit_gib)]
            if case.get("cad_path"):
                cmd += ["--cad", case["cad_path"]]
            print(f"[{lane} {index}/{len(cases)}] {case['case_id']} {Path(case['scan_path']).name}", flush=True)
            started = time.monotonic()
            done = subprocess.run(cmd, cwd=str(bb.ROOT), capture_output=True, text=True, check=False)
            record_path = out / "run_record.json"
            record = bb.read_json(record_path) if record_path.is_file() else {}
            progress["runs"][key] = {
                "status": record.get("status") or f"DRIVER_RC_{done.returncode}",
                "returncode": done.returncode, "wall_s": round(time.monotonic() - started, 1),
                "lane_wall_s": record.get("lane_wall_s"), "peak_rss_bytes": record.get("peak_rss_bytes"),
                "line_count": record.get("line_count"), "stderr_tail": done.stderr[-1500:] if done.returncode else None,
            }
            progress["updated_at"] = bb.utc_now()
            bb.write_json(progress_path, progress)
            print(f"    -> {progress['runs'][key]['status']} lines={record.get('line_count')} "
                  f"wall={progress['runs'][key]['wall_s']}s", flush=True)
    if not args.no_score:
        for lane in lanes:
            for name in SCORED_FILES.get(lane, []):
                variant = name.removeprefix("predictions_").removesuffix(".json")
                template = str(args.runs_root / "{case_id}" / lane / name)
                report = args.runs_root / f"score_{variant}.json"
                subprocess.run([str(bb.APP_PYTHON), "-B", str(bb.HERE / "score.py"), "case-list",
                                "--case-list", str(args.case_list), "--pred-template", template,
                                "--title", f"{lane}:{variant}", "--out", str(report)]
                               + sum((["--only", c["case_id"]] for c in cases), []),
                               cwd=str(bb.ROOT), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
