#!/opt/homebrew/bin/python3.11
"""
Run one bend corpus regression case in an isolated process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from bend_inspection_pipeline import load_bend_runtime_config, run_progressive_bend_inspection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one bend corpus scan case.")
    parser.add_argument("--part-key", required=True)
    parser.add_argument("--cad-path", required=True)
    parser.add_argument("--scan-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--state", default="unknown")
    parser.add_argument("--expected-total")
    parser.add_argument("--expected-completed")
    parser.add_argument("--spec-bend-angles-json", default="[]")
    parser.add_argument("--spec-schedule-type", default="")
    parser.add_argument("--skip-heatmap-consistency", action="store_true")
    parser.add_argument("--heatmap-cache-dir", default=None)
    args = parser.parse_args()

    expected_total = None if args.expected_total in {None, "", "None"} else int(float(args.expected_total))
    expected_completed = None if args.expected_completed in {None, "", "None"} else int(float(args.expected_completed))
    spec_bend_angles: List[float] = [float(v) for v in json.loads(args.spec_bend_angles_json or "[]")]

    runtime_config = load_bend_runtime_config(None)
    report, details = run_progressive_bend_inspection(
        cad_path=args.cad_path,
        scan_path=args.scan_path,
        part_id=args.part_key,
        tolerance_angle=1.0,
        tolerance_radius=0.5,
        runtime_config=runtime_config,
        spec_bend_angles_hint=spec_bend_angles,
        spec_schedule_type=(args.spec_schedule_type or None),
        expected_bend_override=expected_total,
        scan_state=(args.state or None),
    )
    report_dict = report.to_dict()
    payload: Dict[str, Any] = {
        "part_key": args.part_key,
        "cad_path": args.cad_path,
        "scan_path": args.scan_path,
        "scan_name": Path(args.scan_path).name,
        "expectation": {
            "state": args.state,
            "expected_total_bends": expected_total,
            "expected_completed_bends": expected_completed,
        },
        "spec_bend_angles_hint": spec_bend_angles,
        "spec_schedule_type": args.spec_schedule_type or None,
        "skip_heatmap_consistency": bool(args.skip_heatmap_consistency),
        "heatmap_cache_dir": args.heatmap_cache_dir,
        "summary": report_dict.get("summary") or {},
        "details": details.to_dict(),
        "warnings": details.warnings,
        "matches": report_dict.get("matches") or [],
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    gc.collect()
    print(json.dumps({"event": "case_done", "part_key": args.part_key, "scan_name": Path(args.scan_path).name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
