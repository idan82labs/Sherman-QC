#!/usr/bin/env python3
"""Run a typed F1 PLY-only slice with one measured process per scan."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_accuracy_validation import (  # noqa: E402
    _aggregate_validation,
    _write_json,
    _write_review_markdown,
)


MAX_RSS_PATTERN = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[rank]


def _typed_release_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    exact_correct = 0
    range_contains = 0
    raw_exact_correct = 0
    render_exact_correct = 0
    visual_exact_correct = 0
    scored_rows = [
        row for row in rows if row.get("expected_bend_count_scored") is not None
    ]
    for row in scored_rows:
        expected = int(row["expected_bend_count_scored"])
        algorithm = dict(row.get("algorithm") or {})
        accepted_exact = algorithm.get("accepted_exact_bend_count")
        accepted_range = list(algorithm.get("accepted_bend_count_range") or ())
        raw_exact = algorithm.get("raw_f1_map_bend_count")
        render_exact = algorithm.get("render_admissible_exact_bend_count")
        exact_correct += int(accepted_exact is not None and int(accepted_exact) == expected)
        range_contains += int(
            len(accepted_range) == 2 and int(accepted_range[0]) <= expected <= int(accepted_range[1])
        )
        raw_exact_correct += int(raw_exact is not None and int(raw_exact) == expected)
        render_exact_correct += int(render_exact is not None and int(render_exact) == expected)
        visual_exact_correct += int(
            str(algorithm.get("visual_acceptance_status") or "") == "accepted"
            and accepted_exact is not None
            and int(accepted_exact) == expected
        )
    count = len(scored_rows)
    return {
        "scans_processed": count,
        "exact_count_correct": exact_correct,
        "exact_count_accuracy": round(exact_correct / count, 6) if count else None,
        "range_contains_truth_count": range_contains,
        "range_containment_rate": round(range_contains / count, 6) if count else None,
        "raw_f1_exact_correct": raw_exact_correct,
        "raw_f1_exact_accuracy": round(raw_exact_correct / count, 6) if count else None,
        "render_admissible_exact_correct": render_exact_correct,
        "render_admissible_exact_accuracy": round(render_exact_correct / count, 6) if count else None,
        "visually_accepted_exact_correct": visual_exact_correct,
        "visual_exact_accuracy": round(visual_exact_correct / count, 6) if count else None,
    }


def run_slice(
    *,
    slice_payload: Dict[str, Any],
    output_dir: Path,
    review_output_dir: Path,
    cache_dir: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    entries = list(slice_payload.get("entries") or ())
    output_dir.mkdir(parents=True, exist_ok=True)
    review_output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "isolated_cases"
    cases_dir.mkdir(exist_ok=True)
    rows: List[Dict[str, Any]] = []
    operations: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        part_key = str(entry.get("part_key") or f"case_{index}")
        scan_name = Path(str(entry["scan_path"])).name
        case_dir = cases_dir / f"{index:02d}_{part_key}"
        case_review_dir = review_output_dir / f"{index:02d}_{part_key}"
        case_dir.mkdir(parents=True, exist_ok=True)
        slice_path = case_dir / "slice.json"
        _write_json(
            slice_path,
            {
                "schema_version": slice_payload.get("schema_version", 1),
                "target": slice_payload.get("target"),
                "entries": [entry],
            },
        )
        command = [
            "/usr/bin/time",
            "-l",
            sys.executable,
            str(ROOT / "scripts" / "run_f1_accuracy_validation.py"),
            "--known-slice-json",
            str(slice_path),
            "--track",
            "known",
            "--route-mode",
            "apply",
            "--strict-blind-inference",
            "--output-dir",
            str(case_dir / "result"),
            "--review-output-dir",
            str(case_review_dir),
            "--cache-dir",
            str(cache_dir),
        ]
        print(f"[{index}/{len(entries)}] START {part_key}: {scan_name}", flush=True)
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=max(60, int(timeout_sec)),
                check=False,
            )
            wall_sec = perf_counter() - started
            time_match = MAX_RSS_PATTERN.search(completed.stderr or "")
            max_rss_bytes = int(time_match.group(1)) if time_match else None
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "isolated case failed").strip())
            case_summary = json.loads((case_dir / "result" / "summary.json").read_text(encoding="utf-8"))
            case_rows = list(case_summary.get("results") or ())
            if len(case_rows) != 1:
                raise RuntimeError(f"isolated case produced {len(case_rows)} result rows")
            rows.extend(case_rows)
            operations.append(
                {
                    "part_key": part_key,
                    "scan_name": scan_name,
                    "status": "success",
                    "wall_time_sec": round(wall_sec, 3),
                    "maximum_resident_set_bytes": max_rss_bytes,
                }
            )
            print(
                f"[{index}/{len(entries)}] DONE {part_key}: wall={wall_sec:.1f}s "
                f"peak_rss_gib={(max_rss_bytes or 0) / (1024 ** 3):.2f}",
                flush=True,
            )
        except Exception as exc:
            wall_sec = perf_counter() - started
            failure = {"part_key": part_key, "scan_name": scan_name, "error": str(exc)}
            failures.append(failure)
            operations.append(
                {
                    "part_key": part_key,
                    "scan_name": scan_name,
                    "status": "failed",
                    "wall_time_sec": round(wall_sec, 3),
                    "maximum_resident_set_bytes": None,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(entries)}] FAILED {part_key}: {exc}", flush=True)

    successful_latencies = [float(item["wall_time_sec"]) for item in operations if item["status"] == "success"]
    successful_rss = [
        int(item["maximum_resident_set_bytes"])
        for item in operations
        if item["status"] == "success" and item.get("maximum_resident_set_bytes") is not None
    ]
    operational_summary = {
        "schema_version": 1,
        "execution_model": "one_process_per_scan",
        "scan_count": len(entries),
        "successful_scan_count": len(successful_latencies),
        "failed_scan_count": len(failures),
        "job_success_rate": round(len(successful_latencies) / len(entries), 6) if entries else None,
        "p95_latency_sec": _percentile(successful_latencies, 0.95),
        "maximum_resident_set_bytes": max(successful_rss) if successful_rss else None,
        "cases": operations,
    }
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "review_output_dir": str(review_output_dir),
        "route_mode": "apply",
        "validation_target": {"known": "visible observed conventional bend count"},
        "inference_mode": "blind_ply_only_f1",
        "inference_contract": {
            "mode": "strict_blind_scan_only",
            "part_identifier_exposed": False,
            "expectation_exposed": False,
            "manual_truth_exposed": False,
            "promotion_evidence_exposed": False,
        },
        "aggregate": _aggregate_validation(rows),
        "typed_release_metrics": _typed_release_metrics(rows),
        "operational_summary": operational_summary,
        "results": rows,
        "failures": failures,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "operational_summary.json", operational_summary)
    _write_json(review_output_dir / "VALIDATION_SUMMARY.json", summary)
    _write_review_markdown(review_output_dir, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--review-output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.refresh_existing:
        summary_path = args.output_dir.resolve() / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["typed_release_metrics"] = _typed_release_metrics(summary.get("results") or ())
        _write_json(summary_path, summary)
        _write_json(args.review_output_dir.resolve() / "VALIDATION_SUMMARY.json", summary)
        _write_review_markdown(args.review_output_dir.resolve(), summary)
        print(json.dumps({"typed_release_metrics": summary["typed_release_metrics"]}, indent=2))
        return 0
    summary = run_slice(
        slice_payload=json.loads(args.slice_json.read_text(encoding="utf-8")),
        output_dir=args.output_dir.resolve(),
        review_output_dir=args.review_output_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"aggregate": summary["aggregate"], "operational_summary": summary["operational_summary"]}, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
