#!/opt/homebrew/bin/python3.11
"""
Serial regression evaluator for the bend corpus.

Runs the real progressive bend inspection pipeline across the organized corpus and
stores per-scan outputs in a dedicated evaluation folder so future algorithm
changes can be compared against the same baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")


ROOT = Path("/Users/idant/82Labs/Sherman QC/Full Project")
CORPUS_ROOT = ROOT / "data" / "bend_corpus"

import sys

sys.path.insert(0, str(ROOT / "backend"))

from dimension_parser import parse_dimension_file  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preferred_cad_file(metadata: Dict[str, Any]) -> Optional[str]:
    cad_files = metadata.get("cad_files") or []
    if not cad_files:
        return None
    ranked = sorted(
        cad_files,
        key=lambda item: (
            0 if Path(item["path"]).suffix.lower() in {".step", ".stp"} else 1,
            item.get("name", ""),
        ),
    )
    return ranked[0]["path"]


def _load_labels(labels_path: Path) -> Dict[str, Any]:
    if not labels_path.exists():
        return {}
    try:
        return _load_json(labels_path)
    except Exception:
        return {}


def _scan_label_map(labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("filename")): item
        for item in (labels.get("scans") or [])
        if isinstance(item, dict) and item.get("filename")
    }


def _scan_expectation(
    scan_record: Dict[str, Any],
    labels_map: Dict[str, Dict[str, Any]],
    expected_total_bends: Optional[int],
) -> Dict[str, Any]:
    scan_name = str(scan_record.get("name"))
    label = labels_map.get(scan_name, {})
    state = str(label.get("state") or scan_record.get("state") or "unknown").lower()
    completed_bends = label.get("completed_bends")
    if completed_bends is None and state == "full" and expected_total_bends is not None:
        completed_bends = int(expected_total_bends)
    return {
        "state": state,
        "expected_total_bends": expected_total_bends,
        "expected_completed_bends": completed_bends,
        "completed_bend_ids": label.get("completed_bend_ids") or [],
        "all_completed_in_spec": label.get("all_completed_in_spec"),
        "comments": label.get("comments") or "",
    }


def _spec_bend_angles(metadata: Dict[str, Any]) -> List[float]:
    spec_files = metadata.get("spec_files") or []
    for spec in spec_files:
        try:
            result = parse_dimension_file(spec["path"])
        except Exception:
            continue
        if result.success and result.bend_angles:
            return [float(dim.value) for dim in result.bend_angles if dim.value is not None]
    return []


def _spec_schedule_type(metadata: Dict[str, Any]) -> Optional[str]:
    spec_analysis = metadata.get("spec_analysis")
    if not isinstance(spec_analysis, dict):
        return None
    value = str(spec_analysis.get("schedule_type") or "").strip()
    return value or None


def _observability_breakdown(matches: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for match in matches:
        key = str(match.get("observability_state") or "UNKNOWN").upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _score_scan(
    report_dict: Dict[str, Any],
    expectation: Dict[str, Any],
) -> Dict[str, Any]:
    summary = report_dict.get("summary") or {}
    details = report_dict.get("details") or {}
    completed = int(summary.get("completed_bends") or 0)
    expected_total = expectation.get("expected_total_bends")
    expected_completed = expectation.get("expected_completed_bends")

    metrics: Dict[str, Any] = {
        "completed_bends": completed,
        "completed_in_spec": int(summary.get("completed_in_spec") or 0),
        "completed_out_of_spec": int(summary.get("completed_out_of_spec") or 0),
        "remaining_bends": int(summary.get("remaining_bends") or 0),
        "observability_breakdown": _observability_breakdown(report_dict.get("matches") or []),
        "expected_total_bends": expected_total,
        "expected_completed_bends": expected_completed,
        "cad_strategy": details.get("cad_strategy"),
        "expected_bend_count_used": details.get("expected_bend_count"),
    }
    if expected_total is not None:
        metrics["expected_total_error"] = completed - int(expected_total)
        metrics["completed_within_total"] = completed <= int(expected_total)
    if expected_completed is not None:
        metrics["expected_completed_error"] = completed - int(expected_completed)
        metrics["completed_matches_expected"] = completed == int(expected_completed)
    return metrics


CASE_RUNNER = ROOT / "scripts" / "evaluate_bend_case.py"


def _run_case_subprocess(
    *,
    part_key: str,
    cad_path: str,
    scan_path: str,
    output_json: Path,
    expectation: Dict[str, Any],
    spec_bend_angles: List[float],
    spec_schedule_type: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    cmd = [
        "/opt/homebrew/bin/python3.11",
        str(CASE_RUNNER),
        "--part-key",
        part_key,
        "--cad-path",
        cad_path,
        "--scan-path",
        scan_path,
        "--output-json",
        str(output_json),
        "--state",
        str(expectation.get("state") or "unknown"),
        "--expected-total",
        str(expectation.get("expected_total_bends")),
        "--expected-completed",
        str(expectation.get("expected_completed_bends")),
        "--spec-bend-angles-json",
        json.dumps(spec_bend_angles),
        "--spec-schedule-type",
        str(spec_schedule_type or ""),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(60, int(timeout_sec)),
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"case runner exited {completed.returncode}").strip())
    if not output_json.exists():
        raise RuntimeError("case runner finished without writing output json")
    payload = _load_json(output_json)
    if "summary" in payload and "metrics" not in payload:
        payload["metrics"] = _score_scan(payload, expectation)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def evaluate_part(
    part_record: Dict[str, Any],
    output_dir: Path,
    skip_existing: bool,
    timeout_sec: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    metadata_path = Path(part_record["metadata_path"])
    labels_path = Path(part_record["labels_template_path"])
    metadata = _load_json(metadata_path)
    labels = _load_labels(labels_path)
    labels_map = _scan_label_map(labels)
    part_key = str(part_record["part_key"])
    expected_total_bends = labels.get("expected_total_bends")
    if expected_total_bends is None:
        expected_total_bends = metadata.get("expected_bends_override")

    cad_path = _preferred_cad_file(metadata)
    if cad_path is None:
        return [], [f"{part_key}: no CAD file available"]
    spec_bend_angles = _spec_bend_angles(metadata)
    spec_schedule_type = _spec_schedule_type(metadata)

    part_out_dir = output_dir / part_key
    part_out_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: List[str] = []
    scan_results: List[Dict[str, Any]] = []

    for scan_record in metadata.get("scan_files") or []:
        scan_name = str(scan_record.get("name"))
        scan_path = str(scan_record.get("path"))
        result_path = part_out_dir / f"{Path(scan_name).stem}.json"
        if skip_existing and result_path.exists():
            scan_results.append(_load_json(result_path))
            continue

        expectation = _scan_expectation(scan_record, labels_map, expected_total_bends)
        print(
            json.dumps(
                {
                    "event": "scan_start",
                    "part_key": part_key,
                    "scan_name": scan_name,
                    "state": expectation.get("state"),
                    "expected_total_bends": expected_total_bends,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            item = _run_case_subprocess(
                part_key=part_key,
                cad_path=cad_path,
                scan_path=scan_path,
                output_json=result_path,
                expectation=expectation,
                spec_bend_angles=spec_bend_angles,
                spec_schedule_type=spec_schedule_type,
                timeout_sec=timeout_sec,
            )
            scan_results.append(item)
            print(
                json.dumps(
                    {
                        "event": "scan_done",
                        "part_key": part_key,
                        "scan_name": scan_name,
                        "completed_bends": item["summary"].get("completed_bends"),
                        "remaining_bends": item["summary"].get("remaining_bends"),
                        "observability": item["metrics"].get("observability_breakdown"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "part_key": part_key,
                "scan_name": scan_name,
                "scan_path": scan_path,
                "error": str(exc),
                "expectation": expectation,
            }
            result_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
            scan_results.append(failure)
            warnings_out.append(f"{part_key}/{scan_name}: {exc}")
            print(
                json.dumps(
                    {
                        "event": "scan_failed",
                        "part_key": part_key,
                        "scan_name": scan_name,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        finally:
            gc.collect()

    return scan_results, warnings_out


def _aggregate(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed_metric_results = [r for r in all_results if "metrics" in r]
    with_total = [r for r in completed_metric_results if r["metrics"].get("expected_total_bends") is not None]
    with_completed = [r for r in completed_metric_results if r["metrics"].get("expected_completed_bends") is not None]
    full_scans = [r for r in completed_metric_results if str(r.get("expectation", {}).get("state")) == "full"]
    partial_scans = [r for r in completed_metric_results if str(r.get("expectation", {}).get("state")) == "partial"]

    def _mae(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [abs(float(r["metrics"][key])) for r in rows if r["metrics"].get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    return {
        "scans_processed": len(all_results),
        "successful_scans": len(completed_metric_results),
        "failed_scans": sum(1 for r in all_results if "metrics" not in r),
        "scans_with_expected_total": len(with_total),
        "scans_with_expected_completed": len(with_completed),
        "full_scans": len(full_scans),
        "partial_scans": len(partial_scans),
        "mae_vs_expected_total_completed": _mae(with_total, "expected_total_error"),
        "mae_vs_expected_completed": _mae(with_completed, "expected_completed_error"),
        "full_scan_exact_completion_rate": (
            round(
                sum(1 for r in full_scans if r["metrics"].get("completed_matches_expected")) / len(full_scans),
                3,
            )
            if full_scans
            else None
        ),
    }


def _write_markdown(output_dir: Path, aggregate: Dict[str, Any], all_results: List[Dict[str, Any]], warnings_out: List[str]) -> None:
    lines = [
        "# Bend Corpus Regression",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Scans processed: `{aggregate['scans_processed']}`",
        f"- Successful scans: `{aggregate['successful_scans']}`",
        f"- Failed scans: `{aggregate['failed_scans']}`",
        f"- MAE vs expected total completed bends: `{aggregate['mae_vs_expected_total_completed']}`",
        f"- MAE vs expected completed bends: `{aggregate['mae_vs_expected_completed']}`",
        f"- Full-scan exact completion rate: `{aggregate['full_scan_exact_completion_rate']}`",
        "",
        "| Part | Scan | State | CAD strategy | Completed | In spec | Remaining | Expected total | Expected completed | Observability | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in all_results:
        if "metrics" not in item:
            lines.append(
                f"| {item['part_key']} | {item['scan_name']} | {item.get('expectation', {}).get('state', '-')} | - | - | - | - | - | - | - | {item.get('error', 'failed')} |"
            )
            continue
        metrics = item["metrics"]
        summary = item["summary"]
        observability = ",".join(f"{k}:{v}" for k, v in sorted(metrics["observability_breakdown"].items()))
        error_parts = []
        if metrics.get("expected_total_error") is not None:
            error_parts.append(f"Δtot={metrics['expected_total_error']:+}")
        if metrics.get("expected_completed_error") is not None:
            error_parts.append(f"Δcmp={metrics['expected_completed_error']:+}")
        lines.append(
            f"| {item['part_key']} | {item['scan_name']} | {item.get('expectation', {}).get('state', '-')} | "
            f"{metrics.get('cad_strategy', '-') or '-'} | {summary.get('completed_bends', 0)} | {summary.get('completed_in_spec', 0)} | {summary.get('remaining_bends', 0)} | "
            f"{metrics.get('expected_total_bends', '-')} | {metrics.get('expected_completed_bends', '-')} | {observability or '-'} | {'; '.join(error_parts) or '-'} |"
        )
    if warnings_out:
        lines.extend(["", "## Failures", ""])
        for warning in warnings_out:
            lines.append(f"- {warning}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the bend corpus with the progressive inspection pipeline.")
    parser.add_argument("--corpus", default=str(CORPUS_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=420)
    args = parser.parse_args()

    corpus_root = Path(args.corpus)
    ready_path = corpus_root / "ready_regression.json"
    ready_parts = _load_json(ready_path)
    requested_parts = {p.strip() for p in args.part if p and p.strip()}
    if requested_parts:
        ready_parts = [p for p in ready_parts if str(p.get("part_key")) in requested_parts]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else (corpus_root / "evaluation" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings_out: List[str] = []
    all_results: List[Dict[str, Any]] = []

    for part in ready_parts:
        scan_results, part_warnings = evaluate_part(
            part_record=part,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            timeout_sec=args.timeout_sec,
        )
        all_results.extend(scan_results)
        warnings_out.extend(part_warnings)

    aggregate = _aggregate(all_results)
    (output_dir / "summary.json").write_text(
        json.dumps({"aggregate": aggregate, "results": all_results, "warnings": warnings_out}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(output_dir, aggregate, all_results, warnings_out)
    latest = corpus_root / "evaluation" / "latest.json"
    latest.write_text(json.dumps({"path": str(output_dir / 'summary.json')}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
