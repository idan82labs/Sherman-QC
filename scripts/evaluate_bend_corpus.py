#!/opt/homebrew/bin/python3.11
"""
Serial regression evaluator for the bend corpus.

Runs the real progressive bend inspection pipeline across the organized corpus and
stores per-scan outputs in a dedicated evaluation folder so future algorithm
changes can be compared against the same baseline.
"""

from __future__ import annotations

import argparse
import copy
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

from bend_unary_model import load_unary_models, score_case_payload  # noqa: E402
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


def _continuity_inferred_count(matches: List[Dict[str, Any]]) -> int:
    return sum(1 for match in matches if bool(match.get("continuity_inferred")))


def _status_priority(status: Optional[str]) -> int:
    return {
        "PASS": 3,
        "WARNING": 2,
        "FAIL": 1,
        "NOT_DETECTED": 0,
    }.get(str(status or "").upper(), 0)


def _recompute_summary(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(matches)
    detected = sum(1 for match in matches if str(match.get("status") or "").upper() != "NOT_DETECTED")
    passed = sum(1 for match in matches if str(match.get("status") or "").upper() == "PASS")
    failed = sum(1 for match in matches if str(match.get("status") or "").upper() == "FAIL")
    warnings = sum(1 for match in matches if str(match.get("status") or "").upper() == "WARNING")
    observed = sum(
        1
        for match in matches
        if str(match.get("observability_state") or "").upper() in {"OBSERVED_FORMED", "OBSERVED_NOT_FORMED"}
        or (
            str(match.get("status") or "").upper() in {"PASS", "FAIL", "WARNING"}
            and str(match.get("observability_state") or "").upper() == "UNOBSERVED"
        )
    )
    partial_observed = sum(
        1 for match in matches if str(match.get("observability_state") or "").upper() == "PARTIALLY_OBSERVED"
    )
    unobserved = sum(
        1
        for match in matches
        if str(match.get("observability_state") or "").upper() == "UNOBSERVED"
        and str(match.get("status") or "").upper() == "NOT_DETECTED"
    )
    return {
        "total_bends": total,
        "detected": detected,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "progress_pct": round((detected / total) * 100.0, 3) if total else 0.0,
        "completed_bends": detected,
        "completed_in_spec": passed,
        "completed_out_of_spec": failed + warnings,
        "remaining_bends": max(0, total - detected),
        "observed_bends": observed,
        "partially_observed_bends": partial_observed,
        "unobserved_bends": unobserved,
    }


def _should_apply_sequence_continuity(
    previous_item: Optional[Dict[str, Any]],
    current_item: Dict[str, Any],
    previous_scan_record: Optional[Dict[str, Any]],
    current_scan_record: Dict[str, Any],
) -> bool:
    if not previous_item or "matches" not in previous_item or "metrics" not in previous_item:
        return False
    if str(current_item.get("expectation", {}).get("state") or "").lower() != "partial":
        return False
    if str(previous_item.get("expectation", {}).get("state") or "").lower() != "partial":
        return False
    if not bool(previous_scan_record.get("sequence_confident")):
        return False
    if not bool(current_scan_record.get("sequence_confident")):
        return False
    prev_sequence = previous_scan_record.get("sequence")
    curr_sequence = current_scan_record.get("sequence")
    if prev_sequence is None or curr_sequence is None:
        return False
    return float(curr_sequence) > float(prev_sequence)


def _carry_forward_previous_match(
    previous_match: Dict[str, Any],
    current_match: Dict[str, Any],
    previous_scan_name: str,
    current_scan_name: str,
) -> Dict[str, Any]:
    carried = copy.deepcopy(previous_match)
    prev_conf = float(previous_match.get("confidence") or 0.0)
    prev_evidence = float(previous_match.get("local_evidence_score") or 0.0)
    current_evidence = float(current_match.get("local_evidence_score") or 0.0)
    carried["continuity_inferred"] = True
    carried["continuity_source_scan"] = previous_scan_name
    carried["continuity_target_scan"] = current_scan_name
    carried["physical_completion_state"] = previous_match.get("physical_completion_state") or "FORMED"
    carried["measurement_mode"] = "sequence_continuity"
    carried["measurement_method"] = "carry_forward_previous_partial"
    carried["assignment_source"] = "SEQUENCE_CONTINUITY"
    carried["assignment_confidence"] = round(min(0.85, max(prev_conf, prev_evidence)), 3)
    carried["assignment_candidate_id"] = current_match.get("assignment_candidate_id") or "null_candidate"
    carried["assignment_candidate_kind"] = current_match.get("assignment_candidate_kind") or "NULL"
    carried["assignment_candidate_score"] = round(float(current_match.get("assignment_candidate_score") or 0.0), 3)
    carried["assignment_null_score"] = round(float(current_match.get("assignment_null_score") or 0.0), 3)
    carried["assignment_candidate_count"] = int(current_match.get("assignment_candidate_count") or 0)
    carried["observability_state"] = current_match.get("observability_state") or "UNOBSERVED"
    carried["observability_confidence"] = round(
        max(float(current_match.get("observability_confidence") or 0.0), min(0.75, max(prev_conf, prev_evidence) * 0.7)),
        3,
    )
    carried["visibility_score"] = round(float(current_match.get("visibility_score") or 0.0), 3)
    carried["visibility_score_source"] = current_match.get("visibility_score_source") or "heuristic_local_visibility_v0"
    carried["surface_visibility_ratio"] = round(float(current_match.get("surface_visibility_ratio") or 0.0), 3)
    carried["local_support_score"] = round(float(current_match.get("local_support_score") or 0.0), 3)
    carried["side_balance_score"] = round(float(current_match.get("side_balance_score") or 0.0), 3)
    carried["local_point_count"] = int(current_match.get("local_point_count") or 0)
    carried["local_evidence_score"] = round(max(current_evidence, min(0.85, prev_evidence * 0.75)), 3)
    context = dict(previous_match.get("measurement_context") or {})
    context.update(
        {
            "continuity_inferred": True,
            "carried_from_scan": previous_scan_name,
            "current_scan": current_scan_name,
            "current_observability_state": current_match.get("observability_state"),
            "current_local_evidence_score": current_match.get("local_evidence_score"),
        }
    )
    carried["measurement_context"] = context
    return carried


def _apply_sequence_continuity_prior(
    previous_item: Dict[str, Any],
    current_item: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    previous_matches = {
        str(match.get("bend_id")): match
        for match in (previous_item.get("matches") or [])
        if match.get("bend_id")
    }
    updated_matches: List[Dict[str, Any]] = []
    carried_ids: List[str] = []
    previous_scan_name = str(previous_item.get("scan_name") or "")
    current_scan_name = str(current_item.get("scan_name") or "")

    for current_match in current_item.get("matches") or []:
        bend_id = str(current_match.get("bend_id") or "")
        previous_match = previous_matches.get(bend_id)
        if previous_match is None:
            updated_matches.append(current_match)
            continue
        previous_status = str(previous_match.get("status") or "").upper()
        current_status = str(current_match.get("status") or "").upper()
        current_observability = str(current_match.get("observability_state") or "").upper()
        current_evidence = float(current_match.get("local_evidence_score") or 0.0)
        previous_confidence = float(previous_match.get("confidence") or 0.0)
        previous_evidence = float(previous_match.get("local_evidence_score") or 0.0)

        should_carry = (
            _status_priority(previous_status) > 0
            and current_status == "NOT_DETECTED"
            and current_observability in {"UNOBSERVED", "PARTIALLY_OBSERVED"}
            and current_evidence <= 0.25
            and max(previous_confidence, previous_evidence) >= 0.5
        )
        if should_carry:
            updated_matches.append(
                _carry_forward_previous_match(previous_match, current_match, previous_scan_name, current_scan_name)
            )
            carried_ids.append(bend_id)
        else:
            updated_matches.append(current_match)

    if not carried_ids:
        return current_item, []

    updated_item = copy.deepcopy(current_item)
    updated_item["matches"] = updated_matches
    updated_item["summary"] = _recompute_summary(updated_matches)
    updated_item["metrics"] = _score_scan(updated_item, updated_item.get("expectation") or {})
    updated_item["metrics"]["continuity_inferred_bends"] = len(carried_ids)
    updated_item.setdefault("warnings", [])
    updated_item["warnings"] = list(updated_item["warnings"]) + [
        f"Sequence continuity prior carried forward bends from {previous_scan_name}: {', '.join(carried_ids)}"
    ]
    return updated_item, carried_ids


def _ordered_scan_records(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    scans = list(metadata.get("scan_files") or [])

    def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, float, int]:
        idx, scan = item
        state = str(scan.get("state") or "").lower()
        sequence_confident = bool(scan.get("sequence_confident"))
        sequence = scan.get("sequence")
        try:
            sequence_value = float(sequence)
        except (TypeError, ValueError):
            sequence_value = float(idx)
        state_rank = 2 if state == "full" else 0 if state == "partial" else 1
        confidence_rank = 0 if sequence_confident else 1
        return (state_rank, confidence_rank, sequence_value, idx)

    return [scan for _, scan in sorted(enumerate(scans), key=_sort_key)]


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
        "continuity_inferred_bends": _continuity_inferred_count(report_dict.get("matches") or []),
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


def _structured_score_scan(
    report_dict: Dict[str, Any],
    expectation: Dict[str, Any],
) -> Dict[str, Any]:
    structured = (report_dict.get("structured_context") or {}).get("count_posterior") or {}
    median_completed = structured.get("median_completed_bends")
    map_completed = structured.get("map_completed_bends")
    mean_completed = structured.get("mean_completed_bends")
    expected_total = expectation.get("expected_total_bends")
    expected_completed = expectation.get("expected_completed_bends")
    metrics: Dict[str, Any] = {
        "completed_bends_median": median_completed,
        "completed_bends_map": map_completed,
        "completed_bends_mean": mean_completed,
    }
    if median_completed is None:
        return metrics
    median_completed = int(median_completed)
    if expected_total is not None:
        metrics["expected_total_error_median"] = median_completed - int(expected_total)
        metrics["completed_within_total_median"] = median_completed <= int(expected_total)
    if expected_completed is not None:
        metrics["expected_completed_error_median"] = median_completed - int(expected_completed)
        metrics["completed_matches_expected_median"] = median_completed == int(expected_completed)
    return metrics


def _detected_bend_ids(item: Dict[str, Any]) -> set[str]:
    return {
        str(match.get("bend_id"))
        for match in (item.get("matches") or [])
        if str(match.get("status") or "").upper() != "NOT_DETECTED" and match.get("bend_id")
    }


def _staged_progression_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    by_part: Dict[str, List[Dict[str, Any]]] = {}
    for item in all_results:
        if "metrics" not in item:
            continue
        if str(item.get("expectation", {}).get("state") or "").lower() != "partial":
            continue
        if not bool(item.get("sequence_confident")):
            continue
        sequence = item.get("scan_sequence")
        if sequence is None:
            continue
        by_part.setdefault(str(item.get("part_key") or ""), []).append(item)

    pair_count = 0
    monotonic_count = 0
    retained_total = 0
    retained_possible = 0
    part_count = 0
    for items in by_part.values():
        ordered = sorted(items, key=lambda item: float(item.get("scan_sequence") or 0.0))
        if len(ordered) < 2:
            continue
        part_count += 1
        for previous, current in zip(ordered, ordered[1:]):
            pair_count += 1
            if int(current.get("summary", {}).get("completed_bends") or 0) >= int(previous.get("summary", {}).get("completed_bends") or 0):
                monotonic_count += 1
            previous_ids = _detected_bend_ids(previous)
            current_ids = _detected_bend_ids(current)
            retained_total += len(previous_ids & current_ids)
            retained_possible += len(previous_ids)

    return {
        "staged_sequence_part_count": part_count,
        "staged_monotonic_pair_count": pair_count,
        "staged_monotonic_pair_rate": round(monotonic_count / pair_count, 3) if pair_count else None,
        "staged_retained_bend_rate": round(retained_total / retained_possible, 3) if retained_possible else None,
    }


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


def _latest_unary_model_path() -> Optional[Path]:
    latest = ROOT / "output" / "unary_models" / "latest.json"
    if not latest.exists():
        return None
    try:
        payload = _load_json(latest)
    except Exception:
        return None
    candidate = payload.get("model_path") or payload.get("path")
    if not candidate:
        latest_dir = payload.get("latest_bootstrap_dir")
        if latest_dir:
            model_path = Path(str(latest_dir)) / "bend_unary_models.joblib"
            return model_path if model_path.exists() else None
        return None
    model_path = Path(str(candidate))
    if model_path.is_dir():
        model_path = model_path / "bend_unary_models.joblib"
    return model_path if model_path.exists() else None


def _annotate_structured_predictions(
    item: Dict[str, Any],
    expectation: Dict[str, Any],
    unary_bundle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if unary_bundle is None or "matches" not in item:
        return item
    annotated = score_case_payload(copy.deepcopy(item), unary_bundle)
    annotated["structured_metrics"] = _structured_score_scan(annotated, expectation)
    return annotated


def evaluate_part(
    part_record: Dict[str, Any],
    output_dir: Path,
    skip_existing: bool,
    timeout_sec: int,
    unary_bundle: Optional[Dict[str, Any]],
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
    previous_sequence_item: Optional[Dict[str, Any]] = None
    previous_sequence_scan_record: Optional[Dict[str, Any]] = None

    for scan_record in _ordered_scan_records(metadata):
        scan_name = str(scan_record.get("name"))
        scan_path = str(scan_record.get("path"))
        result_path = part_out_dir / f"{Path(scan_name).stem}.json"
        expectation = _scan_expectation(scan_record, labels_map, expected_total_bends)
        if skip_existing and result_path.exists():
            existing = _load_json(result_path)
            if "metrics" in existing:
                existing.setdefault("metrics", _score_scan(existing, expectation))
                existing = _annotate_structured_predictions(existing, expectation, unary_bundle)
                result_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            scan_results.append(existing)
            continue

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
            if _should_apply_sequence_continuity(
                previous_sequence_item,
                item,
                previous_sequence_scan_record,
                scan_record,
            ):
                item, carried_ids = _apply_sequence_continuity_prior(previous_sequence_item, item)
                if carried_ids:
                    result_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            item = _annotate_structured_predictions(item, expectation, unary_bundle)
            result_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            item["scan_sequence"] = scan_record.get("sequence")
            item["sequence_confident"] = bool(scan_record.get("sequence_confident"))
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
                        "continuity_inferred_bends": item["metrics"].get("continuity_inferred_bends", 0),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if str(expectation.get("state") or "").lower() == "partial" and bool(scan_record.get("sequence_confident")):
                previous_sequence_item = item
                previous_sequence_scan_record = scan_record
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

    def _structured_mae(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [
            abs(float((r.get("structured_metrics") or {})[key]))
            for r in rows
            if (r.get("structured_metrics") or {}).get(key) is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    staged = _staged_progression_metrics(completed_metric_results)
    structured_rows = [r for r in completed_metric_results if r.get("structured_metrics")]
    structured_with_total = [
        r for r in structured_rows if (r.get("structured_metrics") or {}).get("expected_total_error_median") is not None
    ]
    structured_with_completed = [
        r for r in structured_rows if (r.get("structured_metrics") or {}).get("expected_completed_error_median") is not None
    ]
    structured_full_scans = [r for r in full_scans if (r.get("structured_metrics") or {}).get("completed_bends_median") is not None]
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
        "partial_mae_vs_expected_total_completed": _mae(partial_scans, "expected_total_error"),
        "partial_mae_vs_expected_completed": _mae(partial_scans, "expected_completed_error"),
        "structured_scans": len(structured_rows),
        "structured_mae_vs_expected_total_completed": _structured_mae(
            structured_with_total, "expected_total_error_median"
        ),
        "structured_mae_vs_expected_completed": _structured_mae(
            structured_with_completed, "expected_completed_error_median"
        ),
        "structured_partial_mae_vs_expected_total_completed": _structured_mae(
            [r for r in partial_scans if r.get("structured_metrics")], "expected_total_error_median"
        ),
        "structured_partial_mae_vs_expected_completed": _structured_mae(
            [r for r in partial_scans if r.get("structured_metrics")], "expected_completed_error_median"
        ),
        "full_scan_exact_completion_rate": (
            round(
                sum(1 for r in full_scans if r["metrics"].get("completed_matches_expected")) / len(full_scans),
                3,
            )
            if full_scans
            else None
        ),
        "structured_full_scan_exact_completion_rate": (
            round(
                sum(1 for r in structured_full_scans if (r.get("structured_metrics") or {}).get("completed_matches_expected_median"))
                / len(structured_full_scans),
                3,
            )
            if structured_full_scans
            else None
        ),
        "continuity_inferred_scan_count": sum(
            1 for r in completed_metric_results if int(r["metrics"].get("continuity_inferred_bends") or 0) > 0
        ),
        "continuity_inferred_bends_total": sum(
            int(r["metrics"].get("continuity_inferred_bends") or 0) for r in completed_metric_results
        ),
        **staged,
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
        f"- Partial MAE vs expected total completed bends: `{aggregate['partial_mae_vs_expected_total_completed']}`",
        f"- Partial MAE vs expected completed bends: `{aggregate['partial_mae_vs_expected_completed']}`",
        f"- Structured MAE vs expected total completed bends: `{aggregate['structured_mae_vs_expected_total_completed']}`",
        f"- Structured MAE vs expected completed bends: `{aggregate['structured_mae_vs_expected_completed']}`",
        f"- Structured partial MAE vs expected total completed bends: `{aggregate['structured_partial_mae_vs_expected_total_completed']}`",
        f"- Structured partial MAE vs expected completed bends: `{aggregate['structured_partial_mae_vs_expected_completed']}`",
        f"- Full-scan exact completion rate: `{aggregate['full_scan_exact_completion_rate']}`",
        f"- Structured full-scan exact completion rate: `{aggregate['structured_full_scan_exact_completion_rate']}`",
        f"- Continuity-inferred scans: `{aggregate['continuity_inferred_scan_count']}`",
        f"- Continuity-inferred bends: `{aggregate['continuity_inferred_bends_total']}`",
        f"- Staged monotonic pair rate: `{aggregate['staged_monotonic_pair_rate']}`",
        f"- Staged retained bend rate: `{aggregate['staged_retained_bend_rate']}`",
        "",
        "| Part | Scan | State | CAD strategy | Completed | Structured K~ | In spec | Remaining | Continuity | Expected total | Expected completed | Observability | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
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
            f"{metrics.get('cad_strategy', '-') or '-'} | {summary.get('completed_bends', 0)} | "
            f"{(item.get('structured_metrics') or {}).get('completed_bends_median', '-')} | "
            f"{summary.get('completed_in_spec', 0)} | {summary.get('remaining_bends', 0)} | "
            f"{metrics.get('continuity_inferred_bends', 0)} | {metrics.get('expected_total_bends', '-')} | {metrics.get('expected_completed_bends', '-')} | {observability or '-'} | {'; '.join(error_parts) or '-'} |"
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
    parser.add_argument("--unary-model", default=None)
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
    unary_model_path = Path(args.unary_model) if args.unary_model else _latest_unary_model_path()
    unary_bundle = None
    if unary_model_path and unary_model_path.exists():
        unary_bundle = load_unary_models(unary_model_path)

    for part in ready_parts:
        scan_results, part_warnings = evaluate_part(
            part_record=part,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            timeout_sec=args.timeout_sec,
            unary_bundle=unary_bundle,
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
