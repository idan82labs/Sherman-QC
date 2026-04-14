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
import hashlib
import json
import os
import re
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

POSITIONAL_WARN_MM = 2.5


ROOT = Path("/Users/idant/82Labs/Sherman QC/Full Project-qc-clean")
CORPUS_ROOT = ROOT / "data" / "bend_corpus"

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from bend_unary_model import load_unary_models, score_case_payload  # noqa: E402
from dimension_parser import parse_dimension_file  # noqa: E402
from domains.bend.services.runtime_semantics import (  # noqa: E402
    is_explicit_observability_evidence,
    normalize_observability_state,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_scan_output_name(scan_name: str) -> str:
    raw_name = Path(str(scan_name)).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
    if not safe_name:
        safe_name = "scan"
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}--{digest}.json"


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
        key = normalize_observability_state(
            match.get("observability_state"),
            physical_completion_state=match.get("physical_completion_state"),
            status=match.get("status"),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _countable_primary_match(match: Dict[str, Any]) -> bool:
    return bool(match.get("countable_in_regression", True)) and str(match.get("feature_type") or "").upper() not in {
        "ROLLED_SECTION",
        "PROCESS_FEATURE",
    }


def _guard_process_match(match: Dict[str, Any]) -> bool:
    return not _countable_primary_match(match)


def _completion_state(match: Dict[str, Any]) -> str:
    if match.get("completion_state"):
        return str(match.get("completion_state"))
    return normalize_observability_state(
        match.get("observability_state"),
        physical_completion_state=match.get("physical_completion_state"),
        status=match.get("status"),
    )


def _metrology_state(match: Dict[str, Any]) -> str:
    if match.get("metrology_state"):
        return str(match.get("metrology_state"))
    status = str(match.get("status") or "").upper()
    if status == "PASS":
        return "IN_TOL"
    if status in {"FAIL", "WARNING"}:
        return "OUT_OF_TOL"
    return "UNMEASURABLE"


def _position_state(match: Dict[str, Any]) -> str:
    if match.get("positional_state"):
        return str(match.get("positional_state"))
    consistency = str(((match.get("heatmap_consistency") or {}).get("status") or "")).upper()
    if consistency == "CONTRADICTED":
        return "MISLOCATED"
    if consistency == "SUPPORTED":
        return "ON_POSITION"
    center_dev = match.get("line_center_deviation_mm")
    if center_dev is not None:
        return "MISLOCATED" if abs(float(center_dev)) > 2.5 else "ON_POSITION"
    return "UNKNOWN_POSITION"


def _correspondence_state(match: Dict[str, Any]) -> str:
    if match.get("correspondence_state"):
        return str(match.get("correspondence_state"))
    status = str(match.get("status") or "").upper()
    candidate_count = int(match.get("correspondence_candidate_count") or match.get("assignment_candidate_count") or 0)
    confidence = float(match.get("correspondence_confidence") or match.get("assignment_confidence") or 0.0)
    raw_margin = match.get("correspondence_margin")
    if raw_margin is None:
        raw_margin = float(match.get("assignment_candidate_score") or 0.0) - float(match.get("assignment_null_score") or 0.0)
    margin = float(raw_margin or 0.0)
    source = str(match.get("correspondence_source") or match.get("assignment_source") or "").upper()
    if status == "NOT_DETECTED" and candidate_count <= 0:
        return "UNRESOLVED"
    if candidate_count <= 0:
        return "AMBIGUOUS" if source not in {"", "NONE"} and confidence > 0.0 else "UNRESOLVED"
    if candidate_count == 1 and confidence >= 0.65 and margin >= 0.15:
        return "CONFIDENT"
    return "AMBIGUOUS"


def _observability_state_internal(match: Dict[str, Any]) -> str:
    if match.get("observability_state_internal"):
        return str(match.get("observability_state_internal"))
    completion = _completion_state(match) != "UNKNOWN"
    metrology = _metrology_state(match) != "UNMEASURABLE"
    position = _position_state(match) != "UNKNOWN_POSITION"
    if completion and metrology and position:
        return "SUFFICIENT"
    if completion or metrology or position:
        return "PARTIAL"
    return "INSUFFICIENT"


def _continuity_inferred_count(matches: List[Dict[str, Any]]) -> int:
    return sum(1 for match in matches if bool(match.get("continuity_inferred")))


def _retained_from_previous_partial_count(matches: List[Dict[str, Any]]) -> int:
    return sum(1 for match in matches if bool(match.get("retained_from_previous_partial")))


def _status_priority(status: Optional[str]) -> int:
    return {
        "PASS": 3,
        "WARNING": 2,
        "FAIL": 1,
        "NOT_DETECTED": 0,
    }.get(str(status or "").upper(), 0)


def _recompute_summary(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    countable_matches = [match for match in matches if _countable_primary_match(match)]
    total = len(countable_matches)
    detected = sum(1 for match in countable_matches if str(match.get("status") or "").upper() != "NOT_DETECTED")
    passed = sum(1 for match in countable_matches if str(match.get("status") or "").upper() == "PASS")
    failed = sum(1 for match in countable_matches if str(match.get("status") or "").upper() == "FAIL")
    warnings = sum(1 for match in countable_matches if str(match.get("status") or "").upper() == "WARNING")
    observed = sum(
        1
        for match in countable_matches
        if is_explicit_observability_evidence(
            match.get("observability_state"),
            physical_completion_state=match.get("physical_completion_state"),
            status=match.get("status"),
        )
    )
    partial_observed = sum(
        1
        for match in countable_matches
        if normalize_observability_state(
            match.get("observability_state"),
            physical_completion_state=match.get("physical_completion_state"),
            status=match.get("status"),
        ) == "UNKNOWN"
        and str(match.get("status") or "").upper() != "NOT_DETECTED"
    )
    unobserved = sum(
        1
        for match in countable_matches
        if normalize_observability_state(
            match.get("observability_state"),
            physical_completion_state=match.get("physical_completion_state"),
            status=match.get("status"),
        ) == "UNKNOWN"
        and str(match.get("status") or "").upper() == "NOT_DETECTED"
    )
    formed = sum(1 for match in countable_matches if _completion_state(match) == "FORMED")
    unformed = sum(1 for match in countable_matches if _completion_state(match) == "UNFORMED")
    unknown_completion = sum(1 for match in countable_matches if _completion_state(match) == "UNKNOWN")
    in_tolerance = sum(1 for match in countable_matches if _metrology_state(match) == "IN_TOL")
    out_of_tolerance = sum(1 for match in countable_matches if _metrology_state(match) == "OUT_OF_TOL")
    unmeasurable = sum(1 for match in countable_matches if _metrology_state(match) == "UNMEASURABLE")
    on_position = sum(1 for match in countable_matches if _position_state(match) == "ON_POSITION")
    mislocated = sum(1 for match in countable_matches if _position_state(match) == "MISLOCATED")
    unknown_position = sum(1 for match in countable_matches if _position_state(match) == "UNKNOWN_POSITION")
    overall_result = "FAIL" if (mislocated > 0 or unformed > 0) else "WARNING" if (
        out_of_tolerance > 0 or unmeasurable > 0 or unknown_completion > 0 or unknown_position > 0
    ) else "PASS"
    release_decision = "AUTO_FAIL" if overall_result == "FAIL" else "HOLD" if overall_result == "WARNING" else "AUTO_PASS"
    return {
        "total_bends": total,
        "countable_total_bends": total,
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
        "formed_bends": formed,
        "unformed_bends": unformed,
        "unknown_completion_bends": unknown_completion,
        "in_tolerance_bends": in_tolerance,
        "out_of_tolerance_bends": out_of_tolerance,
        "unmeasurable_bends": unmeasurable,
        "on_position_bends": on_position,
        "mislocated_bends": mislocated,
        "unknown_position_bends": unknown_position,
        "overall_result": overall_result,
        "release_decision": release_decision,
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
    prev_sequence = previous_scan_record.get("sequence")
    curr_sequence = current_scan_record.get("sequence")
    if (
        bool(previous_scan_record.get("sequence_confident"))
        and bool(current_scan_record.get("sequence_confident"))
        and prev_sequence is not None
        and curr_sequence is not None
    ):
        return float(curr_sequence) > float(prev_sequence)

    previous_expected_total = previous_item.get("expectation", {}).get("expected_total_bends")
    previous_completed = previous_item.get("metrics", {}).get("completed_bends")
    current_completed = current_item.get("metrics", {}).get("completed_bends")
    if previous_expected_total is None or previous_completed is None or current_completed is None:
        return False
    if int(previous_completed) != int(previous_expected_total):
        return False
    if int(current_completed) >= int(previous_completed):
        return False

    previous_scan_name = str(previous_item.get("scan_name") or previous_scan_record.get("name") or "")
    current_scan_name = str(current_item.get("scan_name") or current_scan_record.get("name") or "")
    previous_scan_index = _scan_index_from_name(previous_scan_name)
    current_scan_index = _scan_index_from_name(current_scan_name)
    if previous_scan_index is None or current_scan_index is None:
        return False
    return current_scan_index > previous_scan_index


def _scan_index_from_name(scan_name: str) -> Optional[int]:
    match = re.search(r"SCAN\s*([0-9]+)", str(scan_name).replace("_", " "), re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


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
    carried["observability_state"] = current_match.get("observability_state") or "UNKNOWN"
    carried["observability_detail_state"] = current_match.get("observability_detail_state") or "UNOBSERVED"
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


def _retain_previous_partial_match_for_full(
    previous_match: Dict[str, Any],
    current_match: Dict[str, Any],
    previous_scan_name: str,
    current_scan_name: str,
) -> Dict[str, Any]:
    carried = copy.deepcopy(previous_match)
    prev_conf = float(previous_match.get("confidence") or 0.0)
    prev_evidence = float(previous_match.get("local_evidence_score") or 0.0)
    current_evidence = float(current_match.get("local_evidence_score") or 0.0)
    carried["retained_from_previous_partial"] = True
    carried["retention_source_scan"] = previous_scan_name
    carried["retention_target_scan"] = current_scan_name
    carried["physical_completion_state"] = previous_match.get("physical_completion_state") or "FORMED"
    carried["measurement_mode"] = "partial_to_full_retention"
    carried["measurement_method"] = "retain_explicit_previous_partial_into_full"
    carried["assignment_source"] = "PREVIOUS_PARTIAL_RETENTION"
    carried["assignment_confidence"] = round(min(0.9, max(prev_conf, prev_evidence)), 3)
    carried["assignment_candidate_id"] = current_match.get("assignment_candidate_id") or "null_candidate"
    carried["assignment_candidate_kind"] = current_match.get("assignment_candidate_kind") or "NULL"
    carried["assignment_candidate_score"] = round(float(current_match.get("assignment_candidate_score") or 0.0), 3)
    carried["assignment_null_score"] = round(float(current_match.get("assignment_null_score") or 0.0), 3)
    carried["assignment_candidate_count"] = int(current_match.get("assignment_candidate_count") or 0)
    carried["observability_state"] = "FORMED"
    carried["observability_detail_state"] = "RETAINED_FROM_PREVIOUS_PARTIAL"
    carried["observability_confidence"] = round(max(float(current_match.get("observability_confidence") or 0.0), min(0.8, max(prev_conf, prev_evidence) * 0.75)), 3)
    carried["visibility_score"] = round(max(float(current_match.get("visibility_score") or 0.0), 0.0), 3)
    carried["visibility_score_source"] = current_match.get("visibility_score_source") or "heuristic_local_visibility_v0"
    carried["surface_visibility_ratio"] = round(float(current_match.get("surface_visibility_ratio") or 0.0), 3)
    carried["local_support_score"] = round(float(current_match.get("local_support_score") or 0.0), 3)
    carried["side_balance_score"] = round(float(current_match.get("side_balance_score") or 0.0), 3)
    carried["local_point_count"] = int(current_match.get("local_point_count") or 0)
    carried["local_evidence_score"] = round(max(current_evidence, min(0.85, prev_evidence * 0.8)), 3)
    context = dict(previous_match.get("measurement_context") or {})
    context.update(
        {
            "retained_from_previous_partial": True,
            "retained_from_scan": previous_scan_name,
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
        current_observability = normalize_observability_state(
            current_match.get("observability_state"),
            physical_completion_state=current_match.get("physical_completion_state"),
            status=current_status,
        )
        current_evidence = float(current_match.get("local_evidence_score") or 0.0)
        previous_confidence = float(previous_match.get("confidence") or 0.0)
        previous_evidence = float(previous_match.get("local_evidence_score") or 0.0)

        should_carry = (
            _status_priority(previous_status) > 0
            and current_status == "NOT_DETECTED"
            and current_observability == "UNKNOWN"
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


def _should_apply_partial_to_full_retention(
    previous_item: Optional[Dict[str, Any]],
    current_item: Dict[str, Any],
) -> bool:
    if not previous_item or "matches" not in previous_item or "metrics" not in previous_item:
        return False
    if str(previous_item.get("expectation", {}).get("state") or "").lower() != "partial":
        return False
    if str(current_item.get("expectation", {}).get("state") or "").lower() != "full":
        return False
    previous_completed = previous_item.get("metrics", {}).get("completed_bends")
    current_completed = current_item.get("metrics", {}).get("completed_bends")
    if previous_completed is None or current_completed is None:
        return False
    return int(previous_completed) <= int(current_completed)


def _apply_partial_to_full_retention(
    previous_item: Dict[str, Any],
    current_item: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    previous_matches = {
        str(match.get("bend_id")): match
        for match in (previous_item.get("matches") or [])
        if match.get("bend_id")
    }
    updated_matches: List[Dict[str, Any]] = []
    retained_ids: List[str] = []
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
        previous_observability = normalize_observability_state(
            previous_match.get("observability_state"),
            physical_completion_state=previous_match.get("physical_completion_state"),
            status=previous_status,
        )
        current_observability = normalize_observability_state(
            current_match.get("observability_state"),
            physical_completion_state=current_match.get("physical_completion_state"),
            status=current_status,
        )
        current_evidence = float(current_match.get("local_evidence_score") or 0.0)
        previous_confidence = float(previous_match.get("confidence") or 0.0)
        previous_evidence = float(previous_match.get("local_evidence_score") or 0.0)

        should_retain = (
            _status_priority(previous_status) > 0
            and not bool(previous_match.get("continuity_inferred"))
            and not bool(previous_match.get("retained_from_previous_partial"))
            and previous_observability == "FORMED"
            and current_status == "NOT_DETECTED"
            and current_observability == "UNKNOWN"
            and current_evidence <= 0.2
            and max(previous_confidence, previous_evidence) >= 0.5
        )
        if should_retain:
            updated_matches.append(
                _retain_previous_partial_match_for_full(previous_match, current_match, previous_scan_name, current_scan_name)
            )
            retained_ids.append(bend_id)
        else:
            updated_matches.append(current_match)

    if not retained_ids:
        return current_item, []

    updated_item = copy.deepcopy(current_item)
    updated_item["matches"] = updated_matches
    updated_item["summary"] = _recompute_summary(updated_matches)
    updated_item["metrics"] = _score_scan(updated_item, updated_item.get("expectation") or {})
    updated_item["metrics"]["retained_from_previous_partial_bends"] = len(retained_ids)
    updated_item.setdefault("warnings", [])
    updated_item["warnings"] = list(updated_item["warnings"]) + [
        f"Retained explicit bends from {previous_scan_name} into {current_scan_name}: {', '.join(retained_ids)}"
    ]
    return updated_item, retained_ids


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
    countable_matches = [match for match in (report_dict.get("matches") or []) if _countable_primary_match(match)]
    completed = int(summary.get("completed_bends") or 0)
    expected_total = expectation.get("expected_total_bends")
    expected_completed = expectation.get("expected_completed_bends")
    abstained_bends = sum(
        1
        for match in countable_matches
        if _completion_state(match) == "UNKNOWN"
        or _metrology_state(match) == "UNMEASURABLE"
        or _position_state(match) == "UNKNOWN_POSITION"
    )
    accepted_bends = max(0, len(countable_matches) - abstained_bends)

    metrics: Dict[str, Any] = {
        "completed_bends": completed,
        "completed_in_spec": int(summary.get("completed_in_spec") or 0),
        "completed_out_of_spec": int(summary.get("completed_out_of_spec") or 0),
        "remaining_bends": int(summary.get("remaining_bends") or 0),
        "observability_breakdown": _observability_breakdown(report_dict.get("matches") or []),
        "continuity_inferred_bends": _continuity_inferred_count(report_dict.get("matches") or []),
        "retained_from_previous_partial_bends": _retained_from_previous_partial_count(report_dict.get("matches") or []),
        "expected_total_bends": expected_total,
        "expected_completed_bends": expected_completed,
        "cad_strategy": details.get("cad_strategy"),
        "expected_bend_count_used": details.get("expected_bend_count"),
        "countable_bends": len(countable_matches),
        "formed_bends": sum(1 for match in countable_matches if _completion_state(match) == "FORMED"),
        "unformed_bends": sum(1 for match in countable_matches if _completion_state(match) == "UNFORMED"),
        "unknown_completion_bends": sum(1 for match in countable_matches if _completion_state(match) == "UNKNOWN"),
        "in_tolerance_bends": sum(1 for match in countable_matches if _metrology_state(match) == "IN_TOL"),
        "out_of_tolerance_bends": sum(1 for match in countable_matches if _metrology_state(match) == "OUT_OF_TOL"),
        "unmeasurable_bends": sum(1 for match in countable_matches if _metrology_state(match) == "UNMEASURABLE"),
        "on_position_bends": sum(1 for match in countable_matches if _position_state(match) == "ON_POSITION"),
        "mislocated_bends": sum(1 for match in countable_matches if _position_state(match) == "MISLOCATED"),
        "unknown_position_bends": sum(1 for match in countable_matches if _position_state(match) == "UNKNOWN_POSITION"),
        "correspondence_confident_bends": sum(1 for match in countable_matches if _correspondence_state(match) == "CONFIDENT"),
        "correspondence_ambiguous_bends": sum(1 for match in countable_matches if _correspondence_state(match) == "AMBIGUOUS"),
        "correspondence_unresolved_bends": sum(1 for match in countable_matches if _correspondence_state(match) == "UNRESOLVED"),
        "observability_sufficient_bends": sum(1 for match in countable_matches if _observability_state_internal(match) == "SUFFICIENT"),
        "observability_partial_bends": sum(1 for match in countable_matches if _observability_state_internal(match) == "PARTIAL"),
        "observability_insufficient_bends": sum(1 for match in countable_matches if _observability_state_internal(match) == "INSUFFICIENT"),
        "abstained_bends": abstained_bends,
        "accepted_bends": accepted_bends,
        "accepted_coverage_rate": round(accepted_bends / len(countable_matches), 3) if countable_matches else None,
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


def _heatmap_consistency_metrics(report_dict: Dict[str, Any], expectation: Dict[str, Any]) -> Dict[str, Any]:
    matches = report_dict.get("matches") or []
    countable_matches = [
        match for match in matches
        if bool(match.get("countable_in_regression", True))
        and str(match.get("feature_type") or "").upper() not in {"ROLLED_SECTION", "PROCESS_FEATURE"}
    ]
    counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "NOT_APPLICABLE": 0,
    }
    by_method: Dict[str, Dict[str, int]] = {}
    for match in countable_matches:
        consistency = match.get("heatmap_consistency") or {}
        status = str(consistency.get("status") or "NOT_APPLICABLE").upper()
        counts[status] = counts.get(status, 0) + 1
        method = str(match.get("measurement_method") or "unknown")
        method_counts = by_method.setdefault(method, {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT_EVIDENCE": 0, "NOT_APPLICABLE": 0})
        method_counts[status] = method_counts.get(status, 0) + 1
    summary = (report_dict.get("summary") or {}).get("heatmap_consistency_summary") or {}
    return {
        "scan_state": expectation.get("state"),
        "countable_bends": len(countable_matches),
        "supported_bends": counts.get("SUPPORTED", 0),
        "contradicted_bends": counts.get("CONTRADICTED", 0),
        "insufficient_evidence_bends": counts.get("INSUFFICIENT_EVIDENCE", 0),
        "not_applicable_bends": counts.get("NOT_APPLICABLE", 0),
        "measurement_method_breakdown": by_method,
        "summary_status": summary.get("status"),
        "summary_inconsistent_bend_count": summary.get("inconsistent_bend_count"),
        "summary_insufficient_evidence_bend_count": summary.get("insufficient_evidence_bend_count"),
        "summary_bend_roi_energy_share": summary.get("bend_roi_energy_share"),
        "summary_global_in_tolerance_rate": summary.get("global_in_tolerance_rate"),
    }


def _heatmap_bend_rows(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in all_results:
        if "metrics" not in item:
            continue
        state = str(item.get("expectation", {}).get("state") or "unknown")
        part_key = str(item.get("part_key") or "")
        scan_name = str(item.get("scan_name") or "")
        for match in item.get("matches") or []:
            if not bool(match.get("countable_in_regression", True)):
                continue
            if str(match.get("feature_type") or "").upper() in {"ROLLED_SECTION", "PROCESS_FEATURE"}:
                continue
            consistency = match.get("heatmap_consistency") or {}
            rows.append(
                {
                    "part_key": part_key,
                    "scan_name": scan_name,
                    "scan_state": state,
                    "bend_id": str(match.get("bend_id") or ""),
                    "bend_status": str(match.get("status") or "UNKNOWN").upper(),
                    "measurement_method": str(match.get("measurement_method") or "unknown"),
                    "independence_class": str(consistency.get("independence_class") or "UNKNOWN").upper(),
                    "consistency_status": str(consistency.get("status") or "NOT_APPLICABLE").upper(),
                    "consistency_score": consistency.get("consistency_score"),
                    "roi_mode": consistency.get("roi_mode"),
                    "roi_point_count": consistency.get("roi_point_count"),
                    "roi_coverage_ratio": consistency.get("roi_coverage_ratio"),
                    "axial_half_window_mm": consistency.get("axial_half_window_mm"),
                    "radial_radius_mm": consistency.get("radial_radius_mm"),
                    "local_abs_p95_mm": consistency.get("local_abs_p95_mm"),
                    "local_out_of_tol_ratio": consistency.get("local_out_of_tol_ratio"),
                    "side_asymmetry_score": consistency.get("side_asymmetry_score"),
                }
            )
    return rows


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
    skip_heatmap_consistency: bool,
    heatmap_cache_dir: Optional[Path],
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
    if skip_heatmap_consistency:
        cmd.append("--skip-heatmap-consistency")
    if heatmap_cache_dir is not None:
        cmd.extend(["--heatmap-cache-dir", str(heatmap_cache_dir)])
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
    if "matches" not in item:
        return item
    annotated = copy.deepcopy(item)
    if unary_bundle is not None:
        annotated = score_case_payload(annotated, unary_bundle)
        annotated["structured_metrics"] = _structured_score_scan(annotated, expectation)
    annotated["heatmap_consistency_metrics"] = _heatmap_consistency_metrics(annotated, expectation)
    return annotated


def evaluate_part(
    part_record: Dict[str, Any],
    output_dir: Path,
    skip_existing: bool,
    timeout_sec: int,
    unary_bundle: Optional[Dict[str, Any]],
    skip_heatmap_consistency: bool,
    heatmap_cache_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
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
        return [], [f"{part_key}: no CAD file available"], []
    spec_bend_angles = _spec_bend_angles(metadata)
    spec_schedule_type = _spec_schedule_type(metadata)

    part_out_dir = output_dir / part_key
    part_out_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: List[str] = []
    scan_results: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []
    previous_sequence_item: Optional[Dict[str, Any]] = None
    previous_sequence_scan_record: Optional[Dict[str, Any]] = None

    for scan_record in _ordered_scan_records(metadata):
        scan_name = str(scan_record.get("name"))
        scan_path = str(scan_record.get("path"))
        expectation = _scan_expectation(scan_record, labels_map, expected_total_bends)
        result_path = part_out_dir / _stable_scan_output_name(scan_name)
        entry: Dict[str, Any] = {
            "part_key": part_key,
            "scan_name": scan_name,
            "scan_path": scan_path,
            "scan_state": expectation.get("state"),
            "expected_total_bends": expectation.get("expected_total_bends"),
            "expected_completed_bends": expectation.get("expected_completed_bends"),
            "output_json": str(result_path),
            "status": "pending",
            "sequence": scan_record.get("sequence"),
            "sequence_confident": bool(scan_record.get("sequence_confident")),
        }
        if skip_existing and result_path.exists():
            existing = _load_json(result_path)
            if "metrics" in existing:
                existing.setdefault("metrics", _score_scan(existing, expectation))
                existing = _annotate_structured_predictions(existing, expectation, unary_bundle)
                result_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            scan_results.append(existing)
            entry["status"] = "skipped_existing"
            if str(expectation.get("state") or "").lower() == "partial":
                previous_sequence_item = existing
                previous_sequence_scan_record = scan_record
            manifest_entries.append(entry)
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
                skip_heatmap_consistency=skip_heatmap_consistency,
                heatmap_cache_dir=heatmap_cache_dir,
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
            if _should_apply_partial_to_full_retention(previous_sequence_item, item):
                item, retained_ids = _apply_partial_to_full_retention(previous_sequence_item, item)
                if retained_ids:
                    result_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            item = _annotate_structured_predictions(item, expectation, unary_bundle)
            result_path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            item["scan_sequence"] = scan_record.get("sequence")
            item["sequence_confident"] = bool(scan_record.get("sequence_confident"))
            scan_results.append(item)
            entry["status"] = "success"
            if isinstance(item.get("timing"), dict):
                entry["timing"] = item["timing"]
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
            if str(expectation.get("state") or "").lower() == "partial":
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
            entry["status"] = "failed"
            entry["error"] = str(exc)
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
            manifest_entries.append(entry)
            gc.collect()

    return scan_results, warnings_out, manifest_entries


def _load_results_from_manifest(manifest: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    all_results: List[Dict[str, Any]] = []
    warnings_out: List[str] = list(manifest.get("warnings") or [])
    for entry in manifest.get("entries") or []:
        output_json = Path(str(entry.get("output_json") or ""))
        if output_json.exists():
            try:
                all_results.append(_load_json(output_json))
                continue
            except Exception as exc:
                warnings_out.append(
                    f"{entry.get('part_key')}/{entry.get('scan_name')}: failed to load output json: {exc}"
                )
        all_results.append(
            {
                "part_key": entry.get("part_key"),
                "scan_name": entry.get("scan_name"),
                "scan_path": entry.get("scan_path"),
                "error": entry.get("error") or "missing output json",
                "expectation": {
                    "state": entry.get("scan_state"),
                    "expected_total_bends": entry.get("expected_total_bends"),
                    "expected_completed_bends": entry.get("expected_completed_bends"),
                },
            }
        )
    return all_results, warnings_out


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

    heatmap_rows = [r for r in completed_metric_results if r.get("heatmap_consistency_metrics")]
    full_heatmap_rows = [r for r in heatmap_rows if str(r.get("expectation", {}).get("state")) == "full"]
    partial_heatmap_rows = [r for r in heatmap_rows if str(r.get("expectation", {}).get("state")) == "partial"]

    def _sum_heatmap(rows: List[Dict[str, Any]], key: str) -> int:
        return int(sum(int((r.get("heatmap_consistency_metrics") or {}).get(key) or 0) for r in rows))

    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        if denominator <= 0:
            return None
        return round(numerator / denominator, 3)

    heatmap_bend_rows = _heatmap_bend_rows(completed_metric_results)

    def _count_bend_rows(
        *,
        bend_status: Optional[Any] = None,
        consistency_status: Optional[str] = None,
        scan_state: Optional[str] = None,
        independence_class: Optional[str] = None,
        measurement_method: Optional[str] = None,
    ) -> int:
        total = 0
        for row in heatmap_bend_rows:
            if bend_status is not None:
                if isinstance(bend_status, (list, tuple, set, frozenset)):
                    if row["bend_status"] not in bend_status:
                        continue
                elif row["bend_status"] != bend_status:
                    continue
            if consistency_status is not None and row["consistency_status"] != consistency_status:
                continue
            if scan_state is not None and row["scan_state"] != scan_state:
                continue
            if independence_class is not None and row["independence_class"] != independence_class:
                continue
            if measurement_method is not None and row["measurement_method"] != measurement_method:
                continue
            total += 1
        return total

    staged = _staged_progression_metrics(completed_metric_results)
    structured_rows = [r for r in completed_metric_results if r.get("structured_metrics")]
    structured_with_total = [
        r for r in structured_rows if (r.get("structured_metrics") or {}).get("expected_total_error_median") is not None
    ]
    structured_with_completed = [
        r for r in structured_rows if (r.get("structured_metrics") or {}).get("expected_completed_error_median") is not None
    ]
    structured_full_scans = [r for r in full_scans if (r.get("structured_metrics") or {}).get("completed_bends_median") is not None]
    heatmap_total_bends = sum(int((r.get("heatmap_consistency_metrics") or {}).get("countable_bends") or 0) for r in heatmap_rows)
    heatmap_total_full_bends = sum(int((r.get("heatmap_consistency_metrics") or {}).get("countable_bends") or 0) for r in full_heatmap_rows)
    heatmap_total_partial_bends = sum(int((r.get("heatmap_consistency_metrics") or {}).get("countable_bends") or 0) for r in partial_heatmap_rows)
    total_countable_bends = sum(int((r.get("metrics") or {}).get("countable_bends") or 0) for r in completed_metric_results)
    total_guard_features = sum(
        sum(1 for match in (r.get("matches") or []) if _guard_process_match(match))
        for r in completed_metric_results
    )

    def _sum_metric(rows: List[Dict[str, Any]], key: str) -> int:
        return int(sum(int((r.get("metrics") or {}).get(key) or 0) for r in rows))

    def _rate(numerator: int, denominator: int) -> Optional[float]:
        if denominator <= 0:
            return None
        return round(numerator / denominator, 3)

    angle_errors = [
        abs(float(match.get("angle_deviation") or 0.0))
        for row in completed_metric_results
        for match in (row.get("matches") or [])
        if _countable_primary_match(match) and match.get("measured_angle") is not None
    ]
    radius_errors = [
        abs(float(match.get("radius_deviation") or 0.0))
        for row in completed_metric_results
        for match in (row.get("matches") or [])
        if _countable_primary_match(match) and match.get("measured_radius") is not None
    ]
    line_center_errors = [
        abs(float(match.get("line_center_deviation_mm") or 0.0))
        for row in completed_metric_results
        for match in (row.get("matches") or [])
        if _countable_primary_match(match) and match.get("line_center_deviation_mm") is not None
    ]
    process_detected = sum(
        1
        for row in completed_metric_results
        for match in (row.get("matches") or [])
        if _guard_process_match(match) and str(match.get("status") or "").upper() != "NOT_DETECTED"
    )
    decision_complete_scans = [
        r for r in completed_metric_results
        if int((r.get("metrics") or {}).get("abstained_bends") or 0) == 0
    ]
    decision_complete_partial_scans = [
        r for r in partial_scans
        if int((r.get("metrics") or {}).get("abstained_bends") or 0) == 0
    ]
    contradiction_cases = []
    for row in completed_metric_results:
        for match in (row.get("matches") or []):
            if not _countable_primary_match(match):
                continue
            angle = match.get("measured_angle")
            target = match.get("target_angle")
            line_center = match.get("line_center_deviation_mm")
            if isinstance(angle, (int, float)) and isinstance(target, (int, float)) and isinstance(line_center, (int, float)):
                if abs(angle - target) <= 1.5 and abs(line_center) >= POSITIONAL_WARN_MM:
                    contradiction_cases.append(
                        {
                            "part_key": row.get("part_key"),
                            "scan_name": row.get("scan_name"),
                            "bend_id": match.get("bend_id"),
                            "status": match.get("status"),
                            "measured_angle": angle,
                            "target_angle": target,
                            "line_center_deviation_mm": line_center,
                            "assignment_source": match.get("assignment_source"),
                            "measurement_method": match.get("measurement_method"),
                            "correspondence_state": _correspondence_state(match),
                        }
                    )
            if str(match.get("status") or "").upper() != "NOT_DETECTED" and _correspondence_state(match) != "CONFIDENT":
                contradiction_cases.append(
                    {
                        "part_key": row.get("part_key"),
                        "scan_name": row.get("scan_name"),
                        "bend_id": match.get("bend_id"),
                        "status": match.get("status"),
                        "measured_angle": angle,
                        "target_angle": target,
                        "line_center_deviation_mm": line_center,
                        "assignment_source": match.get("assignment_source"),
                        "measurement_method": match.get("measurement_method"),
                        "correspondence_state": _correspondence_state(match),
                    }
                )
    aggregate = {
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
        "heatmap_consistency_scans": len(heatmap_rows),
        "heatmap_supported_bends": _sum_heatmap(heatmap_rows, "supported_bends"),
        "heatmap_contradicted_bends": _sum_heatmap(heatmap_rows, "contradicted_bends"),
        "heatmap_insufficient_evidence_bends": _sum_heatmap(heatmap_rows, "insufficient_evidence_bends"),
        "heatmap_supported_rate": _ratio(
            _sum_heatmap(heatmap_rows, "supported_bends"),
            heatmap_total_bends,
        ),
        "heatmap_contradicted_rate": _ratio(
            _sum_heatmap(heatmap_rows, "contradicted_bends"),
            heatmap_total_bends,
        ),
        "heatmap_full_scan_supported_rate": _ratio(
            _sum_heatmap(full_heatmap_rows, "supported_bends"),
            heatmap_total_full_bends,
        ),
        "heatmap_partial_scan_supported_rate": _ratio(
            _sum_heatmap(partial_heatmap_rows, "supported_bends"),
            heatmap_total_partial_bends,
        ),
        "heatmap_pass_contradicted": _count_bend_rows(bend_status="PASS", consistency_status="CONTRADICTED"),
        "heatmap_fail_warning_supported": _count_bend_rows(
            bend_status=("FAIL", "WARNING"),
            consistency_status="SUPPORTED",
        ),
        "heatmap_fail_supported": _count_bend_rows(bend_status="FAIL", consistency_status="SUPPORTED"),
        "heatmap_warning_supported": _count_bend_rows(bend_status="WARNING", consistency_status="SUPPORTED"),
        "heatmap_full_pass_contradicted": _count_bend_rows(
            bend_status="PASS",
            consistency_status="CONTRADICTED",
            scan_state="full",
        ),
        "heatmap_partial_pass_contradicted": _count_bend_rows(
            bend_status="PASS",
            consistency_status="CONTRADICTED",
            scan_state="partial",
        ),
        "continuity_inferred_scan_count": sum(
            1 for r in completed_metric_results if int(r["metrics"].get("continuity_inferred_bends") or 0) > 0
        ),
        "continuity_inferred_bends_total": sum(
            int(r["metrics"].get("continuity_inferred_bends") or 0) for r in completed_metric_results
        ),
        **staged,
    }
    aggregate["scoreboards"] = {
        "completion": {
            "countable_bends": total_countable_bends,
            "formed_bends": _sum_metric(completed_metric_results, "formed_bends"),
            "unformed_bends": _sum_metric(completed_metric_results, "unformed_bends"),
            "unknown_completion_bends": _sum_metric(completed_metric_results, "unknown_completion_bends"),
            "false_positive_completion_rate": _rate(
                sum(max(0, int((r.get("metrics") or {}).get("expected_completed_error") or 0)) for r in with_completed),
                sum(int((r.get("metrics") or {}).get("expected_completed_bends") or 0) for r in with_completed),
            ),
            "false_negative_completion_rate": _rate(
                sum(max(0, -int((r.get("metrics") or {}).get("expected_completed_error") or 0)) for r in with_completed),
                sum(int((r.get("metrics") or {}).get("expected_completed_bends") or 0) for r in with_completed),
            ),
            "scan_level_mae_expected_completed": aggregate["mae_vs_expected_completed"],
            "scan_level_mae_expected_total": aggregate["mae_vs_expected_total_completed"],
        },
        "metrology": {
            "angle_mae_deg": round(sum(angle_errors) / len(angle_errors), 3) if angle_errors else None,
            "radius_mae_mm": round(sum(radius_errors) / len(radius_errors), 3) if radius_errors else None,
            "in_tolerance_rate": _rate(_sum_metric(completed_metric_results, "in_tolerance_bends"), total_countable_bends),
            "out_of_tolerance_rate": _rate(_sum_metric(completed_metric_results, "out_of_tolerance_bends"), total_countable_bends),
            "unmeasurable_rate": _rate(_sum_metric(completed_metric_results, "unmeasurable_bends"), total_countable_bends),
        },
        "position": {
            "line_center_mae_mm": round(sum(line_center_errors) / len(line_center_errors), 3) if line_center_errors else None,
            "mislocated_rate": _rate(_sum_metric(completed_metric_results, "mislocated_bends"), total_countable_bends),
            "unknown_position_rate": _rate(_sum_metric(completed_metric_results, "unknown_position_bends"), total_countable_bends),
            "heatmap_supported_rate": aggregate["heatmap_supported_rate"],
            "heatmap_contradicted_rate": aggregate["heatmap_contradicted_rate"],
        },
        "abstention": {
            "abstention_rate": _rate(_sum_metric(completed_metric_results, "abstained_bends"), total_countable_bends),
            "accepted_coverage_rate": _rate(_sum_metric(completed_metric_results, "accepted_bends"), total_countable_bends),
            "decision_complete_scan_count": len(decision_complete_scans),
            "decision_complete_scan_exact_rate": (
                round(sum(1 for r in decision_complete_scans if (r.get("metrics") or {}).get("completed_matches_expected")) / len(decision_complete_scans), 3)
                if decision_complete_scans else None
            ),
            "partial_selective_risk": (
                round(
                    1.0 - (
                        sum(1 for r in decision_complete_partial_scans if (r.get("metrics") or {}).get("completed_matches_expected"))
                        / len(decision_complete_partial_scans)
                    ),
                    3,
                )
                if decision_complete_partial_scans else None
            ),
        },
        "guard_metrics": {
            "process_feature_total": total_guard_features,
            "process_feature_detected": process_detected,
        },
    }
    aggregate["contradiction_case_count"] = len(contradiction_cases)
    aggregate["_contradiction_cases"] = contradiction_cases[:100]
    return aggregate


def _write_markdown(output_dir: Path, aggregate: Dict[str, Any], all_results: List[Dict[str, Any]], warnings_out: List[str]) -> None:
    scoreboards = aggregate.get("scoreboards") or {}
    completion = scoreboards.get("completion") or {}
    metrology = scoreboards.get("metrology") or {}
    position = scoreboards.get("position") or {}
    abstention = scoreboards.get("abstention") or {}
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
        f"- Heatmap consistency scans: `{aggregate['heatmap_consistency_scans']}`",
        f"- Heatmap supported bends: `{aggregate['heatmap_supported_bends']}`",
        f"- Heatmap contradicted bends: `{aggregate['heatmap_contradicted_bends']}`",
        f"- Heatmap insufficient-evidence bends: `{aggregate['heatmap_insufficient_evidence_bends']}`",
        f"- Heatmap supported rate: `{aggregate['heatmap_supported_rate']}`",
        f"- Heatmap contradicted rate: `{aggregate['heatmap_contradicted_rate']}`",
        f"- Heatmap full-scan supported rate: `{aggregate['heatmap_full_scan_supported_rate']}`",
        f"- Heatmap partial-scan supported rate: `{aggregate['heatmap_partial_scan_supported_rate']}`",
        f"- Heatmap PASS contradicted: `{aggregate['heatmap_pass_contradicted']}`",
        f"- Heatmap FAIL/WARNING supported: `{aggregate['heatmap_fail_warning_supported']}`",
        f"- Heatmap FAIL supported: `{aggregate['heatmap_fail_supported']}`",
        f"- Heatmap WARNING supported: `{aggregate['heatmap_warning_supported']}`",
        f"- Heatmap full-scan PASS contradicted: `{aggregate['heatmap_full_pass_contradicted']}`",
        f"- Heatmap partial-scan PASS contradicted: `{aggregate['heatmap_partial_pass_contradicted']}`",
        f"- Full-scan exact completion rate: `{aggregate['full_scan_exact_completion_rate']}`",
        f"- Structured full-scan exact completion rate: `{aggregate['structured_full_scan_exact_completion_rate']}`",
        f"- Continuity-inferred scans: `{aggregate['continuity_inferred_scan_count']}`",
        f"- Continuity-inferred bends: `{aggregate['continuity_inferred_bends_total']}`",
        f"- Staged monotonic pair rate: `{aggregate['staged_monotonic_pair_rate']}`",
        f"- Staged retained bend rate: `{aggregate['staged_retained_bend_rate']}`",
        "",
        "## Scoreboards",
        "",
        f"- Completion false-positive rate: `{completion.get('false_positive_completion_rate')}`",
        f"- Completion false-negative rate: `{completion.get('false_negative_completion_rate')}`",
        f"- Metrology angle MAE: `{metrology.get('angle_mae_deg')}`",
        f"- Position line-center MAE: `{position.get('line_center_mae_mm')}`",
        f"- Abstention rate: `{abstention.get('abstention_rate')}`",
        f"- Accepted coverage rate: `{abstention.get('accepted_coverage_rate')}`",
        f"- Partial selective risk: `{abstention.get('partial_selective_risk')}`",
        f"- Contradiction case count: `{aggregate.get('contradiction_case_count')}`",
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


def _write_scoreboards(output_dir: Path, aggregate: Dict[str, Any]) -> None:
    payload = {
        "scoreboards": aggregate.get("scoreboards") or {},
        "contradiction_case_count": aggregate.get("contradiction_case_count"),
        "heatmap_supported_rate": aggregate.get("heatmap_supported_rate"),
        "heatmap_contradicted_rate": aggregate.get("heatmap_contradicted_rate"),
    }
    (output_dir / "scoreboards.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_heatmap_debug(output_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    rows = [row for row in _heatmap_bend_rows(all_results) if row["consistency_status"] == "CONTRADICTED"]
    rows.sort(
        key=lambda row: (
            -(float(row.get("consistency_score") or 0.0)),
            -(float(row.get("local_out_of_tol_ratio") or 0.0)),
            -(float(row.get("local_abs_p95_mm") or 0.0)),
        )
    )
    (output_dir / "heatmap_consistency_debug_top_contradictions.json").write_text(
        json.dumps(rows[:50], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_contradiction_debug(output_dir: Path, aggregate: Dict[str, Any]) -> None:
    rows = list(aggregate.get("_contradiction_cases") or [])
    (output_dir / "decision_contradiction_cases.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Decision Contradiction Cases",
        "",
        f"- Count: `{len(rows)}`",
        "",
        "| Part | Scan | Bend | Status | Angle | Target | Center Δmm | Correspondence |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('part_key', '-')} | {row.get('scan_name', '-')} | {row.get('bend_id', '-')} | "
            f"{row.get('status', '-')} | {row.get('measured_angle', '-')} | {row.get('target_angle', '-')} | "
            f"{row.get('line_center_deviation_mm', '-')} | {row.get('correspondence_state', '-')} |"
        )
    (output_dir / "decision_contradiction_cases.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the bend corpus with the progressive inspection pipeline.")
    parser.add_argument("--corpus", default=str(CORPUS_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=420)
    parser.add_argument("--unary-model", default=None)
    parser.add_argument("--skip-heatmap-consistency", action="store_true")
    parser.add_argument("--heatmap-cache-dir", default=None)
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
    heatmap_cache_dir = None
    if not args.skip_heatmap_consistency:
        heatmap_cache_dir = (
            Path(args.heatmap_cache_dir)
            if args.heatmap_cache_dir
            else (corpus_root / "cache" / "heatmap_evidence")
        )
        heatmap_cache_dir.mkdir(parents=True, exist_ok=True)

    warnings_out: List[str] = []
    all_results: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []
    unary_model_path = Path(args.unary_model) if args.unary_model else _latest_unary_model_path()
    unary_bundle = None
    if unary_model_path and unary_model_path.exists():
        unary_bundle = load_unary_models(unary_model_path)

    for part in ready_parts:
        scan_results, part_warnings, part_manifest_entries = evaluate_part(
            part_record=part,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            timeout_sec=args.timeout_sec,
            unary_bundle=unary_bundle,
            skip_heatmap_consistency=bool(args.skip_heatmap_consistency),
            heatmap_cache_dir=heatmap_cache_dir,
        )
        all_results.extend(scan_results)
        warnings_out.extend(part_warnings)
        manifest_entries.extend(part_manifest_entries)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "timeout_sec": args.timeout_sec,
        "skip_heatmap_consistency": bool(args.skip_heatmap_consistency),
        "heatmap_cache_dir": str(heatmap_cache_dir) if heatmap_cache_dir is not None else None,
        "skip_existing": bool(args.skip_existing),
        "parts": [str(part.get("part_key")) for part in ready_parts],
        "entries": manifest_entries,
        "warnings": warnings_out,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    all_results, warnings_out = _load_results_from_manifest(manifest)
    aggregate = _aggregate(all_results)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "results": all_results,
                "warnings": warnings_out,
                "manifest_path": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_markdown(output_dir, aggregate, all_results, warnings_out)
    _write_scoreboards(output_dir, aggregate)
    _write_heatmap_debug(output_dir, all_results)
    _write_contradiction_debug(output_dir, aggregate)
    latest = corpus_root / "evaluation" / "latest.json"
    latest.write_text(json.dumps({"path": str(output_dir / 'summary.json')}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
