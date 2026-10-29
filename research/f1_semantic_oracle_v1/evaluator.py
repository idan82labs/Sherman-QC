"""Hidden semantic evaluator that never returns gold geometry or expected counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_f1_against_manual_gold import (
    Thresholds,
    _endpoints,
    is_match,
    match_lines,
    pair_metrics,
)


class OracleContractError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("owned_bend_regions", "manual_bends", "bends", "predictions"):
        rows = payload.get(key)
        if isinstance(rows, list):
            result = [dict(row) for row in rows]
            break
    else:
        raise OracleContractError("submission has no supported candidate array")
    seen: set[str] = set()
    for index, row in enumerate(result):
        candidate_id = str(row.get("bend_id") or row.get("candidate_id") or f"candidate-{index:04d}")
        if candidate_id in seen:
            raise OracleContractError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        row["bend_id"] = candidate_id
        _endpoints(row)
    return result


def _case_for_phase(lock: dict[str, Any], catalog: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_key = "family_A" if phase == "family_A" else "family_B"
    case_ids = lock[phase_key]["case_ids"]
    if len(case_ids) != 1:
        raise OracleContractError("oracle v1 requires exactly one case per family")
    matches = [case for case in catalog.get("cases", []) if case.get("case_id") == case_ids[0]]
    if len(matches) != 1:
        raise OracleContractError("locked case missing or duplicated in catalog")
    case = matches[0]
    if case.get("part_id") != lock[phase_key]["part_id"]:
        raise OracleContractError("locked family membership mismatch")
    if case.get("truth_tier") != lock["catalog"]["required_truth_tier"] or case.get("promotion_eligible") is not True:
        raise OracleContractError("locked case no longer satisfies truth contract")
    return case


def _thresholds(lock: dict[str, Any]) -> Thresholds:
    values = lock["matching"]
    return Thresholds(
        angle_deg=float(values["axis_angle_degrees_max"]),
        transverse_offset_mm=float(values["transverse_offset_mm_max"]),
        gold_span_overlap=float(values["gold_span_overlap_min"]),
        prediction_span_overlap=float(values["prediction_span_overlap_min"]),
    )


def _opaque_object_id(lock_id: str, case_id: str, gold_id: str) -> str:
    digest = hashlib.sha256(f"{lock_id}\0{case_id}\0{gold_id}".encode("utf-8")).hexdigest()
    return f"object-{digest[:20]}"


def _consume_family_b_claim(
    *,
    claim_path: Path,
    lock_sha256: str,
    submission_sha256: str,
) -> dict[str, Any]:
    claim = _read(claim_path)
    if claim.get("status") != "created":
        raise OracleContractError("Family B claim is not fresh and unconsumed")
    if claim.get("lock_sha256") != lock_sha256:
        raise OracleContractError("Family B claim lock mismatch")
    if claim.get("submission_sha256") != submission_sha256:
        raise OracleContractError("Family B claim submission mismatch")
    consumed = dict(claim)
    consumed["status"] = "consumed"
    consumed["consume_token"] = sha256_bytes(
        f"{lock_sha256}\0{submission_sha256}\0CONSUMED".encode("ascii")
    )
    claim_path.write_text(json.dumps(consumed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return consumed


def evaluate_submission(
    *,
    lock_path: Path,
    prediction_path: Path,
    phase: str,
    family_b_claim_path: Path | None = None,
) -> dict[str, Any]:
    if phase not in {"family_A", "family_B"}:
        raise OracleContractError("phase must be family_A or family_B")
    lock_path = lock_path.resolve()
    prediction_path = prediction_path.resolve()
    lock_raw = lock_path.read_bytes()
    lock_sha256 = sha256_bytes(lock_raw)
    lock = json.loads(lock_raw)
    prediction_raw = prediction_path.read_bytes()
    prediction_sha256 = sha256_bytes(prediction_raw)
    if phase == "family_B":
        if family_b_claim_path is None:
            raise OracleContractError("Family B requires a fresh one-shot claim")
        _consume_family_b_claim(
            claim_path=family_b_claim_path.resolve(),
            lock_sha256=lock_sha256,
            submission_sha256=prediction_sha256,
        )
    catalog_path = (lock_path.parents[2] / lock["catalog"]["path"]).resolve()
    if sha256_file(catalog_path) != lock["catalog"]["sha256"]:
        raise OracleContractError("catalog snapshot hash mismatch")
    catalog = _read(catalog_path)
    case = _case_for_phase(lock, catalog, phase)
    predictions = _prediction_rows(json.loads(prediction_raw))
    thresholds = _thresholds(lock)
    result = match_lines(case["bends"], predictions, thresholds)
    matched_by_prediction = {row["prediction_id"]: row for row in result["matches"]}
    matched_gold_ids = {row["gold_id"] for row in result["matches"]}
    gold_rows = {str(row["bend_id"]): row for row in case["bends"]}
    feedback: list[dict[str, Any]] = []
    for prediction in predictions:
        candidate_id = str(prediction["bend_id"])
        matched = matched_by_prediction.get(candidate_id)
        if matched is not None:
            feedback.append(
                {
                    "candidate_id": candidate_id,
                    "ownership_class": "TRUE_BEND_OBJECT",
                    "opaque_object_id": _opaque_object_id(lock["lock_id"], case["case_id"], matched["gold_id"]),
                    "direction_error_degrees": matched["axis_angle_deg"],
                    "line_distance_mm": matched["transverse_offset_mm"],
                    "gold_span_overlap": matched["gold_span_overlap"],
                    "prediction_span_overlap": matched["prediction_span_overlap"],
                }
            )
            continue
        duplicate_object_id = None
        prediction_line = _endpoints(prediction)
        for gold_id, gold in gold_rows.items():
            metrics = pair_metrics(_endpoints(gold), prediction_line)
            is_duplicate_alias = (
                metrics["axis_angle_deg"] <= thresholds.angle_deg
                and metrics["transverse_offset_mm"] <= thresholds.transverse_offset_mm
                and metrics["shorter_span_overlap"]
                >= min(thresholds.gold_span_overlap, thresholds.prediction_span_overlap)
            )
            if gold_id in matched_gold_ids and is_duplicate_alias:
                duplicate_object_id = _opaque_object_id(lock["lock_id"], case["case_id"], gold_id)
                break
        feedback.append(
            {
                "candidate_id": candidate_id,
                "ownership_class": "DUPLICATE_ALIAS" if duplicate_object_id else "UNMATCHED",
                "opaque_object_id": duplicate_object_id,
                "direction_error_degrees": None,
                "line_distance_mm": None,
                "gold_span_overlap": None,
                "prediction_span_overlap": None,
            }
        )
    tp = int(result["true_positive"])
    fp = int(result["false_positive"])
    fn = int(result["false_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_quarantined_semantic_oracle_feedback",
        "phase": phase,
        "lock_sha256": lock_sha256,
        "submission_sha256": prediction_sha256,
        "opaque_submission_id": f"submission-{sha256_bytes((phase + prediction_sha256).encode())[:20]}",
        "candidate_feedback": feedback,
        "metrics": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "aggregate_exact_count": bool(result["exact"]),
            "aggregate_precision": round(precision, 6),
            "aggregate_recall": round(recall, 6),
            "aggregate_f1": round(f1, 6),
        },
        "gold_geometry_exposed": False,
        "expected_bend_count_exposed": False,
        "c006_opened": False,
    }
    report["feedback_sha256"] = sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    )
    return report
