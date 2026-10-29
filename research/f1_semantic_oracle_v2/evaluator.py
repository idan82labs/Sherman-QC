"""Hidden multi-family evaluator returning no gold geometry or expected counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_f1_against_manual_gold import Thresholds, _endpoints, match_lines, pair_metrics


class OracleContractError(RuntimeError):
    pass


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("owned_bend_regions", "manual_bends", "bends", "predictions"):
        if isinstance(payload.get(key), list):
            result = [dict(row) for row in payload[key]]
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


def _locked_case(lock: dict[str, Any], *, phase: str, case_token: str) -> dict[str, Any]:
    key = "development_cases" if phase == "development" else "transfer_cases"
    matches = [row for row in lock.get(key) or [] if row.get("case_token") == case_token]
    if len(matches) != 1:
        raise OracleContractError("case token is not uniquely locked for requested phase")
    return matches[0]


def _consume_claim(
    claim_path: Path,
    *,
    lock_sha256: str,
    submission_sha256: str,
    case_token: str,
) -> None:
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if claim.get("status") != "created":
        raise OracleContractError("transfer claim is not fresh")
    if claim.get("lock_sha256") != lock_sha256:
        raise OracleContractError("transfer claim lock mismatch")
    if claim.get("submission_sha256") != submission_sha256:
        raise OracleContractError("transfer claim submission mismatch")
    if claim.get("case_token") != case_token:
        raise OracleContractError("transfer claim case mismatch")
    consumed = dict(claim)
    consumed["status"] = "consumed"
    consumed["consume_token"] = _sha_bytes(
        f"{lock_sha256}\0{submission_sha256}\0{case_token}\0CONSUMED".encode("ascii")
    )
    claim_path.write_text(
        json.dumps(consumed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def evaluate_submission(
    *,
    lock_path: Path,
    prediction_path: Path,
    phase: str,
    case_token: str,
    transfer_claim_path: Path | None = None,
) -> dict[str, Any]:
    if phase not in {"development", "transfer"}:
        raise OracleContractError("phase must be development or transfer")
    lock_path = lock_path.resolve()
    prediction_path = prediction_path.resolve()
    lock_raw = lock_path.read_bytes()
    lock_sha = _sha_bytes(lock_raw)
    lock = json.loads(lock_raw)
    locked = _locked_case(lock, phase=phase, case_token=case_token)
    if _sha_file((lock_path.parents[2] / locked["candidate_pool_path"]).resolve()) != locked["candidate_pool_sha256"]:
        raise OracleContractError("locked candidate pool changed")
    prediction_raw = prediction_path.read_bytes()
    prediction_sha = _sha_bytes(prediction_raw)
    if phase == "transfer":
        if transfer_claim_path is None:
            raise OracleContractError("transfer evaluation requires a fresh claim")
        _consume_claim(
            transfer_claim_path.resolve(),
            lock_sha256=lock_sha,
            submission_sha256=prediction_sha,
            case_token=case_token,
        )

    catalog_path = (lock_path.parents[2] / lock["catalog"]["path"]).resolve()
    if _sha_file(catalog_path) != lock["catalog"]["sha256"]:
        raise OracleContractError("catalog snapshot changed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases = [case for case in catalog.get("cases") or [] if case.get("case_id") == locked["case_id"]]
    if len(cases) != 1:
        raise OracleContractError("locked catalog case missing or duplicated")
    case = cases[0]
    if str(case.get("part_id")) != locked["part_id"] or case.get("truth_tier") != locked["truth_tier"]:
        raise OracleContractError("locked case metadata changed")

    values = lock["matching"]
    thresholds = Thresholds(
        angle_deg=float(values["axis_angle_degrees_max"]),
        transverse_offset_mm=float(values["transverse_offset_mm_max"]),
        gold_span_overlap=float(values["gold_span_overlap_min"]),
        prediction_span_overlap=float(values["prediction_span_overlap_min"]),
    )
    predictions = _rows(json.loads(prediction_raw))
    result = match_lines(case["bends"], predictions, thresholds)
    matched_by_prediction = {row["prediction_id"]: row for row in result["matches"]}
    matched_gold_ids = {row["gold_id"] for row in result["matches"]}
    gold_rows = {str(row["bend_id"]): row for row in case["bends"]}

    def object_id(gold_id: str) -> str:
        return "object-" + _sha_bytes(
            f"{lock['lock_id']}\0{locked['case_id']}\0{gold_id}".encode("utf-8")
        )[:20]

    feedback: list[dict[str, Any]] = []
    for prediction in predictions:
        candidate_id = str(prediction["bend_id"])
        matched = matched_by_prediction.get(candidate_id)
        if matched is not None:
            feedback.append(
                {
                    "candidate_id": candidate_id,
                    "ownership_class": "TRUE_BEND_OBJECT",
                    "opaque_object_id": object_id(matched["gold_id"]),
                    "direction_error_degrees": matched["axis_angle_deg"],
                    "line_distance_mm": matched["transverse_offset_mm"],
                    "gold_span_overlap": matched["gold_span_overlap"],
                    "prediction_span_overlap": matched["prediction_span_overlap"],
                }
            )
            continue
        duplicate = None
        line = _endpoints(prediction)
        for gold_id, gold in gold_rows.items():
            metrics = pair_metrics(_endpoints(gold), line)
            if (
                gold_id in matched_gold_ids
                and metrics["axis_angle_deg"] <= thresholds.angle_deg
                and metrics["transverse_offset_mm"] <= thresholds.transverse_offset_mm
                and metrics["shorter_span_overlap"]
                >= min(thresholds.gold_span_overlap, thresholds.prediction_span_overlap)
            ):
                duplicate = object_id(gold_id)
                break
        feedback.append(
            {
                "candidate_id": candidate_id,
                "ownership_class": "DUPLICATE_ALIAS" if duplicate else "UNMATCHED",
                "opaque_object_id": duplicate,
                "direction_error_degrees": None,
                "line_distance_mm": None,
                "gold_span_overlap": None,
                "prediction_span_overlap": None,
            }
        )
    tp, fp, fn = int(result["true_positive"]), int(result["false_positive"]), int(result["false_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_multifamily_hidden_oracle_feedback",
        "phase": phase,
        "case_token": case_token,
        "lock_sha256": lock_sha,
        "submission_sha256": prediction_sha,
        "opaque_submission_id": "submission-" + _sha_bytes((phase + case_token + prediction_sha).encode())[:20],
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
        "promotion_evidence": bool(locked["promotion_eligible"]),
    }
    report["feedback_sha256"] = _sha_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    )
    return report
