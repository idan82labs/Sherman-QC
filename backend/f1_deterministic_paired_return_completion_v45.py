"""Deterministic paired-return completion with incumbent-first output NMS."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from backend.f1_paired_return_completion import complete_paired_return_axis
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


PROFILE = "F1_V45_DETERMINISTIC_PAIRED_RETURN_COMPLETION_V1"
COMPLEX_ROUTE = "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("candidate_id") or ""), _canonical(row)


def prediction_row(prediction: dict[str, Any]) -> dict[str, Any]:
    row = deepcopy(prediction)
    row["candidate_id"] = str(
        prediction.get("source_hypothesis_id") or prediction.get("bend_id") or ""
    )
    row["span_endpoints"] = [prediction["line_start"], prediction["line_end"]]
    return row


def incumbent_first_output_nms(
    incumbents: list[dict[str, Any]], additions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run ordinary geometric NMS with incumbents strictly preceding additions."""

    accepted: list[dict[str, Any]] = []
    suppressed: list[str] = []
    for row in [*incumbents, *sorted(additions, key=_row_key)]:
        duplicate = False
        for incumbent in accepted:
            metrics = pair_metrics(endpoints(row), endpoints(incumbent))
            if (
                float(metrics["axis_angle_deg"]) <= 7.0
                and float(metrics["transverse_offset_mm"]) <= 6.0
                and float(metrics["shorter_span_overlap"]) >= 0.20
            ):
                duplicate = True
                break
        if duplicate:
            suppressed.append(str(row.get("candidate_id") or ""))
        else:
            accepted.append(row)
    return accepted, suppressed


def _stable_bend_id(candidate_id: str) -> str:
    return "TRV45-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]


def serialize_prediction(row: dict[str, Any], route: str) -> dict[str, Any]:
    if not str(row.get("candidate_id") or "").startswith("collinear_fragment_stitch::"):
        prediction = deepcopy(row)
        prediction.pop("candidate_id", None)
        prediction.pop("span_endpoints", None)
        return prediction
    line = row["span_endpoints"]
    candidate_id = str(row["candidate_id"])
    return {
        "bend_id": _stable_bend_id(candidate_id),
        "line_start": line[0],
        "line_end": line[1],
        "selection_route": route,
        "selector_score": row.get("selector_score", row.get("score")),
        "source_hypothesis_id": candidate_id,
        "source_kind": "paired_return_completion",
        "stitched_member_candidate_ids": list(row.get("stitched_member_candidate_ids", [])),
    }


def compose_v45(
    v44_result: dict[str, Any], line_objects: list[dict[str, Any]], bbox_span: float
) -> dict[str, Any]:
    """Complete at most one counterpart using canonical source-row ordering."""

    result = deepcopy(v44_result)
    route = str(result.get("route") or "")
    incumbents = [prediction_row(value) for value in result.get("predictions", [])]
    completion = {"selected": incumbents, "changes": []}
    if route == COMPLEX_ROUTE:
        canonical_rows = sorted((deepcopy(value) for value in line_objects), key=_row_key)
        completion = complete_paired_return_axis(incumbents, canonical_rows, float(bbox_span))
    incumbent_ids = {str(value.get("candidate_id") or "") for value in incumbents}
    additions = [
        value for value in completion["selected"]
        if str(value.get("candidate_id") or "") not in incumbent_ids
    ]
    accepted, suppressed = incumbent_first_output_nms(incumbents, additions)
    accepted_ids = {str(value.get("candidate_id") or "") for value in accepted}
    changes = [
        value for value in completion["changes"]
        if str(value.get("candidate_id") or "") in accepted_ids
    ]
    predictions = [serialize_prediction(value, route) for value in accepted]
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v45_deterministic_paired_return_result",
            "profile": PROFILE,
            "v44_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "paired_return_completion_changes": changes,
            "post_completion_nms_suppressed_candidate_ids": suppressed,
            "canonical_all_rows_order": True,
            "incumbent_first_output_nms": True,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {
        "profile": PROFILE,
        "predictions": predictions,
        "changes": changes,
        "suppressed": suppressed,
    }
    result["v45_result_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return result
