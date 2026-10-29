"""Suppress near-equal short-span cross-source duplicates of one physical hinge."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

from backend.f1_deterministic_paired_return_completion_v45 import prediction_row
from backend.f1_elongated_lane_consolidation_v46 import _canonical, _row_key
from backend.f1_nested_partial_duplicate_nms_v48 import _length
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


PROFILE = "F1_V49_SHORT_SPAN_CROSS_SOURCE_DUPLICATE_NMS_V1"
MAXIMUM_PAIR_SPAN_TO_LONGEST_RATIO = 0.15
MINIMUM_PAIR_LENGTH_RATIO = 0.80
MAXIMUM_AXIS_ANGLE_DEG = 15.0
MAXIMUM_TRANSVERSE_OFFSET_MM = 2.0
MINIMUM_SHORTER_SPAN_OVERLAP = 0.90


def compose_v49(v48_result: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic long-first NMS to short, cross-source duplicates."""

    result = deepcopy(v48_result)
    rows = [prediction_row(value) for value in result.get("predictions", [])]
    longest_length = max((_length(row) for row in rows), default=0.0)
    rows.sort(key=lambda row: (-_length(row), _row_key(row)))
    accepted: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        row_length = _length(row)
        suppressor: dict[str, Any] | None = None
        suppressor_metrics: dict[str, float] | None = None
        for incumbent in accepted:
            incumbent_length = _length(incumbent)
            shorter = min(row_length, incumbent_length)
            longer = max(row_length, incumbent_length)
            metrics = pair_metrics(endpoints(row), endpoints(incumbent))
            if (
                str(row.get("source_kind") or "")
                != str(incumbent.get("source_kind") or "")
                and longer <= MAXIMUM_PAIR_SPAN_TO_LONGEST_RATIO * longest_length
                and shorter >= MINIMUM_PAIR_LENGTH_RATIO * longer
                and float(metrics["axis_angle_deg"]) <= MAXIMUM_AXIS_ANGLE_DEG
                and float(metrics["transverse_offset_mm"])
                <= MAXIMUM_TRANSVERSE_OFFSET_MM
                and float(metrics["shorter_span_overlap"])
                >= MINIMUM_SHORTER_SPAN_OVERLAP
            ):
                suppressor = incumbent
                suppressor_metrics = metrics
                break
        if suppressor is None:
            accepted.append(row)
            continue
        retained_length = _length(suppressor)
        changes.append(
            {
                "suppressed_candidate_id": str(row.get("candidate_id") or ""),
                "retained_candidate_id": str(suppressor.get("candidate_id") or ""),
                "suppressed_source_kind": str(row.get("source_kind") or ""),
                "retained_source_kind": str(suppressor.get("source_kind") or ""),
                "suppressed_length_mm": round(row_length, 6),
                "retained_length_mm": round(retained_length, 6),
                "pair_span_to_longest_ratio": round(
                    max(row_length, retained_length) / max(longest_length, 1e-12), 12
                ),
                "pair_length_ratio": round(
                    min(row_length, retained_length) / max(row_length, retained_length), 12
                ),
                "axis_angle_deg": round(float(suppressor_metrics["axis_angle_deg"]), 12),
                "transverse_offset_mm": round(
                    float(suppressor_metrics["transverse_offset_mm"]), 12
                ),
                "shorter_span_overlap": round(
                    float(suppressor_metrics["shorter_span_overlap"]), 12
                ),
            }
        )

    accepted.sort(key=_row_key)
    predictions = []
    for row in accepted:
        prediction = deepcopy(row)
        prediction.pop("candidate_id", None)
        prediction.pop("span_endpoints", None)
        predictions.append(prediction)
    changes.sort(key=lambda value: str(value["suppressed_candidate_id"]))
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v49_short_span_cross_source_duplicate_nms_result",
            "profile": PROFILE,
            "v48_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "short_span_cross_source_duplicate_nms_changes": changes,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "changes": changes}
    result["v49_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
