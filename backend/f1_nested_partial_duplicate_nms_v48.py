"""Suppress short, substantially covered duplicates of longer bend axes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import numpy as np

from backend.f1_deterministic_paired_return_completion_v45 import prediction_row
from backend.f1_elongated_lane_consolidation_v46 import _canonical, _row_key
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


PROFILE = "F1_V48_NESTED_PARTIAL_DUPLICATE_NMS_V1"
MAXIMUM_AXIS_ANGLE_DEG = 10.0
MAXIMUM_TRANSVERSE_OFFSET_MM = 8.0
MINIMUM_SHORTER_SPAN_OVERLAP = 0.70
MAXIMUM_LENGTH_RATIO = 0.60


def _length(row: dict[str, Any]) -> float:
    line = endpoints(row)
    return float(np.linalg.norm(line[1] - line[0]))


def compose_v48(v47_result: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic long-first NMS only to nested partial duplicates."""

    result = deepcopy(v47_result)
    rows = [prediction_row(value) for value in result.get("predictions", [])]
    rows.sort(key=lambda row: (-_length(row), _row_key(row)))
    accepted: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        row_length = _length(row)
        suppressor: dict[str, Any] | None = None
        suppressor_metrics: dict[str, float] | None = None
        for incumbent in accepted:
            incumbent_length = _length(incumbent)
            metrics = pair_metrics(endpoints(row), endpoints(incumbent))
            if (
                row_length <= MAXIMUM_LENGTH_RATIO * incumbent_length
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
        changes.append(
            {
                "suppressed_candidate_id": str(row.get("candidate_id") or ""),
                "retained_candidate_id": str(suppressor.get("candidate_id") or ""),
                "suppressed_length_mm": round(row_length, 6),
                "retained_length_mm": round(_length(suppressor), 6),
                "length_ratio": round(row_length / max(_length(suppressor), 1e-12), 12),
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
            "kind": "sherman_qc_f1_v48_nested_partial_duplicate_nms_result",
            "profile": PROFILE,
            "v47_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "nested_partial_duplicate_nms_changes": changes,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "changes": changes}
    result["v48_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
