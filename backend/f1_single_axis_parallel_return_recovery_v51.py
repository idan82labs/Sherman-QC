"""Recover one strong parallel return axis from a distinct cylindrical family."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any

import numpy as np

from backend.f1_deterministic_paired_return_completion_v45 import (
    _canonical,
    prediction_row,
)
from backend.f1_elongated_lane_consolidation_v46 import _row_key
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


PROFILE = "F1_V51_SINGLE_AXIS_PARALLEL_RETURN_RECOVERY_V1"
MINIMUM_INCUMBENT_TO_BBOX_RATIO = 0.75
MINIMUM_CANDIDATE_TO_INCUMBENT_RATIO = 0.75
MAXIMUM_AXIS_ANGLE_DEG = 2.0
MINIMUM_TRANSVERSE_OFFSET_MM = 8.0
MAXIMUM_TRANSVERSE_OFFSET_TO_INCUMBENT_RATIO = 0.08
MINIMUM_SHORTER_SPAN_OVERLAP = 0.80
_CYLINDRICAL_FAMILY = re.compile(r"^(CB\d+)_")


def _length(row: dict[str, Any]) -> float:
    line = endpoints(row)
    return float(np.linalg.norm(line[1] - line[0]))


def _family(candidate_id: str) -> str | None:
    leaf = candidate_id.rsplit("::", 1)[-1]
    match = _CYLINDRICAL_FAMILY.match(leaf)
    return match.group(1) if match else None


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    line = endpoints(row)
    return {
        "bend_id": "TRV51-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20],
        "line_start": line[0].tolist(),
        "line_end": line[1].tolist(),
        "selection_route": "single_axis_parallel_return_recovery_v51",
        "selector_score": row.get("score"),
        "source_hypothesis_id": candidate_id,
        "source_kind": str(row.get("representative_source") or "cylindrical_standard"),
    }


def compose_v51(v49_result: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    """Add at most one source-only parallel return axis under a strict gate."""

    result = deepcopy(v49_result)
    predictions = list(result.get("predictions", []))
    audit: dict[str, Any] = {
        "gate_active": False,
        "reason": "requires_exactly_one_prediction",
        "eligible_candidate_count": 0,
        "selected_candidate_id": None,
    }
    if len(predictions) == 1:
        incumbent = prediction_row(predictions[0])
        incumbent_id = str(incumbent.get("candidate_id") or "")
        incumbent_family = _family(incumbent_id)
        incumbent_length = _length(incumbent)
        bbox_span = max((float(value) for value in pool.get("scan_bbox_mm") or ()), default=0.0)
        if (
            str(incumbent.get("source_kind") or "") == "cylindrical_standard"
            and incumbent_family is not None
            and incumbent_length >= MINIMUM_INCUMBENT_TO_BBOX_RATIO * bbox_span
        ):
            eligible: list[tuple[float, float, tuple[str, str], dict[str, Any], dict[str, float]]] = []
            for value in pool.get("line_objects", []):
                row = deepcopy(value)
                candidate_id = str(row.get("candidate_id") or "")
                family = _family(candidate_id)
                if (
                    str(row.get("representative_source") or "")
                    != "cylindrical_standard"
                    or family is None
                    or family == incumbent_family
                ):
                    continue
                candidate_length = _length(row)
                metrics = pair_metrics(endpoints(row), endpoints(incumbent))
                if (
                    candidate_length >= MINIMUM_CANDIDATE_TO_INCUMBENT_RATIO * incumbent_length
                    and float(metrics["axis_angle_deg"]) <= MAXIMUM_AXIS_ANGLE_DEG
                    and float(metrics["transverse_offset_mm"])
                    >= MINIMUM_TRANSVERSE_OFFSET_MM
                    and float(metrics["transverse_offset_mm"])
                    <= MAXIMUM_TRANSVERSE_OFFSET_TO_INCUMBENT_RATIO * incumbent_length
                    and float(metrics["shorter_span_overlap"])
                    >= MINIMUM_SHORTER_SPAN_OVERLAP
                ):
                    eligible.append(
                        (
                            -float(row.get("score") or 0.0),
                            -candidate_length,
                            _row_key(row),
                            row,
                            metrics,
                        )
                    )
            eligible.sort(key=lambda value: value[:3])
            audit["eligible_candidate_count"] = len(eligible)
            if eligible:
                selected = eligible[0][3]
                metrics = eligible[0][4]
                selected_id = str(selected.get("candidate_id") or "")
                predictions.append(_serialize(selected))
                audit.update(
                    {
                        "gate_active": True,
                        "reason": "distinct_parallel_cylindrical_family_recovered",
                        "incumbent_candidate_id": incumbent_id,
                        "incumbent_family": incumbent_family,
                        "selected_candidate_id": selected_id,
                        "selected_family": _family(selected_id),
                        "selected_score": float(selected.get("score") or 0.0),
                        "selected_length_mm": round(_length(selected), 6),
                        "axis_angle_deg": round(float(metrics["axis_angle_deg"]), 12),
                        "transverse_offset_mm": round(
                            float(metrics["transverse_offset_mm"]), 12
                        ),
                        "shorter_span_overlap": round(
                            float(metrics["shorter_span_overlap"]), 12
                        ),
                    }
                )
            else:
                audit["reason"] = "no_eligible_distinct_parallel_family"
        else:
            audit["reason"] = "incumbent_not_long_cylindrical_axis"

    predictions.sort(
        key=lambda value: _row_key(prediction_row(value))
    )
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result",
            "profile": PROFILE,
            "v49_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "single_axis_parallel_return_recovery_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "audit": audit}
    result["v51_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
