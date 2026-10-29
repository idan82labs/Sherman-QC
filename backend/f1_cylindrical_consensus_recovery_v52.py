"""Recover long, unrepresented multi-source cylindrical bend axes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any

import numpy as np

from backend.f1_deterministic_paired_return_completion_v45 import _canonical
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics


PROFILE = "F1_V52_CYLINDRICAL_CONSENSUS_RECOVERY_V1"
MINIMUM_CLUSTER_SOURCE_COUNT = 3
MINIMUM_SPAN_TO_BBOX_RATIO = 0.50
MAXIMUM_NMS_AXIS_ANGLE_DEG = 7.0
MAXIMUM_NMS_TRANSVERSE_OFFSET_MM = 6.0
MINIMUM_NMS_SHORTER_SPAN_OVERLAP = 0.50
REQUIRED_SOURCES = frozenset(("cylindrical_standard", "cylindrical_wide"))
SOURCE_PRIORITY = {
    "cylindrical_standard": 4,
    "cylindrical_wide": 3,
    "scan_mesh_adjacency": 2,
    "plane_lattice": 1,
}


def _length(row: dict[str, Any]) -> float:
    line = endpoints(row)
    return float(np.linalg.norm(line[1] - line[0]))


def _rank_score(row: dict[str, Any], bbox_span: float) -> float:
    return (
        2.0 * int(row.get("cluster_source_count") or 0)
        + SOURCE_PRIORITY.get(str(row.get("representative_source") or ""), 0)
        + 2.0 * min(_length(row) / max(bbox_span, 1e-9), 1.5)
        + float(row.get("two_sided_transition_score") or 0.0)
        + 0.2 * math.log1p(float(row.get("support_count") or 0.0))
    )


def _represented(candidate: dict[str, Any], predictions: list[dict[str, Any]]) -> bool:
    for prediction in predictions:
        metrics = pair_metrics(_endpoints(candidate), _endpoints(prediction))
        if (
            metrics["axis_angle_deg"] <= MAXIMUM_NMS_AXIS_ANGLE_DEG
            and metrics["transverse_offset_mm"] <= MAXIMUM_NMS_TRANSVERSE_OFFSET_MM
            and metrics["shorter_span_overlap"] >= MINIMUM_NMS_SHORTER_SPAN_OVERLAP
        ):
            return True
    return False


def _serialize(row: dict[str, Any], score: float) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or "")
    line = endpoints(row)
    return {
        "bend_id": "TRV52-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20],
        "line_start": line[0].tolist(),
        "line_end": line[1].tolist(),
        "selection_route": "cylindrical_consensus_recovery_v52",
        "selector_score": round(score, 12),
        "source_hypothesis_id": candidate_id,
        "source_kind": str(row.get("representative_source") or "cylindrical_consensus"),
    }


def compose_v52(v51_result: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    if v51_result.get("source_only") is not True or v51_result.get("research_only") is not True:
        raise ValueError("V52 requires a source-only research V51 input")
    result = deepcopy(v51_result)
    predictions = [deepcopy(row) for row in result.get("predictions") or []]
    bbox_span = max((float(value) for value in pool.get("scan_bbox_mm") or ()), default=0.0)
    eligible = []
    for row in pool.get("line_objects") or []:
        row = deepcopy(row)
        sources = set(str(value) for value in row.get("cluster_sources") or ())
        if (
            REQUIRED_SOURCES.issubset(sources)
            and int(row.get("cluster_source_count") or 0) >= MINIMUM_CLUSTER_SOURCE_COUNT
            and _length(row) >= MINIMUM_SPAN_TO_BBOX_RATIO * bbox_span
        ):
            eligible.append((_rank_score(row, bbox_span), str(row.get("candidate_id") or ""), row))
    eligible.sort(key=lambda value: (-value[0], value[1]))
    additions = []
    for score, candidate_id, row in eligible:
        if _represented(row, predictions):
            continue
        prediction = _serialize(row, score)
        predictions.append(prediction)
        additions.append(
            {
                "candidate_id": candidate_id,
                "representative_source": str(row.get("representative_source") or ""),
                "cluster_source_count": int(row.get("cluster_source_count") or 0),
                "span_to_bbox_ratio": round(_length(row) / max(bbox_span, 1e-9), 12),
                "rank_score": round(score, 12),
            }
        )
    predictions.sort(key=lambda row: str(row.get("bend_id") or ""))
    audit = {
        "eligible_candidate_count": len(eligible),
        "recovered_candidate_count": len(additions),
        "recovered_candidates": additions,
        "minimum_cluster_source_count": MINIMUM_CLUSTER_SOURCE_COUNT,
        "minimum_span_to_bbox_ratio": MINIMUM_SPAN_TO_BBOX_RATIO,
        "maximum_nms_axis_angle_deg": MAXIMUM_NMS_AXIS_ANGLE_DEG,
        "maximum_nms_transverse_offset_mm": MAXIMUM_NMS_TRANSVERSE_OFFSET_MM,
        "minimum_nms_shorter_span_overlap": MINIMUM_NMS_SHORTER_SPAN_OVERLAP,
    }
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v52_cylindrical_consensus_recovery_result",
            "profile": PROFILE,
            "v51_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "cylindrical_consensus_recovery_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "audit": audit}
    result["v52_result_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return result
