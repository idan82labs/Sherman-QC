"""Route-independent successor to the prospectively falsified V46 cleanup."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import numpy as np

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments
from backend.f1_deterministic_paired_return_completion_v45 import prediction_row
from backend.f1_elongated_lane_consolidation_v46 import (
    MAXIMUM_STITCH_AXIS_ANGLE_DEG,
    MAXIMUM_STITCH_GAP_FRACTION,
    MAXIMUM_TERMINAL_LENGTH_FRACTION,
    MINIMUM_ASPECT_RATIO,
    MINIMUM_DOMINANT_MEMBER_COUNT,
    MINIMUM_STITCH_SPAN_GAIN,
    _canonical,
    _dominant_component,
    _length,
    _row_key,
)


PROFILE = "F1_V47_ROUTE_INDEPENDENT_ELONGATED_LANE_V1"
# Reuse the retained ordinary V27 transverse tolerance rather than tuning it
# to the P03 development case.
MAXIMUM_STITCH_TRANSVERSE_OFFSET_MM = 6.0


def _stable_bend_id(candidate_id: str) -> str:
    return "TRV47-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]


def _serialize(row: dict[str, Any], route: str) -> dict[str, Any]:
    if str(row.get("candidate_source") or "") != "collinear_fragment_stitch":
        prediction = deepcopy(row)
        prediction.pop("candidate_id", None)
        prediction.pop("span_endpoints", None)
        return prediction
    source_id = str(row.get("candidate_id") or "")
    candidate_id = source_id.replace(
        "collinear_fragment_stitch::", "route_independent_elongated_lane::", 1
    )
    line = row["span_endpoints"]
    return {
        "bend_id": _stable_bend_id(candidate_id),
        "line_start": line[0],
        "line_end": line[1],
        "selection_route": route,
        "selector_score": row.get("selector_score", row.get("score")),
        "source_hypothesis_id": candidate_id,
        "source_kind": "route_independent_elongated_lane",
        "stitched_member_candidate_ids": list(row.get("stitched_member_candidate_ids", [])),
    }


def compose_v47(v45_result: dict[str, Any], scan_bbox_mm: list[float]) -> dict[str, Any]:
    """Consolidate elongated physical lanes without consulting router identity."""

    result = deepcopy(v45_result)
    route = str(result.get("route") or "")
    rows = sorted(
        (prediction_row(value) for value in result.get("predictions", [])),
        key=_row_key,
    )
    ordered_bbox = sorted((float(value) for value in scan_bbox_mm), reverse=True)
    longest_span = ordered_bbox[0] if ordered_bbox else 0.0
    cross_section_scale = (
        float(np.sqrt(ordered_bbox[1] * ordered_bbox[2]))
        if len(ordered_bbox) > 2
        else 0.0
    )
    aspect_ratio = longest_span / max(cross_section_scale, 1e-12)
    dominant = _dominant_component(rows, longest_span) if longest_span > 0.0 else []
    gate_active = (
        aspect_ratio >= MINIMUM_ASPECT_RATIO
        and len(dominant) >= MINIMUM_DOMINANT_MEMBER_COUNT
    )

    suppressed_ids: list[str] = []
    retained = rows
    stitch = {"selected": retained, "changes": []}
    if gate_active:
        retained = []
        for row in rows:
            if _length(row) <= MAXIMUM_TERMINAL_LENGTH_FRACTION * longest_span:
                suppressed_ids.append(str(row.get("candidate_id") or ""))
            else:
                retained.append(row)
        stitch = stitch_collinear_selected_fragments(
            retained,
            maximum_axis_angle_deg=MAXIMUM_STITCH_AXIS_ANGLE_DEG,
            maximum_transverse_offset_mm=MAXIMUM_STITCH_TRANSVERSE_OFFSET_MM,
            maximum_axial_gap_mm=MAXIMUM_STITCH_GAP_FRACTION * longest_span,
            minimum_span_gain=MINIMUM_STITCH_SPAN_GAIN,
        )

    predictions = [_serialize(row, route) for row in stitch["selected"]]
    audit = {
        "gate_active": gate_active,
        "source_route": route,
        "scan_aspect_ratio": round(aspect_ratio, 12),
        "longest_scan_span_mm": round(longest_span, 6),
        "dominant_member_candidate_ids": sorted(
            str(rows[index].get("candidate_id") or "") for index in dominant
        ),
        "suppressed_microfragment_candidate_ids": sorted(suppressed_ids),
        "stitch_changes": stitch["changes"],
    }
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v47_route_independent_elongated_lane_result",
            "profile": PROFILE,
            "v45_retained": True,
            "v46_prospective_failure_addressed": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "route_independent_elongated_lane_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "audit": audit}
    result["v47_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
