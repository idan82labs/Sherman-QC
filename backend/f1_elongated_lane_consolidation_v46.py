"""Source-only cleanup for highly elongated open-frame scan predictions.

V46 is deliberately narrow.  It activates only when the scan's longest extent
is at least four times the geometric mean of its two cross-section extents and the selected set contains a
dominant family of at least three substantial, mutually parallel axes.  Under
that gate it removes only ultra-short predictions perpendicular to the
dominant family and allows same-lane fragments to bridge a bounded fraction of
the scan length.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import numpy as np

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments
from backend.f1_deterministic_paired_return_completion_v45 import prediction_row
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


PROFILE = "F1_V46_ELONGATED_LANE_CONSOLIDATION_V1"
COMPLEX_ROUTE = "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair"
MINIMUM_ASPECT_RATIO = 4.0
MINIMUM_DOMINANT_MEMBER_FRACTION = 0.15
MINIMUM_DOMINANT_MEMBER_COUNT = 3
MAXIMUM_DOMINANT_AXIS_ANGLE_DEG = 5.0
MAXIMUM_TERMINAL_LENGTH_FRACTION = 0.05
MINIMUM_TERMINAL_AXIS_ANGLE_DEG = 75.0
MAXIMUM_STITCH_AXIS_ANGLE_DEG = 2.0
MAXIMUM_STITCH_TRANSVERSE_OFFSET_MM = 5.0
MAXIMUM_STITCH_GAP_FRACTION = 0.30
MINIMUM_STITCH_SPAN_GAIN = 1.25


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("candidate_id") or ""), _canonical(row)


def _length(row: dict[str, Any]) -> float:
    line = endpoints(row)
    return float(np.linalg.norm(line[1] - line[0]))


def _connected_components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current].intersection(unseen), reverse=True):
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _dominant_component(rows: list[dict[str, Any]], longest_span: float) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if _length(row) >= MINIMUM_DOMINANT_MEMBER_FRACTION * longest_span
    ]
    adjacency = [set() for _ in eligible]
    for local_first, first in enumerate(eligible):
        for local_second in range(local_first):
            second = eligible[local_second]
            metrics = pair_metrics(endpoints(rows[first]), endpoints(rows[second]))
            if float(metrics["axis_angle_deg"]) <= MAXIMUM_DOMINANT_AXIS_ANGLE_DEG:
                adjacency[local_first].add(local_second)
                adjacency[local_second].add(local_first)
    components = [
        [eligible[local] for local in component]
        for component in _connected_components(adjacency)
    ]
    if not components:
        return []
    components.sort(
        key=lambda component: (
            -sum(_length(rows[index]) for index in component),
            [str(rows[index].get("candidate_id") or "") for index in component],
        )
    )
    winner = components[0]
    if len(winner) < MINIMUM_DOMINANT_MEMBER_COUNT:
        return []
    return winner


def _stable_bend_id(candidate_id: str) -> str:
    return "TRV46-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]


def _serialize(row: dict[str, Any], route: str) -> dict[str, Any]:
    if str(row.get("candidate_source") or "") != "collinear_fragment_stitch":
        prediction = deepcopy(row)
        prediction.pop("candidate_id", None)
        prediction.pop("span_endpoints", None)
        return prediction
    source_id = str(row.get("candidate_id") or "")
    candidate_id = source_id.replace(
        "collinear_fragment_stitch::", "elongated_lane_consolidation::", 1
    )
    line = row["span_endpoints"]
    return {
        "bend_id": _stable_bend_id(candidate_id),
        "line_start": line[0],
        "line_end": line[1],
        "selection_route": route,
        "selector_score": row.get("selector_score", row.get("score")),
        "source_hypothesis_id": candidate_id,
        "source_kind": "elongated_lane_consolidation",
        "stitched_member_candidate_ids": list(row.get("stitched_member_candidate_ids", [])),
    }


def compose_v46(v45_result: dict[str, Any], scan_bbox_mm: list[float]) -> dict[str, Any]:
    """Apply the bounded elongated-frame cleanup to a frozen V45 result."""

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
        route == COMPLEX_ROUTE
        and aspect_ratio >= MINIMUM_ASPECT_RATIO
        and len(dominant) >= MINIMUM_DOMINANT_MEMBER_COUNT
    )

    suppressed_ids: list[str] = []
    retained = rows
    stitch = {"selected": retained, "changes": []}
    if gate_active:
        dominant_lines = [endpoints(rows[index]) for index in dominant]
        retained = []
        for row in rows:
            length = _length(row)
            minimum_angle = min(
                float(pair_metrics(endpoints(row), line)["axis_angle_deg"])
                for line in dominant_lines
            )
            is_terminal_artifact = (
                length <= MAXIMUM_TERMINAL_LENGTH_FRACTION * longest_span
                and minimum_angle >= MINIMUM_TERMINAL_AXIS_ANGLE_DEG
            )
            if is_terminal_artifact:
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
        "scan_aspect_ratio": round(aspect_ratio, 12),
        "longest_scan_span_mm": round(longest_span, 6),
        "dominant_member_candidate_ids": sorted(
            str(rows[index].get("candidate_id") or "") for index in dominant
        ),
        "suppressed_terminal_candidate_ids": sorted(suppressed_ids),
        "stitch_changes": stitch["changes"],
    }
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v46_elongated_lane_consolidation_result",
            "profile": PROFILE,
            "v45_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "elongated_lane_consolidation_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    payload = {"profile": PROFILE, "predictions": predictions, "audit": audit}
    result["v46_result_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    return result
