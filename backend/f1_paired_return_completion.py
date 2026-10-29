"""Source-only completion of a missing member of a paired return profile."""

from __future__ import annotations

import numpy as np

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments
from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import pair_metrics


ELIGIBLE_SOURCES = {
    "cylindrical_standard",
    "cylindrical_wide",
    "scan_mesh_adjacency",
    "plane_lattice",
}


def _represented(candidate: dict, selected: list[dict]) -> bool:
    for row in selected:
        metrics = pair_metrics(endpoints(candidate), endpoints(row))
        if (
            float(metrics["axis_angle_deg"]) <= 7.0
            and float(metrics["transverse_offset_mm"]) <= 6.0
            and float(metrics["shorter_span_overlap"]) >= 0.20
        ):
            return True
    return False


def _configuration(selected: list[dict], bbox_span: float):
    if len(selected) != 3:
        return None
    pairs = []
    for first in range(3):
        for second in range(first):
            metrics = pair_metrics(endpoints(selected[first]), endpoints(selected[second]))
            if float(metrics["axis_angle_deg"]) > 2.0:
                continue
            separation = float(metrics["transverse_offset_mm"])
            if not (0.015 * bbox_span <= separation <= 0.08 * bbox_span):
                continue
            if float(metrics["shorter_span_overlap"]) < 0.75:
                continue
            pairs.append((separation, second, first))
    if len(pairs) != 1:
        return None
    separation, first, second = pairs[0]
    far = ({0, 1, 2} - {first, second}).pop()
    for close in (first, second):
        metrics = pair_metrics(endpoints(selected[far]), endpoints(selected[close]))
        if (
            float(metrics["axis_angle_deg"]) > 2.0
            or float(metrics["transverse_offset_mm"]) < 0.30 * bbox_span
        ):
            return None
    return separation, far


def complete_paired_return_axis(
    selected: list[dict], all_rows: list[dict], bbox_span: float
) -> dict:
    """Add at most one stitched counterpart lane from scan-only evidence."""

    configuration = _configuration(selected, bbox_span)
    if configuration is None:
        return {"selected": list(selected), "changes": []}
    reference_separation, far_index = configuration
    far = selected[far_index]
    eligible = []
    for row in all_rows:
        if str(row.get("representative_source") or "") not in ELIGIBLE_SOURCES:
            continue
        support = [
            float(row.get("finite_support_fraction") or 0.0),
            float(row.get("finite_support_longest_run_fraction") or 0.0),
            float(row.get("endpoint_support_fraction") or 0.0),
        ]
        if min(support) < 0.80:
            continue
        if float(row.get("two_sided_normal_angle_deg") or 0.0) < 70.0:
            continue
        if float(row.get("two_sided_transition_score") or 0.0) < 0.25:
            continue
        if float(row.get("span_mm") or 0.0) < 0.04 * bbox_span:
            continue
        metrics = pair_metrics(endpoints(far), endpoints(row))
        if float(metrics["axis_angle_deg"]) > 2.0:
            continue
        separation = float(metrics["transverse_offset_mm"])
        if not (0.70 * reference_separation <= separation <= 1.50 * reference_separation):
            continue
        if _represented(row, selected):
            continue
        eligible.append(row)

    stitched = stitch_collinear_selected_fragments(
        eligible,
        maximum_axis_angle_deg=2.0,
        maximum_transverse_offset_mm=6.0,
        maximum_axial_gap_mm=0.12 * bbox_span,
        minimum_span_gain=1.25,
    )
    choices = []
    changed_ids = {change["fitted_candidate_id"] for change in stitched["changes"]}
    far_span = float(np.linalg.norm(endpoints(far)[1] - endpoints(far)[0]))
    for row in stitched["selected"]:
        if str(row.get("candidate_id") or "") not in changed_ids:
            continue
        span = float(np.linalg.norm(endpoints(row)[1] - endpoints(row)[0]))
        if span < 0.60 * far_span:
            continue
        metrics = pair_metrics(endpoints(far), endpoints(row))
        choices.append(
            (
                abs(float(metrics["transverse_offset_mm"]) - reference_separation),
                -span,
                str(row.get("candidate_id") or ""),
                row,
            )
        )
    if not choices:
        return {"selected": list(selected), "changes": []}
    _, _, _, recovered = sorted(choices, key=lambda item: item[:3])[0]
    metrics = pair_metrics(endpoints(far), endpoints(recovered))
    return {
        "selected": [*selected, recovered],
        "changes": [
            {
                "candidate_id": str(recovered.get("candidate_id") or ""),
                "repair_basis": "paired_return_profile_stitched_counterpart",
                "reference_candidate_id": str(far.get("candidate_id") or ""),
                "reference_pair_separation_mm": round(reference_separation, 6),
                "counterpart_separation_mm": round(
                    float(metrics["transverse_offset_mm"]), 6
                ),
                "span_mm": round(
                    float(np.linalg.norm(endpoints(recovered)[1] - endpoints(recovered)[0])),
                    6,
                ),
            }
        ],
    }
