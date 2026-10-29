"""Recover fully supported short bend segments from supported detector seeds.

The generated line-object pool sometimes contains a short cylindrical re-fit
of a much longer detector bend seed.  This helper retains only re-fits whose
underlying detector seed is itself strongly supported by the scan.  Interface
births and re-fits that acquire support only after cylindrical fitting fail
closed.

The rule does not consume a part identity, candidate identity, expected bend
count, model score, or manual truth.  Candidate identifiers are used only to
join a line object's declared provenance to the corresponding generated seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ScanSupportedShortSeedThresholds:
    maximum_span_mm: float = 15.0
    minimum_line_finite_support_fraction: float = 0.99
    minimum_line_endpoint_support_fraction: float = 0.99
    minimum_seed_finite_support_fraction: float = 0.90
    minimum_seed_endpoint_support_fraction: float = 2.0 / 3.0
    duplicate_axis_angle_deg: float = 5.0
    duplicate_transverse_offset_mm: float = 4.0
    duplicate_shorter_span_overlap: float = 0.60


def _float(record: Mapping[str, Any], field: str) -> float | None:
    try:
        value = float(record.get(field))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _endpoints(record: Mapping[str, Any]) -> np.ndarray | None:
    endpoints = record.get("span_endpoints")
    if endpoints is None and record.get("line_start") is not None:
        endpoints = [record.get("line_start"), record.get("line_end")]
    try:
        result = np.asarray(endpoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (2, 3) or not np.isfinite(result).all():
        return None
    if float(np.linalg.norm(result[1] - result[0])) <= 1.0e-9:
        return None
    return result


def _is_duplicate(
    candidate: Mapping[str, Any],
    retained: Mapping[str, Any],
    thresholds: ScanSupportedShortSeedThresholds,
) -> bool:
    candidate_endpoints = _endpoints(candidate)
    retained_endpoints = _endpoints(retained)
    if candidate_endpoints is None or retained_endpoints is None:
        return False
    candidate_vector = candidate_endpoints[1] - candidate_endpoints[0]
    retained_vector = retained_endpoints[1] - retained_endpoints[0]
    candidate_span = float(np.linalg.norm(candidate_vector))
    retained_span = float(np.linalg.norm(retained_vector))
    candidate_axis = candidate_vector / candidate_span
    retained_axis = retained_vector / retained_span
    cosine = float(np.clip(abs(np.dot(candidate_axis, retained_axis)), 0.0, 1.0))
    if degrees(acos(cosine)) > thresholds.duplicate_axis_angle_deg:
        return False

    retained_midpoint = np.mean(retained_endpoints, axis=0)
    candidate_midpoint = np.mean(candidate_endpoints, axis=0)
    midpoint_delta = candidate_midpoint - retained_midpoint
    transverse = midpoint_delta - np.dot(midpoint_delta, retained_axis) * retained_axis
    if float(np.linalg.norm(transverse)) > thresholds.duplicate_transverse_offset_mm:
        return False
    projected = sorted(
        float(np.dot(point - retained_midpoint, retained_axis))
        for point in candidate_endpoints
    )
    overlap = max(
        0.0,
        min(0.5 * retained_span, projected[1])
        - max(-0.5 * retained_span, projected[0]),
    )
    return overlap / min(candidate_span, retained_span) >= (
        thresholds.duplicate_shorter_span_overlap
    )


def complete_scan_supported_short_seeds(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
    generated_candidates: Sequence[Mapping[str, Any]],
    thresholds: ScanSupportedShortSeedThresholds = ScanSupportedShortSeedThresholds(),
) -> dict[str, Any]:
    """Append short cylindrical re-fits backed by a supported detector seed.

    The accepted line object must be a fully scan-supported, at-most-15 mm
    consensus between exactly the detector-seed and standard-cylinder source
    families.  Its detector-seed member must be an existing detector bend (not
    a generated interface birth), retain at least 90% finite support, and have
    at least two of three endpoint probes supported.
    """

    output = [dict(record) for record in selected]
    raw_by_id = {
        str(record.get("candidate_id")): record
        for record in generated_candidates
        if record.get("candidate_id") is not None
    }
    additions: list[dict[str, Any]] = []
    ranked = sorted(
        (dict(record) for record in line_objects),
        key=lambda record: str(record.get("bend_id") or record.get("candidate_id") or ""),
    )
    for record in ranked:
        sources = {str(value) for value in record.get("cluster_sources") or ()}
        if sources != {"bend_hypothesis", "cylindrical_standard"}:
            continue
        span = _float(record, "span_mm")
        line_finite = _float(record, "finite_support_fraction")
        line_endpoint = _float(record, "endpoint_support_fraction")
        if span is None or span <= 0.0 or span > thresholds.maximum_span_mm:
            continue
        if (
            line_finite is None
            or line_finite < thresholds.minimum_line_finite_support_fraction
            or line_endpoint is None
            or line_endpoint < thresholds.minimum_line_endpoint_support_fraction
        ):
            continue
        seed_member_ids = [
            str(value)
            for value in record.get("cluster_member_ids") or ()
            if str(value).startswith("bend_hypothesis::")
        ]
        if len(seed_member_ids) != 1:
            continue
        seed = raw_by_id.get(seed_member_ids[0])
        if seed is None or not seed.get("source_bend_ids"):
            continue
        seed_finite = _float(seed, "finite_support_fraction")
        seed_endpoint = _float(seed, "endpoint_support_fraction")
        if (
            seed_finite is None
            or seed_finite < thresholds.minimum_seed_finite_support_fraction
            or seed_endpoint is None
            or seed_endpoint < thresholds.minimum_seed_endpoint_support_fraction
        ):
            continue
        if _endpoints(record) is None:
            continue
        if any(_is_duplicate(record, retained, thresholds) for retained in output):
            continue

        record["selection_reason"] = "scan_supported_short_seed_completion"
        output.append(record)
        additions.append(
            {
                "candidate_id": str(
                    record.get("candidate_id") or record.get("bend_id") or ""
                ),
                "line_object_id": str(record.get("bend_id") or ""),
                "span_mm": span,
                "seed_finite_support_fraction": seed_finite,
                "seed_endpoint_support_fraction": seed_endpoint,
                "seed_has_detector_bend_provenance": True,
            }
        )

    return {
        "enabled": True,
        "selected": output,
        "added": additions,
        "thresholds": {
            "maximum_span_mm": thresholds.maximum_span_mm,
            "minimum_line_finite_support_fraction": (
                thresholds.minimum_line_finite_support_fraction
            ),
            "minimum_line_endpoint_support_fraction": (
                thresholds.minimum_line_endpoint_support_fraction
            ),
            "minimum_seed_finite_support_fraction": (
                thresholds.minimum_seed_finite_support_fraction
            ),
            "minimum_seed_endpoint_support_fraction": (
                thresholds.minimum_seed_endpoint_support_fraction
            ),
        },
    }
