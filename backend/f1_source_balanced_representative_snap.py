"""Source-only representative repair by balanced transverse consensus.

Candidate generators often describe the same physical bend with parallel axes
on different sheet skins.  A dense generator must not win merely because it
emitted more aliases.  This module gives each representative source family one
transverse-location vote, takes their rotation-equivariant geometric median,
and translates the already-selected finite axis toward that consensus.

The primitive cannot add or delete a prediction, change its direction or
span, inspect identity/count/truth, or move an axis by more than the bounded
neighbourhood radius.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.evaluate_f1_against_manual_gold import pair_metrics


MAX_AXIS_ANGLE_DEG = 7.0
MAX_TRANSVERSE_OFFSET_MM = 6.0
MIN_SHORTER_SPAN_OVERLAP = 0.75
MIN_FINITE_SUPPORT_FRACTION = 0.95
MIN_ENDPOINT_SUPPORT_FRACTION = 0.95
MIN_SOURCE_FAMILIES = 3
MIN_TRANSLATION_MM = 0.75
MAX_TRANSLATION_MM = 4.0


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("span_endpoints")
    if value is None and record.get("line_start") is not None:
        value = (record.get("line_start"), record.get("line_end"))
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    if float(np.linalg.norm(result[1] - result[0])) <= 1.0e-9:
        raise ValueError("line span must be nonzero")
    return result


def _source_family(record: Mapping[str, Any]) -> str:
    return str(record.get("representative_source") or record.get("source_kind") or "")


def _quality_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(record.get("p90_line_distance_mm") or float("inf")),
        -float(record.get("finite_support_longest_run_fraction") or 0.0),
        -float(record.get("two_sided_transition_score") or 0.0),
        str(record.get("candidate_id") or record.get("bend_id") or ""),
    )


def _geometric_median(points: np.ndarray) -> np.ndarray:
    """Deterministic Weiszfeld median with a fixed convergence contract."""

    current = np.mean(points, axis=0)
    for _iteration in range(64):
        distances = np.linalg.norm(points - current, axis=1)
        closest = int(np.argmin(distances))
        if float(distances[closest]) <= 1.0e-12:
            return points[closest].copy()
        weights = 1.0 / distances
        updated = np.sum(points * weights[:, None], axis=0) / float(np.sum(weights))
        if float(np.linalg.norm(updated - current)) <= 1.0e-9:
            return updated
        current = updated
    return current


def snap_source_balanced_representatives(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Translate selected axes toward a one-vote-per-source transverse median."""

    output: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for raw_selected in selected:
        source = dict(raw_selected)
        try:
            source_line = _endpoints(source)
        except ValueError:
            output.append(source)
            continue
        source_delta = source_line[1] - source_line[0]
        axis = source_delta / float(np.linalg.norm(source_delta))
        source_midpoint = np.mean(source_line, axis=0)

        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw_candidate in line_objects:
            candidate = dict(raw_candidate)
            family = _source_family(candidate)
            if not family:
                continue
            if float(candidate.get("finite_support_fraction") or 0.0) < MIN_FINITE_SUPPORT_FRACTION:
                continue
            if float(candidate.get("endpoint_support_fraction") or 0.0) < MIN_ENDPOINT_SUPPORT_FRACTION:
                continue
            try:
                candidate_line = _endpoints(candidate)
            except ValueError:
                continue
            metrics = pair_metrics(source_line, candidate_line)
            if (
                float(metrics["axis_angle_deg"]) > MAX_AXIS_ANGLE_DEG
                or float(metrics["transverse_offset_mm"]) > MAX_TRANSVERSE_OFFSET_MM
                or float(metrics["shorter_span_overlap"]) < MIN_SHORTER_SPAN_OVERLAP
            ):
                continue
            by_family[family].append(candidate)

        if len(by_family) < MIN_SOURCE_FAMILIES:
            output.append(source)
            continue

        votes = []
        vote_ids = []
        for family in sorted(by_family):
            representative = min(by_family[family], key=_quality_key)
            candidate_midpoint = np.mean(_endpoints(representative), axis=0)
            delta = candidate_midpoint - source_midpoint
            transverse = delta - float(np.dot(delta, axis)) * axis
            votes.append(transverse)
            vote_ids.append(str(representative.get("candidate_id") or ""))
        translation = _geometric_median(np.asarray(votes, dtype=np.float64))
        magnitude = float(np.linalg.norm(translation))
        if magnitude < MIN_TRANSLATION_MM:
            output.append(source)
            continue
        if magnitude > MAX_TRANSLATION_MM:
            translation *= MAX_TRANSLATION_MM / magnitude
            magnitude = MAX_TRANSLATION_MM

        repaired = dict(source)
        repaired["span_endpoints"] = (source_line + translation).tolist()
        repaired["line_start"] = repaired["span_endpoints"][0]
        repaired["line_end"] = repaired["span_endpoints"][1]
        repaired["candidate_id"] = (
            "source_balanced_snap::" + str(source.get("candidate_id") or source.get("bend_id") or "")
        )
        repaired["representative_source"] = "source_balanced_transverse_median"
        repaired["selection_reason"] = "source_balanced_representative_snap"
        output.append(repaired)
        changes.append(
            {
                "from_candidate_id": str(source.get("candidate_id") or ""),
                "to_candidate_id": repaired["candidate_id"],
                "translation_mm": magnitude,
                "source_family_count": len(by_family),
                "source_family_vote_ids": vote_ids,
            }
        )

    return {"selected": output, "changes": changes}
