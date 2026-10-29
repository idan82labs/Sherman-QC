#!/usr/bin/env python3
"""Append-only V17 recall for rejected full-mesh adjacency axes.

This module implements the single preregistered
``AUTHORIZE_MESH_ADJACENCY_AXIAL_HYSTERESIS_RECALL_V17`` intervention.  It
consumes only serialized source features already present in the frozen
candidate pool.  It does not alter candidate generation, incumbent V11
selection, routing, geometry, or cache schemas.
"""

from __future__ import annotations

import ast
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any

from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, _similar


PROFILE = "F1_MESH_ADJACENCY_AXIAL_HYSTERESIS_RECALL_V17"
MIN_FINITE_SUPPORT = 0.90
MIN_LONGEST_RUN = 0.90
MIN_ENDPOINT_SUPPORT = 2.0 / 3.0
MAX_P90_LINE_DISTANCE_MM = 3.0
MIN_AXIAL_LOCALIZATION_RATIO = 10.0
MIN_TRANSITION_SCORE = 0.25
MIN_NORMAL_ANGLE_DEG = 45.0
MIN_SUPPORT_BALANCE = 0.50
MIN_SIDE_SUPPORT = 0.20
MIN_NORMAL_COHERENCE = 0.50

# SHA-256 of the exact ``_similar`` function source in the frozen V4/V11
# selector.  V17 imports that function directly; it does not recreate or
# modify the duplicate relation.
FROZEN_V11_DUPLICATE_PREDICATE_SHA256 = (
    "104e2f4c8bb3b3b545e3699442cb38582b362d18ec9e3ed84b3be61f08dadd58"
)


def _binary64(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def axial_localization_ratio(row: dict[str, Any]) -> float | None:
    """Return the serialized span/p90 certificate, or ``None`` if invalid."""

    p90 = _binary64(row, "p90_line_distance_mm")
    span = _binary64(row, "span_mm")
    if p90 is None or span is None or p90 <= 0.0:
        return None
    return span / p90


def eligible_rejected_adjacency(row: dict[str, Any]) -> bool:
    """Evaluate the authorized binary64 conjunction in preregistered order."""

    if str(row.get("representative_source") or "") != ADJACENCY_SOURCE:
        return False

    finite = _binary64(row, "finite_support_fraction")
    if finite is None or finite < MIN_FINITE_SUPPORT:
        return False

    longest = _binary64(row, "finite_support_longest_run_fraction")
    if longest is None or longest < MIN_LONGEST_RUN:
        return False

    endpoint = _binary64(row, "endpoint_support_fraction")
    if endpoint is None or endpoint < MIN_ENDPOINT_SUPPORT:
        return False

    p90 = _binary64(row, "p90_line_distance_mm")
    if p90 is None or not (0.0 < p90 <= MAX_P90_LINE_DISTANCE_MM):
        return False

    span = _binary64(row, "span_mm")
    if span is None or span < MIN_AXIAL_LOCALIZATION_RATIO * p90:
        return False

    transition = _binary64(row, "two_sided_transition_score")
    if transition is None or transition < MIN_TRANSITION_SCORE:
        return False

    normal_angle = _binary64(row, "two_sided_normal_angle_deg")
    if normal_angle is None or normal_angle < MIN_NORMAL_ANGLE_DEG:
        return False

    balance = _binary64(row, "two_sided_support_balance")
    if balance is None or balance < MIN_SUPPORT_BALANCE:
        return False

    side_support = _binary64(row, "two_sided_min_support_fraction")
    if side_support is None or side_support < MIN_SIDE_SUPPORT:
        return False

    coherence = _binary64(row, "two_sided_normal_coherence")
    if coherence is None or coherence < MIN_NORMAL_COHERENCE:
        return False

    return True


def _rescue_key(row: dict[str, Any]) -> tuple[Any, ...]:
    ratio = axial_localization_ratio(row)
    if ratio is None:
        raise ValueError("ineligible row reached V17 ranking")
    return (
        -ratio,
        -float(row["two_sided_transition_score"]),
        -float(row["finite_support_longest_run_fraction"]),
        -float(row["endpoint_support_fraction"]),
        -float(row["two_sided_normal_coherence"]),
        -float(row["two_sided_min_support_fraction"]),
        -float(row["two_sided_support_balance"]),
        -float(row["span_mm"]),
        str(row.get("candidate_id") or ""),
    )


def rescue_rejected_adjacency(
    rows: list[dict[str, Any]], incumbent_indices: list[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    """Append eligible rejected adjacency rows using incumbent-first V11 NMS."""

    incumbent_set = set(incumbent_indices)
    eligible = [
        index
        for index, row in enumerate(rows)
        if index not in incumbent_set and eligible_rejected_adjacency(row)
    ]
    eligible.sort(key=lambda index: _rescue_key(rows[index]))

    retained = list(incumbent_indices)
    rescues: list[dict[str, Any]] = []
    for index in eligible:
        if any(_similar(rows[index], rows[prior]) for prior in retained):
            continue
        retained.append(index)
        rescues.append(
            {
                "candidate_id": str(rows[index].get("candidate_id") or ""),
                "axial_localization_ratio": round(
                    float(axial_localization_ratio(rows[index])), 12
                ),
            }
        )
    return retained, rescues


def frozen_duplicate_predicate_sha256() -> str:
    """Hash the imported duplicate predicate's exact defining source region."""

    source_path = Path(__file__).with_name("train_f1_topology_recall_v4.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_similar":
            region = ast.get_source_segment(source, node)
            if region is None:
                break
            return hashlib.sha256(region.encode("utf-8")).hexdigest()
    raise RuntimeError("frozen V11 duplicate predicate source region not found")


def assert_frozen_duplicate_predicate() -> None:
    actual = frozen_duplicate_predicate_sha256()
    if actual != FROZEN_V11_DUPLICATE_PREDICATE_SHA256:
        raise RuntimeError(
            "V11 duplicate predicate drifted: "
            f"expected {FROZEN_V11_DUPLICATE_PREDICATE_SHA256}, got {actual}"
        )
