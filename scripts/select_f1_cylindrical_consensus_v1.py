#!/usr/bin/env python3
"""Select bend objects from a frozen scan-only candidate pool.

The selector uses only candidate-pool geometry and within-case consensus.  It
does not read labels, gold geometry, expected counts, part identity, CAD, or
oracle feedback.  Thresholds are frozen from quarantined Family-A development
and are intended for the one-shot Family-B transfer test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


PROFILE = "F1_CYLINDRICAL_CONSENSUS_SELECTOR_V1"
CYLINDER_SOURCE = "cylindrical_standard"


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    seed_ids = row.get("seed_ids")
    if isinstance(seed_ids, list) and seed_ids:
        return tuple(sorted(str(value) for value in seed_ids))
    candidate_id = str(row.get("candidate_id") or "")
    match = re.search(r"::(CB\d+)_", candidate_id)
    if not match:
        raise ValueError(f"cylindrical row has no stable group key: {candidate_id}")
    return (match.group(1),)


def _eligible_finite(row: dict[str, Any]) -> bool:
    return (
        float(row.get("finite_support_fraction") or 0.0) >= 0.80
        and float(row.get("endpoint_support_fraction") or 0.0) >= 0.50
        and float(row.get("p90_line_distance_mm") or 1e9) <= 3.0
    )


def _prediction(row: dict[str, Any], *, route: str, group_key: tuple[str, ...]) -> dict[str, Any]:
    endpoints = row.get("span_endpoints")
    if not isinstance(endpoints, list) or len(endpoints) < 2:
        raise ValueError("selected line object has no span_endpoints")
    source_id = str(row["candidate_id"])
    digest = hashlib.sha256(
        (PROFILE + "\0" + source_id).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "bend_id": f"CCS-{digest}",
        "line_start": endpoints[0],
        "line_end": endpoints[1],
        "source_line_object_id": source_id,
        "source_kind": row.get("representative_source"),
        "selection_route": route,
        "seed_group": list(group_key),
        "span_mm": row.get("span_mm"),
        "radius_mm": row.get("radius_mm"),
        "consensus_count": row.get("consensus_count"),
        "radius_consensus_count": row.get("radius_consensus_count"),
        "finite_support_fraction": row.get("finite_support_fraction"),
        "endpoint_support_fraction": row.get("endpoint_support_fraction"),
        "p90_line_distance_mm": row.get("p90_line_distance_mm"),
        "two_sided_transition_score": row.get("two_sided_transition_score"),
        "two_sided_normal_angle_deg": row.get("two_sided_normal_angle_deg"),
        "two_sided_support_balance": row.get("two_sided_support_balance"),
    }


def select(payload: dict[str, Any]) -> dict[str, Any]:
    raw = [
        dict(row)
        for row in payload.get("candidates") or []
        if row.get("candidate_source") == CYLINDER_SOURCE
    ]
    line_objects = [
        dict(row)
        for row in payload.get("line_objects") or []
        if row.get("representative_source") == CYLINDER_SOURCE
    ]
    raw_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    object_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in raw:
        raw_groups.setdefault(_group_key(row), []).append(row)
    for row in line_objects:
        object_groups.setdefault(_group_key(row), []).append(row)

    selected: list[dict[str, Any]] = []
    accepted_group_audit: list[dict[str, Any]] = []
    for key in sorted(raw_groups):
        transition_values = [
            float(row.get("two_sided_transition_score") or 0.0)
            for row in raw_groups[key]
        ]
        peak = max(transition_values, default=0.0)
        stable = median(transition_values) if transition_values else 0.0
        accepted = peak >= 0.88 and stable >= 0.30
        accepted_group_audit.append(
            {
                "seed_group": list(key),
                "candidate_count": len(transition_values),
                "peak_transition_score": peak,
                "median_transition_score": stable,
                "near_orthogonal_group_accepted": accepted,
            }
        )
        if not accepted:
            continue
        eligible = [
            row
            for row in object_groups.get(key, [])
            if _eligible_finite(row)
            and float(row.get("two_sided_normal_angle_deg") or 0.0) >= 45.0
        ]
        if not eligible:
            continue
        representative = max(
            eligible,
            key=lambda row: (
                int(row.get("consensus_count") or 0),
                float(row.get("score") or 0.0),
                str(row.get("candidate_id") or ""),
            ),
        )
        selected.append(
            _prediction(representative, route="stable_near_orthogonal_group", group_key=key)
        )

    shallow = [
        row
        for row in line_objects
        if _eligible_finite(row)
        and float(row.get("span_mm") or 1e9) <= 60.0
        and 10.0 <= float(row.get("two_sided_normal_angle_deg") or 0.0) <= 35.0
        and float(row.get("two_sided_transition_score") or 0.0) >= 0.15
        and float(row.get("two_sided_support_balance") or 0.0) >= 0.90
    ]
    if shallow:
        representative = max(
            shallow,
            key=lambda row: (
                int(row.get("consensus_count") or 0),
                float(row.get("score") or 0.0),
                str(row.get("candidate_id") or ""),
            ),
        )
        selected.append(
            _prediction(
                representative,
                route="balanced_shallow_partial_cylinder",
                group_key=_group_key(representative),
            )
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        unique[str(row["source_line_object_id"])] = row
    predictions = sorted(unique.values(), key=lambda row: str(row["bend_id"]))
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cylindrical_consensus_selection",
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "selection_audit": accepted_group_audit,
        "inference_contract": {
            "part_identifier_exposed": False,
            "expected_count_exposed": False,
            "manual_truth_exposed": False,
            "cad_exposed": False,
            "oracle_feedback_exposed": False,
            "filename_features_exposed": False,
        },
        "source_point_serialization": False,
        "c006_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = select(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": result["candidate_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
