#!/usr/bin/env python3
"""Apply a frozen, source-only recovery/reranking layer after retained V51.

The layer consumes only the retained V51 semantic result, the already-built
scan-only candidate pool, and a frozen tree/config artifact.  It has no catalog,
label, expected-count, CAD, drawing, filename, or part-identity input.

This is a development successor.  The default model was fitted on consumed
development labels and therefore is not promotion or generalization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
MODEL_KIND = "sherman_qc_f1_v51_source_only_recovery_tree_v1"
RESULT_KIND = "sherman_qc_f1_v51_source_only_recovery_result_v1"
DEFAULT_PROFILE = "balanced"
REQUIRED_FALSE_FLAGS = (
    "expected_count_used",
    "labels_used",
    "cad_used",
    "part_identity_used",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _finite_number(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("candidate feature is not finite")
    return number


def candidate_endpoints(row: dict[str, Any]) -> np.ndarray:
    """Return finite 2x3 endpoints, reconstructing anchor/axis/span rows."""

    if "span_endpoints" in row:
        endpoints = np.asarray(row["span_endpoints"], dtype=np.float64)
    else:
        anchor = np.asarray(row.get("anchor"), dtype=np.float64)
        axis = np.asarray(row.get("axis_direction"), dtype=np.float64)
        span = _finite_number(row.get("span_mm"))
        if anchor.shape != (3,) or axis.shape != (3,) or span <= 0.0:
            raise ValueError("candidate has no valid endpoint or anchor/axis/span representation")
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("candidate axis is degenerate")
        axis = axis / norm
        endpoints = np.asarray([anchor - axis * span / 2.0, anchor + axis * span / 2.0])
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("candidate endpoints must be finite 2x3 coordinates")
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1e-9:
        raise ValueError("candidate endpoints are degenerate")
    return endpoints


def prediction_endpoints(row: dict[str, Any]) -> np.ndarray:
    endpoints = np.asarray([row.get("line_start"), row.get("line_end")], dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("prediction endpoints must be finite 2x3 coordinates")
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1e-9:
        raise ValueError("prediction endpoints are degenerate")
    return endpoints


def _pair_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_delta = first[1] - first[0]
    second_delta = second[1] - second[0]
    first_length = float(np.linalg.norm(first_delta))
    second_length = float(np.linalg.norm(second_delta))
    first_axis = first_delta / first_length
    second_axis = second_delta / second_length
    dot = float(np.clip(abs(np.dot(first_axis, second_axis)), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    first_midpoint = np.mean(first, axis=0)
    second_midpoint = np.mean(second, axis=0)
    midpoint_delta = second_midpoint - first_midpoint
    first_axis_offset = float(
        np.linalg.norm(midpoint_delta - np.dot(midpoint_delta, first_axis) * first_axis)
    )
    second_axis_offset = float(
        np.linalg.norm(midpoint_delta - np.dot(midpoint_delta, second_axis) * second_axis)
    )
    first_interval = sorted(float(np.dot(point - first_midpoint, first_axis)) for point in first)
    second_interval = sorted(float(np.dot(point - first_midpoint, first_axis)) for point in second)
    overlap = max(
        0.0,
        min(first_interval[1], second_interval[1])
        - max(first_interval[0], second_interval[0]),
    )
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": max(first_axis_offset, second_axis_offset),
        "first_span_overlap": overlap / first_length,
        "second_span_overlap": overlap / second_length,
    }


def duplicate_lines(
    first: np.ndarray,
    second: np.ndarray,
    *,
    angle_deg: float,
    offset_mm: float,
    span_overlap: float,
) -> bool:
    metrics = _pair_metrics(first, second)
    return (
        metrics["axis_angle_deg"] <= angle_deg
        and metrics["transverse_offset_mm"] <= offset_mm
        and metrics["first_span_overlap"] >= span_overlap
        and metrics["second_span_overlap"] >= span_overlap
    )


def validate_model(model: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if model.get("schema_version") != SCHEMA_VERSION or model.get("kind") != MODEL_KIND:
        raise ValueError("recovery model schema/kind mismatch")
    features = model.get("feature_names")
    tree = model.get("tree")
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("model feature_names must be a unique non-empty list")
    if not isinstance(tree, dict):
        raise ValueError("model tree is missing")
    arrays = [
        tree.get("children_left"),
        tree.get("children_right"),
        tree.get("feature_index"),
        tree.get("threshold"),
        tree.get("positive_probability"),
    ]
    if any(not isinstance(values, list) for values in arrays):
        raise ValueError("model tree arrays are missing")
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("model tree arrays have inconsistent lengths")
    profiles = model.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown recovery profile: {profile_name}")
    profile = profiles[profile_name]
    required = {
        "tree_probability_threshold",
        "tree_recovery_cap",
        "duplicate_angle_deg",
        "duplicate_offset_mm",
        "duplicate_span_overlap",
        "adjacency_supplement",
        "narrow_replacement",
    }
    if not isinstance(profile, dict) or not required.issubset(profile):
        raise ValueError("recovery profile is incomplete")
    if int(profile["tree_recovery_cap"]) < 0:
        raise ValueError("tree_recovery_cap must be nonnegative")
    return profile


def tree_probability(row: dict[str, Any], model: dict[str, Any]) -> float:
    feature_names = model["feature_names"]
    tree = model["tree"]
    node = 0
    visited = 0
    node_count = len(tree["children_left"])
    while True:
        if node < 0 or node >= node_count:
            raise ValueError("tree child index is out of bounds")
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        if left == right == -1:
            probability = _finite_number(tree["positive_probability"][node])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("leaf probability is outside [0,1]")
            return probability
        feature_index = int(tree["feature_index"][node])
        if feature_index < 0 or feature_index >= len(feature_names):
            raise ValueError("tree feature index is out of bounds")
        value = _finite_number(row.get(feature_names[feature_index]), default=0.0)
        threshold = _finite_number(tree["threshold"][node])
        node = left if value <= threshold else right
        visited += 1
        if visited > node_count:
            raise ValueError("tree contains a cycle")


def _line_length(row: dict[str, Any]) -> float:
    return float(np.linalg.norm(prediction_endpoints(row)[1] - prediction_endpoints(row)[0]))


def _recovery_prediction(
    candidate: dict[str, Any],
    probability: float,
    *,
    route: str,
) -> dict[str, Any]:
    endpoints = candidate_endpoints(candidate)
    candidate_id = str(candidate.get("candidate_id") or candidate.get("bend_id") or "")
    if not candidate_id:
        raise ValueError("candidate has no stable identifier")
    identity = _sha_bytes(
        _canonical_bytes(
            {
                "candidate_id": candidate_id,
                "line_start": endpoints[0].tolist(),
                "line_end": endpoints[1].tolist(),
                "route": route,
            }
        )
    )[:20]
    representative_source = str(
        candidate.get("representative_source")
        or candidate.get("candidate_source")
        or "unknown"
    )
    return {
        "bend_id": f"V51R1-{identity}",
        "line_start": endpoints[0].tolist(),
        "line_end": endpoints[1].tolist(),
        "selection_route": route,
        "selector_score": round(float(probability), 12),
        "source_hypothesis_id": candidate_id,
        "source_kind": representative_source,
    }


def _is_duplicate(
    first: dict[str, Any],
    second: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    return duplicate_lines(
        prediction_endpoints(first),
        prediction_endpoints(second),
        angle_deg=_finite_number(profile["duplicate_angle_deg"]),
        offset_mm=_finite_number(profile["duplicate_offset_mm"]),
        span_overlap=_finite_number(profile["duplicate_span_overlap"]),
    )


def _supplement_eligible(candidate: dict[str, Any], rule: dict[str, Any]) -> bool:
    transition = _finite_number(candidate.get("two_sided_transition_score"))
    return (
        str(candidate.get("candidate_source")) == str(rule["candidate_source"])
        and _finite_number(candidate.get("confidence")) >= _finite_number(rule["min_confidence"])
        and _finite_number(rule["min_transition_score"])
        <= transition
        <= _finite_number(rule["max_transition_score"])
        and _finite_number(candidate.get("normal_second_eigenvalue"))
        >= _finite_number(rule["min_normal_second_eigenvalue"])
        and _finite_number(candidate.get("corridor_points_per_mm"))
        >= _finite_number(rule["min_corridor_points_per_mm"])
    )


def _narrow_replacement_allowed(
    recovery_candidate: dict[str, Any],
    recovery_probability: float,
    duplicate_incumbents: Iterable[dict[str, Any]],
    rule: dict[str, Any],
) -> bool:
    source = str(
        recovery_candidate.get("representative_source")
        or recovery_candidate.get("candidate_source")
        or ""
    )
    allowed_prefixes = tuple(str(value) for value in rule["recovery_source_prefixes"])
    incumbents = list(duplicate_incumbents)
    allowed_incumbent_sources = {
        str(value)
        for value in rule.get(
            "incumbent_source_kinds",
            [rule.get("incumbent_source_kind")],
        )
        if value is not None
    }
    return (
        bool(incumbents)
        and recovery_probability >= _finite_number(rule["min_tree_probability"])
        and source.startswith(allowed_prefixes)
        and all(
            str(row.get("source_kind")) in allowed_incumbent_sources
            and _finite_number(row.get("selector_score"), default=math.inf)
            < _finite_number(rule["max_incumbent_selector_score"])
            for row in incumbents
        )
    )


def apply_recovery(
    incumbent: dict[str, Any],
    pool: dict[str, Any],
    model: dict[str, Any],
    *,
    profile_name: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Return a deterministic successor result without consulting truth."""

    profile = validate_model(model, profile_name)
    if incumbent.get("source_only") is not True or incumbent.get("research_only") is not True:
        raise ValueError("incumbent is not source-only/research-only")
    if any(incumbent.get(flag) is not False for flag in REQUIRED_FALSE_FLAGS):
        raise ValueError("incumbent reports a forbidden inference input")
    if not isinstance(incumbent.get("predictions"), list):
        raise ValueError("incumbent predictions are missing")
    if int(pool.get("working_point_budget") or 0) != 200000:
        raise ValueError("candidate pool point-budget contract mismatch")
    line_objects = pool.get("line_objects")
    raw_candidates = pool.get("candidates")
    if not isinstance(line_objects, list) or not isinstance(raw_candidates, list):
        raise ValueError("candidate pool arrays are missing")

    incumbent_rows = [dict(row) for row in incumbent["predictions"]]
    for row in incumbent_rows:
        prediction_endpoints(row)

    threshold = _finite_number(profile["tree_probability_threshold"])
    scored: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    for candidate in line_objects:
        probability = tree_probability(candidate, model)
        # Tree leaves can differ by a few binary64 ULPs even when the fitted
        # evidence class and the serialized score are identical.  Ownership is
        # therefore ranked at the emitted 12-decimal precision.
        rank_probability = round(probability, 12)
        if rank_probability < threshold:
            continue
        prediction = _recovery_prediction(
            candidate,
            rank_probability,
            route="v51_source_only_recovery_tree_v1",
        )
        scored.append(
            (rank_probability, str(candidate.get("candidate_id")), candidate, prediction)
        )
    scored.sort(key=lambda row: (-row[0], -_line_length(row[3]), row[1]))

    emitted_recoveries: list[dict[str, Any]] = []
    accepted_candidates: list[dict[str, Any]] = []
    removed_incumbent_ids: list[str] = []
    rejection_audit: list[dict[str, Any]] = []
    replacement_audit: list[dict[str, Any]] = []

    def consider(
        probability: float,
        candidate: dict[str, Any],
        prediction: dict[str, Any],
        *,
        allow_replacement: bool,
    ) -> bool:
        duplicate_recovery = next(
            (row for row in emitted_recoveries if _is_duplicate(prediction, row, profile)),
            None,
        )
        if duplicate_recovery is not None:
            rejection_audit.append(
                {
                    "candidate_id": str(candidate.get("candidate_id")),
                    "reason": "LOWER_RANKED_RECOVERY_DUPLICATE",
                    "owner_bend_id": duplicate_recovery["bend_id"],
                }
            )
            return False
        duplicate_indices = [
            index
            for index, row in enumerate(incumbent_rows)
            if _is_duplicate(prediction, row, profile)
        ]
        if duplicate_indices:
            duplicate_incumbents = [incumbent_rows[index] for index in duplicate_indices]
            replacement_rule = profile["narrow_replacement"]
            if not allow_replacement or not _narrow_replacement_allowed(
                candidate,
                probability,
                duplicate_incumbents,
                replacement_rule,
            ):
                rejection_audit.append(
                    {
                        "candidate_id": str(candidate.get("candidate_id")),
                        "reason": "INCUMBENT_OWNS_DUPLICATE",
                        "owner_bend_ids": [str(row["bend_id"]) for row in duplicate_incumbents],
                    }
                )
                return False
            removed = []
            for index in reversed(duplicate_indices):
                removed.append(str(incumbent_rows[index]["bend_id"]))
                incumbent_rows.pop(index)
            removed.reverse()
            removed_incumbent_ids.extend(removed)
            replacement_audit.append(
                {
                    "candidate_id": str(candidate.get("candidate_id")),
                    "replacement_bend_id": prediction["bend_id"],
                    "replaced_incumbent_bend_ids": removed,
                    "rule": "HIGH_TREE_CYLINDRICAL_OVER_LOW_SCORE_ADJACENCY",
                }
            )
        emitted_recoveries.append(prediction)
        accepted_candidates.append(candidate)
        return True

    tree_cap = int(profile["tree_recovery_cap"])
    tree_accept_count = 0
    for probability, _candidate_id, candidate, prediction in scored:
        if consider(probability, candidate, prediction, allow_replacement=True):
            tree_accept_count += 1
            if tree_accept_count >= tree_cap:
                break

    supplement_rule = profile["adjacency_supplement"]
    supplement_score = _finite_number(supplement_rule["emission_score"])
    supplement_candidates = sorted(
        (row for row in raw_candidates if _supplement_eligible(row, supplement_rule)),
        key=lambda row: (-_finite_number(row.get("confidence")), str(row.get("candidate_id"))),
    )
    supplement_accept_count = 0
    for candidate in supplement_candidates:
        prediction = _recovery_prediction(
            candidate,
            supplement_score,
            route="v51_source_only_high_curvature_adjacency_supplement_v1",
        )
        if consider(supplement_score, candidate, prediction, allow_replacement=False):
            supplement_accept_count += 1

    predictions = incumbent_rows + emitted_recoveries
    if len({str(row["bend_id"]) for row in predictions}) != len(predictions):
        raise ValueError("successor emitted duplicate bend IDs")
    # Final defensive ownership check.  The original incumbent is trusted as a
    # frozen set; all newly emitted rows must be mutually unique and unique from
    # every retained incumbent.
    for index, row in enumerate(emitted_recoveries):
        if any(_is_duplicate(row, other, profile) for other in incumbent_rows):
            raise RuntimeError("recovery duplicate survived incumbent ownership")
        if any(_is_duplicate(row, other, profile) for other in emitted_recoveries[:index]):
            raise RuntimeError("recovery duplicate survived recovery ownership")

    model_identity = _sha_bytes(_canonical_bytes(model))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "profile": profile_name,
        "v51_retained": True,
        "candidate_count": len(predictions),
        "predictions": predictions,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "development_tuned_model": bool(model.get("development_tuned_model")),
        "recovery_model_sha256": model_identity,
        "recovery_audit": {
            "line_object_count": len(line_objects),
            "raw_candidate_count": len(raw_candidates),
            "tree_probability_threshold": threshold,
            "tree_eligible_count": len(scored),
            "tree_recovery_cap": tree_cap,
            "tree_emitted_count": tree_accept_count,
            "adjacency_supplement_eligible_count": len(supplement_candidates),
            "adjacency_supplement_emitted_count": supplement_accept_count,
            "input_incumbent_count": len(incumbent["predictions"]),
            "retained_incumbent_count": len(incumbent_rows),
            "replaced_incumbent_count": len(removed_incumbent_ids),
            "recovery_emitted_count": len(emitted_recoveries),
            "output_count": len(predictions),
            "strict_duplicate_ownership": True,
            "incumbent_owns_by_default": True,
            "replacement_audit": replacement_audit,
            "rejection_audit": rejection_audit,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply_recovery(
        json.loads(args.incumbent.read_text(encoding="utf-8")),
        json.loads(args.pool.read_text(encoding="utf-8")),
        json.loads(args.model.read_text(encoding="utf-8")),
        profile_name=args.profile,
    )
    _write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "profile": result["profile"],
                "recovery_count": result["recovery_audit"]["recovery_emitted_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
