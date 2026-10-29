#!/usr/bin/env python3
"""Post-lock gold attribution for LOCAL_PLANE_FRAGMENT_PROBE_V2.

The inference-safe bundle must be locked and hash-verified before any gold,
grouping, prior attribution, or frozen A0 prediction geometry is opened.  This
script never regenerates the probe and never changes predictions.  Only rows
frozen at transition-probe stage 1 are part of the primary recovery population.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND = "postlock_gold_local_plane_fragment_attribution_v2"
BUNDLE_KIND = "f1_local_plane_fragment_probe_bundle_v2"
BUNDLE_PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
PROBE_KIND = "f1_local_plane_fragment_probe_v2"
LOCK_ATTESTATION_KIND = "f1_local_plane_fragment_probe_lock_attestation_v2"
MATCHING_PROFILE = "LPF_POSTLOCK_EXACT_MATCH_V2"
PRIOR_KIND = "postlock_gold_transition_failure_attribution"

FROZEN_GROUPED_SPLIT_SHA256 = (
    "dbf1cb1f115780a546a72c296a962c37fdcd49602f8489eb4c5b0c9721c9cbfc"
)
FROZEN_PRIOR_ATTRIBUTION_SHA256 = (
    "dda757ce04962c8159ff218924b358e9526d7e56bd13e1845a47c1e61b738438"
)
FROZEN_GOLD_SHA256 = (
    "276fffd42db78629fe3930545f5a1e046fa45e64ea1eb9dc83bcc5916cca6f8b"
)
FROZEN_A0_PREDICTION_SHA256 = (
    "d543b970f3e2830f2bc2e976f490ffb7169eba9f3cef59c32323e1dfd7592fd8"
)
FROZEN_TRANSITION_CENSUS_SHA256 = (
    "dd0007286a2182cdfe1bf0c18219b54fb75226db5543b67caf20a3747ce52ac6"
)

FROZEN_A0_FALSE_PREDICTION_COUNT = 33
PRIMARY_STAGE = 1
PRIMARY_STAGE1_COUNT = 67
MIN_RECOVERED_STAGE1_OBJECTS = 14
MIN_INDEPENDENT_GROUPS = 8
MAX_A0_FALSE_PREDICTIONS_CERTIFIED = 8
EXPECTED_CASE_COUNT = 37

MAX_MATCH_LEFT_VERTICES = 256
MAX_MATCH_RIGHT_VERTICES = 65_536
MAX_MATCH_EDGES = 32_768
MAX_MATCH_EDGE_COST_INT = 10_000_000

LOCK_CRITERION_KEYS = (
    "profile_identity_valid",
    "engineering_profile_hashes_recomputed",
    "exact_nested_schemas_valid",
    "case_count_is_37",
    "successful_case_count_is_37",
    "canonical_rebuilds_identical",
    "scan_hashes_unchanged",
    "candidate_pool_file_hashes_unchanged",
    "a0_prediction_hashes_unchanged",
    "input_table_hashes_unchanged",
    "input_projection_attestations_valid",
    "inherited_v2_caps_valid",
    "no_structural_cap_breach",
    "forbidden_data_absent",
    "source_points_not_consumed",
    "source_points_not_serialized",
    "research_only",
    "inference_safe",
    "prediction_mutated_false",
    "birth_columns_disabled",
    "ownership_not_consumed",
    "solver_not_consumed",
    "truth_not_consumed",
    "designated_replay_attestation_valid",
    "all_measured_peak_rss_below_16_gib",
)

EMPIRICAL_CRITERION_KEYS = (
    "recovered_stage1_objects_at_least_14",
    "independent_family_groups_at_least_8",
    "a0_false_predictions_certified_at_most_8",
)

CAUSE_CATEGORIES = (
    "LOCAL_FRAGMENT_NO_GLOBAL_PLANE",
    "LOCAL_FRAGMENT_PARTIAL_GLOBAL_COVERAGE",
    "FULL_GLOBAL_PAIR_LINE_NON_EQUIVALENT",
    "FULL_GLOBAL_PAIR_LINE_EQUIVALENT_RUN_ABSENT",
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Thresholds:
    angle_deg: float = 5.0
    transverse_offset_mm: float = 4.0
    gold_span_overlap: float = 0.60
    prediction_span_overlap: float = 0.60


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    endpoints = record.get("span_endpoints")
    if endpoints is not None:
        return np.asarray(endpoints, dtype=np.float64)
    if record.get("line_start") is not None and record.get("line_end") is not None:
        return np.asarray([record["line_start"], record["line_end"]], dtype=np.float64)
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    direction = np.asarray(record["axis_direction"], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("line axis must be nonzero")
    direction /= norm
    half_span = direction * float(record["span_mm"]) / 2.0
    return np.asarray([anchor - half_span, anchor + half_span])


def pair_metrics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gold_midpoint = np.mean(gold, axis=0)
    prediction_midpoint = np.mean(prediction, axis=0)
    gold_delta = gold[1] - gold[0]
    prediction_delta = prediction[1] - prediction[0]
    gold_length = float(np.linalg.norm(gold_delta))
    prediction_length = float(np.linalg.norm(prediction_delta))
    if gold_length <= 1e-9 or prediction_length <= 1e-9:
        raise ValueError("bend lines must have nonzero length")
    gold_axis = gold_delta / gold_length
    prediction_axis = prediction_delta / prediction_length
    dot = float(np.clip(np.dot(gold_axis, prediction_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0.0:
        prediction_axis *= -1.0
    midpoint_delta = prediction_midpoint - gold_midpoint
    gold_offset = float(
        np.linalg.norm(
            midpoint_delta - gold_axis * np.dot(midpoint_delta, gold_axis)
        )
    )
    prediction_offset = float(
        np.linalg.norm(
            midpoint_delta
            - prediction_axis * np.dot(midpoint_delta, prediction_axis)
        )
    )
    transverse = max(gold_offset, prediction_offset)
    gold_interval = sorted(
        float(np.dot(point - gold_midpoint, gold_axis)) for point in gold
    )
    prediction_interval = sorted(
        float(np.dot(point - gold_midpoint, gold_axis)) for point in prediction
    )
    overlap_length = max(
        0.0,
        min(gold_interval[1], prediction_interval[1])
        - max(gold_interval[0], prediction_interval[0]),
    )
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": transverse,
        "gold_span_overlap": overlap_length / gold_length,
        "prediction_span_overlap": overlap_length / prediction_length,
        "shorter_span_overlap": overlap_length / min(gold_length, prediction_length),
        "gold_length_mm": gold_length,
        "prediction_length_mm": prediction_length,
    }


def is_match(metrics: Mapping[str, float], thresholds: Thresholds) -> bool:
    return bool(
        float(metrics["axis_angle_deg"]) <= thresholds.angle_deg
        and float(metrics["transverse_offset_mm"])
        <= thresholds.transverse_offset_mm
        and float(metrics["gold_span_overlap"]) >= thresholds.gold_span_overlap
        and float(metrics["prediction_span_overlap"])
        >= thresholds.prediction_span_overlap
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, *, field: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _index_cases(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    raw_rows = payload.get("cases") or ()
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(raw)
    return result


def _line(
    record: Mapping[str, Any],
    *,
    identifier: str,
) -> dict[str, Any]:
    source = dict(record.get("geometry")) if isinstance(record.get("geometry"), Mapping) else dict(record)
    if source.get("span_endpoints") is None and record.get("intersection_span_endpoints") is not None:
        source["span_endpoints"] = deepcopy(record["intersection_span_endpoints"])
    source["bend_id"] = identifier
    try:
        endpoints = _endpoints(source)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{identifier}: invalid line geometry") from exc
    if endpoints.shape != (2, 3) or not all(
        math.isfinite(float(value)) for value in endpoints.flat
    ):
        raise ValueError(f"{identifier}: invalid line endpoints")
    if float(((endpoints[1] - endpoints[0]) ** 2).sum()) <= 1e-18:
        raise ValueError(f"{identifier}: line span must be nonzero")
    return source


def _integer_geometric_cost(metrics: Mapping[str, float]) -> int:
    raw = 1_000_000.0 * (
        float(metrics["axis_angle_deg"]) / 5.0
        + float(metrics["transverse_offset_mm"]) / 4.0
        + 1.0
        - float(metrics["gold_span_overlap"])
        + 1.0
        - float(metrics["prediction_span_overlap"])
    )
    if not math.isfinite(raw) or raw < -1e-6:
        raise ValueError("geometric match cost must be finite and nonnegative")
    return int(math.floor(max(0.0, raw) + 0.5))


@dataclass(frozen=True)
class _CandidateEdge:
    left_id: str
    right_id: str
    cost: int
    metrics: Mapping[str, float]


def _cp_sat_api() -> Any:
    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "LPF_POSTLOCK_EXACT_MATCH_V2 requires the f1-structured OR-Tools runtime"
        ) from exc
    return cp_model


def _solve_exact_matching(
    left_ids: Iterable[str],
    right_ids: Iterable[str],
    candidate_edges: Sequence[_CandidateEdge],
) -> list[_CandidateEdge]:
    """Bounded exact CP-SAT: cardinality, cost, then lexical feasibility."""

    raw_left = [str(value) for value in left_ids]
    raw_right = [str(value) for value in right_ids]
    left = sorted(set(raw_left))
    right = sorted(set(raw_right))
    if len(left) != len(raw_left):
        raise ValueError("left matching IDs must be unique")
    if len(right) != len(raw_right):
        raise ValueError("right matching IDs must be unique")
    if len(left) > MAX_MATCH_LEFT_VERTICES:
        raise ValueError("matching left-vertex cap breached")
    if len(right) > MAX_MATCH_RIGHT_VERTICES:
        raise ValueError("matching right-vertex cap breached")
    if len(candidate_edges) > MAX_MATCH_EDGES:
        raise ValueError("matching edge cap breached")

    edge_by_pair: dict[tuple[str, str], _CandidateEdge] = {}
    for edge in candidate_edges:
        pair = (str(edge.left_id), str(edge.right_id))
        if pair[0] not in left or pair[1] not in right:
            raise ValueError("candidate edge references an absent matching vertex")
        if pair in edge_by_pair:
            raise ValueError("candidate matching edges must be unique")
        if isinstance(edge.cost, bool) or not isinstance(edge.cost, int):
            raise ValueError("candidate matching cost must be an integer")
        if edge.cost < 0 or edge.cost > MAX_MATCH_EDGE_COST_INT:
            raise ValueError("candidate matching cost exceeds its hard bounds")
        edge_by_pair[pair] = edge
    edges = [edge_by_pair[pair] for pair in sorted(edge_by_pair)]
    if not edges:
        return []

    cp_model = _cp_sat_api()
    edges_by_left: dict[str, list[_CandidateEdge]] = {value: [] for value in left}
    edges_by_right: dict[str, list[_CandidateEdge]] = {}
    for edge in edges:
        edges_by_left[edge.left_id].append(edge)
        edges_by_right.setdefault(edge.right_id, []).append(edge)
    for values in edges_by_left.values():
        values.sort(key=lambda edge: edge.right_id)

    def solve(
        *,
        fixed_by_left: Mapping[str, str | None] | None = None,
        target_cardinality: int | None = None,
        target_cost: int | None = None,
        force_left_match: str | None = None,
        force_prefix: Sequence[tuple[str, str]] | None = None,
        objective: str | None = None,
    ) -> tuple[Any, dict[tuple[str, str], Any], Any, Any, Any]:
        model = cp_model.CpModel()
        variables = {
            (edge.left_id, edge.right_id): model.NewBoolVar(
                f"x_{index:05d}"
            )
            for index, edge in enumerate(edges)
        }
        for left_id, row in edges_by_left.items():
            if not row:
                continue
            model.Add(
                sum(variables[(edge.left_id, edge.right_id)] for edge in row) <= 1
            )
        for right_id, column in edges_by_right.items():
            model.Add(
                sum(
                    variables[(edge.left_id, edge.right_id)] for edge in column
                )
                <= 1
            )
        cardinality = sum(variables.values())
        total_cost = sum(
            edge.cost * variables[(edge.left_id, edge.right_id)] for edge in edges
        )
        if target_cardinality is not None:
            model.Add(cardinality == target_cardinality)
        if target_cost is not None:
            model.Add(total_cost == target_cost)
        for left_id, right_id in sorted((fixed_by_left or {}).items()):
            row = edges_by_left[left_id]
            if right_id is None:
                model.Add(
                    sum(
                        variables[(edge.left_id, edge.right_id)] for edge in row
                    )
                    == 0
                )
            else:
                model.Add(variables[(left_id, right_id)] == 1)
        if force_left_match is not None:
            row = edges_by_left[force_left_match]
            model.Add(
                sum(variables[(edge.left_id, edge.right_id)] for edge in row) == 1
            )
        if force_prefix is not None:
            model.Add(sum(variables[pair] for pair in force_prefix) == 1)
        if objective == "maximum_cardinality":
            model.Maximize(cardinality)
        elif objective == "minimum_cost":
            model.Minimize(total_cost)
        elif objective is not None:
            raise ValueError(f"unknown matching objective: {objective}")
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        status = solver.Solve(model)
        return status, variables, cardinality, total_cost, solver

    status, _variables, cardinality, _cost, solver = solve(
        objective="maximum_cardinality"
    )
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT cardinality stage did not prove OPTIMAL")
    target_cardinality = int(solver.Value(cardinality))
    if target_cardinality == 0:
        return []

    status, _variables, _cardinality, total_cost, solver = solve(
        target_cardinality=target_cardinality,
        objective="minimum_cost",
    )
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT geometric-cost stage did not prove OPTIMAL")
    target_cost = int(solver.Value(total_cost))

    fixed_by_left: dict[str, str | None] = {}
    for left_id in left:
        row = edges_by_left[left_id]
        if not row:
            continue
        status, _variables, _cardinality, _cost, _solver = solve(
            fixed_by_left=fixed_by_left,
            target_cardinality=target_cardinality,
            target_cost=target_cost,
            force_left_match=left_id,
        )
        if status == cp_model.INFEASIBLE:
            fixed_by_left[left_id] = None
            continue
        if status != cp_model.OPTIMAL:
            raise RuntimeError("CP-SAT lexical left feasibility was not proven")

        lo = 0
        hi = len(row) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            prefix = [
                (edge.left_id, edge.right_id) for edge in row[: mid + 1]
            ]
            status, _variables, _cardinality, _cost, _solver = solve(
                fixed_by_left=fixed_by_left,
                target_cardinality=target_cardinality,
                target_cost=target_cost,
                force_prefix=prefix,
            )
            if status == cp_model.OPTIMAL:
                hi = mid
            elif status == cp_model.INFEASIBLE:
                lo = mid + 1
            else:
                raise RuntimeError("CP-SAT lexical prefix feasibility was not proven")
        fixed_by_left[left_id] = row[lo].right_id

    status, variables, _cardinality, _cost, solver = solve(
        fixed_by_left=fixed_by_left,
        target_cardinality=target_cardinality,
        target_cost=target_cost,
    )
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT final exact assignment was not proven OPTIMAL")
    selected = [
        edge
        for edge in edges
        if solver.Value(variables[(edge.left_id, edge.right_id)]) == 1
    ]
    if len(selected) != target_cardinality:
        raise AssertionError("CP-SAT selected cardinality differs from locked target")
    if sum(edge.cost for edge in selected) != target_cost:
        raise AssertionError("CP-SAT selected cost differs from locked target")
    return sorted(selected, key=lambda edge: (edge.left_id, edge.right_id))


def _exact_line_matching(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    left_id_field: str,
    right_id_field: str,
    thresholds: Thresholds,
) -> dict[str, Any]:
    left_lines: dict[str, dict[str, Any]] = {}
    right_lines: dict[str, dict[str, Any]] = {}
    for record in left_records:
        identifier = str(record.get(left_id_field) or "")
        if not identifier or identifier in left_lines:
            raise ValueError("left line IDs must be nonempty and unique")
        left_lines[identifier] = _line(record, identifier=identifier)
    for record in right_records:
        identifier = str(record.get(right_id_field) or "")
        if not identifier or identifier in right_lines:
            raise ValueError("right line IDs must be nonempty and unique")
        right_lines[identifier] = _line(record, identifier=identifier)

    candidates: list[_CandidateEdge] = []
    for left_id in sorted(left_lines):
        left_endpoints = _endpoints(left_lines[left_id])
        for right_id in sorted(right_lines):
            metrics = pair_metrics(left_endpoints, _endpoints(right_lines[right_id]))
            if is_match(metrics, thresholds):
                candidates.append(
                    _CandidateEdge(
                        left_id=left_id,
                        right_id=right_id,
                        cost=_integer_geometric_cost(metrics),
                        metrics=dict(metrics),
                    )
                )
    selected = _solve_exact_matching(
        list(left_lines), list(right_lines), candidates
    )
    matched_left = {edge.left_id for edge in selected}
    matched_right = {edge.right_id for edge in selected}
    return {
        "matches": [
            {
                "left_id": edge.left_id,
                "right_id": edge.right_id,
                "integer_geometric_cost": edge.cost,
                **{key: round(float(value), 9) for key, value in edge.metrics.items()},
            }
            for edge in selected
        ],
        "unmatched_left_ids": sorted(set(left_lines) - matched_left),
        "unmatched_right_ids": sorted(set(right_lines) - matched_right),
    }


def _validate_bundle_lock(
    bundle: Mapping[str, Any],
    *,
    actual_bundle_sha256: str,
    expected_bundle_sha256: str,
    expected_lock_attestation_sha256: str,
    expected_case_count: int,
) -> dict[str, bool]:
    actual = _require_hash(actual_bundle_sha256, field="local_plane_fragment_bundle_sha256")
    expected = _require_hash(expected_bundle_sha256, field="expected_bundle_sha256")
    if actual != expected:
        raise ValueError("local-plane-fragment bundle SHA-256 mismatch")
    expected_lock = _require_hash(
        expected_lock_attestation_sha256,
        field="expected_lock_attestation_sha256",
    )
    if (
        int(bundle.get("schema_version") or 0) != 2
        or bundle.get("kind") != BUNDLE_KIND
        or bundle.get("profile") != BUNDLE_PROFILE
    ):
        raise ValueError("local-plane-fragment bundle must have exact V2 identity")
    if bundle.get("locked") is not True or bundle.get("ready_for_gold") is not True:
        raise ValueError("local-plane-fragment V2 bundle is not ready for gold")
    required_flags = {
        "inference_safe": True,
        "research_only": True,
        "prediction_changed": False,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
    }
    for field, expected_value in required_flags.items():
        if bundle.get(field) is not expected_value:
            raise ValueError(f"locked bundle requires {field}={expected_value!r}")
    cases = _index_cases(bundle, label="locked bundle")
    if (
        int(bundle.get("case_count") or -1) != expected_case_count
        or int(bundle.get("successful_case_count") or -1) != expected_case_count
        or len(cases) != expected_case_count
    ):
        raise ValueError(f"locked bundle must contain exactly {expected_case_count} cases")
    replay_hash = _require_hash(
        bundle.get("replay_attestation_sha256"),
        field="bundle.replay_attestation_sha256",
    )
    if not replay_hash:
        raise AssertionError("unreachable empty replay hash")
    raw_lock = bundle.get("lock_attestation")
    if not isinstance(raw_lock, Mapping):
        raise ValueError("V2 bundle lacks its lock attestation")
    lock = dict(raw_lock)
    if set(lock) != {
        "kind",
        "criteria",
        "ready_for_gold",
        "lock_attestation_sha256",
    }:
        raise ValueError("V2 lock-attestation schema is not exact")
    if lock.get("kind") != LOCK_ATTESTATION_KIND:
        raise ValueError("V2 lock-attestation kind is invalid")
    criteria = lock.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(LOCK_CRITERION_KEYS):
        raise ValueError("V2 lock-attestation criteria schema is not exact")
    normalized_criteria = {
        key: criteria.get(key) is True for key in LOCK_CRITERION_KEYS
    }
    if not all(normalized_criteria.values()):
        failed = [key for key, value in normalized_criteria.items() if not value]
        raise ValueError(f"V2 bundle is not ready for gold; failed criteria={failed}")
    if lock.get("ready_for_gold") is not True:
        raise ValueError("V2 lock attestation does not authorize opening gold")
    lock_hash = _require_hash(
        lock.get("lock_attestation_sha256"),
        field="lock_attestation.lock_attestation_sha256",
    )
    recomputed = _sha256_json(
        {key: value for key, value in lock.items() if key != "lock_attestation_sha256"}
    )
    if lock_hash != recomputed:
        raise ValueError("V2 lock-attestation SHA-256 does not recompute")
    if lock_hash != expected_lock:
        raise ValueError("V2 lock-attestation SHA-256 differs from expected lock")
    if _require_hash(
        bundle.get("lock_attestation_sha256"),
        field="bundle.lock_attestation_sha256",
    ) != lock_hash:
        raise ValueError("V2 bundle lock-attestation hash mirror mismatch")
    return normalized_criteria


def _validate_frozen_sources(
    *,
    prior_attribution: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, str]:
    expected = {
        "prior_attribution_sha256": FROZEN_PRIOR_ATTRIBUTION_SHA256,
        "a0_prediction_sha256": FROZEN_A0_PREDICTION_SHA256,
        "gold_sha256": FROZEN_GOLD_SHA256,
        "grouped_split_sha256": FROZEN_GROUPED_SPLIT_SHA256,
    }
    verified: dict[str, str] = {}
    for field, frozen in expected.items():
        actual = _require_hash(source_hashes.get(field), field=field)
        if actual != frozen:
            raise ValueError(f"{field} differs from the preregistered frozen value")
        verified[field] = actual
    if prior_attribution.get("kind") != PRIOR_KIND or prior_attribution.get("postlock") is not True:
        raise ValueError("prior transition attribution is not the frozen post-lock artifact")
    prior_fields = {
        "prediction_sha256": FROZEN_A0_PREDICTION_SHA256,
        "gold_sha256": FROZEN_GOLD_SHA256,
        "grouped_split_sha256": FROZEN_GROUPED_SPLIT_SHA256,
        "diagnostic_census_sha256": FROZEN_TRANSITION_CENSUS_SHA256,
    }
    for field, frozen in prior_fields.items():
        if _require_hash(prior_attribution.get(field), field=f"prior.{field}") != frozen:
            raise ValueError(f"prior attribution {field} changed")
    return verified


def _probe_case(row: Mapping[str, Any], *, token: str) -> dict[str, Any]:
    raw = row.get("probe")
    probe = dict(raw) if isinstance(raw, Mapping) else dict(row)
    if (
        int(probe.get("schema_version") or 0) != 2
        or probe.get("kind") != PROBE_KIND
        or probe.get("profile") != BUNDLE_PROFILE
    ):
        raise ValueError(f"{token}: probe must have exact V2 identity")
    for field, expected in (
        ("inference_safe", True),
        ("research_only", True),
        ("prediction_mutated", False),
        ("birth_columns_enabled", False),
        ("ownership_consumed", False),
        ("solver_consumed", False),
        ("truth_inputs_consumed", False),
        ("source_points_consumed", False),
        ("source_points_serialized", False),
    ):
        if probe.get(field) is not expected:
            raise ValueError(f"{token}: probe requires {field}={expected!r}")
    wrapper_hash = _require_hash(row.get("a0_prediction_sha256"), field=f"{token}.bundle.a0_prediction_sha256")
    probe_hash = _require_hash(probe.get("a0_prediction_sha256"), field=f"{token}.probe.a0_prediction_sha256")
    if wrapper_hash != probe_hash:
        raise ValueError(f"{token}: wrapper/probe A0 prediction hash mismatch")
    return probe


def _component_records(probe: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fragments = {
        str(row.get("local_plane_fragment_id") or ""): dict(row)
        for row in probe.get("local_plane_fragment_table") or ()
        if isinstance(row, Mapping)
    }
    if "" in fragments or len(fragments) != len(probe.get("local_plane_fragment_table") or ()):
        raise ValueError("local fragment IDs must be nonempty and unique")
    pairs: dict[str, dict[str, Any]] = {}
    components: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    for raw_pair in probe.get("local_plane_pair_records") or ():
        if not isinstance(raw_pair, Mapping):
            raise ValueError("local plane-pair records must be objects")
        pair = dict(raw_pair)
        pair_id = str(pair.get("local_plane_pair_id") or "")
        if not pair_id or pair_id in pairs:
            raise ValueError("local plane-pair IDs must be nonempty and unique")
        pairs[pair_id] = pair
        for raw_component in pair.get("components") or ():
            if not isinstance(raw_component, Mapping):
                raise ValueError("local transition components must be objects")
            component = deepcopy(dict(raw_component))
            component_id = str(component.get("local_transition_component_id") or "")
            if not component_id or component_id in seen_components:
                raise ValueError("local transition component IDs must be nonempty and unique")
            seen_components.add(component_id)
            component["_pair_id"] = pair_id
            try:
                component["_line"] = _line(component, identifier=component_id)
                component["_line_matchability"] = "MATCHABLE"
            except ValueError as exc:
                if "line span must be nonzero" not in str(exc):
                    raise
                # Preserve the diagnostic component, but mirror the frozen
                # transition-attribution precedent: a zero-span record is not
                # line geometry and can never enter either exact matching.
                component["_line"] = None
                component["_line_matchability"] = "ZERO_LENGTH_SPAN"
            gates = component.get("geometry_gates")
            if component.get("geometry_certificate_valid") is True:
                if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
                    raise ValueError(f"{component_id}: valid certificate must pass every geometry gate")
            elif not str(component.get("first_rejection_code") or ""):
                raise ValueError(f"{component_id}: invalid component lacks first rejection code")
            components.append(component)
    return sorted(components, key=lambda row: str(row["local_transition_component_id"])), fragments, pairs


def _cause_category(
    component: Mapping[str, Any],
    *,
    fragments: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Any]],
) -> str:
    pair = pairs[str(component["_pair_id"])]
    fragment_ids = tuple(str(value) for value in pair.get("fragment_ids") or ())
    if len(fragment_ids) != 2 or any(value not in fragments for value in fragment_ids):
        raise ValueError("recovered certificate lacks two resolved local fragments")
    states = []
    for identifier in fragment_ids:
        coverage = fragments[identifier].get("global_plane_coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError("local fragment lacks global-plane coverage")
        states.append(str(coverage.get("coverage_state") or ""))
    if "NONE" in states:
        return CAUSE_CATEGORIES[0]
    if "PARTIAL" in states:
        return CAUSE_CATEGORIES[1]
    comparison = pair.get("global_pair_comparison")
    if states != ["FULL", "FULL"] or not isinstance(comparison, Mapping) or comparison.get("state") != "FULL_DISTINCT_GLOBAL_PAIR":
        raise ValueError("recovered certificate has no defined global-plane cause category")
    line_state = str(component.get("global_pair_line_state") or "")
    if line_state == "NON_EQUIVALENT":
        return CAUSE_CATEGORIES[2]
    if line_state == "EQUIVALENT":
        return CAUSE_CATEGORIES[3]
    raise ValueError("recovered certificate has invalid global-pair line state")


def _evaluate_gate(
    *,
    recovered_stage1_objects: int,
    independent_groups: int,
    a0_false_predictions_certified: int,
    lock_criteria: Mapping[str, Any] | None,
    lock_ready_for_gold: bool,
    successful_jobs: int = EXPECTED_CASE_COUNT,
    required_jobs: int = EXPECTED_CASE_COUNT,
) -> dict[str, Any]:
    prelock = {
        key: bool(lock_criteria is not None and lock_criteria.get(key) is True)
        for key in LOCK_CRITERION_KEYS
    }
    empirical = {
        "recovered_stage1_objects_at_least_14": recovered_stage1_objects
        >= MIN_RECOVERED_STAGE1_OBJECTS,
        "independent_family_groups_at_least_8": independent_groups
        >= MIN_INDEPENDENT_GROUPS,
        "a0_false_predictions_certified_at_most_8": a0_false_predictions_certified
        <= MAX_A0_FALSE_PREDICTIONS_CERTIFIED,
    }
    criteria = {**prelock, **empirical}
    ready = bool(
        lock_ready_for_gold is True
        and successful_jobs == required_jobs
        and all(prelock.values())
    )
    return {
        "thresholds": {
            "minimum_recovered_stage1_objects": MIN_RECOVERED_STAGE1_OBJECTS,
            "minimum_independent_family_groups": MIN_INDEPENDENT_GROUPS,
            "maximum_a0_false_predictions_certified": MAX_A0_FALSE_PREDICTIONS_CERTIFIED,
            "frozen_a0_false_prediction_count": FROZEN_A0_FALSE_PREDICTION_COUNT,
            "required_inference_safe_jobs": required_jobs,
        },
        "metrics": {
            "recovered_stage1_objects": recovered_stage1_objects,
            "independent_family_group_count": independent_groups,
            "a0_false_predictions_certified": a0_false_predictions_certified,
            "successful_inference_safe_jobs": successful_jobs,
            "lock_attestation_ready_for_gold": lock_ready_for_gold is True,
            "prelock_criterion_count": len(prelock),
            "empirical_criterion_count": len(empirical),
        },
        "ready_for_gold": ready,
        "prelock_criteria": prelock,
        "empirical_criteria": empirical,
        "criteria": criteria,
        "passed": bool(ready and all(empirical.values())),
    }


def attribute_local_plane_fragment_probe_gold(
    *,
    bundle: Mapping[str, Any],
    prior_attribution: Mapping[str, Any],
    a0_predictions: Mapping[str, Any],
    sealed_gold: Mapping[str, Any],
    grouped_split: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    expected_bundle_sha256: str,
    expected_lock_attestation_sha256: str,
    expected_case_count: int = EXPECTED_CASE_COUNT,
    expected_primary_stage1_count: int = PRIMARY_STAGE1_COUNT,
    enforce_frozen_false_count: bool = True,
) -> dict[str, Any]:
    """Attribute a locked LPF census; validation precedes all truth indexing."""

    # Deliberately first: no gold-bearing mapping is inspected before this lock.
    lock_criteria = _validate_bundle_lock(
        bundle,
        actual_bundle_sha256=source_hashes.get("local_plane_fragment_bundle_sha256", ""),
        expected_bundle_sha256=expected_bundle_sha256,
        expected_lock_attestation_sha256=expected_lock_attestation_sha256,
        expected_case_count=expected_case_count,
    )
    verified_hashes = _validate_frozen_sources(
        prior_attribution=prior_attribution,
        source_hashes=source_hashes,
    )

    bundle_cases = _index_cases(bundle, label="locked bundle")
    prediction_cases = _index_cases(a0_predictions, label="frozen A0 prediction")
    gold_cases = _index_cases(sealed_gold, label="sealed gold")
    split_cases = _index_cases(grouped_split, label="frozen grouped split")
    case_tokens = set(bundle_cases)
    if case_tokens != set(prediction_cases) or case_tokens != set(gold_cases) or case_tokens != set(split_cases):
        raise ValueError("bundle, A0 prediction, gold, and split case sets must match")

    prior_rows = prior_attribution.get("rows") or ()
    if not isinstance(prior_rows, Sequence) or isinstance(prior_rows, (str, bytes)):
        raise ValueError("prior attribution rows must be a sequence")
    primary_rows: dict[tuple[str, str], dict[str, Any]] = {}
    non_primary_stage_counts: Counter[int] = Counter()
    for raw in prior_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("prior attribution rows must be objects")
        stage = int(raw.get("first_failed_stage") or 0)
        if stage != PRIMARY_STAGE:
            non_primary_stage_counts[stage] += 1
            continue
        token = str(raw.get("case_token") or "")
        gold_id = str(raw.get("gold_id") or "")
        key = (token, gold_id)
        if not token or not gold_id or key in primary_rows:
            raise ValueError("frozen stage-1 row keys must be nonempty and unique")
        primary_rows[key] = dict(raw)
    if len(primary_rows) != expected_primary_stage1_count:
        raise ValueError(
            f"frozen primary stage-1 population must contain {expected_primary_stage1_count} rows"
        )

    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    output_rows: list[dict[str, Any]] = []
    recovery_matches: list[dict[str, Any]] = []
    false_matches: list[dict[str, Any]] = []
    recovered_groups: set[str] = set()
    observed_false_count = 0
    unmatchable_zero_length_component_count = 0
    cause_counts: Counter[str] = Counter()

    for token in sorted(case_tokens):
        bundle_row = bundle_cases[token]
        probe = _probe_case(bundle_row, token=token)
        prediction_case = prediction_cases[token]
        gold_case = gold_cases[token]
        split_case = split_cases[token]

        prediction_file_hash = _require_hash(
            prediction_case.get("prediction_owned_regions_sha256"),
            field=f"{token}.prediction_owned_regions_sha256",
        )
        if prediction_file_hash != _require_hash(
            bundle_row.get("a0_prediction_sha256"),
            field=f"{token}.bundle.a0_prediction_sha256",
        ):
            raise ValueError(f"{token}: frozen A0 prediction hash changed")
        scan_hash = _require_hash(bundle_row.get("scan_sha256"), field=f"{token}.bundle.scan_sha256")
        if _require_hash(gold_case.get("source_scan_sha256"), field=f"{token}.gold.source_scan_sha256") != scan_hash:
            raise ValueError(f"{token}: sealed-gold scan hash mismatch")

        group_id = str(
            split_case.get("split_group_id")
            or split_case.get("part_key")
            or split_case.get("part_family")
            or ""
        )
        if not group_id:
            raise ValueError(f"{token}: grouped split lacks an independent group ID")

        gold_records = [dict(value) for value in gold_case.get("manual_bends") or () if isinstance(value, Mapping)]
        gold_by_id = {str(value.get("bend_id") or ""): value for value in gold_records}
        if "" in gold_by_id or len(gold_by_id) != len(gold_records):
            raise ValueError(f"{token}: gold bend IDs must be nonempty and unique")
        prediction_records = [dict(value) for value in prediction_case.get("a0_predictions") or () if isinstance(value, Mapping)]

        # Freeze the 33 false segments independently from LPF certificate use.
        a0_gold_match = _exact_line_matching(
            gold_records,
            prediction_records,
            left_id_field="bend_id",
            right_id_field="bend_id",
            thresholds=thresholds,
        )
        false_ids = set(a0_gold_match["unmatched_right_ids"])
        frozen_false_records = [
            value for value in prediction_records if str(value.get("bend_id")) in false_ids
        ]
        observed_false_count += len(frozen_false_records)

        components, fragments, pairs = _component_records(probe)
        unmatchable_zero_length_component_count += sum(
            component.get("_line_matchability") == "ZERO_LENGTH_SPAN"
            for component in components
        )
        certificates = [
            {**dict(component["_line"]), "certificate_id": str(component["local_transition_component_id"])}
            for component in components
            if component.get("geometry_certificate_valid") is True
            and component.get("_line") is not None
        ]
        component_by_id = {
            str(component["local_transition_component_id"]): component
            for component in components
        }

        case_primary = [
            row for (case_token, _gold_id), row in primary_rows.items() if case_token == token
        ]
        primary_gold = []
        for row in sorted(case_primary, key=lambda value: str(value["gold_id"])):
            gold_id = str(row["gold_id"])
            if gold_id not in gold_by_id:
                raise ValueError(f"{token}: frozen stage-1 gold ID {gold_id} is absent")
            if str(row.get("split_group_id") or group_id) != group_id:
                raise ValueError(f"{token}: frozen stage-1 split group changed")
            if gold_id not in set(a0_gold_match["unmatched_left_ids"]):
                raise ValueError(f"{token}: frozen stage-1 row is no longer an A0 miss")
            primary_gold.append(gold_by_id[gold_id])

        recovery = _exact_line_matching(
            primary_gold,
            certificates,
            left_id_field="bend_id",
            right_id_field="certificate_id",
            thresholds=thresholds,
        )
        recovery_by_gold = {str(row["left_id"]): row for row in recovery["matches"]}
        for match in recovery["matches"]:
            component = component_by_id[str(match["right_id"])]
            cause = _cause_category(component, fragments=fragments, pairs=pairs)
            cause_counts[cause] += 1
            recovered_groups.add(group_id)
            recovery_matches.append(
                {
                    "case_token": token,
                    "split_group_id": group_id,
                    "gold_id": match["left_id"],
                    "certificate_id": match["right_id"],
                    "cause_category": cause,
                    "integer_geometric_cost": match["integer_geometric_cost"],
                }
            )

        # This matching is intentionally independent.  A certificate used for a
        # gold recovery remains eligible to certify an A0 false segment.
        false_result = _exact_line_matching(
            frozen_false_records,
            certificates,
            left_id_field="bend_id",
            right_id_field="certificate_id",
            thresholds=thresholds,
        )
        false_matches.extend(
            {
                "case_token": token,
                "a0_false_prediction_id": row["left_id"],
                "certificate_id": row["right_id"],
                "integer_geometric_cost": row["integer_geometric_cost"],
            }
            for row in false_result["matches"]
        )

        for gold in sorted(primary_gold, key=lambda value: str(value["bend_id"])):
            gold_id = str(gold["bend_id"])
            recovered = recovery_by_gold.get(gold_id)
            if recovered is not None:
                component = component_by_id[str(recovered["right_id"])]
                output_rows.append(
                    {
                        "case_token": token,
                        "split_group_id": group_id,
                        "gold_id": gold_id,
                        "primary_population": True,
                        "recovered": True,
                        "reason_code": "VALID_LOCAL_CERTIFICATE",
                        "certificate_id": recovered["right_id"],
                        "selected_component_id": recovered["right_id"],
                        "cause_category": _cause_category(component, fragments=fragments, pairs=pairs),
                        "integer_geometric_cost": recovered["integer_geometric_cost"],
                    }
                )
                continue

            gold_line = _line(gold, identifier=gold_id)
            matching_components: list[tuple[int, int, str, dict[str, Any]]] = []
            for component in components:
                if component.get("_line") is None:
                    continue
                metrics = pair_metrics(_endpoints(gold_line), _endpoints(component["_line"]))
                if is_match(metrics, thresholds):
                    matching_components.append(
                        (
                            -int(component.get("geometry_gate_depth") or 0),
                            _integer_geometric_cost(metrics),
                            str(component["local_transition_component_id"]),
                            component,
                        )
                    )
            matching_components.sort(key=lambda value: (value[0], value[1], value[2]))
            selected = matching_components[0] if matching_components else None
            output_rows.append(
                {
                    "case_token": token,
                    "split_group_id": group_id,
                    "gold_id": gold_id,
                    "primary_population": True,
                    "recovered": False,
                    "reason_code": (
                        "NO_MATCHING_LOCAL_COMPONENT"
                        if selected is None
                        else str(selected[3].get("first_rejection_code") or "")
                    ),
                    "certificate_id": None,
                    "selected_component_id": None if selected is None else selected[2],
                    "cause_category": None,
                    "integer_geometric_cost": None if selected is None else selected[1],
                    "selected_geometry_gate_depth": None if selected is None else -selected[0],
                }
            )

    if enforce_frozen_false_count and observed_false_count != FROZEN_A0_FALSE_PREDICTION_COUNT:
        raise ValueError(
            f"frozen A0 false-prediction set must contain {FROZEN_A0_FALSE_PREDICTION_COUNT} segments"
        )

    output_rows.sort(key=lambda row: (row["case_token"], row["gold_id"]))
    recovery_matches.sort(key=lambda row: (row["case_token"], row["gold_id"], row["certificate_id"]))
    false_matches.sort(key=lambda row: (row["case_token"], row["a0_false_prediction_id"], row["certificate_id"]))
    gate = _evaluate_gate(
        recovered_stage1_objects=len(recovery_matches),
        independent_groups=len(recovered_groups),
        a0_false_predictions_certified=len(false_matches),
        lock_criteria=lock_criteria,
        lock_ready_for_gold=bundle.get("ready_for_gold") is True,
        successful_jobs=len(bundle_cases),
        required_jobs=expected_case_count,
    )
    gate["metrics"].update(
        {
            "independent_family_group_ids": sorted(recovered_groups),
            "observed_frozen_a0_false_prediction_count": observed_false_count,
            "a0_false_certificate_fraction_of_frozen_33": round(
                len(false_matches) / FROZEN_A0_FALSE_PREDICTION_COUNT, 12
            ),
        }
    )

    return {
        "schema_version": 2,
        "kind": KIND,
        "research_only": True,
        "postlock": True,
        "primary_population": "frozen_transition_stage_1_only",
        "matching_profile": MATCHING_PROFILE,
        "local_plane_fragment_bundle_sha256": _require_hash(
            source_hashes["local_plane_fragment_bundle_sha256"],
            field="local_plane_fragment_bundle_sha256",
        ),
        "lock_attestation_sha256": _require_hash(
            bundle["lock_attestation_sha256"], field="lock_attestation_sha256"
        ),
        "replay_attestation_sha256": _require_hash(
            bundle["replay_attestation_sha256"], field="replay_attestation_sha256"
        ),
        **verified_hashes,
        "line_matching_contract": {
            "profile": MATCHING_PROFILE,
            "axis_angle_deg": thresholds.angle_deg,
            "transverse_offset_mm": thresholds.transverse_offset_mm,
            "gold_span_overlap": thresholds.gold_span_overlap,
            "certificate_span_overlap": thresholds.prediction_span_overlap,
            "stages": [
                "maximum_cardinality",
                "minimum_total_integer_geometric_cost",
                "lexicographically_smallest_ordered_assignment",
            ],
            "integer_cost_scale": 1_000_000,
            "hard_caps": {
                "maximum_left_vertices": MAX_MATCH_LEFT_VERTICES,
                "maximum_right_vertices": MAX_MATCH_RIGHT_VERTICES,
                "maximum_edges": MAX_MATCH_EDGES,
                "maximum_edge_cost_int": MAX_MATCH_EDGE_COST_INT,
            },
            "solver": {
                "engine": "OR_TOOLS_CP_SAT",
                "num_search_workers": 1,
                "random_seed": 0,
                "search_branching": "FIXED_SEARCH",
                "wall_clock_cutoff": None,
            },
        },
        "summary": {
            "primary_stage1_count": len(primary_rows),
            "recovered_stage1_count": len(recovery_matches),
            "unrecovered_stage1_count": len(primary_rows) - len(recovery_matches),
            "unmatchable_zero_length_component_count": int(
                unmatchable_zero_length_component_count
            ),
            "excluded_non_primary_stage_counts": {
                str(stage): count for stage, count in sorted(non_primary_stage_counts.items())
            },
            "new_failure_reason_counts": dict(
                sorted(Counter(row["reason_code"] for row in output_rows if not row["recovered"]).items())
            ),
            "cause_category_counts": {
                category: int(cause_counts.get(category, 0)) for category in CAUSE_CATEGORIES
            },
        },
        "professor_probe_gate": gate,
        "recovery_matches": recovery_matches,
        "a0_false_certificate_matches": false_matches,
        "rows": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-lock-attestation-sha256", required=True)
    parser.add_argument("--prior-attribution", required=True, type=Path)
    parser.add_argument("--a0-predictions", required=True, type=Path)
    parser.add_argument("--sealed-gold", required=True, type=Path)
    parser.add_argument("--grouped-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # Mandatory lock boundary: read and validate only the inference bundle first.
    bundle = _read(args.bundle.resolve())
    bundle_sha = _sha256(args.bundle.resolve())
    _validate_bundle_lock(
        bundle,
        actual_bundle_sha256=bundle_sha,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_lock_attestation_sha256=args.expected_lock_attestation_sha256,
        expected_case_count=EXPECTED_CASE_COUNT,
    )

    protected_paths = {
        "prior_attribution_sha256": args.prior_attribution.resolve(),
        "a0_prediction_sha256": args.a0_predictions.resolve(),
        "gold_sha256": args.sealed_gold.resolve(),
        "grouped_split_sha256": args.grouped_split.resolve(),
    }
    # Hash refusal happens before any protected JSON is parsed.  Verify the
    # non-gold locks first; the sealed-gold file is touched only afterward.
    source_hashes = {"local_plane_fragment_bundle_sha256": bundle_sha}
    for field, expected in (
        ("prior_attribution_sha256", FROZEN_PRIOR_ATTRIBUTION_SHA256),
        ("a0_prediction_sha256", FROZEN_A0_PREDICTION_SHA256),
        ("grouped_split_sha256", FROZEN_GROUPED_SPLIT_SHA256),
    ):
        source_hashes[field] = _sha256(protected_paths[field])
        if source_hashes[field] != expected:
            raise ValueError(f"{field} differs from the preregistered frozen value")
    source_hashes["gold_sha256"] = _sha256(protected_paths["gold_sha256"])
    if source_hashes["gold_sha256"] != FROZEN_GOLD_SHA256:
        raise ValueError("gold_sha256 differs from the preregistered frozen value")

    prior = _read(protected_paths["prior_attribution_sha256"])
    predictions = _read(protected_paths["a0_prediction_sha256"])
    split = _read(protected_paths["grouped_split_sha256"])
    gold = _read(protected_paths["gold_sha256"])
    result = attribute_local_plane_fragment_probe_gold(
        bundle=bundle,
        prior_attribution=prior,
        a0_predictions=predictions,
        sealed_gold=gold,
        grouped_split=split,
        source_hashes=source_hashes,
        expected_bundle_sha256=args.expected_bundle_sha256,
        expected_lock_attestation_sha256=args.expected_lock_attestation_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "primary_stage1_count": result["summary"]["primary_stage1_count"],
                "recovered_stage1_count": result["summary"]["recovered_stage1_count"],
                "a0_false_predictions_certified": result["professor_probe_gate"]["metrics"]["a0_false_predictions_certified"],
                "professor_probe_gate_passed": result["professor_probe_gate"]["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
