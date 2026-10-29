from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import random
from time import perf_counter

import pytest

from scripts.build_f1_local_plane_fragment_probe_bundle import (
    build_lock_attestation,
)

from scripts.attribute_f1_local_plane_fragment_probe_gold import (
    BUNDLE_KIND,
    BUNDLE_PROFILE,
    FROZEN_A0_PREDICTION_SHA256,
    FROZEN_GOLD_SHA256,
    FROZEN_GROUPED_SPLIT_SHA256,
    FROZEN_PRIOR_ATTRIBUTION_SHA256,
    FROZEN_TRANSITION_CENSUS_SHA256,
    LOCK_ATTESTATION_KIND,
    LOCK_CRITERION_KEYS,
    MAX_MATCH_EDGE_COST_INT,
    MAX_MATCH_EDGES,
    MAX_MATCH_LEFT_VERTICES,
    MAX_MATCH_RIGHT_VERTICES,
    PRIOR_KIND,
    PROBE_KIND,
    _CandidateEdge,
    _cause_category,
    _evaluate_gate,
    _sha256_json,
    _solve_exact_matching,
    _validate_bundle_lock,
    attribute_local_plane_fragment_probe_gold,
)


BUNDLE_SHA = "e" * 64
REPLAY_SHA = "3" * 64
PREDICTION_FILE_SHA = "1" * 64
SCAN_SHA = "2" * 64
SOURCE_HASHES = {
    "local_plane_fragment_bundle_sha256": BUNDLE_SHA,
    "prior_attribution_sha256": FROZEN_PRIOR_ATTRIBUTION_SHA256,
    "a0_prediction_sha256": FROZEN_A0_PREDICTION_SHA256,
    "gold_sha256": FROZEN_GOLD_SHA256,
    "grouped_split_sha256": FROZEN_GROUPED_SPLIT_SHA256,
}


def _line(identifier: str, y: float) -> dict:
    return {
        "bend_id": identifier,
        "span_endpoints": [[0.0, y, 0.0], [20.0, y, 0.0]],
    }


def _edge(left: str, right: str, cost: int) -> _CandidateEdge:
    return _CandidateEdge(
        left_id=left,
        right_id=right,
        cost=cost,
        metrics={
            "axis_angle_deg": 0.0,
            "transverse_offset_mm": 0.0,
            "gold_span_overlap": 1.0,
            "prediction_span_overlap": 1.0,
        },
    )


def _probe(*, certificate_y: float = 10.0, valid: bool = True) -> dict:
    gates = {"minimum_parent_count": valid, "line_residual": valid}
    return {
        "schema_version": 2,
        "kind": PROBE_KIND,
        "profile": BUNDLE_PROFILE,
        "inference_safe": True,
        "research_only": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "a0_prediction_sha256": PREDICTION_FILE_SHA,
        "local_plane_fragment_table": [
            {
                "local_plane_fragment_id": "LPF_A",
                "global_plane_coverage": {"coverage_state": "NONE"},
            },
            {
                "local_plane_fragment_id": "LPF_B",
                "global_plane_coverage": {"coverage_state": "FULL"},
            },
        ],
        "local_plane_pair_records": [
            {
                "local_plane_pair_id": "LPP_1",
                "fragment_ids": ["LPF_A", "LPF_B"],
                "global_pair_comparison": {"state": "NO_DISTINCT_FULL_GLOBAL_PAIR"},
                "components": [
                    {
                        "local_transition_component_id": "LTC_1",
                        "intersection_span_endpoints": [
                            [0.0, certificate_y, 0.0],
                            [20.0, certificate_y, 0.0],
                        ],
                        "geometry_gate_depth": 13 if valid else 2,
                        "geometry_gates": gates,
                        "first_rejection_code": None if valid else "COMPONENT_TOO_SMALL",
                        "geometry_certificate_valid": valid,
                        "global_pair_line_state": "UNAVAILABLE",
                    }
                ],
            }
        ],
    }


def _lock_attestation(
    *, criterion_overrides: dict[str, bool] | None = None, ready: bool = True
) -> dict:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    criteria.update(criterion_overrides or {})
    record = {
        "kind": LOCK_ATTESTATION_KIND,
        "criteria": criteria,
        "ready_for_gold": ready,
    }
    record["lock_attestation_sha256"] = _sha256_json(record)
    return record


def _bundle(
    probe: dict,
    *,
    criterion_overrides: dict[str, bool] | None = None,
    ready: bool = True,
) -> dict:
    lock = _lock_attestation(
        criterion_overrides=criterion_overrides,
        ready=ready,
    )
    return {
        "schema_version": 2,
        "kind": BUNDLE_KIND,
        "profile": BUNDLE_PROFILE,
        "locked": ready,
        "ready_for_gold": ready,
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
        "replay_attestation_sha256": REPLAY_SHA,
        "lock_attestation_sha256": lock["lock_attestation_sha256"],
        "lock_attestation": lock,
        "case_count": 1,
        "successful_case_count": 1,
        "cases": [
            {
                "case_token": "C001",
                "scan_sha256": SCAN_SHA,
                "a0_prediction_sha256": PREDICTION_FILE_SHA,
                "probe": probe,
            }
        ],
    }


def _prior(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kind": PRIOR_KIND,
        "postlock": True,
        "prediction_sha256": FROZEN_A0_PREDICTION_SHA256,
        "gold_sha256": FROZEN_GOLD_SHA256,
        "grouped_split_sha256": FROZEN_GROUPED_SPLIT_SHA256,
        "diagnostic_census_sha256": FROZEN_TRANSITION_CENSUS_SHA256,
        "rows": rows,
    }


def _inputs(*, prior_rows: list[dict], gold: list[dict], predictions: list[dict], probe: dict) -> dict:
    bundle = _bundle(probe)
    return {
        "bundle": bundle,
        "prior_attribution": _prior(prior_rows),
        "a0_predictions": {
            "cases": [
                {
                    "case_token": "C001",
                    "prediction_owned_regions_sha256": PREDICTION_FILE_SHA,
                    "a0_predictions": predictions,
                }
            ]
        },
        "sealed_gold": {
            "cases": [
                {
                    "case_token": "C001",
                    "source_scan_sha256": SCAN_SHA,
                    "manual_bends": gold,
                }
            ]
        },
        "grouped_split": {
            "cases": [{"case_token": "C001", "split_group_id": "GROUP_A"}]
        },
        "source_hashes": SOURCE_HASHES,
        "expected_bundle_sha256": BUNDLE_SHA,
        "expected_lock_attestation_sha256": bundle["lock_attestation_sha256"],
        "expected_case_count": 1,
        "enforce_frozen_false_count": False,
    }


def test_exact_match_maximizes_cardinality_before_geometric_cost() -> None:
    selected = _solve_exact_matching(
        ["A", "B"],
        ["X", "Y"],
        [_edge("A", "X", 0), _edge("A", "Y", 10), _edge("B", "X", 10)],
    )
    assert [(row.left_id, row.right_id) for row in selected] == [("A", "Y"), ("B", "X")]


def test_exact_match_minimizes_cost_then_uses_lexicographic_assignment() -> None:
    lower_cost = _solve_exact_matching(
        ["A", "B"],
        ["X", "Y"],
        [
            _edge("A", "X", 2),
            _edge("A", "Y", 0),
            _edge("B", "X", 0),
            _edge("B", "Y", 2),
        ],
    )
    assert [(row.left_id, row.right_id) for row in lower_cost] == [("A", "Y"), ("B", "X")]

    tied = _solve_exact_matching(
        ["B", "A"],
        ["Y", "X"],
        [
            _edge("B", "Y", 1),
            _edge("A", "Y", 1),
            _edge("B", "X", 1),
            _edge("A", "X", 1),
        ],
    )
    assert [(row.left_id, row.right_id) for row in tied] == [("A", "X"), ("B", "Y")]


def test_one_certificate_cannot_recover_two_gold_lines() -> None:
    selected = _solve_exact_matching(
        ["G1", "G2"], ["C1"], [_edge("G1", "C1", 0), _edge("G2", "C1", 0)]
    )
    assert [(row.left_id, row.right_id) for row in selected] == [("G1", "C1")]


def test_bundle_hash_refusal_precedes_truth_validation() -> None:
    inputs = _inputs(
        prior_rows=[],
        gold=[{"malformed": "truth must not be inspected"}],
        predictions=[],
        probe=_probe(),
    )
    inputs["expected_bundle_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bundle SHA-256 mismatch"):
        attribute_local_plane_fragment_probe_gold(
            **inputs,
            expected_primary_stage1_count=0,
        )


def test_only_frozen_stage1_rows_are_primary_and_input_is_not_mutated() -> None:
    prior_rows = [
        {
            "case_token": "C001",
            "split_group_id": "GROUP_A",
            "gold_id": "G_STAGE1",
            "first_failed_stage": 1,
        },
        {
            "case_token": "C001",
            "split_group_id": "GROUP_A",
            "gold_id": "G_STAGE4",
            "first_failed_stage": 4,
        },
    ]
    inputs = _inputs(
        prior_rows=prior_rows,
        gold=[_line("G_STAGE1", 10.0), _line("G_STAGE4", 10.0)],
        predictions=[],
        probe=_probe(certificate_y=10.0),
    )
    frozen = deepcopy(inputs)
    result = attribute_local_plane_fragment_probe_gold(
        **inputs,
        expected_primary_stage1_count=1,
    )
    assert inputs == frozen
    assert result["schema_version"] == 2
    assert result["kind"] == "postlock_gold_local_plane_fragment_attribution_v2"
    assert [row["gold_id"] for row in result["rows"]] == ["G_STAGE1"]
    assert result["rows"][0]["recovered"] is True
    assert result["rows"][0]["cause_category"] == "LOCAL_FRAGMENT_NO_GLOBAL_PLANE"
    assert result["summary"]["excluded_non_primary_stage_counts"] == {"4": 1}


def test_a0_false_certificate_matching_is_independent_and_one_to_one() -> None:
    inputs = _inputs(
        prior_rows=[],
        gold=[_line("G_OTHER", 0.0)],
        predictions=[_line("FP_A", 10.0), _line("FP_B", 10.0)],
        probe=_probe(certificate_y=10.0),
    )
    result = attribute_local_plane_fragment_probe_gold(
        **inputs,
        expected_primary_stage1_count=0,
    )
    assert result["professor_probe_gate"]["metrics"]["observed_frozen_a0_false_prediction_count"] == 2
    assert result["professor_probe_gate"]["metrics"]["a0_false_predictions_certified"] == 1
    assert result["a0_false_certificate_matches"][0]["a0_false_prediction_id"] == "FP_A"


def test_zero_length_component_is_audited_but_never_line_matched() -> None:
    probe = _probe(certificate_y=10.0)
    component = probe["local_plane_pair_records"][0]["components"][0]
    component["intersection_span_endpoints"] = [
        [4.0, 10.0, 2.0],
        [4.0, 10.0, 2.0],
    ]
    inputs = _inputs(
        prior_rows=[
            {
                "case_token": "C001",
                "split_group_id": "GROUP_A",
                "gold_id": "G_STAGE1",
                "first_failed_stage": 1,
            }
        ],
        gold=[_line("G_STAGE1", 10.0)],
        predictions=[_line("A0_FP", 30.0)],
        probe=probe,
    )
    result = attribute_local_plane_fragment_probe_gold(
        **inputs,
        expected_primary_stage1_count=1,
    )
    assert result["summary"]["unmatchable_zero_length_component_count"] == 1
    assert result["summary"]["recovered_stage1_count"] == 0
    assert result["rows"][0]["reason_code"] == "NO_MATCHING_LOCAL_COMPONENT"
    assert result["a0_false_certificate_matches"] == []


@pytest.mark.parametrize(
    ("recovered", "groups", "false_count", "expected"),
    [
        (14, 8, 8, True),
        (13, 8, 8, False),
        (14, 7, 8, False),
        (14, 8, 9, False),
    ],
)
def test_exact_14_8_8_gate_boundaries(
    recovered: int, groups: int, false_count: int, expected: bool
) -> None:
    result = _evaluate_gate(
        recovered_stage1_objects=recovered,
        independent_groups=groups,
        a0_false_predictions_certified=false_count,
        lock_criteria={key: True for key in LOCK_CRITERION_KEYS},
        lock_ready_for_gold=True,
    )
    assert result["passed"] is expected
    assert result["thresholds"]["maximum_a0_false_predictions_certified"] == 8


def test_all_four_postlock_cause_categories_are_exact() -> None:
    component = {"_pair_id": "PAIR", "global_pair_line_state": "UNAVAILABLE"}
    pair = {
        "fragment_ids": ["A", "B"],
        "global_pair_comparison": {"state": "NO_DISTINCT_FULL_GLOBAL_PAIR"},
    }
    fragments = {
        "A": {"global_plane_coverage": {"coverage_state": "NONE"}},
        "B": {"global_plane_coverage": {"coverage_state": "FULL"}},
    }
    assert _cause_category(component, fragments=fragments, pairs={"PAIR": pair}) == "LOCAL_FRAGMENT_NO_GLOBAL_PLANE"

    fragments["A"]["global_plane_coverage"]["coverage_state"] = "PARTIAL"
    assert _cause_category(component, fragments=fragments, pairs={"PAIR": pair}) == "LOCAL_FRAGMENT_PARTIAL_GLOBAL_COVERAGE"

    fragments["A"]["global_plane_coverage"]["coverage_state"] = "FULL"
    pair["global_pair_comparison"]["state"] = "FULL_DISTINCT_GLOBAL_PAIR"
    component["global_pair_line_state"] = "NON_EQUIVALENT"
    assert _cause_category(component, fragments=fragments, pairs={"PAIR": pair}) == "FULL_GLOBAL_PAIR_LINE_NON_EQUIVALENT"

    component["global_pair_line_state"] = "EQUIVALENT"
    assert _cause_category(component, fragments=fragments, pairs={"PAIR": pair}) == "FULL_GLOBAL_PAIR_LINE_EQUIVALENT_RUN_ABSENT"


def test_expected_lock_hash_and_replay_readiness_precede_truth_validation() -> None:
    inputs = _inputs(
        prior_rows=[],
        gold=[{"malformed": "must remain unopened"}],
        predictions=[],
        probe=_probe(),
    )
    inputs["expected_lock_attestation_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from expected lock"):
        attribute_local_plane_fragment_probe_gold(
            **inputs,
            expected_primary_stage1_count=0,
        )
    blocked_bundle = _bundle(
        _probe(),
        criterion_overrides={"designated_replay_attestation_valid": False},
        ready=False,
    )
    inputs["bundle"] = blocked_bundle
    inputs["expected_lock_attestation_sha256"] = blocked_bundle[
        "lock_attestation_sha256"
    ]
    with pytest.raises(ValueError, match="not ready for gold"):
        attribute_local_plane_fragment_probe_gold(
            **inputs,
            expected_primary_stage1_count=0,
        )


def test_postlock_accepts_bundle_builders_exact_lock_record_contract() -> None:
    bundle = _bundle(_probe())
    lock = build_lock_attestation({key: True for key in LOCK_CRITERION_KEYS})
    bundle["lock_attestation"] = lock
    bundle["lock_attestation_sha256"] = lock["lock_attestation_sha256"]
    criteria = _validate_bundle_lock(
        bundle,
        actual_bundle_sha256=BUNDLE_SHA,
        expected_bundle_sha256=BUNDLE_SHA,
        expected_lock_attestation_sha256=lock["lock_attestation_sha256"],
        expected_case_count=1,
    )
    assert criteria == {key: True for key in LOCK_CRITERION_KEYS}

@pytest.mark.parametrize("criterion", LOCK_CRITERION_KEYS)
def test_each_missing_or_false_prelock_criterion_forces_stop(criterion: str) -> None:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    criteria[criterion] = False
    result = _evaluate_gate(
        recovered_stage1_objects=14,
        independent_groups=8,
        a0_false_predictions_certified=8,
        lock_criteria=criteria,
        lock_ready_for_gold=True,
    )
    assert result["prelock_criteria"][criterion] is False
    assert result["ready_for_gold"] is False
    assert result["passed"] is False


def test_missing_lock_mapping_never_passes_even_at_14_8_8() -> None:
    result = _evaluate_gate(
        recovered_stage1_objects=14,
        independent_groups=8,
        a0_false_predictions_certified=8,
        lock_criteria=None,
        lock_ready_for_gold=True,
    )
    assert result["passed"] is False
    assert not any(result["prelock_criteria"].values())


def test_ready_for_gold_false_stops_even_with_all_criteria_and_14_8_8() -> None:
    result = _evaluate_gate(
        recovered_stage1_objects=14,
        independent_groups=8,
        a0_false_predictions_certified=8,
        lock_criteria={key: True for key in LOCK_CRITERION_KEYS},
        lock_ready_for_gold=False,
    )
    assert all(result["criteria"].values())
    assert result["ready_for_gold"] is False
    assert result["passed"] is False


def _bruteforce(
    left_ids: list[str], right_ids: list[str], edges: list[_CandidateEdge]
) -> tuple[tuple[str, str], ...]:
    best: tuple[tuple[int, int, tuple[tuple[str, str], ...]], tuple[tuple[str, str], ...]] | None = None
    for count in range(len(edges) + 1):
        for selected in combinations(edges, count):
            if len({edge.left_id for edge in selected}) != count:
                continue
            if len({edge.right_id for edge in selected}) != count:
                continue
            pairs = tuple(sorted((edge.left_id, edge.right_id) for edge in selected))
            key = (-count, sum(edge.cost for edge in selected), pairs)
            if best is None or key < best[0]:
                best = (key, pairs)
    assert best is not None
    return best[1]


def test_matching_matches_bruteforce_and_is_edge_order_invariant() -> None:
    generator = random.Random(20260805)
    for size in range(1, 5):
        left = [f"L{index}" for index in range(size)]
        right = [f"R{index}" for index in range(size)]
        for _trial in range(25):
            edges = [
                _edge(left_id, right_id, generator.randrange(5))
                for left_id in left
                for right_id in right
                if generator.random() < 0.55
            ]
            expected = _bruteforce(left, right, edges)
            first = _solve_exact_matching(left, right, edges)
            replay = _solve_exact_matching(left[::-1], right[::-1], edges[::-1])
            first_pairs = tuple((edge.left_id, edge.right_id) for edge in first)
            replay_pairs = tuple((edge.left_id, edge.right_id) for edge in replay)
            assert first_pairs == expected
            assert replay_pairs == expected


@pytest.mark.parametrize(
    ("left", "right", "edges", "message"),
    [
        (["L"], ["R"], [_edge("L", "R", -1)], "cost exceeds"),
        (
            ["L"],
            ["R"],
            [_edge("L", "R", MAX_MATCH_EDGE_COST_INT + 1)],
            "cost exceeds",
        ),
        (
            [f"L{i}" for i in range(MAX_MATCH_LEFT_VERTICES + 1)],
            [],
            [],
            "left-vertex cap",
        ),
        (
            [],
            [f"R{i}" for i in range(MAX_MATCH_RIGHT_VERTICES + 1)],
            [],
            "right-vertex cap",
        ),
    ],
)
def test_matching_hard_caps_fail_closed(
    left: list[str], right: list[str], edges: list[_CandidateEdge], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _solve_exact_matching(left, right, edges)


def test_matching_rejects_duplicate_edges_and_edge_cap() -> None:
    duplicate = _edge("L", "R", 0)
    with pytest.raises(ValueError, match="unique"):
        _solve_exact_matching(["L"], ["R"], [duplicate, duplicate])
    repeated = [_edge("L", "R", 0)] * (MAX_MATCH_EDGES + 1)
    with pytest.raises(ValueError, match="edge cap"):
        _solve_exact_matching(["L"], ["R"], repeated)


def test_configured_vertex_cap_fixture_is_deterministic_and_bounded() -> None:
    left = [f"L{index:03d}" for index in range(MAX_MATCH_LEFT_VERTICES)]
    right = [f"R{index:05d}" for index in range(MAX_MATCH_RIGHT_VERTICES)]
    edges = [_edge(left[index], right[index], index) for index in range(len(left))]
    started = perf_counter()
    first = _solve_exact_matching(left, right, edges)
    replay = _solve_exact_matching(left[::-1], right[::-1], edges[::-1])
    elapsed = perf_counter() - started
    assert len(first) == MAX_MATCH_LEFT_VERTICES
    assert [(edge.left_id, edge.right_id) for edge in first] == [
        (edge.left_id, edge.right_id) for edge in replay
    ]
    # This is a resource fixture, not a wall-clock acceptance threshold.
    assert elapsed >= 0.0
