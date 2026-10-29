from __future__ import annotations

import hashlib
import itertools
import json
import math

import pytest

from research.f1_candidate_ir_v1 import (
    CandidateIRValidationError,
    canonical_json_bytes,
    canonicalize_candidate,
    canonicalize_candidate_set,
    validate_stage_graph,
)


BASE = {"line_start": [0, 0, 0], "line_end": [20, 0, 0], "source_route": "mesh", "source_stage": "birth", "source_ids": ["x"]}


def test_historical_line_shapes_canonicalize_to_same_geometry() -> None:
    variants = [
        BASE,
        {"span_endpoints": [[20, 0, 0], [0, 0, 0]], "route": "mesh", "stage": "birth", "bend_id": "x"},
        {"anchor": [10, 0, 0], "axis_direction": [-2, 0, 0], "span_mm": 20, "source_route": "mesh", "source_stage": "birth", "source_hypothesis_id": "x"},
        {"render_line_start": [0, 0, 0], "render_line_end": [20, 0, 0], "source_route": "mesh", "source_stage": "birth", "candidate_id": "x"},
    ]
    rows = [canonicalize_candidate(value) for value in variants]
    for row in rows:
        assert row["line_start_mm"] == [0.0, 0.0, 0.0]
        assert row["line_end_mm"] == [20.0, 0.0, 0.0]
        assert row["axis_direction"] == [1.0, 0.0, 0.0]
        assert row["span_mm"] == 20.0
    assert len({row["candidate_id"] for row in rows}) == 1
    assert len({canonical_json_bytes(row) for row in rows}) == 1


def test_multiple_representations_must_agree() -> None:
    value = {**BASE, "span_endpoints": [[0, 1, 0], [20, 1, 0]]}
    with pytest.raises(CandidateIRValidationError, match="disagree"):
        canonicalize_candidate(value)


def test_features_are_finite_sorted_and_do_not_change_identity() -> None:
    first = canonicalize_candidate({**BASE, "features": {"z": 1, "a": -0.0}})
    second = canonicalize_candidate({**BASE, "features": {"a": 0.0, "z": 2}})
    assert list(first["features"]) == ["a", "z"]
    assert first["features"]["a"] == 0.0
    assert first["candidate_id"] == second["candidate_id"]
    with pytest.raises(CandidateIRValidationError):
        canonicalize_candidate({**BASE, "features": {"bad": float("nan")}})


@pytest.mark.parametrize("bad", [
    {"line_start": [0, 0, 0], "source_route": "x", "source_stage": "y"},
    {**BASE, "line_end": [float("inf"), 0, 0]},
    {**BASE, "line_end": [0, 0, 0]},
    {**BASE, "points": [[0, 0, 0]]},
    {**BASE, "metadata": {"family_id": "secret"}},
    {**BASE, "metadata": {"expected count": 3}},
    {**BASE, "metadata": {"expectedCount": 3}},
    {**BASE, "features": {"target-count": 3}},
    {**BASE, "support_status": "NO_BEND"},
    {**BASE, "source_ids": []},
    {**BASE, "features": {"a": 1, " a ": 2}},
])
def test_invalid_or_leaky_candidates_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(CandidateIRValidationError):
        canonicalize_candidate(bad)


def test_set_serialization_is_permutation_invariant_and_rejects_duplicates() -> None:
    rows = [BASE, {**BASE, "line_start": [0, 5, 0], "line_end": [20, 5, 0], "source_ids": ["y"]}]
    hashes = set()
    for ordering in itertools.permutations(rows):
        hashes.add(hashlib.sha256(canonical_json_bytes(canonicalize_candidate_set(list(ordering)))).hexdigest())
    assert len(hashes) == 1
    with pytest.raises(CandidateIRValidationError, match="duplicate"):
        canonicalize_candidate_set([BASE, dict(BASE)])


def test_half_even_quantization_and_negative_zero_are_stable() -> None:
    row = canonicalize_candidate({**BASE, "line_start": [-0.0, 0.0000005, 0], "line_end": [20, 0.0000005, 0]})
    assert row["line_start_mm"][0] == 0.0
    assert row["line_start_mm"][1] == 0.0
    assert "-0.0" not in canonical_json_bytes(row).decode()


def test_stage_graph_is_canonical_and_rejects_cycles() -> None:
    sha = "a" * 64
    graph = validate_stage_graph({"schema": "F1_STAGE_GRAPH_V1", "stages": [
        {"stage_id": "select", "implementation_sha256": sha, "parent_stage_ids": ["birth"], "input_schema": "F1_CANDIDATE_SET_V1", "output_schema": "F1_CANDIDATE_SET_V1"},
        {"stage_id": "birth", "implementation_sha256": sha, "parent_stage_ids": [], "input_schema": "PLY", "output_schema": "F1_CANDIDATE_SET_V1"},
    ]})
    assert [row["stage_id"] for row in graph["stages"]] == ["birth", "select"]
    with pytest.raises(CandidateIRValidationError, match="cycle"):
        validate_stage_graph({"schema": "F1_STAGE_GRAPH_V1", "stages": [
            {"stage_id": "a", "implementation_sha256": sha, "parent_stage_ids": ["b"], "input_schema": "x", "output_schema": "x"},
            {"stage_id": "b", "implementation_sha256": sha, "parent_stage_ids": ["a"], "input_schema": "x", "output_schema": "x"},
        ]})
