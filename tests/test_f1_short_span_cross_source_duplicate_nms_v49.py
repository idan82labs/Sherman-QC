from __future__ import annotations

from copy import deepcopy

from backend.f1_short_span_cross_source_duplicate_nms_v49 import compose_v49


def _prediction(candidate_id, start, end, source_kind):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": list(start),
        "line_end": list(end),
        "selection_route": "test",
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": source_kind,
    }


def test_suppresses_near_equal_short_cross_source_duplicate():
    source = {
        "predictions": [
            _prediction("long", (0, 0, 0), (450, 0, 0), "cylindrical_standard"),
            _prediction("short-a", (0, 100, 0), (36, 100, 0), "plane_lattice"),
            _prediction("short-b", (1, 101, 0), (35, 101, 0), "scan_mesh_adjacency"),
        ]
    }
    result = compose_v49(source)
    assert len(result["predictions"]) == 2
    change = result["short_span_cross_source_duplicate_nms_changes"][0]
    assert change["suppressed_candidate_id"] == "short-b"
    assert change["retained_candidate_id"] == "short-a"


def test_preserves_same_source_short_lines():
    source = {
        "predictions": [
            _prediction("long", (0, 0, 0), (450, 0, 0), "cylindrical_standard"),
            _prediction("short-a", (0, 100, 0), (36, 100, 0), "plane_lattice"),
            _prediction("short-b", (1, 101, 0), (35, 101, 0), "plane_lattice"),
        ]
    }
    assert len(compose_v49(source)["predictions"]) == 3


def test_preserves_long_cross_source_axes():
    source = {
        "predictions": [
            _prediction("long-a", (0, 0, 0), (100, 0, 0), "plane_lattice"),
            _prediction("long-b", (1, 1, 0), (99, 1, 0), "scan_mesh_adjacency"),
        ]
    }
    assert len(compose_v49(source)["predictions"]) == 2


def test_preserves_short_lines_with_material_offset_or_incomplete_overlap():
    source = {
        "predictions": [
            _prediction("long", (0, 0, 0), (450, 0, 0), "cylindrical_standard"),
            _prediction("short-a", (0, 100, 0), (36, 100, 0), "plane_lattice"),
            _prediction("offset", (0, 103, 0), (36, 103, 0), "scan_mesh_adjacency"),
            _prediction("partial", (20, 101, 0), (54, 101, 0), "scan_mesh_adjacency"),
        ]
    }
    assert len(compose_v49(source)["predictions"]) == 4


def test_input_permutation_is_exact():
    predictions = [
        _prediction("long", (0, 0, 0), (450, 0, 0), "cylindrical_standard"),
        _prediction("short-a", (0, 100, 0), (36, 100, 0), "plane_lattice"),
        _prediction("short-b", (1, 101, 0), (35, 101, 0), "scan_mesh_adjacency"),
    ]
    first = compose_v49({"predictions": predictions})
    second = compose_v49({"predictions": list(reversed(deepcopy(predictions)))})
    assert first["predictions"] == second["predictions"]
    assert first["short_span_cross_source_duplicate_nms_changes"] == second[
        "short_span_cross_source_duplicate_nms_changes"
    ]
    assert first["v49_result_sha256"] == second["v49_result_sha256"]
