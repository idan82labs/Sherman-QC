from pathlib import Path

import numpy as np

import backend.f1_mesh_section_turn_locus_reselection_v55 as v55


def _sharp_fixture():
    s_values, yz = v55._profile_points(((0.0, 90.0),))
    vertices, triangles = v55._extruded_fixture_mesh(s_values, yz)
    index = v55.MeshSectionIndex.from_arrays(
        vertices, triangles, scale_override=(200.0, 0.5)
    )
    return index, s_values, yz


def test_canonical_line_reverses_direction_and_normalizes_negative_zero():
    first = v55.canonical_line(
        {"line_start": [5.0, -0.0, 0.0], "line_end": [0.0, -0.0, 0.0]}, 0
    )
    second = v55.canonical_line(
        {"line_start": [0.0, 0.0, 0.0], "line_end": [5.0, 0.0, 0.0]}, 1
    )
    assert first.geometry_bytes == second.geometry_bytes
    assert first.geometry_sha256 == second.geometry_sha256
    assert np.signbit(first.canonical_start).sum() == 0


def test_centered_sharp_fold_is_measured_at_all_nine_stations():
    index, s_values, yz = _sharp_fixture()
    measurement = v55.measure_candidate(
        index, v55._fixture_line(s_values, yz, 0.0)
    )
    assert measurement.valid_turn_count == 9
    assert measurement.centered_count == 9
    assert measurement.offset_count == 0
    assert v55._addition_decision(measurement) == (True, "PROPOSED")
    assert v55._prune_decision(measurement)[0] is False


def test_offset_fold_prunes_flat_axis_and_accepts_true_axis():
    offset = 0.40 * 9.6
    s_values, yz = v55._profile_points(((offset, 90.0),))
    vertices, triangles = v55._extruded_fixture_mesh(s_values, yz)
    index = v55.MeshSectionIndex.from_arrays(
        vertices, triangles, scale_override=(200.0, 0.5)
    )
    incumbent = v55.measure_candidate(
        index, v55._fixture_line(s_values, yz, 0.0)
    )
    true_axis = v55.measure_candidate(
        index, v55._fixture_line(s_values, yz, offset, 1)
    )
    assert incumbent.offset_count == 9
    assert v55._prune_decision(incumbent)[0] is True
    assert true_axis.centered_count == 9
    assert v55._addition_decision(true_axis)[0] is True


def test_full_frozen_source_free_preflight_passes():
    report = v55.run_preflight_fixtures()
    assert report["passed"] is True
    assert report["precommit_token"] == v55.PRECOMMIT_TOKEN
    assert len(report["fixtures"]) == 10
    assert all(row["passed"] for row in report["fixtures"])


def test_optional_fixed_spacing_profile_is_symmetric_and_bounded():
    index, s_values, yz = _sharp_fixture()
    line = v55._fixture_line(s_values, yz, 0.0)
    parameters = index.scale.values(line.length)
    u, profile = v55._section_profile(
        index,
        0.5 * (line.start + line.end),
        line.direction,
        parameters,
        seed_point=np.asarray([0.0, yz[len(yz) // 2, 0], yz[len(yz) // 2, 1]]),
        output_spacing=1.0,
        maximum_output_points=65,
        allow_partial_symmetric_extent=True,
        minimum_section_segments_per_side=1,
    )

    assert 3 <= len(u) <= 65
    assert len(u) == len(profile)
    assert np.all(np.diff(u) > 0.0)
    assert np.allclose(u, -u[::-1], rtol=0.0, atol=1e-12)
    assert np.allclose(np.diff(u), 1.0, rtol=0.0, atol=1e-12)


def test_compose_prunes_and_adds_without_mutating_v52_geometry():
    offset = 0.40 * 9.6
    s_values, yz = v55._profile_points(((offset, 90.0),))
    vertices, triangles = v55._extruded_fixture_mesh(s_values, yz)
    index = v55.MeshSectionIndex.from_arrays(
        vertices, triangles, scale_override=(200.0, 0.5)
    )
    incumbent_line = v55._fixture_line(s_values, yz, 0.0)
    true_line = v55._fixture_line(s_values, yz, offset, 0)
    incumbent = {
        "bend_id": "V52-fixed",
        "line_start": incumbent_line.start.tolist(),
        "line_end": incumbent_line.end.tolist(),
        "source_kind": "fixed",
    }
    pool_candidate = {
        "candidate_id": "fixture",
        "span_endpoints": [true_line.start.tolist(), true_line.end.tolist()],
    }
    result, _v52, _pool = v55._compose_from_index(
        {"source_only": True, "research_only": True, "predictions": [incumbent]},
        {"candidates": [pool_candidate]},
        index,
    )
    assert result["candidate_count"] == 1
    assert result["predictions"][0]["line_start"] == pool_candidate["span_endpoints"][0]
    audit = result["mesh_section_turn_locus_reselection_audit"]
    assert audit["pruned_v52_count"] == 1
    assert audit["accepted_addition_count"] == 1
