from __future__ import annotations

from backend.f1_mesh_surface_topology_filter import (
    filter_mesh_surface_topology,
    route_mesh_topology_profile,
)


def _candidate(
    candidate_id: str,
    *,
    start=(0.0, 0.0, 0.0),
    end=(100.0, 0.0, 0.0),
    surfaces=(1, 2),
    areas=(10_000.0, 2_000.0),
    gap=2.0,
    angle=90.0,
    transition=0.9,
):
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "line_start": list(start),
        "line_end": list(end),
        "surface1_id": surfaces[0],
        "surface2_id": surfaces[1],
        "surface1_area": areas[0],
        "surface2_area": areas[1],
        "min_surface_gap": gap,
        "bend_angle": angle,
        "two_sided_transition_score": transition,
    }


def _ids(result):
    return [item["candidate_id"] for item in result["selected"]]


def test_direct_transition_requires_post_completion_gap_budget():
    candidates = [
        _candidate("keep"),
        _candidate("gap-veto", start=(0, 20, 0), end=(100, 20, 0), gap=6.01),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["keep"]


def test_area_span_support_prunes_weak_outline():
    candidates = [
        _candidate("keep", surfaces=(1, 2)),
        _candidate(
            "outline",
            start=(0, 20, 0),
            end=(100, 20, 0),
            surfaces=(2, 3),
            areas=(2_000, 400),
            transition=0.95,
        ),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["keep"]


def test_dominant_face_retains_large_low_ratio_direct_transition():
    candidates = [
        _candidate("dominant", end=(200, 0, 0), surfaces=(1, 2), areas=(20_000, 1_200)),
        _candidate("other", start=(0, 20, 0), end=(200, 20, 0), surfaces=(3, 4), areas=(2_000, 1_200)),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["dominant"]


def test_two_nonparallel_short_supported_folds_complete_sparse_base():
    candidates = [
        _candidate("base"),
        _candidate(
            "short-a",
            start=(0, 20, 0),
            end=(25, 20, 0),
            surfaces=(3, 4),
            areas=(300, 250),
            gap=0.5,
            transition=0.2,
        ),
        _candidate(
            "short-b",
            start=(50, 0, 0),
            end=(50, 25, 0),
            surfaces=(5, 6),
            areas=(300, 250),
            gap=0.5,
            transition=0.2,
        ),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["base", "short-a", "short-b"]


def test_parallel_short_fragments_do_not_complete():
    candidates = [
        _candidate("base"),
        _candidate("short-a", end=(25, 0, 0), surfaces=(3, 4), areas=(300, 250), gap=0.5),
        _candidate(
            "short-b",
            start=(0, 20, 0),
            end=(25, 20, 0),
            surfaces=(5, 6),
            areas=(300, 250),
            gap=0.5,
        ),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["base"]


def test_paired_high_angle_transitions_complete_sparse_graph_and_dedupe_alias():
    candidates = [
        _candidate("direct", surfaces=(1, 2)),
        _candidate(
            "alias",
            start=(0, 3, 0),
            end=(100, 3, 0),
            surfaces=(2, 3),
            areas=(2_000, 600),
            angle=138,
            transition=0.85,
            gap=0.2,
        ),
        _candidate(
            "missing-direct",
            start=(0, 50, 0),
            end=(100, 50, 0),
            surfaces=(2, 4),
            areas=(2_000, 600),
            angle=138,
            transition=0.85,
            gap=0.2,
        ),
    ]
    assert _ids(filter_mesh_surface_topology(candidates)) == ["direct", "missing-direct"]


def test_filter_is_deterministic_and_ignores_non_mesh_candidates():
    candidates = [_candidate("a"), {**_candidate("ignored"), "candidate_source": "other"}]
    first = filter_mesh_surface_topology(candidates)
    second = filter_mesh_surface_topology(candidates)
    assert first == second
    assert _ids(first) == ["a"]


def test_closed_six_surface_graph_routes_to_scan_supported_formed_completion():
    candidates = [
        _candidate(
            f"c{index}",
            start=(0, index * 20, 0),
            end=(100, index * 20, 0),
            surfaces=(1, index + 2),
        )
        for index in range(6)
    ]
    payload = {
        "candidates": candidates,
        "generator_diagnostics": {"scan_mesh_adjacency": {"major_surface_count": 6}},
    }
    result = filter_mesh_surface_topology(candidates)
    route = route_mesh_topology_profile(payload, result)
    assert route["route_id"] == "mesh_closed6_formed_prismatic_completion"
    assert route["enable_prismatic_plane_lattice_completion"] is True
    assert route["enable_interior_formed_transition_completion"] is True


def test_closed_six_route_requires_independent_surface_count_signature():
    candidates = [
        _candidate(
            f"c{index}",
            start=(0, index * 20, 0),
            end=(100, index * 20, 0),
            surfaces=(1, index + 2),
        )
        for index in range(6)
    ]
    payload = {
        "candidates": candidates,
        "generator_diagnostics": {"scan_mesh_adjacency": {"major_surface_count": 5}},
    }
    route = route_mesh_topology_profile(payload, filter_mesh_surface_topology(candidates))
    assert route["route_id"] == "mesh_surface_topology_filter"
    assert route["enable_prismatic_plane_lattice_completion"] is False
