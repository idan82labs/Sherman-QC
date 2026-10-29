from __future__ import annotations

import math

from backend.f1_mesh_adjacency_veto import (
    MESH_NORMAL_COHERENCE_CAP_GRID,
    retains_mesh_adjacency_candidate,
)


def _mesh_candidate(**overrides):
    candidate = {
        "candidate_source": "scan_mesh_adjacency",
        "min_surface_gap": 3.0,
        "two_sided_normal_coherence": 0.85,
    }
    candidate.update(overrides)
    return candidate


def test_mesh_veto_uses_inclusive_tuned_caps() -> None:
    assert MESH_NORMAL_COHERENCE_CAP_GRID == (0.80, 0.85, 0.90, 0.95, 1.0)
    assert retains_mesh_adjacency_candidate(
        _mesh_candidate(min_surface_gap=6.0, two_sided_normal_coherence=0.90),
        maximum_surface_gap_mm=6.0,
        maximum_two_sided_normal_coherence=0.90,
    )
    assert not retains_mesh_adjacency_candidate(
        _mesh_candidate(two_sided_normal_coherence=0.90001),
        maximum_surface_gap_mm=6.0,
        maximum_two_sided_normal_coherence=0.90,
    )


def test_mesh_veto_fails_closed_for_missing_or_nonfinite_evidence() -> None:
    for candidate in (
        _mesh_candidate(min_surface_gap=None),
        _mesh_candidate(two_sided_normal_coherence="bad"),
        _mesh_candidate(two_sided_normal_coherence=math.nan),
        _mesh_candidate(min_surface_gap=-math.inf),
        _mesh_candidate(two_sided_normal_coherence=math.inf),
    ):
        assert not retains_mesh_adjacency_candidate(
            candidate,
            maximum_surface_gap_mm=6.0,
            maximum_two_sided_normal_coherence=0.90,
        )


def test_mesh_veto_does_not_filter_other_scan_only_sources() -> None:
    assert retains_mesh_adjacency_candidate(
        {"candidate_source": "cylindrical_standard"},
        maximum_surface_gap_mm=0.0,
        maximum_two_sided_normal_coherence=0.0,
    )
