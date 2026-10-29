from __future__ import annotations

from backend.f1_overextended_mesh_fragment_replacement import replace_overextended_mesh_fragments


def _mesh(probability: float = 0.35) -> dict:
    return {
        "candidate_id": "MESH",
        "candidate_source": "scan_mesh_adjacency",
        "span_endpoints": [[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]],
        "span_mm": 150.0,
        "selection_probability": probability,
    }


def _fragment(transition: float = 0.97) -> dict:
    return {
        "candidate_id": "FRAGMENT",
        "representative_source": "cylindrical_standard",
        "span_endpoints": [[40.0, 2.0, 0.0], [110.0, 2.0, 0.0]],
        "span_mm": 70.0,
        "cluster_source_count": 2,
        "cluster_max_two_sided_transition_score": transition,
        "case_percentile_score": 0.98,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
    }


def test_replaces_overextended_low_probability_mesh_line() -> None:
    result = replace_overextended_mesh_fragments([_mesh()], [_fragment()])

    assert result["activated"] is True
    assert result["selected"][0]["candidate_id"] == "FRAGMENT"
    assert result["selected"][0]["candidate_source"] == "overextended_mesh_fragment_replacement"


def test_preserves_high_probability_mesh_line() -> None:
    assert replace_overextended_mesh_fragments([_mesh(0.8)], [_fragment()])["activated"] is False


def test_rejects_weak_or_offset_fragment() -> None:
    weak = _fragment(0.90)
    offset = _fragment()
    offset["span_endpoints"] = [[40.0, 8.0, 0.0], [110.0, 8.0, 0.0]]

    assert replace_overextended_mesh_fragments([_mesh()], [weak])["activated"] is False
    assert replace_overextended_mesh_fragments([_mesh()], [offset])["activated"] is False

