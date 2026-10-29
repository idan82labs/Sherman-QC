from __future__ import annotations

from backend.f1_v43_fragment_stitch_v44 import compose_v44


def _prediction(candidate_id, x0, x1, y=0.0):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": [x0, y, 0.0],
        "line_end": [x1, y, 0.0],
        "selection_route": "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair",
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": "cylindrical_standard",
    }


def test_stitches_collinear_fragments_after_v43():
    source = {
        "route": "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair",
        "predictions": [
            _prediction("a", 0.0, 40.0),
            _prediction("b", 45.0, 85.0),
            _prediction("c", 90.0, 130.0),
        ],
    }
    result = compose_v44(source)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["source_kind"] == "collinear_fragment_stitch"
    assert len(result["collinear_fragment_stitch_changes"]) == 1


def test_non_complex_route_is_exact_passthrough():
    prediction = _prediction("a", 0.0, 40.0)
    source = {"route": "full_resolution_mesh_v11", "predictions": [prediction]}
    result = compose_v44(source)
    assert result["predictions"] == [prediction]
    assert result["collinear_fragment_stitch_changes"] == []
