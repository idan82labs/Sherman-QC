from __future__ import annotations

from scripts.run_f1_gold27_unseen_profile import mesh_prediction, zero_predictions


def test_zero_predictions_uses_anonymous_render_geometry_only() -> None:
    result = {
        "part_id": None,
        "renderable_bend_objects": [
            {
                "bend_id": "S1",
                "detected_line": {
                    "display_start": [1.0, 2.0, 3.0],
                    "display_end": [4.0, 5.0, 6.0],
                },
                "metadata": {"strip_family": "discrete_fold_strip"},
            },
            {"bend_id": "S2", "detected_line": {}},
        ],
    }

    assert zero_predictions(result) == [
        {
            "bend_id": "S1",
            "span_endpoints": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "source_kind": "zero_shot_support_strip",
            "strip_family": "discrete_fold_strip",
        }
    ]


def test_mesh_prediction_abstains_when_external_pool_has_no_mesh_candidates() -> None:
    assert mesh_prediction({"candidates": []}, model=object()) == []
