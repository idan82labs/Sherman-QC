from __future__ import annotations

from scripts.probe_zero_shot_repeated_near_90_recipes import RECIPE_CONFIGS, score_probe_result


def test_score_probe_result_prefers_exact_supported_result() -> None:
    exact = score_probe_result(
        {
            "estimated_bend_count": 10,
            "estimated_bend_count_range": [10, 10],
            "abstained": False,
        },
        10,
    )
    abstained = score_probe_result(
        {
            "estimated_bend_count": None,
            "estimated_bend_count_range": [9, 11],
            "abstained": True,
        },
        10,
    )

    assert exact["exact_match"] == 1
    assert exact["range_contains_truth"] == 1
    assert exact["score"] > abstained["score"]


def test_score_probe_result_penalizes_truth_miss() -> None:
    missed = score_probe_result(
        {
            "estimated_bend_count": None,
            "estimated_bend_count_range": [3, 7],
            "abstained": True,
        },
        10,
    )

    assert missed["range_contains_truth"] == 0
    assert missed["score"] < 0


def test_probe_recipe_catalog_has_baseline_and_focusable_entries() -> None:
    assert "current_baseline" in RECIPE_CONFIGS
    assert "outlier_only_after_voxel" in RECIPE_CONFIGS
