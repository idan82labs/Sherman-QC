from __future__ import annotations

from scripts.run_isolated_f1_blind_ply_benchmark import _typed_release_metrics


def test_typed_release_metrics_ignore_unscored_source_only_rows():
    metrics = _typed_release_metrics([
        {
            "expected_bend_count_scored": None,
            "algorithm": {"accepted_exact_bend_count": 2},
        }
    ])
    assert metrics["scans_processed"] == 0
    assert metrics["exact_count_correct"] == 0
    assert metrics["exact_count_accuracy"] is None


def test_typed_release_metrics_still_score_labeled_rows():
    metrics = _typed_release_metrics([
        {
            "expected_bend_count_scored": 2,
            "algorithm": {
                "accepted_exact_bend_count": 2,
                "accepted_bend_count_range": [2, 2],
                "raw_f1_map_bend_count": 2,
                "render_admissible_exact_bend_count": 2,
                "visual_acceptance_status": "accepted",
            },
        }
    ])
    assert metrics["scans_processed"] == 1
    assert metrics["exact_count_accuracy"] == 1.0
    assert metrics["visual_exact_accuracy"] == 1.0
