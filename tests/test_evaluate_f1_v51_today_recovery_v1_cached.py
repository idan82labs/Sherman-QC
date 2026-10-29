from __future__ import annotations

from scripts.evaluate_f1_v51_today_recovery_v1_cached import _delta, _metrics


def test_metrics_and_delta_preserve_count_accounting() -> None:
    baseline_rows = [
        {
            "true_positive": 1,
            "false_positive": 2,
            "false_negative": 3,
            "exact": False,
        }
    ]
    successor_rows = [
        {
            "true_positive": 2,
            "false_positive": 3,
            "false_negative": 2,
            "exact": False,
        }
    ]
    baseline = _metrics(baseline_rows)
    successor = _metrics(successor_rows)
    assert baseline["gold_bend_count"] == successor["gold_bend_count"] == 4
    assert baseline["prediction_count"] == 3
    assert successor["prediction_count"] == 5
    delta = _delta(successor, baseline)
    assert delta["true_positive"] == 1
    assert delta["false_negative"] == -1
    assert delta["prediction_count"] == 2


def test_metrics_zero_prediction_case_is_finite() -> None:
    metrics = _metrics(
        [
            {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 2,
                "exact": False,
            }
        ]
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
