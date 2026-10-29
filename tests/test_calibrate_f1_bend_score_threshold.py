from scripts.calibrate_f1_bend_score_threshold import _micro


def test_micro_metrics_are_finite_and_exact_counted():
    result = _micro([
        {"true_positive": 2, "false_positive": 1, "false_negative": 1, "exact": False},
        {"true_positive": 1, "false_positive": 0, "false_negative": 0, "exact": True},
    ])
    assert result["true_positive"] == 3
    assert result["precision"] == 0.75
    assert result["recall"] == 0.75
    assert result["exact_scans"] == 1
