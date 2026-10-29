from __future__ import annotations

from scripts.run_gold11_v11_single_scan_luna_max_v2 import explicit_metrics


def test_explicit_metrics_are_derived_only_from_match_counts():
    result = explicit_metrics({
        "true_positive": 4,
        "false_positive": 0,
        "false_negative": 0,
        "gold_count": 4,
        "prediction_count": 4,
    })
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["formula"].startswith("TP/")


def test_explicit_metrics_preserve_abstention_as_null_precision():
    result = explicit_metrics({
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 4,
        "gold_count": 4,
        "prediction_count": 0,
    })
    assert result["precision"] is None
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
