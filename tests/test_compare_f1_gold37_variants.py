from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.compare_f1_gold37_variants import compare_scores


def _score() -> dict:
    return {
        "corpus_freeze_id": "freeze",
        "metrics": {"scan_count": 2},
        "part_family_metrics": {
            "family_count": 2,
            "families": [
                {
                    "part_family": "F1",
                    "case_tokens": ["C1"],
                    "precision": 0.5,
                    "recall": 0.5,
                    "line_f1": 0.5,
                    "exact_count_accuracy": 0.0,
                },
                {
                    "part_family": "F2",
                    "case_tokens": ["C2"],
                    "precision": 0.5,
                    "recall": 0.5,
                    "line_f1": 0.5,
                    "exact_count_accuracy": 0.0,
                },
            ],
        },
    }


def test_self_comparison_is_exactly_zero_and_deterministic() -> None:
    score = _score()
    first = compare_scores(score, score, resamples=100)
    second = compare_scores(deepcopy(score), deepcopy(score), resamples=100)

    assert first == second
    for metric in first["metric_deltas"].values():
        assert metric["observed_family_macro_delta"] == 0.0
        assert metric["paired_family_bootstrap_95_ci"] == [0.0, 0.0]
        assert metric["families_tied"] == 2


def test_paired_family_comparison_counts_broad_and_mixed_gains() -> None:
    baseline = _score()
    variant = deepcopy(baseline)
    first, second = variant["part_family_metrics"]["families"]
    first["recall"] = 0.75
    second["recall"] = 0.25
    first["line_f1"] = 0.75
    second["line_f1"] = 0.50

    report = compare_scores(baseline, variant, resamples=500)

    recall = report["metric_deltas"]["recall"]
    line_f1 = report["metric_deltas"]["line_f1"]
    assert recall["observed_family_macro_delta"] == 0.0
    assert recall["families_improved"] == 1
    assert recall["families_worse"] == 1
    assert line_f1["observed_family_macro_delta"] == 0.125
    assert line_f1["families_improved"] == 1
    assert line_f1["families_tied"] == 1


def test_comparison_rejects_freeze_family_or_case_drift() -> None:
    baseline = _score()
    variant = deepcopy(baseline)
    variant["corpus_freeze_id"] = "other"
    with pytest.raises(ValueError, match="freeze IDs"):
        compare_scores(baseline, variant, resamples=10)

    variant = deepcopy(baseline)
    variant["part_family_metrics"]["families"][1]["part_family"] = "F3"
    with pytest.raises(ValueError, match="family sets"):
        compare_scores(baseline, variant, resamples=10)

    variant = deepcopy(baseline)
    variant["part_family_metrics"]["families"][1]["case_tokens"] = ["C3"]
    with pytest.raises(ValueError, match="case-token sets"):
        compare_scores(baseline, variant, resamples=10)
