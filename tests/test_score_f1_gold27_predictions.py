from __future__ import annotations

from scripts.score_f1_gold27_predictions import build_part_family_metrics


def _line(identifier: str, y: float = 0.0) -> dict:
    return {
        "bend_id": identifier,
        "line_start": [0.0, y, 0.0],
        "line_end": [100.0, y, 0.0],
    }


def test_family_metrics_are_equal_weighted_and_expose_worst_family() -> None:
    cases = {
        "C1": {"part_key": "F1"},
        "C2": {"part_key": "F1"},
        "C3": {"part_key": "F2"},
    }
    gold = {
        "C1": [_line("G1")],
        "C2": [_line("G2")],
        "C3": [_line("G3")],
    }
    predictions = {
        "C1": [_line("P1")],
        "C2": [_line("P2")],
        "C3": [],
    }

    result = build_part_family_metrics(predictions, gold, cases)

    assert result["family_count"] == 2
    assert result["family_macro"]["recall"] == 0.5
    assert result["family_macro"]["exact_count_accuracy"] == 0.5
    assert result["minimum_family_recall"] == 0.0
    assert result["families_below_0_80_recall"] == ["F2"]
    assert result["families_with_zero_recall"] == ["F2"]


def test_family_metrics_include_false_positive_precision_penalty() -> None:
    cases = {"C1": {"part_key": "F1"}}
    gold = {"C1": [_line("G1")]}
    predictions = {"C1": [_line("P1"), _line("P2", y=20.0)]}

    result = build_part_family_metrics(predictions, gold, cases)
    row = result["families"][0]

    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["precision"] == 0.5
    assert row["recall"] == 1.0
    assert row["exact_count_accuracy"] == 0.0


def test_family_metrics_keep_repeated_cases_in_one_family() -> None:
    cases = {
        "C1": {"part_key": "F1"},
        "C2": {"part_key": "F1"},
    }
    gold = {"C1": [_line("G1")], "C2": [_line("G2")]}
    predictions = {"C1": [_line("P1")], "C2": []}

    result = build_part_family_metrics(predictions, gold, cases)

    assert result["family_count"] == 1
    assert result["families"][0]["scan_count"] == 2
    assert result["families"][0]["recall"] == 0.5
    assert result["families"][0]["exact_count_accuracy"] == 0.5
