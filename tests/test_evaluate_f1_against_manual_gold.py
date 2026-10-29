import numpy as np

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics


def _line(bend_id, start, end):
    return {"bend_id": bend_id, "line_start": list(start), "line_end": list(end)}


def test_pair_metrics_treats_reversed_axis_as_same_line():
    metrics = pair_metrics(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.asarray([[10.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert metrics["axis_angle_deg"] == 0.0
    assert metrics["transverse_offset_mm"] == 1.0
    assert metrics["gold_span_overlap"] == 1.0


def test_matching_is_one_to_one_for_duplicate_predictions():
    result = match_lines(
        [_line("G1", (0, 0, 0), (10, 0, 0))],
        [
            _line("P1", (0, 0, 0), (10, 0, 0)),
            _line("P2", (0, 0.5, 0), (10, 0.5, 0)),
        ],
        Thresholds(),
    )
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0
    assert not result["exact"]


def test_parallel_line_beyond_four_mm_is_rejected():
    result = match_lines(
        [_line("G1", (0, 0, 0), (10, 0, 0))],
        [_line("P1", (0, 4.8, 0), (10, 4.8, 0))],
        Thresholds(transverse_offset_mm=4.0),
    )
    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1


def test_short_fragment_below_gold_overlap_is_rejected():
    result = match_lines(
        [_line("G1", (0, 0, 0), (100, 0, 0))],
        [_line("P1", (0, 0, 0), (20, 0, 0))],
        Thresholds(gold_span_overlap=0.60, prediction_span_overlap=0.60),
    )
    assert result["true_positive"] == 0
