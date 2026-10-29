from __future__ import annotations

import math

import numpy as np

from research.cibec_v1.agreement import AgreementThresholds, is_agreement, match_lines, pair_metrics
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match as legacy_is_match


def _event(event_id: str, start: list[float], end: list[float]) -> dict[str, object]:
    return {"event_id": event_id, "line_start_mm": start, "line_end_mm": end}


def test_cibec_uses_frozen_shorter_span_overlap_not_legacy_dual_overlap() -> None:
    short = np.asarray(((0, 0, 0), (10, 0, 0)), dtype=float)
    long = np.asarray(((-45, 0, 0), (55, 0, 0)), dtype=float)
    metrics = pair_metrics(short, long)
    assert metrics["shorter_span_overlap"] == 1.0
    assert metrics["second_span_overlap"] == 0.1
    assert is_agreement(metrics) is True
    legacy_metrics = {
        "axis_angle_deg": metrics["axis_angle_deg"],
        "transverse_offset_mm": metrics["transverse_offset_mm"],
        "gold_span_overlap": metrics["first_span_overlap"],
        "prediction_span_overlap": metrics["second_span_overlap"],
    }
    assert legacy_is_match(legacy_metrics, Thresholds()) is False


def test_endpoint_reversal_is_invariant() -> None:
    first = [[0, 0, 0], [20, 0, 0]]
    second = [[20, 0, 0], [0, 0, 0]]
    assert is_agreement(pair_metrics(first, second)) is True


def test_frozen_angle_offset_and_overlap_boundaries() -> None:
    base = [[0, 0, 0], [20, 0, 0]]
    angle = math.radians(5.01)
    angled = [[0, 0, 0], [20 * math.cos(angle), 20 * math.sin(angle), 0]]
    offset = [[0, 4.01, 0], [20, 4.01, 0]]
    short_overlap = [[8.2, 0, 0], [28.2, 0, 0]]
    assert is_agreement(pair_metrics(base, angled)) is False
    assert is_agreement(pair_metrics(base, offset)) is False
    assert abs(pair_metrics(base, short_overlap)["shorter_span_overlap"] - 0.59) < 1e-12
    assert is_agreement(pair_metrics(base, short_overlap)) is False


def test_matching_is_one_to_one_and_byte_stable_in_order() -> None:
    first = [_event("E0002", [0, 10, 0], [20, 10, 0]), _event("E0001", [0, 0, 0], [20, 0, 0])]
    second = [_event("B", [20, 10, 0], [0, 10, 0]), _event("A", [20, 0, 0], [0, 0, 0])]
    result = match_lines(first, second)
    assert result["true_positive"] == 2
    assert result["exact_localized_target_set"] is True
    assert [(row["first_event_id"], row["second_event_id"]) for row in result["matches"]] == [("E0001", "A"), ("E0002", "B")]


def test_nonfinite_and_short_lines_are_rejected() -> None:
    for line in ([[0, 0, 0], [7.99, 0, 0]], [[0, 0, 0], [float("nan"), 0, 0]]):
        try:
            pair_metrics(line, [[0, 0, 0], [10, 0, 0]])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid line was accepted")


def test_decimal_minimum_span_boundary_is_stable_without_admitting_grid_undercut() -> None:
    exact_decimal_boundary = [[-8.513, -34.94, -33.027], [-8.513, -26.94, -33.027]]
    assert float(np.linalg.norm(np.asarray(exact_decimal_boundary[1]) - np.asarray(exact_decimal_boundary[0]))) < 8.0
    assert is_agreement(pair_metrics(exact_decimal_boundary, list(reversed(exact_decimal_boundary))))
    one_micrometre_short = [[-8.513, -34.94, -33.027], [-8.513, -26.940001, -33.027]]
    try:
        pair_metrics(one_micrometre_short, exact_decimal_boundary)
    except ValueError:
        pass
    else:
        raise AssertionError("canonical-grid minimum-span undercut was accepted")
