import itertools
import pytest
from backend.cad_bend_reference_bounds_v1 import metric_bounds


def test_small_unknown_completions_are_enclosed():
    checked = 0
    for t, m, f, u in itertools.product(range(4), repeat=4):
        bound = metric_bounds(matched=t, known_missed=m, complete_false_positive=f,
                              unscored_predictions=u, all_complete=False)
        for x in range(u + 1):
            for extra_misses in range(9):
                tp, fp, fn = t + x, f + u - x, m + extra_misses
                ratios = {"precision": (tp, tp + fp), "recall": (tp, tp + fn),
                          "f1": (2 * tp, 2 * tp + fp + fn)}
                for name, (numerator, denominator) in ratios.items():
                    if denominator and bound[name]["upper"] is not None:
                        actual = numerator / denominator
                        assert bound[name]["lower"] <= actual <= bound[name]["upper"]
                        checked += 1
    assert checked > 10_000


def test_nontrivial_upper_bound_with_unbounded_unknown_truth():
    bound = metric_bounds(matched=2, known_missed=5, complete_false_positive=3,
                          unscored_predictions=1, all_complete=False)
    assert bound["recall"] == {"lower": 0.0, "upper": 3 / 8}
    assert bound["f1"] == {"lower": 0.0, "upper": 6 / 14}
    assert bound["precision"] == {"lower": 2 / 6, "upper": 3 / 6}


def test_complete_is_exact_and_empty_ratios_are_null():
    bound = metric_bounds(matched=3, known_missed=2, complete_false_positive=1,
                          unscored_predictions=0, all_complete=True)
    assert bound["f1"] == {"lower": 6 / 9, "upper": 6 / 9}
    empty = metric_bounds(matched=0, known_missed=0, complete_false_positive=0,
                          unscored_predictions=0, all_complete=True)
    assert all(empty[key] == {"lower": None, "upper": None} for key in ("precision", "recall", "f1"))


def test_partial_empty_optimistic_upper_is_null():
    bound = metric_bounds(matched=0, known_missed=0, complete_false_positive=0,
                          unscored_predictions=0, all_complete=False)
    assert bound["precision"] == {"lower": None, "upper": None}
    assert bound["recall"] == {"lower": 0.0, "upper": None}


@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_invalid_counts(value):
    with pytest.raises(ValueError):
        metric_bounds(matched=value, known_missed=0, complete_false_positive=0,
                      unscored_predictions=0, all_complete=False)


def test_complete_cannot_hide_unscored_predictions():
    with pytest.raises(ValueError):
        metric_bounds(matched=0, known_missed=0, complete_false_positive=0,
                      unscored_predictions=1, all_complete=True)
