"""Optimistic whole-corpus metric bounds from frozen partial-reference counts.

Does not match geometry, change labels, or consume candidate artifacts. Unknown
truth may add arbitrary false negatives. Every unscored prediction is optimistically
allowed to match a distinct unknown bend, even if geometry would prevent this.
"""


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metric_bounds(*, matched, known_missed, complete_false_positive,
                  unscored_predictions, all_complete):
    """Return conservative intervals conditional on exact reviewed ownership."""
    values = (matched, known_missed, complete_false_positive, unscored_predictions)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("counts must be nonnegative Python integers")
    if type(all_complete) is not bool:
        raise ValueError("all_complete must be boolean")
    if all_complete and unscored_predictions:
        raise ValueError("complete reference cannot have unscored predictions")
    t, m, f, u = values
    n = t + f + u
    if all_complete:
        precision = _ratio(t, n)
        recall = _ratio(t, t + m)
        f1 = _ratio(2 * t, 2 * t + f + m)
        return {"precision": {"lower": precision, "upper": precision},
                "recall": {"lower": recall, "upper": recall},
                "f1": {"lower": f1, "upper": f1}, "all_complete": True,
                "prediction_denominator": n, "known_positive_denominator": t + m}
    # A 0/0 optimistic configuration is undefined, but unknown misses can give
    # a defined zero metric. Never fabricate a ratio for that empty configuration.
    return {"precision": {"lower": _ratio(t, n), "upper": _ratio(t + u, n)},
            "recall": {"lower": 0.0, "upper": _ratio(t + u, t + u + m)},
            "f1": {"lower": 0.0, "upper": _ratio(2 * (t + u), 2 * (t + u) + f + m)},
            "all_complete": False, "prediction_denominator": n,
            "known_positive_denominator": t + m,
            "semantics": "optimistic outer bounds; lower zero is an infimum; undefined upper ratios remain null"}
