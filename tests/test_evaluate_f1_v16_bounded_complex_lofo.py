import numpy as np

from scripts.evaluate_f1_v16_bounded_complex_lofo import (
    ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES,
    ESTIMATOR_REGULARIZED_LOGISTIC,
    _aggregate,
    _case_metrics,
    _estimator,
    _family_class_balanced_weights,
    _gold_candidate_coverage,
)


def test_aggregate_is_micro_f1_and_counts_exact_cases():
    result = _aggregate(
        [
            {"true_positive": 3, "false_positive": 1, "false_negative": 0, "exact": False},
            {"true_positive": 2, "false_positive": 0, "false_negative": 0, "exact": True},
        ]
    )
    assert result == {
        "true_positive": 5,
        "false_positive": 1,
        "false_negative": 0,
        "precision": 0.833333,
        "recall": 1.0,
        "f1": 0.909091,
        "exact_case_count": 1,
        "case_count": 2,
    }


def test_case_metrics_adds_rates_to_match_counts():
    assert _case_metrics(
        {"true_positive": 2, "false_positive": 1, "false_negative": 2, "exact": False}
    ) == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 2,
        "precision": 0.666667,
        "recall": 0.5,
        "f1": 0.571429,
        "exact": False,
    }


def test_gold_candidate_coverage_separates_representation_from_selection():
    gold = [{"bend_id": "G1", "line_start": [0, 0, 0], "line_end": [10, 0, 0]}]
    rows = [
        {
            "candidate_id": "S1",
            "representative_source": "scan_mesh_adjacency",
            "source_kind": "mesh_edge_band",
            "span_endpoints": [[0, 0.5, 0], [10, 0.5, 0]],
        }
    ]
    coverage = _gold_candidate_coverage(gold, rows)
    assert coverage[0]["covered"] is True
    assert coverage[0]["matching_candidate_count"] == 1
    assert coverage[0]["matching_candidates"][0]["candidate_id"] == "S1"


def test_regularized_estimator_produces_probabilities():
    estimator = _estimator(ESTIMATOR_REGULARIZED_LOGISTIC)
    assert hasattr(estimator, "predict_proba")


def test_family_class_balancing_equalizes_family_and_class_mass():
    groups = np.asarray(["a", "a", "a", "b", "b", "b", "b"])
    target = np.asarray([0, 0, 1, 0, 1, 1, 1])
    weights = _family_class_balanced_weights(groups, target)
    for family in ("a", "b"):
        assert np.isclose(weights[groups == family].sum(), 0.5)
        assert np.isclose(weights[(groups == family) & (target == 0)].sum(), 0.25)
        assert np.isclose(weights[(groups == family) & (target == 1)].sum(), 0.25)


def test_family_balanced_extra_trees_supports_weighted_fit():
    estimator = _estimator(ESTIMATOR_FAMILY_CLASS_BALANCED_EXTRA_TREES)
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    target = np.asarray([0, 0, 1, 1])
    estimator.fit(features, target, sample_weight=np.ones(4))
    assert estimator.predict_proba(features).shape == (4, 2)
