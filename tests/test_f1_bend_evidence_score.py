from backend.f1_bend_evidence_score import score_bend_evidence


def test_dense_two_sided_multisource_line_rates_above_weak_line():
    bend = {"bend_id": "B1", "span_endpoints": [[0, 0, 0], [100, 0, 0]], "confidence": 0.8}
    strong = {
        "bend_id": "LO1", "span_endpoints": [[0, 0, 0], [100, 0, 0]],
        "cluster_max_finite_support_longest_run_fraction": 1.0,
        "cluster_max_endpoint_support_fraction": 1.0,
        "cluster_max_two_sided_transition_score": 0.9,
        "cluster_max_two_sided_normal_coherence": 0.95,
        "cluster_source_count": 3,
        "cluster_mean_median_line_distance_mm": 1.0,
    }
    weak = {
        "bend_id": "LO2", "span_endpoints": [[0, 0, 0], [100, 0, 0]],
        "cluster_max_finite_support_longest_run_fraction": 0.2,
        "cluster_max_endpoint_support_fraction": 0.0,
        "cluster_max_two_sided_transition_score": 0.05,
        "cluster_max_two_sided_normal_coherence": 0.4,
        "cluster_source_count": 1,
        "cluster_mean_median_line_distance_mm": 12.0,
    }
    assert score_bend_evidence(bend, [strong])["evidence_score"] > score_bend_evidence(bend, [weak])["evidence_score"]


def test_score_is_identity_and_truth_free():
    bend = {"bend_id": "B1", "span_endpoints": [[0, 0, 0], [10, 0, 0]]}
    result = score_bend_evidence(bend, [])
    assert result["uses_part_identifier"] is False
    assert result["uses_expected_count"] is False
    assert result["uses_manual_truth"] is False
    assert result["interpretation"] == "intrinsic_geometry_evidence_not_probability"


def test_parallel_lane_margin_reports_stronger_competitor_without_rejecting():
    selected = {
        "bend_id": "LO1", "span_endpoints": [[0, 0, 0], [100, 0, 0]],
        "cluster_max_finite_support_longest_run_fraction": 0.5,
        "cluster_max_two_sided_transition_score": 0.2,
        "cluster_max_two_sided_normal_coherence": 0.7,
        "cluster_source_count": 1,
        "cluster_mean_median_line_distance_mm": 2.0,
    }
    competitor = {
        "bend_id": "LO2", "span_endpoints": [[0, 5, 0], [100, 5, 0]],
        "cluster_max_finite_support_longest_run_fraction": 1.0,
        "cluster_max_endpoint_support_fraction": 1.0,
        "cluster_max_two_sided_transition_score": 0.9,
        "cluster_max_two_sided_normal_coherence": 0.95,
        "cluster_source_count": 3,
        "cluster_mean_median_line_distance_mm": 1.0,
    }
    result = score_bend_evidence({"bend_id": "B", "span_endpoints": [[0, 0, 0], [100, 0, 0]]}, [selected, competitor])
    assert result["parallel_lane_ownership"]["state"] == "competitor_stronger"
    assert result["parallel_lane_ownership"]["margin"] < 0.0
    assert result["review_queue"] == "semantic_ownership_review"
