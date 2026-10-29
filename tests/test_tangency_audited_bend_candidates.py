"""Additive engine-boundary checks on invented samples, no physical inputs."""
import json

import numpy as np
import pytest

from research.geometry.tangency_audited_bend_candidates import analyze_tangency_audited_bend_candidates
from tests.test_adaptive_bend_candidates import synthetic_bend
from tests.test_cap_supported_bend_candidates import analyze as baseline_analyze


def analyze(points, *, scale=1., **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6*scale, neighbor_counts=[8],
        max_neighborhood_radius=1.5*scale, max_normal_angle_bound=0.1,
        min_region_points=9, min_seed_points=3, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5*scale, max_axis_angle_bound=0.1,
        max_axial_gap=0.51*scale, max_radius_width=0.1*scale,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255, max_plane_evaluations=2048,
        graph_symmetry="union", max_center_evaluations=256,
    )
    policy.update(changes)
    return analyze_tangency_audited_bend_candidates(points, **policy)


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_bend_count"] is None
    assert result["policy"]["radius_analysis_recomputed"] is False
    assert result["policy"]["base_candidates_modified"] is False
    assert result["policy"]["two_radius_candidates_checked_as_single_face"] is False
    for item in result["assessments"]:
        source = result["base_analysis"]["candidates"][item["candidate_id"]]
        assert source["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"
        assert item["analysis"]["two_radius_or_multiple_surface_model_is_excluded"] is False
        assert item["analysis"]["original_joint_compatibility"] == "UNESTABLISHED"
        assert item["analysis"]["tangency_is_verified"] is False


@pytest.mark.parametrize("mixture,missing", [(False, False), (False, True), (True, False)])
def test_base_outputs_preserved_exactly_and_relation_diagnostic_is_additive(mixture, missing):
    first, second, arc = synthetic_bend(mixture=mixture)
    if missing:
        arc = np.concatenate([arc[:15], arc[20:]])
    points = np.concatenate([first, second, arc])
    result = analyze(points)
    scope(result)
    assert result["base_analysis"] == baseline_analyze(points)
    assert result["search_complete_within_declared_policy"] is True
    single_ids = [c["candidate_id"] for c in result["base_analysis"]["candidates"]
                  if c["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"]
    assert [item["candidate_id"] for item in result["assessments"]] == single_ids
    assert result["diagnostics"]["center_enclosure_evaluation_count"] == len(single_ids)
    if missing:
        # This specific 90-degree/extent-2 fixture has no conditional alias.
        # Missing a station alone is not evidence of a wrong tangent relation.
        assert result["diagnostics"]["same_face_excluded_count"] == 0
    if mixture:
        assert result["diagnostics"]["conditional_two_count"] > 0


def test_new_center_cap_preflights_without_a_successful_prefix():
    points = np.concatenate([np.concatenate(synthetic_bend()),
                             np.concatenate(synthetic_bend())+[15, 0, 0]])
    result = analyze(points, max_center_evaluations=1)
    scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["reasons"] == ["CENTER_ENCLOSURE_BUDGET_EXCEEDED"]
    assert result["assessments"] == []
    assert result["diagnostics"]["center_enclosure_evaluation_count"] == 0
    assert result["base_analysis"] == baseline_analyze(points)


def test_incomplete_base_search_cannot_be_reinterpreted_as_complete_audit():
    result = analyze(np.concatenate(synthetic_bend()), max_plane_evaluations=1)
    scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["reasons"] == ["BASE_CANDIDATE_SEARCH_INCOMPLETE"]
    assert result["assessments"] == []


def test_empty_cloud_is_not_a_complete_physical_empty_part():
    result = analyze([])
    scope(result)
    assert result["search_complete_within_declared_policy"]
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["NO_CONDITIONAL_SINGLE_RADIUS_CANDIDATES"]


def test_permutation_preserves_base_and_diagnostic_identity():
    first, second, arc = synthetic_bend()
    points = np.concatenate([first, second, arc[:15], arc[20:]])
    assert analyze(points) == analyze(points[np.random.default_rng(54).permutation(len(points))])


@pytest.mark.parametrize("value", [0, 257, True, 1.5, None])
def test_center_budget_is_explicit_and_bounded(value):
    with pytest.raises(ValueError):
        analyze([], max_center_evaluations=value)
