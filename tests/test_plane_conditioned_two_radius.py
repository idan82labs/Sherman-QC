"""All-input synthetic checks for two-radius witnesses, not physical metrology."""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.geometry import plane_conditioned_two_radius as module


def fixture(radii=(2, 2.2)):
    first = np.array([(x, y, 2) for x in (-2, -1, 0) for y in (-1, 0, 1)], float)
    second = np.array([(2, y, z) for z in (-2, -1, 0) for y in (-1, 0, 1)], float)
    mixed = np.array([(r*math.sin(t), y, r*math.cos(t)) for r in radii
                      for t in np.linspace(0.15, 1.4, 9) for y in (-1, 0, 1)], float)
    return first, second, mixed


def analyze(first=None, second=None, mixed=None, **changes):
    default = fixture()
    arguments = dict(unit="mm", position_error_bound=1e-7, max_axis_angle_bound=0.1,
                     max_axial_gap=1.1, max_radius_width=0.1,
                     min_band_points=6, max_split_evaluations=255)
    arguments.update(changes)
    return module.analyze_plane_conditioned_two_radius(
        default[0] if first is None else first, default[1] if second is None else second,
        default[2] if mixed is None else mixed, **arguments,
    )


def check_scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_face_count"] is result["thickness"] is None
    assert result["partition_is_unique"] is False
    assert result["partition_uniqueness"] == "UNESTABLISHED"
    assert "radius_interval" not in result
    for model in result["models"]:
        assert model["scope"] == "TWO_CONCENTRIC_CIRCLE_WITNESS_IN_ROTATED_ERROR_RELAXATION_ONLY"
        assert model["all_points_accounted_for"] is True
        assert model["physical_face_identity_is_verified"] is False
        assert model["partition_is_verified"] is False
        assert len(model["bands"]) == 2
        members = [set(band["point_indices"]) for band in model["bands"]]
        assert members[0].isdisjoint(members[1])
        assert members[0] | members[1] == set(range(len(result["mixed_positions"])))
        for band in model["bands"]:
            assert band["membership_is_verified"] is False
            assert band["radius_interval_scope"] == (
                "CONDITIONAL_ON_PROPOSED_BAND_MEMBERSHIP_AND_AXIS_ROTATION_RELAXATION")
            assert band["position_error_bound"] == result["global_band_input_error_bound"]
            assert band["position_error_bound"] >= result["uncertainty_propagation"]["radius_effective_error_bound"]


def test_two_radius_sample_witness_preserves_every_position_without_supplied_band_labels():
    result = analyze()
    check_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["models"]) == 1
    model = result["models"][0]
    assert_allclose(model["witness_radii"], [2, 2.2], rtol=0, atol=1e-12)
    points = np.asarray(result["mixed_positions"])
    q = ((points-model["projection_pivot"])/model["projection_scale"]) @ np.asarray(model["projection_basis"])
    distances = np.linalg.norm(q-model["normalized_projection_center"], axis=1)*model["projection_scale"]
    for band, radius in zip(model["bands"], (2, 2.2)):
        indices = band["point_indices"]
        assert len(indices) == 27
        assert_allclose(np.hypot(points[indices, 0], points[indices, 2]), radius, rtol=0, atol=1e-15)
        assert np.max(np.abs(distances[indices]-band["witness_radius"])) <= result["global_band_input_error_bound"]
        assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
    assert result["search_diagnostics"]["algebraic_rank"] == 8
    assert result["single_radius_alternative"]["compatible_with_error_bound"] is False


def test_permutations_and_exact_repeats_preserve_witness_not_extra_evidence():
    first, second, mixed = fixture()
    expected = analyze(first, second, mixed)
    actual = analyze(first[::-1], second[::-1], np.repeat(mixed[::-1], 2, axis=0))
    check_scope(actual)
    assert actual["models"] == expected["models"]
    assert actual["mixed_positions"] == expected["mixed_positions"]
    assert actual["input_count"] == 126


@pytest.mark.parametrize("seed", [8, 72, 1109])
def test_proper_rigid_transform_retains_known_radii(seed):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    shift = rng.uniform(-10, 10, size=3)
    result = analyze(*(points @ rotation.T+shift for points in fixture()))
    check_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    model = result["models"][0]
    assert_allclose(model["witness_radii"], [2, 2.2], rtol=0, atol=1e-10)
    for band, radius in zip(model["bands"], (2, 2.2)):
        assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]


def test_units_scale_all_length_policies_and_both_conditional_intervals():
    mm = analyze()
    meters = analyze(*(p*0.001 for p in fixture()), unit="m", position_error_bound=1e-10,
                     max_axial_gap=0.0011, max_radius_width=0.0001)
    check_scope(meters)
    assert meters["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert_allclose(np.asarray(meters["models"][0]["witness_radii"])*1000,
                    mm["models"][0]["witness_radii"], rtol=0, atol=1e-10)
    for first, second in zip(mm["models"][0]["bands"], meters["models"][0]["bands"]):
        assert_allclose(np.asarray(second["radius_interval"])*1000, first["radius_interval"], atol=1e-10)


def test_single_radius_alternative_does_not_get_relabelled_as_two_faces():
    result = analyze(*fixture((2,)))
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["SINGLE_RADIUS_ALTERNATIVE_COMPATIBLE"]
    assert result["models"] == []


def test_three_well_separated_radii_do_not_discard_one_face_as_outliers():
    result = analyze(*fixture((2, 2.2, 2.5)))
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["models"] == []
    assert len(result["mixed_positions"]) == 81


def test_one_unexplained_contaminant_cannot_be_trimmed_from_model():
    first, second, mixed = fixture()
    mixed = np.concatenate((mixed, [[2.8/math.sqrt(2), 0, 2.8/math.sqrt(2)]]))
    result = analyze(first, second, mixed)
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["models"] == []
    assert len(result["mixed_positions"]) == 55


def test_low_split_budget_stops_before_center_or_band_selection(monkeypatch):
    monkeypatch.setattr(module, "_algebraic_center", lambda *args: pytest.fail("cap checked too late"))
    result = analyze(max_split_evaluations=52)
    check_scope(result)
    assert result["reasons"] == ["RADIAL_SPLIT_BUDGET_EXCEEDED"]
    assert result["search_complete_within_declared_policy"] is False
    assert result["search_diagnostics"]["band_evaluation_count"] == 0


def test_shared_raw_cap_precedes_conversion(monkeypatch):
    monkeypatch.setattr(np, "asarray", lambda *args, **kwargs: pytest.fail("oversize conversion"))
    with pytest.raises(ValueError, match="256 total"):
        analyze([[0, 0, 0]]*85, [[0, 0, 0]]*86, [[0, 0, 0]]*86)


@pytest.mark.parametrize("field,value", [
    ("min_band_points", 2), ("min_band_points", 129), ("min_band_points", True),
    ("min_band_points", 6.0), ("max_split_evaluations", 0),
    ("max_split_evaluations", 256), ("max_split_evaluations", False),
    ("max_split_evaluations", math.inf), ("max_split_evaluations", "255"),
])
def test_invalid_explicit_policies_raise(field, value):
    with pytest.raises(ValueError):
        analyze(**{field: value})


def test_unavailable_algebraic_analysis_abstains_incomplete(monkeypatch):
    monkeypatch.setattr(module, "_algebraic_center", lambda *args: (
        None, "ALGEBRAIC_CENTER_NUMERICALLY_UNAVAILABLE", None))
    result = analyze()
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False


def test_unavailable_second_band_clears_the_entire_two_radius_model(monkeypatch):
    original = module.analyze_axis_conditioned_radius
    calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            return {"reasons": ["NUMERICAL_RANGE_UNSUPPORTED"]}
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "analyze_axis_conditioned_radius", failure)
    result = analyze()
    check_scope(result)
    assert result["models"] == []
    assert result["reasons"] == ["BAND_ANALYSIS_NUMERICAL_RANGE_UNSUPPORTED"]
    assert result["search_complete_within_declared_policy"] is False


def test_failed_geometric_checks_do_not_claim_no_feasible_two_radius_model(monkeypatch):
    # A deliberately bad center proposal must not be rescued by trimming points.
    monkeypatch.setattr(module, "_algebraic_center", lambda *args: (np.array([0.0, 0.0]), None, 8))
    result = analyze()
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["NO_TWO_RADIUS_WITNESS_WITHIN_DECLARED_SEARCH"]
    assert "NO_COMPLETE_SEARCH_OVER_FEASIBLE_CENTERS_OR_PARTITIONS" in result["limitations"]


def test_unclassified_legacy_proposal_failure_does_not_bypass_geometric_checks(monkeypatch):
    # The unchanged legacy circle fitter collapses an initial solver exception
    # to a missing proposal. This is not an explicit numerical-unavailability
    # signal. A later verified two-band witness can still be retained, but its
    # declared-policy completion must not imply complete solver-health reporting.
    original = np.linalg.lstsq
    failures = []

    def fail_initial_circle(matrix, rhs, **kwargs):
        if matrix.shape[1] == 3 and not failures:
            failures.append(True)
            raise np.linalg.LinAlgError("invented initial circle solver failure")
        return original(matrix, rhs, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", fail_initial_circle)
    result = analyze()
    check_scope(result)
    assert failures == [True]
    assert result["single_radius_alternative"] is None
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert result["search_complete_within_declared_policy"] is True
    assert "LEGACY_CIRCLE_PROPOSAL_DOES_NOT_CLASSIFY_ALL_SOLVER_FAILURES" in result["limitations"]
    model = result["models"][0]
    points = np.asarray(result["mixed_positions"])
    q = ((points-model["projection_pivot"])/model["projection_scale"]) @ np.asarray(model["projection_basis"])
    radial = np.linalg.norm(q-model["normalized_projection_center"], axis=1)*model["projection_scale"]
    for band, truth in zip(model["bands"], (2, 2.2)):
        assert np.max(np.abs(radial[band["point_indices"]]-band["witness_radius"])) <= result["global_band_input_error_bound"]
        assert band["radius_interval"][0] <= truth <= band["radius_interval"][1]
