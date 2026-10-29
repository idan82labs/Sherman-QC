"""Focused exact-math controls; one frozen full-matrix run, no native imports."""
import importlib.util
import json
from fractions import Fraction as F
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "cad_root_axis_math_checks.py"


@pytest.fixture(scope="module")
def math_api():
    spec = importlib.util.spec_from_file_location("isolated_cad_root_axis_math", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("signs", [(-1, -1), (-1, 1), (1, -1), (1, 1)])
def test_right_angle_offset_is_sqrt2_radius_for_every_sign_alternative(math_api, signs):
    first, second = math_api.plane_normals(F(1))
    center = math_api.solve_axis_displacement(first, second, F(5), signs)
    assert math_api.dot(center, center) == 50
    assert math_api.dot(first, center) == signs[0] * 5
    assert math_api.dot(second, center) == signs[1] * 5
    if signs == (1, 1):
        assert center == (F(-1), F(7), F(0))


@pytest.mark.parametrize("half_angle,large_signs", [(F(1, 1_000_000), (1, -1)), (F(1_000_000), (1, 1))])
def test_near_parallel_condition_and_offset_amplification_are_exact(math_api, half_angle, large_signs):
    first, second = math_api.plane_normals(half_angle)
    assert math_api.normal_system_condition_squared(first, second) == 10**12
    center = math_api.solve_axis_displacement(first, second, F(1), large_signs)
    assert math_api.dot(center, center) == 10**12 + 1


def test_parallel_planes_are_not_rescued_by_a_tolerance(math_api):
    first = (F(1), F(0), F(0))
    with pytest.raises(ValueError, match="parallel"):
        math_api.solve_axis_displacement(first, first, F(1), (1, 1))
    with pytest.raises(ValueError, match="nonparallel"):
        math_api.normal_system_condition_squared(first, first)


def test_two_finite_intervals_remain_separate_despite_same_infinite_geometry(math_api):
    first, second = math_api.plane_normals(F(1))
    center = math_api.solve_axis_displacement(first, second, F(1), (1, 1))
    generator = math_api.sub(center, first)
    for interval in math_api.FINITE_INTERVALS:
        low, high = interval
        assert low < high
        for parameter in (low, high, high + 1):
            point = math_api.add(generator, (F(0), F(0), parameter))
            assert math_api.dot(first, point) == 0
            radius_vector = math_api.sub(point, math_api.add(center, (F(0), F(0), parameter)))
            assert math_api.dot(radius_vector, radius_vector) == 1
    assert math_api.FINITE_INTERVALS[0][1] < math_api.FINITE_INTERVALS[1][0]


def test_frozen_matrix_identity_and_budget_are_predeclared(math_api):
    assert math_api.EXPECTED_CASES == 5120 and math_api.MAX_CASES == 8192
    assert math_api.MAX_NUMERATOR_OR_DENOMINATOR_BITS == 512
    assert len(math_api.HALF_ANGLES) == 10 and len(math_api.SIGNS) == 4
    assert len(math_api.RADII) == len(math_api.FRAMES) == len(math_api.SCALES) == 4
    assert len(math_api.FINITE_INTERVALS) == 2


def test_one_complete_exact_matrix_and_non_authoritative_json_aggregate(math_api):
    result = math_api.run_checks()
    assert result["status"] == "EXACT_IDENTITIES_PASSED"
    assert result["case_count"] == result["expected_case_count"] == 5120
    assert result["strata"] == {"near_parallel": 512, "near_antiparallel": 512,
                                "right_angle": 512, "other_oblique": 3584}
    assert set(result["signed_offset_cases"].values()) == {1280}
    assert set(result["frame_cases"].values()) == {1280}
    assert set(result["scale_cases"].values()) == {1280}
    assert set(result["declared_finite_interval_cases"].values()) == {2560}
    assert result["exact_check_counts"]["closed_form_offset_squared"] == 5120
    assert result["exact_check_counts"]["right_angle_sqrt2_offset_squared"] == 512
    assert result["exact_check_counts"]["contact_on_analytic_plane"] == 30720
    assert result["maximum_gram_condition_number"] == "1000000000000"
    assert result["maximum_normal_system_condition_squared"] == "1000000000000"
    assert result["root_role_status"] == "NOT_NOMINATED" and result["grouping_eligible"] is False
    assert result["event_identity_is_verified"] is result["native_runtime_verified"] is False
    assert result["physical_geometry_is_verified"] is False and result["observed_state"] == "UNKNOWN"
    assert json.loads(json.dumps(result)) == result
    # With pytest -s, root receives the aggregate from this one full execution.
    # Do not additionally run the script CLI merely to duplicate the same matrix.
    print("CAD_ROOT_AXIS_MATH_AGGREGATE=" + json.dumps(result, sort_keys=True))
