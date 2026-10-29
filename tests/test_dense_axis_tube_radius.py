"""Independent invented contracts for dense declared-axis radius constraints.

Only NumPy, pytest, the standard library and the new source seam are imported.
No corpus, protected input, native geometry, app, service or data file is used.
The analytic cylinders below supply test-only truth. The engine receives a
stipulated axis-line family and raw positions, never a fitted truth label.
Nonempty per-row outer constraints are deliberately NOT a shared-model witness.
"""
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry.dense_axis_tube_radius import analyze_dense_axis_tube_radius


def analyze(chunks, **changes):
    policy = dict(
        unit="mm", axis=[0.0, 0.0, 1.0], axis_point=[0.0, 0.0, 0.0],
        axis_angle_bound=0.0, axis_point_error_bound=0.0, position_error_bound=0.0,
        radius_bounds=[1.0, 3.0], max_total_rows=2_000_000,
        max_chunk_rows=65_536, max_chunks=4096,
    )
    policy.update(changes)
    return analyze_dense_axis_tube_radius(chunks, **policy)


def cylinder_rows(radius=2.0, *, tilt=0.0, origin=(0.0, 0.0, 0.0),
                  stations=(-3.0, 0.0, 4.0), angular_count=16):
    """Analytic orthonormal extrusion, independent of the source's distance code."""
    axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    first = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    second = np.array([0.0, 1.0, 0.0])
    theta = np.arange(angular_count)*2*math.pi/angular_count
    radial = radius*(np.cos(theta)[:, None]*first+np.sin(theta)[:, None]*second)
    points = np.concatenate([np.asarray(origin)+station*axis+radial for station in stations])
    np.testing.assert_allclose(np.column_stack([first, second, axis]).T
                               @np.column_stack([first, second, axis]), np.eye(3), atol=1e-14, rtol=0)
    return points, axis


def assert_scope(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["kind"] == "DENSE_AXIS_TUBE_RADIUS_CONSTRAINTS_V1"
    assert result["radius_interval_scope"] == "NECESSARY_OUTER_CONSTRAINT_FOR_DECLARED_AXIS_FAMILY_SUPPORT_AND_PRIOR"
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_bend_count"] is None
    assert result["shared_model_feasibility_is_established"] is False
    hypothesis = result["hypothesis"]
    assert hypothesis["axis_and_point_are_verified"] is False
    assert hypothesis["support_is_verified"] is False
    assert hypothesis["axis_point_bound_is_inferred_from_measurement_epsilon"] is False
    policy = result["policy"]
    assert policy["all_supplied_rows_are_in_the_single_support_hypothesis"] is True
    assert policy["raw_duplicates_consume_budget"] is True
    assert policy["sampling_or_residual_trimming"] is False
    assert policy["retained_raw_rows_scope"] == "NO_RAW_ARCHIVE_BEYOND_TRANSIENT_CHUNK_WORKSPACE"
    assert policy["producer_allocation_or_blocking_time_is_bounded"] is False
    assert policy["eof_required_for_any_geometric_conclusion"] is True
    assert policy["input_row_count_scope"] == "FULLY_VALIDATED_AND_CONSTRAINED_CHUNKS_EXCLUDES_REJECTED_CHUNK"
    assert policy["extremum_tie_rule"] == "EARLIEST_STREAM_ROW_WITH_EXACT_EQUAL_VALUE"
    assert result["diagnostics"]["retained_raw_rows"] == 0
    for limitation in (
        "NONEMPTY_OUTER_INTERSECTION_NOT_SHARED_MODEL_FEASIBILITY",
        "EMPTY_INTERSECTION_EXCLUDES_ONLY_DECLARED_FAMILY_SUPPORT_AND_RADIUS_PRIOR",
        "NO_TOPOLOGY_GRAPH_OR_PHYSICAL_EVENT_COUNT", "EOF_NOT_PHYSICAL_CAPTURE_COVERAGE",
        "STREAM_ROWS_AUDITED_NOT_DURABLY_RETAINED_MEASUREMENT_ARCHIVE",
        "AXIS_POINT_BALL_REQUIRES_SEPARATE_JUSTIFICATION_NOT_FIT_RESIDUAL_OR_EPSILON",
    ):
        assert limitation in result["limitations"]
    if result["constraint_extrema"] is not None:
        assert result["constraint_extrema"]["scope"] == (
            "ALL_ROW_NECESSARY_CONSTRAINTS_NOT_FEASIBLE_SET_OR_COMPLETE_PROVENANCE_ARCHIVE")


def assert_contains(result, radius):
    assert_scope(result)
    assert result["stream_fully_consumed"] is True
    assert result["search_complete_within_declared_policy"] is True
    assert result["status"] == "NECESSARY_RADIUS_CONSTRAINTS"
    lower, upper = result["radius_interval"]
    assert lower <= radius <= upper
    assert result["constraint_extrema"]["prior_intersection_lower"] == lower
    assert result["constraint_extrema"]["prior_intersection_upper"] == upper


def assert_incomplete(result, reason, accepted_rows):
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["stream_fully_consumed"] is False
    assert result["radius_interval"] is result["constraint_extrema"] is None
    assert result["reasons"] == [reason]
    assert result["input_row_count"] == result["diagnostics"]["constrained_raw_rows"] == accepted_rows


def test_public_api_requires_every_hypothesis_and_budget_before_stream_consumption():
    signature = inspect.signature(analyze_dense_axis_tube_radius)
    assert set(signature.parameters) == {
        "chunks", "unit", "axis", "axis_point", "axis_angle_bound", "axis_point_error_bound",
        "position_error_bound", "radius_bounds", "max_total_rows", "max_chunk_rows", "max_chunks",
    }
    assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for name, parameter in signature.parameters.items() if name != "chunks")


def test_exact_fixed_line_contains_known_radius_with_only_numerical_allowance():
    points, _ = cylinder_rows()
    result = analyze([points])
    assert_contains(result, 2.0)
    assert result["input_row_count"] == result["diagnostics"]["constrained_raw_rows"] == 48
    assert result["radius_interval"][1]-result["radius_interval"][0] < 1e-9
    assert result["diagnostics"]["max_propagated_error_bound"] == 0
    assert result["diagnostics"]["max_numerical_allowance"] > 0


@pytest.mark.parametrize("tilt", [-0.08, 0.0, 0.08])
def test_one_shared_true_cylinder_inside_declared_point_ball_and_axis_cone_survives_bounded_noise(tilt):
    nominal_origin = np.array([1.0, -2.0, 3.0])
    displacement = np.array([0.04, -0.06, 0.03])
    epsilon, point_bound, angle_bound = 0.02, 0.1, 0.1
    exact, true_axis = cylinder_rows(tilt=tilt, origin=nominal_origin+displacement)
    indices = np.arange(1, len(exact)+1, dtype=float)
    directions = np.column_stack([np.sin(indices), np.cos(indices), np.sin(2*indices)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    observed = exact+0.75*epsilon*directions
    assert np.linalg.norm(displacement) <= point_bound
    assert math.acos(abs(float(true_axis@[0.0, 0.0, 1.0]))) <= angle_bound
    errors = np.linalg.norm(observed-exact, axis=1)
    assert np.all((errors > 0) & (errors <= epsilon))
    result = analyze([observed[:17], observed[17:]], axis_point=nominal_origin,
                     axis_angle_bound=angle_bound, axis_point_error_bound=point_bound,
                     position_error_bound=epsilon)
    assert_contains(result, 2.0)
    assert result["diagnostics"]["max_propagated_error_bound"] >= epsilon+point_bound
    assert result["input_row_count"] == len(observed)


def test_distant_axial_lever_arm_requires_the_declared_direction_cone():
    points, _ = cylinder_rows(tilt=0.1, stations=(100.0,))
    fixed_wrong_line = analyze([points])
    assert_scope(fixed_wrong_line)
    assert fixed_wrong_line["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    cone = analyze([points], axis_angle_bound=0.1)
    assert_contains(cone, 2.0)
    assert cone["diagnostics"]["max_propagated_error_bound"] > 9.0


@pytest.mark.parametrize("scale,unit", [(1.0, "mm"), (0.001, "m")])
@pytest.mark.parametrize("translation", [(0.0, 0.0, 0.0), (20.0, -30.0, 40.0)])
def test_metric_scaling_and_joint_translation_preserve_radius_constraints(scale, unit, translation):
    points, _ = cylinder_rows()
    baseline = analyze([points], position_error_bound=0.01, axis_point_error_bound=0.03,
                       axis_angle_bound=0.01)
    moved = (points+translation)*scale
    result = analyze([moved], unit=unit, axis_point=np.asarray(translation)*scale,
                     position_error_bound=0.01*scale, axis_point_error_bound=0.03*scale,
                     axis_angle_bound=0.01, radius_bounds=[scale, 3*scale])
    assert_contains(result, 2*scale)
    # Translation changes the explicitly disclosed engineering allowance. This
    # comparison tolerates that padding, not a relaxed geometric error budget.
    np.testing.assert_allclose(result["radius_interval"], np.asarray(baseline["radius_interval"])*scale,
                               atol=1e-9*scale, rtol=0)
    assert result["position_error_bound"] == 0.01*scale
    assert result["hypothesis"]["axis_point_error_bound"] == 0.03*scale


def test_unoriented_axis_sign_keeps_exact_bounds_and_extremal_row_provenance():
    points, _ = cylinder_rows(tilt=0.06)
    first = analyze([points], axis_angle_bound=0.1)
    second = analyze([points], axis=[0.0, 0.0, -1.0], axis_angle_bound=0.1)
    for key in ("status", "radius_interval", "constraint_extrema", "diagnostics"):
        assert second[key] == first[key]
    assert_contains(first, 2.0)
    assert_contains(second, 2.0)


def test_more_than_256_duplicate_raw_rows_are_all_constrained_and_consume_budget():
    original = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    repeated = np.repeat(original, 300, axis=0)
    base, result = analyze([original]), analyze([repeated[:257], repeated[257:]])
    assert_contains(result, 2.0)
    assert result["radius_interval"] == base["radius_interval"]
    assert result["input_row_count"] == result["diagnostics"]["constrained_raw_rows"] == 600
    assert result["diagnostics"]["retained_raw_rows"] == 0
    assert result["constraint_extrema"]["max_sample_lower_row"] == 0
    assert result["constraint_extrema"]["min_sample_upper_row"] == 0
    limited = analyze([repeated[:257]], max_total_rows=256)
    assert_incomplete(limited, "TOTAL_ROW_BUDGET_EXCEEDED", 0)


def test_late_contradiction_is_never_trimmed_and_later_rows_still_update_extrema_until_eof():
    state = dict(tail_read=False, eof=False)

    def chunks():
        yield np.array([[2.0, 0.0, 0.0]])
        yield np.array([[4.0, 0.0, 0.0]])
        state["tail_read"] = True
        yield np.array([[1.5, 0.0, 0.0]])
        state["eof"] = True

    result = analyze(chunks(), radius_bounds=[1.0, 5.0])
    assert_scope(result)
    assert state == dict(tail_read=True, eof=True)
    assert result["stream_fully_consumed"] and result["search_complete_within_declared_policy"]
    assert result["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    assert result["radius_interval"] is None
    assert result["reasons"] == ["EMPTY_NECESSARY_RADIUS_INTERSECTION"]
    extrema = result["constraint_extrema"]
    assert extrema["max_sample_lower_row"] == 1 and extrema["min_sample_upper_row"] == 2
    assert extrema["prior_intersection_lower"] > extrema["prior_intersection_upper"]
    assert result["input_row_count"] == result["diagnostics"]["constrained_raw_rows"] == 3


def test_excluding_a_wrong_declared_axis_does_not_exclude_another_valid_cylinder():
    points, _ = cylinder_rows(origin=(5.0, 0.0, 0.0), stations=(0.0,), angular_count=4)
    wrong = analyze([points])
    assert_scope(wrong)
    assert wrong["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    assert wrong["radius_interval"] is None and wrong["constraint_extrema"] is not None
    correct = analyze([points], axis_point=[5.0, 0.0, 0.0])
    assert_contains(correct, 2.0)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2]-[5.0, 0.0], axis=1), 2.0,
                               atol=1e-14, rtol=0)


def test_nonempty_rowwise_relaxation_does_not_establish_a_shared_axis_line_model():
    points = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    result = analyze([points], axis_point_error_bound=1.0, radius_bounds=[1.0, 1.0])
    assert_contains(result, 1.0)
    assert result["radius_interval"] == [1.0, 1.0]
    # A common axis is parallel to Z because alpha=0. Projected points are four
    # apart, but any two points on one radius-one circle are at most two apart.
    # Individually, centers (+1,0) and (-1,0) in the point ball satisfy one row
    # each; no shared center can satisfy both. No optimizer output is assumed.
    assert np.linalg.norm(points[0, :2]-points[1, :2]) == 4.0 > 2*1.0
    assert result["shared_model_feasibility_is_established"] is False


def test_rechunking_preserves_exact_interval_and_earliest_tied_extremal_global_rows():
    points = np.array([[2.0, 0.0, 0.0], [2.3, 0.0, 0.0], [1.7, 0.0, 0.0],
                       [2.3, 0.0, 0.0], [1.7, 0.0, 0.0]])
    results = []
    for size in (1, 2, 5):
        chunks = [np.empty((0, 3))]+[points[i:i+size] for i in range(0, len(points), size)]
        result = analyze(chunks, position_error_bound=0.5)
        assert_contains(result, 2.0)
        assert result["input_row_count"] == 5
        assert result["constraint_extrema"]["max_sample_lower_row"] == 1
        assert result["constraint_extrema"]["min_sample_upper_row"] == 2
        results.append(result)
    for result in results[1:]:
        for key in ("radius_interval", "constraint_extrema", "status", "hypothesis", "reasons"):
            assert result[key] == results[0][key]
        for key in ("max_numerical_allowance", "max_propagated_error_bound",
                    "max_nominal_axis_distance", "max_origin_distance", "constrained_raw_rows"):
            assert result["diagnostics"][key] == results[0]["diagnostics"][key]


def test_axis_point_and_radius_prior_are_snapshotted_before_the_producer_mutates_them():
    axis = np.array([0.0, 0.0, 1.0])
    origin = np.zeros(3)
    prior = np.array([1.0, 3.0])
    state = dict(mutated=False)

    class MutatingProducer:
        def __iter__(self):
            axis[:] = [1.0, 0.0, 0.0]
            origin[:] = [100.0, 100.0, 100.0]
            prior[:] = [10.0, 20.0]
            state["mutated"] = True
            return iter([np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])])

    result = analyze(MutatingProducer(), axis=axis, axis_point=origin, radius_bounds=prior)
    assert state["mutated"]
    assert_contains(result, 2.0)
    assert result["hypothesis"]["axis"] == [0.0, 0.0, 1.0]
    assert result["hypothesis"]["axis_point"] == [0.0, 0.0, 0.0]
    assert result["hypothesis"]["radius_prior"] == [1.0, 3.0]


def test_reused_producer_chunk_is_constrained_before_its_next_mutation():
    chunk = np.array([[2.0, 0.0, 0.0]])

    def chunks():
        yield chunk
        chunk[0, 0] = 4.0
        yield chunk

    result = analyze(chunks(), radius_bounds=[1.0, 5.0])
    assert_scope(result)
    assert result["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    assert result["constraint_extrema"]["max_sample_lower_row"] == 1
    assert result["constraint_extrema"]["min_sample_upper_row"] == 0
    assert result["input_row_count"] == 2


@pytest.mark.parametrize("interrupted", [False, True])
def test_late_invalid_or_interrupted_stream_erases_even_an_earlier_exclusion(interrupted):
    def chunks():
        yield np.array([[1.0, 0.0, 0.0]])
        yield np.array([[3.0, 0.0, 0.0]])
        if interrupted:
            raise RuntimeError("invented producer failure after contradictory finite rows")
        yield np.array([[np.nan, 0.0, 0.0]])

    result = analyze(chunks())
    assert_incomplete(result, "INPUT_STREAM_FAILED" if interrupted else "NONFINITE_CHUNK_COORDINATE", 2)
    assert result["diagnostics"]["processed_chunks"] == 2
    assert result["diagnostics"]["received_chunks"] == (2 if interrupted else 3)


@pytest.mark.parametrize("chunks,processed", [([], 0), ([np.empty((0, 3))], 1),
                                               ([np.empty((0, 3)), np.empty((0, 3))], 2)])
def test_empty_eof_is_complete_ingestion_but_not_a_geometric_or_empty_part_conclusion(chunks, processed):
    result = analyze(chunks)
    assert_scope(result)
    assert result["stream_fully_consumed"] and result["search_complete_within_declared_policy"]
    assert result["status"] == "ABSTAINED" and result["reasons"] == ["NO_SUPPLIED_ROWS"]
    assert result["radius_interval"] is result["constraint_extrema"] is None
    assert result["input_row_count"] == 0
    assert result["diagnostics"]["processed_chunks"] == processed


@pytest.mark.parametrize("bad,reason", [
    ([[2.0, 0.0, 0.0]], "INVALID_CHUNK_ARRAY"),
    (np.array([2.0, 0.0, 0.0]), "INVALID_CHUNK_ARRAY"),
    (np.zeros((1, 4)), "INVALID_CHUNK_ARRAY"),
    (np.zeros((1, 3), dtype=bool), "INVALID_CHUNK_ARRAY"),
    (np.zeros((1, 3), dtype=complex), "INVALID_CHUNK_ARRAY"),
    (np.array([["2", "0", "0"]]), "INVALID_CHUNK_ARRAY"),
    (np.array([[2.0, 0.0, 0.0]], dtype=object), "INVALID_CHUNK_ARRAY"),
    (np.array([[np.inf, 0.0, 0.0]]), "NONFINITE_CHUNK_COORDINATE"),
    (np.array([[2**53+1, 0, 0]], dtype=np.int64), "INTEGER_COORDINATE_PRECISION_UNSUPPORTED"),
    (np.array([[2**53+1, 0, 0]], dtype=np.uint64), "INTEGER_COORDINATE_PRECISION_UNSUPPORTED"),
])
def test_every_late_rejected_chunk_clears_all_radius_constraints(bad, reason):
    result = analyze([np.array([[2.0, 0.0, 0.0]]), bad])
    assert_incomplete(result, reason, 1)
    assert result["diagnostics"]["processed_chunks"] == 1
    assert result["diagnostics"]["received_chunks"] == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int64, np.uint16, np.uint64])
def test_materialized_supported_numeric_chunk_types_preserve_exact_cardinal_radius(dtype):
    result = analyze([np.array([[2, 0, 0], [0, 2, 0]], dtype=dtype)])
    assert_contains(result, 2.0)


def test_platform_extended_float_is_accepted_only_if_its_width_is_at_most_eight_bytes():
    points = np.array([[2.0, 0.0, 0.0]], dtype=np.longdouble)
    result = analyze([points])
    if points.dtype.itemsize > 8:
        assert_incomplete(result, "INVALID_CHUNK_ARRAY", 0)
    else:
        assert_contains(result, 2.0)


def test_finite_but_unsupported_numerical_range_clears_the_valid_prefix():
    result = analyze([np.array([[2.0, 0.0, 0.0]]), np.array([[1.7e308, 1.7e308, 1.7e308]])])
    assert_incomplete(result, "NUMERICAL_RANGE_UNSUPPORTED", 1)
    assert result["diagnostics"]["processed_chunks"] == 1


def test_subnormal_distance_cannot_publish_a_falsely_collapsed_zero_guard_interval():
    eta = float(np.nextafter(0.0, 1.0))
    assert eta > 0 and eta < np.finfo(float).tiny
    # The exact mathematical radial distance is sqrt(2)*eta, between the prior
    # endpoints, but it cannot be represented with a reliable relative guard.
    result = analyze([np.array([[eta, eta, 0.0]])], radius_bounds=[eta, 2*eta])
    assert_incomplete(result, "NUMERICAL_RANGE_UNSUPPORTED", 0)
    assert result["diagnostics"]["processed_chunks"] == 0


def test_exact_all_zero_arithmetic_scale_remains_supported_but_excludes_a_positive_radius():
    result = analyze([np.zeros((1, 3))])
    assert_scope(result)
    assert result["stream_fully_consumed"] and result["search_complete_within_declared_policy"]
    assert result["status"] == "DECLARED_AXIS_TUBE_MODEL_EXCLUDED"
    assert result["radius_interval"] is None
    assert result["constraint_extrema"]["max_sample_lower"] == 0.0
    assert result["constraint_extrema"]["min_sample_upper"] == 0.0
    assert result["diagnostics"]["max_numerical_allowance"] == 0.0


@pytest.mark.parametrize("changes,chunks,reason,accepted", [
    ({"max_total_rows": 3}, [np.array([[2.0, 0.0, 0.0]]*2), np.array([[2.0, 0.0, 0.0]]*2)],
     "TOTAL_ROW_BUDGET_EXCEEDED", 2),
    ({"max_chunk_rows": 1}, [np.array([[2.0, 0.0, 0.0]]), np.array([[2.0, 0.0, 0.0]]*2)],
     "CHUNK_ROW_BUDGET_EXCEEDED", 1),
    ({"max_chunks": 1}, [np.array([[2.0, 0.0, 0.0]]), np.empty((0, 3))],
     "CHUNK_COUNT_BUDGET_EXCEEDED", 1),
])
def test_each_declared_stream_budget_fails_closed_before_counting_rejected_chunk(changes, chunks, reason, accepted):
    result = analyze(chunks, **changes)
    assert_incomplete(result, reason, accepted)
    assert result["diagnostics"]["received_chunks"] == 2
    assert result["diagnostics"]["processed_chunks"] == 1


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("unit", True),
    ("axis", [0, 0, 0]), ("axis", [0, 0, 2]), ("axis", [0, 0]),
    ("axis", [False, 0, 1]), ("axis", [0, np.nan, 1]),
    ("axis_point", [0, 0]), ("axis_point", [0, 0, math.inf]),
    ("axis_point", [2**53+1, 0, 0]),
    ("axis_angle_bound", -0.01), ("axis_angle_bound", math.pi/2),
    ("axis_angle_bound", math.nan), ("axis_angle_bound", True),
    ("axis_point_error_bound", -1), ("axis_point_error_bound", math.inf),
    ("axis_point_error_bound", True), ("position_error_bound", -1),
    ("position_error_bound", math.inf), ("position_error_bound", True),
    ("radius_bounds", [0, 2]), ("radius_bounds", [2, 1]),
    ("radius_bounds", [1, math.inf]), ("radius_bounds", [1]), ("radius_bounds", [True, 2]),
    ("max_total_rows", 0), ("max_total_rows", 2_000_001), ("max_total_rows", True),
    ("max_total_rows", 2.5), ("max_chunk_rows", 0), ("max_chunk_rows", 65_537),
    ("max_chunk_rows", True), ("max_chunks", 0), ("max_chunks", 4097), ("max_chunks", True),
])
def test_invalid_hypothesis_or_budget_is_rejected_before_iterating_any_input(field, value):
    class MustNotRead:
        def __iter__(self):
            raise AssertionError("invalid policy attempted producer access")

    with pytest.raises(ValueError):
        analyze(MustNotRead(), **{field: value})


def test_combined_uncertainty_overflow_is_rejected_before_input_access():
    class MustNotRead:
        def __iter__(self):
            raise AssertionError("overflowing policy attempted producer access")

    with pytest.raises(ValueError):
        analyze(MustNotRead(), position_error_bound=1e308, axis_point_error_bound=1e308)


def test_producer_initialization_failure_is_not_a_geometric_exclusion():
    class BrokenProducer:
        def __iter__(self):
            raise RuntimeError("invented initialization failure")

    assert_incomplete(analyze(BrokenProducer()), "INPUT_STREAM_FAILED", 0)


def test_noniterable_input_is_an_invalid_call_not_an_empty_stream():
    with pytest.raises(ValueError):
        analyze(None)
