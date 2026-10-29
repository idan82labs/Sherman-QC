"""Independent rejection-only box-screen checks; invented arrays, no execution.

Exact Fraction oracles use represented binary64 values, not production arithmetic
or a geometric query implementation. A returned box lower bound can prove only
outside-distance rows. Its complement never supplies finite-face support.
A box enclosing a hole or an oblique patch must remain inconclusive inside it.

Complete row aggregation, triangle-to-box enclosure, transformed-row identity,
source ownership and work charging remain caller/integration obligations. The
misuse demonstrations below explicitly DO NOT qualify those unimplemented gates.
Prospective isolated gate <=60s wall/45s worker CPU/1GiB sampled tree RSS/1MiB
stdout. Root must independently freeze this source/helper/dependency before run.
No native, physical, Gold, prediction, application or database imports/data.
"""
from fractions import Fraction
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"backend/cad_guided_face_box_screen_v1.py"
BOUNDS_SOURCE = ROOT/"backend/cad_guided_triangle_distance_bounds_v3.py"
RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
GEOMETRY_SCOPE = "CONDITIONAL_CALLER_BOUND_ENCLOSING_BOX_REJECTION_ONLY"
COUNT_SCOPE = "CALLER_COMPLETE_SUPPORT_ROW_DENOMINATOR_REQUIRED"
ARITHMETIC_MODEL = "CONDITIONAL_BINARY64_RN_GRADUAL_UNDERFLOW_CORRECT_SQRT"


@pytest.fixture
def screen():
    # Exact named source modules under an isolated package; never backend startup.
    package = "_independent_face_box_screen_v1"
    names = (package, package+".cad_guided_triangle_distance_bounds_v3",
             package+".cad_guided_face_box_screen_v1")
    assert not any(name in sys.modules for name in names)
    root = ModuleType(package)
    root.__path__ = [str(SOURCE.parent)]
    sys.modules[package] = root
    try:
        for name, path in zip(names[1:], (BOUNDS_SOURCE, SOURCE)):
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules[names[-1]]
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def _q(value):
    return Fraction(*float(value).as_integer_ratio())


def _exact_box_distances(point, lower, upper):
    gaps = tuple(max(Fraction(0), _q(lo)-_q(p), _q(p)-_q(hi))
                 for p, lo, hi in zip(point, lower, upper))
    return max(gaps), sum((gap*gap for gap in gaps), Fraction(0))


def _assert_complete(result, points, lower, upper, gate):
    assert result["status"] == "COMPLETE", result
    assert result["reason"] == "CONDITIONAL_BOX_DISTANCE_LOWER_BOUND"
    assert result["arithmetic_model"] == ARITHMETIC_MODEL
    assert result["geometry_scope"] == GEOMETRY_SCOPE
    assert result["finite_surface_support_admitted"] is False
    count = len(points)
    assert result["input_rows"] == result["attempted_rows"] == result["completed_rows"] == count
    bounds, outside = result["distance_lower_mm"], result["definitely_outside_mask"]
    assert type(bounds) is np.ndarray and bounds.dtype == np.dtype(np.float64) and bounds.shape == (count,)
    assert type(outside) is np.ndarray and outside.dtype == np.dtype(bool) and outside.shape == (count,)
    assert np.isfinite(bounds).all() and np.all(bounds >= 0)
    assert type(result["definitely_outside_count"]) is int
    assert result["definitely_outside_count"] == sum(bool(value) for value in outside)
    exact_gate = _q(gate)
    for point, bound, rejected in zip(points, bounds, outside):
        exact_inf, exact_euclidean_squared = _exact_box_distances(point, lower, upper)
        represented = _q(bound)
        assert 0 <= represented <= exact_inf
        assert represented*represented <= exact_euclidean_squared
        # A useful tight screen, not the vacuous always-zero lower bound.
        assert exact_inf-represented <= 2*_q(math.ulp(float(exact_inf)))
        assert bool(rejected) == (represented > exact_gate)
        if rejected:
            assert exact_inf > exact_gate and exact_euclidean_squared > exact_gate*exact_gate
    return result


def _assert_cleared(result):
    assert result["status"] == "UNAVAILABLE" and type(result["reason"]) is str and result["reason"]
    assert result["completed_rows"] == 0
    assert result["distance_lower_mm"] is None and result["definitely_outside_mask"] is None
    assert result["definitely_outside_count"] is None
    assert result["finite_surface_support_admitted"] is False
    assert result["geometry_scope"] == GEOMETRY_SCOPE


def _assert_decision(result, count, outside, minimum):
    exact = Fraction(count-outside, count)
    upper = _q(result["possible_inlier_fraction_upper"])
    assert exact <= upper <= 1
    assert upper-exact <= 2*_q(math.ulp(float(exact)))
    assert result["support_count"] == count and result["definitely_outside_count"] == outside
    assert result["possible_inlier_count_upper"] == count-outside
    assert result["minimum_coverage"] == minimum
    assert result["count_scope"] == COUNT_SCOPE and result["finite_surface_support_admitted"] is False
    assert result["decision"] == ("BOUND_EXCLUDED" if upper < _q(minimum) else "NEEDS_SURFACE_QUERY")
    if result["decision"] == "BOUND_EXCLUDED":
        assert exact < _q(minimum)
    return result


def _base():
    return np.array([[2., 0., 0.], [0., 0., 0.]]), np.array([-1., -1., -1.]), np.array([1., 1., 1.])


@pytest.mark.parametrize("power", (-1000, -400, 0, 250))
@pytest.mark.parametrize("frame", ("identity", "permuted", "translated", "translated_permuted"))
def test_exact_represented_bounds_across_scales_and_frames(screen, power, frame):
    points = np.array([[0., 0., 1.], [-1., -2., 0.], [2., 3., 4.],
        [-2., 1., 1.], [4., 1., 1.], [0., -4., 1.], [0., 5., 1.],
        [0., 0., -2.], [0., 0., 6.], [-3., -5., -4.], [4., 6., 8.]])
    lower, upper = np.array([-1., -2., 0.]), np.array([2., 3., 4.])
    if "permuted" in frame:
        points, lower, upper = points[:, [2, 0, 1]], lower[[2, 0, 1]], upper[[2, 0, 1]]
    offset = np.array([2.**40, -2.**40, 2.**40]) if "translated" in frame else np.zeros(3)
    scale = math.ldexp(1., power)
    points, lower, upper = (points+offset)*scale, (lower+offset)*scale, (upper+offset)*scale
    originals = [value.copy() for value in (points, lower, upper)]
    result = screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5*scale)
    _assert_complete(result, points, lower, upper, .5*scale)
    assert result["definitely_outside_count"] == 8
    for value, original in zip((points, lower, upper), originals):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("point,gate,excluded", (
    (1., 1., False),
    (1., math.nextafter(1., 0.), False),
    (1., math.nextafter(math.nextafter(1., 0.), 0.), True),
    (math.nextafter(1., math.inf), 1., False),
    (math.nextafter(math.nextafter(1., math.inf), math.inf), 1., True),
))
def test_strict_threshold_and_nextafter_neighbors_never_round_into_exclusion(screen, point, gate, excluded):
    points, lower, upper = np.array([[point, 0., 0.]]), np.zeros(3), np.zeros(3)
    result = _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=gate),
                              points, lower, upper, gate)
    assert bool(result["definitely_outside_mask"][0]) is excluded


def test_extreme_admitted_coordinates_and_normal_to_subnormal_intermediates(screen):
    maximum, tiny = 1e100, float(np.finfo(float).tiny)
    for point, lo, hi, gate in ((maximum, -maximum, -maximum, maximum),
                               (-maximum, maximum, maximum, maximum),
                               (maximum, math.nextafter(maximum, 0.), math.nextafter(maximum, 0.), 0.),
                               (tiny, 0., 0., 0.), (-tiny, 0., 0., 0.)):
        points, lower, upper = np.array([[point, 0., 0.]]), np.array([lo, 0., 0.]), np.array([hi, 0., 0.])
        result = _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=gate),
                                  points, lower, upper, gate)
        assert result["definitely_outside_count"] == 1


@pytest.mark.parametrize("kind", ("expanded_zero", "positive_subnormal_box", "cancellation"))
def test_subnormal_box_endpoints_and_gradual_underflow_are_conservative(screen, kind):
    smallest, tiny = math.ulp(0.), float(np.finfo(float).tiny)
    if kind == "expanded_zero":
        points, lower, upper = np.zeros((1, 3)), np.full(3, -smallest), np.full(3, smallest)
    elif kind == "positive_subnormal_box":
        points, lower, upper = np.zeros((1, 3)), np.array([smallest, 0., 0.]), np.array([2*smallest, 0., 0.])
    else:
        points = np.array([[tiny, 0., 0.]])
        lower = upper = np.array([math.nextafter(tiny, math.inf), 0., 0.])
    result = _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=0.),
                              points, lower, upper, 0.)
    assert result["definitely_outside_count"] == 0


@pytest.mark.parametrize("sign", (-1., 1.))
def test_original_subnormal_point_refuses_whole_result(screen, sign):
    points, lower, upper = _base()
    points[-1, 2] = sign*math.ulp(0.)
    _assert_cleared(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=0.))


@pytest.mark.parametrize("budget", (0, 65536))
def test_empty_chunk_is_complete_but_cannot_be_a_complete_support_denominator(screen, budget):
    points, lower, upper = np.empty((0, 3)), np.zeros(3), np.zeros(3)
    _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=0., max_rows=budget),
                     points, lower, upper, 0.)
    with pytest.raises(ValueError, match="INVALID_COMPLETE_SUPPORT_COUNTS"):
        screen.decide_face_box_exclusion(support_count=0, definitely_outside_count=0, minimum_coverage=.5)


def test_maximum_chunk_is_not_thinned_and_next_row_refuses_all(screen):
    points = np.zeros((65536, 3))
    points[:, 0] = 2.
    lower, upper = np.zeros(3), np.ones(3)
    result = screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5)
    assert result["status"] == "COMPLETE" and result["completed_rows"] == 65536
    assert result["definitely_outside_count"] == 65536
    assert result["distance_lower_mm"].shape == result["definitely_outside_mask"].shape == (65536,)
    over = np.vstack((points, [[2., 0., 0.]]))
    _assert_cleared(screen.screen_face_box_rows(over, lower, upper, distance_gate_mm=.5))


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("kind", ("list", "tuple", "none", "vector", "wrong_columns", "rank3",
    "float32", "integers", "object", "big_endian", "subclass"))
def test_malformed_point_representation_is_never_coerced_into_support(screen, kind):
    points, lower, upper = _base()
    if kind == "list":
        points = points.tolist()
    elif kind == "tuple":
        points = tuple(map(tuple, points))
    elif kind == "none":
        points = None
    elif kind == "vector":
        points = np.zeros(3)
    elif kind == "wrong_columns":
        points = np.zeros((2, 4))
    elif kind == "rank3":
        points = np.zeros((2, 1, 3))
    elif kind == "subclass":
        points = points.view(_ArraySubclass)
    else:
        points = points.astype({"float32": np.float32, "integers": np.int64,
                               "object": object, "big_endian": ">f8"}[kind])
    _assert_cleared(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5))


@pytest.mark.parametrize("kind", ("list", "shape", "float32", "subclass", "nan", "infinity", "over_range", "inverted"))
def test_box_admission_and_order_fail_closed(screen, kind):
    points, lower, upper = _base()
    if kind == "list":
        lower = lower.tolist()
    elif kind == "shape":
        upper = np.ones((1, 3))
    elif kind == "float32":
        lower = lower.astype(np.float32)
    elif kind == "subclass":
        upper = upper.view(_ArraySubclass)
    elif kind == "nan":
        lower[0] = float("nan")
    elif kind == "infinity":
        upper[0] = float("inf")
    elif kind == "over_range":
        upper[0] = math.nextafter(1e100, math.inf)
    else:
        lower[0] = upper[0]+1.
    _assert_cleared(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5))


class _NoCoerceFloat(float):
    def __float__(self):
        raise AssertionError("SCALAR_SUBCLASS_MUST_NOT_BE_COERCED")


class _NoCoerceInt(int):
    def __float__(self):
        raise AssertionError("SCALAR_SUBCLASS_MUST_NOT_BE_COERCED")


INVALID_SCALARS = (
    pytest.param(True, id="bool_true"), pytest.param(False, id="bool_false"),
    pytest.param(None, id="none"), pytest.param("1", id="string"),
    pytest.param([], id="list"), pytest.param(1+0j, id="complex"),
    pytest.param(np.float64(.5), id="numpy_float"), pytest.param(np.int64(1), id="numpy_int"),
    pytest.param(np.array(.5), id="numpy_scalar_array"),
    pytest.param(Fraction(1, 2), id="fraction_object"),
    pytest.param(_NoCoerceFloat(.5), id="float_subclass"), pytest.param(_NoCoerceInt(1), id="int_subclass"),
    pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="positive_inf"),
    pytest.param(-float("inf"), id="negative_inf"), pytest.param(-1., id="negative"),
    pytest.param(math.nextafter(1e100, math.inf), id="above_coordinate_limit"),
    pytest.param(2**53+1, id="inexact_integer"), pytest.param(2**54, id="oversized_exact_integer"),
    pytest.param(10**400, id="huge_integer"),
)


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_scalar_abuse_clears_screen_and_raises_value_error_for_decision(screen, value):
    points, lower, upper = _base()
    _assert_cleared(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=value))
    with pytest.raises(ValueError):
        screen.decide_face_box_exclusion(support_count=4, definitely_outside_count=2, minimum_coverage=value)


def test_exact_integer_boundary_and_signed_zero_scalars_are_explicit(screen):
    points, lower, upper = _base()
    for gate in (0, -0., 2**53, 1e100):
        _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=gate),
                         points, lower, upper, gate)
    for minimum in (0, -0., 1):
        _assert_decision(screen.decide_face_box_exclusion(support_count=4, definitely_outside_count=2,
            minimum_coverage=minimum), 4, 2, minimum)
    with pytest.raises(ValueError):
        screen.decide_face_box_exclusion(support_count=4, definitely_outside_count=2,
                                        minimum_coverage=math.nextafter(1., math.inf))


@pytest.mark.parametrize("budget", (True, False, -1, 65537, 1., np.int64(1), None, 10**400))
def test_chunk_budget_must_be_exact_bounded_integer(screen, budget):
    points, lower, upper = _base()
    _assert_cleared(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5, max_rows=budget))


@pytest.mark.parametrize("layout", ("c", "fortran", "strided", "readonly"))
def test_owned_outputs_do_not_alias_or_modify_input_arrays(screen, layout):
    points, lower, upper = _base()
    if layout == "fortran":
        points = np.asfortranarray(points)
    elif layout == "strided":
        store = np.repeat(points, 2, axis=0)
        points = store[::2]
    elif layout == "readonly":
        for value in (points, lower, upper):
            value.flags.writeable = False
    original = [value.copy() for value in (points, lower, upper)]
    result = _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5),
                              points, lower, upper, .5)
    for value, before in zip((points, lower, upper), original):
        np.testing.assert_array_equal(value, before)
        assert not np.shares_memory(result["distance_lower_mm"], value)
        assert not np.shares_memory(result["definitely_outside_mask"], value)
    result["distance_lower_mm"][:] = 999.
    result["definitely_outside_mask"][:] = False
    for value, before in zip((points, lower, upper), original):
        np.testing.assert_array_equal(value, before)


def test_arrays_are_owned_before_arithmetic_sentinel_can_mutate_caller_storage(screen, monkeypatch):
    points, lower, upper = _base()
    original = [value.copy() for value in (points, lower, upper)]
    real = screen._arithmetic_sentinels
    def mutate():
        points[:] = 100.
        lower[:] = -200.
        upper[:] = 200.
        return real()
    monkeypatch.setattr(screen, "_arithmetic_sentinels", mutate)
    result = screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=.5)
    _assert_complete(result, *original, .5)
    assert result["definitely_outside_count"] == 1


def test_failed_arithmetic_model_never_yields_screen_or_exclusion(screen, monkeypatch):
    monkeypatch.setattr(screen, "_arithmetic_sentinels", lambda: False)
    result = screen.screen_face_box_rows(*_base(), distance_gate_mm=.5)
    _assert_cleared(result)
    assert result["attempted_rows"] == 0
    assert result["reason"] == "UNSUPPORTED_ARITHMETIC_MODEL_SENTINEL"
    with pytest.raises(ValueError, match="UNSUPPORTED_ARITHMETIC_MODEL_SENTINEL"):
        screen.decide_face_box_exclusion(support_count=10, definitely_outside_count=10, minimum_coverage=.5)


@pytest.mark.parametrize("fault", ("raise_during_bound", "nonfinite_bound", "raise_during_count"))
def test_arithmetic_failure_after_attempt_never_leaks_prefix_outputs(screen, monkeypatch, fault):
    class ArrayProxy:
        def __getattr__(self, name):
            if name == "nextafter" and fault != "raise_during_count":
                def nextafter(values, direction):
                    if fault == "raise_during_bound":
                        raise FloatingPointError("INJECTED_BOUND_ARITHMETIC_FAILURE")
                    return np.full_like(values, np.nan)
                return nextafter
            if name == "count_nonzero" and fault == "raise_during_count":
                def count(_):
                    raise TypeError("INJECTED_COUNT_FAILURE")
                return count
            return getattr(np, name)
    monkeypatch.setattr(screen, "np", ArrayProxy())
    result = screen.screen_face_box_rows(*_base(), distance_gate_mm=.5)
    _assert_cleared(result)
    assert result["attempted_rows"] == 2


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), math.nextafter(1e100, math.inf)))
def test_late_invalid_point_clears_preceding_obviously_outside_rows(screen, bad):
    points = np.zeros((1000, 3))
    points[:, 0] = 10.
    points[-1, 2] = bad
    result = screen.screen_face_box_rows(points, np.zeros(3), np.ones(3), distance_gate_mm=.5)
    _assert_cleared(result)
    assert result["attempted_rows"] == 0


@pytest.mark.parametrize("count", (3, 6, 7, 17, 255, 999983, 1000000))
def test_possible_fraction_upper_and_decisions_against_exact_rationals(screen, count):
    for possible in sorted({0, 1, count//2, count-1, count}):
        exact = Fraction(possible, count)
        rounded = float(exact)
        thresholds = sorted({0., 1., rounded, max(0., math.nextafter(rounded, -math.inf)),
                             min(1., math.nextafter(rounded, math.inf))})
        for minimum in thresholds:
            outside = count-possible
            result = screen.decide_face_box_exclusion(support_count=count, definitely_outside_count=outside,
                                                     minimum_coverage=minimum)
            _assert_decision(result, count, outside, minimum)


@pytest.mark.parametrize("count,outside,minimum", ((2, 1, .5), (4, 1, .75), (8, 6, .25), (1000000, 500000, .5)))
def test_exact_fraction_equality_never_excludes(screen, count, outside, minimum):
    assert Fraction(count-outside, count) == _q(minimum)
    result = _assert_decision(screen.decide_face_box_exclusion(support_count=count,
        definitely_outside_count=outside, minimum_coverage=minimum), count, outside, minimum)
    assert result["decision"] == "NEEDS_SURFACE_QUERY"


def test_five_sixths_rounding_cannot_turn_a_near_equal_gate_into_exclusion(screen):
    represented = float(Fraction(5, 6))
    assert _q(represented) > Fraction(5, 6)
    next_gate = math.nextafter(represented, math.inf)
    for minimum in (represented, next_gate):
        result = _assert_decision(screen.decide_face_box_exclusion(support_count=6,
            definitely_outside_count=1, minimum_coverage=minimum), 6, 1, minimum)
        assert result["decision"] == "NEEDS_SURFACE_QUERY"
    higher = math.nextafter(next_gate, math.inf)
    result = _assert_decision(screen.decide_face_box_exclusion(support_count=6,
        definitely_outside_count=1, minimum_coverage=higher), 6, 1, higher)
    assert result["decision"] == "BOUND_EXCLUDED"


def test_zero_possible_support_keeps_directed_up_subnormal_and_zero_gate_semantics(screen):
    for minimum, expected in ((0., "NEEDS_SURFACE_QUERY"), (math.ulp(0.), "NEEDS_SURFACE_QUERY"),
                              (2*math.ulp(0.), "BOUND_EXCLUDED"), (1., "BOUND_EXCLUDED")):
        result = _assert_decision(screen.decide_face_box_exclusion(support_count=10,
            definitely_outside_count=10, minimum_coverage=minimum), 10, 10, minimum)
        assert result["possible_inlier_fraction_upper"] == math.ulp(0.)
        assert result["decision"] == expected


@pytest.mark.parametrize("count,outside", (
    (0, 0), (-1, 0), (1000001, 0), (True, 0), (2., 1), (np.int64(2), 1), (10**400, 0),
    (2, -1), (2, 3), (2, True), (2, 1.), (2, np.int64(1)), (2, None), (2, 10**400),
))
def test_counts_require_complete_exact_bounded_integers(screen, count, outside):
    with pytest.raises(ValueError, match="INVALID_COMPLETE_SUPPORT_COUNTS"):
        screen.decide_face_box_exclusion(support_count=count, definitely_outside_count=outside, minimum_coverage=.75)


def test_chunk_denominator_misuse_is_caller_responsibility_and_full_aggregation_prevents_false_refusal(screen):
    # A valid first chunk is not the whole face support. This helper cannot
    # distinguish the two declarations without row/face binding from its caller.
    points = np.zeros((100, 3))
    points[:10, 0] = 2.
    lower, upper = np.zeros(3), np.ones(3)
    chunks = [screen.screen_face_box_rows(part, lower, upper, distance_gate_mm=.5)
              for part in (points[:10], points[10:])]
    assert all(row["status"] == "COMPLETE" for row in chunks)
    counts = [row["definitely_outside_count"] for row in chunks]
    wrong = screen.decide_face_box_exclusion(support_count=10, definitely_outside_count=counts[0], minimum_coverage=.75)
    complete = screen.decide_face_box_exclusion(support_count=len(points),
        definitely_outside_count=sum(counts), minimum_coverage=.75)
    assert wrong["decision"] == "BOUND_EXCLUDED"  # Invalid whole-support attribution, not a qualified exclusion.
    assert complete["decision"] == "NEEDS_SURFACE_QUERY"
    assert complete["possible_inlier_count_upper"] == 90
    assert wrong["count_scope"] == complete["count_scope"] == COUNT_SCOPE
    assert wrong["finite_surface_support_admitted"] is complete["finite_surface_support_admitted"] is False


@pytest.mark.parametrize("geometry", ("hole", "oblique_patch"))
def test_boxes_with_holes_or_loose_oblique_extent_never_admit_finite_support(screen, geometry):
    if geometry == "hole":
        # Four rectangular bands form a square frame with an actual central hole.
        rectangles = ((-2., -1., -2., 2.), (1., 2., -2., 2.),
                      (-1., 1., -2., -1.), (-1., 1., 1., 2.))
        triangles = []
        for x0, x1, y0, y1 in rectangles:
            a, b, c, d = [x0, y0, 0.], [x1, y0, 0.], [x0, y1, 0.], [x1, y1, 0.]
            triangles.extend(((a, b, c), (c, b, d)))
        vertices, point = np.asarray(triangles).reshape(-1, 3), np.array([0., 0., 0.])
        # Exact distance to this union is the minimum exact rectangle distance.
        exact_surface_distance_squared = min(_exact_box_distances(point,
            [x0, y0, 0.], [x1, y1, 0.])[1] for x0, x1, y0, y1 in rectangles)
        assert exact_surface_distance_squared == 1
    else:
        vertices = np.array([[-1., -1., -1.], [-1., 1., -1.], [1., -1., 1.], [1., 1., 1.]])
        point = np.array([1., 0., -1.])
        # Plane x-z=0: squared distance is (x-z)^2/(1^2+(-1)^2)=2.
        # Its foot (0,0,0) belongs to the finite patch.
        normal = (Fraction(1), Fraction(0), Fraction(-1))
        signed = sum((coordinate*_q(value) for coordinate, value in zip(normal, point)), Fraction(0))
        norm_squared = sum((coordinate*coordinate for coordinate in normal), Fraction(0))
        exact_surface_distance_squared = signed*signed/norm_squared
        foot = tuple(_q(value)-signed*coordinate/norm_squared for value, coordinate in zip(point, normal))
        assert foot == tuple(sum((_q(vertex[axis])/4 for vertex in vertices), Fraction(0)) for axis in range(3))
        assert exact_surface_distance_squared == 2
    lower, upper, points, gate = vertices.min(axis=0), vertices.max(axis=0), point[None, :], .5
    assert exact_surface_distance_squared > _q(gate)**2
    result = _assert_complete(screen.screen_face_box_rows(points, lower, upper, distance_gate_mm=gate),
                              points, lower, upper, gate)
    assert result["distance_lower_mm"][0] == 0. and result["definitely_outside_count"] == 0
    decision = _assert_decision(screen.decide_face_box_exclusion(support_count=1,
        definitely_outside_count=0, minimum_coverage=1.), 1, 0, 1.)
    assert decision["decision"] == "NEEDS_SURFACE_QUERY" and decision["finite_surface_support_admitted"] is False


def test_invalid_triangle_box_binding_cannot_be_authenticated_by_array_only_helper(screen):
    # The stipulated finite triangle contains p with exact barycentric weights.
    vertices = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])
    weights = (Fraction(1, 8), Fraction(5, 8), Fraction(1, 4))
    point = tuple(sum((_q(vertices[i, axis])*weights[i] for i in range(3)), Fraction(0))
                  for axis in range(3))
    points = np.array([[float(value) for value in point]])
    correct_lower, correct_upper = vertices.min(axis=0), vertices.max(axis=0)
    correct = screen.screen_face_box_rows(points, correct_lower, correct_upper, distance_gate_mm=.1)
    invalid = screen.screen_face_box_rows(points, np.zeros(3), np.zeros(3), distance_gate_mm=.1)
    assert correct["definitely_outside_count"] == 0
    assert invalid["definitely_outside_count"] == 1  # The supplied box does NOT enclose the intended triangle.
    assert invalid["geometry_scope"] == GEOMETRY_SCOPE
    assert invalid["finite_surface_support_admitted"] is False


def test_source_closure_and_prospective_limits_are_explicit():
    tree = ast.parse(SOURCE.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 2
    assert isinstance(imports[0], ast.Import) and [name.name for name in imports[0].names] == ["numpy"]
    assert isinstance(imports[1], ast.ImportFrom) and imports[1].module == "cad_guided_triangle_distance_bounds_v3"
    forbidden = {"open", "exec", "eval", "compile", "__import__"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in forbidden for node in ast.walk(tree))
    assert RESOURCE_LIMITS == dict(wall_seconds=60, cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
