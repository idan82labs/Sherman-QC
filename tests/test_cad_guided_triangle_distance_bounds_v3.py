"""Independent exact-rational certificate checks; invented arrays only.

No implementation arithmetic is reused as an oracle. Root must review this
source and the helper, freeze the fixed cases, and run the isolated pure gate.
"""
from fractions import Fraction
from itertools import combinations
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"backend/cad_guided_triangle_distance_bounds_v3.py"
ARRAY_KEYS = ("distance_lower", "distance_upper", "closest_points", "closest_point_error",
              "candidate_distance", "candidate_distance_lower", "candidate_distance_upper")


@pytest.fixture
def bounds():
    name = "independent_triangle_distance_bounds_v3"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _fraction(value):
    """Exactly the represented binary64 value, never decimal-string truth."""
    return Fraction(*float(value).as_integer_ratio())


def _vector(values):
    return tuple(_fraction(value) for value in values)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction(0))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _solve_exact(matrix, rhs):
    """Generic exact elimination; singular affine supports are skipped."""
    rows = [list(row)+[value] for row, value in zip(matrix, rhs)]
    size = len(rows)
    assert 1 <= size <= 4 and all(len(row) == size+1 for row in rows)
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column] != 0), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value/scale for value in rows[column]]
        for row in range(size):
            if row != column:
                scale = rows[row][column]
                rows[row] = [value-scale*reference for value, reference in zip(rows[row], rows[column])]
    return tuple(row[-1] for row in rows)


def _exact_projection(point, triangle):
    """Seven active sets, followed by a GLOBAL normal-cone certificate.

    This solves equality-constrained least squares in vertex coordinates, not
    the implementation's anchored interval arithmetic or a copied edge kernel.
    It also works for a segment/point hull because singular supports are skipped.
    """
    p, vertices = _vector(point), tuple(_vector(row) for row in triangle)
    witnesses = []
    for count in (1, 2, 3):
        for active in combinations(range(3), count):
            selected = [vertices[index] for index in active]
            matrix = [[_dot(a, b) for b in selected]+[Fraction(1)] for a in selected]
            matrix.append([Fraction(1)]*count+[Fraction(0)])
            rhs = [_dot(p, vertex) for vertex in selected]+[Fraction(1)]
            solution = _solve_exact(matrix, rhs)
            if solution is None or any(value < 0 for value in solution[:-1]):
                continue
            weights = [Fraction(0)]*3
            for index, value in zip(active, solution[:-1]):
                weights[index] = value
            assert sum(weights) == 1
            y = tuple(sum((weight*vertex[axis] for weight, vertex in zip(weights, vertices)), Fraction(0))
                      for axis in range(3))
            residual = _sub(p, y)
            if any(_dot(residual, _sub(vertex, y)) > 0 for vertex in vertices):
                continue
            witnesses.append((y, _dot(residual, residual), tuple(weights)))
    assert witnesses, "Exact convex-hull projection must have a certified active set"
    assert all(row[:2] == witnesses[0][:2] for row in witnesses)
    return witnesses[0]


def _exact_candidate(point, triangle, barycentric):
    p = _vector(point)
    a, b, c = (_vector(row) for row in triangle)
    alpha, beta = (_fraction(value) for value in barycentric)
    assert alpha >= 0 and beta >= 0 and alpha+beta <= 1
    q = tuple(x+alpha*(y-x)+beta*(z-x) for x, y, z in zip(a, b, c))
    residual = _sub(p, q)
    return q, _dot(residual, residual)


def _certified(result, points, triangles, barycentric):
    assert result["status"] == "COMPLETE", result
    count = len(points)
    assert result["closest_points"].shape == (count, 3)
    for key in ARRAY_KEYS:
        array = result[key]
        assert isinstance(array, np.ndarray) and array.dtype == np.dtype(np.float64)
        assert np.isfinite(array).all()
        if key != "closest_points":
            assert array.shape == (count,) and np.all(array >= 0)
    assert np.all(result["distance_lower"] <= result["distance_upper"])
    assert np.all(result["candidate_distance_lower"] <= result["candidate_distance"])
    assert np.all(result["candidate_distance"] <= result["candidate_distance_upper"])
    diagnostic = result["diagnostics"]
    assert diagnostic["input_pairs"] == diagnostic["attempted_pairs"] == diagnostic["completed_pairs"] == count
    assert diagnostic["full_result_cleared"] is False
    oracle = []
    for row, (point, triangle, candidate) in enumerate(zip(points, triangles, barycentric)):
        optimal, exact_distance_squared, weights = _exact_projection(point, triangle)
        q, candidate_squared = _exact_candidate(point, triangle, candidate)
        lower, upper = _fraction(result["distance_lower"][row]), _fraction(result["distance_upper"][row])
        assert lower*lower <= exact_distance_squared <= candidate_squared <= upper*upper
        candidate_lower = _fraction(result["candidate_distance_lower"][row])
        candidate_upper = _fraction(result["candidate_distance_upper"][row])
        assert candidate_lower*candidate_lower <= candidate_squared <= candidate_upper*candidate_upper
        represented_q = _vector(result["closest_points"][row])
        point_error_squared = _dot(_sub(represented_q, q), _sub(represented_q, q))
        radius = _fraction(result["closest_point_error"][row])
        assert point_error_squared <= radius*radius
        oracle.append((optimal, exact_distance_squared, weights, q, candidate_squared))
    return oracle


def _cleared(result):
    assert result["status"] == "UNAVAILABLE", result
    assert all(result[key] is None for key in ARRAY_KEYS)
    assert isinstance(result["diagnostics"], dict)
    assert result["diagnostics"]["full_result_cleared"] is True
    assert result["diagnostics"]["completed_pairs"] == 0
    assert type(result["diagnostics"]["attempted_pairs"]) is int
    assert result["diagnostics"]["attempted_pairs"] >= 0


def _base_arrays():
    return (np.array([[1., .5, 1.]], dtype=np.float64),
            np.array([[[0., 0., 0.], [4., 0., 0.], [0., 2., 0.]]], dtype=np.float64),
            np.array([[.25, .25]], dtype=np.float64))


@pytest.mark.parametrize("shape", ("axis", "oblique"))
@pytest.mark.parametrize("power", (-200, 0, 200))
@pytest.mark.parametrize("frame", ("identity", "permuted", "translated", "translated_permuted"))
def test_near_optimal_certificates_are_tight_in_represented_local_geometry(bounds, shape, power, frame):
    # Signed/permutation frames and dyadic inputs stay exact after conversion.
    # No presumed float rotation or intended pre-quantization surface is truth.
    triangle = np.array([[0., 0., 0.], [4., 0., 0.], [0., 2., 0.]]) if shape == "axis" else np.array(
        [[0., 0., 0.], [2., 1., -2.], [2., -2., 1.]])
    normal = np.array([0., 0., 1.]) if shape == "axis" else np.array([1., 2., 2.])
    barycentric = np.array([[.25, .25], [.125, .5], [.5, .125]])
    points = triangle[0]+barycentric[:, :1]*(triangle[1]-triangle[0])+barycentric[:, 1:]*(triangle[2]-triangle[0])+.25*normal
    if "permuted" in frame:
        triangle, points = triangle[:, [2, 0, 1]], points[:, [2, 0, 1]]
    scale = math.ldexp(1., power)
    translation = np.array([2.**40, -2.**40, 2.**40]) if "translated" in frame else np.zeros(3)
    triangles = np.repeat(((triangle+translation)*scale)[None, :, :], len(points), axis=0)
    points = (points+translation)*scale
    originals = [array.copy() for array in (points, triangles, barycentric)]
    result = bounds.enclose_triangle_distances(points, triangles, barycentric)
    oracle = _certified(result, points, triangles, barycentric)
    for row, (optimal, squared, _, q, candidate_squared) in enumerate(oracle):
        assert optimal == q and squared == candidate_squared
        # Fixed engineering targets, not inferred from the returned certificate.
        # These depend on local scale, never world-coordinate magnitude.
        width_limit = _fraction(math.ldexp(scale, -36))
        assert _fraction(result["distance_upper"][row])-_fraction(result["distance_lower"][row]) <= width_limit
        distance = _fraction(result["candidate_distance"][row])
        error = _fraction(math.ldexp(scale, -40))
        assert max(Fraction(0), distance-error)**2 <= candidate_squared <= (distance+error)**2
    for original, value in zip(originals, (points, triangles, barycentric)):
        np.testing.assert_array_equal(original, value)


@pytest.mark.parametrize("candidate", ((0., 0.), (1., 0.), (0., 1.), (.25, .25)))
def test_poor_feasible_candidates_still_enclose_true_distance(bounds, candidate):
    points = np.array([[1., .5, 1.], [5., 3., 0.], [-1., -.5, .25]])
    triangle = _base_arrays()[1][0]
    triangles = np.repeat(triangle[None, :, :], 3, axis=0)
    barycentric = np.repeat(np.array(candidate)[None, :], 3, axis=0)
    oracle = _certified(bounds.enclose_triangle_distances(points, triangles, barycentric), points, triangles, barycentric)
    assert any(item[4] > item[1] for item in oracle)


def test_exact_projection_oracle_covers_seven_finite_regions(bounds):
    points = np.array([[1., .5, 1.], [1., -1., .5], [-1., .5, .5], [3., 1.5, .5],
                       [-1., -1., .5], [5., -.5, .5], [-.5, 3., .5]])
    triangles = np.repeat(_base_arrays()[1], len(points), axis=0)
    expected_support_sizes = [3, 2, 2, 2, 1, 1, 1]
    coefficients = []
    for point, triangle, size in zip(points, triangles, expected_support_sizes):
        _, _, weights = _exact_projection(point, triangle)
        assert sum(weight > 0 for weight in weights) == size
        alpha, beta = float(weights[1]), float(weights[2])
        # Test-side exact-feasibility preparation, not the helper's TwoSum path.
        for _ in range(2):
            if _fraction(alpha)+_fraction(beta) <= 1:
                break
            if alpha >= beta:
                alpha = math.nextafter(alpha, 0.)
            else:
                beta = math.nextafter(beta, 0.)
        assert _fraction(alpha)+_fraction(beta) <= 1
        coefficients.append([alpha, beta])
    barycentric = np.array(coefficients)
    _certified(bounds.enclose_triangle_distances(points, triangles, barycentric), points, triangles, barycentric)


@pytest.mark.parametrize("triangle", (
    [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]],
    [[0., 0., 0.], [0., 0., 0.], [4., 0., 0.]],
    [[0., 0., 0.], [4., 0., 0.], [2., 2.**-40, 0.]],
))
def test_degenerate_and_sliver_convex_hulls_are_valid_for_the_bound_primitive(bounds, triangle):
    points, barycentric = np.array([[1., .5, 1.]]), np.array([[.25, .25]])
    triangles = np.array([triangle])
    _certified(bounds.enclose_triangle_distances(points, triangles, barycentric), points, triangles, barycentric)


def test_world_point_rounding_is_separate_from_candidate_distance(bounds):
    offset = np.array([2.**40, -2.**40, 2.**40])
    triangle = np.array([[0., 0., 0.], [2., 1., -2.], [2., -2., 1.]])+offset
    triangles = triangle[None, :, :]
    barycentric = np.array([[.1, .2]])
    points = np.array([[.75, -.25, .5]])+offset
    result = bounds.enclose_triangle_distances(points, triangles, barycentric)
    oracle = _certified(result, points, triangles, barycentric)
    q = oracle[0][3]
    assert any(_fraction(value) != exact for value, exact in zip(result["closest_points"][0], q))
    assert result["closest_point_error"][0] > 0
    candidate, tolerance = _fraction(result["candidate_distance"][0]), Fraction(1, 2**38)
    assert max(Fraction(0), candidate-tolerance)**2 <= oracle[0][4] <= (candidate+tolerance)**2


def test_one_call_preserves_independent_mixed_scale_pair_enclosures(bounds):
    points, triangles, barycentric = [], [], []
    for power in (-150, 0, 150):
        scale = math.ldexp(1., power)
        offset = np.array([2.**40, -2.**40, 2.**40])
        for candidate in ((0., 0.), (1., 0.), (0., 1.), (.25, .25)):
            points.append((np.array([1., .5, 1.])+offset)*scale)
            triangles.append((_base_arrays()[1][0]+offset)*scale)
            barycentric.append(candidate)
    arrays = tuple(np.array(value) for value in (points, triangles, barycentric))
    _certified(bounds.enclose_triangle_distances(*arrays), *arrays)


@pytest.mark.parametrize("max_pairs", (0, 65536))
def test_empty_batch_is_complete_even_with_zero_pair_budget(bounds, max_pairs):
    arrays = (np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 2)))
    result = bounds.enclose_triangle_distances(*arrays, max_pairs=max_pairs)
    _certified(result, *arrays)


@pytest.mark.parametrize("argument", (0, 1, 2))
@pytest.mark.parametrize("kind", ("list", "float32", "swapped", "subclass"))
def test_only_exact_native_float64_ndarrays_are_admitted(bounds, argument, kind):
    arrays = list(_base_arrays())
    value = arrays[argument]
    if kind == "list":
        arrays[argument] = value.tolist()
    elif kind == "float32":
        arrays[argument] = value.astype(np.float32)
    elif kind == "swapped":
        arrays[argument] = value.astype(np.dtype(np.float64).newbyteorder("S"))
    else:
        class Subclass(np.ndarray):
            pass
        arrays[argument] = value.view(Subclass)
    _cleared(bounds.enclose_triangle_distances(*arrays))


@pytest.mark.parametrize("fault", ("point_width", "triangle_width", "barycentric_width", "triangle_count", "barycentric_count"))
def test_shape_and_pair_alignment_failure_clears_every_array(bounds, fault):
    arrays = list(_base_arrays())
    if fault == "point_width":
        arrays[0] = np.zeros((1, 2))
    elif fault == "triangle_width":
        arrays[1] = np.zeros((1, 3, 2))
    elif fault == "barycentric_width":
        arrays[2] = np.zeros((1, 3))
    elif fault == "triangle_count":
        arrays[1] = np.repeat(arrays[1], 2, axis=0)
    else:
        arrays[2] = np.repeat(arrays[2], 2, axis=0)
    _cleared(bounds.enclose_triangle_distances(*arrays))


@pytest.mark.parametrize("argument", (0, 1, 2))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf"), 1.01e100, math.ulp(0.)))
def test_late_nonfinite_out_of_range_or_original_subnormal_clears_whole_batch(bounds, argument, value):
    arrays = [np.repeat(array, 2, axis=0) for array in _base_arrays()]
    arrays[argument][1].flat[0] = value
    _cleared(bounds.enclose_triangle_distances(*arrays))


@pytest.mark.parametrize("candidate", ((-2.**-52, 0.), (1.+2.**-52, 0.), (.5, math.nextafter(.5, 1.)), (1., 2.**-53)))
def test_exact_barycentric_infeasibility_is_not_hidden_by_rounded_sum(bounds, candidate):
    points, triangles, barycentric = _base_arrays()
    barycentric[0] = candidate
    alpha, beta = (_fraction(value) for value in candidate)
    assert alpha < 0 or beta < 0 or alpha+beta > 1
    if candidate[0] >= 0 and candidate[0] <= 1:
        assert candidate[0]+candidate[1] == 1.  # Both positive-residual traps round to one.
    _cleared(bounds.enclose_triangle_distances(points, triangles, barycentric))


@pytest.mark.parametrize("max_pairs", (None, True, -1, 65537, 1.5))
def test_invalid_or_raised_pair_budget_is_refused(bounds, max_pairs):
    _cleared(bounds.enclose_triangle_distances(*_base_arrays(), max_pairs=max_pairs))


def test_pair_budget_overflow_clears_all_rows(bounds):
    arrays = tuple(np.repeat(value, 2, axis=0) for value in _base_arrays())
    _cleared(bounds.enclose_triangle_distances(*arrays, max_pairs=1))


def test_late_arithmetic_failure_retains_attempted_count_and_clears_all_enclosures(bounds, monkeypatch):
    original, calls = bounds._sqrt_nonnegative, []
    def fail_at_restoration(value):
        calls.append(1)
        if len(calls) == 2:
            raise FloatingPointError("Invented failure after distance enclosures, before publication")
        return original(value)
    monkeypatch.setattr(bounds, "_sqrt_nonnegative", fail_at_restoration)
    arrays = tuple(np.repeat(value, 2, axis=0) for value in _base_arrays())
    result = bounds.enclose_triangle_distances(*arrays)
    _cleared(result)
    assert len(calls) == 2 and result["diagnostics"]["attempted_pairs"] == 2


@pytest.mark.parametrize("case", (
    "zero_vertex", "zero_interior", "zero_translated_point_hull", "normal_inputs_subnormal_differences",
    "coordinate_limit", "obtuse_optimal", "moderately_thin_optimal",
))
def test_additional_zero_underflow_range_and_shape_certificate_branches(bounds, case):
    points, triangles, barycentric = _base_arrays()
    if case == "zero_vertex":
        points[0] = triangles[0, 0]
        barycentric[0] = [0., 0.]
    elif case == "zero_interior":
        points[0] = [1., .5, 0.]
    elif case == "zero_translated_point_hull":
        world = np.array([2.**40, -2.**40, 2.**40])
        triangles[0] = world
        points[0] = world
    elif case == "normal_inputs_subnormal_differences":
        tiny = np.finfo(np.float64).tiny
        adjacent = np.nextafter(tiny, np.inf)
        triangles[0] = [[tiny, tiny, tiny], [adjacent, tiny, tiny], [tiny, adjacent, tiny]]
        points[0] = [adjacent, adjacent, adjacent]
        barycentric[0] = [.5, .5]
        # Every original coordinate is NORMAL. Only the local represented
        # differences are subnormal; this is not the forbidden original-input case.
        assert np.all(triangles >= tiny) and np.all(points >= tiny)
        differences = triangles[0]-triangles[0, 0]
        assert np.any((np.abs(differences) > 0) & (np.abs(differences) < tiny))
    elif case == "coordinate_limit":
        triangles[0] = [[1e100, 0., 0.], [1e100, 1e100, 0.], [1e100, 0., 1e100]]
        points[0] = [0., 0., 0.]
        barycentric[0] = [0., 0.]
        assert np.max(np.abs(triangles)) == 1e100
    elif case == "obtuse_optimal":
        triangles[0] = [[0., 0., 0.], [4., 0., 0.], [-1., 1., 0.]]
        points[0] = [.75, .25, 1.]
    else:
        triangles[0] = [[0., 0., 0.], [4., 0., 0.], [2., 2.**-20, 0.]]
        points[0] = [1.5, 2.**-22, 1.]
    result = bounds.enclose_triangle_distances(points, triangles, barycentric)
    oracle = _certified(result, points, triangles, barycentric)
    optimal, distance_squared, _, candidate, candidate_squared = oracle[0]
    assert optimal == candidate and distance_squared == candidate_squared
    if case.startswith("zero_"):
        assert distance_squared == 0
    else:
        assert distance_squared > 0
