"""Source/synthetic contract checks; no test opens CAD, scans or truth.

The explicitly named native fixture helper at the end is NOT a pytest test and
is never called by this suite. An independently bounded native harness may call
it after auditing its imports and authority. OCP interactions in pytest tests use
fake modules only; importing this test file does not import OCP.
"""

import hashlib
import json
import sys
import types
from fractions import Fraction

import numpy as np
import pytest

from backend import cad_guided_cad_sampling_v1 as sampling


def _triangles():
    return np.array([[[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]],
                     [[0., 0., 2.], [6., 0., 2.], [0., 1., 2.]]])


def _sample(triangles=None, *, faces=None, ids=None, count=40):
    if triangles is None:
        triangles = _triangles()
    n = len(triangles)
    return sampling.sample_triangle_interiors_v1(
        triangles, source_face_indices=np.arange(n, dtype=np.int64) if faces is None else faces,
        source_triangle_indices=np.ones(n, dtype=np.int64) if ids is None else ids,
        sample_count=count)


def _identity_location():
    return np.column_stack((np.eye(3), np.zeros(3)))


def _assert_unavailable(result, reason):
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == [reason]
    assert result["points"] is None
    assert result["normals"] is None
    assert result["production_use_allowed"] is False


def test_area_allocation_interior_normals_and_source_lineage():
    triangles = _triangles()
    result = _sample(triangles, faces=np.array([7, 19]), ids=np.array([3, 8]))
    assert result["status"] == "AVAILABLE"
    assert result["reasons"] == []
    rows = result["sample_triangle_rows"]
    assert np.bincount(rows).tolist() == [10, 30]
    assert np.all(rows[:-1] <= rows[1:])
    assert np.array_equal(result["sample_source_face_indices"], np.array([7, 19])[rows])
    assert np.array_equal(result["sample_source_triangle_indices"], np.array([3, 8])[rows])
    weights = result["sample_barycentric_weights"]
    assert np.all((weights > 0) & (weights < 1))
    assert np.allclose(weights.sum(axis=1), 1, rtol=0, atol=4e-16)
    chosen = triangles[rows]
    affine_points = chosen[:, 0] + (weights[:, 1, None] * (chosen[:, 1] - chosen[:, 0])
                                   + weights[:, 2, None] * (chosen[:, 2] - chosen[:, 0]))
    assert np.array_equal(result["points"], affine_points)
    assert np.array_equal(weights[:, 0], 1. - (weights[:, 1] + weights[:, 2]))
    assert not np.any(np.all(result["points"][:, None, :] == triangles[rows], axis=2))
    assert np.array_equal(result["normals"], np.tile([0., 0., 1.], (40, 1)))
    coverage = result["coverage"]
    assert coverage["total_triangulated_area_coordinate_units_squared"] == 4.
    assert coverage["area_by_source_face"] == {7: 1., 19: 3.}
    assert coverage["sample_count_by_source_face"] == {7: 10, 19: 30}
    assert coverage["every_face_sampled"] is True
    assert coverage["sampled_triangle_count"] == 2
    assert coverage["unsampled_triangle_count"] == 0
    assert coverage["cdf_lost_increment_triangle_count"] == 0
    assert coverage["continuous_surface_coverage"] == "NOT_ESTABLISHED"
    assert result["native_surface_point_status"] == "NOT_CERTIFIED"
    assert result["production_use_allowed"] is False
    assert result["sampling_policy"]["radical_inverse_evaluation"] == "EXACT_INT64_DIGIT_REVERSAL_SINGLE_FLOAT64_DIVISION"
    assert result["sampling_policy"]["position_evaluation"] == "ANCHOR_PLUS_SUM_OF_WEIGHTED_EDGE_DIFFERENCES"


def test_area_allocations_at_many_sample_counts():
    for count in range(36, 97):
        result = _sample(count=count)
        assert result["status"] == "AVAILABLE"
        allocations = np.bincount(result["sample_triangle_rows"], minlength=2)
        assert allocations.sum() == count
        assert np.all(np.abs(allocations - count * np.array([.25, .75])) <= 1.)


def test_fixed_radical_inverse_sequence_and_sqrt_warp():
    ids = np.arange(1, 7, dtype=np.int64)
    assert np.array_equal(sampling._radical_inverse_positive(ids, 2), [.5, .25, .75, .125, .625, .375])
    assert np.array_equal(sampling._radical_inverse_positive(ids, 3), [1/3, 2/3, 1/9, 4/9, 7/9, 2/9])
    result = _sample()
    root = np.sqrt(.5)
    first, second = root * (1. - 1/3), root * (1/3)
    assert np.array_equal(result["sample_barycentric_weights"][0], [1-(first+second), first, second])
    assert np.array_equal(ids, np.arange(1, 7, dtype=np.int64))


@pytest.mark.parametrize("base", [2, 3])
def test_radical_inverse_matches_independent_exact_rational_oracle(base):
    indices = set(range(1, 25)) | {99_999, 100_000}
    power = base
    while power <= 100_000:
        indices.update((power-1, power, power+1))
        power *= base
    ids = np.array(sorted(index for index in indices if 1 <= index <= 100_000), dtype=np.int64)
    expected = []
    for index in ids.tolist():
        rational, place = Fraction(0), 1
        while index:
            index, digit = divmod(index, base)
            rational += Fraction(digit, base**place)
            place += 1
        expected.append(float(rational))
    assert np.array_equal(sampling._radical_inverse_positive(ids, base), expected)


@pytest.mark.parametrize("base", [True, 1, 4, 2., np.int64(2)])
def test_radical_inverse_rejects_unsupported_base(base):
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_RADICAL_INVERSE_BASE"):
        sampling._radical_inverse_positive(np.array([1]), base)


@pytest.mark.parametrize("indices", [np.array([0]), np.array([100_001]), np.array([1.]), np.array([True]),
                                      np.array([], dtype=int), np.array([[1]]),
                                      np.array([2**64-1], dtype=np.uint64)])
def test_radical_inverse_checks_original_index_range(indices):
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_RADICAL_INVERSE_INDICES"):
        sampling._radical_inverse_positive(indices, 3)


@pytest.mark.parametrize("constant", [-1e6, -30., 0., 30., 1e6])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_constant_coordinate_planes_are_preserved_exactly(constant, axis):
    triangle = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]]])
    triangle = triangle[:, :, np.roll(np.arange(3), axis-2)]
    triangle[:, :, axis] = constant
    result = _sample(triangle)
    assert result["status"] == "AVAILABLE"
    assert np.array_equal(result["points"][:, axis], np.full(40, constant))


def test_affine_evaluation_error_against_exact_rational_bound():
    # No arbitrary epsilon: compare each result to the real affine formula on
    # represented inputs using exact Fraction arithmetic and the operation-count
    # bound documented in the source. Tiny examples exercise gradual underflow;
    # this helper test does not claim their triangles have representable area.
    tiny = np.nextafter(0., 1.)
    triangle_cases = [
        [[-3.25, 1.5, 2.], [7.75, -4.5, 9.], [2.25, 8.5, -3.]],
        [[999_990., -999_990., 1e6], [999_999., -999_992., 1e6], [999_992., -999_980., 1e6]],
        [[-1e6, -1e6, -1e6], [1e6, -1e6, 1e6], [-1e6, 1e6, 1e6]],
        [[0., 0., 0.], [tiny, 2*tiny, 3*tiny], [2*tiny, tiny, 0.]],
        [[1e-200, -2e-200, 3e-200], [5e-200, 7e-200, -2e-200], [-1e-200, 3e-200, 9e-200]],
    ]
    weight_cases = np.array([[.5, .25, .25], [.25, .25, .5], [.1, .2, .7],
                             [1.-(1e-5+.99998), 1e-5, .99998]])
    triangles = np.repeat(np.asarray(triangle_cases), len(weight_cases), axis=0)
    weights = np.tile(weight_cases, (len(triangle_cases), 1))
    actual = sampling._affine_interior_points(triangles, weights)
    unit_roundoff = Fraction(1, 2**53)
    gamma4 = 4 * unit_roundoff / (1 - 4 * unit_roundoff)
    half_min_subnormal = Fraction(1, 2**1075)
    for triangle, weight, point in zip(triangles, weights, actual):
        w1, w2 = (Fraction.from_float(float(value)) for value in weight[1:])
        for axis in range(3):
            p0, p1, p2 = (Fraction.from_float(float(value)) for value in triangle[:, axis])
            exact = p0 + w1*(p1-p0) + w2*(p2-p0)
            bound = unit_roundoff*abs(p0) + gamma4*(abs(w1)*abs(p1-p0) + abs(w2)*abs(p2-p0))
            bound += 8*half_min_subnormal
            assert abs(Fraction.from_float(float(point[axis])) - exact) <= bound


def test_replay_digests_and_deep_array_immutability():
    triangles, faces, ids = _triangles(), np.array([0, 1]), np.array([1, 1])
    first = _sample(triangles, faces=faces, ids=ids)
    second = _sample(triangles.copy(), faces=faces.copy(), ids=ids.copy())
    assert first["digests"] == second["digests"]
    assert first["coverage"] == second["coverage"]
    fields = ("points", "normals", "sample_triangle_rows", "sample_source_face_indices",
              "sample_source_triangle_indices", "sample_barycentric_weights")
    for field in fields:
        assert np.array_equal(first[field], second[field])
        assert not first[field].flags.writeable
        with pytest.raises(ValueError):
            first[field].flags.writeable = True
    for key, array in (("sample_points_sha256", first["points"]),
                       ("sample_normals_sha256", first["normals"]),
                       ("sample_triangle_rows_sha256", first["sample_triangle_rows"])):
        assert first["digests"][key] == hashlib.sha256(array.tobytes(order="C")).hexdigest()
    policy = json.dumps(first["sampling_policy"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert first["digests"]["sampling_policy_sha256"] == hashlib.sha256(policy).hexdigest()
    old_points = first["points"].copy()
    triangles[:] = 99.
    faces[:] = 99
    ids[:] = 99
    assert np.array_equal(first["points"], old_points)


def test_maximum_sample_count_is_finite_and_exact():
    result = _sample(_triangles()[:1], count=100_000)
    assert result["status"] == "AVAILABLE"
    assert result["points"].shape == (100_000, 3)
    assert np.isfinite(result["points"]).all()
    assert np.all(result["sample_barycentric_weights"] > 0)
    assert result["coverage"]["sample_count"] == 100_000


def test_tiny_unsampled_face_and_lost_cdf_increment_are_disclosed():
    triangles = _triangles()
    triangles[1] = np.array([[0., 0., 2.], [1e-9, 0., 2.], [0., 1e-9, 2.]])
    result = _sample(triangles)
    assert result["status"] == "AVAILABLE"
    assert result["coverage"]["every_face_sampled"] is False
    assert result["coverage"]["unsampled_source_face_indices"] == [1]
    assert result["coverage"]["unsampled_triangle_count"] == 1
    assert result["coverage"]["cdf_lost_increment_triangle_count"] == 1
    assert result["coverage"]["area_by_source_face"][1] > 0


@pytest.mark.parametrize("scale", [1e-100, 1e-10, 1., 1e5])
def test_scale_safe_triangle_area_and_normals(scale):
    triangle = np.array([[[0., 0., 0.], [scale, 0., 0.], [0., scale, 0.]]])
    result = _sample(triangle)
    assert result["status"] == "AVAILABLE"
    assert result["coverage"]["total_triangulated_area_coordinate_units_squared"] == .5 * scale * scale
    assert np.array_equal(result["normals"], np.tile([0., 0., 1.], (40, 1)))


def test_unrepresentable_area_refuses_whole_input():
    triangle = np.array([[[0., 0., 0.], [1e-200, 0., 0.], [0., 1e-200, 0.]]])
    _assert_unavailable(_sample(triangle), "TRIANGLE_AREA_NOT_REPRESENTABLE")


def test_unsampled_degenerate_triangle_still_refuses_all_samples():
    triangles = _triangles()
    triangles[1, 2] = triangles[1, 1]
    _assert_unavailable(_sample(triangles), "DEGENERATE_TRIANGLE")


def test_positive_weights_do_not_hide_float64_vertex_collapse():
    x = 100_000.
    upper = np.nextafter(x, np.inf)
    triangle = np.array([[[x, x, x], [upper, x, x], [x, upper, x]]])
    _assert_unavailable(_sample(triangle), "BARYCENTRIC_POINT_COLLAPSED_TO_VERTEX")


@pytest.mark.parametrize("count", [True, False, 35, 100_001, 36., np.int64(36), "36", None, 10**100])
def test_sample_count_refusals(count):
    _assert_unavailable(_sample(count=count), "CAD_SAMPLE_COUNT_BUDGET")


@pytest.mark.parametrize("value", [[], np.empty((0, 3, 3)), np.zeros((1, 3)), np.zeros((1, 4, 3)),
                                   np.ones((1, 3, 3), dtype=bool), np.ones((1, 3, 3), dtype=object)])
def test_triangle_shape_dtype_refusals(value):
    result = sampling.sample_triangle_interiors_v1(value, source_face_indices=np.array([0]),
                                                   source_triangle_indices=np.array([1]), sample_count=36)
    _assert_unavailable(result, "INVALID_TRIANGLE_ARRAY")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1_000_001., -1_000_001., np.finfo(float).max])
def test_all_triangle_numerics_validated(value):
    triangles = _triangles()
    triangles[1, 2, 0] = value
    _assert_unavailable(_sample(triangles), "INVALID_TRIANGLE_NUMERICS")


@pytest.mark.parametrize("faces", [np.array([-1, 0]), np.array([0, 1024]), np.array([False, True]),
                                   np.array([0., 1.]), np.array([0]), [0, 1],
                                   np.array([0, 2**64-1], dtype=np.uint64)])
def test_original_face_id_range_and_dtype_before_cast(faces):
    _assert_unavailable(_sample(faces=faces), "INVALID_SOURCE_FACE_IDS")


@pytest.mark.parametrize("ids", [np.array([0, 1]), np.array([1, 500_001]), np.array([True, True]),
                                 np.array([1., 1.]), np.array([1]), [1, 1],
                                 np.array([1, 2**64-1], dtype=np.uint64)])
def test_original_triangle_id_range_and_dtype_before_cast(ids):
    _assert_unavailable(_sample(ids=ids), "INVALID_SOURCE_TRIANGLE_IDS")


def test_unique_composite_ids_allow_repeated_local_id_across_faces_only():
    assert _sample(ids=np.array([7, 7]), faces=np.array([0, 1]))["status"] == "AVAILABLE"
    _assert_unavailable(_sample(ids=np.array([7, 7]), faces=np.array([1, 1])), "DUPLICATE_SOURCE_TRIANGLE_ID")
    assert _sample(ids=np.array([7, 8]), faces=np.array([1, 1]))["status"] == "AVAILABLE"


def test_triangle_budget_guard_is_checked_before_processing(monkeypatch):
    monkeypatch.setattr(sampling, "MAX_TRIANGLES", 1)
    _assert_unavailable(_sample(), "INVALID_TRIANGLE_ARRAY")


def test_rigid_location_applied_once_and_reversed_face_winding():
    nodes = _triangles()[0]
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    translation = np.array([10., 20., 30.])
    location = np.column_stack((rotation, translation))
    indices = np.array([[1, 2, 3]])
    forward = sampling._oriented_transformed_triangles(nodes, indices, location, reversed_face=False)
    reverse = sampling._oriented_transformed_triangles(nodes, indices, location, reversed_face=True)
    assert np.array_equal(forward[0], nodes @ rotation.T + translation)
    assert np.array_equal(reverse, forward[:, [0, 2, 1]])
    f, r = _sample(forward), _sample(reverse)
    assert f["status"] == r["status"] == "AVAILABLE"
    assert np.array_equal(f["normals"], np.tile([1., 0., 0.], (40, 1)))
    assert np.array_equal(r["normals"], -f["normals"])
    assert np.array_equal(indices, [[1, 2, 3]])
    assert np.array_equal(nodes, _triangles()[0])


@pytest.mark.parametrize("rotation,reason", [
    (np.diag([-1., 1., 1.]), "REFLECTED_OR_SINGULAR_NATIVE_LOCATION"),
    (np.diag([0., 1., 1.]), "REFLECTED_OR_SINGULAR_NATIVE_LOCATION"),
    (np.diag([2., 2., 2.]), "NONRIGID_NATIVE_LOCATION"),
    (np.diag([2., .5, 1.]), "NONRIGID_NATIVE_LOCATION"),
    (np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]]), "NONRIGID_NATIVE_LOCATION"),
])
def test_reflection_scale_singular_and_shear_fail_closed(rotation, reason):
    with pytest.raises(sampling.CADSamplingUnavailable, match=reason):
        sampling._proper_location_matrix(np.column_stack((rotation, np.zeros(3))))


@pytest.mark.parametrize("location", [np.eye(4), np.ones((3, 4), dtype=bool),
                                      np.full((3, 4), np.nan), np.full((3, 4), np.inf)])
def test_invalid_location_structure_and_numerics(location):
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_NATIVE_LOCATION"):
        sampling._proper_location_matrix(location)


@pytest.mark.parametrize("indices,reason", [
    (np.array([[0, 1, 2]]), "INVALID_NATIVE_TRIANGLE_INDICES"),
    (np.array([[1, 2, 4]]), "INVALID_NATIVE_TRIANGLE_INDICES"),
    (np.array([[1, 2, 2]]), "DEGENERATE_NATIVE_TRIANGLE_INDICES"),
    (np.array([[1., 2., 3.]]), "INVALID_NATIVE_TRIANGLE_INDICES"),
    (np.array([[True, True, True]]), "INVALID_NATIVE_TRIANGLE_INDICES"),
    (np.array([[1, 2, 2**64-1]], dtype=np.uint64), "INVALID_NATIVE_TRIANGLE_INDICES"),
])
def test_native_local_index_validation(indices, reason):
    with pytest.raises(sampling.CADSamplingUnavailable, match=reason):
        sampling._oriented_transformed_triangles(_triangles()[0], indices, _identity_location(), reversed_face=False)


def test_all_mesh_nodes_validated_even_if_unused():
    nodes = np.vstack((_triangles()[0], [np.nan, 0, 0]))
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_MESH_NODE_NUMERICS"):
        sampling._oriented_transformed_triangles(nodes, np.array([[1, 2, 3]]), _identity_location(), reversed_face=False)


def test_invalid_orientation_and_transformed_bounds():
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_FACE_ORIENTATION"):
        sampling._oriented_transformed_triangles(_triangles()[0], np.array([[1, 2, 3]]),
                                                  _identity_location(), reversed_face=1)
    location = _identity_location()
    location[0, 3] = 1e6
    with pytest.raises(sampling.CADSamplingUnavailable, match="INVALID_TRANSFORMED_MESH_NODES"):
        sampling._oriented_transformed_triangles(_triangles()[0], np.array([[1, 2, 3]]), location, reversed_face=False)


@pytest.mark.parametrize("opt_in,binding", [(False, "ocp"), (1, "ocp"), (np.bool_(True), "ocp"),
                                            (True, "OCP"), (True, "pythonocc"), (True, None)])
def test_explicit_opt_in_gate_precedes_native_access(monkeypatch, opt_in, binding):
    def forbidden(*args):
        pytest.fail("native seam must not be reached")
    monkeypatch.setattr(sampling, "_native_triangles", forbidden)
    _assert_unavailable(sampling.sample_native_cad_surface_v1(object(), allow_ocp=opt_in, native_binding=binding),
                        "EXPLICIT_OCP_BINDING_REQUIRED")


@pytest.mark.parametrize("kwargs,reason", [
    ({"sample_count": 35}, "CAD_SAMPLE_COUNT_BUDGET"),
    ({"linear_deflection_mm": True}, "LINEAR_DEFLECTION_POLICY"),
    ({"linear_deflection_mm": 0.}, "LINEAR_DEFLECTION_POLICY"),
    ({"linear_deflection_mm": 10.01}, "LINEAR_DEFLECTION_POLICY"),
    ({"linear_deflection_mm": np.nan}, "LINEAR_DEFLECTION_POLICY"),
    ({"linear_deflection_mm": 10**1000}, "LINEAR_DEFLECTION_POLICY"),
    ({"angular_deflection_radians": 0.}, "ANGULAR_DEFLECTION_POLICY"),
    ({"angular_deflection_radians": 1.01}, "ANGULAR_DEFLECTION_POLICY"),
    ({"angular_deflection_radians": np.inf}, "ANGULAR_DEFLECTION_POLICY"),
    ({"angular_deflection_radians": np.float64(.2)}, "ANGULAR_DEFLECTION_POLICY"),
])
def test_policy_validation_precedes_native_access(monkeypatch, kwargs, reason):
    def forbidden(*args):
        pytest.fail("native seam must not be reached")
    monkeypatch.setattr(sampling, "_native_triangles", forbidden)
    _assert_unavailable(sampling.sample_native_cad_surface_v1(object(), allow_ocp=True, **kwargs), reason)


def test_native_wrapper_receipt_and_shared_points_normals(monkeypatch):
    handle, calls = object(), []
    def synthetic_native(shape, linear, angular):
        calls.append((shape, linear, angular))
        return _triangles(), np.array([0, 1]), np.array([1, 1]), {"fixture": "SYNTHETIC_NOT_NATIVE"}
    monkeypatch.setattr(sampling, "_native_triangles", synthetic_native)
    result = sampling.sample_native_cad_surface_v1(handle, allow_ocp=True, sample_count=40)
    assert calls == [(handle, .1, .2)]
    assert result["status"] == "AVAILABLE"
    assert result["schema"] == "CAD_GUIDED_NATIVE_SURFACE_SAMPLING_V1"
    assert np.array_equal(result["points"], _sample()["points"])
    assert np.array_equal(result["normals"], _sample()["normals"])
    assert result["native_facts"] == {"fixture": "SYNTHETIC_NOT_NATIVE"}
    assert result["mesher_policy"]["parallel"] is False
    assert result["mesher_policy"]["relative_deflection"] is False
    assert result["mesher_policy"]["deflection_error_is_certified"] is False
    assert result["mesher_policy"]["max_triangles"] == 500_000
    assert result["production_use_allowed"] is False
    assert result["frame_status"] == "NATIVE_MODEL_COORDINATES_NOT_SCAN_REGISTRATION"
    assert result["source_unit_status"] == "CALLER_IMPORT_CONTRACT_NOT_VERIFIED_HERE"
    assert "MUTATE" in result["cache_effect"]
    encoded = json.dumps(result["mesher_policy"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert result["digests"]["mesher_policy_sha256"] == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("error", [ImportError(), RuntimeError(), TypeError(), AttributeError(), OSError()])
def test_native_errors_are_unavailable_not_part_failures(monkeypatch, error):
    def unavailable(*args):
        raise error
    monkeypatch.setattr(sampling, "_native_triangles", unavailable)
    _assert_unavailable(sampling.sample_native_cad_surface_v1(object(), allow_ocp=True), "NATIVE_CAD_SAMPLING_UNAVAILABLE")


class _FakeLocation:
    def __init__(self, matrix=None):
        self.matrix = _identity_location() if matrix is None else matrix

    def Transformation(self):
        return self

    def Value(self, row, column):
        return self.matrix[row-1, column-1]


class _FakeShape:
    def __init__(self, *, children=(), face=False, orientation="FORWARD", location=None,
                 node_count=3, triangle_count=1, identity=None):
        self.children, self.face, self.orientation = tuple(children), face, orientation
        self.location = _FakeLocation() if location is None else location
        self.node_count, self.triangle_count = node_count, triangle_count
        self.identity = object() if identity is None else identity
        self.node_reads = 0
        self.triangle_reads = 0

    def IsNull(self):
        return False

    def Location(self):
        return self.location

    def ShapeType(self):
        return "FACE" if self.face else "COMPOUND"

    def Orientation(self):
        return self.orientation

    def Oriented(self, value):
        assert value == self.orientation
        return self

    def IsSame(self, other):
        return self.identity is other.identity


def _install_fake_ocp(monkeypatch, *, status=0, done=True, missing=False, triangle=(1, 2, 3)):
    """Synthetic native API double, never imports or executes the OCP binary."""
    calls = []

    class Iterator:
        def __init__(self, current, cumulative_orientation, cumulative_location):
            assert cumulative_orientation is cumulative_location is True
            self.children, self.index = current.children, 0

        def More(self):
            return self.index < len(self.children)

        def Value(self):
            return self.children[self.index]

        def Next(self):
            self.index += 1

    class Triangulation:
        def __init__(self, face):
            self.face = face

        def NbNodes(self):
            return self.face.node_count

        def NbTriangles(self):
            return self.face.triangle_count

        def Node(self, index):
            self.face.node_reads += 1
            point = _triangles()[0][index-1]
            return types.SimpleNamespace(X=lambda: point[0], Y=lambda: point[1], Z=lambda: point[2])

        def Triangle(self, index):
            self.face.triangle_reads += 1
            return types.SimpleNamespace(Get=lambda: triangle)

    def triangulation(face, location):
        location.matrix = face.location.matrix
        return None if missing else Triangulation(face)

    def mesher(*args):
        calls.append(args)
        return types.SimpleNamespace(IsDone=lambda: done, GetStatusFlags=lambda: status)

    modules = {
        "OCP": {},
        "OCP.BRep": {"BRep_Tool": types.SimpleNamespace(Triangulation_s=triangulation)},
        "OCP.BRepMesh": {"BRepMesh_IncrementalMesh": mesher},
        "OCP.TopAbs": {"TopAbs_FACE": "FACE", "TopAbs_FORWARD": "FORWARD", "TopAbs_REVERSED": "REVERSED"},
        "OCP.TopLoc": {"TopLoc_Location": _FakeLocation},
        "OCP.TopoDS": {"TopoDS": types.SimpleNamespace(Face_s=lambda face: face),
                       "TopoDS_Shape": _FakeShape, "TopoDS_Iterator": Iterator},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_fake_native_constructor_once_and_triangle_location_winding(monkeypatch):
    calls = _install_fake_ocp(monkeypatch)
    location = _identity_location()
    location[:, 3] = [10., 20., 30.]
    face = _FakeShape(face=True, orientation="REVERSED", location=_FakeLocation(location))
    shape = _FakeShape(children=(face,))
    result = sampling.sample_native_cad_surface_v1(shape, allow_ocp=True, sample_count=36)
    assert result["status"] == "AVAILABLE"
    assert calls == [(shape, .1, False, .2, False)]
    assert face.node_reads == 3 and face.triangle_reads == 1
    assert np.array_equal(result["normals"], np.tile([0., 0., -1.], (36, 1)))
    assert np.all(result["points"] >= [10., 20., 30.])
    assert np.array_equal(result["points"][:, 2], np.full(36, 30.))
    assert result["native_facts"]["face_records"] == [
        {"source_face_index": 0, "node_count": 3, "triangle_count": 1, "orientation": "REVERSED"}]
    assert result["native_facts"]["preexisting_triangulated_faces"] == 1


@pytest.mark.parametrize("case,reason", [
    ("face_count", "NATIVE_FACE_BUDGET"),
    ("node_count", "NATIVE_TOPOLOGY_BUDGET"),
    ("depth", "NATIVE_TOPOLOGY_BUDGET"),
    ("repeat", "REPEATED_NATIVE_FACE_OCCURRENCE"),
    ("orientation", "UNSUPPORTED_FACE_ORIENTATION"),
    ("mirror", "REFLECTED_OR_SINGULAR_NATIVE_LOCATION"),
    ("no_faces", "NO_NATIVE_FACES"),
])
def test_fake_native_preflight_refuses_before_meshing(monkeypatch, case, reason):
    calls = _install_fake_ocp(monkeypatch)
    face = _FakeShape(face=True)
    shape = _FakeShape(children=(face,))
    if case == "face_count":
        monkeypatch.setattr(sampling, "MAX_FACES", 1)
        shape = _FakeShape(children=(face, _FakeShape(face=True)))
    elif case == "node_count":
        monkeypatch.setattr(sampling, "MAX_TOPOLOGY_NODES", 1)
    elif case == "depth":
        monkeypatch.setattr(sampling, "MAX_TOPOLOGY_DEPTH", 0)
    elif case == "repeat":
        shape = _FakeShape(children=(face, face))
    elif case == "orientation":
        face.orientation = "INTERNAL"
    elif case == "mirror":
        shape.location.matrix[0, 0] = -1.
    else:
        shape = _FakeShape()
    _assert_unavailable(sampling.sample_native_cad_surface_v1(shape, allow_ocp=True), reason)
    assert calls == []
    assert face.node_reads == face.triangle_reads == 0


@pytest.mark.parametrize("node_count,triangle_count", [(1_000_001, 1), (3, 500_001)])
def test_fake_native_streaming_mesh_caps_before_array_reads(monkeypatch, node_count, triangle_count):
    calls = _install_fake_ocp(monkeypatch)
    face = _FakeShape(face=True, node_count=node_count, triangle_count=triangle_count)
    result = sampling.sample_native_cad_surface_v1(face, allow_ocp=True)
    _assert_unavailable(result, "NATIVE_MESH_CARDINALITY_BUDGET")
    assert len(calls) == 1
    assert face.node_reads == face.triangle_reads == 0


def test_fake_native_mesh_caps_are_aggregate_not_per_face(monkeypatch):
    _install_fake_ocp(monkeypatch)
    monkeypatch.setattr(sampling, "MAX_MESH_NODES", 5)
    first, second = _FakeShape(face=True), _FakeShape(face=True)
    result = sampling.sample_native_cad_surface_v1(_FakeShape(children=(first, second)), allow_ocp=True)
    _assert_unavailable(result, "NATIVE_MESH_CARDINALITY_BUDGET")
    assert first.node_reads == 3
    assert second.node_reads == second.triangle_reads == 0


@pytest.mark.parametrize("options,reason", [
    ({"done": False}, "NATIVE_MESHING_NOT_DONE"),
    ({"status": 1}, "NATIVE_MESHING_STATUS_FLAGS"),
    ({"status": True}, "NATIVE_MESHING_STATUS_FLAGS"),
    ({"missing": True}, "NATIVE_FACE_TRIANGULATION_MISSING"),
    ({"triangle": (1, 2, 4)}, "INVALID_NATIVE_TRIANGLE_INDICES"),
    ({"triangle": (True, 2, 3)}, "INVALID_NATIVE_TRIANGLE_INDICES"),
    ({"triangle": (1, 2, 2)}, "DEGENERATE_NATIVE_TRIANGLE_INDICES"),
])
def test_fake_native_failures_do_not_publish_partial_points(monkeypatch, options, reason):
    _install_fake_ocp(monkeypatch, **options)
    _assert_unavailable(sampling.sample_native_cad_surface_v1(_FakeShape(face=True), allow_ocp=True), reason)


@pytest.mark.parametrize("flags", [0, 128])
def test_only_no_error_or_informational_native_cache_reuse_is_admitted(monkeypatch, flags):
    calls = _install_fake_ocp(monkeypatch, status=flags)
    result = sampling.sample_native_cad_surface_v1(_FakeShape(face=True), allow_ocp=True, sample_count=36)
    assert result["status"] == "AVAILABLE"
    assert len(calls) == 1
    assert result["native_facts"]["mesher_status_flags"] == flags
    assert result["native_facts"]["mesher_cache_reused"] is (flags == 128)
    assert result["mesher_policy"]["deflection_error_is_certified"] is False


@pytest.mark.parametrize("flags", [1, 2, 4, 8, 16, 32, 64, 256, 512, 1024,
                                    129, 130, 132, 136, 144, 160, 192, 384, 640,
                                    -1, -128, True, 128.])
def test_cache_bit_never_masks_native_problem_or_unknown_status(monkeypatch, flags):
    _install_fake_ocp(monkeypatch, status=flags)
    result = sampling.sample_native_cad_surface_v1(_FakeShape(face=True), allow_ocp=True, sample_count=36)
    _assert_unavailable(result, "NATIVE_MESHING_STATUS_FLAGS")


def constructed_native_sampling_fixture_shapes_v1():
    """Return source-created native shapes ONLY when a separate harness calls it.

    This is not a pytest test/fixture and does not run during collection. No file
    access, real-part geometry, labels, references or benchmark inputs are used.
    The harness owns process wall/CPU/RSS limits and must record native versions.
    Expected box area is 2*(10*20+10*30+20*30)=2200 square coordinate units;
    cylinder triangulated area is only approximate, never exact-surface evidence.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    box = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
    reversed_box = box.Reversed()
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0., 0., 0.), gp_Dir(1., 2., 3.)), .37)
    transform.SetTranslationPart(gp_Vec(40., -25., 17.))
    moved_box = box.Moved(TopLoc_Location(transform))
    cylinder = BRepPrimAPI_MakeCylinder(7., 25.).Shape()
    return {"box": box, "reversed_box": reversed_box, "proper_rigid_moved_box": moved_box,
            "cylinder": cylinder}


def run_constructed_native_sampling_checks_v1():
    """Fixed four-case native API check, called only by its bounded harness.

    Analytic distance checks qualify these invented shapes, not an arbitrary
    STEP mesh or a certified mesher error bound. Every case remains in report.
    """
    shapes = constructed_native_sampling_fixture_shapes_v1()
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    rotation = np.eye(3) + np.sin(.37)*cross + (1-np.cos(.37))*(cross@cross)
    records = []
    for name, shape in shapes.items():
        row = dict(case=name, passed=False)
        try:
            result = sampling.sample_native_cad_surface_v1(shape, allow_ocp=True, sample_count=2048)
            row.update(status=result["status"], reasons=result["reasons"],
                       native_mesher_status_flags=result.get("native_mesher_status_flags"))
            assert result["status"] == "AVAILABLE", result["reasons"]
            p, n = result["points"], result["normals"]
            assert p.shape == n.shape == (2048, 3)
            assert np.all(np.isfinite(p)) and np.all(np.isfinite(n))
            assert np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-12, rtol=0)
            if name == "proper_rigid_moved_box":
                p = (p-np.array([40., -25., 17.])) @ rotation
                n = n @ rotation
            if name != "cylinder":
                size = np.array([10., 20., 30.])
                assert np.all(p >= -1e-12) and np.all(p <= size+1e-12)
                distance = np.minimum(np.abs(p), np.abs(p-size)).min(axis=1)
                assert float(distance.max()) <= 1e-12
                dot = np.einsum("ij,ij->i", p-size/2, n)
                assert np.all(dot < 0) if name == "reversed_box" else np.all(dot > 0)
                row["maximum_analytic_surface_distance_mm"] = float(distance.max())
            else:
                # Chordal cylinder points lie inside the exact analytic wall;
                # caps lie on z=0 or25. This fixture's declared check is0.1mm.
                radius = np.linalg.norm(p[:, :2], axis=1)
                distance = np.minimum(np.abs(radius-7.), np.minimum(np.abs(p[:, 2]), np.abs(p[:, 2]-25.)))
                assert np.all(radius <= 7.+1e-12)
                assert np.all(p[:, 2] >= -1e-12) and np.all(p[:, 2] <= 25.+1e-12)
                assert float(distance.max()) <= .1
                assert np.all(np.einsum("ij,ij->i", p-[0., 0., 12.5], n) > 0)
                row["maximum_analytic_surface_distance_mm"] = float(distance.max())
            row.update(passed=True, native_facts=result["native_facts"],
                       digests=result["digests"], calibration_claim=False)
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error)[:300])
        records.append(row)
    return dict(scope="FOUR_INVENTED_NATIVE_CAD_SAMPLING_CASES", expected=4,
                passed=sum(row["passed"] for row in records), records=records,
                physical_generalization_claim=False)
