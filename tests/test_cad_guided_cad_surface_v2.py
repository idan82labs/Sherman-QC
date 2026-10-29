"""Pure invented geometry and mocked native seam; no OCP/raw/Gold access."""

import hashlib

import numpy as np
import pytest

from backend import cad_guided_cad_sampling_v1 as v1
from backend import cad_guided_cad_surface_v2 as v2


def _geometry():
    return (np.array([[[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]],
                     [[0., 0., 2.], [0., 1., 2.], [6., 0., 2.]]]),
            np.array([7, 19]), np.array([3, 8]))


def _prepare(triangles=None, faces=None, ids=None, count=40):
    original = _geometry()
    return v2.prepare_cad_triangle_surface_v2(
        original[0] if triangles is None else triangles,
        source_face_indices=original[1] if faces is None else faces,
        source_triangle_indices=original[2] if ids is None else ids, sample_count=count)


def _refused(result):
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"]
    assert all(result[key] is None for key in v2._ARRAY_KEYS)
    assert result["production_use_allowed"] is False


def test_exact_v1_samples_and_retained_finite_geometry():
    triangles, faces, ids = _geometry()
    expected = v1.sample_triangle_interiors_v1(triangles, source_face_indices=faces,
        source_triangle_indices=ids, sample_count=40)
    actual = _prepare(triangles, faces, ids)
    assert actual["status"] == "AVAILABLE"
    assert actual["schema"] == "CAD_GUIDED_FINITE_TRIANGLE_SURFACE_V2"
    for key in v2._ARRAY_KEYS:
        if key in expected:
            assert np.array_equal(actual[key], expected[key]), key
    assert np.array_equal(actual["triangles"], triangles)
    assert np.array_equal(actual["source_face_indices"], faces)
    assert np.array_equal(actual["source_triangle_indices"], ids)
    for key in ("sampling_policy", "coverage", "digests"):
        assert actual[key] == expected[key]
    assert np.array_equal(actual["normals"][:10], np.tile([0., 0., 1.], (10, 1)))
    assert np.array_equal(actual["normals"][10:], np.tile([0., 0., -1.], (30, 1)))
    assert actual["native_surface_distance_status"] == "NOT_ANALYTIC_OR_CERTIFIED"
    assert actual["production_use_allowed"] is False


def test_bytes_backed_ownership_and_digests_survive_caller_mutation():
    triangles, faces, ids = _geometry()
    result = _prepare(triangles, faces, ids)
    triangles[:] = 99
    faces[:] = 1
    ids[:] = 1
    for key in v2._ARRAY_KEYS:
        with pytest.raises(ValueError):
            result[key].setflags(write=True)
    for key, digest_key in (("triangles", "oriented_triangle_xyz_sha256"),
                            ("source_face_indices", "source_face_ids_sha256"),
                            ("source_triangle_indices", "source_triangle_ids_sha256"),
                            ("points", "sample_points_sha256"), ("normals", "sample_normals_sha256")):
        assert hashlib.sha256(result[key].tobytes()).hexdigest() == result["digests"][digest_key]
    assert np.array_equal(result["triangles"], _geometry()[0])


@pytest.mark.parametrize("value", [True, 35, 100001, 40., np.int64(40), None])
def test_invalid_sample_count_refuses_before_native(value, monkeypatch):
    monkeypatch.setattr(v1, "_native_triangles", lambda *args: pytest.fail("native invoked"))
    _refused(v2.sample_native_cad_surface_v2(object(), allow_ocp=True, sample_count=value))


@pytest.mark.parametrize("values", [[], np.empty((0, 3, 3)), np.zeros((2, 3, 3)),
                                    np.full((2, 3, 3), np.nan), np.ones((2, 3)),
                                    np.ones((2, 3, 3), dtype=bool)])
def test_bad_triangle_refuses_whole_export(values):
    _refused(_prepare(triangles=values))


@pytest.mark.parametrize("faces,ids", [(np.array([-1, 0]), np.array([1, 2])),
    (np.array([1024, 0]), np.array([1, 2])), (np.array([0, 0]), np.array([1, 1])),
    (np.array([0., 1.]), np.array([1, 2])), (np.array([0, 1]), np.array([0, 2])),
    (np.array([0, 1]), np.array([1, 2**64-1], dtype=np.uint64))])
def test_invalid_source_identity_refuses_whole_export(faces, ids):
    _refused(_prepare(faces=faces, ids=ids))


@pytest.mark.parametrize("kwargs", [{}, {"allow_ocp": 1}, {"allow_ocp": True, "native_binding": "occ"},
    {"allow_ocp": True, "linear_deflection_mm": 0.},
    {"allow_ocp": True, "angular_deflection_radians": float("nan")}])
def test_native_opt_in_and_policy_are_checked_first(kwargs, monkeypatch):
    monkeypatch.setattr(v1, "_native_triangles", lambda *args: pytest.fail("native invoked"))
    _refused(v2.sample_native_cad_surface_v2(object(), **kwargs))


def test_native_export_calls_same_seam_once_and_same_policy_as_v1(monkeypatch):
    calls = []
    shape = object()
    def seam(given, linear, angular):
        calls.append((given, linear, angular))
        return (*_geometry(), {"mesher_status_flags": 128, "triangle_count": 2})
    monkeypatch.setattr(v1, "_native_triangles", seam)
    result = v2.sample_native_cad_surface_v2(shape, allow_ocp=True, sample_count=40)
    assert calls == [(shape, .1, .2)]
    assert result["status"] == "AVAILABLE"
    expected = v1.sample_native_cad_surface_v1(shape, allow_ocp=True, sample_count=40)
    assert len(calls) == 2  # One independent mocked seam call for each API.
    for key in ("mesher_policy", "native_facts", "sampling_policy", "coverage", "digests"):
        assert result[key] == expected[key], key
    assert np.array_equal(result["points"], expected["points"])


def test_native_failure_is_single_terminal_no_partial_surface(monkeypatch):
    calls = []
    def seam(*args):
        calls.append(args)
        error = v1.CADSamplingUnavailable("NATIVE_MESHING_STATUS_FLAGS")
        error.native_mesher_status_flags = 8
        raise error
    monkeypatch.setattr(v1, "_native_triangles", seam)
    result = v2.sample_native_cad_surface_v2(object(), allow_ocp=True)
    assert len(calls) == 1
    assert result["reasons"] == ["NATIVE_MESHING_STATUS_FLAGS"]
    assert result["native_mesher_status_flags"] == 8
    _refused(result)


def test_second_bad_triangle_does_not_leave_first_triangle_prefix(monkeypatch):
    triangles, faces, ids = _geometry()
    triangles[1] = 0
    monkeypatch.setattr(v1, "_native_triangles", lambda *args: (triangles, faces, ids, {}))
    _refused(v2.sample_native_cad_surface_v2(object(), allow_ocp=True, sample_count=40))


@pytest.mark.parametrize("key", ["oriented_triangle_xyz_sha256", "source_face_ids_sha256", "source_triangle_ids_sha256"])
def test_inconsistent_sample_binding_clears_all_surface_arrays(key, monkeypatch):
    original = v1.sample_triangle_interiors_v1
    def corrupt(*args, **kwargs):
        sampled = original(*args, **kwargs)
        return {**sampled, "digests": {**sampled["digests"], key: "0" * 64}}
    monkeypatch.setattr(v1, "sample_triangle_interiors_v1", corrupt)
    result = _prepare()
    assert result["reasons"] == ["TRIANGLE_SAMPLE_BINDING_MISMATCH"]
    _refused(result)
