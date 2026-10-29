"""Independent near-roundoff mesh-patch controls, invented arrays only.

No native importer, PLY file, registration, physical scan, reference labels or
retained run is used. Rectangle/hole moments are analytic area integrals, not
an implementation-generated oracle. This is not noisy-scanner qualification.
Root must independently review this file before its first supervised execution.
"""

from dataclasses import fields, replace
import hashlib
import math

import numpy as np
import pytest

from backend import cad_guided_mesh_plane_proposals_v4 as v4


RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
                       process_tree_rss_bytes=1024**3)


def _grid(xs=(0., 1., 2., 3.), ys=(0., 1., 2.), *, height=None, omit_cells=()):
    height = height or (lambda x, y: 0.)
    points = np.asarray([[x, y, height(x, y)] for y in ys for x in xs], dtype=np.float64)
    triangles = []
    for iy in range(len(ys)-1):
        for ix in range(len(xs)-1):
            if (ix, iy) in omit_cells:
                continue
            a = iy*len(xs)+ix
            triangles.extend(((a, a+1, a+len(xs)+1), (a, a+len(xs)+1, a+len(xs))))
    return points, np.asarray(triangles, dtype=np.int64)


def _scan(points, triangles, **kwargs):
    return v4.extract_coplanar_scan_patches_v4(points, triangles, **kwargs)


def _cad(points, triangles, *, face_ids=None, local_ids=None, **kwargs):
    count = len(triangles)
    if face_ids is None:
        face_ids = np.full(count, 71, dtype=np.int64)
    if local_ids is None:
        local_ids = np.arange(1, count+1, dtype=np.int64)
    return v4.fit_cad_face_planes_v4(points[triangles], source_face_indices=face_ids,
                                    source_triangle_indices=local_ids, **kwargs)


def _immutable(array):
    assert array.dtype == np.dtype("<i8") and not array.flags.writeable
    base = array
    while type(base) is np.ndarray:
        with pytest.raises(ValueError):
            base.setflags(write=True)
        base = base.base
    assert type(base) is bytes


def _available(result, patches, *, components=None):
    assert result["status"] == "AVAILABLE", result
    assert result["reasons"] == []
    assert len(result["patches"]) == patches
    statistics = result["statistics"]
    assert statistics["component_search_complete"] is True
    assert statistics["qualified_patch_count"] == patches
    assert statistics["components_evaluated"] == statistics["component_count"]
    assert statistics["qualified_patch_count"] + statistics["rejected_component_count"] == statistics["component_count"]
    assert len(result["components"]) == statistics["component_count"]
    if components is not None:
        assert len(result["components"]) == components
    _immutable(result["triangle_component_labels"])
    assert result["completeness_for_noisy_planes"] is False
    assert result["accepted_registration"] is False
    assert result["production_use_allowed"] is False
    assert result["material_topology_verified"] is False
    assert result["finite_support_representation"] == "ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON"
    for patch in result["patches"]:
        _immutable(patch["triangle_rows"])
        assert patch["disposition"] == "PROPOSED" and patch["reasons"] == []
        assert patch["plane_fit_measure"] == "UNIFORM_TRIANGLE_AREA_TLS"
        assert patch["area_measure"] == "TESSELLATION_AREA_WITH_MULTIPLICITY_NOT_MATERIAL_AREA"
        assert np.linalg.norm(patch["normal"]) == pytest.approx(1., abs=1e-12)
        assert patch["surface_area_mm2"] > 0.
        assert np.isfinite(patch["area_centroid_mm"]).all()
        assert np.isfinite(patch["surface_covariance_mm2"]).all()
    return result["patches"]


def _refused(result, reason=None):
    assert result["status"] == "UNAVAILABLE", result
    assert result["reasons"]
    if reason is not None:
        assert result["reasons"] == [reason]
    assert result["patches"] == []
    assert result["components"] == []
    assert result["triangle_component_labels"] is None
    assert result["statistics"]["component_search_complete"] is False
    assert result["production_use_allowed"] is False


def _rectangle_moments(patch, *, scale=1., shift=np.zeros(3), rotation=np.eye(3)):
    expected_center = np.array([1.5, 1., 0.])*scale @ rotation.T + shift
    expected_covariance = rotation @ np.diag([.75, 1./3., 0.]) @ rotation.T * scale**2
    np.testing.assert_allclose(patch["area_centroid_mm"], expected_center, rtol=0., atol=2e-10)
    np.testing.assert_allclose(patch["surface_covariance_mm2"], expected_covariance, rtol=0., atol=2e-10)
    assert patch["surface_area_mm2"] == pytest.approx(6.*scale**2, rel=2e-12, abs=1e-12)


@pytest.mark.parametrize("kind", ("scan", "cad"))
@pytest.mark.parametrize("nonuniform", (False, True))
@pytest.mark.parametrize("reverse_winding", (False, True))
def test_rectangle_area_moments_ignore_vertex_density_and_triangle_winding(kind, nonuniform, reverse_winding):
    points, triangles = _grid((0., .2, .8, 3.) if nonuniform else (0., 1., 2., 3.),
                              (0., .3, 2.) if nonuniform else (0., 1., 2.))
    if reverse_winding:
        triangles = triangles[:, [0, 2, 1]].copy()
    result = _scan(points, triangles) if kind == "scan" else _cad(points, triangles)
    patch, = _available(result, 1, components=1)
    _rectangle_moments(patch)
    np.testing.assert_allclose(patch["support_extent_mm"], [3., 2.], rtol=0., atol=1e-12)
    np.testing.assert_allclose(np.abs(patch["normal"]), [0., 0., 1.], rtol=0., atol=1e-12)
    assert patch["full_support_max_abs_residual_mm"] < 1e-12
    assert np.array_equal(patch["triangle_rows"], np.arange(12))
    if kind == "scan":
        assert patch["support_count"] == 12
        assert patch["rms_measure"] == "UNIQUE_RAW_VERTICES"
        assert np.array_equal(patch["raw_vertex_rows"], np.arange(12))
        _immutable(patch["raw_vertex_rows"])
    else:
        assert patch["support_count"] == 36
        assert patch["rms_measure"] == "TRIANGLE_CORNERS_WITH_MULTIPLICITY"
        assert np.array_equal(patch["source_triangle_ids"], np.arange(1, 13))
        _immutable(patch["source_triangle_ids"])
    if nonuniform:
        assert np.linalg.norm(points.mean(axis=0) - [1.5, 1., 0.]) > .5


def test_single_cad_triangle_matches_independent_barycentric_area_integrals():
    points = np.array([[0., 0., 0.], [6., 0., 0.], [0., 3., 0.]])
    patch, = _available(_cad(points, np.array([[0, 1, 2]], dtype=np.int64)), 1)
    np.testing.assert_allclose(patch["area_centroid_mm"], [2., 1., 0.], rtol=0., atol=1e-12)
    np.testing.assert_allclose(patch["surface_covariance_mm2"],
        [[2., -.5, 0.], [-.5, .5, 0.], [0., 0., 0.]], rtol=0., atol=1e-12)
    assert patch["surface_area_mm2"] == pytest.approx(9., abs=1e-12)
    assert patch["support_count"] == 3


def test_overlapping_distinct_id_triangles_disclose_area_multiplicity_not_union_area():
    points = np.array([[0., 0., 0.], [6., 0., 0.], [0., 3., 0.]])
    triangles = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    patch, = _available(_cad(points, triangles), 1)
    assert patch["surface_area_mm2"] == pytest.approx(18., abs=1e-12)
    assert patch["support_count"] == 6
    assert np.array_equal(patch["source_triangle_ids"], [1, 2])
    assert patch["area_measure"] == "TESSELLATION_AREA_WITH_MULTIPLICITY_NOT_MATERIAL_AREA"


@pytest.mark.parametrize("kind", ("scan", "cad"))
def test_square_hole_is_retained_not_convex_hulled_or_filled(kind):
    points, triangles = _grid(tuple(range(5)), tuple(range(5)),
                              omit_cells=((1, 1), (1, 2), (2, 1), (2, 2)))
    result = _scan(points, triangles) if kind == "scan" else _cad(points, triangles)
    patch, = _available(result, 1, components=1)
    assert len(triangles) == 24
    assert np.array_equal(patch["triangle_rows"], np.arange(24))
    assert patch["surface_area_mm2"] == pytest.approx(12., abs=1e-12)
    np.testing.assert_allclose(patch["area_centroid_mm"], [2., 2., 0.], rtol=0., atol=1e-12)
    np.testing.assert_allclose(patch["surface_covariance_mm2"],
                               np.diag([5./3., 5./3., 0.]), rtol=0., atol=2e-12)
    assert patch["extent_axis_status"] == "ARBITRARY_WITHIN_REPEATED_EIGENSPACE"
    if kind == "scan":
        assert patch["support_count"] == 24
        assert np.array_equal(patch["raw_vertex_rows"], np.delete(np.arange(25), 12))
    else:
        assert patch["support_count"] == 72


@pytest.mark.parametrize("side,admitted", (
    (.2, False), (.3, False), (.5, True),
    (.25*math.sqrt(3.)*(1.-1e-4), False),
    (.25*math.sqrt(3.)*(1.+1e-4), True),
))
@pytest.mark.parametrize("angle", (0., math.pi/4., math.pi/7.))
def test_repeated_inplane_basis_cannot_change_width_admission_by_rotating_a_square(side, admitted, angle):
    points, triangles = _grid(tuple(np.linspace(-side/2., side/2., 4)),
                              (-side/2., 0., side/2.))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    result = _scan(points @ rotation.T, triangles)
    patches = _available(result, int(admitted), components=1)
    component, = result["components"]
    # Uniform-square variance is side²/12 in EVERY in-plane direction.
    assert component["inplane_min_width_lower_bound_mm"] == pytest.approx(side/math.sqrt(3.),
                                                                           rel=1e-12, abs=1e-14)
    assert component["extent_admission_basis"] == "INVARIANT_WIDTH_SUFFICIENT_BOUND_FOR_REPEATED_EIGENSPACE"
    if admitted:
        assert component["reasons"] == []
        assert patches[0]["extent_axis_status"] == "ARBITRARY_WITHIN_REPEATED_EIGENSPACE"
        np.testing.assert_allclose(patches[0]["surface_covariance_mm2"],
                                   np.diag([side*side/12., side*side/12., 0.]), rtol=0., atol=1e-13)
    else:
        assert "REPEATED_EIGENSPACE_EXTENT_UNRESOLVED" in component["reasons"]
        if side >= .25:
            # Actual square width is sufficient: this intentionally documents
            # the conservative bound's false refusal, not a physical defect.
            assert component["reasons"] == ["REPEATED_EIGENSPACE_EXTENT_UNRESOLVED"]


@pytest.mark.parametrize("offset", (0., 10.))
def test_scan_coordinate_coincidence_or_coplanarity_does_not_weld_raw_components(offset):
    points, triangles = _grid()
    all_points = np.vstack((points, points + [offset, 0., 0.]))
    all_triangles = np.vstack((triangles, triangles+len(points)))
    result = _scan(all_points, all_triangles)
    patches = _available(result, 2, components=2)
    support_sets = {tuple(patch["raw_vertex_rows"]) for patch in patches}
    assert support_sets == {tuple(range(12)), tuple(range(12, 24))}
    assert result["statistics"]["support_memberships"] == 24
    assert all(patch["surface_area_mm2"] == pytest.approx(6., abs=1e-12) for patch in patches)


def test_cad_face_tag_keeps_disconnected_finite_triangle_union_without_filling_gap():
    points, triangles = _grid()
    all_points = np.vstack((points, points + [10., 0., 0.]))
    all_triangles = np.vstack((triangles, triangles+len(points)))
    patch, = _available(_cad(all_points, all_triangles), 1, components=1)
    assert patch["source_face_id"] == 71
    assert patch["surface_area_mm2"] == pytest.approx(12., abs=1e-12)
    np.testing.assert_allclose(patch["area_centroid_mm"], [6.5, 1., 0.], rtol=0., atol=1e-12)
    np.testing.assert_allclose(patch["surface_covariance_mm2"],
                               np.diag([25.75, 1./3., 0.]), rtol=0., atol=2e-12)
    assert np.array_equal(patch["triangle_rows"], np.arange(24))
    assert patch["support_count"] == 72


def test_small_remote_perpendicular_cap_survives_a_much_larger_planar_patch():
    points, triangles = _grid(tuple(range(31)), tuple(range(21)))
    cap_points, cap_triangles = _grid((0., .2, .5, .8), (0., 2., 4.))
    cap_points = cap_points[:, [2, 0, 1]] + [50., 0., 0.]
    result = _scan(np.vstack((points, cap_points)),
                   np.vstack((triangles, cap_triangles+len(points))))
    patches = _available(result, 2, components=2)
    cap, = [patch for patch in patches if patch["support_count"] == 12]
    assert np.array_equal(cap["raw_vertex_rows"], np.arange(len(points), len(points)+12))
    np.testing.assert_allclose(np.abs(cap["normal"]), [1., 0., 0.], rtol=0., atol=1e-12)
    assert cap["surface_area_mm2"] == pytest.approx(3.2, abs=1e-12)
    assert max(patch["surface_area_mm2"] for patch in patches) / cap["surface_area_mm2"] > 100.


@pytest.mark.parametrize("pose", range(4))
def test_constructed_flange_cap_and_curved_strip_keep_support_under_four_fixed_poses(pose):
    first_points, first_triangles = _grid()
    second_points, second_triangles = _grid()
    cap_points, cap_triangles = _grid((0., .2, .5, .8), (0., 2., 4.))
    first_points += [20., 0., 0.]
    second_points = second_points[:, [2, 0, 1]] + [0., 10., 0.]
    cap_points = cap_points[:, [0, 2, 1]] + [40., 20., 0.]
    angles = np.linspace(0., math.pi/2., 9)
    curved_points = np.asarray([[-10.+2.*math.cos(angle), 2.*math.sin(angle), z]
                                for z in (0., 1., 2.) for angle in angles])
    _, curved_triangles = _grid(tuple(range(9)), (0., 1., 2.))
    points = np.vstack((first_points, second_points, cap_points, curved_points))
    triangles = np.vstack((first_triangles, second_triangles+12, cap_triangles+24,
                            curved_triangles+36))
    rotations = (np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
                 np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
                 np.array([[.36, -.48, .8], [.8, .6, 0.], [-.48, .64, .6]]))
    translations = ((0., 0., 0.), (30., -20., 10.), (-100., 70., 25.), (1000., -2000., 3000.))
    result = _scan(points @ rotations[pose].T + translations[pose], triangles)
    patches = _available(result, 3, components=11)
    assert {tuple(patch["raw_vertex_rows"]) for patch in patches} == {
        tuple(range(12)), tuple(range(12, 24)), tuple(range(24, 36))}
    rejected = [component for component in result["components"] if component["disposition"] == "REJECTED"]
    assert len(rejected) == 8
    assert all(component["support_count"] == 6 for component in rejected)
    assert all("INSUFFICIENT_SUPPORT_VERTICES" in component["reasons"] for component in rejected)


def test_pairwise_coplanar_drift_chain_fails_the_whole_component_without_salvage():
    points, triangles = _grid(tuple(range(13)), (0., 1.), height=lambda x, y: 4e-9*x*x)
    result = _scan(points, triangles)
    _available(result, 0, components=1)
    assert result["statistics"]["compatible_edge_pairs"] == result["statistics"]["shared_edge_pairs"]
    component, = result["components"]
    assert component["support_count"] == 26 and component["triangle_count"] == 24
    assert component["reasons"] == ["FULL_COMPONENT_MAX_RESIDUAL_EXCEEDS_GATE"]
    assert component["full_support_max_abs_residual_mm"] > 8e-8
    assert np.array_equal(result["triangle_component_labels"], np.zeros(24, dtype=np.int64))


def test_rejected_chain_does_not_consume_an_independent_valid_component():
    bad_points, bad_triangles = _grid(tuple(range(13)), (0., 1.), height=lambda x, y: 4e-9*x*x)
    good_points, good_triangles = _grid()
    good_points += [30., 0., 2.]
    result = _scan(np.vstack((bad_points, good_points)),
                   np.vstack((bad_triangles, good_triangles+len(bad_points))))
    patch, = _available(result, 1, components=2)
    assert np.array_equal(patch["raw_vertex_rows"], np.arange(26, 38))
    assert patch["triangle_count"] == 12
    assert result["statistics"]["rejected_component_count"] == 1


def test_small_nonplanarity_fragments_exact_policy_not_qualified_noisy_extraction():
    points, triangles = _grid(height=lambda x, y: 1e-6*(x*x+2*y*y))
    scan = _scan(points, triangles)
    _available(scan, 0, components=6)
    assert all(component["support_count"] == 4 for component in scan["components"])
    assert all("INSUFFICIENT_SUPPORT_VERTICES" in component["reasons"] for component in scan["components"])
    cad = _cad(points, triangles)
    _available(cad, 0, components=1)
    assert cad["components"][0]["reasons"] == ["FULL_COMPONENT_MAX_RESIDUAL_EXCEEDS_GATE"]


@pytest.mark.parametrize("condition", ("inplane_rank", "second_extent"))
def test_planar_but_inadequate_two_dimensional_support_has_an_explicit_component_refusal(condition):
    if condition == "inplane_rank":
        points, triangles = _grid((0., 30000., 70000., 100000.), (0., 1., 2.))
        expected = "DEGENERATE_INPLANE_SUPPORT"
    else:
        points, triangles = _grid((0., 1., 2., 3.), (0., .1, .2))
        expected = "INSUFFICIENT_SECOND_EXTENT"
    result = _scan(points, triangles)
    _available(result, 0, components=1)
    assert result["components"][0]["reasons"] == [expected]
    assert result["components"][0]["full_support_max_abs_residual_mm"] < 1e-12


def test_rms_uses_unique_scan_rows_but_declared_cad_corner_multiplicity():
    points, triangles = _grid(height=lambda x, y: 1e-9*x*x)
    scan, = _available(_scan(points, triangles), 1)
    cad, = _available(_cad(points, triangles), 1)
    for patch, support in ((scan, points), (cad, points[triangles].reshape(-1, 3))):
        residuals = (support-np.asarray(patch["area_centroid_mm"])) @ np.asarray(patch["normal"])
        assert patch["support_rms_mm"] == pytest.approx(float(np.sqrt(np.mean(residuals**2))),
                                                       rel=1e-7, abs=1e-14)
    np.testing.assert_allclose(scan["area_centroid_mm"], cad["area_centroid_mm"], rtol=0., atol=1e-12)
    assert abs(scan["support_rms_mm"]-cad["support_rms_mm"]) > 1e-12


def test_reported_sorted_extents_and_axes_remain_paired_for_tiny_remote_support():
    points = np.array([[0., 0., 0.], [10., 0., 0.], [10., 1., 0.], [0., 1., 0.],
                       [5., 100., 0.], [5.001, 100., 0.], [5., 100.001, 0.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    patch, = _available(_cad(points, triangles), 1)
    assert patch["support_extent_mm"][0] > 90.
    axes = np.asarray(patch["support_extent_axes"])
    actual = np.ptp(points[triangles].reshape(-1, 3) @ axes.T, axis=0)
    np.testing.assert_allclose(actual, patch["support_extent_mm"], rtol=0., atol=1e-10)


def test_distant_low_area_cad_contamination_is_not_hidden_by_area_or_vertex_rms():
    points, triangles = _grid()
    remote = np.array([[50., 0., .01], [50.001, 0., .01], [50., .001, .01]])
    points = np.vstack((points, remote))
    triangles = np.vstack((triangles, [[12, 13, 14]]))
    result = _cad(points, triangles)
    _available(result, 0, components=1)
    component, = result["components"]
    assert component["triangle_count"] == 13 and component["support_count"] == 39
    assert component["reasons"] == ["FULL_COMPONENT_MAX_RESIDUAL_EXCEEDS_GATE"]
    assert component["support_rms_mm"] < .01
    assert component["full_support_max_abs_residual_mm"] > .009


@pytest.mark.parametrize("kind", ("scan", "cad"))
@pytest.mark.parametrize("scale", (.1, 1., 10.))
def test_frame_and_consistent_unit_changes_preserve_area_support_and_unoriented_plane(kind, scale):
    points, triangles = _grid()
    rotation = np.array([[.36, -.48, .8], [.8, .6, 0.], [-.48, .64, .6]])
    shift = np.array([1000., -2000., 3000.])*scale
    moved = points*scale @ rotation.T + shift
    policy = replace(v4.CoplanarPatchPolicyV4(), max_pair_anchored_distance_mm=1e-8*scale,
                     max_full_support_distance_mm=1e-8*scale, min_second_extent_mm=.25*scale)
    result = _scan(moved, triangles, policy=policy) if kind == "scan" else _cad(moved, triangles, policy=policy)
    patch, = _available(result, 1)
    _rectangle_moments(patch, scale=scale, shift=shift, rotation=rotation)
    expected = rotation[:, 2]
    normal = np.asarray(patch["normal"])
    sign = 1. if normal @ expected > 0. else -1.
    np.testing.assert_allclose(normal, sign*expected, rtol=0., atol=2e-10)
    assert patch["offset_mm"] == pytest.approx(sign*float(expected @ shift), abs=2e-8)
    np.testing.assert_allclose(np.asarray(patch["support_extent_mm"])/scale,
                               [3., 2.], rtol=0., atol=2e-9)


@pytest.mark.parametrize("chunk", (1, 3, 65536))
def test_chunking_and_triangle_row_permutation_preserve_geometry_and_explicit_lineage(chunk):
    points, triangles = _grid((0., .2, .8, 3.), (0., .3, 2.))
    order = np.arange(len(triangles))[::-1]
    changed = triangles[order].copy()
    changed[::2] = changed[::2, [0, 2, 1]]
    result = _scan(points, changed, budgets=replace(v4.CoplanarPatchBudgetsV4(), chunk_triangles=chunk))
    patch, = _available(result, 1)
    _rectangle_moments(patch)
    assert np.array_equal(patch["triangle_rows"], np.arange(12))
    assert np.array_equal(patch["raw_vertex_rows"], np.arange(12))
    assert result["digests"]["triangle_raw_rows_sha256"] == hashlib.sha256(changed.astype("<i8").tobytes()).hexdigest()


@pytest.mark.parametrize("field,value,reason", (
    ("max_vertices", 11, "PATCH_VERTEX_BUDGET_EXCEEDED"),
    ("max_triangles", 11, "PATCH_TRIANGLE_BUDGET_EXCEEDED"),
    ("max_edge_records", 35, "PATCH_EDGE_BUDGET_EXCEEDED"),
    ("max_support_memberships", 35, "PATCH_MEMBERSHIP_BUDGET_EXCEEDED"),
    ("max_full_vertex_tests", 35, "PATCH_VERTEX_TEST_BUDGET_EXCEEDED"),
))
def test_input_work_caps_refuse_before_triangle_geometry(monkeypatch, field, value, reason):
    points, triangles = _grid()
    def forbidden(*args, **kwargs):
        pytest.fail("input cap must precede triangle numerical work")
    monkeypatch.setattr(v4, "_geometry", forbidden)
    result = _scan(points, triangles, budgets=replace(v4.CoplanarPatchBudgetsV4(), **{field: value}))
    _refused(result, reason)


@pytest.mark.parametrize("field", ("max_components", "max_output_patches"))
def test_late_component_or_output_cap_clears_all_arrays_and_proposed_dispositions(field):
    points, triangles = _grid()
    result = _scan(np.vstack((points, points+[10., 0., 0.])),
                   np.vstack((triangles, triangles+12)),
                   budgets=replace(v4.CoplanarPatchBudgetsV4(), **{field: 1}))
    _refused(result, "PATCH_COMPONENT_BUDGET_EXCEEDED" if field == "max_components" else "PATCH_OUTPUT_BUDGET_EXCEEDED")


def test_late_eigensolve_failure_clears_component_labels_and_all_candidate_evidence(monkeypatch):
    points, triangles = _grid()
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("invented covariance refusal")
    monkeypatch.setattr(v4.np.linalg, "eigh", fail)
    _refused(_scan(points, triangles))


def test_more_than_two_triangle_fan_on_one_raw_edge_refuses_entire_scan():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., -1., 0.], [0., 0., 1.]])
    triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    _refused(_scan(points, triangles), "NONMANIFOLD_SHARED_RAW_EDGE_FAN")


@pytest.mark.parametrize("kind", ("scan", "cad"))
@pytest.mark.parametrize("bad", ("late_nan", "late_degenerate", "extreme", "subnormal"))
def test_unsupported_geometry_cannot_leave_an_earlier_valid_patch(kind, bad):
    points, triangles = _grid()
    extra = np.array([[20., 0., 0.], [21., 0., 0.], [20., 1., 0.]])
    if bad == "late_nan":
        extra[-1, 0] = np.nan
    elif bad == "late_degenerate":
        extra[-1] = [22., 0., 0.]
    elif bad == "extreme":
        extra[-1, 0] = 1_000_001.
    else:
        extra = np.array([[0., 0., 0.], [1e-300, 0., 0.], [0., 1e-300, 0.]])
    all_points = np.vstack((points, extra))
    all_triangles = np.vstack((triangles, [[12, 13, 14]]))
    result = _scan(all_points, all_triangles) if kind == "scan" else _cad(all_points, all_triangles)
    _refused(result)


def test_unused_raw_nan_is_still_validated_not_hidden_outside_triangle_support():
    points, triangles = _grid()
    _refused(_scan(np.vstack((points, [[np.nan, 0., 0.]])), triangles), "INVALID_PATCH_POSITION")


@pytest.mark.parametrize("bad", ("negative", "too_large", "duplicate_corner", "float_indices", "boolean_points", "object_points"))
def test_invalid_raw_identity_or_array_types_are_not_coerced_into_support(bad):
    points, triangles = _grid()
    if bad == "negative":
        triangles[-1, -1] = -1
    elif bad == "too_large":
        triangles[-1, -1] = len(points)
    elif bad == "duplicate_corner":
        triangles[-1, -1] = triangles[-1, 0]
    elif bad == "float_indices":
        triangles = triangles.astype(float)
    elif bad == "boolean_points":
        points = points.astype(bool)
    else:
        points = points.astype(object)
    _refused(_scan(points, triangles))


@pytest.mark.parametrize("bad", ("duplicate_pair", "negative_face", "float_face", "short_ids"))
def test_cad_local_source_id_contract_is_revalidated(bad):
    points, triangles = _grid()
    faces = np.full(12, 71, dtype=np.int64)
    local = np.arange(1, 13, dtype=np.int64)
    if bad == "duplicate_pair":
        local[-1] = local[0]
    elif bad == "negative_face":
        faces[-1] = -1
    elif bad == "float_face":
        faces = faces.astype(float)
    else:
        local = local[:-1]
    _refused(_cad(points, triangles, face_ids=faces, local_ids=local))


@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
def test_largest_admitted_cad_face_id_is_preserved_without_float_round_trip(dtype):
    points, triangles = _grid()
    faces = np.full(12, 2**63-1, dtype=dtype)
    patch, = _available(_cad(points, triangles, face_ids=faces), 1)
    assert patch["source_face_id"] == patch["plane_id"] == 2**63-1


def test_unsigned_cad_id_outside_int64_does_not_wrap_into_a_valid_group():
    points, triangles = _grid()
    faces = np.full(12, 71, dtype=np.uint64)
    faces[-1] = 2**63
    _refused(_cad(points, triangles, face_ids=faces), "INVALID_CAD_PATCH_SOURCE_IDS")


@pytest.mark.parametrize("kind,field,value", (
    ("policy", "min_support_vertices", True),
    ("policy", "max_pair_normal_sine", 0.),
    ("policy", "max_full_support_distance_mm", float("nan")),
    ("budgets", "max_components", True),
    ("budgets", "chunk_triangles", 0),
))
def test_malformed_policy_or_budget_scalars_fail_closed(kind, field, value):
    points, triangles = _grid()
    default = v4.CoplanarPatchPolicyV4() if kind == "policy" else v4.CoplanarPatchBudgetsV4()
    _refused(_scan(points, triangles, **{kind: replace(default, **{field: value})}))


@pytest.mark.parametrize("kind", ("policy", "budgets"))
def test_extended_policy_dataclass_is_not_an_authorized_contract(kind):
    parent = v4.CoplanarPatchPolicyV4 if kind == "policy" else v4.CoplanarPatchBudgetsV4
    class Extended(parent):
        extra_authority = True
    points, triangles = _grid()
    _refused(_scan(points, triangles, **{kind: Extended()}))


@pytest.mark.parametrize("field", tuple(field.name for field in fields(v4.CoplanarPatchBudgetsV4)))
def test_declared_hard_budgets_cannot_be_raised(field):
    points, triangles = _grid()
    budget = v4.CoplanarPatchBudgetsV4()
    _refused(_scan(points, triangles, budgets=replace(budget, **{field: getattr(budget, field)+1})),
             "INVALID_PATCH_BUDGET")


def test_input_mutations_after_return_cannot_rewrite_support_identity():
    points, triangles = _grid()
    result = _scan(points, triangles)
    patch, = _available(result, 1)
    saved = (patch["raw_vertex_rows"].copy(), patch["triangle_rows"].copy(),
             result["triangle_component_labels"].copy(), result["digests"].copy())
    points[:] = 123.
    triangles[:] = 0
    assert np.array_equal(patch["raw_vertex_rows"], saved[0])
    assert np.array_equal(patch["triangle_rows"], saved[1])
    assert np.array_equal(result["triangle_component_labels"], saved[2])
    assert result["digests"] == saved[3]
    _immutable(patch["raw_vertex_rows"])
    _immutable(patch["triangle_rows"])
