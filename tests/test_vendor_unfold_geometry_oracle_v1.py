"""Independently constructed charts; imports neither engine nor native CAD."""
import copy
import math

import numpy as np
import pytest

from backend.vendor_unfold_geometry_oracle_v1 import (
    BendEvidence, DevelopedBend, DevelopedFace, FaceEvidence, GeometryOracleError,
    SeamEvidence, validate_development,
)


def rectangle(index=0, shift=0.):
    chart = np.array([[0., 0.], [10., 0.], [10., 6.], [0., 6.]])
    vertices = np.column_stack((chart, np.zeros(4)))
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    source = FaceEvidence(index, vertices, triangles, chart, np.tile([0., 0., 1.], (2, 1)), 1, 60., 0.)
    flat = DevelopedFace(index, vertices+np.array([shift, 0., 0.]), triangles.copy())
    return source, flat


def quarter_cylinder(index=1, radius=4., width=12., pieces=8):
    angles = np.linspace(.2, .2+math.pi/2, pieces+1)
    points = np.array([[radius*math.cos(u), radius*math.sin(u), z] for u in angles for z in (0., width)])
    triangles = np.array([t for j in range(pieces) for t in
                          ([2*j, 2*j+2, 2*j+3], [2*j, 2*j+3, 2*j+1])], dtype=np.int32)
    chart = np.array([[radius*(u-angles[0]), z] for u in angles for z in (0., width)])
    centroid_angles = np.array([sum(angles[i//2] for i in t)/3 for t in triangles])
    normals = np.column_stack((np.cos(centroid_angles), np.sin(centroid_angles), np.zeros(len(triangles))))
    source = FaceEvidence(index, points, triangles, chart, normals, 1, radius*width*math.pi/2, 0.)
    flat = DevelopedFace(index, np.column_stack((chart, np.zeros(len(chart)))), triangles.copy())
    return source, flat


def hole():
    chart = np.array([[0., 0.], [10., 0.], [10., 6.], [0., 6.],
                      [3., 2.], [7., 2.], [7., 4.], [3., 4.]])
    triangles = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    points = np.column_stack((chart, np.zeros(len(chart))))
    source = FaceEvidence(0, points, triangles, chart, np.tile([0., 0., 1.], (8, 1)), 2, 52., 0.)
    return source, DevelopedFace(0, points.copy(), triangles.copy())


def test_plane_cylinder_intrinsic_area_differs_from_chord_mesh_area():
    source, flat = quarter_cylinder()
    report = validate_development([source], [flat])
    area = 4*12*math.pi/2
    assert report["total_flat_area_mm2"] == pytest.approx(area)
    q = source.vertices[source.triangles]
    chord_area = np.linalg.norm(np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]), axis=1).sum()/2
    assert chord_area < area-.1
    wrong = FaceEvidence(**{**source.__dict__, "continuous_area_mm2": float(chord_area)})
    with pytest.raises(GeometryOracleError, match="CONTINUOUS_AREA"):
        validate_development([wrong], [flat])


def test_rotation_translation_of_flat_chart_preserves_intrinsic_geometry():
    source, flat = quarter_cylinder()
    angle = .75
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    flat.vertices[:, :2] = flat.vertices[:, :2] @ rot.T+[500., -123.]
    assert validate_development([source], [flat])["face_count"] == 1


def test_all_hole_boundary_cycles_and_area_preserved():
    source, flat = hole()
    report = validate_development([source], [flat])
    assert report["faces"][0]["boundary_cycles"] == 2
    assert report["total_flat_area_mm2"] == pytest.approx(52.)


def test_filled_hole_cannot_pass_equal_vertex_or_visible_mesh_check():
    source, flat = hole()
    output = DevelopedFace(0, flat.vertices, np.vstack((flat.triangles, [[4, 5, 6], [4, 6, 7]])))
    with pytest.raises(GeometryOracleError, match="TRIANGLE_LINEAGE"):
        validate_development([source], [output])


def test_false_native_hole_count_refuses_independently_of_engine_diagnostic():
    source, flat = hole()
    wrong = FaceEvidence(**{**source.__dict__, "boundary_cycles": 1})
    with pytest.raises(GeometryOracleError, match="HOLE_BOUNDARY"):
        validate_development([wrong], [flat])


@pytest.mark.parametrize("all_triangles", [False, True])
def test_mixed_or_whole_face_negative_winding_refuses(all_triangles):
    source, flat = rectangle()
    selected = slice(None) if all_triangles else slice(0, 1)
    flat.triangles[selected] = flat.triangles[selected][:, [0, 2, 1]]
    with pytest.raises(GeometryOracleError, match="FLAT_WINDING|LOCAL_TRIANGLE_PARITY"):
        validate_development([source], [flat])


def test_source_reversed_native_face_requires_orientation_applied():
    source, flat = rectangle()
    wrong = FaceEvidence(**{**source.__dict__, "oriented_normals": -source.oriented_normals})
    with pytest.raises(GeometryOracleError, match="NATIVE_FACE_ORIENTATION"):
        validate_development([wrong], [flat])
    corrected = FaceEvidence(**{**wrong.__dict__, "triangles": source.triangles[:, [0, 2, 1]]})
    assert validate_development([corrected], [flat])["face_count"] == 1


def test_equal_area_shear_is_detected_by_global_intrinsic_metric():
    source, flat = rectangle()
    flat.vertices[:, 0] += flat.vertices[:, 1]*.3
    with pytest.raises(GeometryOracleError, match="INTRINSIC_METRIC"):
        validate_development([source], [flat])


def test_seam_checks_relative_pose_that_face_metric_cannot_detect():
    first, a = rectangle(0)
    second, b = rectangle(1, 10.)
    seam = SeamEvidence(0, (1, 2), 1, (0, 3))
    assert validate_development([first, second], [a, b], [seam])["max_seam_error_mm"] == 0
    b.vertices[:, 0] += .02
    assert validate_development([first, second], [a, b])["face_count"] == 2
    with pytest.raises(GeometryOracleError, match="DEVELOPED_SEAM"):
        validate_development([first, second], [a, b], [seam])


def test_unexplained_area_tolerance_is_never_inferred_from_tessellation():
    source, flat = rectangle()
    wrong = FaceEvidence(**{**source.__dict__, "continuous_area_mm2": 60.01})
    with pytest.raises(GeometryOracleError, match="CONTINUOUS_AREA"):
        validate_development([wrong], [flat])
    declared = FaceEvidence(**{**wrong.__dict__, "area_error_bound_mm2": .011})
    assert validate_development([declared], [flat])["continuous_trim_proof"] is False


def test_inputs_remain_unchanged_and_no_engine_diagnostics_are_consumed():
    source, flat = hole()
    before = copy.deepcopy((source, flat))
    validate_development([source], [flat])
    assert np.array_equal(source.vertices, before[0].vertices)
    assert np.array_equal(flat.triangles, before[1].triangles)


def test_missing_duplicate_and_reordered_source_ids_refuse():
    source, flat = rectangle()
    with pytest.raises(GeometryOracleError, match="BIJECTION"):
        validate_development([source, source], [flat, flat])
    wrong = DevelopedFace(5, flat.vertices, flat.triangles)
    with pytest.raises(GeometryOracleError, match="BIJECTION"):
        validate_development([source], [wrong])


def test_nonplanar_face_and_duplicate_triangle_refuse():
    source, flat = rectangle()
    flat.vertices[1, 2] = .1
    with pytest.raises(GeometryOracleError, match="FLAT_NOT_PLANAR"):
        validate_development([source], [flat])
    source, flat = rectangle()
    bad = DevelopedFace(0, flat.vertices, np.array([[0, 1, 2], [0, 1, 2]]))
    with pytest.raises(GeometryOracleError, match="DUPLICATE"):
        validate_development([source], [bad])


def test_bend_mid_sweep_location_finite_width_and_endpoint_reversal():
    source, flat = quarter_cylinder()
    chart = np.array([[math.pi, 0.], [math.pi, 12.]])
    evidence = BendEvidence("bend-1", 1, chart)
    bend = DevelopedBend("bend-1", np.column_stack((chart[::-1], np.zeros(2))))
    result = validate_development([source], [flat], bend_evidence=[evidence], developed_bends=[bend])
    assert result["bend_count"] == 1
    for shift in ([.1, 0., 0.], [0., .1, 0.]):
        wrong = DevelopedBend("bend-1", bend.endpoints+shift)
        with pytest.raises(GeometryOracleError, match="LOCATION_OR_SPAN"):
            validate_development([source], [flat], bend_evidence=[evidence], developed_bends=[wrong])
    shorter = DevelopedBend("bend-1", np.array([[math.pi, 1., 0.], [math.pi, 11., 0.]]))
    with pytest.raises(GeometryOracleError, match="LOCATION_OR_SPAN"):
        validate_development([source], [flat], bend_evidence=[evidence], developed_bends=[shorter])


def test_missing_bend_and_duplicate_bend_id_refuse():
    source, flat = quarter_cylinder()
    evidence = BendEvidence("bend-1", 1, np.array([[math.pi, 0.], [math.pi, 12.]]))
    with pytest.raises(GeometryOracleError, match="BEND_BIJECTION"):
        validate_development([source], [flat], bend_evidence=[evidence])


def test_per_face_reflection_cannot_hide_in_individual_triangle_reversal():
    first, a = rectangle(0)
    second, b = rectangle(1)
    # A reflection retains every metric, area and boundary count, and flipping
    # its triangle order makes +Z shading look valid. Global source parity
    # must still detect this independently of the visual appearance.
    b.vertices[:, 0] *= -1
    b.triangles[:] = b.triangles[:, [0, 2, 1]]
    with pytest.raises(GeometryOracleError, match="CROSS_FACE_TRIANGLE_PARITY"):
        validate_development([first, second], [a, b])


def test_discrete_native_chart_verification_does_not_claim_continuous_area():
    source, flat = quarter_cylinder()
    discrete = FaceEvidence(**{**source.__dict__, "continuous_area_mm2": None,
                              "area_error_bound_mm2": None})
    report = validate_development([discrete], [flat])
    assert report["verification_model"] == "DISCRETE_NATIVE_METRIC_CHART"
    assert report["continuous_area_check"] == "NOT_ESTABLISHED"
    assert report["faces"][0]["continuous_area_mm2"] is None
    assert report["faces"][0]["declared_area_error_bound_mm2"] is None
    assert report["faces"][0]["discrete_metric_polygon_area_check"] == "PASSED"
    assert report["manufacturing_ready"] is False


@pytest.mark.parametrize("missing", ["continuous_area_mm2", "area_error_bound_mm2"])
def test_partial_continuous_area_model_refuses(missing):
    source, flat = rectangle()
    incomplete = FaceEvidence(**{**source.__dict__, missing: None})
    with pytest.raises(GeometryOracleError, match="INCOMPLETE_ORACLE_CONTINUOUS_AREA_MODEL"):
        validate_development([incomplete], [flat])


def test_discrete_mode_still_checks_small_area_change_below_metric_tolerance():
    source, flat = rectangle()
    discrete = FaceEvidence(**{**source.__dict__, "continuous_area_mm2": None,
                              "area_error_bound_mm2": None})
    # Displacement 1e-6mm passes the metric tolerance, but its material area
    # change is much larger than the declared fixed area roundoff allowance.
    flat.vertices[1:3, 0] += 1e-6
    with pytest.raises(GeometryOracleError, match="DISCRETE_METRIC_POLYGON_AREA"):
        validate_development([discrete], [flat])


def test_mixed_discrete_continuous_evidence_keeps_whole_continuous_gate_pending():
    first, a = rectangle(0)
    second, b = rectangle(1, 10.)
    discrete = FaceEvidence(**{**second.__dict__, "continuous_area_mm2": None,
                              "area_error_bound_mm2": None})
    report = validate_development([first, discrete], [a, b])
    assert report["continuous_area_check"] == "NOT_ESTABLISHED"
    assert report["faces"][0]["continuous_area_check"] == "PASSED_UNDER_DECLARED_ERROR_MODEL"
    assert report["faces"][1]["continuous_area_check"] == "NOT_ESTABLISHED"
