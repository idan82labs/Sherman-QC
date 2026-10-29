"""Independent planar material charts; no native, physical or retained input."""
import numpy as np
import pytest

from backend import vendor_unfold_finite_width as incumbent
from backend import vendor_unfold_local_strip_guard_v1 as guard


def rectangle(x0, x1, y0, y1):
    return np.array([[x0, y0, 0.], [x1, y0, 0.], [x1, y1, 0.], [x0, y1, 0.]]), np.array([[0, 1, 2], [0, 2, 3]])


def material(*rectangles):
    points, rows, offset = [], [], 0
    for params in rectangles:
        vertices, triangles = rectangle(*params)
        points.extend(vertices)
        rows.extend(triangles+offset)
        offset += len(vertices)
    return np.array(points), np.array(rows)


def options(side="parent"):
    return dict(cylinder_origin=[0., 0., 0.], cylinder_axis=[0., 1., 0.],
        signed_tangent=[1., 0., 0.], seam_endpoints=[[0., 0., 0.], [0., 10., 0.]],
        axial_min=0., axial_max=10., developed_strip_width=3., side=side)


def c_flange():
    # A left flange returns beyond the bend through a bridge outside its finite
    # axial range. The complete developed bend strip is x[0,3] y[0,10].
    return material((-5., 0., 0., 10.), (-5., 0., -2., 0.),
                    (0., 10., -2., -1.), (8., 10., -1., 10.))


def test_c_return_beyond_finite_strip_is_retained_with_local_seam_witness():
    vertices, triangles = c_flange()
    old_options = options()
    old_options["tangent"] = old_options.pop("signed_tangent")
    old_options.pop("developed_strip_width")
    with pytest.raises(incumbent.FiniteWidthUnavailable, match="FLANGE_EXTENDS"):
        incumbent.certify_finite_width_flange(vertices, triangles, **old_options)
    prior_vertices, prior_triangles = vertices.copy(), triangles.copy()
    result = guard.certify_local_finite_strip(vertices, triangles, **options())
    assert result["status"] == "CERTIFIED"
    assert result["native_seam_boundary_interval_count"] == 1
    assert result["native_seam_local_material_side_checked"] is True
    assert result["retained_remote_side_triangles"] == 2
    assert result["work"]["triangles"] == len(triangles)
    assert np.array_equal(vertices, prior_vertices) and np.array_equal(triangles, prior_triangles)
    assert result["manufacturing_ready"] is False


def test_ordinary_halfspace_flange_still_certifies():
    result = guard.certify_local_finite_strip(*rectangle(-5., 0., 0., 10.), **options())
    assert result["retained_remote_side_triangles"] == 0


def test_child_uses_opposite_local_material_polarity():
    assert guard.certify_local_finite_strip(*rectangle(0., 5., 0., 10.), **options("child"))["status"] == "CERTIFIED"
    with pytest.raises(guard.LocalStripUnavailable, match="INTERSECTS|MATERIAL_SIDE"):
        guard.certify_local_finite_strip(*rectangle(0., 5., 0., 10.), **options("parent"))


def test_actual_strip_intrusion_refuses_even_when_all_triangle_vertices_outside():
    vertices, triangles = c_flange()
    offset = len(vertices)
    vertices = np.vstack([vertices, [[1., -1., 0.], [1., 11., 0.], [2., -1., 0.]]])
    triangles = np.vstack([triangles, [[offset, offset+1, offset+2]]])
    with pytest.raises(guard.LocalStripUnavailable, match="FLANGE_INTERSECTS_FINITE_DEVELOPED_STRIP"):
        guard.certify_local_finite_strip(vertices, triangles, **options())


def test_no_local_native_seam_coverage_cannot_receive_certificate():
    vertices, triangles = rectangle(-5., 0., 0., 4.)
    with pytest.raises(guard.LocalStripUnavailable, match="COVERAGE_GAP"):
        guard.certify_local_finite_strip(vertices, triangles, **options())


def test_shared_interior_edge_is_not_native_seam_boundary():
    vertices, triangles = rectangle(-5., 0., 0., 10.)
    # A second triangle owns the seam, with distinct other mesh edges.
    vertices = np.vstack([vertices, [[-2., 5., 0.]]])
    triangles = np.vstack([triangles, [[1, 4, 2]]])
    with pytest.raises(guard.LocalStripUnavailable, match="NOT_MESH_BOUNDARY"):
        guard.certify_local_finite_strip(vertices, triangles, **options())


def test_nonmanifold_mesh_edge_refuses():
    vertices, triangles = rectangle(-5., 0., 0., 10.)
    with pytest.raises(guard.LocalStripUnavailable, match="NON_MANIFOLD"):
        guard.certify_local_finite_strip(vertices, np.vstack([triangles]*3), **options())


def test_boundary_triangle_without_definite_inward_side_refuses():
    vertices, triangles = rectangle(-1e-7, 0., 0., 10.)
    with pytest.raises(guard.LocalStripUnavailable, match="MATERIAL_SIDE_NOT_PROVEN"):
        guard.certify_local_finite_strip(vertices, triangles, **options())


def test_seam_endpoint_reversal_does_not_change_guard():
    first = guard.certify_local_finite_strip(*c_flange(), **options())
    values = options()
    values["seam_endpoints"] = values["seam_endpoints"][::-1]
    assert guard.certify_local_finite_strip(*c_flange(), **values) == first


def test_proper_pose_preserves_strip_material_relation():
    vertices, triangles = c_flange()
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    shift = np.array([30., -12., 42.])
    values = options()
    for key in ("cylinder_axis", "signed_tangent"):
        values[key] = rotation @ np.array(values[key])
    for key in ("cylinder_origin",):
        values[key] = rotation @ np.array(values[key])+shift
    values["seam_endpoints"] = np.array(values["seam_endpoints"]) @ rotation.T+shift
    assert guard.certify_local_finite_strip(vertices @ rotation.T+shift, triangles, **values)["status"] == "CERTIFIED"


def test_shared_budget_cannot_reset_between_certificates(monkeypatch):
    monkeypatch.setitem(guard.CAPS, "triangles", 3)
    budget = guard.LocalStripBudget()
    guard.certify_local_finite_strip(*rectangle(-5., 0., 0., 10.), **options(), budget=budget)
    with pytest.raises(guard.LocalStripUnavailable, match="WORK_BUDGET_EXCEEDED"):
        guard.certify_local_finite_strip(*rectangle(-5., 0., 0., 10.), **options(), budget=budget)


@pytest.mark.parametrize("fault,value,reason", [
    ("signed_tangent", [2., 0., 0.], "FRAME"),
    ("developed_strip_width", -1., "INTERVAL"),
    ("axial_max", 0., "INTERVAL"),
    ("seam_endpoints", [[0., 0., 0.], [0., 9., 0.]], "COVER_AXIAL_SPAN"),
])
def test_malformed_physical_strip_or_frame_refuses(fault, value, reason):
    values = options()
    values[fault] = value
    with pytest.raises(guard.LocalStripUnavailable, match=reason):
        guard.certify_local_finite_strip(*rectangle(-5., 0., 0., 10.), **values)
