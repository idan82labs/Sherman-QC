"""Synthetic arithmetic charts only: no CAD kernels, imports, files, or jobs."""

import copy
import json
import math

import pytest

from backend import vendor_unfold_finite_width as fw


def certify(vertices, triangles, **updates):
    arguments = {"cylinder_origin": [0, 0, 0], "cylinder_axis": [0, 0, 1],
                 "tangent": [0, 1, 0], "seam_endpoints": [[2, 0, 0], [2, 0, 20]],
                 "axial_min": 0, "axial_max": 20, "side": "parent"}
    arguments.update(updates)
    return fw.certify_finite_width_flange(vertices, triangles, **arguments)


def rectangle(y0=-10, y1=0, z0=0, z1=20):
    return [[2, y0, z0], [2, y1, z0], [2, y1, z1], [2, y0, z1]], [[0, 1, 2], [0, 2, 3]]


def assert_refusal(vertices, triangles, reason, **updates):
    with pytest.raises(fw.FiniteWidthUnavailable, match=reason):
        certify(vertices, triangles, **updates)


def test_old_global_false_refusal_tab_is_certified_without_mesh_changes():
    vertices, triangles = rectangle(z1=25)
    tab, tab_triangles = rectangle(y0=0, y1=4, z0=21, z1=25)
    vertices += tab
    triangles += [[index+4 for index in row] for row in tab_triangles]
    before = copy.deepcopy((vertices, triangles))
    assert max(point[1] for point in vertices) == 4  # Old infinite guard refuses.
    proof = certify(vertices, triangles)
    assert proof["status"] == "CERTIFIED"
    assert proof["intersecting_triangles"] == 2
    assert proof["max_oriented_side_distance_mm"] == 0
    assert (vertices, triangles) == before
    assert "vertices" not in proof and "triangles" not in proof
    assert proof["manufacturing_ready"] is False
    assert "whole_unchanged_developed_mesh_overlap" in proof["required_external_gates"]
    json.dumps(proof, allow_nan=False)


def test_all_vertices_outside_slab_crossing_triangle_is_not_skipped():
    vertices = [[2, 2, -1], [2, 2, 21], [2, 5, -1]]
    assert all(not 0 <= point[2] <= 20 for point in vertices)
    assert_refusal(vertices, [[0, 1, 2]], "FLANGE_EXTENDS")


def test_all_vertices_outside_safe_crossing_triangle_is_evaluated():
    vertices = [[2, -2, -1], [2, -2, 21], [2, -5, -1]]
    assert certify(vertices, [[0, 1, 2]])["intersecting_triangles"] == 1


def test_outside_span_overlap_is_not_claimed_safe_by_local_certificate():
    # Independent mathematical flat chart, NOT an integrated CAD/unfold test.
    # Parent tab is y[0,4], z[21,25]. A quarter bend r2 translates a child
    # strip to developed y[pi, pi+5], z[0,25], overlapping that outside-span tab.
    vertices, triangles = rectangle(z1=25)
    tab, tab_triangles = rectangle(y0=0, y1=4, z0=21, z1=25)
    vertices += tab
    triangles += [[i+4 for i in row] for row in tab_triangles]
    proof = certify(vertices, triangles)
    overlap_width = min(4, math.pi+5)-max(0, math.pi)
    overlap_height = min(25, 25)-max(21, 0)
    assert overlap_width*overlap_height > fw.GEOM_TOL**2
    assert proof["geometry_mutated"] is False
    assert "whole_unchanged_developed_mesh_overlap" in proof["required_external_gates"]


def test_child_signed_side_and_wrong_side():
    vertices, triangles = rectangle(y0=0, y1=10)
    assert certify(vertices, triangles, side="child")["side"] == "child"
    assert_refusal(vertices, triangles, "FLANGE_EXTENDS", side="parent")
    reversed_vertices = [[x, -y, z] for x, y, z in vertices]
    assert certify(reversed_vertices, triangles, side="child", tangent=[0, -1, 0])


def test_reversed_seam_order_and_axis_span_chart():
    vertices, triangles = rectangle()
    forward = certify(vertices, triangles)
    backward = certify(vertices, triangles, seam_endpoints=[[2, 0, 20], [2, 0, 0]])
    assert backward == forward
    reverse_axis = certify(vertices, triangles, cylinder_axis=[0, 0, -1],
                           axial_min=-20, axial_max=0)
    assert reverse_axis["max_oriented_side_distance_mm"] == 0
    assert_refusal(vertices, triangles, "AXIAL_SPAN", axial_min=20, axial_max=0)


def test_rigid_pose_and_mirrored_chart_are_synthetic_only():
    vertices, triangles = rectangle()
    angle = .73
    c, s = math.cos(angle), math.sin(angle)
    def pose(point):
        x, y, z = point
        return [c*x-s*y+17, s*x+c*y-9, z+4]
    proof = certify([pose(point) for point in vertices], triangles,
                    cylinder_origin=pose([0, 0, 0]), tangent=[-s, c, 0],
                    seam_endpoints=[pose([2, 0, 0]), pose([2, 0, 20])])
    assert proof["status"] == "CERTIFIED"
    mirrored = [[-x, y, z] for x, y, z in vertices]
    assert certify(mirrored, triangles, seam_endpoints=[[-2, 0, 0], [-2, 0, 20]])


@pytest.mark.parametrize("span", [(0, 0), (0, 1e-6)])
def test_degenerate_or_unresolved_axial_spans_refuse(span):
    assert_refusal(*rectangle(), "AXIAL_SPAN", axial_min=span[0], axial_max=span[1])


@pytest.mark.parametrize("z", [0, 20, -fw.GEOM_TOL, 20+fw.GEOM_TOL])
def test_closed_slab_endpoint_and_expansion_boundaries_are_included(z):
    # A triangle with an edge at the slab boundary is still checked.
    outside = z-1 if z <= 0 else z+1
    vertices = [[2, 0, z], [2, 1, z], [2, 0, outside]]
    assert_refusal(vertices, [[0, 1, 2]], "FLANGE_EXTENDS")


def test_side_tolerance_not_enlarged_and_exact_unit_boundary_allowed():
    vertices, triangles = rectangle(y0=-1, y1=fw.GEOM_TOL)
    assert certify(vertices, triangles)["max_oriented_side_distance_mm"] == fw.GEOM_TOL
    vertices[1][1] = vertices[2][1] = math.nextafter(fw.GEOM_TOL, math.inf)
    assert_refusal(vertices, triangles, "FLANGE_EXTENDS")


def test_negative_reported_normalized_distance_is_an_upper_bound():
    vertices, triangles = rectangle(y0=-1_000_001, y1=-1_000_000)
    proof = certify(vertices, triangles, tangent=[0, 1-32*math.ulp(1), 0])
    assert proof["max_oriented_side_distance_mm"] >= -1_000_000
    assert proof["max_oriented_side_distance_mm"] < -999_999.999


def test_unsafe_material_just_outside_expansion_does_not_enter_local_proof():
    vertices, triangles = rectangle()
    just_outside = math.nextafter(20+fw.GEOM_TOL, math.inf)
    tab, tab_triangles = rectangle(y0=0, y1=4, z0=just_outside, z1=21)
    vertices += tab
    triangles += [[i+4 for i in row] for row in tab_triangles]
    assert certify(vertices, triangles)["intersecting_triangles"] == 2


def test_near_parallel_clipping_is_exact_not_division_threshold_sampling():
    lower = -fw.GEOM_TOL
    vertices = [[2, -1, math.nextafter(lower, -math.inf)],
                [2, -2, math.nextafter(lower, math.inf)], [2, -3, 20]]
    proof = certify(vertices, [[0, 1, 2]])
    assert proof["exact_binary_clipping"] is True
    assert proof["intersecting_triangles"] == 1


def test_no_vacuous_outside_only_certificate():
    assert_refusal(*rectangle(z0=21, z1=25), "NO_TRIANGLE")


@pytest.mark.parametrize("seam", [[[2, 0, 1], [2, 0, 20]],
                                   [[2, 0, 0], [2, 0, 19]],
                                   [[2, 0, 0], [2.1, 0, 20]]])
def test_seam_span_gap_and_nonaxial_seam_refuse(seam):
    assert_refusal(*rectangle(), "SEAM", seam_endpoints=seam)


def test_shared_seam_uncertainty_is_charged_not_added_to_tolerance():
    vertices, triangles = rectangle(y0=-1, y1=fw.GEOM_TOL)
    # Midpoint y=-tol/32 shifts the side chart; a near-tolerance edge cannot
    # borrow another seam tolerance. The total budget is still exactly tol.
    seam = [[2, -fw.GEOM_TOL/16, 0], [2, 0, 20]]
    assert_refusal(vertices, triangles, "FLANGE_EXTENDS", seam_endpoints=seam)


@pytest.mark.parametrize("updates", [{"cylinder_axis": [0, 0, 2]},
                                      {"tangent": [0, 0, 1]},
                                      {"tangent": [0, 0, 0]},
                                      {"cylinder_axis": [1e-9, 0, 1]}])
def test_invalid_frames_refuse(updates):
    # The last frame is unit-ish but is nonorthogonal to a tailored tangent.
    if updates.get("cylinder_axis") == [1e-9, 0, 1]:
        updates = {**updates, "tangent": [1, 0, 0]}
    assert_refusal(*rectangle(), "FRAME", **updates)


@pytest.mark.parametrize("value", [True, None, "0", math.nan, math.inf, -math.inf, 10**400])
def test_bad_coordinate_numbers_raise_typed_refusal(value):
    vertices, triangles = rectangle()
    vertices[0][0] = value
    assert_refusal(vertices, triangles, "NUMBER")


def test_huge_coordinate_resolution_refuses_even_if_subtraction_cancels():
    vertices, triangles = rectangle()
    vertices = [[x+1e12, y, z] for x, y, z in vertices]
    assert_refusal(vertices, triangles, "COORDINATE_RESOLUTION",
                   cylinder_origin=[1e12, 0, 0],
                   seam_endpoints=[[1e12+2, 0, 0], [1e12+2, 0, 20]])


@pytest.mark.parametrize("row", [[0, 0, 2], [0, 1, 4], [-1, 1, 2],
                                 [0, True, 2], [0, 1.0, 2], [0, 1], None])
def test_bad_triangle_tables_raise_typed_refusal(row):
    vertices, _ = rectangle()
    assert_refusal(vertices, [row], "TABLE|INDEX")


def test_degenerate_triangle_refuses():
    assert_refusal([[2, -1, 0], [2, -1, 1], [2, -1, 2]], [[0, 1, 2]], "DEGENERATE")


def test_unused_nonfinite_vertex_still_refuses():
    vertices, triangles = rectangle()
    vertices.append([2, math.nan, 0])
    assert_refusal(vertices, triangles, "NONFINITE")


def test_fixed_shared_work_cap_is_charged_on_refusal(monkeypatch):
    monkeypatch.setattr(fw, "MAX_TOTAL_TRIANGLES", 3)
    budget = fw.FiniteWidthBudget()
    assert certify(*rectangle(), budget=budget)["shared_work"]["triangles"] == 2
    assert_refusal(*rectangle(), "WORK_BUDGET", budget=budget)
    assert budget.snapshot()["triangles"] == 3
    assert budget.snapshot()["projected_vertices"] == 8


@pytest.mark.parametrize("limit_name", ["MAX_TOTAL_PROJECTED_VERTICES", "MAX_TOTAL_CLIP_EDGE_TESTS"])
def test_projection_and_edge_work_are_separately_bounded(monkeypatch, limit_name):
    monkeypatch.setattr(fw, limit_name, 1)
    budget = fw.FiniteWidthBudget()
    assert_refusal(*rectangle(), "WORK_BUDGET", budget=budget)
    assert all(value >= 0 for value in budget.snapshot().values())


def test_input_cap_and_rational_complexity_cap_fail_closed(monkeypatch):
    monkeypatch.setattr(fw, "MAX_TRIANGLES", 1)
    assert_refusal(*rectangle(), "INPUT_BUDGET")
    monkeypatch.setattr(fw, "MAX_TRIANGLES", 200_000)
    monkeypatch.setattr(fw, "MAX_FRACTION_BITS", 8)
    assert_refusal(*rectangle(), "RATIONAL_CLIP")


def test_mesh_with_hole_is_not_retriangulated_or_filled():
    # Ring tessellation around an omitted rectangle entirely on safe side.
    vertices = [[2, -10, 0], [2, 0, 0], [2, 0, 20], [2, -10, 20],
                [2, -7, 5], [2, -3, 5], [2, -3, 15], [2, -7, 15]]
    triangles = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                 [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    before = copy.deepcopy((vertices, triangles))
    assert certify(vertices, triangles)["intersecting_triangles"] == 8
    assert (vertices, triangles) == before


@pytest.mark.parametrize("updates", [{"side": []}, {"side": "neither"}, {"budget": {}},
                                      {"axial_min": True}, {"seam_endpoints": []}])
def test_invalid_public_arguments_are_typed(updates):
    assert_refusal(*rectangle(), "INVALID", **updates)
