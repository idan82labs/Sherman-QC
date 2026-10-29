import copy
import pytest
from backend.cad_finite_native_curve_graph_domain_v2 import certify_native_curve_graph_domain


def fixture():
    points = [[[0,0],[0,10]], [[0,10],[.5,12],[1.5,11],[2,10]], [[2,10],[2,0]], [[2,0],[1.5,-1],[.5,-2],[0,0]]]
    curves=[]
    for i,poles in enumerate(points):
        basis=dict(kind="LINE",degree=1,actual_interval=[0.,1.],restricted_interval=[0.,1.],actual_interval_proven=True)
        if i in (1,3): basis.update(kind="BSPLINE",degree=2,knots=[0.,.5,1.],multiplicities=[3,1,3],periodic=False)
        curves.append(dict(native_edge_id=i+1,native_start_vertex_id=i+1,native_end_vertex_id=(i+1)%4+1,poles=poles,weights=[1.]*len(poles),basis=basis,kernel_vertex_tolerance_mm=.1))
    return curves


def certify(curves, **kw):
    return certify_native_curve_graph_domain(curves,face_tolerance_mm=1e-7,native_brep_valid=True,complete_outer_wire=True,chart_branch_proven=True,**kw)


def test_original_rational_bspline_no_decomposition_and_uncertainty_disclosed():
    c=fixture(); c[1]["weights"]=[1,.1,10,1]
    result=certify(c)
    assert result["status"]=="PROVED"
    assert result["ordered_original_native_curves"]==c
    assert result["kernel_vertex_uncertainty_retained"]==[.1]*4
    assert result["full_boundary_geometry_calibrated"] is False
    assert result["safe_rectangular_interior_core_mm"][0]>0


@pytest.mark.parametrize("mutation,reason", [
    (lambda c:c[1]["basis"].update(multiplicities=[3,3,3]),"UNCLAMPED_OR_DISCONTINUOUS_BASIS"),
    (lambda c:c[1]["basis"].update(actual_interval=[-1e-15,1.]),"EXACT_ACTIVE_INTERVAL_REQUIRED"),
    (lambda c:c[1]["weights"].__setitem__(1,0),"NONPOSITIVE_WEIGHT"),
    (lambda c:c[1]["poles"][1].__setitem__(0,3),"CONTROL_MONOTONICITY_UNPROVEN"),
    (lambda c:c[1].update(native_start_vertex_id=99),"SHARED_NATIVE_VERTEX_IDENTITY_REQUIRED"),
])
def test_invalid_basis_geometry_or_topology_refuses(mutation,reason):
    c=fixture(); mutation(c)
    assert certify(c)["reason"]==reason


def test_tiny_join_and_rail_band_erosion_without_snapping():
    c=fixture(); c[0]["poles"][1][0]=1e-9; c[1]["poles"][0][0]=2e-9
    result=certify(c)
    assert result["status"]=="PROVED"
    assert result["ordered_original_native_curves"][0]["poles"][1][0]==1e-9
    assert result["safe_rectangular_interior_core_mm"][0]>1e-7


def test_periodic_basis_and_missing_wire_census_refuse():
    c=fixture(); c[1]["basis"]["periodic"]=True
    assert certify(c)["status"]=="REFUSED"
    assert certify_native_curve_graph_domain(fixture(),face_tolerance_mm=1e-7,native_brep_valid=True,complete_outer_wire=False,chart_branch_proven=True)["status"]=="REFUSED"


def test_flat_active_span_not_a_single_valued_end_graph():
    c=fixture(); c[1]["poles"]=[[0,10],[0,11],[0,12],[2,10]]
    assert certify(c)["reason"]=="CONSTANT_COORDINATE_ACTIVE_SPAN"


def test_oriented_reverse_preserves_original_basis_and_core():
    c=fixture(); original=certify(c)
    reversed_curves=[]
    for curve in reversed(c):
        item=copy.deepcopy(curve)
        item["poles"].reverse(); item["weights"].reverse()
        item["native_start_vertex_id"],item["native_end_vertex_id"]=item["native_end_vertex_id"],item["native_start_vertex_id"]
        if item["basis"]["kind"]=="BSPLINE":
            knots=item["basis"]["knots"]
            item["basis"]["knots"]=[knots[0]+knots[-1]-k for k in reversed(knots)]
            item["basis"]["multiplicities"].reverse()
        reversed_curves.append(item)
    result=certify(reversed_curves)
    assert result["status"]=="PROVED"
    assert result["safe_rectangular_interior_core_mm"]==original["safe_rectangular_interior_core_mm"]
