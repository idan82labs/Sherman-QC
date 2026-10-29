"""Synthetic topology and visual-winding successor checks, no physical CAD."""
import copy
import math

import numpy as np
import pytest

from backend import vendor_unfold_preview as native
from backend import vendor_unfold_preview_v5 as v5
from backend.vendor_unfold_contract_v4 import validate_preview,PreviewContractError
from tests.test_vendor_unfold_coplanar_stitch_v1 import split_two_bend_skin


def payload():
    faces,axes,candidates=split_two_bend_skin()
    result=v5.develop_faces_v5(faces,axes,candidates,"a"*64)
    result["diagnostics"]["transport_version"]=4
    return faces,candidates,result


def test_split_flange_recovers_both_bends_with_exact_native_slices():
    faces,candidates,result=payload()
    assert validate_preview(result,"a"*64,candidates) is result
    diag=result["diagnostics"]
    assert diag["selected_native_face_indices"]==list(range(6))
    assert diag["face_index_space"]=="STITCHED_NATIVE_CHART"
    assert len(diag["developed_native_face_slices"])==6
    points=np.asarray(result["vertices"]).reshape(-1,3)
    assert np.ptp(points[:,0])==pytest.approx(12.)
    assert np.ptp(points[:,1])==pytest.approx(47.+4.*math.pi)
    tri=np.asarray(result["triangles"]).reshape(-1,3)
    for row in diag["developed_native_face_slices"]:
        source=faces[row["source_face_index"]]
        local=tri[row["body_triangle_start"]:row["body_triangle_start"]+row["body_triangle_count"]]-row["body_vertex_start"]
        assert np.array_equal(np.sort(local,axis=1),np.sort(source.triangles,axis=1))


def test_winding_repair_preserves_geometry_and_all_overlays():
    faces,candidates,result=payload()
    for surface in [result,*result["flange_surfaces"]]:
        p=np.asarray(surface["vertices"]).reshape(-1,3)
        q=p[np.asarray(surface["triangles"]).reshape(-1,3)]
        cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
        assert np.all(cross[:,2]>0)
    assert result["diagnostics"]["orientation_changes_vertex_geometry"] is False


@pytest.mark.parametrize("field",["body_vertex_start","body_triangle_count","source_face_index"])
def test_forged_source_slice_refuses(field):
    _,candidates,result=payload()
    result["diagnostics"]["developed_native_face_slices"][0][field]+=1
    with pytest.raises(PreviewContractError,match="V5_"):
        validate_preview(result,"a"*64,candidates)


def test_opposite_export_winding_refuses_even_when_double_side_would_display():
    _,candidates,result=payload()
    result["triangles"][:3]=result["triangles"][:3][::-1]
    with pytest.raises(PreviewContractError,match="V5_FLAT_WINDING_INVALID"):
        validate_preview(result,"a"*64,candidates)


def test_orient_function_never_moves_vertices_and_refuses_collapsed_triangle():
    p=[0.,0.,0.,1.,0.,0.,0.,1.,0.]
    old=p.copy()
    assert v5._orient_flat(p,[0,2,1])==([0,1,2],1)
    assert p==old
    with pytest.raises(native.UnfoldUnavailable,match="DEGENERATE"):
        v5._orient_flat([0.,0.,0.,1.,0.,0.,2.,0.,0.],[0,1,2])


def test_incumbent_globals_unchanged_and_inputs_not_mutated():
    from backend import vendor_unfold_preview_v4 as old
    envelope=old._envelope
    faces,axes,candidates=split_two_bend_skin()
    original=copy.deepcopy(faces)
    v5.develop_faces_v5(faces,axes,candidates,"a"*64)
    assert old._envelope is envelope
    for source,prior in zip(faces,original):
        assert np.array_equal(source.vertices,prior.vertices)
        assert np.array_equal(source.triangles,prior.triangles)


def test_overlay_winding_cannot_disagree_with_body():
    _,candidates,result=payload()
    surface=result["flange_surfaces"][0]
    surface["triangles"]=np.asarray(surface["triangles"]).reshape(-1,3)[:,::-1].reshape(-1).tolist()
    with pytest.raises(PreviewContractError,match="V5_FLANGE_OVERLAY"):
        validate_preview(result,"a"*64,candidates)


def test_selected_chart_cannot_forge_boundary_wall_role():
    _,candidates,result=payload()
    for chart in result["diagnostics"]["chart_topology"]["charts"]:
        chart["kind"]="extruded_boundary"
    with pytest.raises(PreviewContractError,match="V5_CHART_KIND|V5_SELECTED_CHART"):
        validate_preview(result,"a"*64,candidates)


def test_swapped_flange_ids_or_partial_overlay_refuses():
    _,candidates,result=payload()
    a,b=result["flange_surfaces"]
    a["feature_id"],b["feature_id"]=b["feature_id"],a["feature_id"]
    result["flange_surfaces"].sort(key=lambda row:row["feature_id"])
    with pytest.raises(PreviewContractError,match="V5_FLANGE_OVERLAY"):
        validate_preview(result,"a"*64,candidates)
    _,candidates,result=payload()
    result["flange_surfaces"][0]["triangles"]=result["flange_surfaces"][0]["triangles"][:3]
    with pytest.raises(PreviewContractError,match="V5_FLANGE_OVERLAY"):
        validate_preview(result,"a"*64,candidates)


def test_negative_reorientation_count_refuses():
    _,candidates,result=payload()
    result["diagnostics"]["reoriented_triangle_count"]=-1
    with pytest.raises(PreviewContractError,match="V5_REORIENTATION_COUNT"):
        validate_preview(result,"a"*64,candidates)
