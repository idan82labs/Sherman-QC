"""Pure inherited geometry checks for the V6 diagnostic, no native inputs."""
import copy
import numpy as np
import pytest
from backend import vendor_unfold_preview as native
from backend import vendor_unfold_preview_v5 as v5
from backend import vendor_unfold_preview_v6 as v6
from backend.vendor_unfold_contract_v5 import validate_preview, PreviewContractError
from tests.test_vendor_unfold_coplanar_stitch_v1 import split_two_bend_skin


def test_v6_preserves_all_v5_geometry_and_native_lineage():
    faces, axes, candidates = split_two_bend_skin()
    previous = copy.deepcopy(faces)
    old = v5.develop_faces_v5(faces, axes, candidates, "a" * 64)
    new = v6.develop_faces_v6(faces, axes, candidates, "a" * 64)
    for key in ("vertices", "triangles", "bend_lines", "flange_surfaces"):
        assert new[key] == old[key]
    for key in ("developed_native_face_slices", "chart_topology",
                "selected_native_face_indices", "exported_triangle_winding"):
        assert new["diagnostics"][key] == old["diagnostics"][key]
    assert new["diagnostics"]["mechanism_version"] == 6
    assert new["manufacturing_ready"] is False
    new["diagnostics"]["transport_version"] = 5
    assert validate_preview(new, "a" * 64, candidates) is new
    for actual, before in zip(faces, previous):
        np.testing.assert_array_equal(actual.vertices, before.vertices)
        np.testing.assert_array_equal(actual.triangles, before.triangles)


def test_v6_does_not_accept_missing_axis_faces():
    faces, axes, candidates = split_two_bend_skin()
    axes[0]["face_indices"].append(999)
    with pytest.raises(native.UnfoldUnavailable):
        v6.develop_faces_v6(faces, axes, candidates, "a" * 64)


def test_v6_does_not_modify_incumbent_globals():
    from backend import vendor_unfold_preview_v4 as v4
    envelope, reader = v4._envelope, v5.read_native_v5
    faces, axes, candidates = split_two_bend_skin()
    v6.develop_faces_v6(faces, axes, candidates, "a" * 64)
    assert v4._envelope is envelope and v5.read_native_v5 is reader


@pytest.mark.parametrize("mutation", ["version", "winding", "slice", "overlay"])
def test_v6_contract_refuses_forged_geometry_and_version(mutation):
    faces, axes, candidates = split_two_bend_skin()
    result = v6.develop_faces_v6(faces, axes, candidates, "a" * 64)
    result["diagnostics"]["transport_version"] = 5
    if mutation == "version":
        result["diagnostics"]["mechanism_version"] = 5
    elif mutation == "winding":
        result["triangles"][:3] = result["triangles"][:3][::-1]
    elif mutation == "slice":
        result["diagnostics"]["developed_native_face_slices"][0]["body_vertex_start"] += 1
    else:
        result["flange_surfaces"][0]["triangles"] = result["flange_surfaces"][0]["triangles"][:3]
    with pytest.raises(PreviewContractError):
        validate_preview(result, "a" * 64, candidates)
