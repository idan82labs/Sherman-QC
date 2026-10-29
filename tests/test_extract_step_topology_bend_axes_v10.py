"""Authored pairing negatives; no native or real-input execution."""
import copy
from scripts.extract_step_topology_bend_axes_v10 import _qualify_new_pair, _select_safe_marker


def rows():
    return {1:{"radius_mm":2.,"native_face_orientation":"TopAbs_Orientation.TopAbs_REVERSED","native_cap_neighbor_ids":[3,4]},
            2:{"radius_mm":3.,"native_face_orientation":"TopAbs_Orientation.TopAbs_FORWARD","native_cap_neighbor_ids":[3,4]}}


def test_opposite_normals_and_two_shared_caps_qualify():
    assert _qualify_new_pair({"face_indices":[1,2],"canonical_line_start":[1.,1.,0.],"canonical_line_end":[1.,1.,10.]},rows()) == (True,None)


def test_wrong_material_orientation_refuses():
    r=rows();r[1]["native_face_orientation"]="TopAbs_Orientation.TopAbs_FORWARD"
    assert _qualify_new_pair({"face_indices":[1,2]},r)[1] == "OPPOSITE_NATIVE_MATERIAL_NORMALS_REQUIRED"


def test_unrelated_caps_and_skin_neighbors_do_not_qualify():
    for neighbors in ([4,5],[1,2]):
        r=copy.deepcopy(rows());r[2]["native_cap_neighbor_ids"]=neighbors
        assert _qualify_new_pair({"face_indices":[1,2]},r)[1] == "TWO_SHARED_NATIVE_END_CAPS_REQUIRED"


def marker_fixture(reverse=False):
    axis={"face_indices":[1,2],"radii_mm":[2.,3.],"line_start":[0.,0.,0.],
          "carrier_receipt":{"angular_frame":{"axis":[0.,0.,1.],"x_axis":[1.,0.,0.],"y_axis":[0.,1.,0.]}},
          "domain_receipt":{"angular_intervals_rad":[[0.,1.5]],"axial_interval_mm":[0.,10.]}}
    r=rows()
    for i in (1,2):
        r[i].update(graph_native_frame={"origin":[0.,0.,0.],"x":[1.,0.,0.],"y":[0.,1.,0.]},
                    graph_core_angular_native_rad=[.1,.6],
                    graph_core_angular_native_components_rad=[[.1,.6],[.9,1.4]])
    if reverse:
        r[2]["graph_native_frame"]["y"]=[0.,-1.,0.]
        r[2]["graph_core_angular_native_components_rad"]=[[-1.4,-.9],[-.6,-.1]]
        r[2]["graph_core_angular_native_rad"]=[-.6,-.1]
    return axis,r


def test_marker_uses_common_safe_component_not_excluded_middle():
    for reverse in (False,True):
        axis,r=marker_fixture(reverse)
        assert _select_safe_marker(axis,r)
        assert _qualify_new_pair(axis,r)==(True,None)
        theta=axis["domain_receipt"]["marker_source_angle_common_rad"]
        assert .1<theta<.6 or .9<theta<1.4
        assert axis["canonical_line_rule"]=="MID_THICKNESS_SOURCE_COMMON_SAFE_COMPONENT_INTERIOR_MARKER"


def test_disjoint_safe_components_refuse_without_marker_fabrication():
    axis,r=marker_fixture()
    r[2]["graph_core_angular_native_components_rad"]=[[.65,.85]]
    assert not _select_safe_marker(axis,r)
    assert "canonical_line_start" not in axis


def test_narrow_excluded_join_is_never_epsilon_filled():
    axis,r=marker_fixture()
    for patch in r.values():
        patch["graph_core_angular_native_components_rad"]=[[.1,.75-1e-10],[.75+1e-10,1.4]]
    assert _select_safe_marker(axis,r)
    low,high=axis["domain_receipt"]["marker_safe_component_common_rad"]
    assert high<.75 or low>.75
    assert _qualify_new_pair(axis,r)==(True,None)
