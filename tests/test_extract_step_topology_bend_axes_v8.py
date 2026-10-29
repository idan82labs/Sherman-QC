"""Authored pairing negatives; no native or real-input execution."""
import copy
from scripts.extract_step_topology_bend_axes_v8 import _qualify_new_pair


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
