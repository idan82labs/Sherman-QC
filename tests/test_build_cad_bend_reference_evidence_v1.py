"""Pure synthetic checks; importing the collector does not import native OCP."""
import importlib.util
from pathlib import Path
import json
import os
import pytest

spec = importlib.util.spec_from_file_location("neutral_evidence", Path(__file__).parents[1] / "scripts/build_cad_bend_reference_evidence_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_surface_classes_and_failures_remain_in_census():
    faces = [{"id": 1, "surface_class": "Plane", "edge_ids": [1]},
             {"id": 2, "surface_class": "BSplineSurface", "edge_ids": [1]},
             {"id": 3, "surface_class": "Unclassified", "errors": ["native failure"]}]
    result = module.coverage(faces, [{"id": 1, "face_ids": [1, 2]}])
    assert result["surface_census"] == {"BSplineSurface": 1, "Plane": 1, "Unclassified": 1}
    assert result["face_errors"] == [3]
    assert result["bend_reference_complete"] is False


def test_missing_topology_and_duplicate_ids_are_not_silent():
    result = module.coverage([{"id": 1, "surface_class": "Plane", "edge_ids": [7]},
                              {"id": 1, "surface_class": "Plane"}], [{"id": 2, "face_ids": [9]}])
    assert set(result["topology_errors"]) == {"duplicate_index", "missing_edge_index", "missing_face_index"}


def test_nonfinite_bounds_are_explicit_and_valid_json():
    result = module.finite_json({"bounds": [float("inf"), float("nan"), 1.0]})
    assert result["bounds"][0] == {"nonfinite": "inf"}
    json.dumps(result, allow_nan=False)


@pytest.mark.skipif(os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE") != "1", reason="authored native phase requires root gate")
def test_authored_compound_complete_census_and_step_roundtrip(tmp_path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_FACE
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, BRepPrimAPI_MakeBox(3, 4, 5).Shape())
    builder.Add(compound, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(20, 4, 7), gp_Dir(1, 2, 3)), 2, 6).Shape())
    indexed = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, TopAbs_FACE, indexed)
    direct = module.collect_shape(compound)
    assert len(direct["faces"]) == indexed.Extent() == 9
    assert direct["coverage"]["surface_census"] == {"Cylinder": 1, "Plane": 8}
    assert direct["brep_valid"] is True
    assert direct["solid_count"] == 2
    assert not direct["coverage"]["face_errors"]
    assert all(f["solid_ids"] and f["display_mesh"]["triangles_one_based"] for f in direct["faces"])
    assert all(e["face_ids"] for e in direct["edges"])
    assert any("line" in e["carrier"] for e in direct["edges"])
    assert any("circle" in e["carrier"] for e in direct["edges"])
    assert all(w["order_is_boundary_order"] for f in direct["faces"] for w in f["wires"])
    assert all("pcurve" in occurrence and not occurrence["errors"] for f in direct["faces"] for w in f["wires"] for occurrence in w["edge_occurrences"])
    assert [f["id"] for f in direct["faces"]] == list(range(1, 10))
    for face in direct["faces"]:
        for eid in face["edge_ids"]:
            assert face["id"] in direct["edges"][eid - 1]["face_ids"]
    writer = STEPControl_Writer()
    assert writer.Transfer(compound, STEPControl_AsIs) == IFSelect_RetDone
    path = tmp_path / "authored.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    result = module.collect_evidence(path)
    assert result["coverage"]["surface_census"] == direct["coverage"]["surface_census"]
    assert result["automatic_bend_labels"] is None
    assert result["input_sha256"]
