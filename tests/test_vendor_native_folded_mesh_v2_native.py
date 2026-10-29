"""Authored geometry only; gated into a root-supervised native phase."""
import os
import importlib.util
from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE") != "1",
    reason="root-coordinated bounded native phase required")


def test_indexed_identity_matches_same_not_orientation_or_geometry():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(10, 0, 0)).Edge()
    same_geometry = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(10, 0, 0)).Edge()
    transform = gp_Trsf(); transform.SetTranslation(gp_Vec(0, 2, 0))
    translated = edge.Moved(TopLoc_Location(transform))
    candidates = [edge, edge.Reversed(), same_geometry, translated]
    index = TopTools_IndexedMapOfShape()
    for candidate in candidates:
        before = index.Extent()
        expected = next((i for i in range(1, before + 1)
                         if candidate.IsSame(index.FindKey(i))), 0)
        assert index.FindIndex(candidate) == expected
        actual = index.Add(candidate)
        assert actual == (expected or before + 1)
    assert index.Extent() == 3


def test_indexed_display_matches_incumbent_complete_authored_material(tmp_path):
    source = Path(__file__).with_name("test_extract_step_topology_bend_axes_v3_native.py")
    spec = importlib.util.spec_from_file_location("authored_bends", source)
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    from backend.vendor_native_folded_mesh_v1 import read_folded_native as incumbent
    from backend.vendor_native_folded_mesh_v2 import read_folded_native as candidate
    path = tmp_path / "authored.step"
    fixtures.write_step(fixtures.authored_l_bend(), path)
    left, right = incumbent(path), candidate(path)
    assert len(left) == len(right) == 10
    for a, b in zip(left, right):
        assert a.index == b.index and a.native_orientation == b.native_orientation
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.triangles, b.triangles)
