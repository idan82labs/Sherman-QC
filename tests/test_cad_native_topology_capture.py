"""Exact-source native-incidence checks using invented Python shape doubles.

Only this named source is read. No backend import, kernel/native package,
filesystem CAD input, service, protected artifact or physical geometry is used.
"""
import ast
import builtins
import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend" / "cad_import.py"


def source_namespace():
    tree = ast.parse(SOURCE.read_text())
    declarations = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    constants = {"MAX_TESSELLATION_FACE_RECORDS", "MAX_NATIVE_TOPOLOGY_EDGE_USES",
                 "MAX_NATIVE_TOPOLOGY_TREE_NODES", "MAX_NATIVE_TOPOLOGY_DEPTH",
                 "MAX_NATIVE_TOPOLOGY_IDENTITY_COMPARISONS"}
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    assert len(nodes) == len(constants)
    nodes.extend(copy.deepcopy(declarations[name]) for name in ("_tessellation_provenance", "CADMesh"))
    importer = copy.deepcopy(declarations["CADImporter"])
    names = {"__init__", "_tessellate", "_capture_native_topology", "_calculate_face_normals"}
    importer.body = [n for n in importer.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in importer.body} == names
    nodes.append(importer)

    def reject_import(name, *args, **kwargs):
        raise AssertionError("Native/package import forbidden: " + name)

    namespace = {"__name__": __name__, "np": np, "dataclass": dataclass, "field": field,
                 "Optional": Optional, "Dict": Dict, "List": List, "Any": Any,
                 "__builtins__": dict(vars(builtins), __import__=reject_import)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


class Shape:
    """TShape token, integer location and orientation; no coordinate identity."""
    def __init__(self, kind, token, children=(), *, location=0, orientation="FORWARD",
                 degenerate=False, null=False, triangulated=True):
        self.kind, self.token, self.children = kind, token, tuple(children)
        self.location, self.orientation = location, orientation
        self.degenerate, self.null, self.triangulated = degenerate, null, triangulated

    def ShapeType(self):
        return self.kind

    def IsNull(self):
        return self.null

    def IsSame(self, other):
        return self.token == other.token and self.location == other.location

    def Orientation(self):
        return self.orientation


def compose(parent, child):
    if parent == "FORWARD":
        return child
    if parent == "REVERSED" and child in ("FORWARD", "REVERSED"):
        return "REVERSED" if child == "FORWARD" else "FORWARD"
    return parent


class Iterator:
    calls = []

    def __init__(self, shape, cum_ori, cum_loc):
        assert cum_ori is True and cum_loc is True
        self.calls.append((shape.token, cum_ori, cum_loc))
        self.children = []
        for child in shape.children:
            current = copy.copy(child)
            current.location += shape.location
            current.orientation = compose(shape.orientation, child.orientation)
            self.children.append(current)
        self.index = 0

    def More(self):
        return self.index < len(self.children)

    def Value(self):
        return self.children[self.index]

    def Next(self):
        self.index += 1


def faces_in_tree(root):
    """Independent invented explorer enumeration, cumulative location/orientation."""
    result = []

    def visit(node, location, orientation):
        current = copy.copy(node)
        current.location = location + node.location
        current.orientation = compose(orientation, node.orientation)
        if node.kind == "FACE":
            result.append(current)
            return
        for child in node.children:
            visit(child, current.location, current.orientation)

    visit(root, 0, "FORWARD")
    return result


def importer(namespace):
    value = namespace["CADImporter"]()
    value._native_topology_api_available = True
    value._TopoDS_Shape = copy.copy
    value._TopoDS_Iterator = Iterator
    value._native_edge_cast = lambda edge: edge
    value._native_orientations = {key: key for key in ("FORWARD", "REVERSED", "INTERNAL", "EXTERNAL")}
    value._BRep_Tool = SimpleNamespace(Degenerated=lambda edge: edge.degenerate)
    for kind in ("SOLID", "SHELL", "FACE", "EDGE"):
        setattr(value, "_TopAbs_" + kind, kind)
    return value


def capture(shape, *, tessellation_faces=None, namespace=None):
    namespace = namespace or source_namespace()
    value = importer(namespace)
    result = value._capture_native_topology(shape, faces_in_tree(shape)
                                           if tessellation_faces is None else tessellation_faces)
    return result


def assert_unavailable(result, reason):
    assert result == {"schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
                      "status": "UNAVAILABLE", "face_records": [], "owner_records": [],
                      "reasons": [reason]}


def pair(*, owner_kind="SOLID"):
    edge = Shape("EDGE", "shared")
    reverse = copy.copy(edge)
    reverse.orientation = "REVERSED"
    first = Shape("FACE", "first", [Shape("WIRE", "wire-a", [edge])])
    second = Shape("FACE", "second", [Shape("WIRE", "wire-b", [reverse])])
    shell = Shape("SHELL", "shell", [first, second])
    return shell if owner_kind == "SHELL" else Shape("SOLID", "solid", [shell])


def test_fifth_field_is_optional_and_not_in_repr():
    namespace = source_namespace()
    mesh = namespace["CADMesh"](np.zeros((0, 3)), np.zeros((0, 3)), None, {"v1": True})
    assert mesh.native_topology is None
    assert "native_topology" not in repr(mesh)
    assert mesh.tessellation_provenance == {"v1": True}


def test_shared_native_edge_keeps_oriented_uses_and_nearest_solid_owner():
    result = capture(pair())
    assert result["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert result["owner_records"] == [{"owner_id": "owner:0", "kind": "SOLID", "occurrence_path": []}]
    records = result["face_records"]
    assert [r["face_id"] for r in records] == ["face:0", "face:1"]
    assert [r["source_face_index"] for r in records] == [0, 1]
    assert [r["edge_uses"] for r in records] == [
        [{"edge_id": "owner:0:edge:0", "orientation": "FORWARD", "degenerate": False}],
        [{"edge_id": "owner:0:edge:0", "orientation": "REVERSED", "degenerate": False}],
    ]
    assert set(result) == {"schema_version", "kind", "status", "face_records", "owner_records", "reasons"}


@pytest.mark.parametrize("kind", ["SHELL", "FACE"])
def test_loose_shell_and_face_are_actual_owner_occurrences(kind):
    node = pair(owner_kind="SHELL") if kind == "SHELL" else Shape("FACE", "loose")
    result = capture(Shape("COMPOUND", "root", [node]))
    assert result["owner_records"] == [{"owner_id": "owner:0", "kind": kind, "occurrence_path": [0]}]


def test_compound_does_not_proxy_as_owner_or_connect_loose_faces():
    edge = Shape("EDGE", "same-kernel-edge")
    root = Shape("COMPOUND", "root", [Shape("FACE", "a", [edge]), Shape("FACE", "b", [edge])])
    result = capture(root)
    assert [r["kind"] for r in result["owner_records"]] == ["FACE", "FACE"]
    assert [r["occurrence_path"] for r in result["owner_records"]] == [[0], [1]]
    records = result["face_records"]
    assert records[0]["edge_uses"][0]["edge_id"] != records[1]["edge_uses"][0]["edge_id"]


def test_located_instances_keep_distinct_owners_and_accumulate_parent_location():
    model = pair()
    other = copy.copy(model)
    other.location = 10
    root = Shape("COMPOUND", "assembly", [model, other], location=100)
    result = capture(root)
    assert result["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert [r["occurrence_path"] for r in result["owner_records"]] == [[0], [1]]
    assert [r["owner_id"] for r in result["face_records"]] == ["owner:0", "owner:0", "owner:1", "owner:1"]
    assert all(call[1:] == (True, True) for call in Iterator.calls)


def test_parent_orientation_is_cumulative_on_edge_uses():
    root = pair()
    root.orientation = "REVERSED"
    result = capture(root)
    assert [r["edge_uses"][0]["orientation"] for r in result["face_records"]] == ["REVERSED", "FORWARD"]


def test_face_ids_follow_exact_tessellation_order_not_tree_order():
    root = pair()
    result = capture(root, tessellation_faces=list(reversed(faces_in_tree(root))))
    assert [r["edge_uses"][0]["orientation"] for r in result["face_records"]] == ["REVERSED", "FORWARD"]
    assert [r["source_face_index"] for r in result["face_records"]] == [0, 1]


@pytest.mark.parametrize("different_owner", [False, True])
def test_repeated_coincident_native_face_clears_every_record(different_owner):
    face = Shape("FACE", "repeated")
    children = [Shape("SOLID", "a", [face]), Shape("SOLID", "b", [face])] if different_owner else [face, face]
    result = capture(Shape("COMPOUND", "root", children))
    assert_unavailable(result, "AMBIGUOUS_REPEATED_NATIVE_FACE_OCCURRENCE")


def test_geometrically_coincident_but_distinct_native_tokens_are_not_welded():
    result = capture(Shape("SOLID", "root", [Shape("FACE", "a"), Shape("FACE", "b")]))
    assert result["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert len(result["face_records"]) == 2


def test_seam_repetitions_and_degenerate_edges_are_retained_not_manifold_claims():
    first = Shape("EDGE", "seam", degenerate=True)
    second = copy.copy(first)
    second.orientation = "REVERSED"
    result = capture(Shape("FACE", "face", [Shape("WIRE", "wire", [first, second])]))
    assert result["face_records"][0]["edge_uses"] == [
        {"edge_id": "owner:0:edge:0", "orientation": "FORWARD", "degenerate": True},
        {"edge_id": "owner:0:edge:0", "orientation": "REVERSED", "degenerate": True},
    ]


@pytest.mark.parametrize("orientation", ["INTERNAL", "EXTERNAL"])
def test_non_boundary_orientation_is_retained_without_reinterpretation(orientation):
    result = capture(Shape("FACE", "face", [Shape("EDGE", "edge", orientation=orientation)]))
    assert result["face_records"][0]["edge_uses"][0]["orientation"] == orientation


@pytest.mark.parametrize("change,reason", [
    ("count", "NATIVE_TESSELLATION_FACE_COUNT_MISMATCH"),
    ("missing", "NATIVE_TESSELLATION_FACE_IDENTITY_MISMATCH"),
    ("repeated", "NATIVE_TESSELLATION_FACE_IDENTITY_MISMATCH"),
    ("location", "NATIVE_TESSELLATION_FACE_IDENTITY_MISMATCH"),
    ("orientation", "NATIVE_TESSELLATION_FACE_ORIENTATION_MISMATCH"),
])
def test_tessellation_correspondence_cannot_be_guessed(change, reason):
    root = pair()
    faces = faces_in_tree(root)
    if change == "count":
        faces.pop()
    elif change == "missing":
        faces[1] = Shape("FACE", "not-in-tree")
    elif change == "repeated":
        faces[1] = faces[0]
    elif change == "location":
        faces[1].location += 1
    else:
        faces[1].orientation = "REVERSED"
    assert_unavailable(capture(root, tessellation_faces=faces), reason)


@pytest.mark.parametrize("key,value,reason", [
    ("MAX_TESSELLATION_FACE_RECORDS", 1, "SOURCE_FACE_RECORD_BUDGET_EXCEEDED"),
    ("MAX_NATIVE_TOPOLOGY_EDGE_USES", 1, "NATIVE_EDGE_USE_BUDGET_EXCEEDED"),
    ("MAX_NATIVE_TOPOLOGY_TREE_NODES", 3, "NATIVE_TREE_NODE_BUDGET_EXCEEDED"),
    ("MAX_NATIVE_TOPOLOGY_DEPTH", 2, "NATIVE_TREE_DEPTH_BUDGET_EXCEEDED"),
    ("MAX_NATIVE_TOPOLOGY_IDENTITY_COMPARISONS", 1, "NATIVE_IDENTITY_COMPARISON_BUDGET_EXCEEDED"),
])
def test_every_cap_clears_complete_mapping_without_a_prefix(key, value, reason):
    namespace = source_namespace()
    namespace[key] = value
    assert_unavailable(capture(pair(), namespace=namespace), reason)


def test_exact_small_work_boundaries_are_inclusive():
    namespace = source_namespace()
    namespace.update(MAX_TESSELLATION_FACE_RECORDS=2, MAX_NATIVE_TOPOLOGY_EDGE_USES=2,
                     MAX_NATIVE_TOPOLOGY_TREE_NODES=8, MAX_NATIVE_TOPOLOGY_DEPTH=4,
                     MAX_NATIVE_TOPOLOGY_IDENTITY_COMPARISONS=6)
    assert capture(pair(), namespace=namespace)["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"


@pytest.mark.parametrize("mutation,reason", [
    ("orientation", "INVALID_NATIVE_EDGE_ORIENTATION"),
    ("degenerate", "INVALID_NATIVE_EDGE_DEGENERATE_FLAG"),
    ("identity", "INVALID_NATIVE_IDENTITY_RESULT"),
    ("api", "NATIVE_TOPOLOGY_APIS_UNAVAILABLE"),
    ("exception", "NATIVE_TOPOLOGY_CAPTURE_FAILED"),
])
def test_native_api_failures_remain_unavailable(mutation, reason):
    namespace = source_namespace()
    value = importer(namespace)
    root = pair()
    if mutation == "orientation":
        value._native_orientations = {}
    elif mutation == "degenerate":
        value._BRep_Tool.Degenerated = lambda edge: 1
    elif mutation == "identity":
        def bad_identity(shape):
            cloned = copy.copy(shape)
            cloned.IsSame = lambda other: 1
            return cloned
        value._TopoDS_Shape = bad_identity
    elif mutation == "api":
        value._native_topology_api_available = False
    else:
        value._TopoDS_Iterator = lambda *args: (_ for _ in ()).throw(RuntimeError("native detail"))
    assert_unavailable(value._capture_native_topology(root, faces_in_tree(root)), reason)


def test_inconsistent_degenerate_flag_for_same_native_edge_is_not_overwritten():
    first = Shape("EDGE", "same", degenerate=False)
    second = Shape("EDGE", "same", degenerate=True)
    result = capture(Shape("FACE", "face", [first, second]))
    assert_unavailable(result, "INCONSISTENT_NATIVE_EDGE_DEGENERATE_FLAG")


def test_empty_and_null_shapes_are_not_complete_topology():
    assert_unavailable(capture(Shape("COMPOUND", "empty")), "NO_NATIVE_FACES")
    face = Shape("FACE", "null", null=True)
    assert_unavailable(capture(face), "NULL_NATIVE_SHAPE")


def test_invalid_face_under_face_hierarchy_is_unavailable():
    root = Shape("FACE", "outer", [Shape("FACE", "inner")])
    assert_unavailable(capture(root), "INVALID_NATIVE_OCCURRENCE_HIERARCHY")


class Point:
    def __init__(self, xyz):
        self.xyz = xyz

    def X(self):
        return self.xyz[0]

    def Y(self):
        return self.xyz[1]

    def Z(self):
        return self.xyz[2]


def tessellation_importer(namespace):
    value = importer(namespace)
    value._BRepMesh_IncrementalMesh = lambda *args: SimpleNamespace(Perform=lambda: None)
    value._TopLoc_Location = lambda: SimpleNamespace(IsIdentity=lambda: True)

    class Explorer:
        def __init__(self, shape, kind):
            assert kind == "FACE"
            self.faces, self.index = faces_in_tree(shape), 0

        def More(self):
            return self.index < len(self.faces)

        def Current(self):
            return self.faces[self.index]

        def Next(self):
            self.index += 1

    value._TopExp_Explorer = Explorer
    xyz = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
    triangulation = SimpleNamespace(NbNodes=lambda: 3, NbTriangles=lambda: 1,
                                   Node=lambda index: Point(xyz[index-1]),
                                   Triangle=lambda index: SimpleNamespace(Get=lambda: (1, 2, 3)))
    value._BRep_Tool.Triangulation = lambda face, location: triangulation if face.triangulated else None
    return value


def test_exact_tessellation_path_captures_incidence_and_preserves_geometry_and_v1():
    namespace = source_namespace()
    root = pair()
    value = tessellation_importer(namespace)
    captured = value._tessellate(root)
    value._native_topology_api_available = False
    unavailable = value._tessellate(root)
    assert captured.native_topology["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE"
    assert unavailable.native_topology["status"] == "UNAVAILABLE"
    for name in ("vertices", "faces", "normals"):
        np.testing.assert_array_equal(getattr(captured, name), getattr(unavailable, name))
    assert captured.tessellation_provenance == unavailable.tessellation_provenance
    assert captured.tessellation_provenance["native_topology_status"] == "UNKNOWN"
    np.testing.assert_array_equal(captured.faces, [[0, 1, 2], [3, 4, 5]])


def test_native_faces_without_triangles_still_have_incidence():
    namespace = source_namespace()
    root = pair()
    root.children[0].children[1].triangulated = False
    mesh = tessellation_importer(namespace)._tessellate(root)
    assert len(mesh.native_topology["face_records"]) == 2
    assert mesh.tessellation_provenance["face_records"][1]["tessellation_status"] == "NO_TRIANGULATION"
    assert mesh.tessellation_provenance["face_records"][1]["triangle_range"] == [1, 1]


@pytest.mark.parametrize("failure", ["copy", "capture"])
def test_unexpected_native_capture_failure_never_changes_geometry(failure):
    namespace = source_namespace()
    value = tessellation_importer(namespace)
    root = pair()
    baseline = value._tessellate(root)
    if failure == "copy":
        value._TopoDS_Shape = lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable"))
    else:
        value._capture_native_topology = lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable"))
    actual = value._tessellate(root)
    assert actual.native_topology["status"] == "UNAVAILABLE"
    assert actual.native_topology["face_records"] == []
    for name in ("vertices", "faces", "normals"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(baseline, name))
    assert actual.tessellation_provenance == baseline.tessellation_provenance
