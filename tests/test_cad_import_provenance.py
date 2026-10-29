"""Source-extracted CAD bookkeeping checks with Python-only native doubles.

No backend import, OCC/gmsh package, CAD kernel, filesystem input, service or
physical artifact is executed. The named source files are the only files read. The
mocked tessellation API supplies invented coordinates and native return values.
"""
import ast
import builtins
import copy
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend" / "cad_import.py"


def extracted_source(*, gmsh_double=None):
    tree = ast.parse(SOURCE.read_text())
    declarations = {node.name: node for node in tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    nodes = []
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "MAX_TESSELLATION_FACE_RECORDS"
                        for t in node.targets)):
            nodes.append(copy.deepcopy(node))
    assert len(nodes) == 1
    for name in ("_tessellation_provenance", "CADFormat", "BoundingBox", "CADMesh", "CADImportResult"):
        nodes.append(copy.deepcopy(declarations[name]))
    importer = copy.deepcopy(declarations["CADImporter"])
    importer.body = [node for node in importer.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("__init__", "_tessellate", "_calculate_face_normals")]
    assert len(importer.body) == 3
    nodes.extend([importer, copy.deepcopy(declarations["_import_with_gmsh"])])
    import_calls = []

    def guarded_import(name, *args, **kwargs):
        import_calls.append(name)
        if name == "gmsh" and gmsh_double is not None:
            return gmsh_double
        raise AssertionError(f"Unexpected import from extracted source: {name}")

    class SyntheticPath:
        def __init__(self, value):
            self.value = value
            self.suffix = ".step"

        def __str__(self):
            return self.value

    namespace = {
        "__name__": __name__, "__builtins__": dict(vars(builtins), __import__=guarded_import),
        "np": np, "dataclass": dataclass, "field": field, "Enum": Enum,
        "Optional": Optional, "Dict": Dict, "List": List, "Tuple": Tuple, "Any": Any,
        "Path": SyntheticPath, "logger": SimpleNamespace(error=lambda *args, **kwargs: None),
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    assert import_calls == []
    return namespace, import_calls


class Node:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)

    def X(self):
        return float(self.xyz[0])

    def Y(self):
        return float(self.xyz[1])

    def Z(self):
        return float(self.xyz[2])

    def Transformed(self, translation):
        return Node(self.xyz+translation)


class Triangulation:
    def __init__(self, vertices, triangles, *, reported_nodes=None, reported_triangles=None):
        self.vertices = list(vertices)
        self.triangles = list(triangles)
        self.reported_nodes = len(vertices) if reported_nodes is None else reported_nodes
        self.reported_triangles = len(triangles) if reported_triangles is None else reported_triangles

    def NbNodes(self):
        return self.reported_nodes

    def Node(self, index):
        return Node(self.vertices[index-1])

    def NbTriangles(self):
        return self.reported_triangles

    def Triangle(self, index):
        return SimpleNamespace(Get=lambda: self.triangles[index-1])


class Face:
    def __init__(self, triangulation, translation=(0., 0., 0.)):
        self.triangulation = triangulation
        self.translation = np.asarray(translation, dtype=float)
        # The adapter must not guess native ownership from arbitrary properties.
        self.owner_id = "not-a-verified-owner"


class Location:
    def __init__(self):
        self.translation = np.zeros(3)

    def IsIdentity(self):
        return bool(np.array_equal(self.translation, np.zeros(3)))

    def Transformation(self):
        return self.translation


class Explorer:
    def __init__(self, faces):
        self.faces, self.index = faces, 0

    def More(self):
        return self.index < len(self.faces)

    def Current(self):
        return self.faces[self.index]

    def Next(self):
        self.index += 1


def importer_with_doubles(namespace):
    importer = namespace["CADImporter"](linear_deflection=0.05, angular_deflection=0.5)
    calls = []

    def mesh_constructor(*args):
        calls.append(("mesh", args))
        return SimpleNamespace(Perform=lambda: calls.append(("perform",)))

    def triangulation(face, location):
        location.translation = face.translation.copy()
        return face.triangulation

    importer._BRepMesh_IncrementalMesh = mesh_constructor
    importer._TopExp_Explorer = lambda shape, kind: Explorer(shape)
    importer._TopAbs_FACE = "mock-face-kind"
    importer._TopLoc_Location = Location
    importer._BRep_Tool = SimpleNamespace(Triangulation=triangulation)
    return importer, calls


def assert_unknown_native_topology(provenance):
    json.dumps(provenance, allow_nan=False)
    assert provenance["schema_version"] == 1
    assert provenance["kind"] == "CAD_TESSELLATION_PROVENANCE_V1"
    assert provenance["native_topology_status"] == "UNKNOWN"
    assert provenance["native_owner_identity_status"] == "UNKNOWN"
    assert provenance["native_edge_identity_status"] == "UNKNOWN"
    assert provenance["source_id"] is None
    assert provenance["source_identity_is_verified"] is False
    assert provenance["frame_id"] is None
    assert provenance["frame_is_verified"] is False
    assert provenance["units"] is None
    assert provenance["units_verified"] is False
    assert provenance["source_face_identity_scope"] == "IMPORT_LOCAL_ENUMERATION_ORDINAL_NOT_PERSISTENT_NATIVE_ID"
    assert "NOT_ELIGIBLE_AS_COMPLETE_CAD_ROOT_GROUPING_TOPOLOGY" in provenance["limitations"]
    for record in provenance["face_records"]:
        assert record["owner_id"] is None
        assert record["edge_ids"] is None


def triangle_sources(provenance):
    result = [None]*provenance["triangle_count"]
    for record in provenance["face_records"]:
        start, stop = record["triangle_range"]
        for index in range(start, stop):
            assert result[index] is None
            result[index] = record["source_face_index"]
    assert all(value is not None for value in result)
    return result


def test_old_cadmesh_constructors_and_import_result_serialization_remain_compatible():
    namespace, imports = extracted_source()
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    normals = np.asarray([[0., 0., 1.]])
    mesh = namespace["CADMesh"](vertices, triangles, normals)
    assert mesh.vertices is vertices and mesh.faces is triangles and mesh.normals is normals
    assert mesh.tessellation_provenance is None
    assert namespace["CADMesh"](vertices, triangles).tessellation_provenance is None
    result = namespace["CADImportResult"](True, namespace["CADFormat"].STEP, mesh=mesh)
    before = result.to_dict()
    mesh.tessellation_provenance = {"invented": "optional sidecar"}
    assert result.to_dict() == before
    assert "tessellation_provenance" not in before
    assert imports == []


def test_occ_ranges_map_every_emitted_triangle_and_preserve_missing_and_empty_faces():
    namespace, imports = extracted_source()
    importer, calls = importer_with_doubles(namespace)
    first = Triangulation([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)],
                          [(1, 2, 3), (1, 3, 4)])
    empty = Triangulation([(4., 5., 6.), (5., 5., 6.), (4., 6., 6.)], [])
    reversed_triangle = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 3, 2)])
    faces = [Face(first), Face(None), Face(empty), Face(reversed_triangle, (10., 0., 0.))]
    mesh = importer._tessellate(faces)
    expected_vertices = np.asarray([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [4., 5., 6.], [5., 5., 6.], [4., 6., 6.],
        [10., 0., 0.], [11., 0., 0.], [10., 1., 0.],
    ], dtype=np.float64)
    assert np.array_equal(mesh.vertices, expected_vertices)
    assert np.array_equal(mesh.faces, np.asarray([[0, 1, 2], [0, 2, 3], [7, 9, 8]], dtype=np.int32))
    assert np.array_equal(mesh.normals, np.asarray([[0., 0., 1.], [0., 0., 1.], [0., 0., -1.]]))
    assert mesh.vertices.dtype == np.float64 and mesh.faces.dtype == np.int32
    assert calls == [("mesh", (faces, 0.05, False, 0.5, True)), ("perform",)]
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["backend"] == "pythonocc"
    assert provenance["mapping_status"] == "FACE_OCCURRENCE_RANGES"
    assert provenance["source_face_count"] == 4
    assert provenance["vertex_count"] == 10 and provenance["triangle_count"] == 3
    assert provenance["source_faces_without_triangles"] == 2
    assert provenance["reasons"] == ["SOURCE_FACES_WITHOUT_TRIANGLES"]
    assert [r["source_face_index"] for r in provenance["face_records"]] == [0, 1, 2, 3]
    assert [r["vertex_range"] for r in provenance["face_records"]] == [[0, 4], [4, 4], [4, 7], [7, 10]]
    assert [r["triangle_range"] for r in provenance["face_records"]] == [[0, 2], [2, 2], [2, 2], [2, 3]]
    assert [r["tessellation_status"] for r in provenance["face_records"]] == [
        "TESSELLATED", "NO_TRIANGULATION", "EMPTY_TRIANGULATION", "TESSELLATED"]
    assert triangle_sources(provenance) == [0, 0, 3]
    assert imports == []


def test_coincident_face_coordinates_are_not_welded_or_promoted_to_common_owner():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    native = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 2, 3)])
    mesh = importer._tessellate([Face(native), Face(native)])
    assert np.array_equal(mesh.vertices[:3], mesh.vertices[3:])
    assert np.array_equal(mesh.faces, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert triangle_sources(provenance) == [0, 1]
    assert provenance["source_face_count"] == 2
    assert provenance["reasons"] == []


def test_actual_tessellation_sidecar_survives_actual_opt_in_loader_handoff():
    namespace, native_imports = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    first = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 2, 3)])
    last = Triangulation([(2., 0., 0.), (3., 0., 0.), (2., 1., 0.)], [(1, 2, 3)])
    mesh = importer._tessellate([Face(first), Face(None), Face(last)])
    vertices_before, faces_before = mesh.vertices.copy(), mesh.faces.copy()
    provenance_before = copy.deepcopy(mesh.tessellation_provenance)
    imported = namespace["CADImportResult"](True, namespace["CADFormat"].STEP, mesh=mesh)

    pipeline_source = SOURCE.parent / "bend_inspection_pipeline.py"
    tree = ast.parse(pipeline_source.read_text())
    loader = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_cad_geometry")
    extensions = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "CAD_BREP_EXTS" for t in node.targets))
    import_calls, cad_calls = [], []

    def forbidden_io(*args, **kwargs):
        raise AssertionError("The CAD handoff must not open any geometry path")

    open3d_double = SimpleNamespace(io=SimpleNamespace(
        read_triangle_mesh=forbidden_io, read_point_cloud=forbidden_io))

    def guarded_import(name, *args, **kwargs):
        import_calls.append(name)
        if name == "open3d":
            return open3d_double
        raise AssertionError(f"Unexpected import from extracted loader: {name}")

    def import_cad_file(path, *, deflection):
        cad_calls.append((path, deflection))
        return imported

    loader_namespace = {
        "__name__": __name__, "__builtins__": dict(vars(builtins), __import__=guarded_import),
        "np": np, "copy": copy, "Path": Path, "Tuple": Tuple, "Optional": Optional,
        "Dict": Dict, "Any": Any, "import_cad_file": import_cad_file,
    }
    module = ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(extensions), copy.deepcopy(loader)], type_ignores=[]))
    exec(compile(module, str(pipeline_source), "exec"), loader_namespace)
    assert import_calls == []
    vertices, faces, source, provenance = loader_namespace["load_cad_geometry"](
        "invented-handoff.step", include_provenance=True)

    assert vertices is mesh.vertices and faces is mesh.faces
    assert np.array_equal(vertices, vertices_before) and np.array_equal(faces, faces_before)
    assert source == "cad_import"
    assert provenance == provenance_before
    assert provenance is not mesh.tessellation_provenance
    assert provenance["face_records"] is not mesh.tessellation_provenance["face_records"]
    assert [r["triangle_range"] for r in provenance["face_records"]] == [[0, 1], [1, 1], [1, 2]]
    assert [r["vertex_range"] for r in provenance["face_records"]] == [[0, 3], [3, 3], [3, 6]]
    assert provenance["source_face_count"] == 3 and provenance["source_faces_without_triangles"] == 1
    assert provenance["vertex_count"] == 6 and provenance["triangle_count"] == 2
    assert triangle_sources(provenance) == [0, 2]
    assert_unknown_native_topology(provenance)
    provenance["face_records"][0]["triangle_range"][1] = 999
    assert mesh.tessellation_provenance == provenance_before  # No mutable sidecar alias.
    assert native_imports == [] and import_calls == ["open3d"]
    assert cad_calls == [("invented-handoff.step", 0.05)]


@pytest.mark.parametrize("extra", [0, 1])
def test_face_record_cap_preserves_geometry_and_never_returns_a_mapping_prefix(extra):
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    cap = namespace["MAX_TESSELLATION_FACE_RECORDS"]
    native = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 2, 3)])
    faces = [Face(native)]+[Face(None)]*(cap+extra-1)
    mesh = importer._tessellate(faces)
    assert np.array_equal(mesh.vertices, np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
    assert np.array_equal(mesh.faces, np.asarray([[0, 1, 2]], dtype=np.int32))
    assert np.array_equal(mesh.normals, np.asarray([[0., 0., 1.]]))
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["source_face_count"] == cap+extra
    assert provenance["vertex_count"] == 3 and provenance["triangle_count"] == 1
    if extra:
        assert provenance["mapping_status"] == "UNAVAILABLE"
        assert provenance["face_records"] == []
        assert provenance["source_faces_without_triangles"] is None
        assert provenance["reasons"] == ["SOURCE_FACE_RECORD_BUDGET_EXCEEDED"]
    else:
        assert provenance["mapping_status"] == "FACE_OCCURRENCE_RANGES"
        assert len(provenance["face_records"]) == cap
        assert provenance["face_records"][-1]["source_face_index"] == cap-1
        assert triangle_sources(provenance) == [0]


@pytest.mark.parametrize("reported_nodes,reported_triangles", [(-1, 0), (0, -1), (True, 0)])
def test_malformed_native_counts_make_mapping_unavailable_without_rescuing_geometry(reported_nodes, reported_triangles):
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    native = Triangulation([(0., 0., 0.)], [], reported_nodes=reported_nodes,
                           reported_triangles=reported_triangles)
    mesh = importer._tessellate([Face(native)])
    assert mesh.num_vertices == (1 if reported_nodes is True else 0)
    assert mesh.num_faces == 0 and mesh.normals is None
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["mapping_status"] == "UNAVAILABLE"
    assert provenance["face_records"] == []
    assert "INVALID_NATIVE_FACE_TESSELLATION_COUNTS" in provenance["reasons"]


def test_numpy_native_integer_counts_retain_exact_mapping():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    native = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 2, 3)],
                           reported_nodes=np.int64(3), reported_triangles=np.int64(1))
    mesh = importer._tessellate([Face(native)])
    assert mesh.tessellation_provenance["mapping_status"] == "FACE_OCCURRENCE_RANGES"
    assert triangle_sources(mesh.tessellation_provenance) == [0]


@pytest.mark.parametrize("local_index", [0, -1, 1.5, True, np.bool_(False)])
def test_bad_local_triangle_indices_clear_all_mapping_even_when_globally_indexable(local_index):
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    first = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(1, 2, 3)])
    second = Triangulation([(2., 0., 0.), (3., 0., 0.), (2., 1., 0.)], [(local_index, 2, 3)])
    mesh = importer._tessellate([Face(first), Face(second)])
    # This is the unchanged legacy flatten-and-cast behavior, not a repair.
    # In particular, local zero resolves to global index 2 in the FIRST face.
    expected_index = int(local_index-1+3)
    assert np.array_equal(mesh.faces, np.asarray([[0, 1, 2], [expected_index, 4, 5]], dtype=np.int32))
    assert mesh.num_vertices == 6 and mesh.num_faces == 2
    assert np.isfinite(mesh.normals).all()
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["mapping_status"] == "UNAVAILABLE"
    assert provenance["face_records"] == []  # Never return the valid first-face prefix.
    assert provenance["source_face_count"] == 2
    assert provenance["reasons"] == ["INVALID_NATIVE_FACE_TRIANGLE_INDICES"]


def test_bad_upper_local_index_into_a_later_face_does_not_gain_valid_mapping():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    first = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(4, 2, 3)])
    second = Triangulation([(2., 0., 0.), (3., 0., 0.), (2., 1., 0.)], [(1, 2, 3)])
    mesh = importer._tessellate([Face(first), Face(second)])
    assert np.array_equal(mesh.faces, np.asarray([[3, 1, 2], [3, 4, 5]], dtype=np.int32))
    assert mesh.tessellation_provenance["mapping_status"] == "UNAVAILABLE"
    assert mesh.tessellation_provenance["face_records"] == []
    assert mesh.tessellation_provenance["reasons"] == ["INVALID_NATIVE_FACE_TRIANGLE_INDICES"]


def test_globally_out_of_bounds_native_triangle_retains_existing_index_exception():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    native = Triangulation([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [(4, 2, 3)])
    with pytest.raises(IndexError):
        importer._tessellate([Face(native)])


def test_existing_noninteger_native_count_exception_is_not_silently_repaired():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    native = Triangulation([], [], reported_nodes=1.5)
    with pytest.raises(TypeError):
        importer._tessellate([Face(native)])


def test_empty_shape_preserves_legacy_empty_array_shapes_and_has_no_physical_claim():
    namespace, _ = extracted_source()
    importer, _ = importer_with_doubles(namespace)
    mesh = importer._tessellate([])
    assert mesh.vertices.shape == mesh.faces.shape == (0,)
    assert mesh.vertices.dtype == np.float64 and mesh.faces.dtype == np.int32
    assert mesh.normals is None
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["source_face_count"] == 0
    assert provenance["vertex_count"] == provenance["triangle_count"] == 0
    assert provenance["face_records"] == []


def gmsh_double(*, include_missing_node=False):
    calls = []
    node_rows = [11, 13, 17]+([11, 13, 999] if include_missing_node else [])
    element_tags = [101, 102] if include_missing_node else [101]
    mesh = SimpleNamespace(
        generate=lambda dimension: calls.append(("generate", dimension)),
        getNodes=lambda: (np.asarray([11, 13, 17]), np.asarray([0., 0., 0., 1., 0., 0., 0., 1., 0.]), []),
        getElements=lambda dimension: ([2], [np.asarray(element_tags)], [np.asarray(node_rows)]),
        getElementProperties=lambda kind: ("Triangle", 2, 1, 3, None, None),
    )
    result = SimpleNamespace(
        initialize=lambda: calls.append(("initialize",)),
        finalize=lambda: calls.append(("finalize",)),
        option=SimpleNamespace(setNumber=lambda name, value: calls.append(("option", name, value))),
        model=SimpleNamespace(
            add=lambda name: calls.append(("add", name)),
            occ=SimpleNamespace(importShapes=lambda path: calls.append(("mock-import", path)),
                                synchronize=lambda: calls.append(("synchronize",))),
            getBoundingBox=lambda *args: (0., 0., 0., 1., 1., 0.),
            mesh=mesh,
        ),
    )
    return result, calls


@pytest.mark.parametrize("include_missing_node", [False, True])
def test_gmsh_source_path_declares_unavailable_mapping_without_fabricating_face_identity(include_missing_node):
    fake, calls = gmsh_double(include_missing_node=include_missing_node)
    namespace, imports = extracted_source(gmsh_double=fake)
    result = namespace["_import_with_gmsh"]("invented-no-file.step", 0.1)
    assert result.success is True
    mesh = result.mesh
    assert np.array_equal(mesh.vertices, np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
    assert np.array_equal(mesh.faces, np.asarray([[0, 1, 2]], dtype=np.int32))
    assert np.array_equal(mesh.normals, np.asarray([[0., 0., 1.]]))
    assert result.num_faces == 1  # Existing gmsh triangle-count field is unchanged.
    provenance = mesh.tessellation_provenance
    assert_unknown_native_topology(provenance)
    assert provenance["backend"] == "gmsh"
    assert provenance["mapping_status"] == "UNAVAILABLE"
    assert provenance["source_face_count"] is None
    assert provenance["face_records"] == []
    assert provenance["vertex_count"] == 3 and provenance["triangle_count"] == 1
    assert provenance["reasons"] == ["GMSH_NATIVE_FACE_ENTITY_MAPPING_UNAVAILABLE"]
    assert imports == ["gmsh"]  # Resolved ONLY by the guarded Python double.
    assert ("mock-import", "invented-no-file.step") in calls
    assert calls.count(("finalize",)) == 1
