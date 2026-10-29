"""Native-free contract checks for the exact CAD loader source function.

Only the loader's AST is compiled: importing the pipeline would import Open3D
and other runtime modules. All geometry below is invented, and the CAD path
strings are never opened. The importer and both Open3D readers are stubs.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend" / "bend_inspection_pipeline.py"
VERTICES = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
FACES = np.array([[0, 1, 2]], dtype=np.int32)
MISSING = object()


def face_sidecar():
    """An import-local bookkeeping declaration, not native event topology."""
    return {
        "schema_version": 1,
        "kind": "CAD_TESSELLATION_PROVENANCE_V1",
        "backend": "INVENTED_NATIVE_STUB",
        "mapping_status": "FACE_OCCURRENCE_RANGES",
        "mapping_scope": "EMITTED_TRIANGLES_TO_IMPORT_LOCAL_SOURCE_FACE_OCCURRENCE",
        "source_face_identity_scope": "IMPORT_LOCAL_ENUMERATION_ORDINAL_NOT_PERSISTENT_NATIVE_ID",
        "source_face_count": 1,
        "vertex_count": 3,
        "triangle_count": 1,
        "face_records": [{
            "source_face_index": 0,
            "vertex_range": [0, 3],
            "triangle_range": [0, 1],
            "tessellation_status": "TESSELLATED",
            "owner_id": None,
            "edge_ids": None,
        }],
        "max_face_records": 4096,
        "source_faces_without_triangles": 0,
        "native_topology_status": "UNKNOWN",
        "native_owner_identity_status": "UNKNOWN",
        "native_edge_identity_status": "UNKNOWN",
        "source_id": None,
        "source_identity_is_verified": False,
        "frame_id": None,
        "frame_is_verified": False,
        "units": None,
        "units_verified": False,
        "reasons": [],
        "limitations": ["NOT_ELIGIBLE_AS_COMPLETE_CAD_ROOT_GROUPING_TOPOLOGY"],
    }


def imported_mesh(*, sidecar=MISSING, vertices=None, faces=None):
    mesh = SimpleNamespace(
        vertices=VERTICES.copy() if vertices is None else vertices,
        faces=FACES.copy() if faces is None else faces,
    )
    if sidecar is not MISSING:
        mesh.tessellation_provenance = sidecar
    return mesh


def direct_mesh(*, vertices=None, faces=None):
    mesh = SimpleNamespace(
        vertices=VERTICES.copy() if vertices is None else vertices,
        triangles=FACES.copy() if faces is None else faces,
    )
    mesh.has_vertices = lambda: len(mesh.vertices) > 0
    return mesh


def point_cloud(points=None):
    cloud = SimpleNamespace(points=VERTICES.copy() if points is None else points)
    cloud.has_points = lambda: len(cloud.points) > 0
    return cloud


@pytest.fixture
def loader_harness(monkeypatch):
    """Compile the exact function, without executing any pipeline top level."""
    syntax = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    functions = [node for node in syntax.body
                 if isinstance(node, ast.FunctionDef) and node.name == "load_cad_geometry"]
    assert len(functions) == 1
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + functions,
        type_ignores=[],
    )
    calls = []
    state = SimpleNamespace(
        imported=SimpleNamespace(success=True, mesh=imported_mesh(), error=None),
        mesh=direct_mesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)),
        cloud=point_cloud(np.empty((0, 3))),
    )

    def fake_import(path, *, deflection):
        calls.append(("import", path, deflection))
        return state.imported

    def read_mesh(path):
        calls.append(("mesh", path))
        return state.mesh

    def read_cloud(path):
        calls.append(("cloud", path))
        return state.cloud

    native_stub = ModuleType("open3d")
    native_stub.io = SimpleNamespace(read_triangle_mesh=read_mesh, read_point_cloud=read_cloud)
    monkeypatch.setitem(sys.modules, "open3d", native_stub)
    namespace = {
        "np": np, "Path": Path, "copy": copy,
        "CAD_BREP_EXTS": {".step", ".stp", ".iges", ".igs"},
        "import_cad_file": fake_import,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["load_cad_geometry"], state, calls


@pytest.mark.parametrize("suffix", [".step", ".STP", ".iges", ".IGS"])
def test_legacy_importer_tuple_and_deflection_remain_unchanged(loader_harness, suffix):
    load, state, calls = loader_harness
    state.imported.mesh = imported_mesh(sidecar=face_sidecar())
    path = "invented-part" + suffix
    result = load(path, cad_import_deflection=0.07)
    assert len(result) == 3
    vertices, faces, source = result
    np.testing.assert_array_equal(vertices, VERTICES)
    np.testing.assert_array_equal(faces, FACES)
    assert vertices.dtype == np.float64 and faces.dtype == np.int32
    assert source == "cad_import"
    assert calls == [("import", path, 0.07)]


@pytest.mark.parametrize("sidecar", [MISSING, None])
def test_legacy_importer_without_sidecar_still_returns_tuple3(loader_harness, sidecar):
    load, state, _ = loader_harness
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    assert len(load("invented-part.step")) == 3


def test_legacy_direct_mesh_and_cloud_defaults_remain_tuple3(loader_harness):
    load, state, calls = loader_harness
    state.mesh = direct_mesh()
    mesh_result = load("invented-part.stl")
    assert len(mesh_result) == 3 and mesh_result[2] == "open3d_mesh"
    np.testing.assert_array_equal(mesh_result[0], VERTICES)
    np.testing.assert_array_equal(mesh_result[1], FACES)
    assert calls == [("mesh", "invented-part.stl")]

    calls.clear()
    state.mesh = direct_mesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32))
    state.cloud = point_cloud()
    cloud_result = load("invented-part.ply")
    assert len(cloud_result) == 3 and cloud_result[1:] == (None, "open3d_point_cloud")
    np.testing.assert_array_equal(cloud_result[0], VERTICES)
    assert calls == [("mesh", "invented-part.ply"), ("cloud", "invented-part.ply")]


def assert_no_authority(provenance):
    assert provenance["kind"] == "CAD_TESSELLATION_PROVENANCE_V1"
    assert provenance["schema_version"] == 1
    for key in ("native_topology_status", "native_owner_identity_status", "native_edge_identity_status"):
        assert provenance[key] == "UNKNOWN"
    for key in ("source_identity_is_verified", "frame_is_verified", "units_verified"):
        assert provenance[key] is False
    for key in ("source_id", "frame_id", "units"):
        assert provenance[key] is None
    assert provenance["mapping_scope"] == "EMITTED_TRIANGLES_TO_IMPORT_LOCAL_SOURCE_FACE_OCCURRENCE"
    assert provenance["source_face_identity_scope"] == "IMPORT_LOCAL_ENUMERATION_ORDINAL_NOT_PERSISTENT_NATIVE_ID"
    assert provenance["max_face_records"] == 4096


def assert_unavailable(provenance, *, source, vertex_count=3, triangle_count=1):
    assert_no_authority(provenance)
    assert provenance["backend"] == source
    assert provenance["mapping_status"] == "UNAVAILABLE"
    assert provenance["vertex_count"] == vertex_count
    assert provenance["triangle_count"] == triangle_count
    assert provenance["source_face_count"] is None
    assert provenance["source_faces_without_triangles"] is None
    assert provenance["face_records"] == []
    assert "MISSING_OR_INCONSISTENT_TESSELLATION_PROVENANCE" in provenance["reasons"]


def test_opt_in_importer_transports_exact_bookkeeping_snapshot(loader_harness):
    load, state, calls = loader_harness
    sidecar = face_sidecar()
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    result = load("invented-part.step", include_provenance=True)
    assert len(result) == 4
    vertices, faces, source, provenance = result
    np.testing.assert_array_equal(vertices, VERTICES)
    np.testing.assert_array_equal(faces, FACES)
    assert source == "cad_import"
    assert provenance == sidecar and provenance is not sidecar
    assert provenance["face_records"] is not sidecar["face_records"]
    assert provenance["face_records"][0]["vertex_range"] is not sidecar["face_records"][0]["vertex_range"]
    assert_no_authority(provenance)
    assert calls == [("import", "invented-part.step", 0.05)]


def test_sidecar_mutations_are_isolated_in_both_directions_and_between_calls(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    first = load("invented-part.step", include_provenance=True)[3]
    second = load("invented-part.step", include_provenance=True)[3]
    first["face_records"][0]["vertex_range"][0] = 999
    first["reasons"].append("OUTPUT_ONLY_MUTATION")
    assert sidecar == before and second == before
    sidecar["face_records"][0]["triangle_range"][1] = 99
    sidecar["limitations"].append("INPUT_ONLY_MUTATION")
    assert second == before
    assert first["face_records"][0]["triangle_range"] == [0, 1]
    assert "INPUT_ONLY_MUTATION" not in first["limitations"]


def test_default_loader_does_not_copy_or_consult_sidecar_contents(loader_harness):
    load, state, _ = loader_harness

    class UncopyableSidecar(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Default tuple3 path must not inspect provenance contents")

        def __deepcopy__(self, memo):
            raise AssertionError("Default tuple3 path must not copy provenance")

    state.imported.mesh = imported_mesh(sidecar=UncopyableSidecar())
    assert len(load("invented-part.step")) == 3


@pytest.mark.parametrize("sidecar", [MISSING, None, [], "not-a-sidecar", {}])
def test_absent_or_nondictionary_importer_sidecar_is_explicitly_unavailable(loader_harness, sidecar):
    load, state, _ = loader_harness
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    result = load("invented-part.step", include_provenance=True)
    assert len(result) == 4
    assert_unavailable(result[3], source="cad_import")


@pytest.mark.parametrize("key,value", [
    ("kind", "OTHER_PROVENANCE"), ("schema_version", 2),
    ("schema_version", True), ("schema_version", 1.0),
    ("vertex_count", 4), ("triangle_count", 2),
    ("vertex_count", None), ("triangle_count", None),
    ("vertex_count", 3.0), ("triangle_count", True), ("triangle_count", 1.0),
    ("mapping_status", "COMPLETE"), ("mapping_status", "UNKNOWN"),
])
def test_mismatched_schema_or_geometry_counts_discard_mapping(loader_harness, key, value):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar[key] = value
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_unavailable(provenance, source="cad_import")
    assert sidecar == before


def test_canonical_forged_authority_is_downgraded_without_mutating_source(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar.update(
        native_topology_status="COMPLETE", native_owner_identity_status="VERIFIED",
        native_edge_identity_status="VERIFIED", source_id="invented-authority",
        source_identity_is_verified=True, frame_id="invented-frame", frame_is_verified=True,
        units="mm", units_verified=True,
    )
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_no_authority(provenance)
    assert provenance["mapping_status"] == "FACE_OCCURRENCE_RANGES"
    assert provenance["face_records"] == sidecar["face_records"]
    assert sidecar == before


def test_extra_authority_flags_and_per_face_owner_edge_labels_cannot_travel(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar.update(native_topology_verified=True, physical_bend_count=7, release_authorized=True)
    sidecar["face_records"][0].update(
        owner_id="forged-native-owner", edge_ids=["forged-native-edge"], native_face_id="forged-face",
    )
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_no_authority(provenance)
    for key in ("native_topology_verified", "physical_bend_count", "release_authorized"):
        assert key not in provenance
    record = provenance["face_records"][0]
    assert record["owner_id"] is None and record["edge_ids"] is None
    assert "native_face_id" not in record
    assert sidecar == before


def test_scope_and_budget_fields_cannot_claim_persistent_native_identity(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar.update(mapping_scope="VERIFIED_NATIVE_TOPOLOGY",
                   source_face_identity_scope="PERSISTENT_NATIVE_ID",
                   max_face_records=999999)
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_no_authority(provenance)
    assert provenance["face_records"] == sidecar["face_records"]
    assert sidecar == before


@pytest.mark.parametrize("mapping_available", [False, True])
def test_missing_triangle_count_is_derived_not_trusted(loader_harness, mapping_available):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar["source_faces_without_triangles"] = 999
    if not mapping_available:
        sidecar.update(mapping_status="UNAVAILABLE", face_records=[], source_face_count=None)
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert provenance["source_faces_without_triangles"] == (0 if mapping_available else None)
    assert sidecar == before
    assert_no_authority(provenance)


@pytest.mark.parametrize("mutation", [
    "not_a_record", "wrong_index", "bool_index", "tuple_range", "noncontiguous_vertices",
    "noncontiguous_triangles", "negative_range", "bool_range", "float_range",
    "uncovered_vertices", "uncovered_triangles", "overrun_vertices", "overrun_triangles",
    "unknown_status", "empty_status_with_triangle", "source_count_mismatch",
])
def test_malformed_or_uncovered_face_ranges_clear_the_mapping(loader_harness, mutation):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    record = sidecar["face_records"][0]
    if mutation == "not_a_record":
        sidecar["face_records"][0] = None
    elif mutation == "wrong_index":
        record["source_face_index"] = 1
    elif mutation == "bool_index":
        record["source_face_index"] = False
    elif mutation == "tuple_range":
        record["vertex_range"] = (0, 3)
    elif mutation == "noncontiguous_vertices":
        record["vertex_range"][0] = 1
    elif mutation == "noncontiguous_triangles":
        record["triangle_range"][0] = 1
    elif mutation == "negative_range":
        record["vertex_range"][0] = -1
    elif mutation == "bool_range":
        record["triangle_range"][1] = True
    elif mutation == "float_range":
        record["vertex_range"][1] = 3.0
    elif mutation == "uncovered_vertices":
        record["vertex_range"][1] = 2
    elif mutation == "uncovered_triangles":
        record["triangle_range"][1] = 0
    elif mutation == "overrun_vertices":
        record["vertex_range"][1] = 4
    elif mutation == "overrun_triangles":
        record["triangle_range"][1] = 2
    elif mutation == "unknown_status":
        record["tessellation_status"] = "COMPLETE"
    elif mutation == "empty_status_with_triangle":
        record["tessellation_status"] = "EMPTY_TRIANGULATION"
    elif mutation == "source_count_mismatch":
        sidecar["source_face_count"] = 2
    else:
        raise AssertionError("Unknown test construction")
    before = copy.deepcopy(sidecar)
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_unavailable(provenance, source="cad_import")
    assert sidecar == before


def test_later_invalid_face_clears_an_earlier_valid_range(loader_harness):
    load, state, _ = loader_harness
    vertices = np.vstack((VERTICES, VERTICES + [0.0, 0.0, 2.0]))
    faces = np.vstack((FACES, FACES + 3))
    sidecar = face_sidecar()
    sidecar.update(source_face_count=2, vertex_count=6, triangle_count=2)
    second = copy.deepcopy(sidecar["face_records"][0])
    second.update(source_face_index=1, vertex_range=[4, 6], triangle_range=[1, 2])
    sidecar["face_records"].append(second)
    state.imported.mesh = imported_mesh(vertices=vertices, faces=faces, sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_unavailable(provenance, source="cad_import", vertex_count=6, triangle_count=2)


def test_face_record_budget_overflow_does_not_keep_prefix(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar["face_records"] = [copy.deepcopy(sidecar["face_records"][0]) for _ in range(4097)]
    sidecar["source_face_count"] = 4097
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_unavailable(provenance, source="cad_import")


def test_missing_native_face_is_preserved_as_bookkeeping_not_completeness(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar["face_records"][0]["source_face_index"] = 1
    sidecar["face_records"].insert(0, {
        "source_face_index": 0, "vertex_range": [0, 0], "triangle_range": [0, 0],
        "tessellation_status": "NO_TRIANGULATION", "owner_id": None, "edge_ids": None,
    })
    sidecar.update(source_face_count=2, source_faces_without_triangles=1)
    sidecar["reasons"].append("SOURCE_FACES_WITHOUT_TRIANGLES")
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert provenance == sidecar
    assert provenance["source_faces_without_triangles"] == 1
    assert provenance["face_records"][0]["tessellation_status"] == "NO_TRIANGULATION"
    assert_no_authority(provenance)


def test_declared_unavailable_importer_bookkeeping_stays_unavailable(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar.update(mapping_status="UNAVAILABLE", face_records=[], source_face_count=None,
                   source_faces_without_triangles=None)
    sidecar["reasons"] = ["INVENTED_MAPPING_NOT_CAPTURED"]
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert provenance == sidecar
    assert_no_authority(provenance)


def test_unavailable_status_cannot_smuggle_face_records(loader_harness):
    load, state, _ = loader_harness
    sidecar = face_sidecar()
    sidecar["mapping_status"] = "UNAVAILABLE"
    state.imported.mesh = imported_mesh(sidecar=sidecar)
    provenance = load("invented-part.step", include_provenance=True)[3]
    assert_unavailable(provenance, source="cad_import")


@pytest.mark.parametrize("has_faces", [False, True])
def test_direct_mesh_cannot_inherit_fake_native_sidecar(loader_harness, has_faces):
    load, state, calls = loader_harness
    state.mesh = direct_mesh(faces=FACES.copy() if has_faces else np.empty((0, 3), dtype=np.int32))
    state.mesh.tessellation_provenance = face_sidecar()
    result = load("invented-part.obj", include_provenance=True)
    assert len(result) == 4 and result[2] == "open3d_mesh"
    assert_unavailable(result[3], source="open3d_mesh", triangle_count=int(has_faces))
    assert calls == [("mesh", "invented-part.obj")]


def test_point_cloud_fallback_has_no_face_identity(loader_harness):
    load, state, calls = loader_harness
    state.cloud = point_cloud()
    state.cloud.tessellation_provenance = face_sidecar()
    result = load("invented-part.ply", include_provenance=True)
    assert len(result) == 4 and result[1:3] == (None, "open3d_point_cloud")
    assert_unavailable(result[3], source="open3d_point_cloud", triangle_count=0)
    assert calls == [("mesh", "invented-part.ply"), ("cloud", "invented-part.ply")]


def test_importer_zero_faces_preserves_none_and_zero_count(loader_harness):
    load, state, _ = loader_harness
    state.imported.mesh = imported_mesh(faces=np.empty((0, 3), dtype=np.int32))
    result = load("invented-part.step", include_provenance=True)
    assert result[1] is None and result[2] == "cad_import"
    assert_unavailable(result[3], source="cad_import", triangle_count=0)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True)])
def test_provenance_flag_requires_explicit_bool_before_any_reader(loader_harness, flag):
    load, _, calls = loader_harness
    with pytest.raises(ValueError, match="explicit boolean"):
        load("invented-part.step", include_provenance=flag)
    assert calls == []


def test_provenance_flag_is_keyword_only(loader_harness):
    load, _, calls = loader_harness
    with pytest.raises(TypeError):
        load("invented-part.step", 0.05, True)
    assert calls == []


@pytest.mark.parametrize("include_provenance", [False, True])
@pytest.mark.parametrize("failure", ["import_failed", "no_mesh", "empty_vertices"])
def test_importer_failure_cannot_return_sidecar_or_geometry(loader_harness, include_provenance, failure):
    load, state, calls = loader_harness
    if failure == "import_failed":
        state.imported.success = False
        state.imported.error = "invented failure"
    elif failure == "no_mesh":
        state.imported.mesh = None
    else:
        state.imported.mesh = imported_mesh(vertices=np.empty((0, 3)), sidecar=face_sidecar())
    with pytest.raises(ValueError, match="Failed to import CAD|CAD import yielded no vertices"):
        load("invented-part.step", include_provenance=include_provenance)
    assert calls == [("import", "invented-part.step", 0.05)]


@pytest.mark.parametrize("include_provenance", [False, True])
def test_empty_mesh_and_point_cloud_fail_without_fabricated_sidecar(loader_harness, include_provenance):
    load, _, calls = loader_harness
    with pytest.raises(ValueError, match="Failed to load CAD geometry"):
        load("invented-part.ply", include_provenance=include_provenance)
    assert calls == [("mesh", "invented-part.ply"), ("cloud", "invented-part.ply")]
