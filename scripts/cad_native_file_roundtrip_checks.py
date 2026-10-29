"""Fixed invented-input reader prerequisite; root executes in its isolated cache.

No command-line/data arguments, application imports, existing CAD inputs or
backend promotion. The quarter-cylinder is a CURVED IMPORT CONTROL, not a formed
bend, tangent flange or nominated root. STEP/BREP are shape round trips; IGES
mode 0 deliberately exports faces and loses solid ownership. Neither supplies
persistent source identity, verified physical units, labels or F1/count evidence.

Frozen matrix: 3 shapes x STEP/BREP; one IGES mode-0 box; malformed and absent
files for all three formats = 13 cases. No fixture/threshold rescue after a run.
"""
import ast
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


GEOMETRY_ABSOLUTE_TOLERANCE = 2e-6
MAX_GENERATED_FILE_BYTES = 8 * 1024**2
# This bounds the accepted generated file AFTER write, not native write admission.
ROUNDTRIP_MATRIX = (
    ("box", "step", "solid_preserved"),
    ("box", "brep", "solid_preserved"),
    ("quarter_cylinder", "step", "solid_preserved"),
    ("quarter_cylinder", "brep", "solid_preserved"),
    ("located_compound", "step", "solid_preserved"),
    ("located_compound", "brep", "solid_preserved"),
    ("box", "iges", "surface_mode_solid_ownership_lost"),
    ("malformed", "step", "import_failure"),
    ("malformed", "brep", "import_failure"),
    ("malformed", "iges", "import_failure"),
    ("absent", "step", "import_failure"),
    ("absent", "brep", "import_failure"),
    ("absent", "iges", "import_failure"),
)


def _isolated_cache():
    """Require the root worker's fresh, consistently isolated HOME/cwd/TMPDIR."""
    cache = Path(os.environ["TMPDIR"]).resolve()
    root = Path(__file__).resolve().parents[1]
    if (cache != Path.cwd().resolve() or cache != Path(os.environ["HOME"]).resolve()
            or cache == root or root in cache.parents):
        raise RuntimeError("Root-owned isolated cache required")
    return cache


def _importer_namespace(np):
    source = Path(__file__).resolve().parents[1] / "backend" / "cad_import.py"
    tree = ast.parse(source.read_text())
    names = {"_select_native_cad_binding", "_tessellation_provenance", "CADFormat", "BoundingBox",
             "CADMesh", "CADImportResult", "CADImporter", "import_cad_file"}
    nodes = [node for node in tree.body if
             isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("MAX_")
                                                for t in node.targets)]
    methods = {"__init__", "_check_occ_available", "_load_occ", "_load_ocp", "detect_format",
               "import_file", "_read_step", "_read_iges", "_read_brep", "_analyze_topology",
               "_get_bounding_box", "_tessellate", "_capture_native_topology", "_calculate_face_normals"}
    for node in nodes:
        if isinstance(node, ast.ClassDef) and node.name == "CADImporter":
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            assert {item.name for item in node.body} == methods
    fallbacks = []

    def forbidden_fallback(*args, **kwargs):
        fallbacks.append("FALLBACK_ATTEMPT")
        raise AssertionError("Chosen native reader must not retry another backend")

    namespace = dict(__name__=__name__, np=np, dataclass=dataclass, field=field, Enum=Enum,
                     Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Path=Path,
                     logger=SimpleNamespace(warning=lambda *a: None, error=lambda *a: None),
                     _import_with_gmsh=forbidden_fallback)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, fallbacks


def run_checks():
    cache = _isolated_cache()
    import numpy as np
    import OCP
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepTools import BRepTools
    from OCP.IGESControl import IGESControl_Writer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf

    namespace, fallbacks = _importer_namespace(np)
    assert namespace["_select_native_cad_binding"](allow_ocp=True) == "ocp"
    tolerance = GEOMETRY_ABSOLUTE_TOLERANCE
    counts, results = Counter(), []

    def locate(shape, rotation, translation):
        transform = gp_Trsf()
        transform.SetValues(*[float(value) for row, offset in zip(rotation, translation)
                              for value in (*row, offset)])
        return shape.Located(TopLoc_Location(transform))

    def fixture(kind):
        lengths = np.array([10., 20., 30.])
        corners = np.array(list(itertools.product(*[(0., value) for value in lengths])))
        if kind == "box":
            return (BRepPrimAPI_MakeBox(*lengths).Shape(), corners,
                    [(np.eye(3), np.zeros(3), lengths)], (6, 1, 12, 24))
        if kind == "quarter_cylinder":
            return (BRepPrimAPI_MakeCylinder(7., 11., math.pi / 2).Shape(),
                    np.array([[0., 0., 0.], [7., 7., 11.]]), [], (5, 1, 9, 18))
        assert kind == "located_compound"
        base = BRepPrimAPI_MakeBox(*lengths).Shape()
        builder, compound = BRep_Builder(), TopoDS_Compound()
        builder.MakeCompound(compound)
        offsets = [np.zeros(3), np.array([70., -90., 110.])]
        for offset in offsets:
            builder.Add(compound, locate(base, np.eye(3), offset))
        rotation = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
        translation = np.array([7., -11., 13.])
        assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-15)
        assert abs(np.linalg.det(rotation) - 1) < 1e-15
        return (locate(compound, rotation, translation),
                np.concatenate([(corners + offset) @ rotation.T + translation for offset in offsets]),
                [(rotation, rotation @ offset + translation, lengths) for offset in offsets],
                (12, 2, 24, 48))

    def write_shape(shape, path, format_):
        if format_ == "step":
            writer = STEPControl_Writer()
            assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
            assert writer.Write(str(path)) == IFSelect_RetDone
        elif format_ == "brep":
            assert BRepTools.Write_s(shape, str(path)) is True
        else:
            assert format_ == "iges"
            # Explicit faces mode; the solid owner loss is intentional, not a
            # detector-count failure or permission to infer ownership by proximity.
            writer = IGESControl_Writer("MM", 0)
            assert writer.AddShape(shape) is True
            writer.ComputeModel()
            assert writer.Write(str(path)) is True
        assert 0 < path.stat().st_size <= MAX_GENERATED_FILE_BYTES

    def verify_geometry(result, kind, extrema, boxes):
        vertices, triangles, normals = result.mesh.vertices, result.mesh.faces, result.mesh.normals
        assert vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0
        assert triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) > 0
        assert np.isfinite(vertices).all() and np.isfinite(normals).all()
        assert np.max(np.abs(vertices.min(0) - extrema.min(0))) <= tolerance
        assert np.max(np.abs(vertices.max(0) - extrema.max(0))) <= tolerance
        assert result.bounding_box is not None
        assert np.max(np.abs(np.asarray(result.bounding_box.min_point) - extrema.min(0))) <= tolerance
        assert np.max(np.abs(np.asarray(result.bounding_box.max_point) - extrema.max(0))) <= tolerance
        if kind == "quarter_cylinder":
            radius = np.hypot(vertices[:, 0], vertices[:, 1])
            inside = ((vertices[:, 0] >= -tolerance) & (vertices[:, 1] >= -tolerance)
                      & (radius <= 7. + tolerance) & (vertices[:, 2] >= -tolerance)
                      & (vertices[:, 2] <= 11. + tolerance))
            boundary = ((np.abs(vertices[:, 0]) <= tolerance) | (np.abs(vertices[:, 1]) <= tolerance)
                        | (np.abs(radius - 7.) <= tolerance) | (np.abs(vertices[:, 2]) <= tolerance)
                        | (np.abs(vertices[:, 2] - 11.) <= tolerance))
            assert np.all(inside & boundary)
        else:
            covered = np.zeros(len(vertices), dtype=bool)
            for rotation, translation, lengths in boxes:
                local = (vertices - translation) @ rotation
                inside = np.all((local >= -tolerance) & (local <= lengths + tolerance), axis=1)
                boundary = np.any((np.abs(local) <= tolerance) | (np.abs(local - lengths) <= tolerance), axis=1)
                covered |= inside & boundary
            assert covered.all()

    def verify_provenance(result, kind, expected, loss):
        face_count, solid_count, edge_count, use_count = expected
        if loss:
            solid_count, edge_count = 0, 24
        assert result.num_solids == solid_count and result.num_faces == face_count
        provenance, topology = result.mesh.tessellation_provenance, result.mesh.native_topology
        assert provenance["backend"] == "ocp" and provenance["mapping_status"] == "FACE_OCCURRENCE_RANGES"
        assert provenance["source_face_count"] == face_count
        assert topology["status"] == "CAPTURED_IMPORT_LOCAL_INCIDENCE", topology["reasons"]
        assert len(topology["face_records"]) == len(provenance["face_records"]) == face_count
        assert provenance["units_verified"] is False and provenance["frame_is_verified"] is False
        assert provenance["source_identity_is_verified"] is False
        vstop = tstop = 0
        for index, (record, face) in enumerate(zip(provenance["face_records"], topology["face_records"])):
            assert record["source_face_index"] == face["source_face_index"] == index
            assert face["face_id"] == f"face:{index}"
            start, stop = record["vertex_range"]
            first, last = record["triangle_range"]
            assert start == vstop and first == tstop and stop > start and last > first
            block = result.mesh.faces[first:last]
            assert np.all((block >= start) & (block < stop))
            vstop, tstop = stop, last
        assert (vstop, tstop) == (len(result.mesh.vertices), len(result.mesh.faces))
        owners = topology["owner_records"]
        assert len(owners) == (6 if loss else solid_count)
        assert all(owner["kind"] == ("FACE" if loss else "SOLID") for owner in owners)
        assert len({tuple(owner["occurrence_path"]) for owner in owners}) == len(owners)
        owner_faces, edge_uses = Counter(), {}
        for face in topology["face_records"]:
            owner_faces[face["owner_id"]] += 1
            for use in face["edge_uses"]:
                assert use["edge_id"].startswith(face["owner_id"] + ":edge:")
                assert use["degenerate"] is False
                edge_uses.setdefault(use["edge_id"], []).append(use["orientation"])
        assert set(owner_faces) == {owner["owner_id"] for owner in owners}
        assert set(owner_faces.values()) == ({1} if loss else {5} if kind == "quarter_cylinder" else {6})
        assert len(edge_uses) == edge_count and sum(map(len, edge_uses.values())) == use_count
        if loss:
            assert all(len(uses) == 1 and uses[0] in ("FORWARD", "REVERSED") for uses in edge_uses.values())
        else:
            assert all(sorted(uses) == ["FORWARD", "REVERSED"] for uses in edge_uses.values())
        return dict(native_faces=face_count, native_solids=solid_count,
                    owner_occurrences=len(owners), owner_local_edges=len(edge_uses), edge_uses=use_count)

    with tempfile.TemporaryDirectory(prefix="reader-roundtrip-", dir=str(cache)) as directory:
        owned = Path(directory)
        for index, (kind, format_, expectation) in enumerate(ROUNDTRIP_MATRIX):
            record = dict(case_id=f"reader:{index}", fixture=kind, format=format_, expectation=expectation,
                          failure_codes=[])
            path = owned / f"invented_{index}.{format_}"
            stage = "PREPARE_OWN_CACHE_FILE"
            try:
                if kind == "malformed":
                    path.write_bytes(b"INVENTED INVALID CAD\n")
                elif kind != "absent":
                    shape, extrema, boxes, expected = fixture(kind)
                    write_shape(shape, path, format_)
                stage = "NATIVE_READER"
                result = namespace["import_cad_file"](str(path), deflection=.02, allow_ocp=True)
                topology = result.mesh.native_topology if result.mesh is not None else None
                record['observed_import'] = dict(
                    success=result.success, faces=result.num_faces, solids=result.num_solids,
                    vertices=None if result.mesh is None else len(result.mesh.vertices),
                    triangles=None if result.mesh is None else len(result.mesh.faces),
                    topology_status=None if topology is None else topology.get('status'),
                    topology_reasons=[] if topology is None else topology.get('reasons', []))
                assert not fallbacks
                if expectation == "import_failure":
                    assert result.success is False and result.mesh is None and result.error
                    record.update(status="PASS", expected_import_failure=True)
                    counts["expected_import_failures"] += 1
                else:
                    assert result.success is True and result.mesh is not None
                    stage = "ANALYTIC_GEOMETRY"
                    verify_geometry(result, kind, extrema, boxes)
                    loss = expectation == "surface_mode_solid_ownership_lost"
                    stage = "NATIVE_INCIDENCE"
                    record.update(verify_provenance(result, kind, expected, loss))
                    record.update(status="PASS", known_transfer_losses=(
                        ["IGES_MODE_0_SOLID_OWNER_REPLACED_BY_LOOSE_FACE_OWNERS"] if loss else []))
                    counts["positive_roundtrips"] += 1
                    if loss:
                        counts["expected_surface_mode_ownership_losses"] += 1
            except Exception as exc:
                trace = exc.__traceback__
                while trace is not None and trace.tb_next is not None:
                    trace = trace.tb_next
                record.update(status="FAIL", failure_type=type(exc).__name__,
                              failure_codes=[stage + ":" + type(exc).__name__],
                              failure_function=None if trace is None else trace.tb_frame.f_code.co_name,
                              failure_source_line=None if trace is None else trace.tb_lineno)
            results.append(record)
    assert len(results) == 13
    failed = sum(record["status"] != "PASS" for record in results)
    return dict(status="FAIL" if failed or fallbacks else "PASS", matrix_cases=13,
                failed_cases=failed, counts=dict(counts), cases=results, ocp_version=OCP.__version__,
                generated_file_max_bytes=MAX_GENERATED_FILE_BYTES,
                fixed_geometry_tolerance=GEOMETRY_ABSOLUTE_TOLERANCE,
                forbidden_fallback_attempts=len(fallbacks), files_origin="OWN_ISOLATED_CACHE_INVENTED_SHAPES",
                quarter_cylinder_scope="CURVED_IMPORT_CONTROL_NOT_FORMED_BEND_OR_ROOT",
                label_identity_status="NOT_TESTED_NON_XCAF_WRITERS",
                persistent_identity_verified=False, units_verified=False, grouping_eligible=False,
                cross_binding_geometry_equivalence=False, physical_inputs_accessed=False,
                count_accuracy_or_f1_evidence=False, default_backend_changed=False)
