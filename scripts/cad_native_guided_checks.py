"""Fixed invented native CAD -> position measurement -> report checks.

Run only inside the named bounded native worker's fresh cache. No application
module/package is imported, and no existing CAD/scan file is read. The planar
cross-section is independently constructed as a closed annular-sector strip
with two tangent flanges, then extruded by OCCT; the scan is independently
sampled from analytic positions, never a tessellation or extracted CAD feature.

Frozen before native execution: 54 in-memory cases (3 radii x 3 sweep magnitudes
x 3 proper poses x 2 actual-radius factors), 4 STEP/BREP round trips, and 12
fail-closed controls. Every positive must yield one conditional measurement,
with 2e-6 mm radius/finite-axis/contact error and 2e-6 degree sweep error. No
threshold/fixture rescue after a run. This is not automatic registration,
calibrated metrology, manufacturing event identity, physical count or F1.
"""

import ast
import builtins
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


GEOMETRY_ABSOLUTE_TOLERANCE_MM = 2e-6
ANGLE_ABSOLUTE_TOLERANCE_DEG = 2e-6
MAX_GENERATED_FILE_BYTES = 8 * 1024**2
RADII_MM = (1., 5., 30.)
SWEEPS_DEG = (60., 90., 135.)
POSES = ("identity", "oblique", "axial_flip")
RADIUS_FACTORS = (1., 1.1)
MEMORY_MATRIX = tuple(itertools.product(RADII_MM, SWEEPS_DEG, POSES, RADIUS_FACTORS))
ROUNDTRIP_MATRIX = (
    (5., 90., "identity", 1.2, "step"),
    (5., 90., "identity", 1.2, "brep"),
    (5., 135., "oblique", 1.2, "step"),
    (5., 135., "oblique", 1.2, "brep"),
)
NEGATIVE_MATRIX = (
    "reversed_solid", "shell_not_solid", "box_without_bend", "convex_full_cylinder",
    "flange_window_exceeds_native_face", "unverified_units", "untrusted_registration",
    "ocp_opt_in_missing", "wrong_native_binding",
    "arc_unobserved", "translated_scan_outside_roi", "disconnected_compound",
)


def _isolated_cache():
    cache = Path(os.environ["TMPDIR"]).resolve()
    root = Path(__file__).resolve().parents[1]
    if (cache != Path.cwd().resolve() or cache != Path(os.environ["HOME"]).resolve()
            or cache == root or root in cache.parents):
        raise RuntimeError("Root-owned isolated cache required")
    return cache


@contextmanager
def _source_modules(np):
    """Named source only; no backend initializer or application import graph."""
    root = Path(__file__).resolve().parents[1]
    installed = []
    missing = object()
    real_import = builtins.__import__
    kernel = None
    adapter = None

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("backend.cad_guided_measurement", "cad_guided_measurement"):
            assert kernel is not None
            return kernel
        if name in ("backend.cad_guided_native_features", "cad_guided_native_features"):
            assert adapter is not None
            return adapter
        if name == "backend" or name.startswith("backend."):
            raise AssertionError("Unexpected application import")
        return real_import(name, globals, locals, fromlist, level)

    def load(filename, name):
        source = root / "backend" / filename
        module = ModuleType(name)
        module.__file__ = str(source)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": guarded_import}
        installed.append((name, sys.modules.get(name, missing)))
        sys.modules[name] = module
        exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
        return module

    try:
        kernel = load("cad_guided_measurement.py", "_invented_native_guided_kernel")
        adapter = load("cad_guided_native_features.py", "_invented_native_guided_adapter")
        namespace = dict(__name__=__name__, np=np, dataclass=dataclass, field=field,
                         Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                         BendMatch=Any, DetectedBend=Any,
                         __builtins__={**vars(builtins), "__import__": guarded_import})
        report_source = root / "backend" / "feature_detection" / "bend_detector.py"
        tree = ast.parse(report_source.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "BendInspectionReport"]
        assert len(selected) == 1
        # The actual full report class executes; its offline serialization branch
        # must not reach the unavailable legacy semantic/detector dependencies.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(report_source), "exec"), namespace)
        pipeline_source = root / "backend" / "bend_inspection_pipeline.py"
        names = {"inspect_cad_guided_offline", "inspect_native_cad_guided_offline"}
        tree = ast.parse(pipeline_source.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in selected} == names
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(pipeline_source), "exec"), namespace)
        yield kernel, adapter, namespace["inspect_native_cad_guided_offline"]
    finally:
        for name, previous in reversed(installed):
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _importer_namespace(np):
    source = Path(__file__).resolve().parents[1] / "backend" / "cad_import.py"
    tree = ast.parse(source.read_text())
    names = {"_select_native_cad_binding", "_tessellation_provenance", "CADFormat", "BoundingBox",
             "CADMesh", "CADImportResult", "CADImporter"}
    selected = [node for node in tree.body if
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("MAX_")
                                                   for t in node.targets)]
    methods = {"__init__", "_check_occ_available", "_load_occ", "_load_ocp", "detect_format",
               "import_file", "_read_step", "_read_iges", "_read_brep", "_analyze_topology",
               "_get_bounding_box", "_tessellate", "_capture_native_topology", "_calculate_face_normals"}
    for node in selected:
        if isinstance(node, ast.ClassDef) and node.name == "CADImporter":
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            assert {item.name for item in node.body} == methods
    namespace = dict(__name__=__name__, np=np, dataclass=dataclass, field=field, Enum=Enum,
                     Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Path=Path,
                     logger=SimpleNamespace(warning=lambda *a: None, error=lambda *a: None))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _pose(np, name, radius):
    if name == "identity":
        return np.eye(3), np.zeros(3)
    if name == "axial_flip":
        return np.diag([1., -1., -1.]), radius * np.array([-13., 17., 29.])
    assert name == "oblique"
    direction = np.array([2., -1., 3.]) / math.sqrt(14.)
    x, y, z = direction
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .73
    rotation = (math.cos(angle) * np.eye(3) + (1. - math.cos(angle)) * np.outer(direction, direction)
                + math.sin(angle) * cross)
    return rotation, radius * np.array([31., -47., 19.])


def _independent_scan(np, radius, sweep, pose, factor, include_arc=True):
    actual_radius = radius * factor
    phi = math.radians(sweep)
    axial = np.linspace(0., 6. * radius, 9)
    distance = np.linspace(.6 * radius, 2.4 * radius, 17)
    positions = [[actual_radius * math.cos(theta), actual_radius * math.sin(theta), z]
                 for theta in np.linspace(0., phi, 81) for z in axial] if include_arc else []
    positions += [[actual_radius, -along, z] for along in distance for z in axial]
    positions += [[actual_radius * math.cos(phi) - along * math.sin(phi),
                   actual_radius * math.sin(phi) + along * math.cos(phi), z]
                  for along in distance for z in axial]
    rotation, translation = _pose(np, pose, radius)
    return np.asarray(positions, dtype=np.float64) @ rotation.T + translation


def run_checks():
    cache = _isolated_cache()
    import numpy as np
    import OCP
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
    from OCP.BRepTools import BRepTools
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec

    importer_namespace = _importer_namespace(np)
    assert importer_namespace["_select_native_cad_binding"](allow_ocp=True) == "ocp"
    results, counts = [], Counter()
    maxima = Counter()

    def fixture(radius, sweep, pose, flange_factor=3.):
        """Positive planar region then prism; no adapter calls in construction."""
        phi, thickness, length = math.radians(sweep), .2 * radius, flange_factor * radius

        def polar(rad, angle):
            return np.array([rad * math.cos(angle), rad * math.sin(angle), 0.])

        def point(value):
            return gp_Pnt(*map(float, value))

        def line(first, second):
            maker = BRepBuilderAPI_MakeEdge(point(first), point(second))
            assert maker.IsDone()
            return maker.Edge()

        def arc(rad, start, stop):
            curve = GC_MakeArcOfCircle(point(polar(rad, start)),
                                      point(polar(rad, (start + stop) / 2.)), point(polar(rad, stop)))
            assert curve.IsDone()
            maker = BRepBuilderAPI_MakeEdge(curve.Value())
            assert maker.IsDone()
            return maker.Edge()

        # CCW closed cross-section: outer arc forward, inner arc backward.
        inner0, inner1 = polar(radius, 0.), polar(radius, phi)
        outer0, outer1 = polar(radius + thickness, 0.), polar(radius + thickness, phi)
        first_ray = np.array([0., -1., 0.])
        last_ray = np.array([-math.sin(phi), math.cos(phi), 0.])
        edges = [line(outer0 + length * first_ray, outer0), arc(radius + thickness, 0., phi),
                 line(outer1, outer1 + length * last_ray),
                 line(outer1 + length * last_ray, inner1 + length * last_ray),
                 line(inner1 + length * last_ray, inner1), arc(radius, phi, 0.),
                 line(inner0, inner0 + length * first_ray),
                 line(inner0 + length * first_ray, outer0 + length * first_ray)]
        wire = BRepBuilderAPI_MakeWire()
        for edge in edges:
            wire.Add(edge)
            assert wire.IsDone()
        face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
        assert face.IsDone()
        prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 6. * radius), True, True)
        assert prism.IsDone()
        shape = prism.Shape()
        assert shape.ShapeType() == TopAbs_SOLID and BRepCheck_Analyzer(shape).IsValid()
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, properties)
        expected_volume = (0.5 * ((radius + thickness)**2 - radius**2) * phi
                           + 2. * length * thickness) * 6. * radius
        # This is an independent signed-solid-orientation and shape-volume check.
        # Never reverse/repair a failed fixture to rescue the adapter's result.
        assert abs(properties.Mass() - expected_volume) <= 1e-9 * expected_volume
        rotation, translation = _pose(np, pose, radius)
        assert np.max(np.abs(rotation @ rotation.T - np.eye(3))) < 1e-14
        assert abs(np.linalg.det(rotation) - 1.) < 1e-14
        transform = gp_Trsf()
        transform.SetValues(*[float(value) for row, offset in zip(rotation, translation)
                              for value in (*row, offset)])
        return shape.Located(TopLoc_Location(transform))

    def policy(adapter, radius):
        return adapter.NativeGuidedPolicy(
            flange_window_mm=(.5 * radius, 2.5 * radius), search_margin_mm=.6 * radius,
            max_fit_residual_mm=.001 * radius, plane_exclusion_mm=.015 * radius,
            max_tangency_error_mm=.02 * radius, fit_model="measured_flange_tangent")

    def point_set_error(actual, expected):
        distances = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
        return float(max(np.max(np.min(distances, axis=0)), np.max(np.min(distances, axis=1))))

    def verify_feature(feature, radius, sweep, pose):
        rotation, translation = _pose(np, pose, radius)
        origin, axis = np.asarray(feature.origin), np.asarray(feature.axis)
        start = np.asarray(feature.start_radial)
        angle = math.radians(feature.nominal_sweep_deg)
        end = math.cos(angle) * start + math.sin(angle) * np.cross(axis, start)
        expected_axis_endpoints = np.array([[0., 0., 0.], [0., 0., 6. * radius]]) @ rotation.T + translation
        actual_axis_endpoints = origin + np.asarray(feature.span_mm)[:, None] * axis
        expected_radials = np.array([[1., 0., 0.],
                                     [math.cos(math.radians(sweep)), math.sin(math.radians(sweep)), 0.]]) @ rotation.T
        errors = dict(nominal_radius_error_mm=abs(feature.nominal_radius_mm - radius),
                      nominal_sweep_error_deg=abs(abs(feature.nominal_sweep_deg) - sweep),
                      finite_axis_endpoint_error_mm=point_set_error(actual_axis_endpoints, expected_axis_endpoints),
                      contact_radial_error_mm=radius * point_set_error(np.array([start, end]), expected_radials),
                      axis_direction_error=float(min(np.linalg.norm(axis - rotation[:, 2]),
                                                     np.linalg.norm(axis + rotation[:, 2]))))
        for name, value in errors.items():
            limit = ANGLE_ABSOLUTE_TOLERANCE_DEG if name.endswith("_deg") else GEOMETRY_ABSOLUTE_TOLERANCE_MM
            assert value <= limit, (name, value)
            maxima[name] = max(maxima[name], value)
        assert feature.unit == "mm" and feature.surface == "inner" and feature.nominal_source
        assert feature.fit_model == "measured_flange_tangent"
        return errors

    def verify_safety(output):
        assert output["production_use_allowed"] is False
        assert output["observed_bend_count"] is None
        assert output["conformance"] == "INDETERMINATE"
        assert "summary" not in output and "matches" not in output
        assert "passed" not in output and "failed" not in output
        metadata = output["native_cad_extraction"]
        assert metadata["source_identity_status"] == "IMPORT_LOCAL_NOT_PERSISTENT"
        assert metadata["frame_status"] == "NATIVE_COORDINATES_NOT_SCAN_REGISTRATION"
        assert metadata["feature_identity_status"] == "NATIVE_INSPECTION_FEATURE_NOT_MANUFACTURING_EVENT"
        assert metadata["exhaustive_bend_inventory"] is False
        assert metadata["production_use_allowed"] is False
        json.dumps(output, allow_nan=False)

    def verify_measurement(output, radius, sweep, factor):
        verify_safety(output)
        assert output["status"] == "COMPLETE", output.get("reasons")
        assert output["expected_feature_count"] == output["measured_conditional_feature_count"] == 1
        assert len(output["features"]) == 1
        row = output["features"][0]
        assert row["status"] == "MEASURED_CONDITIONAL", row.get("reasons")
        assert row["uncertainty_status"] == "UNCALIBRATED"
        assert row["physical_completion_state"] == "UNKNOWN"
        measurement = row["measurements"]
        assert measurement["fit_model"] == "measured_flange_tangent"
        errors = dict(measured_radius_error_mm=abs(measurement["measured_radius_mm"] - radius * factor),
                      radius_deviation_error_mm=abs(measurement["radius_deviation_mm"] - radius * (factor - 1.)),
                      measured_sweep_error_deg=abs(abs(measurement["measured_sweep_deg"]) - sweep))
        for name, value in errors.items():
            limit = ANGLE_ABSOLUTE_TOLERANCE_DEG if name.endswith("_deg") else GEOMETRY_ABSOLUTE_TOLERANCE_MM
            assert value <= limit, (name, value)
            maxima[name] = max(maxima[name], value)
        assert measurement["physical_trim_endpoints"] is None
        assert measurement["physical_extent_deviation_mm"] is None
        assert measurement["continuous_surface_coverage"] == "NOT_ESTABLISHED"
        assert measurement["tangency_check_role"] == "ENFORCED_CONSTRAINT_NOT_INDEPENDENT_VALIDATION"
        return errors

    def write_shape(shape, path, format_):
        if format_ == "step":
            writer = STEPControl_Writer()
            assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
            assert writer.Write(str(path)) == IFSelect_RetDone
        else:
            assert format_ == "brep" and BRepTools.Write_s(shape, str(path)) is True
        assert 0 < path.stat().st_size <= MAX_GENERATED_FILE_BYTES

    with _source_modules(np) as (_, adapter, entrypoint):
        with tempfile.TemporaryDirectory(prefix="native-guided-", dir=str(cache)) as directory:
            owned = Path(directory)
            matrix = ([('memory', item) for item in MEMORY_MATRIX]
                      + [('roundtrip', item) for item in ROUNDTRIP_MATRIX]
                      + [('negative', item) for item in NEGATIVE_MATRIX])
            for index, (kind, item) in enumerate(matrix):
                record = dict(case_id=f"native-guided:{index}", kind=kind, parameters=item)
                stage = "PREPARE_INVENTED_NATIVE_FIXTURE"
                try:
                    if kind == "negative":
                        radius, sweep, pose, factor = 5., 90., "identity", 1.1
                    else:
                        radius, sweep, pose, factor = item[:4]
                    shape = fixture(radius, sweep, pose,
                                    flange_factor=1. if item == "flange_window_exceeds_native_face" else 3.)
                    if item == "reversed_solid":
                        shape = shape.Reversed()
                    elif item == "shell_not_solid":
                        explorer = TopExp_Explorer(shape, TopAbs_SHELL)
                        assert explorer.More()
                        shape = explorer.Current()
                    elif item == "box_without_bend":
                        shape = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
                    elif item == "convex_full_cylinder":
                        shape = BRepPrimAPI_MakeCylinder(5., 30.).Shape()
                    elif item == "disconnected_compound":
                        builder, compound = BRep_Builder(), TopoDS_Compound()
                        builder.MakeCompound(compound)
                        builder.Add(compound, shape)
                        builder.Add(compound, fixture(radius, sweep, "oblique"))
                        shape = compound
                    retained = SimpleNamespace(success=True, native_shape=shape, native_binding="ocp")
                    if kind == "roundtrip":
                        format_ = item[4]
                        path = owned / f"invented_guided_{index}.{format_}"
                        write_shape(shape, path, format_)
                        stage = "NATIVE_READER_WITH_EXPLICIT_RETENTION"
                        importer = importer_namespace["CADImporter"](linear_deflection=.02, allow_ocp=True)
                        retained = importer.import_file(str(path), retain_native_shape=True)
                        assert retained.success is True and retained.native_shape is not None
                        assert retained.native_binding == "ocp"
                        assert retained.mesh is not None and retained.num_solids == 1
                        assert "native_shape" not in retained.to_dict() and "native_binding" not in retained.to_dict()
                        record["native_reader"] = dict(format=format_, solids=retained.num_solids,
                                                       faces=retained.num_faces, vertices=retained.mesh.num_vertices)
                    actual_policy = policy(adapter, radius)
                    unit_status = "UNVERIFIED" if item == "unverified_units" else "DECLARED_MM"
                    registration = "UNTRUSTED" if item == "untrusted_registration" else "DECLARED_ALIGNED"
                    allow_ocp = item != "ocp_opt_in_missing"
                    if item == "wrong_native_binding":
                        retained.native_binding = "occ"
                    stage = "NATIVE_FEATURE_EXTRACTION"
                    extraction = adapter.extract_ocp_guided_features(
                        retained.native_shape, policy=actual_policy, unit_status=unit_status,
                        allow_ocp=allow_ocp, native_binding=retained.native_binding)
                    native_positive = kind != "negative" or item in (
                        "untrusted_registration", "arc_unobserved", "translated_scan_outside_roi")
                    if native_positive:
                        assert extraction["status"] == "AVAILABLE", extraction.get("metadata")
                        assert len(extraction["features"]) == 1
                        record["nominal_errors"] = verify_feature(extraction["features"][0], radius, sweep, pose)
                    else:
                        assert extraction["status"] == "UNAVAILABLE"
                        assert len(extraction["features"]) == 0
                        record["native_reasons"] = extraction["metadata"].get("reasons", [])
                    stage = "CONNECTED_ACTUAL_ENTRYPOINT_AND_REPORT"
                    points = _independent_scan(np, radius, sweep, pose, factor,
                                               include_arc=item != "arc_unobserved")
                    if item == "translated_scan_outside_roi":
                        # This deliberately false alignment declaration checks
                        # only gross absent-ROI support, not registration accuracy.
                        points = points + np.array([1000., 0., 0.])
                    output = entrypoint(retained, points, f"invented:{index}", policy=actual_policy,
                                        unit_status=unit_status, registration_status=registration,
                                        allow_ocp=allow_ocp).to_dict()
                    verify_safety(output)
                    if kind == "negative":
                        assert output["status"] == "UNAVAILABLE"
                        assert output["measured_conditional_feature_count"] == 0
                        assert all(row["measurements"] is None for row in output["features"])
                        if not native_positive:
                            assert output["expected_feature_count"] is None and output["features"] == []
                        counts["expected_abstentions"] += 1
                    else:
                        record["measurement_errors"] = verify_measurement(output, radius, sweep, factor)
                        counts["conditional_measurements"] += 1
                        counts["positive_real_reader_roundtrips" if kind == "roundtrip" else "positive_memory_cases"] += 1
                        if factor != 1.:
                            counts["nonzero_radius_deviation_cases"] += 1
                    record["status"] = "PASS"
                except Exception as exc:
                    trace = exc.__traceback__
                    while trace is not None and trace.tb_next is not None:
                        trace = trace.tb_next
                    record.update(status="FAIL", failure_stage=stage, failure_type=type(exc).__name__,
                                  failure_message=str(exc)[:1000],
                                  failure_function=None if trace is None else trace.tb_frame.f_code.co_name,
                                  failure_source_line=None if trace is None else trace.tb_lineno)
                results.append(record)

    assert len(results) == 70
    failed = sum(record["status"] != "PASS" for record in results)
    return dict(status="FAIL" if failed else "PASS", matrix_cases=len(results), failed_cases=failed,
                counts=dict(counts), maximum_errors=dict(maxima),
                fixed_geometry_tolerance_mm=GEOMETRY_ABSOLUTE_TOLERANCE_MM,
                fixed_angle_tolerance_deg=ANGLE_ABSOLUTE_TOLERANCE_DEG,
                generated_file_max_bytes=MAX_GENERATED_FILE_BYTES, ocp_version=OCP.__version__,
                files_origin="OWN_ISOLATED_CACHE_INVENTED_SHAPES",
                truth_origin="INDEPENDENT_CLOSED_CROSS_SECTION_AND_ANALYTIC_SCAN_POSITIONS",
                registration_status="DECLARED_NOT_VERIFIED", units_status="DECLARED_MM_NOT_SOURCE_VERIFIED",
                manufacturing_event_count="NOT_CLAIMED", physical_accuracy="NOT_TESTED",
                cases=results)
