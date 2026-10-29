"""Exact-source handoff checks with invented doubles; no native/app/file inputs."""
import ast
import builtins
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _importer_namespace(path_factory):
    source = ROOT / "backend/cad_import.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in {"CADFormat", "CADImportResult", "CADImporter"}]
    assert len(nodes) == 3
    for node in nodes:
        if node.name == "CADImporter":
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name == "import_file"]
            assert len(node.body) == 1
    namespace = {
        "__name__": "cad_native_retention_source_check", "dataclass": dataclass,
        "field": field, "Enum": Enum, "Optional": Optional, "Any": Any,
        "Dict": Dict, "List": List, "CADMesh": Any, "BoundingBox": Any,
        "Path": path_factory, "logger": SimpleNamespace(error=lambda *args: None),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _fixture(format_name="STEP", failure=None):
    paths, used = [], []

    def path_factory(value):
        paths.append(value)
        return SimpleNamespace(exists=lambda: failure != "absent", suffix=".invented")

    namespace = _importer_namespace(path_factory)
    importer = namespace["CADImporter"]()
    importer.native_binding = "ocp"
    format_ = getattr(namespace["CADFormat"], "UNKNOWN" if failure == "unsupported" else format_name)
    importer.detect_format = lambda path: format_
    shape = object()
    mesh = SimpleNamespace(num_vertices=12, num_faces=8)

    def stage(name, result, incoming=None):
        used.append((name, incoming))
        if failure == name:
            raise ValueError("INVENTED_STAGE_FAILURE")
        return result

    def load():
        if failure == "load":
            raise ImportError("INVENTED_BINDING_FAILURE")

    importer._load_occ = load
    for suffix in ("step", "iges", "brep"):
        setattr(importer, "_read_" + suffix, lambda path: stage("read", shape, path))
    importer._analyze_topology = lambda value: stage(
        "analysis", {"solids": 1, "shells": 1, "faces": 6, "edges": 12}, value)
    importer._get_bounding_box = lambda value: stage("bbox", None, value)
    importer._tessellate = lambda value: stage("tessellate", mesh, value)
    return namespace, importer, shape, mesh, paths, used


@pytest.mark.parametrize("flag", [None, 0, 1, "true", object()])
def test_retention_flag_is_strict_and_rejected_before_path_access(flag):
    def forbidden_path(value):
        raise AssertionError("invalid flag must precede every file/path action")

    importer = _importer_namespace(forbidden_path)["CADImporter"]()
    with pytest.raises(TypeError, match="retain_native_shape must be a bool"):
        importer.import_file("invented.step", retain_native_shape=flag)


@pytest.mark.parametrize("format_name", ["STEP", "IGES", "BREP"])
@pytest.mark.parametrize("retain", [False, True])
def test_only_explicit_success_retains_exact_reader_shape_and_binding(format_name, retain):
    _, importer, shape, mesh, paths, used = _fixture(format_name)
    result = importer.import_file("invented", retain_native_shape=retain)
    assert result.success is True and result.mesh is mesh
    assert result.native_shape is (shape if retain else None)
    assert result.native_binding == ("ocp" if retain else None)
    assert paths == ["invented"]
    assert [name for name, _ in used] == ["read", "analysis", "bbox", "tessellate"]
    assert all(incoming is shape for _, incoming in used[1:])
    assert "native_shape" not in repr(result) and "native_binding" not in repr(result)
    assert "native_shape" not in result.to_dict() and "native_binding" not in result.to_dict()
    assert not hasattr(importer, "last_shape")


def test_default_retention_and_serialization_stay_unchanged():
    _, importer, _, _, _, _ = _fixture()
    result = importer.import_file("invented")
    assert result.native_shape is None and result.native_binding is None
    assert result.to_dict() == {
        "success": True, "format": "step", "units": "mm", "num_vertices": 12,
        "num_faces": 8, "num_solids": 1, "bounding_box": None,
        "error": None, "warnings": [],
    }


@pytest.mark.parametrize("failure", ["absent", "unsupported", "load", "read", "analysis", "bbox", "tessellate"])
def test_failure_never_returns_a_retained_shape_or_binding(failure):
    _, importer, _, _, _, _ = _fixture(failure=failure)
    result = importer.import_file("invented", retain_native_shape=True)
    assert result.success is False
    assert result.native_shape is None and result.native_binding is None


def _entrypoint(adapter, measurement=None, import_function=None):
    source = ROOT / "backend/bend_inspection_pipeline.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "inspect_native_cad_guided_offline"]
    assert len(nodes) == 1

    def only_adapter(name, *args, **kwargs):
        assert name == "backend.cad_guided_native_features"
        return SimpleNamespace(extract_ocp_guided_features=adapter)

    namespace = {
        "np": SimpleNamespace(ndarray=object),
        "BendInspectionReport": lambda **kwargs: SimpleNamespace(**kwargs),
        "inspect_cad_guided_offline": measurement,
        "__builtins__": {**vars(builtins), "__import__": import_function or only_adapter},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["inspect_native_cad_guided_offline"]


def test_available_native_features_reach_existing_measurement_entry_without_overclaim():
    shape, scan, policy, feature = object(), object(), object(), object()
    features = (feature,)
    metadata = {"status": "AVAILABLE", "source_identity_is_verified": False}
    calls = []

    def adapter(value, **kwargs):
        calls.append((value, kwargs))
        return {"status": "AVAILABLE", "features": features, "metadata": metadata}

    def measurement(points, received_features, part_id, **kwargs):
        assert points is scan and received_features is features and part_id == "invented"
        assert kwargs == {"registration_status": "DECLARED_ALIGNED"}
        return SimpleNamespace(cad_guided_measurements={
            "status": "COMPLETE", "expected_feature_count": 1,
            "measured_conditional_feature_count": 1,
            "conformance": "INDETERMINATE", "observed_bend_count": None,
            "uncertainty_status": "UNCALIBRATED", "production_use_allowed": False,
            "feature_identity_status": "DECLARED_NOT_AUTOMATICALLY_VERIFIED",
        })

    cad = SimpleNamespace(success=True, native_shape=shape, native_binding="ocp")
    report = _entrypoint(adapter, measurement)(
        cad, scan, "invented", policy=policy, allow_ocp=True,
        unit_status="DECLARED_MM", registration_status="DECLARED_ALIGNED")
    assert calls == [(shape, {"policy": policy, "allow_ocp": True,
                              "unit_status": "DECLARED_MM", "native_binding": "ocp"})]
    assert report.cad_guided_measurements == {
        "status": "COMPLETE", "computation_status": "COMPLETE",
        "measurement_availability": "AVAILABLE", "expected_feature_count": 1,
        "measured_conditional_feature_count": 1,
        "conformance": "INDETERMINATE", "observed_bend_count": None,
        "uncertainty_status": "UNCALIBRATED", "production_use_allowed": False,
        "feature_identity_status": "DECLARED_NOT_AUTOMATICALLY_VERIFIED",
        "native_cad_extraction": metadata,
    }


@pytest.mark.parametrize("computation,measured,expected_status,availability", [
    ("COMPLETE", 0, "UNAVAILABLE", "UNAVAILABLE"),
    ("COMPLETE", 1, "PARTIAL", "PARTIAL"),
    ("COMPLETE", 3, "COMPLETE", "AVAILABLE"),
    ("UNAVAILABLE", 0, "UNAVAILABLE", "UNAVAILABLE"),
])
def test_native_report_separates_computation_from_measurement_availability(
        computation, measured, expected_status, availability):
    features = tuple(object() for _ in range(3))
    metadata = {"status": "AVAILABLE"}
    rows = [{"feature_id": "invented:" + str(index),
             "status": "MEASURED_CONDITIONAL" if index < measured else "UNAVAILABLE",
             "measurements": {"radius_deviation_mm": 1.} if index < measured else None,
             "conformance": "INDETERMINATE"}
            for index in range(3)]

    def adapter(shape, **kwargs):
        return {"status": "AVAILABLE", "features": features, "metadata": metadata}

    def measurement(*args, **kwargs):
        return SimpleNamespace(cad_guided_measurements={
            "status": computation, "expected_feature_count": 3,
            "measured_conditional_feature_count": measured, "features": rows,
            "conformance": "INDETERMINATE", "observed_bend_count": None,
            "production_use_allowed": False,
        })

    cad = SimpleNamespace(success=True, native_shape=object(), native_binding="ocp")
    payload = _entrypoint(adapter, measurement)(
        cad, object(), "invented", policy=object(), allow_ocp=True).cad_guided_measurements
    assert payload["computation_status"] == computation
    assert payload["status"] == expected_status
    assert payload["measurement_availability"] == availability
    assert payload["expected_feature_count"] == 3
    assert payload["measured_conditional_feature_count"] == measured
    assert payload["features"] is rows
    assert [row["measurements"] for row in rows] == [
        {"radius_deviation_mm": 1.} if index < measured else None for index in range(3)]
    assert payload["conformance"] == "INDETERMINATE"
    assert payload["observed_bend_count"] is None and payload["production_use_allowed"] is False
    assert payload["native_cad_extraction"] is metadata


@pytest.mark.parametrize("status,features", [("UNAVAILABLE", ()), ("AVAILABLE", ()), ("UNKNOWN", (object(),))])
def test_unavailable_or_empty_extraction_cannot_become_measurements_or_zero_bends(status, features):
    metadata = {"status": status, "reasons": ["INVENTED_UNAVAILABLE"]}

    def adapter(shape, **kwargs):
        assert shape is None
        assert kwargs == {"policy": "invented", "unit_status": "UNVERIFIED",
                          "native_binding": None, "allow_ocp": False}
        return {"status": status, "features": features, "metadata": metadata}

    report = _entrypoint(adapter)(SimpleNamespace(success=True), object(), "invented", policy="invented")
    payload = report.cad_guided_measurements
    assert payload["status"] == "UNAVAILABLE" and payload["features"] == []
    assert payload["computation_status"] == "NOT_RUN"
    assert payload["measurement_availability"] == "UNAVAILABLE"
    assert payload["expected_feature_count"] is None and payload["observed_bend_count"] is None
    assert payload["measured_conditional_feature_count"] == 0
    assert payload["conformance"] == "INDETERMINATE"
    assert payload["production_use_allowed"] is False
    assert payload["native_cad_extraction"] is metadata


def test_failed_import_cannot_reuse_a_stale_shape():
    def forbidden_adapter(*args, **kwargs):
        raise AssertionError("failed import must not reach native adapter")

    cad = SimpleNamespace(success=False, native_shape=object(), native_binding="ocp")
    payload = _entrypoint(forbidden_adapter)(cad, object(), "invented", policy=object()).cad_guided_measurements
    assert payload["native_cad_extraction"]["reasons"] == ["NATIVE_CAD_IMPORT_UNAVAILABLE"]
    assert payload["computation_status"] == "NOT_RUN"
    assert payload["measurement_availability"] == "UNAVAILABLE"
    assert payload["features"] == [] and payload["observed_bend_count"] is None


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_native_entry_rejects_nonboolean_opt_in_before_any_import(flag):
    def forbidden_import(*args, **kwargs):
        raise AssertionError("invalid flag must precede imports")

    entry = _entrypoint(None, import_function=forbidden_import)
    with pytest.raises(TypeError, match="allow_ocp must be a bool"):
        entry(None, None, "invented", policy=None, allow_ocp=flag)


@pytest.mark.parametrize("missing", ["backend", "OCP", "backend.cad_guided_native_features"])
def test_lazy_import_fallback_only_handles_absent_backend_package(missing):
    imports = []

    def controlled_import(name, *args, **kwargs):
        imports.append(name)
        if name == "backend.cad_guided_native_features":
            raise ModuleNotFoundError("invented missing module", name=missing)
        assert name == "cad_guided_native_features"
        return SimpleNamespace(extract_ocp_guided_features=None)

    entry = _entrypoint(None, import_function=controlled_import)
    if missing == "backend":
        entry(SimpleNamespace(success=False), None, "invented", policy=None)
        assert imports == ["backend.cad_guided_native_features", "cad_guided_native_features"]
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            entry(SimpleNamespace(success=False), None, "invented", policy=None)
        assert error.value.name == missing
        assert imports == ["backend.cad_guided_native_features"]
