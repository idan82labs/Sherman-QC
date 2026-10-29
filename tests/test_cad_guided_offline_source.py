"""Whole-module import and lazy dependency wiring with invented arrays/doubles.

No native library, file input, application or legacy report is imported. The
existing 22 entry-boundary cases remain in test_cad_registered_guided_entry_source.
"""
import ast
import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/cad_guided_offline.py"


def load_runtime():
    # Standard whole-module loader: no source rewriting or AST-pruned engine.
    spec = importlib.util.spec_from_file_location("invented_offline_core", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    return dict(
        cad_result=SimpleNamespace(success=True, native_binding="ocp", native_shape=object()),
        scan_points=np.arange(180, dtype=float).reshape(60, 3) / 20,
        cad_datum_points=np.zeros((36, 3)), cad_datum_normals=np.ones((36, 3)),
        scan_datum_row_indices=np.arange(36), registration_policy=object(),
        measurement_policy=object(), instance_policy=object(),
        unit_status="DECLARED_MM", allow_ocp=True)


def install_double(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def accepted_registration():
    return {"status": "ACCEPTED_BOUNDED", "transform_scan_to_cad": np.eye(4).tolist(),
            "evidence": {"candidate_search_status": "SEARCH_BOUNDED_NOT_GLOBAL"}}


def test_normal_package_import_has_no_legacy_or_native_runtime_dependencies(monkeypatch):
    forbidden = (
        "open3d", "OCP", "OCC", "sqlalchemy", "server", "apps", "domains",
        "feature_detection", "bend_inspection_pipeline", "cad_guided_native_features",
        "cad_guided_registration", "cad_guided_measurement", "cad_guided_evaluation",
        "backend.feature_detection", "backend.bend_inspection_pipeline",
        "backend.cad_guided_native_features", "backend.cad_guided_registration",
        "backend.cad_guided_measurement", "backend.cad_guided_evaluation")
    real_import = builtins.__import__
    seen = []

    def guarded_import(name, *args, **kwargs):
        seen.append(name)
        assert not any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
        return real_import(name, *args, **kwargs)

    name = "backend.cad_guided_offline"
    previous = sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(ROOT))
    try:
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", guarded_import)
            module = importlib.import_module(name)
        assert callable(module.inspect_registered_native)
        assert "numpy" in seen
        assert not hasattr(module, "BendInspectionReport")
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


@pytest.mark.parametrize("measured,expected,status,availability", [
    (0, 1, "UNAVAILABLE", "UNAVAILABLE"),
    (1, 2, "PARTIAL", "PARTIAL"),
    (2, 2, "COMPLETE", "AVAILABLE"),
])
def test_direct_native_and_kernel_wiring_preserves_computation_vs_availability(
        monkeypatch, measured, expected, status, availability):
    module = load_runtime()
    arguments = inputs()
    registration = accepted_registration()
    metadata = {"status": "AVAILABLE", "exhaustive_bend_inventory": False}
    features = tuple(object() for _ in range(expected))
    calls = []

    def register(scan, reference, normals, *, policy):
        assert np.array_equal(scan, arguments["scan_points"][:36])
        assert reference is arguments["cad_datum_points"]
        assert normals is arguments["cad_datum_normals"]
        assert policy is arguments["registration_policy"]
        calls.append("register")
        return registration

    def extract(shape, **kwargs):
        assert shape is arguments["cad_result"].native_shape
        assert kwargs == {"policy": arguments["measurement_policy"], "unit_status": "DECLARED_MM",
                          "allow_ocp": True, "native_binding": "ocp"}
        calls.append("extract")
        return {"status": "AVAILABLE", "features": features, "metadata": metadata}

    def measure(points, supplied_features, **kwargs):
        assert np.array_equal(points, arguments["scan_points"][36:])
        assert supplied_features is features
        assert kwargs == {"registration_status": "DECLARED_ALIGNED",
                          "instance_policy": arguments["instance_policy"]}
        calls.append("measure")
        return {"status": "COMPLETE", "measured_conditional_feature_count": measured,
                "expected_feature_count": expected, "features": [], "observed_bend_count": None,
                "conformance": "INDETERMINATE",
                "production_use_allowed": False}

    install_double(monkeypatch, "cad_guided_registration", register_datum_clouds=register)
    install_double(monkeypatch, "cad_guided_native_features", extract_ocp_guided_features=extract)
    install_double(monkeypatch, "cad_guided_measurement", inspect_cad_guided_scan=measure)
    payload = module.inspect_registered_native(**arguments)
    assert calls == ["register", "extract", "measure"]
    assert payload["computation_status"] == "COMPLETE"
    assert payload["status"] == status and payload["measurement_availability"] == availability
    assert payload["registration_status"] == "ACCEPTED_BOUNDED"
    assert payload["registration"] is registration and payload["native_cad_extraction"] is metadata
    assert payload["scope"] == "CAD_GUIDED_REGISTERED_DATUM_OFFLINE"
    assert payload["uncertainty_status"] == "UNCALIBRATED"
    assert payload["physical_completion_state"] == "UNKNOWN"
    assert payload["observed_bend_count"] is None and payload["production_use_allowed"] is False
    assert payload["finite_support_instances"] == [] and payload["observed_support_instance_count"] == 0
    assert payload["instance_extraction_status"] == "UNAVAILABLE"


@pytest.mark.parametrize("extraction_status", ["UNAVAILABLE", "AVAILABLE"])
def test_unavailable_or_empty_native_features_do_not_import_measurement(monkeypatch, extraction_status):
    module = load_runtime()
    install_double(monkeypatch, "cad_guided_registration",
                   register_datum_clouds=lambda *args, **kwargs: accepted_registration())
    metadata = {"status": extraction_status, "reasons": ["INVENTED_UNSUPPORTED_NATIVE_GEOMETRY"]}
    install_double(monkeypatch, "cad_guided_native_features", extract_ocp_guided_features=
                   lambda *args, **kwargs: {"status": extraction_status, "features": (), "metadata": metadata})
    # Importing this placeholder succeeds, but the missing callable would fail
    # immediately if the measurement import were incorrectly reached.
    install_double(monkeypatch, "cad_guided_measurement")
    payload = module.inspect_registered_native(**inputs())
    assert payload["reasons"] == ["NATIVE_CAD_FEATURES_UNAVAILABLE"]
    assert payload["status"] == payload["measurement_availability"] == "UNAVAILABLE"
    assert payload["computation_status"] == "NOT_RUN"
    assert payload["registration_status"] == "ACCEPTED_BOUNDED"
    assert payload["native_cad_extraction"] is metadata
    assert payload["finite_support_instances"] == [] and payload["observed_bend_count"] is None
    assert payload["expected_feature_count"] is None


def test_backend_namespace_relative_dependency_route(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    importlib.import_module("backend")
    spec = importlib.util.spec_from_file_location("backend.invented_offline_core", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registration = {"status": "UNAVAILABLE", "transform_scan_to_cad": None,
                    "reasons": ["AMBIGUOUS_GENERATED_POSES"]}
    install_double(monkeypatch, "backend.cad_guided_registration",
                   register_datum_clouds=lambda *args, **kwargs: registration)
    payload = module.inspect_registered_native(**inputs())
    assert payload["registration"] is registration
    assert payload["reasons"] == ["DATUM_REGISTRATION_UNAVAILABLE"]


def test_legacy_registered_wrapper_delegates_without_reimplementing_runtime():
    # The legacy wrapper itself cannot safely be imported during source-only
    # checks. This sole AST extraction tests report wiring, never the engine.
    source = ROOT / "backend/bend_inspection_pipeline.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "inspect_registered_native_cad_guided_offline"]
    assert len(nodes) == 1
    arguments = inputs()
    expected = {**arguments}
    part_id = "invented-wrapper"
    returned = {"status": "UNAVAILABLE", "observed_bend_count": None}
    calls = []

    def inspect(cad_result, scan_points, **kwargs):
        assert cad_result is expected["cad_result"] and scan_points is expected["scan_points"]
        assert set(kwargs) == set(expected) - {"cad_result", "scan_points"}
        assert all(kwargs[name] is expected[name] for name in kwargs)
        calls.append("runtime")
        return returned

    def importer(name, *args, **kwargs):
        assert name in ("backend.cad_guided_offline", "cad_guided_offline")
        return SimpleNamespace(inspect_registered_native=inspect)

    namespace = {"np": np, "BendInspectionReport": lambda **kwargs: SimpleNamespace(**kwargs),
                 "__builtins__": {**vars(builtins), "__import__": importer}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    report = namespace[nodes[0].name](part_id=part_id, **arguments)
    assert calls == ["runtime"] and report.cad_guided_measurements is returned
    assert report.part_id == part_id and report.matches == []
