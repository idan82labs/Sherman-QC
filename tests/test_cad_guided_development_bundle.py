"""Invented source snapshots only; no real freeze, metadata discovery or native imports."""

import ast
import base64
import copy
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROL = "backend/cad_guided_development_control.py"
CONTROL_BYTES = b'CONTROL_ID = "INVENTED"\ndef control_specification():\n    return {"control_id": CONTROL_ID}\n'


def _module():
    path = ROOT / "scripts/cad_guided_development_bundle.py"
    spec = importlib.util.spec_from_file_location("invented_development_bundle_test", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _sources(module):
    return {name: (CONTROL_BYTES if name == CONTROL else b'VALUE = "captured"\n')
            for name in module.SOURCES}


def _bundle(module, sources=None):
    sources = _sources(module) if sources is None else sources
    records = {name: {"sha256": module.digest(blob), "bytes": len(blob),
                      "source_base64": base64.b64encode(blob).decode("ascii")}
               for name, blob in sources.items()}
    payload = {
        "schema": "CAD_DEVELOPMENT_SOURCE_BUNDLE_V1", "sources": records,
        "specification": {"control_id": "INVENTED"},
        "runtime_metadata": {"assurance": "INVENTED_METADATA_NOT_RUNTIME_ATTESTATION"},
        "physical_access_authorized": False, "production_use_allowed": False,
    }
    return {"bundle_sha256": module.digest(module.canonical(payload)), "payload": payload}


def _rebind(module, bundle):
    bundle["bundle_sha256"] = module.digest(module.canonical(bundle["payload"]))


def test_roundtrip_contains_exact_full_fixed_source_closure_without_metadata_access(monkeypatch):
    module = _module()
    sources = _sources(module)
    calls = []

    def read(path, limit):
        name = str(Path(path).relative_to(module.ROOT))
        assert name in sources and limit == module.MAX_SOURCE_BYTES
        calls.append(name)
        return sources[name]

    monkeypatch.setattr(module, "read_regular", read)
    metadata = {"assurance": "INVENTED_METADATA_NOT_RUNTIME_ATTESTATION"}
    monkeypatch.setattr(module, "runtime_metadata", lambda: copy.deepcopy(metadata))
    result = module.capture_bundle()
    assert module.validate_bundle(json.loads(module.canonical(result))) == sources
    assert sorted(calls) == sorted(list(module.SOURCES) * 2)
    assert result["payload"]["specification"] == {"control_id": "INVENTED"}
    assert result["payload"]["runtime_metadata"] == metadata
    assert result["payload"]["physical_access_authorized"] is False
    assert result["payload"]["production_use_allowed"] is False


def test_capture_detects_source_drift_without_any_real_source_read(monkeypatch):
    module = _module()
    sources = _sources(module)
    counts = {}

    def read(path, limit):
        name = str(Path(path).relative_to(module.ROOT))
        counts[name] = counts.get(name, 0) + 1
        return sources[name] + (b"# changed\n" if counts[name] > 1 else b"")

    monkeypatch.setattr(module, "read_regular", read)
    monkeypatch.setattr(module, "runtime_metadata", lambda: {})
    with pytest.raises(ValueError, match="SOURCE_CHANGED_DURING_FREEZE"):
        module.capture_bundle()


@pytest.mark.parametrize("change", ["root_extra", "root_missing", "not_mapping",
                                    "payload_extra", "payload_missing", "wrong_schema",
                                    "physical_true", "production_true", "hash"])
def test_rejects_invalid_envelope_and_authority_fields(change):
    module = _module()
    bundle = _bundle(module)
    if change == "root_extra":
        bundle["extra"] = True
    elif change == "root_missing":
        del bundle["bundle_sha256"]
    elif change == "not_mapping":
        bundle = []
    elif change == "payload_extra":
        bundle["payload"]["extra"] = True
    elif change == "payload_missing":
        del bundle["payload"]["specification"]
    elif change == "wrong_schema":
        bundle["payload"]["schema"] = "OTHER"
    elif change == "physical_true":
        bundle["payload"]["physical_access_authorized"] = True
    elif change == "production_true":
        bundle["payload"]["production_use_allowed"] = True
    else:
        bundle["bundle_sha256"] = "0" * 64
    if change.startswith("payload_") or change in {"wrong_schema", "physical_true", "production_true"}:
        _rebind(module, bundle)
    with pytest.raises(ValueError):
        module.validate_bundle(bundle)


@pytest.mark.parametrize("change", ["missing", "extra", "not_mapping"])
def test_rejects_nonexact_source_closure_even_with_rebound_outer_digest(change):
    module = _module()
    bundle = _bundle(module)
    records = bundle["payload"]["sources"]
    if change == "missing":
        del records[module.SOURCES[0]]
    elif change == "extra":
        records["backend/unfrozen.py"] = copy.deepcopy(records[module.SOURCES[0]])
    else:
        bundle["payload"]["sources"] = list(records)
    _rebind(module, bundle)
    with pytest.raises(ValueError, match="INVALID_SOURCE_CLOSURE"):
        module.validate_bundle(bundle)


@pytest.mark.parametrize("change", ["extra", "missing", "hash", "bytes", "bool_bytes",
                                    "empty", "base64", "base64_type", "syntax"])
def test_rejects_corrupt_source_records(change):
    module = _module()
    bundle = _bundle(module)
    row = bundle["payload"]["sources"][module.SOURCES[0]]
    if change == "extra":
        row["extra"] = 0
    elif change == "missing":
        del row["bytes"]
    elif change == "hash":
        row["sha256"] = "0" * 64
    elif change == "bytes":
        row["bytes"] += 1
    elif change == "bool_bytes":
        row["bytes"] = True
    elif change == "empty":
        row.update(bytes=0, source_base64="", sha256=module.digest(b""))
    elif change == "base64":
        row["source_base64"] = "%%%"
    elif change == "base64_type":
        row["source_base64"] = []
    else:
        invalid = b"def broken(:\n"
        row.update(bytes=len(invalid), source_base64=base64.b64encode(invalid).decode(),
                   sha256=module.digest(invalid))
    _rebind(module, bundle)
    with pytest.raises((ValueError, SyntaxError)):
        module.validate_bundle(bundle)


def test_rejects_mismatched_configuration_source_even_after_outer_rebinding():
    module = _module()
    bundle = _bundle(module)
    bundle["payload"]["specification"]["control_id"] = "DETACHED"
    _rebind(module, bundle)
    with pytest.raises(ValueError, match="CONFIGURATION_SOURCE_MISMATCH"):
        module.validate_bundle(bundle)


def test_individual_source_byte_budget_is_enforced(monkeypatch):
    module = _module()
    bundle = _bundle(module)
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 8)
    with pytest.raises(ValueError):
        module.validate_bundle(bundle)


def test_direct_validation_enforces_aggregate_bundle_byte_budget(monkeypatch):
    module = _module()
    bundle = _bundle(module)
    monkeypatch.setattr(module, "MAX_BUNDLE_BYTES", len(module.canonical(bundle)) - 1)
    with pytest.raises(ValueError, match="BUNDLE_BYTE_BUDGET"):
        module.validate_bundle(bundle)


@pytest.mark.parametrize("json_bytes", [b'{"payload":{},"payload":{}}',
                                       b'{"payload":{"schema":1,"schema":2}}'])
def test_read_bundle_rejects_duplicate_keys_without_opening_any_file(monkeypatch, json_bytes):
    module = _module()
    calls = []

    def read(path, limit):
        calls.append((path, limit))
        return json_bytes

    monkeypatch.setattr(module, "read_regular", read)
    with pytest.raises(ValueError, match="DUPLICATE_BUNDLE_KEY"):
        module.read_bundle()
    assert calls == [(module.BUNDLE, module.MAX_BUNDLE_BYTES)]


@pytest.fixture
def fresh_backend_namespace():
    previous = {name: module for name, module in sys.modules.items()
                if name == "backend" or name.startswith("backend.")}
    for name in previous:
        del sys.modules[name]
    finders = list(sys.meta_path)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "backend" or name.startswith("backend."):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.meta_path[:] = finders


def test_captured_imports_ignore_filesystem_bytecode_and_later_bundle_mutation(monkeypatch, fresh_backend_namespace):
    module = _module()
    sources = _sources(module)
    sources["backend/cad_guided_offline.py"] = b"from .cad_guided_registration import VALUE\n"
    bundle = _bundle(module, sources)
    finder = module.install_captured_importer(bundle)
    original_digest = bundle["bundle_sha256"]

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen import attempted mutable source/bytecode access")

    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "get_code", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    row = bundle["payload"]["sources"]["backend/cad_guided_registration.py"]
    row["source_base64"] = base64.b64encode(b'VALUE = "mutated"\n').decode()
    bundle["bundle_sha256"] = "mutated-caller-declaration"
    imported = importlib.import_module("backend.cad_guided_offline")
    assert imported.VALUE == "captured"
    assert imported.__file__.startswith("<frozen:")
    assert original_digest in imported.__file__ and "mutated-caller" not in imported.__file__
    assert finder is sys.meta_path[0]
    with pytest.raises(ModuleNotFoundError, match="UNFROZEN_BACKEND_IMPORT_REFUSED"):
        importlib.import_module("backend.not_in_snapshot")


@pytest.mark.parametrize("name", ["backend", "backend.cad_guided_offline"])
def test_captured_importer_rejects_preloaded_backend_namespace(name, fresh_backend_namespace):
    module = _module()
    sys.modules[name] = ModuleType(name)
    with pytest.raises(ValueError, match="FROZEN_IMPORT_REQUIRES_FRESH_BACKEND_NAMESPACE"):
        module.install_captured_importer(_bundle(module))


@pytest.mark.parametrize("prefix", ["len([])\n", "VALUE = len([])\n",
                                     "@list\ndef extra(): pass\n",
                                     "def extra(value=len([])): pass\n"])
def test_control_specification_rejects_startup_expressions_without_executing_them(prefix):
    module = _module()
    blob = (prefix + 'def control_specification():\n    return {"scope": "invented"}\n').encode()
    with pytest.raises(ValueError):
        module.specification_from_source(blob)


def test_actual_control_has_only_explicit_declarations_and_fresh_policy_data():
    module = _module()
    source = (ROOT / CONTROL).read_bytes()
    tree = ast.parse(source)
    assert all(isinstance(node, (ast.Expr, ast.Assign, ast.FunctionDef)) for node in tree.body)
    assert all(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
               for node in tree.body if isinstance(node, ast.Expr))
    assert all(not node.decorator_list and not node.args.defaults and not node.args.kw_defaults
               for node in tree.body if isinstance(node, ast.FunctionDef))
    spec = module.specification_from_source(source)
    assert spec["historical_gold11_comparable"] is False
    assert spec["physical_score"] is None
    assert spec["physical_access_authorized"] is spec["production_use_allowed"] is False
    assert spec["measurement"]["fit_model"] == "measured_flange_tangent"
    assert spec["native_loader"]["meshing_parallel"] is True
    assert spec["native_loader"]["native_thread_count"] == "UNVERIFIED"
    assert spec["preprocessing"] == "NONE_RAW_ROW_ORDER_PRESERVED"
    spec["registration"]["max_iterations"] = -1
    assert module.specification_from_source(source)["registration"]["max_iterations"] == 40
    feature_tree = ast.parse((ROOT / "backend/cad_guided_measurement.py").read_bytes())
    feature = next(node for node in feature_tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "CADGuidedFeature")
    defaults = {node.target.id: ast.literal_eval(node.value) for node in feature.body
                if isinstance(node, ast.AnnAssign) and node.value is not None}
    assert all(defaults[name] == value for name, value in spec["inherited_feature_defaults"].items())
