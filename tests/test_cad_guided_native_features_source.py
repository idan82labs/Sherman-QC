"""Source-only admission checks; no native library, shape or physical inputs.

The native adapter source is compiled in both supported module contexts. Its
measurement-policy dependency is the exact CADGuidedFeature class extracted
from the named source, not a permissive substitute. Native imports are guarded;
geometry is covered separately by the bounded constructed-native worker.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, dataclass
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "backend/cad_guided_native_features.py"
KERNEL_PATH = ROOT / "backend/cad_guided_measurement.py"


@pytest.fixture(params=("backend", ""), ids=("package", "backend-directory"))
def loaded(request, monkeypatch):
    calls = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise RuntimeError("NATIVE_IMPORT_FORBIDDEN_IN_SOURCE_CHECK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    kernel_name = "backend.cad_guided_measurement" if request.param else "cad_guided_measurement"
    kernel = ModuleType(kernel_name)
    kernel.__dict__.update(dataclass=dataclass, math=math, np=np)
    monkeypatch.setitem(sys.modules, kernel_name, kernel)
    kernel_tree = ast.parse(KERNEL_PATH.read_text(), filename=str(KERNEL_PATH))
    selected = [node for node in kernel_tree.body
                if isinstance(node, ast.ClassDef) and node.name == "CADGuidedFeature"]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(KERNEL_PATH), "exec"),
         kernel.__dict__)
    if request.param:
        package = ModuleType("backend")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "backend", package)
    name = "backend.cad_guided_native_features" if request.param else "cad_guided_native_features"
    adapter = ModuleType(name)
    adapter.__package__ = request.param
    monkeypatch.setitem(sys.modules, name, adapter)
    tree = ast.parse(ADAPTER_PATH.read_text(), filename=str(ADAPTER_PATH))
    exec(compile(tree, str(ADAPTER_PATH), "exec"), adapter.__dict__)
    assert adapter.CADGuidedFeature is kernel.CADGuidedFeature
    assert calls == []
    yield adapter, calls, tree


def _policy(adapter, **overrides):
    values = dict(flange_window_mm=(1., 4.), search_margin_mm=2.,
                  max_fit_residual_mm=.01, plane_exclusion_mm=.05,
                  max_tangency_error_mm=.05)
    values.update(overrides)
    return adapter.NativeGuidedPolicy(**values)


def _unavailable(result, reason):
    assert result["status"] == "UNAVAILABLE"
    assert result["features"] == ()
    assert result["metadata"]["features"] == []
    assert result["metadata"]["reasons"] == [reason]
    assert result["metadata"]["exhaustive_bend_inventory"] is False
    assert result["metadata"]["production_use_allowed"] is False


@pytest.mark.parametrize("flag", (False, None, 1, "true", np.bool_(True)))
def test_binding_opt_in_is_strict_before_native_import(loaded, flag):
    adapter, calls, _ = loaded
    result = adapter.extract_ocp_guided_features(object(), policy=object(),
                                                unit_status="DECLARED_MM", allow_ocp=flag)
    _unavailable(result, "EXPLICIT_OCP_BINDING_REQUIRED")
    assert calls == []


@pytest.mark.parametrize("binding", (None, "occ", "OCP", ""))
def test_no_other_binding_activates_native_import(loaded, binding):
    adapter, calls, _ = loaded
    result = adapter.extract_ocp_guided_features(object(), policy=object(), allow_ocp=True,
                                                unit_status="DECLARED_MM", native_binding=binding)
    _unavailable(result, "EXPLICIT_OCP_BINDING_REQUIRED")
    assert calls == []


@pytest.mark.parametrize("units", (None, "UNVERIFIED", "mm", "DECLARED_INCH", True))
def test_units_must_be_explicit_before_native_import(loaded, units):
    adapter, calls, _ = loaded
    result = adapter.extract_ocp_guided_features(object(), policy=object(),
                                                unit_status=units, allow_ocp=True)
    _unavailable(result, "EXPLICIT_NATIVE_MILLIMETRE_DECLARATION_REQUIRED")
    assert calls == []


def test_policy_required_before_native_import(loaded):
    adapter, calls, _ = loaded
    result = adapter.extract_ocp_guided_features(object(), policy={}, allow_ocp=True,
                                                unit_status="DECLARED_MM")
    _unavailable(result, "VALIDATED_SEARCH_POLICY_REQUIRED")
    assert calls == []


def test_native_import_failure_does_not_retry_another_binding(loaded):
    adapter, calls, _ = loaded
    result = adapter.extract_ocp_guided_features(object(), policy=_policy(adapter), allow_ocp=True,
                                                unit_status="DECLARED_MM")
    _unavailable(result, "NATIVE_EXTRACTION_ERROR:RuntimeError")
    assert calls == ["OCP.BRep"]


def test_policy_owns_frozen_snapshot_and_reuses_exact_kernel_validation(loaded):
    adapter, calls, _ = loaded
    window = [1, 4]
    policy = _policy(adapter, flange_window_mm=window)
    window[0] = 3
    assert policy.flange_window_mm == (1., 4.)
    assert policy.fit_model == "measured_flange_tangent"
    with pytest.raises(FrozenInstanceError):
        policy.search_margin_mm = 10
    assert calls == []


@pytest.mark.parametrize("override", (
    {"flange_window_mm": (0, 4)},
    {"flange_window_mm": (4, 1)},
    {"search_margin_mm": True},
    {"search_margin_mm": float("nan")},
    {"max_fit_residual_mm": .1},
    {"plane_exclusion_mm": 2.},
    {"max_tangency_error_mm": -1.},
    {"fit_model": "copy_cad_radius"},
))
def test_invalid_search_policy_rejected_without_native_import(loaded, override):
    adapter, calls, _ = loaded
    with pytest.raises(ValueError):
        _policy(adapter, **override)
    assert calls == []


def _nested(tree, name, namespace):
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER_PATH), "exec"), scope)
    return scope[name]


def test_native_tolerances_cannot_relax_fixed_geometry_guard(loaded):
    adapter, calls, tree = loaded
    helper = _nested(tree, "tolerance_in_scope", dict(
        math=math, GEOMETRY_TOL_MM=adapter.GEOMETRY_TOL_MM,
        BRep_Tool=SimpleNamespace(Tolerance_s=lambda value: value)))
    for invalid in (-math.inf, -1., -1e-12, math.nan, math.inf, 2.00001e-6):
        assert not helper(invalid)
    assert helper(0.) and helper(1e-7) and helper(2e-6)
    assert calls == []


def test_explorer_snapshots_mutating_native_current_handles(loaded):
    adapter, calls, tree = loaded

    class Handle:
        def __init__(self, value):
            self.value = value

        def Orientation(self):
            return "FWD"

        def Oriented(self, orientation):
            assert orientation == "FWD"
            return Handle(self.value)

    class Explorer:
        def __init__(self, item, kind):
            self.handle = Handle(0)

        def More(self):
            return self.handle.value < 3

        def Current(self):
            return self.handle

        def Next(self):
            self.handle.value += 1

    explore = _nested(tree, "explore", dict(TopExp_Explorer=Explorer, _require=adapter._require))
    result = explore(None, None, 3)
    assert [handle.value for handle in result] == [0, 1, 2]
    assert len({id(handle) for handle in result}) == 3
    with pytest.raises(adapter._Unsupported, match="NATIVE_INCIDENCE_BUDGET"):
        explore(None, None, 2)
    assert calls == []


def test_frozen_budgets_and_stack_handle_copy_are_present(loaded):
    adapter, calls, tree = loaded
    assert (adapter.MAX_FACES, adapter.MAX_EDGE_USES, adapter.MAX_TREE_NODES,
            adapter.MAX_TREE_DEPTH, adapter.GEOMETRY_TOL_MM, adapter.DIRECTION_TOL) == (
                128, 1024, 4096, 16, 2e-6, 1e-7)
    pushes = [node for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "stack"
              and node.func.attr == "append"]
    assert len(pushes) == 1
    pushed = pushes[0].args[0]
    assert isinstance(pushed, ast.Tuple)
    assert isinstance(pushed.elts[0], ast.Call)
    assert isinstance(pushed.elts[0].func, ast.Attribute)
    assert pushed.elts[0].func.attr == "Oriented"
    assert calls == []
