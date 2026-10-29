"""Fixture-only V2 successor for the eleven unreached public adapter checks.

The V1 attempt remains terminal RED. Only the original module fixture, policy
helper, adapter fixture and final two test functions are retained here. The sole
behavioral correction supplies NativeGuidedPolicy's five required keywords with
the unchanged values used by the frozen V13 adapter tests. No earlier passing
outcome is repeated. Native/search returns and measurement refusal remain mocks;
actual tiny composite preparation, role admission and context binding remain real.
No native/connected/physical qualification or registration positive is claimed.
"""


from dataclasses import asdict
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules():
    # V7 has a literal backend import: one inert namespace preserves exact
    # dataclass identities without importing the server package initializer.
    def owned(key): return key == "backend" or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved: del sys.modules[key]
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    sys.modules["backend"] = package
    mapping = dict(initialize="cad_guided_plane_initialization_v12",
        previous="cad_guided_plane_initialization_v8",
        composite="cad_guided_scan_plane_preparation_v1",
        composite_roles="cad_guided_composite_surface_roles_v1",
        roles="cad_guided_context_surface_roles_v1",
        workflow_v12="cad_guided_mesh_offline_v12", workflow_v13="cad_guided_mesh_offline_v13",
        mesh="cad_guided_mesh_input_v1", topology="cad_guided_mesh_topology_v4",
        native="cad_guided_native_features", native_v1="cad_guided_native_features_v1",
        surface="cad_guided_cad_surface_v2", instances="cad_guided_instances",
        measurement="cad_guided_context_measurement_v1", cover="cad_guided_support_cover_measurement_v2")
    try:
        yield types.SimpleNamespace(**{key: importlib.import_module("backend."+name)
            for key, name in mapping.items()})
    finally:
        for key in tuple(sys.modules):
            if owned(key): del sys.modules[key]
        sys.modules.update(saved)


def _policy(modules):
    return modules.initialize._v1.WholePartRegistrationPolicyV1(coarse_points=36,
        refine_points=36, coarse_iterations=1, refine_iterations=1)


@pytest.fixture
def adapter_case(modules, monkeypatch):
    # Reuse only two ordinary fixture constructors from the fully reviewed
    # 43-check module. No pytest fixture is called and none of its tests run.
    # They provide invented bytes, dyadic CAD samples, and explicitly MOCKED
    # registration contexts, not geometric or native qualification.
    spec = importlib.util.spec_from_file_location("reviewed_v13_invented_fixture_helpers",
        ROOT/"tests/test_cad_guided_mesh_offline_v13.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scan, topology, cad = helper._arrays()
    registration = helper._registration(types.SimpleNamespace(workflow=modules.workflow_v13), scan, topology, cad)
    patches = modules.composite.prepare_composite_scan_planes_v1(scan["raw_points"],
        topology["triangle_raw_rows"], position_error_bounds_mm=.05)
    assert patches["status"] == "COMPLETE", patches
    encoded = modules.previous._compact.encode_metadata_v7(patches)
    prepared = modules.composite_roles.prepare_composite_surface_roles_v1(scan["raw_points"],
        topology["triangle_raw_rows"], patches, encoded, patches["input_digests"]["position_error_bounds_mm_sha256"])
    seeded = modules.composite._seeded.SeededPlaneBudgetsV1()
    combined = modules.composite.CompositeScanPlaneBudgetsV1()
    expected = dict(mode="composite_v1", seeded_budgets=asdict(seeded), composite_budgets=asdict(combined))
    evidence = registration["evidence"]
    evidence["scan_plane_configuration"] = copy.deepcopy(expected)
    evidence["initialization"]["scan_plane_configuration"] = copy.deepcopy(expected)
    evidence["initialization"]["patch_evidence"] = dict(scan=encoded)
    evidence["surface_role_preparation"] = dict(patches["input_digests"],
        metadata_sha256=prepared.metadata_sha256, raw_row_count=len(scan["raw_points"]))
    # Establish a positive BINDING control on both generated contexts before
    # introducing the fault. Search/pose evidence is still explicitly invented.
    for index in (0, 1):
        binding = modules.workflow_v12._registration_binding(registration, scan, topology, cad,
            scan["engine_to_raw_row_indices"], .05, context_index=index,
            expected_scan_plane_configuration=expected)
        assert binding["scan_plane_mode"] == "composite_v1"
        assert binding["scan_plane_metadata_sha256"] == prepared.metadata_sha256
    state = types.SimpleNamespace(scan=scan, topology=topology, cad=cad, registration=registration,
        prepared=prepared, expected=expected, seeded=seeded, combined=combined,
        native_calls=[], register_calls=[], measurement_calls=[], binding_failures=[], scan_binding_calls=[])
    def prepare(**kwargs):
        assert kwargs == dict(scan_bytes=helper.SCAN_BYTES, max_engine_rows=36)
        return scan
    def parse(blob):
        assert blob is helper.SCAN_BYTES
        return topology
    shape = object()
    def read(blob):
        assert blob is helper.CAD_BYTES
        state.native_calls.append("read")
        return shape, dict(target_unit="mm", physical_calibration="NOT_ESTABLISHED")
    def extract(value, **kwargs):
        assert value is shape and kwargs["allow_ocp"] is True
        state.native_calls.append("extract")
        return dict(status="AVAILABLE", metadata=dict(fixture="MOCK_NATIVE_NOT_EXECUTED"), features=(object(),))
    def sample(value, **kwargs):
        assert value is shape and kwargs["sample_count"] == 36 and kwargs["allow_ocp"] is True
        state.native_calls.append("sample")
        return cad
    def register(*args, **kwargs):
        state.register_calls.append((args, kwargs))
        return registration, prepared
    def no_measurement(*args, **kwargs):
        state.measurement_calls.append(True)
        raise AssertionError("CONFIGURATION_MISMATCH_MUST_PRECEDE_MEASUREMENT")
    real_scan_binding = modules.workflow_v12._scan_plane_binding
    def bound(*args, **kwargs):
        state.scan_binding_calls.append((args, kwargs))
        try:
            return real_scan_binding(*args, **kwargs)
        except ValueError as error:
            state.binding_failures.append(str(error))
            raise
    monkeypatch.setattr(modules.mesh, "prepare_mesh_scan_v1", prepare)
    monkeypatch.setattr(modules.topology, "parse_triangle_mesh_ply_v4", parse)
    monkeypatch.setattr(modules.workflow_v12._v2, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(modules.native, "extract_ocp_guided_features", extract)
    monkeypatch.setattr(modules.native_v1, "extract_ocp_guided_features_v1", extract)
    monkeypatch.setattr(modules.surface, "sample_native_cad_surface_v2", sample)
    monkeypatch.setattr(modules.initialize, "register_measurement_contexts_v12", register)
    monkeypatch.setattr(modules.measurement, "inspect_generated_context_windows_v1", no_measurement)
    monkeypatch.setattr(modules.cover, "inspect_support_cover_context_windows_v2", no_measurement)
    monkeypatch.setattr(modules.workflow_v12, "_scan_plane_binding", bound)
    def run(version, requested):
        workflow = getattr(modules, "workflow_"+version)
        policy_class = getattr(workflow, "MeshWorkflowPolicy"+version.upper())
        kwargs = dict(scan_bytes=helper.SCAN_BYTES, cad_bytes=helper.CAD_BYTES,
            position_error_bound_mm=.05, allow_ocp=True,
            workflow_policy=policy_class(max_registration_scan_rows=36, cad_sample_count=36),
            registration_policy=_policy(modules),
            measurement_policy=modules.native.NativeGuidedPolicy(flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
                max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
            instance_policy=modules.instances.SupportPolicy(4., 5., 20., 3), scan_plane_mode=requested)
        if requested == "composite_v1":
            kwargs.update(seeded_plane_budgets=seeded, composite_plane_budgets=combined)
        return getattr(workflow, "inspect_cad_guided_mesh_"+version)(**kwargs)
    state.run = run
    return state


@pytest.mark.parametrize("version", ["v12", "v13"])
@pytest.mark.parametrize("fault,expected_reason", [
    ("missing_returned_configuration", "INVALID_EXPECTED_SCAN_PLANE_CONFIGURATION"),
    ("top_budget_mismatch", "REGISTRATION_SCAN_PLANE_CONFIGURATION_MISMATCH"),
    ("initialization_budget_mismatch", "REGISTRATION_SCAN_PLANE_CONFIGURATION_MISMATCH"),
    ("unrequested_composite", "UNREQUESTED_COMPOSITE_SCAN_PLANE_CONFIGURATION"),
    ("hidden_composite_schema", "UNREQUESTED_OR_INVALID_SCAN_PLANE_SCHEMA")])
def test_actual_workflow_requested_returned_configuration_mismatch_never_measures(adapter_case, version, fault, expected_reason):
    state, requested = adapter_case, "composite_v1"
    evidence = state.registration["evidence"]
    initialization = evidence["initialization"]
    if fault == "missing_returned_configuration": initialization.pop("scan_plane_configuration")
    elif fault == "top_budget_mismatch": evidence["scan_plane_configuration"]["seeded_budgets"]["max_seeds"] = 31
    elif fault == "initialization_budget_mismatch": initialization["scan_plane_configuration"]["composite_budgets"]["max_qualified_planes"] = 65551
    elif fault == "unrequested_composite": requested = "v7"
    elif fault == "hidden_composite_schema":
        requested = "v7"
        evidence.pop("scan_plane_configuration")
        initialization.pop("scan_plane_configuration")
    result = state.run(version, requested)
    assert state.native_calls == ["read", "extract", "sample"]
    assert len(state.register_calls) == 1
    assert state.binding_failures == [expected_reason]
    assert state.measurement_calls == []
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION", result
    assert result["measurements"] is None and result["predictions"] == []
    assert result["registration"]["status"] == "UNAVAILABLE" and result["registration"]["transform_scan_to_cad"] is None
    assert "registration_array_binding" not in result
    args, kwargs = state.register_calls[0]
    assert args[0] is state.scan["raw_points"] and args[1] is state.topology["triangle_raw_rows"]
    if requested == "composite_v1":
        assert kwargs["scan_plane_mode"] == "composite_v1"
        assert kwargs["seeded_plane_budgets"] is not state.seeded
        assert kwargs["composite_plane_budgets"] is not state.combined
        assert asdict(kwargs["seeded_plane_budgets"]) == state.expected["seeded_budgets"]
        assert asdict(kwargs["composite_plane_budgets"]) == state.expected["composite_budgets"]
        assert result["policies"]["scan_plane_configuration"] == state.expected
    else:
        assert all(key not in kwargs for key in ("scan_plane_mode", "seeded_plane_budgets", "composite_plane_budgets"))
        assert "scan_plane_configuration" not in result["policies"]


def test_v13_prepares_composite_metadata_once_and_reuses_same_token_for_every_context(adapter_case, modules, monkeypatch):
    state, bound_contexts = adapter_case, []
    real = modules.workflow_v13._registration_binding
    def bind(*args, **kwargs):
        bound_contexts.append((kwargs["context_index"], kwargs["_prepared_scan_plane_binding"]))
        return real(*args, **kwargs)
    monkeypatch.setattr(modules.workflow_v13, "_registration_binding", bind)
    result = state.run("v13", "composite_v1")
    assert len(state.scan_binding_calls) == 1 and state.binding_failures == []
    assert [index for index, _ in bound_contexts] == [0, 1]
    first, second = bound_contexts[0][1], bound_contexts[1][1]
    assert first is second and type(first) is modules.workflow_v12._PreparedScanPlaneBinding
    assert first.registration is state.registration and first.scan is state.scan and first.topology is state.topology
    assert first.metadata_sha256 == state.prepared.metadata_sha256
    assert len(result["registration_array_binding"]) == 2
    assert all(row["scan_plane_mode"] == "composite_v1"
               and row["scan_plane_metadata_sha256"] == state.prepared.metadata_sha256
               for row in result["registration_array_binding"])
    # The complete contract binding intentionally reaches a REFUSING test
    # measurement stub. It is not a successful native/registration/fit result.
    assert state.measurement_calls == [True]
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "FULL_RAW_LOCAL_MEASUREMENT"
    assert result["measurements"] is None and result["predictions"] == []
