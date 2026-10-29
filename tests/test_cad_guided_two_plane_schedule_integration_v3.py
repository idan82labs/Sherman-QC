"""Fixture-corrected successor to the terminal V2 integration gate.

V2 retained 25 passes and 44 setup errors: its mock omitted the scan-envelope
header after adding CAD patch evidence. This successor adds only that required
mock header. It does not qualify complete V7 metadata, native input or geometry.
The same 69 outcomes, engine bytes, physical policies and budgets remain.

Independent tiny-array and mocked-byte opt-in integration controls.

The two-rectangle checks really prepare owned supports, schedule and finite-query
them; they are not native, global-pose or measurement qualification. Byte tests
mock decoding/native/search/measurement boundaries but exercise both maintained
workflows and their real requested-returned configuration/context binders.
No target import or test executes at module import. No pytest fixture is called
as an ordinary function; only two reviewed ordinary V13 constructors are reused.
"""
from dataclasses import asdict, replace
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


@pytest.fixture
def modules():
    def belongs(key): return key == "backend" or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if belongs(key)}
    for key in saved: del sys.modules[key]
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    sys.modules["backend"] = package
    names = dict(initialize="cad_guided_plane_initialization_v12",
        workflow_v12="cad_guided_mesh_offline_v12", workflow_v13="cad_guided_mesh_offline_v13",
        schedule_v1="cad_guided_two_plane_schedule_v1", schedule_v2="cad_guided_two_plane_schedule_v2",
        mesh="cad_guided_mesh_input_v1", topology="cad_guided_mesh_topology_v4",
        native="cad_guided_native_features", native_v1="cad_guided_native_features_v1",
        surface="cad_guided_cad_surface_v2", measurement="cad_guided_context_measurement_v1",
        cover="cad_guided_support_cover_measurement_v2", instances="cad_guided_instances")
    try:
        yield types.SimpleNamespace(**{key: importlib.import_module("backend."+name)
            for key, name in names.items()})
    finally:
        for key in tuple(sys.modules):
            if belongs(key): del sys.modules[key]
        sys.modules.update(saved)


def _policy(m, rows=50):
    return m.initialize._v1.WholePartRegistrationPolicyV1(coarse_points=rows,
        refine_points=rows, coarse_iterations=1, refine_iterations=1)


def _owned(value, dtype="<f8"):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)


def _sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.tobytes()).hexdigest()


def _tiny_rectangles():
    """Two finite orthogonal rectangles, 25+25 distinct original row IDs.

    Shared positions along the intersection do not weld the two raw components.
    No CAD label, pose or known count is passed as a scan inference answer.
    """
    def mesh(n, plane):
        points = []
        for u in np.linspace(-8., 8., n):
            for v in np.linspace(-6., 6., n):
                points.append((0., u, v) if plane == 0 else (u, v, 0.))
        faces = []
        for i in range(n-1):
            for j in range(n-1):
                a = i*n+j
                faces.extend(((a, a+n, a+1), (a+1, a+n, a+n+1)))
        return np.asarray(points), np.asarray(faces, dtype=np.int64)
    parts = [mesh(5, i) for i in range(2)]
    nominal = np.vstack([p for p, _ in parts])
    raw = np.vstack([t+25*i for i, (_, t) in enumerate(parts)])
    coarse = [mesh(2, i) for i in range(2)]
    cad = np.vstack([p[t] for p, t in coarse])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    translation = np.array([7., -11., 3.])
    observed = nominal@rotation.T+translation
    inputs = (_owned(observed), _owned(raw, "<i8"), _owned(cad),
        _owned([101, 101, 307, 307], "<i8"), _owned([0, 1, 0, 1], "<i8"))
    return inputs, nominal


def _prefilter(m, inputs, **options):
    return m.initialize.prefilter_plane_initializations_v12(*inputs,
        position_error_bounds_mm=.05, registration_policy=_policy(m), unit_assumption=UNIT,
        search_budgets=m.initialize._search.PlaneSearchBudgetsV6(max_bank_planes=2), **options)


@pytest.mark.parametrize("enabled", (False, True))
def test_default_configuration_adds_no_schedule_policy_or_publication(modules, enabled):
    assert modules.initialize._two_plane_schedule_configuration(enabled, "v1", None) == {}
    modules.initialize._bind_initialization_schedule_configuration({"two_plane_schedule": None}, {})
    with pytest.raises(ValueError):
        modules.initialize._bind_initialization_schedule_configuration(
            {"two_plane_schedule": {"schema_version": "two_plane_schedule_v2"}}, {})
    for function in (modules.initialize.prefilter_plane_initializations_v12,
            modules.initialize.register_whole_part_planes_v12,
            modules.workflow_v12.inspect_cad_guided_mesh_v12,
            modules.workflow_v13.inspect_cad_guided_mesh_v13):
        signature = inspect.signature(function)
        assert signature.parameters["two_plane_schedule_mode"].default == "v1"
        assert signature.parameters["two_plane_schedule_budgets"].default is None


def test_v2_configuration_owns_exact_typed_budget_snapshot(modules):
    source = modules.schedule_v2.TwoPlaneScheduleBudgetsV2(max_selected_scan_pairs=3)
    expected = asdict(source)
    configuration = modules.initialize._two_plane_schedule_configuration(True, "v2", source)
    assert configuration["two_plane_schedule_budgets"] is not source
    assert type(configuration["two_plane_schedule_budgets"]) is modules.schedule_v2.TwoPlaneScheduleBudgetsV2
    object.__setattr__(source, "max_selected_scan_pairs", 0)
    assert asdict(configuration["two_plane_schedule_budgets"]) == expected
    assert configuration["two_plane_schedule_configuration"] == dict(mode="v2", budgets=expected)
    publication = dict(two_plane_schedule_configuration=copy.deepcopy(configuration["two_plane_schedule_configuration"]))
    modules.initialize._bind_initialization_schedule_configuration(publication, configuration)
    with pytest.raises(ValueError): modules.initialize._bind_initialization_schedule_configuration({}, configuration)
    publication["two_plane_schedule_configuration"]["budgets"]["max_selected_scan_pairs"] = 3.
    with pytest.raises(ValueError): modules.initialize._bind_initialization_schedule_configuration(publication, configuration)


@pytest.mark.parametrize("fault", ("boolean_mode", "unknown_mode", "missing_v2_budget", "v1_budget_type",
    "dictionary_budget", "budget_with_v1", "disabled_v2", "nonboolean_enabled", "boolean_cap"))
def test_invalid_or_ignored_schedule_requests_refuse_before_geometry(modules, fault):
    enabled, mode, budget = True, "v2", modules.schedule_v2.TwoPlaneScheduleBudgetsV2()
    if fault == "boolean_mode": mode = True
    elif fault == "unknown_mode": mode = "V2"
    elif fault == "missing_v2_budget": budget = None
    elif fault == "v1_budget_type": budget = modules.schedule_v1.TwoPlaneScheduleBudgetsV1()
    elif fault == "dictionary_budget": budget = asdict(budget)
    elif fault == "budget_with_v1": mode = "v1"
    elif fault == "disabled_v2": enabled = False
    elif fault == "nonboolean_enabled": enabled = 1
    elif fault == "boolean_cap": budget = replace(budget, max_residual_pair_visits=True)
    with pytest.raises(ValueError): modules.initialize._two_plane_schedule_configuration(enabled, mode, budget)


@pytest.mark.parametrize("mode", ("v1", "v2"))
def test_actual_owned_fifty_row_prefilter_dispatches_both_orders_and_all_slots(modules, monkeypatch, mode):
    inputs, nominal = _tiny_rectangles()
    before = tuple(a.tobytes() for a in inputs)
    calls = []
    for version in ("v1", "v2"):
        module = getattr(modules, "schedule_"+version)
        name = "schedule_two_plane_pairs_"+version
        original = getattr(module, name)
        def call(*args, _original=original, _version=version, **kwargs):
            calls.append((_version, args, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, call)
    options = {} if mode == "v1" else dict(two_plane_schedule_mode="v2",
        two_plane_schedule_budgets=modules.schedule_v2.TwoPlaneScheduleBudgetsV2())
    answer = _prefilter(modules, inputs, **options)
    assert answer["status"] == "COMPLETE" and answer["start_groups"], answer["reasons"]
    assert [row[0] for row in calls] == [mode]
    assert len(answer["hypothesis_evidence"]["scan_bank"]) == 2
    joint = answer["joint_proposal_accounting"]
    assert joint["triplet_records"] == 0 and joint["ordered_pair_matches"] == 2
    assert joint["pair_slot_reservation"] == joint["total_hypothesis_records"] == 64
    assert len(answer["hypotheses"]) == 64
    assert answer["two_plane_context"]["statistics"]["slot_records_completed"] == 64
    matches = answer["two_plane_schedule"]["scheduled_matches"]
    assert matches[0]["scan_plane_ids"] == matches[1]["scan_plane_ids"]
    assert matches[0]["cad_plane_ids"] == matches[1]["cad_plane_ids"][::-1]
    assert all(len(row["correspondences"]) == 2 for row in answer["hypotheses"])
    assert answer["input_digests"]["scan_points_sha256"] == _sha(inputs[0])
    assert answer["input_digests"]["scan_triangle_raw_rows_sha256"] == _sha(inputs[1])
    assert answer["input_digests"]["position_error_bounds_mm_sha256"] == _sha(np.full(50, .05, dtype="<f8"))
    assert before == tuple(a.tobytes() for a in inputs)
    assert any(np.max(np.linalg.norm(inputs[0]@np.asarray(group["initialized_rotation"]).T
        +group["initialized_translation_mm"]-nominal, axis=1)) < 1e-8 for group in answer["start_groups"])
    assert answer["accepted_registration"] is False and answer["observed_bend_count"] is None
    if mode == "v1": assert "two_plane_schedule_configuration" not in answer
    else:
        assert answer["two_plane_schedule_configuration"] == dict(mode="v2",
            budgets=asdict(options["two_plane_schedule_budgets"]))
        assert answer["two_plane_schedule"]["schema_version"] == "two_plane_schedule_v2"


@pytest.mark.parametrize("fault", ("schema", "budgets", "scan_digest", "cad_digest", "joint_count", "helper_refusal", "prefix_bool"))
def test_actual_prepared_supports_do_not_reach_context_from_invalid_v2_schedule(modules, monkeypatch, fault):
    inputs, _ = _tiny_rectangles(); calls = []
    original = modules.schedule_v2.schedule_two_plane_pairs_v2
    def damaged(*args, **kwargs):
        answer = original(*args, **kwargs)
        assert answer["status"] == "COMPLETE" and answer["scheduled_matches"]
        if fault == "schema": answer["schema_version"] = "two_plane_schedule_v1"
        elif fault == "budgets": answer["budgets"]["max_selected_scan_pairs"] -= 1
        elif fault == "scan_digest": answer["input_digests"]["scan_bank_equations_sha256"] = "f"*64
        elif fault == "cad_digest": answer["input_digests"]["cad_bank_equations_sha256"] = "f"*64
        elif fault == "joint_count": answer["joint_budget"]["reserved_two_plane_hypothesis_records"] -= 1
        elif fault == "prefix_bool":
            answer["breadth_ledger"] = copy.deepcopy(answer["breadth_ledger"])
            answer["breadth_ledger"]["scheduled_matches"][0]["ordered_match_index"] = False
            assert type(answer["scheduled_matches"][0]["ordered_match_index"]) is int
        else: answer.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False)
        return answer
    def forbidden(*args, **kwargs):
        calls.append("context")
        raise AssertionError("INVALID_SCHEDULE_MUST_PRECEDE_CONTEXT")
    monkeypatch.setattr(modules.schedule_v2, "schedule_two_plane_pairs_v2", damaged)
    monkeypatch.setattr(modules.initialize._context, "propose_two_plane_context_v1", forbidden)
    answer = _prefilter(modules, inputs, two_plane_schedule_mode="v2",
        two_plane_schedule_budgets=modules.schedule_v2.TwoPlaneScheduleBudgetsV2())
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["numerical_consumer_complete"] is False
    assert answer["statistics"]["completed_query_calls"] == 0 and calls == []


@pytest.mark.parametrize("marker", ("schema_version", "breadth_ledger", "residual_windows", "residual_statistics"))
def test_default_prefilter_rejects_unrequested_successor_before_context(modules, monkeypatch, marker):
    inputs, _ = _tiny_rectangles()
    calls = []
    original = modules.schedule_v1.schedule_two_plane_pairs_v1
    def damaged(*args, **kwargs):
        answer = original(*args, **kwargs)
        assert answer["status"] == "COMPLETE" and answer["scheduled_matches"]
        answer[marker] = "two_plane_schedule_v2" if marker == "schema_version" else {}
        return answer
    def forbidden(*args, **kwargs):
        calls.append("context")
        raise AssertionError("UNREQUESTED_SUCCESSOR_MUST_PRECEDE_CONTEXT")
    monkeypatch.setattr(modules.schedule_v1, "schedule_two_plane_pairs_v1", damaged)
    monkeypatch.setattr(modules.initialize._context, "propose_two_plane_context_v1", forbidden)
    answer = _prefilter(modules, inputs)
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["numerical_consumer_complete"] is False
    assert answer["statistics"]["completed_query_calls"] == 0 and calls == []
    assert "UNREQUESTED_V2_SCHEDULE_PUBLICATION" in answer["reasons"]


@pytest.fixture
def adapter(modules, monkeypatch):
    # Read-only ordinary constructors, not a fixture invocation or test import
    # side effect. Native/search returns remain deliberately mocked below.
    spec = importlib.util.spec_from_file_location("reviewed_v13_schedule_fixture_helpers",
        ROOT/"tests/test_cad_guided_mesh_offline_v13.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scan, topology, cad = helper._arrays()
    native = modules.native
    feature = native.CADGuidedFeature(feature_id="invented-schedule-binding-feature",
        nominal_radius_mm=5., nominal_sweep_deg=90., origin=(0., 0., 0.), axis=(0., 0., 1.),
        start_radial=(1., 0., 0.), span_mm=(0., 12.), flange_window_mm=(2.5, 12.5),
        search_margin_mm=3., max_fit_residual_mm=.05, plane_exclusion_mm=.075,
        max_tangency_error_mm=.1, fit_model="free_circle", nominal_source="MOCK_NATIVE_NO_EXECUTION")
    state = types.SimpleNamespace(scan=scan, topology=topology, cad=cad, roles=object(), shape=object(),
        events=[], binding_calls=[], register_calls=[], measurement_calls=[], validation_calls=[],
        damage_second=None, mutate_during_register=None,
        schedule_budgets=modules.schedule_v2.TwoPlaneScheduleBudgetsV2(max_selected_scan_pairs=0))
    originals = {version: getattr(modules, "workflow_"+version)._registration_binding for version in ("v12", "v13")}
    def install(mode):
        registration = helper._registration(types.SimpleNamespace(workflow=modules.workflow_v13), scan, topology, cad)
        if mode == "v2":
            initialization = registration["evidence"]["initialization"]
            normals = ((0., 0., 1.), (0., 0., 1.), (1., 0., 0.))
            centers = ((8./3., 4./3., 0.), (8./3., 4./3., 2.), (3., 8./3., 4./3.))
            equation = modules.initialize._search.PlaneEquationV6
            scan_eq = tuple(equation(i, normal, (0., 0., 0.), 0., 0.) for i, normal in enumerate(normals))
            cad_eq = tuple(equation(face, normal, center, 0., 0.)
                           for face, normal, center in zip((2, 5, 9), normals, centers))
            configuration = dict(mode="v2", budgets=asdict(state.schedule_budgets))
            schedule = modules.schedule_v2.schedule_two_plane_pairs_v2(scan_eq, cad_eq,
                retained_triplet_hypothesis_count=2, retained_triplet_unsigned_count=2,
                budgets=state.schedule_budgets)
            assert schedule["status"] == "COMPLETE" and schedule["scheduled_matches"] == []
            initialization.update(two_plane_schedule=schedule,
                two_plane_schedule_configuration=copy.deepcopy(configuration),
                budgets=asdict(modules.initialize.PlaneInitializationBudgetsV12()),
                hypothesis_evidence=dict(scan_bank=[0, 1, 2], scan_planes=[
                    dict(equation=modules.initialize._search._equation(row)) for row in scan_eq]),
                patch_evidence=dict(scan=dict(format="noisy_mesh_metadata_schema_v6",
                    schemas=[["schema_version"]], root=["D", 0, ["noisy_mesh_planes_v7"]]),
                    cad=dict(patches=[dict(plane_id=face, normal=list(normal),
                    area_centroid_mm=list(center)) for face, normal, center in zip((2, 5, 9), normals, centers)])))
            registration["evidence"]["two_plane_schedule_configuration"] = copy.deepcopy(configuration)
            state.expected = configuration
        else: state.expected = None
        state.registration = registration
        # Each malformed-publication test starts from a real successful binding
        # of BOTH mocked contexts. Zero pair selection is explicit, not a claim
        # of two-plane geometric recovery in this byte-plumbing fixture.
        kwargs = {} if mode == "v1" else dict(expected_two_plane_schedule_configuration=state.expected)
        for index in (0, 1):
            originals["v12"](registration, scan, topology, cad, scan["engine_to_raw_row_indices"], .05,
                context_index=index, **kwargs)
    state.install = install
    install("v2")
    state.measured = dict(status="PARTIAL", computation_status="COMPLETE", reasons=[],
        predictions=[dict(fixture="MOCK_CONDITIONAL_PREDICTION_NOT_GEOMETRY")],
        observed_bend_count=None, production_use_allowed=False)
    def prepare(**kwargs):
        state.events.append("prepare")
        assert kwargs == dict(scan_bytes=helper.SCAN_BYTES, max_engine_rows=36)
        return scan
    def parse(blob):
        state.events.append("parse"); assert blob is helper.SCAN_BYTES
        return topology
    def read(blob):
        state.events.append("native_read"); assert blob is helper.CAD_BYTES
        return state.shape, dict(target_unit="mm", physical_calibration="NOT_ESTABLISHED")
    def extract(shape, **kwargs):
        state.events.append("native_features"); assert shape is state.shape
        return dict(status="PARTIAL", features=(feature,), metadata=dict(fixture="MOCK_NATIVE"))
    def sample(shape, **kwargs):
        state.events.append("sample"); assert shape is state.shape and kwargs["sample_count"] == 36
        return cad
    def register(*args, **kwargs):
        state.events.append("register"); state.register_calls.append((args, kwargs))
        if state.mutate_during_register is not None: state.mutate_during_register()
        return state.registration, state.roles
    def measure(raw, features, **kwargs):
        state.events.append("measure"); state.measurement_calls.append((raw, features, kwargs))
        assert raw is scan["raw_points"] and kwargs["surface_roles"] is state.roles
        assert kwargs["registration"] is state.registration and kwargs["scan_sha256"] == _sha(helper.SCAN_BYTES)
        return state.measured
    def validate(measured, **kwargs):
        state.events.append("validate"); state.validation_calls.append((measured, kwargs))
        assert measured is state.measured
        return dict(fixture="MOCK_VALIDATOR_NO_METROLOGY")
    monkeypatch.setattr(modules.mesh, "prepare_mesh_scan_v1", prepare)
    monkeypatch.setattr(modules.topology, "parse_triangle_mesh_ply_v4", parse)
    monkeypatch.setattr(modules.workflow_v12._v2, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(modules.native, "extract_ocp_guided_features", extract)
    monkeypatch.setattr(modules.native_v1, "extract_ocp_guided_features_v1", extract)
    monkeypatch.setattr(modules.surface, "sample_native_cad_surface_v2", sample)
    monkeypatch.setattr(modules.initialize, "register_measurement_contexts_v12", register)
    monkeypatch.setattr(modules.measurement, "inspect_generated_context_windows_v1", measure)
    monkeypatch.setattr(modules.measurement, "validate_context_measurement_publication_v1", validate)
    monkeypatch.setattr(modules.cover, "inspect_support_cover_context_windows_v2", measure)
    monkeypatch.setattr(modules.cover, "validate_support_cover_publication_v2", validate)
    for version in ("v12", "v13"):
        def bound(*args, _original=originals[version], **kwargs):
            index = kwargs["context_index"]
            state.events.append("bind:"+str(index)); state.binding_calls.append((args, kwargs))
            if index == 1 and state.damage_second is not None: state.damage_second()
            return _original(*args, **kwargs)
        monkeypatch.setattr(getattr(modules, "workflow_"+version), "_registration_binding", bound)
    def run(version, requested="v2", **overrides):
        workflow = getattr(modules, "workflow_"+version)
        policy_type = getattr(workflow, "MeshWorkflowPolicy"+version.upper())
        kwargs = dict(scan_bytes=helper.SCAN_BYTES, cad_bytes=helper.CAD_BYTES,
            position_error_bound_mm=.05, allow_ocp=True, connected_measurement_contexts=True,
            workflow_policy=policy_type(max_registration_scan_rows=36, cad_sample_count=36, native_adapter_version="v1"),
            registration_policy=_policy(modules, 36),
            measurement_policy=native.NativeGuidedPolicy(flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
                max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
            instance_policy=modules.instances.SupportPolicy(4., 5., 20., 3))
        if requested == "v2":
            kwargs.update(two_plane_schedule_mode="v2", two_plane_schedule_budgets=state.schedule_budgets)
        elif requested == "explicit_v1": kwargs.update(two_plane_schedule_mode="v1", two_plane_schedule_budgets=None)
        kwargs.update(overrides)
        return getattr(workflow, "inspect_cad_guided_mesh_"+version)(**kwargs)
    state.run = run
    return state


@pytest.mark.parametrize("version", ("v12", "v13"))
def test_actual_byte_workflows_bind_requested_v2_configuration_before_all_measurements(adapter, version):
    state = adapter; result = state.run(version)
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE", result
    assert state.events == ["prepare", "parse", "native_read", "native_features", "sample", "register",
                            "bind:0", "bind:1", "measure", "validate"]
    assert result["policies"]["two_plane_schedule_configuration"] == state.expected
    assert len(result["registration_array_binding"]) == 2
    for binding in result["registration_array_binding"]:
        assert binding["two_plane_schedule_mode"] == "v2"
        assert binding["two_plane_schedule_input_digests"] == state.registration["evidence"]["initialization"]["two_plane_schedule"]["input_digests"]
        assert binding["two_plane_schedule_configuration_sha256"] == hashlib.sha256(json.dumps(
            state.expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    args, kwargs = state.register_calls[0]
    assert args[0] is state.scan["raw_points"] and args[1] is state.topology["triangle_raw_rows"]
    assert kwargs["two_plane_schedule_mode"] == "v2"
    assert kwargs["two_plane_schedule_budgets"] is not state.schedule_budgets
    assert asdict(kwargs["two_plane_schedule_budgets"]) == state.expected["budgets"]
    assert result["observed_bend_count"] is None and result["registration"]["transform_scan_to_cad"] is None


@pytest.mark.parametrize("version", ("v12", "v13"))
def test_omitted_and_explicit_v1_byte_paths_have_identical_outputs_and_no_extra_kwargs(adapter, version):
    state = adapter; state.install("v1")
    default = state.run(version, "v1")
    explicit = state.run(version, "explicit_v1")
    assert default["status"] == explicit["status"] == "PARTIAL"
    assert default == explicit
    assert "two_plane_schedule_configuration" not in default["policies"]
    assert all("two_plane_schedule_mode" not in kwargs and "two_plane_schedule_budgets" not in kwargs
               for _, kwargs in state.register_calls)
    assert all("two_plane_schedule_mode" not in row for row in default["registration_array_binding"])


@pytest.mark.parametrize("version", ("v12", "v13"))
@pytest.mark.parametrize("fault", ("missing_top", "missing_initialization", "top_mode", "top_budget",
    "initialization_budget", "initialization_float_budget", "boolean_budget", "schedule_schema", "schedule_budget", "scan_digest",
    "cad_digest", "source_equation", "cad_equation", "joint_count", "disabled_lane", "unrequested"))
def test_stale_requested_returned_schedule_contract_never_measures(adapter, version, fault):
    state = adapter; evidence = state.registration["evidence"]; init = evidence["initialization"]
    requested = "v2"
    if fault == "missing_top": evidence.pop("two_plane_schedule_configuration")
    elif fault == "missing_initialization": init.pop("two_plane_schedule_configuration")
    elif fault == "top_mode": evidence["two_plane_schedule_configuration"]["mode"] = "v1"
    elif fault == "top_budget": evidence["two_plane_schedule_configuration"]["budgets"]["max_window_pair_tests"] -= 1
    elif fault == "initialization_budget": init["budgets"]["max_hypothesis_records"] -= 1
    elif fault == "initialization_float_budget": init["budgets"]["max_hypothesis_records"] = float(init["budgets"]["max_hypothesis_records"])
    elif fault == "boolean_budget": init["two_plane_schedule_configuration"]["budgets"]["max_selected_scan_pairs"] = False
    elif fault == "schedule_schema": init["two_plane_schedule"]["schema_version"] = "two_plane_schedule_v1"
    elif fault == "schedule_budget": init["two_plane_schedule"]["budgets"]["max_cad_pair_records"] -= 1
    elif fault == "scan_digest": init["two_plane_schedule"]["input_digests"]["scan_bank_equations_sha256"] = "f"*64
    elif fault == "cad_digest": init["two_plane_schedule"]["input_digests"]["cad_bank_equations_sha256"] = "f"*64
    elif fault == "source_equation": init["hypothesis_evidence"]["scan_planes"][0]["equation"]["offset_from_anchor_mm"] = 1.
    elif fault == "cad_equation": init["patch_evidence"]["cad"]["patches"][0]["area_centroid_mm"][0] += 1.
    elif fault == "joint_count": init["two_plane_schedule"]["joint_budget"]["joint_hypothesis_records_reserved"] += 1
    elif fault == "disabled_lane": init["joint_proposal_accounting"]["pair_lane_enabled"] = False
    elif fault == "unrequested": requested = "v1"
    result = state.run(version, requested)
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION", result
    assert result["measurements"] is None and result["predictions"] == []
    assert "registration_array_binding" not in result
    assert result["registration"]["status"] == "UNAVAILABLE"
    assert result["registration"]["evidence"]["measurement_context_indices"] == []
    assert result["registration"]["evidence"]["attempted_measurement_context_indices"] == [0, 1]
    assert state.measurement_calls == state.validation_calls == []


@pytest.mark.parametrize("version", ("v12", "v13"))
def test_second_context_schedule_binding_failure_discards_successful_first_binding(adapter, version):
    state = adapter
    def damage():
        state.registration["evidence"]["initialization"]["hypothesis_evidence"]["scan_planes"][0]["equation"]["offset_from_anchor_mm"] = 2.
    state.damage_second = damage
    result = state.run(version)
    assert [kwargs["context_index"] for _, kwargs in state.binding_calls] == [0, 1]
    assert result["status"] == "UNAVAILABLE" and result["measurements"] is None and result["predictions"] == []
    assert "registration_array_binding" not in result
    evidence = result["registration"]["evidence"]
    assert evidence["measurement_context_indices"] == [] and evidence["attempted_measurement_context_indices"] == [0, 1]
    assert state.measurement_calls == state.validation_calls == []


@pytest.mark.parametrize("version", ("v12", "v13"))
def test_byte_workflow_request_snapshot_survives_caller_budget_mutation(adapter, version):
    state = adapter
    state.mutate_during_register = lambda: object.__setattr__(state.schedule_budgets, "max_selected_scan_pairs", 1)
    result = state.run(version)
    assert result["status"] == "PARTIAL"
    assert state.schedule_budgets.max_selected_scan_pairs == 1
    assert result["policies"]["two_plane_schedule_configuration"]["budgets"]["max_selected_scan_pairs"] == 0
    assert state.register_calls[0][1]["two_plane_schedule_budgets"].max_selected_scan_pairs == 0


@pytest.mark.parametrize("version", ("v12", "v13"))
@pytest.mark.parametrize("fault", ("v1_budget", "disabled_v2"))
def test_byte_request_contradictions_refuse_before_any_input_access(adapter, version, fault):
    state = adapter
    options = dict(two_plane_schedule_mode="v1") if fault == "v1_budget" else dict(two_plane_enabled=False)
    result = state.run(version, **options)
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert state.events == []

