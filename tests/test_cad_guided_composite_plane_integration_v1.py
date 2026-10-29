"""Independent explicit composite-mode integration; source-only until gated.

Tiny invented meshes use actual V7/seeded preparation, CAD plane fitting, plane
search projection, and owned role admission. A deliberately invented start only
opens the role-dispatch seam; refinement is explicitly refused and never earns
a registration/context/measurement positive. Adapter controls use the already
reviewed invented-byte fixture with its native/search/measurement mocks.
No physical files, truth labels, known bend count, or native execution is used.
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
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


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


def _mesh(corner=False, registration=False):
    points = [[0., 0., 0.], [16., 0., 0.], [16., 16., 0.], [0., 16., 0.]]
    triangles, faces = [[0, 1, 2], [0, 2, 3]], [7, 7]
    if corner:
        points += [[0., 0., 16.], [16., 0., 16.]]
        triangles += [[0, 5, 1], [0, 4, 5]]
        faces += [11, 11]
    # Exactly 36 unique interior rows meet the separate registration API's
    # admission floor. They remain unreferenced, not invented mesh support.
    interior = np.array([[x, y, 0.] for x in np.linspace(2., 14., 6)
                         for y in np.linspace(2., 14., 6)])
    if registration: points += interior.tolist()
    points, triangles = np.array(points), np.array(triangles, dtype="<i8")
    return dict(points=points, triangles=triangles, cad=points[triangles].copy(),
        faces=np.array(faces, dtype="<i8"), local=np.tile(np.array([0, 1], dtype="<i8"), len(faces)//2),
        cad_samples=interior)


def _policy(modules):
    return modules.initialize._v1.WholePartRegistrationPolicyV1(coarse_points=36,
        refine_points=36, coarse_iterations=1, refine_iterations=1)


def _configuration(modules, **overrides):
    init = modules.initialize
    return init._configuration(_policy(modules), init._noisy.NoisyMeshPlanePolicyV5(),
        init._noisy.NoisyMeshPlaneBudgetsV5(), init._patch.CoplanarPatchPolicyV4(),
        init._patch.CoplanarPatchBudgetsV4(), init._search.PlaneSearchPolicyV6(),
        init._search.PlaneSearchBudgetsV6(), init.PlaneInitializationBudgetsV12(), None,
        two_plane_enabled=False, **overrides)


def _call(modules, api, mesh, **overrides):
    init = modules.initialize
    args = (mesh["points"], mesh["triangles"], mesh["cad"], mesh["faces"], mesh["local"])
    kwargs = dict(position_error_bounds_mm=.05, registration_policy=_policy(modules),
                  unit_assumption=UNIT, two_plane_enabled=False)
    kwargs.update(overrides)
    if api == "prefilter": return init.prefilter_plane_initializations_v12(*args, **kwargs)
    assert api == "register"
    return init.register_whole_part_planes_v12(*args, cad_initialization_points=mesh["cad_samples"], **kwargs)


def _sha(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _origin_record(patches, origin):
    lane, kind = origin["lane"], origin["kind"]
    table = ("cores" if kind == "CORE" else "closures") if lane == "v7" else "regions"
    return patches["lanes"][lane][table][origin["record_id"]]


@pytest.mark.parametrize("api", ["prefilter", "register"])
def test_default_and_explicit_v7_preserve_exact_result_without_composite_dispatch(modules, monkeypatch, api):
    mesh, calls = _mesh(registration=api == "register"), []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("DEFAULT_V7_MUST_NOT_DISPATCH_COMPOSITE_OR_REFINEMENT")
    monkeypatch.setattr(modules.composite, "prepare_composite_scan_planes_v1", forbidden)
    monkeypatch.setattr(modules.initialize._v3, "_refine", forbidden)
    default = _call(modules, api, mesh)
    explicit = _call(modules, api, mesh, scan_plane_mode="v7")
    assert default == explicit and calls == []
    initialization = default if api == "prefilter" else default["evidence"]["initialization"]
    assert initialization["status"] == "COMPLETE"
    assert initialization["hypotheses"] == initialization["start_groups"] == []
    assert initialization["hypothesis_evidence"]["selected_policy_evaluation_complete"] is True
    decoded = modules.previous._compact.decode_metadata_v7(initialization["patch_evidence"]["scan"])
    assert decoded["schema_version"] == "noisy_mesh_planes_v7"
    assert "scan_plane_configuration" not in initialization
    if api == "register":
        assert default["status"] == "UNAVAILABLE" and default["transform_scan_to_cad"] is None
        assert default["reasons"] == ["NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"]
        assert "scan_plane_configuration" not in default["evidence"]


@pytest.mark.parametrize("corner", [False, True])
def test_actual_composite_preparation_projects_every_original_origin_into_real_search(modules, monkeypatch, corner):
    mesh, held, searched = _mesh(corner=corner), [], []
    prepare, search = modules.initialize._prepare_triplets, modules.previous._search.propose_plane_search_v6
    def retain(*args, **kwargs):
        context = prepare(*args, **kwargs)
        held.append(context)
        return context
    def record_search(observed, nominal, **kwargs):
        searched.append((observed, nominal))
        return search(observed, nominal, **kwargs)
    monkeypatch.setattr(modules.initialize, "_prepare_triplets", retain)
    monkeypatch.setattr(modules.previous._search, "propose_plane_search_v6", record_search)
    result = _call(modules, "prefilter", mesh, scan_plane_mode="composite_v1")
    assert result["status"] == "COMPLETE" and result["start_groups"] == [], result
    assert len(held) == len(searched) == 1
    context, observed = held[0], searched[0][0]
    patches = context["scan_patches"]
    assert patches["schema_version"] == "composite_scan_planes_v1"
    assert patches["lanes"]["v7"]["schema_version"] == "noisy_mesh_planes_v7"
    assert patches["lanes"]["seeded_v1"]["schema_version"] == "seeded_plane_proposals_v1"
    expected = [(lane, ref["kind"], ref["record_id"]) for lane in ("v7", "seeded_v1")
                for ref in patches["lanes"][lane]["proposals"]]
    assert [(row["lane"], row["kind"], row["record_id"]) for row in patches["planes"]] == expected
    assert len(expected) == len(context["scan_supports"]) == len(observed)
    assert len(expected) == (2 if corner else 4)
    assert any(row["lane"] == "seeded_v1" for row in patches["planes"])
    if corner: assert patches["lanes"]["v7"]["proposals"] == []
    for index, origin in enumerate(patches["planes"]):
        record, support, projected = _origin_record(patches, origin), context["scan_supports"][index], observed[index]
        analysis = record["analysis"]
        assert origin["plane_id"] == support["plane_id"] == support["source_record_id"] == index
        assert support["proposal_kind"] == origin["lane"]+":"+origin["kind"]
        assert support["triangle_rows"] is record["triangle_rows"]
        assert support["raw_vertex_rows"] is record["raw_vertex_rows"]
        assert np.array_equal(record["raw_vertex_rows"], np.unique(mesh["triangles"][record["triangle_rows"]]))
        modules.previous._bound_scan(context, support)
        assert projected.equation.plane_id == index
        assert projected.equation.normal == tuple(analysis["normal"])
        assert projected.equation.anchor_mm == tuple(analysis["witness_anchor_mm"])
        assert projected.equation.offset_from_anchor_mm == analysis["witness_offset_from_anchor_mm"]
        assert projected.equation.normal_angle_bound_radians == analysis["normal_angle_bound_radians"]
        assert projected.support_count == len(record["raw_vertex_rows"])
        assert projected.proposal_kind == (origin["kind"] if origin["lane"] == "v7" else "DECLARED")
    assert result["scan_plane_configuration"] == dict(mode="composite_v1",
        seeded_budgets=asdict(modules.composite._seeded.SeededPlaneBudgetsV1()),
        composite_budgets=asdict(modules.composite.CompositeScanPlaneBudgetsV1()))
    assert result["hypothesis_evidence"]["raw_support_provenance_verified"] is False
    assert result["hypothesis_evidence"]["accepted_registration"] is False


@pytest.mark.parametrize("fault", ["plane_id_bool", "wrong_lane_kind", "stale_hash", "incomplete", "missing_last_origin"])
def test_malformed_returned_composite_origin_cannot_reach_plane_search(modules, monkeypatch, fault):
    producer, calls = modules.composite.prepare_composite_scan_planes_v1, []
    def damaged(*args, **kwargs):
        output = producer(*args, **kwargs)
        assert output["status"] == "COMPLETE" and output["planes"]
        if fault == "plane_id_bool": output["planes"][0]["plane_id"] = False
        elif fault == "wrong_lane_kind": output["planes"][-1]["kind"] = "CORE"
        elif fault == "stale_hash": output["planes"][-1]["raw_vertex_rows_sha256"] = "f"*64
        elif fault == "incomplete": output["statistics"]["selected_policy_evaluation_complete"] = False
        elif fault == "missing_last_origin": output["planes"].pop()
        return output
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("MALFORMED_COMPOSITE_MUST_NOT_REACH_SEARCH")
    monkeypatch.setattr(modules.composite, "prepare_composite_scan_planes_v1", damaged)
    monkeypatch.setattr(modules.previous._search, "propose_plane_search_v6", forbidden)
    result = _call(modules, "prefilter", _mesh(), scan_plane_mode="composite_v1")
    assert result["status"] == "UNAVAILABLE" and calls == []
    assert result["hypotheses"] == result["start_groups"] == []
    assert result["selected_policy_evaluation_complete"] is False
    assert any("COMPOSITE" in reason for reason in result["reasons"])


@pytest.mark.parametrize("fault", ["exception", "refusal"])
def test_later_seeded_lane_failure_cannot_search_a_successful_v7_prefix(modules, monkeypatch, fault):
    seeded, completed_v7, searched = modules.composite._seeded.extract_seeded_plane_proposals_v1, [], []
    v7 = modules.composite._v7.extract_noisy_scan_plane_proposals_v7
    def first(*args, **kwargs):
        value = v7(*args, **kwargs)
        assert value["status"] == "COMPLETE" and value["proposals"]
        completed_v7.append(value)
        return value
    def later(*args, **kwargs):
        value = seeded(*args, **kwargs)
        assert value["status"] == "COMPLETE"
        if fault == "exception": raise ValueError("INVENTED_LATE_SEEDED_FAILURE")
        value.update(status="UNAVAILABLE", reasons=["INVENTED_LATE_SEEDED_REFUSAL"])
        return value
    def forbidden(*args, **kwargs):
        searched.append(True)
        raise AssertionError("NO_SEARCH_OF_SUCCESSFUL_LANE_PREFIX")
    monkeypatch.setattr(modules.composite._v7, "extract_noisy_scan_plane_proposals_v7", first)
    monkeypatch.setattr(modules.composite._seeded, "extract_seeded_plane_proposals_v1", later)
    monkeypatch.setattr(modules.previous._search, "propose_plane_search_v6", forbidden)
    result = _call(modules, "prefilter", _mesh(), scan_plane_mode="composite_v1")
    assert len(completed_v7) == 1 and searched == []
    assert result["status"] == "UNAVAILABLE" and result["hypotheses"] == result["start_groups"] == []
    patches = modules.previous._compact.decode_metadata_v7(result["patch_evidence"]["scan"])
    assert patches["status"] == "UNAVAILABLE" and patches["lanes"] == {} and patches["planes"] == []
    assert patches["statistics"]["selected_policy_evaluation_complete"] is False


@pytest.mark.parametrize("api,fault,reason", [
    ("prefilter", "bool_mode", "INVALID_SCAN_PLANE_MODE"),
    ("register", "none_mode", "INVALID_SCAN_PLANE_MODE"),
    ("prefilter", "unknown_mode", "INVALID_SCAN_PLANE_MODE"),
    ("register", "v7_seeded", "COMPOSITE_BUDGETS_REQUIRE_EXPLICIT_COMPOSITE_MODE"),
    ("prefilter", "v7_composite", "COMPOSITE_BUDGETS_REQUIRE_EXPLICIT_COMPOSITE_MODE"),
    ("register", "seeded_type", "INVALID_SEEDED_PLANE_BUDGET_TYPE"),
    ("prefilter", "composite_type", "INVALID_COMPOSITE_BUDGET_TYPE"),
    ("register", "seeded_above_maximum", "INVALID_SEEDED_PLANE_BUDGET"),
    ("prefilter", "composite_boolean_cap", "INVALID_COMPOSITE_BUDGET")])
def test_strict_explicit_mode_and_budget_admission_precedes_all_preparation(modules, monkeypatch, api, fault, reason):
    seeded, combined = modules.composite._seeded.SeededPlaneBudgetsV1, modules.composite.CompositeScanPlaneBudgetsV1
    options = dict(scan_plane_mode="composite_v1")
    if fault == "bool_mode": options["scan_plane_mode"] = True
    elif fault == "none_mode": options["scan_plane_mode"] = None
    elif fault == "unknown_mode": options["scan_plane_mode"] = "composite"
    elif fault == "v7_seeded": options.update(scan_plane_mode="v7", seeded_plane_budgets=seeded())
    elif fault == "v7_composite": options.update(scan_plane_mode="v7", composite_plane_budgets=combined())
    elif fault == "seeded_type": options["seeded_plane_budgets"] = {}
    elif fault == "composite_type": options["composite_plane_budgets"] = {}
    elif fault == "seeded_above_maximum": options["seeded_plane_budgets"] = seeded(max_seeds=33)
    elif fault == "composite_boolean_cap": options["composite_plane_budgets"] = combined(max_qualified_planes=True)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("CONFIGURATION_MUST_REFUSE_BEFORE_PREPARATION")
    monkeypatch.setattr(modules.initialize, "_prepare_triplets", forbidden)
    result = _call(modules, api, _mesh(registration=api == "register"), **options)
    assert result["status"] == "UNAVAILABLE" and result["reasons"] == [reason] and calls == []


def test_configuration_owns_budget_snapshots_before_caller_mutation(modules):
    seeded = modules.composite._seeded.SeededPlaneBudgetsV1(max_seeds=2)
    combined = modules.composite.CompositeScanPlaneBudgetsV1(max_qualified_planes=16)
    expected = dict(mode="composite_v1", seeded_budgets=asdict(seeded), composite_budgets=asdict(combined))
    configuration = _configuration(modules, scan_plane_mode="composite_v1",
        seeded_plane_budgets=seeded, composite_plane_budgets=combined)
    assert configuration["seeded_plane_budgets"] is not seeded
    assert configuration["composite_plane_budgets"] is not combined
    object.__setattr__(seeded, "max_seeds", 0)
    object.__setattr__(combined, "max_qualified_planes", 0)
    mesh = _mesh()
    result, context = modules.initialize._initialize(mesh["points"], mesh["triangles"], mesh["cad"],
        mesh["faces"], mesh["local"], .05, configuration, UNIT)
    assert result["status"] == "COMPLETE" and context is not None, result
    assert result["scan_plane_configuration"] == expected
    assert context["scan_patches"]["requested_lane_budgets"]["seeded_v1"]["max_seeds"] == 2
    assert len(context["scan_patches"]["planes"]) == 4


def test_initializer_snapshot_prevents_caller_mutation_from_splitting_lane_binding(modules, monkeypatch):
    mesh, producer, captured = _mesh(), modules.composite.prepare_composite_scan_planes_v1, []
    points, triangles = mesh["points"].copy(), mesh["triangles"].copy()
    def mutate_caller_then_prepare(p, t, **kwargs):
        assert p is not mesh["points"] and t is not mesh["triangles"]
        mesh["points"][:] = np.nan
        mesh["triangles"][:] = -1
        value = producer(p, t, **kwargs)
        captured.append(value)
        return value
    monkeypatch.setattr(modules.composite, "prepare_composite_scan_planes_v1", mutate_caller_then_prepare)
    result = _call(modules, "prefilter", mesh, scan_plane_mode="composite_v1")
    assert result["status"] == "COMPLETE" and len(captured) == 1, result
    binding = dict(points_sha256=_sha(points, "<f8"), triangle_raw_rows_sha256=_sha(triangles, "<i8"),
        position_error_bounds_mm_sha256=_sha(np.full(len(points), .05), "<f8"))
    assert captured[0]["input_digests"] == binding
    assert all(lane["input_digests"] == binding for lane in captured[0]["lanes"].values())
    assert result["input_digests"]["scan_points_sha256"] == binding["points_sha256"]
    assert result["input_digests"]["scan_triangle_raw_rows_sha256"] == binding["triangle_raw_rows_sha256"]


@pytest.mark.parametrize("mode", ["v7", "composite_v1"])
def test_actual_single_role_preparation_dispatch_with_explicit_refinement_refusal(modules, monkeypatch, mode):
    init, mesh, held, calls = modules.initialize, _mesh(registration=True), [], []
    prefilter = init._prefilter
    def invented_start(context, result, configuration):
        prefilter(context, result, configuration)
        assert result["status"] == "COMPLETE" and result["start_groups"] == []
        # This is NOT a finite-distance or registration positive. It only opens
        # the downstream role-admission seam; refinement is refused below.
        result["start_groups"] = [dict(index=0, exact_group_index=0, hypothesis_indices=[],
            initialized_rotation=np.eye(3).tolist(), initialized_translation_mm=[0., 0., 0.],
            fixture="INVENTED_START_FOR_ROLE_DISPATCH_NOT_FINITE_SUPPORT_EVIDENCE")]
    role_module = modules.composite_roles if mode == "composite_v1" else modules.roles
    role_name = "prepare_composite_surface_roles_v1" if mode == "composite_v1" else "prepare_context_surface_roles_v1"
    real_roles = getattr(role_module, role_name)
    def actual_roles(*args, **kwargs):
        value = real_roles(*args, **kwargs)
        held.append((args, value))
        return value
    def refused_refinement(*args, **kwargs):
        calls.append(True)
        raise ValueError("EXPLICIT_TEST_REFINEMENT_REFUSAL_NO_REGISTRATION_POSITIVE")
    monkeypatch.setattr(init, "_prefilter", invented_start)
    monkeypatch.setattr(role_module, role_name, actual_roles)
    monkeypatch.setattr(init._v3, "_refine", refused_refinement)
    result = _call(modules, "register", mesh, scan_plane_mode=mode, connected_measurement_contexts=True)
    assert len(held) == len(calls) == 1, result
    args, prepared = held[0]
    assert type(prepared) is modules.roles.PreparedContextSurfaceRolesV1
    assert prepared.raw_row_count == len(mesh["points"])
    assert prepared.points_sha256 == _sha(mesh["points"], "<f8")
    assert prepared.triangle_raw_rows_sha256 == _sha(mesh["triangles"], "<i8")
    assert result["evidence"]["surface_role_preparation"]["metadata_sha256"] == prepared.metadata_sha256
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    assert result["evidence"]["candidate_numerical_failure_count"] == 1
    assert result["evidence"].get("measurement_context_indices", []) == []
    if mode == "composite_v1":
        origins = args[2]["planes"]
        assert [(row.kind, row.record_id) for row in prepared._supports] == [
            (row["lane"]+":"+row["kind"], row["plane_id"]) for row in origins]
        assert result["evidence"]["scan_plane_configuration"] == result["evidence"]["initialization"]["scan_plane_configuration"]
    else:
        assert all(row.kind in ("CORE", "CLOSURE") for row in prepared._supports)
        assert "scan_plane_configuration" not in result["evidence"]


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
            registration_policy=_policy(modules), measurement_policy=modules.native.NativeGuidedPolicy(fit_model="free_circle"),
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
