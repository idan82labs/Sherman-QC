"""Independent actual-V13-adapter orchestration with explicit boundary mocks.

Byte decoders, native import/extraction/sampling and registration search are
invented stubs. Immutable array/source binding and EVERY V12 context-binding
call remain real. Most measurement/validator calls are separate contract stubs;
the invalid-role tests use the actual V2->V1 refusal path, stopped before role
selection or geometry. No native/physical input, recovered pose, fit accuracy,
event count or end-to-end geometry qualification is claimed.
"""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCAN_BYTES = b"INVENTED-V13-SCAN-BYTES-NOT-A-PHYSICAL-FILE\x00\xff"
CAD_BYTES = b"INVENTED-V13-CAD-BYTES-NOT-A-STEP-FILE\x00\xff"
BOX_POLICY = "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
DISTANCE = "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
SUPPORT = "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
SURROGATE = "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
CONSUMER = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"


def _owned(value, dtype="<f8"):
    value = np.asarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape)


def _sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.tobytes()).hexdigest()


@pytest.fixture
def modules():
    name = "isolated_mesh_offline_v13_dispatch"
    def owned(key):
        return key in (name, "backend") or key.startswith(name+".") or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved: del sys.modules[key]
    for prefix in (name, "backend"):
        package = types.ModuleType(prefix)
        package.__path__ = [str(ROOT/"backend")]
        sys.modules[prefix] = package
    mapping = dict(workflow="cad_guided_mesh_offline_v13", mesh="cad_guided_mesh_input_v1",
        topology="cad_guided_mesh_topology_v4", initialize="cad_guided_plane_initialization_v12",
        registration="cad_guided_global_registration_v1", native="cad_guided_native_features",
        native_v1="cad_guided_native_features_v1", surface="cad_guided_cad_surface_v2",
        cover="cad_guided_support_cover_measurement_v2")
    try:
        yield types.SimpleNamespace(**{key: importlib.import_module(name+"."+value)
            for key, value in mapping.items()})
    finally:
        for key in tuple(sys.modules):
            if owned(key): del sys.modules[key]
        sys.modules.update(saved)


def _arrays():
    rows = np.arange(72)
    raw = _owned(np.column_stack((rows % 6, rows//6 % 4, rows//24)))
    selection = _owned(np.arange(0, 72, 2), "<i8")
    engine = _owned(raw[selection])
    triangles = _owned([[0, 1, 6], [6, 7, 12]], "<i8")
    scan = dict(status="AVAILABLE", reasons=[], raw_points=raw, points=engine,
        engine_to_raw_row_indices=selection, input_admission=dict(face_count=2, face_index_count=6),
        coverage=dict(raw_vertex_count=72, selected_engine_rows=36, scope="MOCK_LOADER_RECEIPT"),
        measurement_cloud_contract="FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
        sample_cloud_contract="REGISTRATION_AND_PROPOSALS_ONLY", units_status="UNVERIFIED",
        coordinate_frame_status="UNVERIFIED", source_bindings=dict(scan_sha256=_sha(SCAN_BYTES),
            raw_xyz_sha256=_sha(raw), sample_xyz_sha256=_sha(engine), engine_to_raw_sha256=_sha(selection)))
    topology = dict(status="AVAILABLE", reasons=[], points=raw, triangle_raw_rows=triangles,
        source_sha256=_sha(SCAN_BYTES), points_sha256=_sha(raw), triangle_raw_rows_sha256=_sha(triangles),
        vertex_count=72, triangle_count=2, raw_row_order="ORIGINAL_PLY_VERTEX_ROWS",
        triangle_row_order="ORIGINAL_PLY_FACE_ROWS", topology_status="INVENTED_NOT_MATERIAL_TOPOLOGY")
    # Hand-written exact dyadic affine samples. No sampler/mesher is run to
    # construct this fixture; actual _surface_binding checks their incidence.
    tri = _owned([[[0, 0, 0], [8, 0, 0], [0, 4, 0]],
                  [[0, 0, 2], [8, 0, 2], [0, 4, 2]],
                  [[3, 0, 0], [3, 8, 0], [3, 0, 4]]])
    faces, ids = _owned([2, 5, 9], "<i8"), _owned([1, 1, 1], "<i8")
    sample_rows = _owned(np.arange(36) % 3, "<i8")
    points = _owned(np.array([[2., 1., 0.], [2., 1., 2.], [3., 2., 1.]])[sample_rows])
    normals = _owned(np.array([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]])[sample_rows])
    weights = _owned(np.tile([.5, .25, .25], (36, 1)))
    policy = dict(sample_count=36, fixture="EXACT_DYADIC_SAMPLES_NOT_NATIVE_EXPORT")
    cad = dict(status="AVAILABLE", reasons=[], triangles=tri, source_face_indices=faces,
        source_triangle_indices=ids, points=points, normals=normals, sample_triangle_rows=sample_rows,
        sample_source_face_indices=_owned(faces[sample_rows], "<i8"),
        sample_source_triangle_indices=_owned(ids[sample_rows], "<i8"),
        sample_barycentric_weights=weights, sampling_policy=policy,
        digests=dict(oriented_triangle_xyz_sha256=_sha(tri), source_face_ids_sha256=_sha(faces),
            source_triangle_ids_sha256=_sha(ids), sample_points_sha256=_sha(points),
            sample_normals_sha256=_sha(normals), sample_triangle_rows_sha256=_sha(sample_rows),
            sampling_policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()))
    return scan, topology, cad


def _registration(modules, scan, topology, cad):
    stages = modules.workflow._stages
    digests = dict(scan_points_sha256=_sha(scan["raw_points"]),
        scan_triangle_raw_rows_sha256=_sha(topology["triangle_raw_rows"]),
        position_error_bounds_mm_sha256=_sha(np.full(72, .05, dtype="<f8")),
        cad_triangles_sha256=_sha(cad["triangles"]), cad_face_ids_sha256=_sha(cad["source_face_indices"]),
        cad_source_triangle_ids_sha256=_sha(cad["source_triangle_indices"]))
    hypotheses, groups, candidates = [], [], []
    for index in range(2):
        r, t = np.eye(3).tolist(), [float(index), 2.*index, 3.*index]
        correspondences = [dict(scan_plane_id=i, cad_plane_id=face, normal_sign=1)
                           for i, face in enumerate((2, 5, 9))]
        hypotheses.append(dict(hypothesis_index=index, proposal_lane="TRIPLET", status="PROPOSED",
            prefilter_status="PASSED", exact_start_group_index=index,
            correspondences=copy.deepcopy(correspondences),
            assigned_support=[dict(row, status="PASSED") for row in correspondences],
            rotation=r, translation_mm=t))
        group = dict(index=index, exact_group_index=index, hypothesis_indices=[index],
            initialized_rotation=copy.deepcopy(r), initialized_translation_mm=list(t),
            refinement_disposition="SCHEDULED_FOR_V3_REFINEMENT")
        groups.append(group)
        candidates.append(dict(copy.deepcopy(group), rotation=copy.deepcopy(r), translation=list(t),
            reasons=["MOCK_LEGACY_RANK_REFUSAL_NOT_CONTEXT_DISTANCE_REFUSAL"], execution_status="ATTEMPTED",
            numerical_consumer_status=CONSUMER, numerical_support_policy=SUPPORT,
            distance_model=DISTANCE, optimizer_objective=SURROGATE,
            conditional_context_support=dict(status="SUPPORTED_CONTEXT", accepted_registration=False,
                proximity_policy=stages.PROXIMITY,
                pose_sha256=_sha(np.r_[np.array(r).ravel(), t].astype("<f8")))))
    initialization = dict(schema_version="noisy_plane_initialization_prefilter_v12", status="COMPLETE",
        search_complete=True, selected_policy_evaluation_complete=True, search_exhaustive=False,
        numerical_consumer_complete=True, box_screening_complete=True, box_screening_policy=BOX_POLICY,
        numerical_support_policy=stages.PROXIMITY, input_digests=copy.deepcopy(digests),
        hypotheses=hypotheses, start_groups=copy.deepcopy(groups), all_passing_start_groups=copy.deepcopy(groups),
        joint_proposal_accounting=dict(pair_lane_enabled=True, triplet_records=2,
            triplet_unsigned_reservation=2, ordered_pair_matches=0, pair_slot_reservation=0,
            total_hypothesis_records=2),
        two_plane_schedule=dict(status="COMPLETE", selected_policy_evaluation_complete=True, scheduled_matches=[]))
    evidence = dict(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12",
        initialization=initialization, candidate_evaluation_complete=True, numerical_consumer_complete=True,
        connected_measurement_contexts=True, accepted_registration=False, numerical_support_policy=SUPPORT,
        distance_model=DISTANCE, optimizer_objective=SURROGATE,
        stage_policies=stages.ablation_stage_policies_v1(connected=True), input_digests=digests,
        surface_role_preparation=dict(metadata_sha256="d"*64), candidates=candidates,
        measurement_context_indices=[0, 1], prefilter_full_scan_raw_rows=72, registration_input_rows=36,
        registration_input_points_sha256=_sha(scan["points"]),
        registration_raw_rows=dict(count=36, selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING",
                                   sha256=_sha(scan["engine_to_raw_row_indices"])),
        cad_initialization_points_sha256=_sha(np.unique(cad["points"], axis=0)))
    return dict(schema_version="noisy_plane_initialized_finite_registration_v12",
        status="CONDITIONAL_MEASUREMENT_CONTEXTS", reasons=[], transform_scan_to_cad=None, evidence=evidence)


@pytest.fixture
def plumbing(modules, monkeypatch):
    workflow, native, cover = modules.workflow, modules.native, modules.cover
    scan, topology, cad = _arrays()
    feature = native.CADGuidedFeature(feature_id="invented-native-feature", nominal_radius_mm=5.,
        nominal_sweep_deg=90., origin=(0., 0., 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.),
        span_mm=(0., 12.), flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.05,
        plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle",
        nominal_source="INVENTED_NATIVE_EXTRACTION_SEAM_NOT_NATIVE_EXECUTION")
    state = types.SimpleNamespace(modules=modules, scan=scan, topology=topology, cad=cad,
        feature=feature, shape=object(), owned_roles=object(), events=[], bindings=[],
        measurement_calls=[], validator_calls=[], register_calls=[], damaged=None,
        real_cover=cover.inspect_support_cover_context_windows_v2,
        real_validator=cover.validate_support_cover_publication_v2)
    state.registration = _registration(modules, scan, topology, cad)
    state.measured = dict(schema="CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2", status="PARTIAL",
        computation_status="COMPLETE", reasons=[], predictions=[{"fixture_cover_carrier": "only-top-level"}],
        support_groups=[{"fixture": "MOCK_MEASUREMENT_NOT_GEOMETRIC_OUTPUT"}],
        source_context_result=dict(predictions=[{"nested_member_carrier": "must-not-leak"}],
            support_records=[{"raw_support": {"arc_rows": [1, 5, 69]}, "fixture": "EVIDENCE_ONLY"}]),
        observed_bend_count=None, production_use_allowed=False)
    state.validation = dict(cover_plan_revalidated=True, fixture="MOCK_PUBLICATION_CONTRACT_NOT_PROOF")
    def prepare(**kwargs):
        state.events.append("prepare")
        assert kwargs == dict(scan_bytes=SCAN_BYTES, max_engine_rows=36)
        return state.scan
    def parse(blob):
        state.events.append("parse_topology")
        assert blob is SCAN_BYTES
        return state.topology
    def read(blob):
        state.events.append("native_read")
        assert blob is CAD_BYTES
        return state.shape, dict(target_unit="mm", physical_calibration="NOT_ESTABLISHED")
    def extract(shape, **kwargs):
        state.events.append("native_extract_v1")
        assert shape is state.shape and kwargs["allow_ocp"] is True
        return dict(status="PARTIAL", features=(state.feature,), metadata={"status": "PARTIAL", "fixture": "FAKE_NATIVE"})
    def sample(shape, **kwargs):
        state.events.append("native_sample")
        assert shape is state.shape and kwargs["sample_count"] == 36 and kwargs["allow_ocp"] is True
        return state.cad
    def register(*args, **kwargs):
        state.events.append("register_contexts")
        state.register_calls.append((args, kwargs))
        if state.damaged is not None: state.damaged(state.registration)
        return state.registration, state.owned_roles
    original_binding = workflow._registration_binding
    def binding(*args, **kwargs):
        state.events.append("bind:"+str(kwargs["context_index"]))
        state.bindings.append((args, kwargs))
        return original_binding(*args, **kwargs)
    def measure(raw, features, **kwargs):
        state.events.append("measure_cover")
        state.measurement_calls.append((raw, features, kwargs))
        assert raw is state.scan["raw_points"] and features == (state.feature,)
        assert kwargs["registration"] is state.registration and kwargs["surface_roles"] is state.owned_roles
        assert kwargs["scan_sha256"] == _sha(SCAN_BYTES)
        return state.measured
    def validate(result, **kwargs):
        state.events.append("validate_cover")
        state.validator_calls.append((result, kwargs))
        assert result is state.measured and kwargs["raw_points"] is state.scan["raw_points"]
        assert kwargs["scan_sha256"] == _sha(SCAN_BYTES)
        return state.validation
    monkeypatch.setattr(modules.mesh, "prepare_mesh_scan_v1", prepare)
    monkeypatch.setattr(modules.topology, "parse_triangle_mesh_ply_v4", parse)
    monkeypatch.setattr(workflow._v2, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(modules.native_v1, "extract_ocp_guided_features_v1", extract)
    monkeypatch.setattr(modules.surface, "sample_native_cad_surface_v2", sample)
    monkeypatch.setattr(modules.initialize, "register_measurement_contexts_v12", register)
    monkeypatch.setattr(workflow, "_registration_binding", binding)
    monkeypatch.setattr(cover, "inspect_support_cover_context_windows_v2", measure)
    monkeypatch.setattr(cover, "validate_support_cover_publication_v2", validate)
    def run(**overrides):
        args = dict(scan_bytes=SCAN_BYTES, cad_bytes=CAD_BYTES, position_error_bound_mm=.05, allow_ocp=True,
            workflow_policy=workflow.MeshWorkflowPolicyV13(max_registration_scan_rows=36, cad_sample_count=36),
            registration_policy=modules.registration.WholePartRegistrationPolicyV1(coarse_points=36,
                refine_points=36, coarse_iterations=1, refine_iterations=1),
            measurement_policy=native.NativeGuidedPolicy(flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
                max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
            instance_policy=cover._v1.SupportPolicy(4., 5., 20., 3))
        args.update(overrides)
        return workflow.inspect_cad_guided_mesh_v13(**args)
    state.run = run
    return state


def test_actual_adapter_binds_every_context_and_passes_exact_full_raw_role_object(plumbing):
    state = plumbing
    result = state.run()
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE", result
    assert state.events == ["prepare", "parse_topology", "native_read", "native_extract_v1",
        "native_sample", "register_contexts", "bind:0", "bind:1", "measure_cover", "validate_cover"]
    assert [row[1]["context_index"] for row in state.bindings] == [0, 1]
    assert len(result["registration_array_binding"]) == 2
    for args, kwargs in state.bindings:
        assert args[0] is state.registration and args[1] is state.scan and args[2] is state.topology
        assert args[3] is state.cad and args[4] is state.scan["engine_to_raw_row_indices"] and args[5] == .05
    args, kwargs = state.register_calls[0]
    assert args[0] is state.scan["raw_points"] and len(args[0]) == 72
    assert args[1] is state.topology["triangle_raw_rows"]
    assert args[2] is state.cad["triangles"] and args[3] is state.cad["source_face_indices"]
    assert args[4] is state.cad["source_triangle_indices"]
    assert kwargs["cad_initialization_points"] is state.cad["points"]
    assert kwargs["registration_raw_rows"] is state.scan["engine_to_raw_row_indices"]
    assert kwargs["position_error_bounds_mm"] == .05 and kwargs["connected_measurement_contexts"] is True
    raw, _, measured_kwargs = state.measurement_calls[0]
    assert raw is state.scan["raw_points"] and raw is not state.scan["points"]
    assert measured_kwargs["surface_roles"] is state.owned_roles
    assert "position_error_bounds_mm" not in measured_kwargs
    assert result["measurements"] is state.measured
    assert result["predictions"] is state.measured["predictions"]
    assert result["predictions"] != state.measured["source_context_result"]["predictions"]
    assert result["support_cover_publication_validation"] is state.validation
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    assert result["registration"]["transform_scan_to_cad"] is None
    assert result["measurement_geometry_frame"] == result["prediction_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"


@pytest.mark.parametrize("fault", ["missing_role", "role_id", "role_boolean", "refused_role",
    "role_sign", "pose", "candidate_type", "context_pose", "origin_index"])
def test_real_second_context_binding_failure_clears_all_output_before_measurement(plumbing, fault):
    state = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        origin = evidence["initialization"]["hypotheses"][1]
        candidate = evidence["candidates"][1]
        if fault == "missing_role": origin["assigned_support"].pop()
        elif fault == "role_id": origin["assigned_support"][0]["cad_plane_id"] = 99
        elif fault == "role_boolean": origin["assigned_support"][0]["scan_plane_id"] = False
        elif fault == "refused_role": origin["assigned_support"][0]["status"] = "REJECTED"
        elif fault == "role_sign": origin["assigned_support"][0]["normal_sign"] = -1
        elif fault == "pose": origin["translation_mm"][0] += 1.
        elif fault == "candidate_type": candidate["index"] = True
        elif fault == "context_pose": candidate["rotation"] = [[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]
        elif fault == "origin_index": candidate["hypothesis_indices"] = [0]
    state.damaged = damage
    result = state.run()
    assert [row[1]["context_index"] for row in state.bindings] == [0, 1]
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION"
    assert result["measurements"] is None and result["predictions"] == []
    assert "registration_array_binding" not in result, "A successful binding prefix cannot be published"
    assert result["registration"]["status"] == "UNAVAILABLE" and result["registration"]["transform_scan_to_cad"] is None
    assert state.measurement_calls == state.validator_calls == []


@pytest.mark.parametrize("fault", ["indices_missing", "indices_empty", "indices_duplicate", "indices_bool",
    "pair_lane", "scan_digest", "triangle_digest", "epsilon_digest", "sample_digest"])
def test_incomplete_context_or_global_array_bindings_never_measure(plumbing, fault):
    state = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        if fault == "indices_missing": evidence.pop("measurement_context_indices")
        elif fault == "indices_empty": evidence["measurement_context_indices"] = []
        elif fault == "indices_duplicate": evidence["measurement_context_indices"] = [0, 0]
        elif fault == "indices_bool": evidence["measurement_context_indices"] = [False, 1]
        elif fault == "pair_lane": evidence["initialization"]["joint_proposal_accounting"]["pair_lane_enabled"] = False
        elif fault == "scan_digest": evidence["input_digests"]["scan_points_sha256"] = "f"*64
        elif fault == "triangle_digest": evidence["input_digests"]["scan_triangle_raw_rows_sha256"] = "f"*64
        elif fault == "epsilon_digest": evidence["input_digests"]["position_error_bounds_mm_sha256"] = "f"*64
        elif fault == "sample_digest": evidence["cad_initialization_points_sha256"] = "f"*64
    state.damaged = damage
    result = state.run()
    assert result["status"] == "UNAVAILABLE" and result["measurements"] is None and result["predictions"] == []
    assert state.measurement_calls == state.validator_calls == []


@pytest.mark.parametrize("fault", ["scan_sha", "raw_sha", "triangle_sha", "row_order", "sample_sha", "sample_membership"])
def test_actual_immutable_byte_and_triangle_bindings_reject_mock_export_drift(plumbing, fault):
    state = plumbing
    if fault == "scan_sha": state.scan["source_bindings"]["scan_sha256"] = "f"*64
    elif fault == "raw_sha": state.scan["source_bindings"]["raw_xyz_sha256"] = "f"*64
    elif fault == "triangle_sha": state.topology["triangle_raw_rows_sha256"] = "f"*64
    elif fault == "row_order": state.topology["raw_row_order"] = "SORTED_ROWS"
    elif fault == "sample_sha": state.cad["digests"]["sample_points_sha256"] = "f"*64
    elif fault == "sample_membership": state.cad["sample_source_face_indices"] = _owned(np.zeros(36), "<i8")
    result = state.run()
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert state.register_calls == state.measurement_calls == state.validator_calls == []


@pytest.mark.parametrize("fault", ["none", "wrong_type", "points", "triangles", "epsilon", "metadata", "count"])
def test_actual_v2_v1_owned_role_admission_refuses_before_any_geometry(plumbing, monkeypatch, fault):
    state = plumbing
    cover = state.modules.cover
    roles = cover._v1._roles
    evidence = state.registration["evidence"]
    values = dict(points_sha256=evidence["input_digests"]["scan_points_sha256"],
        triangle_raw_rows_sha256=evidence["input_digests"]["scan_triangle_raw_rows_sha256"],
        position_error_bounds_mm_sha256=evidence["input_digests"]["position_error_bounds_mm_sha256"],
        metadata_sha256="d"*64, raw_row_count=72, triangle_count=2, triangle_membership_count=0,
        raw_membership_count=0, encoded_metadata_bytes=0, decoded_metadata_bytes=0,
        unscheduled_qualified_core_count=0, _triangles=state.topology["triangle_raw_rows"],
        _referenced=_owned(np.ones(72), "?"), _supports=(),
        _budgets=roles.ContextSurfaceRoleBudgetsV1(), _token=object())
    # This deliberately INVALID token object is never used as a valid prepared
    # context: every case must fail a preceding type/digest/count admission.
    if fault == "none": state.owned_roles = None
    elif fault == "wrong_type": state.owned_roles = object()
    else:
        key = dict(points="points_sha256", triangles="triangle_raw_rows_sha256",
                   epsilon="position_error_bounds_mm_sha256", metadata="metadata_sha256", count="raw_row_count")[fault]
        values[key] = 71 if fault == "count" else "f"*64
        state.owned_roles = roles.PreparedContextSurfaceRolesV1(**values)
    downstream = []
    def forbidden(*args, **kwargs):
        downstream.append(True)
        raise AssertionError("ROLE_ADMISSION_MUST_PRECEDE_GEOMETRY")
    monkeypatch.setattr(roles, "select_context_measurement_rows_v1", forbidden)
    monkeypatch.setattr(cover._v1, "_canonical_fit", forbidden)
    monkeypatch.setattr(cover, "inspect_support_cover_context_windows_v2", state.real_cover)
    monkeypatch.setattr(cover, "validate_support_cover_publication_v2", state.real_validator)
    result = state.run()
    assert [row[1]["context_index"] for row in state.bindings] == [0, 1]
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "MEASUREMENT_REFUSAL", result
    assert result["predictions"] == [] and downstream == []
    measured = result["measurements"]
    assert measured["computation_status"] == "SOURCE_CONTEXT_REFUSED"
    assert "MISSING_OR_UNBOUND_OWNED_SURFACE_ROLES" in measured["source_context_result"]["reasons"][0]
    assert result["support_cover_publication_validation"]["source_refusal"] is True


def test_validator_exception_discards_returned_measurement_and_cover_predictions(plumbing, monkeypatch):
    state = plumbing
    def bad(*args, **kwargs):
        raise ValueError("INVENTED_PUBLICATION_VETO")
    monkeypatch.setattr(state.modules.cover, "validate_support_cover_publication_v2", bad)
    result = state.run()
    assert len(state.measurement_calls) == 1
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "FULL_RAW_LOCAL_MEASUREMENT"
    assert result["measurements"] is None and result["predictions"] == []
    assert "support_cover_publication_validation" not in result


@pytest.mark.parametrize("computation,status,phase", [
    ("COMPLETE", "UNAVAILABLE", "COMPLETE"),
    ("SOURCE_CONTEXT_REFUSED", "UNAVAILABLE", "MEASUREMENT_REFUSAL"),
    ("COMPLETE", "PARTIAL", "COMPLETE")])
def test_legal_refusal_or_quantity_without_carrier_retains_exact_measurement_ledger(plumbing, computation, status, phase):
    state = plumbing
    state.measured.update(status=status, computation_status=computation, predictions=[],
        reasons=["MOCK_LEGAL_REFUSAL"], support_groups=[{"fixture_quantity_without_carrier": 5.3}])
    result = state.run()
    assert result["status"] == status and result["phase"] == phase
    assert result["measurements"] is state.measured and result["predictions"] == []
    assert result["reasons"] == state.measured["reasons"]
    assert len(state.validator_calls) == 1
    assert result["observed_bend_count"] is None


def test_detailed_report_overflow_discards_entire_adapter_result(plumbing, monkeypatch):
    state = plumbing
    workflow = state.modules.workflow
    original = workflow._bounded_workflow
    monkeypatch.setattr(workflow, "_bounded_workflow", lambda output: original(output, maximum=128))
    result = state.run()
    assert len(state.measurement_calls) == len(state.validator_calls) == 1
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13" and result["phase"] == "REPORT_PUBLICATION"
    assert result["detailed_report_discarded_whole"] is True
    assert result["predictions"] == [] and result["measurements"] is None and result["registration"] is None
    assert result["detailed_report_limit_bytes"] == 128 and result["encoded_bytes_examined"] > 128


@pytest.mark.parametrize("mode", [False, None, 0, 1, "true", np.bool_(True)])
def test_connected_only_admission_is_strict_and_precedes_all_byte_calls(plumbing, mode):
    result = plumbing.run(connected_measurement_contexts=mode)
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "ADMISSION"
    assert result["reasons"] == ["V13_CONNECTED_CONTEXTS_REQUIRED"] and plumbing.events == []
