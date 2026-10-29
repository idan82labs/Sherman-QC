"""Prospective initializer-V13/workflow-V14 context-V3 closure, invented only.

No target import, collection or test is authorized by this source file. Root
must freeze exact sources, denominator, resource limits, protocol and PRE first.
The real numerical fixture contains50 scan rows/64 scan triangles and80 CAD
triangles, with two independently triangulated planar sheets. It exercises real
extraction, scheduling, context proposal and finite-face prefilter only: no
native CAD, registration refinement, local bend fit or physical input.
Byte-workflow checks explicitly mock native/registration/measurement boundaries
but leave the actual requested/returned configuration and every selected-context
binding live. Neither category is physical accuracy or end-to-end qualification.
"""
import copy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
INIT_ID = "CAD_GUIDED_PLANE_INITIALIZATION_V13_CONTEXT_V3"
CONTEXT_ID = "RAW_EXTREMA_CONTEXT_V3_STAGED_SCAN_PROJECTION"
WORKFLOW_ID = "CAD_GUIDED_MESH_OFFLINE_V14_CONTEXT_V3"
SCAN_BYTES = b"NEW-INVENTED-CONTEXT-V3-WORKFLOW-SCAN-NOT-A-FILE"
CAD_BYTES = b"NEW-INVENTED-CONTEXT-V3-WORKFLOW-CAD-NOT-STEP"
DECLARATIONS = {
    "scan_projection_arithmetic": "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1",
    "scan_projection_bound_scope": "ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
    "cad_projection_arithmetic": "UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
    "scan_projection_chunk_scope": "FIXED_EXTREMA_LEAF256_NOT_CHUNK_ROWS",
    "chunk_rows_scope": "INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
    "extrema_guard_status": "NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_INVARIANTS",
    "extrema_membership_scope": "SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN",
    "extrema_memory_scope": "SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS",
}
SLOTS = ("WHOLE_MIN", "WHOLE_MAX", "FACE0_MIN", "FACE0_MAX", "FACE1_MIN", "FACE1_MAX",
    "WHOLE_CONTEXT_MIDPOINT", "ASSIGNED_CONTEXT_INTERSECTION_MIDPOINT")


@pytest.fixture(scope="module")
def modules():
    # Some frozen helpers deliberately use absolute backend imports. Use one
    # inert backend graph so strict dataclass identities match relative imports.
    prefix = "backend"
    def selected(name):
        return name == prefix or name.startswith(prefix+".")
    saved = {name: value for name, value in sys.modules.items() if selected(name)}
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[prefix] = package
    mapping = dict(init="cad_guided_plane_initialization_v13", frozen="cad_guided_plane_initialization_v12",
        context="cad_guided_two_plane_context_v3", schedule2="cad_guided_two_plane_schedule_v2",
        workflow="cad_guided_mesh_offline_v14", mesh="cad_guided_mesh_input_v1",
        topology="cad_guided_mesh_topology_v4", native="cad_guided_native_features",
        native1="cad_guided_native_features_v1", cad_surface="cad_guided_cad_surface_v2",
        cover="cad_guided_support_cover_measurement_v2")
    try:
        yield types.SimpleNamespace(**{name: importlib.import_module(prefix+"."+path)
            for name, path in mapping.items()})
    finally:
        for name in tuple(sys.modules):
            if selected(name):
                del sys.modules[name]
        sys.modules.update(saved)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.tobytes(order="C")).hexdigest()


def _owned(value, dtype="<f8"):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _expected_configuration(m, budget=None):
    cap = m.context.TwoPlaneContextBudgetsV3() if budget is None else budget
    return dict(mode="v3", implementation=CONTEXT_ID, schema_version="two_plane_context_v3",
        budgets=asdict(cap), **DECLARATIONS)


def _publication(m, budget=None):
    configuration = _expected_configuration(m, budget)
    return dict(initialization_implementation=INIT_ID, two_plane_context_configuration=configuration,
        two_plane_context_configuration_sha256=_sha(_canonical(configuration)))


def _sheet(axis, lateral, axial, alternate):
    points = np.asarray([(0., a, b) if axis == 0 else (a, 0., b)
        for a in lateral for b in axial], dtype=np.float64)
    triangles, width = [], len(axial)
    for i in range(len(lateral)-1):
        for j in range(width-1):
            a, b = i*width+j, (i+1)*width+j
            c, d = a+1, b+1
            triangles.extend(((a, b, c), (c, b, d)) if alternate else ((a, b, d), (a, d, c)))
    return points, np.asarray(triangles, dtype=np.int64)


def _arrays():
    scans = [_sheet(0, (2., 5., 8., 11., 14.), (-6., -3., 0., 3., 6.), False),
             _sheet(1, (3., 6., 9., 12., 15.), (-6., -3., 0., 3., 6.), True)]
    nominal = np.concatenate([value[0] for value in scans])
    scan_triangles = np.concatenate([value[1]+25*i for i, value in enumerate(scans)])
    # Proper cyclic permutation and integral translation; retained oracle only.
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    translation = np.array((9., -5., 7.))
    observed = nominal@rotation.T+translation
    cads = [_sheet(0, (1., 4.5, 8., 11.5, 15.), (-8., -5., -2., 2., 5., 8.), True),
            _sheet(1, (2., 5.5, 9., 12.5, 16.), (-8., -5., -2., 2., 5., 8.), False)]
    cad = np.concatenate([value[0][value[1]] for value in cads])
    faces = np.repeat(np.array((17, 29), dtype=np.int64), 40)
    local = np.tile(np.arange(40, dtype=np.int64), 2)
    assert observed.shape == (50, 3) and scan_triangles.shape == (64, 3) and cad.shape == (80, 3, 3)
    return (observed, scan_triangles, cad, faces, local), nominal


def _policy(m):
    return m.init._v1.WholePartRegistrationPolicyV1(coarse_points=50, refine_points=50,
        coarse_iterations=1, refine_iterations=1, max_correspondence_distance_mm=.5,
        max_inlier_plane_p95_mm=.1, min_scan_coverage=.95, min_cad_coverage=.25)


def _prefilter_args(m, mode="v1", **overrides):
    options = dict(position_error_bounds_mm=.025, registration_policy=_policy(m), unit_assumption=UNIT,
        search_budgets=m.init._search.PlaneSearchBudgetsV6(max_bank_planes=2),
        two_plane_schedule_mode=mode, two_plane_context_mode="v3")
    if mode == "v2":
        options["two_plane_schedule_budgets"] = m.schedule2.TwoPlaneScheduleBudgetsV2()
    options.update(overrides)
    return options


def _configuration(m, **overrides):
    i = m.init
    options = dict(two_plane_enabled=True, two_plane_schedule_mode="v1",
        two_plane_context_mode="v3", two_plane_context_budgets=None)
    options.update(overrides)
    return i._configuration(_policy(m), i._noisy.NoisyMeshPlanePolicyV5(), i._noisy.NoisyMeshPlaneBudgetsV5(),
        i._patch.CoplanarPatchPolicyV4(), i._patch.CoplanarPatchBudgetsV4(), i._search.PlaneSearchPolicyV6(),
        i._search.PlaneSearchBudgetsV6(max_bank_planes=2), i.PlaneInitializationBudgetsV12(), None, **options)


def _assert_prefilter_refusal(answer):
    assert answer["status"] == "UNAVAILABLE"
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["numerical_consumer_complete"] is False
    assert answer["box_screening_complete"] is False
    assert answer["start_groups"] == []
    assert all(row.get("usable_initialization") is not True for row in answer["hypotheses"])


def test_independent_fixture_and_frozen_numeric_aliases(modules):
    arrays, nominal = _arrays()
    assert len(np.unique(arrays[0], axis=0)) == 50
    assert np.array_equal(np.unique(arrays[1]), np.arange(50))
    assert nominal.shape == (50, 3)
    for name in ("_prepare_triplets", "_bound_scan", "_prefilter", "_registration_rows", "_context_support", "_publish_contexts"):
        assert getattr(modules.init, name) is getattr(modules.frozen, name)
    assert modules.init.PlaneInitializationBudgetsV12 is modules.frozen.PlaneInitializationBudgetsV12


@pytest.mark.parametrize("mode", ("v1", "v2"))
def test_actual_initializer_scheduler_proposer_finite_prefilter_chain(modules, monkeypatch, mode):
    arrays, nominal = _arrays()
    calls, finite_calls, schedules = [], [], []
    actual_propose = modules.context.propose_two_plane_context_v3
    actual_query = modules.frozen._FaceQueries.__call__
    schedule_module = modules.frozen._pair_schedule if mode == "v1" else modules.schedule2
    schedule_name = "schedule_two_plane_pairs_"+mode
    actual_schedule = getattr(schedule_module, schedule_name)
    def schedule_call(*args, **kwargs):
        result = actual_schedule(*args, **kwargs)
        schedules.append(copy.deepcopy(result))
        return result
    def propose(*args, **kwargs):
        result = actual_propose(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result
    def query(*args, **kwargs):
        finite_calls.append(len(args[2]))
        return actual_query(*args, **kwargs)
    monkeypatch.setattr(modules.context, "propose_two_plane_context_v3", propose)
    monkeypatch.setattr(modules.frozen._FaceQueries, "__call__", query)
    monkeypatch.setattr(schedule_module, schedule_name, schedule_call)
    answer = modules.init.prefilter_plane_initializations_v13(*arrays, **_prefilter_args(modules, mode))
    assert answer["status"] == "COMPLETE", answer["reasons"]
    assert answer["initialization_implementation"] == INIT_ID
    assert answer["two_plane_context_configuration"] == _expected_configuration(modules)
    assert answer["two_plane_context_configuration_sha256"] == _sha(_canonical(_expected_configuration(modules)))
    assert answer["schema_version"] == "noisy_plane_initialization_prefilter_v12"
    assert len(calls) == 1 and finite_calls and all(count == 25 for count in finite_calls)
    args, kwargs, context = calls[0]
    assert len(args[0]) == 50 and len(args[1]) == 64 and len(args[2]) == 80
    assert type(kwargs["budgets"]) is modules.context.TwoPlaneContextBudgetsV3
    assert answer["two_plane_context"] is context
    schedule = answer["two_plane_schedule"]
    assert schedules == [schedule], "No post-proposal rescheduling, quota refund or table pruning"
    assert schedule["schema_version"] == "two_plane_schedule_"+mode
    matches = schedule["scheduled_matches"]
    assert len(matches) == 2
    assert matches[0]["scan_plane_ids"] == matches[1]["scan_plane_ids"]
    assert matches[0]["cad_plane_ids"] == matches[1]["cad_plane_ids"][::-1]
    assert answer["joint_proposal_accounting"]["triplet_records"] == 0
    assert answer["joint_proposal_accounting"]["pair_slot_reservation"] == 64
    assert len(context["families"]) == 8 and len(answer["hypotheses"]) == 64
    assert context["statistics"]["signed_families_completed"] == 8
    assert context["statistics"]["slot_records_completed"] == 64
    assert all(tuple(slot["slot"] for slot in f["slots"]) == SLOTS for f in context["families"])
    assert len(context["scan_extrema_indexes"]) == 3
    assert sorted(row["support_row_count"] for row in context["scan_extrema_indexes"]) == [25, 25, 50]
    stats = answer["statistics"]
    assert stats["omitted_proposed_hypothesis_count"] == stats["omitted_unsigned_assignment_count"] == 0
    assert stats["hypotheses_evaluated"] == stats["scheduled_hypothesis_count"]
    assert stats["completed_query_rows"]+stats["bound_excluded_support_rows"] == stats["planned_query_rows"]
    assert all(support["support_count"] == 25 for row in answer["hypotheses"] for support in row["assigned_support"])
    assert answer["start_groups"]
    errors = [np.max(np.linalg.norm(arrays[0]@np.asarray(row["initialized_rotation"]).T+
        np.asarray(row["initialized_translation_mm"])-nominal, axis=1)) for row in answer["start_groups"]]
    assert min(errors) <= 1e-7
    assert answer["search_exhaustive"] is False and answer["global_pose_uniqueness"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("mode", ("v1", "v2"))
@pytest.mark.parametrize("scan_mode", ("v7", "composite_v1"))
def test_configuration_owns_context_caps_independent_of_schedule_mode(modules, mode, scan_mode):
    cap = modules.context.TwoPlaneContextBudgetsV3(chunk_rows=3, max_extrema_indexes=7)
    options = dict(two_plane_schedule_mode=mode, two_plane_context_budgets=cap, scan_plane_mode=scan_mode)
    if mode == "v2":
        options["two_plane_schedule_budgets"] = modules.schedule2.TwoPlaneScheduleBudgetsV2()
    configuration = _configuration(modules, **options)
    assert configuration["two_plane_context_budgets"] is not cap
    expected = _expected_configuration(modules, cap)
    assert configuration["two_plane_context_configuration"] == expected
    publication = modules.init._context_publication(configuration)
    assert publication == _publication(modules, cap)
    expected["budgets"]["chunk_rows"] = 9
    assert configuration["two_plane_context_configuration"]["budgets"]["chunk_rows"] == 3


@pytest.mark.parametrize("mode", (None, False, 3, "v1", "v2", "V3", "v3 "))
def test_context_mode_is_strict_and_precedes_array_preparation(modules, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid context policy must stop before preparation")
    monkeypatch.setattr(modules.init, "_prepare", forbidden)
    answer = modules.init.prefilter_plane_initializations_v13(*(object() for _ in range(5)),
        **_prefilter_args(modules, two_plane_context_mode=mode))
    _assert_prefilter_refusal(answer)


@pytest.mark.parametrize("enabled", (False, None, 0, 1, "true"))
def test_context_v3_cannot_silently_disable_pair_lane(modules, enabled):
    with pytest.raises(ValueError, match="CONTEXT_V3_REQUIRES_ENABLED_TWO_PLANE_LANE"):
        _configuration(modules, two_plane_enabled=enabled)


@pytest.mark.parametrize("fault", ("missing", "extra", "mode", "implementation", "schema", "boolean_budget", "too_large", "arithmetic", "scope"))
def test_context_configuration_snapshot_rejects_tamper(modules, fault):
    value = _expected_configuration(modules)
    if fault == "missing": value.pop("implementation")
    elif fault == "extra": value["unexpected"] = True
    elif fault == "mode": value["mode"] = "v2"
    elif fault == "implementation": value["implementation"] = "LEGACY_CONTEXT"
    elif fault == "schema": value["schema_version"] = "two_plane_context_v2"
    elif fault == "boolean_budget": value["budgets"]["max_extrema_indexes"] = True
    elif fault == "too_large": value["budgets"]["max_extrema_indexes"] = 130
    elif fault == "arithmetic": value["scan_projection_arithmetic"] = "BLAS"
    else: value["extrema_membership_scope"] = "WHOLE_SCAN_NOT_CHARGED"
    with pytest.raises(ValueError):
        modules.init._context_configuration_snapshot(value)


@pytest.mark.parametrize("fault", ("implementation", "missing_configuration", "digest", "budget"))
def test_requested_returned_publication_requires_exact_config_and_hash(modules, fault):
    publication = _publication(modules)
    if fault == "implementation": publication["initialization_implementation"] = "V12"
    elif fault == "missing_configuration": publication.pop("two_plane_context_configuration")
    elif fault == "digest": publication["two_plane_context_configuration_sha256"] = "0"*64
    else:
        publication["two_plane_context_configuration"]["budgets"]["chunk_rows"] = 3
        publication["two_plane_context_configuration_sha256"] = _sha(_canonical(publication["two_plane_context_configuration"]))
    with pytest.raises(ValueError):
        modules.init._bind_context_publication(publication, _expected_configuration(modules))


@pytest.mark.parametrize("fault", ("schema", "budget", "arithmetic", "scope", "capacity",
    "scan_points_sha256", "scan_triangles_sha256", "cad_triangles_sha256", "cad_face_ids_sha256",
    "cad_source_triangle_ids_sha256", "schedule", "sign", "slot", "pose"))
def test_tampered_actual_context_cannot_enter_finite_prefilter(modules, monkeypatch, fault):
    actual = modules.context.propose_two_plane_context_v3
    proposer_calls, prefilter_calls = [], []
    def tampered(*args, **kwargs):
        result = actual(*args, **kwargs)
        assert result["status"] == "COMPLETE", result["reasons"]
        proposer_calls.append(result)
        if fault == "schema": result["schema_version"] = "two_plane_context_v2"
        elif fault == "budget": result["budgets"]["max_extrema_indexes"] -= 1
        elif fault == "arithmetic": result["scan_projection_arithmetic"] = "OLD_MATMUL"
        elif fault == "scope": result["extrema_membership_scope"] = "ASSIGNED_ONLY"
        elif fault == "capacity": result["scheduling_capacity"]["slots_per_sign"] = 7
        elif fault.endswith("_sha256"): result["input_digests"][fault] = "0"*64
        elif fault == "schedule": result["scheduled_pairs"].reverse()
        elif fault == "sign": result["families"][1]["normal_signs"] = result["families"][0]["normal_signs"][:]
        elif fault == "slot": result["families"][-1]["slots"][-1]["slot"] = "WHOLE_MIN"
        else: result["candidates"][0]["translation_mm"][0] += 1.
        return result
    def forbidden(*args, **kwargs):
        prefilter_calls.append(True)
        raise AssertionError("tampered helper must stop before finite prefilter")
    monkeypatch.setattr(modules.context, "propose_two_plane_context_v3", tampered)
    monkeypatch.setattr(modules.init, "_prefilter", forbidden)
    arrays, _ = _arrays()
    answer = modules.init.prefilter_plane_initializations_v13(*arrays, **_prefilter_args(modules))
    assert len(proposer_calls) == 1 and not prefilter_calls
    _assert_prefilter_refusal(answer)


@pytest.mark.parametrize("site", ("context_budget", "late_finite_query"))
def test_actual_initializer_whole_refusal_keeps_no_usable_start_prefix(modules, monkeypatch, site):
    arrays, _ = _arrays()
    options = _prefilter_args(modules)
    calls = []
    if site == "context_budget":
        options["two_plane_context_budgets"] = modules.context.TwoPlaneContextBudgetsV3(max_extrema_indexes=2)
    else:
        actual = modules.frozen._FaceQueries.__call__
        def broken(*args, **kwargs):
            calls.append(True)
            if len(calls) == 3:
                raise ValueError("INVENTED_THIRD_FINITE_QUERY_FAILURE")
            return actual(*args, **kwargs)
        monkeypatch.setattr(modules.frozen._FaceQueries, "__call__", broken)
    answer = modules.init.prefilter_plane_initializations_v13(*arrays, **options)
    _assert_prefilter_refusal(answer)
    if site == "context_budget":
        assert answer["two_plane_context"]["status"] == "UNAVAILABLE"
        assert answer["two_plane_context"]["statistics"]["extrema_indexes"] == 3
        assert answer["two_plane_context"]["candidates"] == []
    else:
        assert len(calls) == 3 and answer["statistics"]["completed_query_calls"] == 2
        assert all(group["initialized_rotation"] is None and group["initialized_translation_mm"] is None
            for group in answer["all_passing_start_groups"])


def _byte_arrays():
    # New48row fixture distinct from50row real numerical fixture. All bytes are
    # in-memory declarations, never files or native decoder input.
    raw = _owned([(float(i % 4), float((i//4) % 4), float(i//16)) for i in range(48)])
    selection = _owned(range(36), "<i8")
    sample = _owned(raw[selection])
    tri = _owned(((0, 1, 4), (1, 5, 4)), "<i8")
    scan = dict(status="AVAILABLE", reasons=[], raw_points=raw, points=sample,
        engine_to_raw_row_indices=selection, input_admission=dict(face_count=2, face_index_count=6),
        coverage=dict(raw_vertex_count=48, selected_engine_rows=36),
        measurement_cloud_contract="FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
        sample_cloud_contract="REGISTRATION_AND_PROPOSALS_ONLY", units_status="UNVERIFIED",
        coordinate_frame_status="UNVERIFIED", source_bindings=dict(scan_sha256=_sha(SCAN_BYTES),
            raw_xyz_sha256=_sha(raw), sample_xyz_sha256=_sha(sample), engine_to_raw_sha256=_sha(selection)))
    topology = dict(status="AVAILABLE", reasons=[], points=raw, triangle_raw_rows=tri,
        source_sha256=_sha(SCAN_BYTES), points_sha256=_sha(raw), triangle_raw_rows_sha256=_sha(tri),
        vertex_count=48, triangle_count=2, raw_row_order="ORIGINAL_PLY_VERTEX_ROWS",
        triangle_row_order="ORIGINAL_PLY_FACE_ROWS", topology_status="INVENTED_BOUNDARY_STUB")
    triangles = _owned((((0, 0, 0), (8, 0, 0), (0, 8, 0)),
                        ((0, 0, 4), (8, 0, 4), (0, 8, 4)),
                        ((6, 0, 0), (6, 8, 0), (6, 0, 8))))
    faces, local = _owned((13, 23, 37), "<i8"), _owned((2, 4, 6), "<i8")
    sample_rows = _owned([i % 3 for i in range(36)], "<i8")
    points = _owned(np.asarray(((2., 2., 0.), (2., 2., 4.), (6., 2., 2.)))[sample_rows])
    normals = _owned(np.asarray(((0., 0., 1.), (0., 0., 1.), (1., 0., 0.)))[sample_rows])
    weights = _owned([(0.5, 0.25, 0.25)]*36)
    sampling = dict(sample_count=36, fixture="INVENTED_EXACT_DYADIC_SAMPLING")
    cad = dict(status="AVAILABLE", reasons=[], triangles=triangles, source_face_indices=faces,
        source_triangle_indices=local, points=points, normals=normals, sample_triangle_rows=sample_rows,
        sample_source_face_indices=_owned(faces[sample_rows], "<i8"),
        sample_source_triangle_indices=_owned(local[sample_rows], "<i8"), sample_barycentric_weights=weights,
        sampling_policy=sampling, digests=dict(oriented_triangle_xyz_sha256=_sha(triangles),
            source_face_ids_sha256=_sha(faces), source_triangle_ids_sha256=_sha(local),
            sample_points_sha256=_sha(points), sample_normals_sha256=_sha(normals),
            sample_triangle_rows_sha256=_sha(sample_rows), sampling_policy_sha256=_sha(_canonical(sampling))))
    return scan, topology, cad


def _registration(m, scan, topology, cad):
    digests = dict(scan_points_sha256=_sha(scan["raw_points"]),
        scan_triangle_raw_rows_sha256=_sha(topology["triangle_raw_rows"]),
        position_error_bounds_mm_sha256=_sha(np.full(48, .025, dtype="<f8")),
        cad_triangles_sha256=_sha(cad["triangles"]), cad_face_ids_sha256=_sha(cad["source_face_indices"]),
        cad_source_triangle_ids_sha256=_sha(cad["source_triangle_indices"]))
    hypotheses, groups, candidates = [], [], []
    support_policy = "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
    distance = "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
    optimizer = "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS"
    for index in range(3):
        rotation, translation = np.eye(3).tolist(), [float(index), -float(index), 2.*index]
        matches = [dict(scan_plane_id=i, cad_plane_id=face, normal_sign=1) for i, face in enumerate((13, 23, 37))]
        hypotheses.append(dict(hypothesis_index=index, unsigned_assignment_id=index, proposal_lane="TRIPLET",
            status="PROPOSED", prefilter_status="PASSED", exact_start_group_index=index,
            correspondences=copy.deepcopy(matches), assigned_support=[dict(item, status="PASSED") for item in matches],
            rotation=copy.deepcopy(rotation), translation_mm=list(translation)))
        group = dict(index=index, exact_group_index=index, hypothesis_indices=[index],
            initialized_rotation=rotation, initialized_translation_mm=translation,
            refinement_disposition="SCHEDULED_FOR_V3_REFINEMENT")
        groups.append(group)
        candidates.append(dict(copy.deepcopy(group), rotation=copy.deepcopy(rotation), translation=list(translation),
            execution_status="ATTEMPTED", reasons=["INVENTED_NOT_ACTUAL_REFINEMENT"],
            numerical_consumer_status="COMPLETE_V3_INTERVAL_SUPPORT_POLICY", numerical_support_policy=support_policy,
            distance_model=distance, optimizer_objective=optimizer,
            conditional_context_support=dict(status="SUPPORTED_CONTEXT", accepted_registration=False,
                proximity_policy=m.workflow._stages.PROXIMITY,
                pose_sha256=_sha(np.asarray([*np.asarray(rotation).ravel(), *translation], dtype="<f8")))))
    initialization = dict(schema_version="noisy_plane_initialization_prefilter_v12", status="COMPLETE",
        search_complete=True, selected_policy_evaluation_complete=True, search_exhaustive=False,
        numerical_consumer_complete=True, box_screening_complete=True,
        box_screening_policy="REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1",
        numerical_support_policy=m.workflow._stages.PROXIMITY, input_digests=copy.deepcopy(digests),
        hypotheses=hypotheses, start_groups=copy.deepcopy(groups), all_passing_start_groups=copy.deepcopy(groups),
        joint_proposal_accounting=dict(pair_lane_enabled=True, triplet_records=3, triplet_unsigned_reservation=3,
            ordered_pair_matches=0, pair_slot_reservation=0, total_hypothesis_records=3),
        two_plane_schedule=dict(status="COMPLETE", selected_policy_evaluation_complete=True, scheduled_matches=[]),
        **_publication(m))
    evidence = dict(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12",
        initialization=initialization, candidate_evaluation_complete=True, numerical_consumer_complete=True,
        connected_measurement_contexts=True, accepted_registration=False, numerical_support_policy=support_policy,
        distance_model=distance, optimizer_objective=optimizer,
        stage_policies=m.workflow._stages.ablation_stage_policies_v1(connected=True), input_digests=digests,
        surface_role_preparation=dict(metadata_sha256="a"*64), candidates=candidates,
        measurement_context_indices=[0, 1, 2], prefilter_full_scan_raw_rows=48, registration_input_rows=36,
        registration_input_points_sha256=_sha(scan["points"]),
        registration_raw_rows=dict(count=36, selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING",
            sha256=_sha(scan["engine_to_raw_row_indices"])),
        cad_initialization_points_sha256=_sha(np.unique(cad["points"], axis=0)), **_publication(m))
    return dict(schema_version="noisy_plane_initialized_finite_registration_v12",
        status="CONDITIONAL_MEASUREMENT_CONTEXTS", reasons=[], transform_scan_to_cad=None, evidence=evidence)


@pytest.fixture
def plumbing(modules, monkeypatch):
    m = modules
    scan, topology, cad = _byte_arrays()
    state = types.SimpleNamespace(m=m, scan=scan, topology=topology, cad=cad, shape=object(), roles=object(),
        events=[], bindings=[], measurement_calls=[], validator_calls=[], registration_calls=[], damage=None)
    state.registration = _registration(m, scan, topology, cad)
    state.feature = object()  # Native boundary mock only; never passed to real geometry.
    state.measured = dict(status="PARTIAL", computation_status="COMPLETE", reasons=[],
        predictions=[dict(invented="sole_top_level_carrier")], source_context_result=dict(
            predictions=[dict(invented="nested_must_not_be_concatenated")]), observed_bend_count=None,
        production_use_allowed=False)
    def prepare(**kwargs):
        state.events.append("prepare")
        assert kwargs == dict(scan_bytes=SCAN_BYTES, max_engine_rows=36)
        return state.scan
    def parse(blob):
        state.events.append("parse")
        assert blob is SCAN_BYTES
        return state.topology
    def read(blob):
        state.events.append("native_read_mock")
        assert blob is CAD_BYTES
        return state.shape, dict(target_unit="mm", physical_calibration="NOT_ESTABLISHED")
    def extract(shape, **kwargs):
        state.events.append("native_features_mock")
        assert shape is state.shape
        return dict(status="PARTIAL", features=(state.feature,), metadata=dict(fixture="NATIVE_BOUNDARY_STUB"))
    def sample(shape, **kwargs):
        state.events.append("native_sampling_mock")
        assert shape is state.shape
        return state.cad
    def register(*args, **kwargs):
        state.events.append("register_mock")
        state.registration_calls.append((args, kwargs))
        if state.damage is not None:
            state.damage(state.registration)
        return state.registration, state.roles
    original_binding = m.workflow._registration_binding
    def binding(*args, **kwargs):
        index = kwargs["context_index"]
        state.events.append("bind:"+str(index))
        state.bindings.append(index)
        return original_binding(*args, **kwargs)
    def measure(raw, features, **kwargs):
        state.events.append("measure_mock")
        state.measurement_calls.append((raw, features, kwargs))
        assert raw is state.scan["raw_points"] and kwargs["surface_roles"] is state.roles
        return state.measured
    def validate(*args, **kwargs):
        state.events.append("validate_mock")
        state.validator_calls.append((args, kwargs))
        return dict(fixture="PUBLICATION_BOUNDARY_STUB")
    monkeypatch.setattr(m.mesh, "prepare_mesh_scan_v1", prepare)
    monkeypatch.setattr(m.topology, "parse_triangle_mesh_ply_v4", parse)
    monkeypatch.setattr(m.workflow._v2, "_read_step_owned_snapshot", read)
    monkeypatch.setattr(m.native1, "extract_ocp_guided_features_v1", extract)
    monkeypatch.setattr(m.native, "extract_ocp_guided_features", extract)
    monkeypatch.setattr(m.cad_surface, "sample_native_cad_surface_v2", sample)
    monkeypatch.setattr(m.init, "register_measurement_contexts_v13", register)
    monkeypatch.setattr(m.workflow, "_registration_binding", binding)
    monkeypatch.setattr(m.cover, "inspect_support_cover_context_windows_v2", measure)
    monkeypatch.setattr(m.cover, "validate_support_cover_publication_v2", validate)
    def run(**overrides):
        options = dict(scan_bytes=SCAN_BYTES, cad_bytes=CAD_BYTES, position_error_bound_mm=.025,
            workflow_policy=m.workflow.MeshWorkflowPolicyV14(max_registration_scan_rows=36, cad_sample_count=36),
            registration_policy=_policy(m), measurement_policy=m.native.NativeGuidedPolicy(
                flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.05,
                plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
            instance_policy=m.cover._v1.SupportPolicy(4., 5., 20., 3), allow_ocp=True,
            connected_measurement_contexts=True, two_plane_context_mode="v3")
        options.update(overrides)
        return m.workflow.inspect_cad_guided_mesh_v14(**options)
    state.run = run
    return state


def _assert_workflow_refusal(answer):
    assert answer["status"] == "UNAVAILABLE"
    assert answer["implementation"] == WORKFLOW_ID
    assert answer["predictions"] == [] and answer["measurements"] is None
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False


def test_actual_workflow_binds_all_three_contexts_and_full_raw_transport(plumbing):
    state = plumbing
    answer = state.run()
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", answer
    assert answer["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13"
    assert answer["implementation"] == WORKFLOW_ID
    assert state.bindings == [0, 1, 2]
    assert len(state.registration_calls) == len(state.measurement_calls) == len(state.validator_calls) == 1
    args, kwargs = state.registration_calls[0]
    assert args[0] is state.scan["raw_points"] and len(args[0]) == 48
    assert kwargs["registration_raw_rows"] is state.scan["engine_to_raw_row_indices"]
    assert kwargs["two_plane_context_mode"] == "v3"
    assert type(kwargs["two_plane_context_budgets"]) is state.m.context.TwoPlaneContextBudgetsV3
    expected_sha = _sha(_canonical(_expected_configuration(state.m)))
    for publication in (answer, answer["policies"], state.registration["evidence"],
                        state.registration["evidence"]["initialization"]):
        assert publication["initialization_implementation"] == INIT_ID
        assert publication["two_plane_context_configuration_sha256"] == expected_sha
    assert len(answer["registration_array_binding"]) == 3
    for binding in answer["registration_array_binding"]:
        assert binding["initialization_implementation"] == INIT_ID
        assert binding["two_plane_context_mode"] == "v3"
        assert binding["two_plane_context_implementation"] == CONTEXT_ID
        assert binding["two_plane_context_configuration_sha256"] == expected_sha
    assert state.events[-5:] == ["bind:0", "bind:1", "bind:2", "measure_mock", "validate_mock"]
    assert answer["predictions"] is state.measured["predictions"]
    assert answer["predictions"] != state.measured["source_context_result"]["predictions"]


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("fault", ("candidate_pose", "role", "group_origin", "initialized_pose"))
def test_every_selected_context_is_bound_before_any_measurement(plumbing, index, fault):
    state = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        candidate = evidence["candidates"][index]
        if fault == "candidate_pose": candidate["rotation"][2][2] = -1.
        elif fault == "role": evidence["initialization"]["hypotheses"][index]["assigned_support"][0]["cad_plane_id"] = 99
        elif fault == "group_origin": candidate["hypothesis_indices"] = [(index+1) % 3]
        else: candidate["initialized_translation_mm"][0] += 1.
    state.damage = damage
    answer = state.run()
    _assert_workflow_refusal(answer)
    assert state.bindings == list(range(index+1))
    assert state.measurement_calls == state.validator_calls == []
    assert answer["registration_array_binding"] == []
    evidence = answer["registration"]["evidence"]
    assert evidence["measurement_context_indices"] == []
    assert evidence["attempted_measurement_context_indices"] == [0, 1, 2]


@pytest.mark.parametrize("location", ("evidence", "initialization"))
@pytest.mark.parametrize("fault", ("implementation", "missing_configuration", "digest", "budget", "arithmetic"))
def test_empty_pair_schedule_does_not_bypass_mandatory_v3_publication(plumbing, location, fault):
    state = plumbing
    def damage(registration):
        publication = registration["evidence"]
        if location == "initialization": publication = publication["initialization"]
        if fault == "implementation": publication["initialization_implementation"] = "V12"
        elif fault == "missing_configuration": publication.pop("two_plane_context_configuration")
        elif fault == "digest": publication["two_plane_context_configuration_sha256"] = "0"*64
        elif fault == "budget":
            publication["two_plane_context_configuration"]["budgets"]["chunk_rows"] = 3
            publication["two_plane_context_configuration_sha256"] = _sha(_canonical(publication["two_plane_context_configuration"]))
        else: publication["two_plane_context_configuration"]["scan_projection_arithmetic"] = "OLD_BLAS"
    state.damage = damage
    answer = state.run()
    _assert_workflow_refusal(answer)
    assert state.bindings == state.measurement_calls == state.validator_calls == []
    assert answer["registration"]["evidence"]["measurement_context_indices"] == []


@pytest.mark.parametrize("fault", ("missing", "duplicate", "boolean", "range"))
def test_invalid_complete_context_index_table_cannot_publish_prefix(plumbing, fault):
    def damage(registration):
        evidence = registration["evidence"]
        if fault == "missing": evidence.pop("measurement_context_indices")
        elif fault == "duplicate": evidence["measurement_context_indices"] = [0, 1, 1]
        elif fault == "boolean": evidence["measurement_context_indices"] = [0, 1, True]
        else: evidence["measurement_context_indices"] = [0, 1, 3]
    plumbing.damage = damage
    answer = plumbing.run()
    _assert_workflow_refusal(answer)
    assert plumbing.measurement_calls == plumbing.validator_calls == []
    assert answer["registration_array_binding"] == []


@pytest.mark.parametrize("site", ("measurement", "validator", "final_report"))
def test_late_workflow_refusal_never_exposes_measurement_prefix(plumbing, monkeypatch, site):
    state = plumbing
    def broken(*args, **kwargs):
        raise ValueError("INVENTED_LATE_WORKFLOW_FAILURE")
    if site == "measurement":
        monkeypatch.setattr(state.m.cover, "inspect_support_cover_context_windows_v2", broken)
    elif site == "validator":
        monkeypatch.setattr(state.m.cover, "validate_support_cover_publication_v2", broken)
    else:
        original = state.m.workflow._bounded_workflow
        monkeypatch.setattr(state.m.workflow, "_bounded_workflow", lambda output: original(output, maximum=128))
    answer = state.run()
    _assert_workflow_refusal(answer)
    assert state.bindings == [0, 1, 2]
    if site == "final_report":
        assert answer["registration"] is None and answer["detailed_report_discarded_whole"] is True


@pytest.mark.parametrize("publication_call", (1, 2))
@pytest.mark.parametrize("fault", ("digest", "arithmetic"))
def test_workflow_top_and_policy_publications_are_independently_bound(plumbing, monkeypatch, publication_call, fault):
    state = plumbing
    actual = state.m.init._context_publication
    count = 0
    def altered(*args, **kwargs):
        nonlocal count
        count += 1
        value = actual(*args, **kwargs)
        if count == publication_call:
            if fault == "digest": value["two_plane_context_configuration_sha256"] = "0"*64
            else: value["two_plane_context_configuration"]["scan_projection_arithmetic"] = "OLD_BLAS"
        return value
    monkeypatch.setattr(state.m.init, "_context_publication", altered)
    answer = state.run()
    assert count == 2
    _assert_workflow_refusal(answer)
    assert state.bindings == state.measurement_calls == state.validator_calls == []
    assert answer["registration"]["evidence"]["measurement_context_indices"] == []


@pytest.mark.parametrize("mode", (None, "v1", "v2", False))
def test_workflow_invalid_context_mode_never_reaches_native_or_array_decoding(plumbing, mode):
    answer = plumbing.run(two_plane_context_mode=mode)
    _assert_workflow_refusal(answer)
    assert plumbing.events == []


def test_context_capacity_is_not_misidentified_as_v2_scheduler_marker(modules):
    registration = dict(evidence=dict(initialization=dict(two_plane_schedule=dict(status="COMPLETE"),
        two_plane_context=dict(schema_version="two_plane_context_v3", scheduling_capacity=dict(maximum_ordered_matches=256)))))
    assert modules.workflow._two_plane_schedule_binding(registration, None) is None
    registration["evidence"]["initialization"]["two_plane_schedule"]["schema_version"] = "two_plane_schedule_v2"
    with pytest.raises(ValueError, match="UNREQUESTED_V2_TWO_PLANE_SCHEDULE_PUBLICATION"):
        modules.workflow._two_plane_schedule_binding(registration, None)


def test_actual_workflow_v2_schedule_request_retains_mandatory_context_v3(plumbing):
    state, m = plumbing, plumbing.m
    cap = m.schedule2.TwoPlaneScheduleBudgetsV2()
    configuration = dict(mode="v2", budgets=asdict(cap))
    # Deliberate empty equation-bank boundary stub: this validates request and
    # publication plumbing, not triplet-search or pose recovery. The real V2
    # scheduler constructs its full empty breadth/residual reservation ledgers.
    schedule = m.schedule2.schedule_two_plane_pairs_v2((), (),
        retained_triplet_hypothesis_count=3, retained_triplet_unsigned_count=3,
        max_joint_hypothesis_records=8192, max_joint_unsigned_assignments=1024, budgets=cap)
    assert schedule["status"] == "COMPLETE" and schedule["scheduled_matches"] == []
    evidence = state.registration["evidence"]
    initialization = evidence["initialization"]
    evidence["two_plane_schedule_configuration"] = copy.deepcopy(configuration)
    initialization.update(two_plane_schedule_configuration=copy.deepcopy(configuration), two_plane_schedule=schedule,
        budgets=asdict(m.init.PlaneInitializationBudgetsV12()),
        hypothesis_evidence=dict(scan_bank=[], scan_planes=[]), patch_evidence=dict(cad=dict(patches=[])))
    answer = state.run(two_plane_schedule_mode="v2", two_plane_schedule_budgets=cap)
    assert answer["status"] == "PARTIAL", answer
    assert state.bindings == [0, 1, 2]
    _, kwargs = state.registration_calls[0]
    assert kwargs["two_plane_schedule_mode"] == "v2" and kwargs["two_plane_context_mode"] == "v3"
    assert kwargs["two_plane_schedule_budgets"] is not cap
    assert asdict(kwargs["two_plane_schedule_budgets"]) == asdict(cap)
    for binding in answer["registration_array_binding"]:
        assert binding["two_plane_schedule_mode"] == "v2"
        assert binding["two_plane_context_mode"] == "v3"
        assert binding["two_plane_context_configuration_sha256"] == _sha(_canonical(_expected_configuration(m)))


@pytest.mark.parametrize("kind", ("dict", "boolean", "v1", "v2"))
def test_context_budget_selector_rejects_untyped_or_legacy_profiles(modules, kind):
    if kind == "dict": budget = asdict(modules.context.TwoPlaneContextBudgetsV3())
    elif kind == "boolean": budget = True
    elif kind == "v1": budget = modules.context._v1.TwoPlaneContextBudgetsV1()
    else: budget = modules.context._v2.TwoPlaneContextBudgetsV2()
    with pytest.raises(ValueError):
        _configuration(modules, two_plane_context_budgets=budget)


def test_direct_selected_binding_requires_explicit_expected_context_config(plumbing):
    state = plumbing
    with pytest.raises(ValueError, match="INVALID_CONTEXT_V3_CONFIGURATION"):
        state.m.workflow._registration_binding(state.registration, state.scan, state.topology,
            state.cad, state.scan["engine_to_raw_row_indices"], .025, context_index=0)
    assert state.measurement_calls == state.validator_calls == []
