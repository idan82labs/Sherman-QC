"""Initializer-V14/workflow-V15 closure on context V4 and surface V4, invented only.

Derived from the terminal V2 integration selection (all original cases retained
with V14/V15/V4 identities) plus a live V4 registration-implementation binding.

New fixture revision after the terminal original qualification failure. All93
original parameter cases are retained, plus one explicit valid-policy guard.
The already-passed dense-support test is not imported, collected or rerun here.

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
INIT_ID = "CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1"
CONTEXT_ID = "RAW_EXTREMA_KD_PARTITION_CONTEXT_V4_STAGED_SCAN_PROJECTION"
WORKFLOW_ID = "CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1"
REGISTRATION_ID = "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
SCAN_BYTES = b"NEW-INVENTED-CONTEXT-V4-WORKFLOW-SCAN-NOT-A-FILE"
CAD_BYTES = b"NEW-INVENTED-CONTEXT-V4-WORKFLOW-CAD-NOT-STEP"
DECLARATIONS = {
    "scan_projection_arithmetic": "STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1",
    "scan_extrema_partition": "RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3",
    "scan_projection_bound_scope": "ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
    "cad_projection_arithmetic": "UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
    "scan_projection_chunk_scope": "FIXED_EXTREMA_LEAF64_NOT_CHUNK_ROWS",
    "chunk_rows_scope": "INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
    "extrema_guard_status": "NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_V3_INVARIANTS",
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
    mapping = dict(init="cad_guided_plane_initialization_v16", frozen="cad_guided_plane_initialization_v12",
        context="cad_guided_two_plane_context_v4", schedule2="cad_guided_two_plane_schedule_v2",
        workflow="cad_guided_mesh_offline_v18", mesh="cad_guided_mesh_input_v1",
        topology="cad_guided_mesh_topology_v4", native="cad_guided_native_features",
        native1="cad_guided_native_features_v1", cad_surface="cad_guided_cad_surface_v2",
        kernel1="cad_guided_measurement",
        cover="cad_guided_support_cover_measurement_v5")
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
    cap = m.context.TwoPlaneContextBudgetsV4() if budget is None else budget
    return dict(mode="v4", implementation=CONTEXT_ID, schema_version="two_plane_context_v4",
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
        huber_delta_mm=.05, max_inlier_plane_p95_mm=.1, min_scan_coverage=.95, min_cad_coverage=.25)


def _prefilter_args(m, mode="v1", **overrides):
    options = dict(position_error_bounds_mm=.025, registration_policy=_policy(m), unit_assumption=UNIT,
        search_budgets=m.init._search.PlaneSearchBudgetsV6(max_bank_planes=2),
        two_plane_schedule_mode=mode, two_plane_context_mode="v4")
    if mode == "v2":
        options["two_plane_schedule_budgets"] = m.schedule2.TwoPlaneScheduleBudgetsV2()
    options.update(overrides)
    return options


def _configuration(m, **overrides):
    i = m.init
    options = dict(two_plane_enabled=True, two_plane_schedule_mode="v1",
        two_plane_context_mode="v4", two_plane_context_budgets=None)
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
    for name in ("_prepare_triplets", "_bound_scan", "_registration_rows", "_publish_contexts"):
        assert getattr(modules.init, name) is getattr(modules.frozen, name)
    # V14 carries verbatim V12 copies of these, bound to the V4 modules.
    import inspect
    frozen_strata = '    indices = _search._strata(len(groups), budget.max_refined_start_groups)\n'
    completion_insert = ('    main_group_count = len(groups)\n'
        '    _complete_two_of_three(context, result, configuration, scheduled, scans, faces, groups, lookup)\n'
        '    stats.update(main_passing_start_group_count=main_group_count,\n'
        '                 completion_passing_start_group_count=len(groups)-main_group_count)\n'
        '    # The frozen strata schedule applies to the main-lane groups exactly as in V14;\n'
        '    # completion-only groups fill the remaining refinement slots by their own strata\n'
        '    # and can never displace a main-lane group.\n'
        '    indices = list(_search._strata(main_group_count, budget.max_refined_start_groups))\n'
        '    remaining = budget.max_refined_start_groups-len(indices)\n'
        '    completion_groups = len(groups)-main_group_count\n'
        '    completion_indices = (list(_search._strata(completion_groups, remaining)) if remaining > 0 and completion_groups else [])\n'
        '    indices.extend(main_group_count+index for index in completion_indices)\n'
        '    result["completion_lane"]["refinement_schedule"] = dict(main_groups=main_group_count,\n'
        '        main_scheduled=len(indices)-len(completion_indices), completion_groups=completion_groups,\n'
        '        completion_scheduled=len(completion_indices), completion_omitted=completion_groups-len(completion_indices),\n'
        '        rule="MAIN_LANE_STRATA_FIRST_THEN_COMPLETION_STRATA_IN_REMAINING_SLOTS")\n'
        '    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])\n')
    v15 = importlib.import_module("backend.cad_guided_plane_initialization_v15")
    for name in ("_context_support", "_FaceQueries"):
        assert getattr(modules.init, name) is not getattr(modules.frozen, name)
        copied = inspect.getsource(getattr(modules.init, name))
        assert copied.replace("TriangleSurfaceBudgetsV4(", "TriangleSurfaceBudgets(") == inspect.getsource(getattr(modules.frozen, name))
    # V16 _prefilter == V15 _prefilter with exactly the declared allowance schedule block substituted.
    v15_source, v16_source = inspect.getsource(v15._prefilter), inspect.getsource(modules.init._prefilter)
    v15_block_start = v15_source.index("    main_group_count = len(groups)\n")
    v15_block_end = v15_source.index('    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])\n') + len('    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])\n')
    v16_block_start = v16_source.index("    main_group_count = len(groups)\n")
    v16_block_end = v16_source.index('    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])\n') + len('    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])\n')
    assert v15_source[:v15_block_start] == v16_source[:v16_block_start] and v15_source[v15_block_end:] == v16_source[v16_block_end:]
    assert "allocation_remaining=completion_quota" in v16_source[v16_block_start:v16_block_end]
    assert "_rollback_completion_lane" in v16_source[v16_block_start:v16_block_end]
    # The configuration copy: renamed def, one added docstring line, V4 constructor.
    copied = inspect.getsource(modules.init._frozen_configuration_v4)
    copied = copied.replace("def _frozen_configuration_v4(", "def _configuration(")
    copied = copied.replace('    """Verbatim V12 configuration body; surface budgets are the V4 type here."""\n', "")
    copied = copied.replace("TriangleSurfaceBudgetsV4(", "TriangleSurfaceBudgets(")
    assert copied == inspect.getsource(modules.frozen._configuration)
    assert modules.init._base()["finite_surface_implementation"] == REGISTRATION_ID
    assert modules.init._surface.__name__.endswith("cad_guided_triangle_surface_v4")
    assert modules.init._v3.__name__.endswith("cad_guided_global_registration_v4")
    assert modules.init._stages.__name__.endswith("cad_guided_stage_policies_v2")
    assert modules.init.PlaneInitializationBudgetsV12 is modules.frozen.PlaneInitializationBudgetsV12


def test_fixture_policy_order_and_owned_configuration_are_valid(modules):
    policy = _policy(modules)
    assert (policy.huber_delta_mm, policy.max_inlier_plane_p95_mm,
        policy.max_correspondence_distance_mm) == (.05, .1, .5)
    assert 0 < policy.huber_delta_mm <= policy.max_inlier_plane_p95_mm <= policy.max_correspondence_distance_mm
    assert policy.coarse_points == policy.refine_points == 50
    assert policy.coarse_iterations == policy.refine_iterations == 1
    assert .5 <= policy.trim_fraction <= 1 and policy.max_rotation_step_radians <= .25
    configuration = _configuration(modules)
    assert asdict(configuration["policy"]) == asdict(policy)
    assert configuration["two_plane_context_configuration"] == _expected_configuration(modules)
    # Constructor-only negative control for the old invalid cross-field tuple;
    # no old test, attempt, geometry or policy-bearing inference is rerun.
    with pytest.raises(ValueError, match="^INVALID_WHOLE_PART_POLICY_ORDER$"):
        modules.init._v1.WholePartRegistrationPolicyV1(**dict(asdict(policy), huber_delta_mm=.25))


@pytest.mark.parametrize("mode", ("v1", "v2"))
def test_actual_initializer_scheduler_proposer_finite_prefilter_chain(modules, monkeypatch, mode):
    arrays, nominal = _arrays()
    calls, finite_calls, schedules = [], [], []
    actual_propose = modules.context.propose_two_plane_context_v4
    actual_query = modules.init._FaceQueries.__call__
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
    monkeypatch.setattr(modules.context, "propose_two_plane_context_v4", propose)
    monkeypatch.setattr(modules.init._FaceQueries, "__call__", query)
    monkeypatch.setattr(schedule_module, schedule_name, schedule_call)
    answer = modules.init.prefilter_plane_initializations_v16(*arrays, **_prefilter_args(modules, mode))
    assert answer["status"] == "COMPLETE", answer["reasons"]
    assert answer["initialization_implementation"] == INIT_ID
    assert answer["two_plane_context_configuration"] == _expected_configuration(modules)
    assert answer["two_plane_context_configuration_sha256"] == _sha(_canonical(_expected_configuration(modules)))
    assert answer["schema_version"] == "noisy_plane_initialization_prefilter_v12"
    assert len(calls) == 1 and finite_calls and all(count == 25 for count in finite_calls)
    args, kwargs, context = calls[0]
    assert len(args[0]) == 50 and len(args[1]) == 64 and len(args[2]) == 80
    assert type(kwargs["budgets"]) is modules.context.TwoPlaneContextBudgetsV4
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
    cap = modules.context.TwoPlaneContextBudgetsV4(chunk_rows=3, max_extrema_indexes=7)
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


@pytest.mark.parametrize("mode", (None, False, 3, "v1", "v2", "v3", "V4", "v4 "))
def test_context_mode_is_strict_and_precedes_array_preparation(modules, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid context policy must stop before preparation")
    monkeypatch.setattr(modules.init, "_prepare", forbidden)
    answer = modules.init.prefilter_plane_initializations_v16(*(object() for _ in range(5)),
        **_prefilter_args(modules, two_plane_context_mode=mode))
    _assert_prefilter_refusal(answer)


@pytest.mark.parametrize("enabled", (False, None, 0, 1, "true"))
def test_context_v4_cannot_silently_disable_pair_lane(modules, enabled):
    with pytest.raises(ValueError, match="CONTEXT_V4_REQUIRES_ENABLED_TWO_PLANE_LANE"):
        _configuration(modules, two_plane_enabled=enabled)


@pytest.mark.parametrize("fault", ("missing", "extra", "mode", "implementation", "schema", "boolean_budget", "too_large", "arithmetic", "scope"))
def test_context_configuration_snapshot_rejects_tamper(modules, fault):
    value = _expected_configuration(modules)
    if fault == "missing": value.pop("implementation")
    elif fault == "extra": value["unexpected"] = True
    elif fault == "mode": value["mode"] = "v3"
    elif fault == "implementation": value["implementation"] = "LEGACY_CONTEXT"
    elif fault == "schema": value["schema_version"] = "two_plane_context_v3"
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
    actual = modules.context.propose_two_plane_context_v4
    proposer_calls, prefilter_calls = [], []
    def tampered(*args, **kwargs):
        result = actual(*args, **kwargs)
        assert result["status"] == "COMPLETE", result["reasons"]
        proposer_calls.append(result)
        if fault == "schema": result["schema_version"] = "two_plane_context_v3"
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
    monkeypatch.setattr(modules.context, "propose_two_plane_context_v4", tampered)
    monkeypatch.setattr(modules.init, "_prefilter", forbidden)
    arrays, _ = _arrays()
    answer = modules.init.prefilter_plane_initializations_v16(*arrays, **_prefilter_args(modules))
    assert len(proposer_calls) == 1 and not prefilter_calls
    _assert_prefilter_refusal(answer)


@pytest.mark.parametrize("site", ("context_budget", "late_finite_query"))
def test_actual_initializer_whole_refusal_keeps_no_usable_start_prefix(modules, monkeypatch, site):
    arrays, _ = _arrays()
    options = _prefilter_args(modules)
    calls = []
    if site == "context_budget":
        options["two_plane_context_budgets"] = modules.context.TwoPlaneContextBudgetsV4(max_extrema_indexes=2)
    else:
        actual = modules.init._FaceQueries.__call__
        def broken(*args, **kwargs):
            calls.append(True)
            if len(calls) == 3:
                raise ValueError("INVENTED_THIRD_FINITE_QUERY_FAILURE")
            return actual(*args, **kwargs)
        monkeypatch.setattr(modules.init._FaceQueries, "__call__", broken)
    answer = modules.init.prefilter_plane_initializations_v16(*arrays, **options)
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
        registration_implementation=REGISTRATION_ID,
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
    # V17: the native factory double must return an exact kernel-V1 feature (the real bridge rebuilds it as
    # kernel V2 before the mocked cover receives it); the object is never passed to real geometry.
    state.feature = m.kernel1.CADGuidedFeature(feature_id="fixture:window:0", nominal_radius_mm=1., nominal_sweep_deg=90.,
        origin=(0., 0., 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.), span_mm=(0., 4.), flange_window_mm=(2.5, 12.5),
        search_margin_mm=3., max_fit_residual_mm=.05, plane_exclusion_mm=.075, max_tangency_error_mm=.1,
        nominal_source="FIXTURE_NOT_NATIVE")
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
    monkeypatch.setattr(m.init, "register_measurement_contexts_v16", register)
    monkeypatch.setattr(m.workflow, "_registration_binding", binding)
    monkeypatch.setattr(m.cover, "inspect_support_cover_context_windows_v5", measure)
    monkeypatch.setattr(m.cover, "validate_support_cover_publication_v5", validate)
    def run(**overrides):
        options = dict(scan_bytes=SCAN_BYTES, cad_bytes=CAD_BYTES, position_error_bound_mm=.025,
            workflow_policy=m.workflow.MeshWorkflowPolicyV18(max_registration_scan_rows=36, cad_sample_count=36),
            registration_policy=_policy(m), measurement_policy=m.native.NativeGuidedPolicy(
                flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.05,
                plane_exclusion_mm=.075, max_tangency_error_mm=.1, fit_model="free_circle"),
            instance_policy=m.cover._v1.SupportPolicy(4., 5., 20., 3), allow_ocp=True,
            connected_measurement_contexts=True, two_plane_context_mode="v4")
        options.update(overrides)
        return m.workflow.inspect_cad_guided_mesh_v18(**options)
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
    assert kwargs["two_plane_context_mode"] == "v4"
    assert type(kwargs["two_plane_context_budgets"]) is state.m.context.TwoPlaneContextBudgetsV4
    assert answer["registration_implementation"] == REGISTRATION_ID
    expected_sha = _sha(_canonical(_expected_configuration(state.m)))
    for publication in (answer, answer["policies"], state.registration["evidence"],
                        state.registration["evidence"]["initialization"]):
        assert publication["initialization_implementation"] == INIT_ID
        assert publication["two_plane_context_configuration_sha256"] == expected_sha
    assert len(answer["registration_array_binding"]) == 3
    for binding in answer["registration_array_binding"]:
        assert binding["initialization_implementation"] == INIT_ID
        assert binding["two_plane_context_mode"] == "v4"
        assert binding["two_plane_context_implementation"] == CONTEXT_ID
        assert binding["registration_implementation"] == REGISTRATION_ID
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
def test_empty_pair_schedule_does_not_bypass_mandatory_v4_publication(plumbing, location, fault):
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
        monkeypatch.setattr(state.m.cover, "inspect_support_cover_context_windows_v5", broken)
    elif site == "validator":
        monkeypatch.setattr(state.m.cover, "validate_support_cover_publication_v5", broken)
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


@pytest.mark.parametrize("mode", (None, "v1", "v2", "v3", False))
def test_workflow_invalid_context_mode_never_reaches_native_or_array_decoding(plumbing, mode):
    answer = plumbing.run(two_plane_context_mode=mode)
    _assert_workflow_refusal(answer)
    assert plumbing.events == []


def test_context_capacity_is_not_misidentified_as_v2_scheduler_marker(modules):
    registration = dict(evidence=dict(initialization=dict(two_plane_schedule=dict(status="COMPLETE"),
        two_plane_context=dict(schema_version="two_plane_context_v4", scheduling_capacity=dict(maximum_ordered_matches=256)))))
    assert modules.workflow._two_plane_schedule_binding(registration, None) is None
    registration["evidence"]["initialization"]["two_plane_schedule"]["schema_version"] = "two_plane_schedule_v2"
    with pytest.raises(ValueError, match="UNREQUESTED_V2_TWO_PLANE_SCHEDULE_PUBLICATION"):
        modules.workflow._two_plane_schedule_binding(registration, None)


# The former v2-schedule variant (stale legacy patch-evidence fixture, known
# terminal failure in the V2 selection) is superseded by the actual-codec
# fixture in tests/test_cad_guided_context_v4_schedule_fixture_v1.py.


@pytest.mark.parametrize("kind", ("dict", "boolean", "v1", "v2"))
def test_context_budget_selector_rejects_untyped_or_legacy_profiles(modules, monkeypatch, kind):
    if kind == "dict": budget = asdict(modules.context.TwoPlaneContextBudgetsV4())
    elif kind == "boolean": budget = True
    elif kind == "v1": budget = modules.context._v1.TwoPlaneContextBudgetsV1()
    else: budget = modules.context._v2.TwoPlaneContextBudgetsV2()
    # The earlier fixture policy could raise before this selector. Establish a
    # valid complete configuration first, then require the real budget boundary.
    valid = _configuration(modules)
    assert valid["two_plane_context_mode"] == "v4"
    actual_selector = modules.context._configuration
    reached = []
    def selector(value):
        reached.append(value)
        return actual_selector(value)
    monkeypatch.setattr(modules.context, "_configuration", selector)
    with pytest.raises(ValueError, match="^INVALID_BUDGET_TYPE$"):
        _configuration(modules, two_plane_context_budgets=budget)
    assert len(reached) == 1 and reached[0] is budget


def test_direct_selected_binding_requires_explicit_expected_context_config(plumbing):
    state = plumbing
    with pytest.raises(ValueError, match="INVALID_CONTEXT_V4_CONFIGURATION"):
        state.m.workflow._registration_binding(state.registration, state.scan, state.topology,
            state.cad, state.scan["engine_to_raw_row_indices"], .025, context_index=0)
    assert state.measurement_calls == state.validator_calls == []


@pytest.mark.parametrize("fault", ("missing", "stale_v3_label", "non_string"))
def test_missing_or_wrong_registration_implementation_is_refused_by_workflow_binding(plumbing, fault):
    state = plumbing
    def damage(registration):
        evidence = registration["evidence"]
        if fault == "missing":
            evidence.pop("registration_implementation")
        elif fault == "stale_v3_label":
            evidence["registration_implementation"] = "SURFACE_V3_NOT_V4"
        else:
            evidence["registration_implementation"] = True
    state.damage = damage
    answer = state.run()
    _assert_workflow_refusal(answer)
    assert state.bindings == [0]
    assert state.measurement_calls == state.validator_calls == []
    assert answer["registration_array_binding"] == []


def test_workflow_derives_and_passes_the_measurement_context_cap(plumbing):
    state = plumbing
    answer = state.run()
    assert answer["status"] == "PARTIAL", answer
    _, kwargs = state.registration_calls[0]
    # One mocked native feature, 48 raw rows: the policy maximum (8) binds.
    assert kwargs["max_measurement_contexts"] == 8
    cap = answer["measurement_context_cap"]
    assert cap["value"] == 8 and cap["feature_count"] == 1 and cap["raw_rows"] == 48
    assert cap["window_budget"] == 4096 and cap["crop_row_budget"] == 64_000_000
    assert cap["schedule"] == state.m.init.MEASUREMENT_CONTEXT_SCHEDULE
    state.registration_calls.clear()
    answer = state.run(workflow_policy=state.m.workflow.MeshWorkflowPolicyV18(
        max_registration_scan_rows=36, cad_sample_count=36, max_measurement_contexts=2))
    assert answer["status"] == "PARTIAL"
    assert state.registration_calls[0][1]["max_measurement_contexts"] == 2
    assert answer["measurement_context_cap"]["value"] == 2
    for bad in (0, 49, True, 2.):
        with pytest.raises(ValueError, match="INVALID_MAX_MEASUREMENT_CONTEXTS"):
            state.m.workflow.MeshWorkflowPolicyV18(max_registration_scan_rows=36, cad_sample_count=36,
                max_measurement_contexts=bad)


@pytest.mark.parametrize("branch", ("window_budget", "crop_budget", "floor"))
def test_measurement_context_cap_budget_branches_and_floor(plumbing, monkeypatch, branch):
    state, m = plumbing, plumbing.m
    consumer = importlib.import_module("backend.cad_guided_context_measurement_v4")
    if branch == "window_budget":
        # 64 mocked features (the bridge/cover boundary; distinct ids) under a 384-window cap: 384 // 64 = 6
        # binds below the policy maximum. 600 identical features would now be refused by the bridge before work.
        bridge = importlib.import_module("backend.cad_guided_native_feature_bridge_v2")
        sixty_four = tuple(type(state.feature)(**dict(asdict(state.feature), feature_id="native:%d:window:0" % i))
                           for i in range(bridge.MAX_BRIDGED_FEATURES))
        monkeypatch.setattr(m.native1, "extract_ocp_guided_features_v1",
            lambda shape, **kwargs: dict(status="PARTIAL", features=sixty_four, metadata={}))
        monkeypatch.setattr(consumer, "MAX_CONTEXT_WINDOWS", 384)
        expected = 6
    elif branch == "crop_budget":
        monkeypatch.setattr(consumer, "MAX_CROP_ROW_VISITS", 100)
        expected = 2  # 100 // (1 feature * 48 raw rows) = 2
    else:
        monkeypatch.setattr(consumer, "MAX_CROP_ROW_VISITS", 40)
        expected = 1  # 40 // 48 = 0 -> floor 1
    answer = state.run()
    assert answer["status"] == "PARTIAL", answer.get("reasons")
    assert answer["measurement_context_cap"]["value"] == expected
    assert state.registration_calls[0][1]["max_measurement_contexts"] == expected


def _completion_registration(m, scan, topology, cad, *, mixed):
    """The main-only fixture with group 0 re-originated from the completion lane: completion-only
    (its former main hypothesis becomes the REJECTED two-of-three parent) or mixed (a fourth
    hypothesis is the parent and the child's exact pose merges into the main group, as V15 does
    when a completion child's pose key equals an existing main group's key)."""
    registration = _registration(m, scan, topology, cad)
    init = registration["evidence"]["initialization"]
    hyps = init["hypotheses"]
    axis, shift = [0., 0., 1.], 2.
    pose_t = list(hyps[0]["translation_mm"])
    parent_t = [pose_t[0], pose_t[1], pose_t[2]-shift]
    matches = copy.deepcopy(hyps[0]["correspondences"])
    parent_index = len(hyps) if mixed else 0
    parent = dict(hypothesis_index=parent_index, unsigned_assignment_id=parent_index, proposal_lane="TRIPLET",
        status="PROPOSED", prefilter_status="REJECTED", prefilter_reasons=["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"],
        exact_start_group_index=None, correspondences=copy.deepcopy(matches), normal_signs=[1, 1, 1],
        assigned_support=[dict(item, status="PASSED" if i < 2 else "REJECTED") for i, item in enumerate(matches)],
        rotation=np.eye(3).tolist(), translation_mm=parent_t)
    child_matches = copy.deepcopy(matches)
    child_matches[2] = dict(scan_plane_id=2, cad_plane_id=41, normal_sign=1)
    child = dict(completion_index=0, parent_hypothesis_index=parent_index, proposal_lane=m.init.COMPLETION_LANE,
        prefilter_status="PASSED", prefilter_reasons=[], exact_start_group_index=0, correspondences=child_matches,
        assigned_support=[dict(item, status="PASSED") for item in child_matches], normal_signs=[1, 1, 1],
        rotation=np.eye(3).tolist(), translation_mm=list(pose_t), replaced_cad_plane_id=37, completed_cad_plane_id=41,
        completed_normal_sign=1, shift_mm=shift, axis_direction=list(axis), status="PROPOSED")
    if mixed:
        hyps.append(parent)
        init["joint_proposal_accounting"].update(triplet_records=4, triplet_unsigned_reservation=4, total_hypothesis_records=4)
    else:
        hyps[0] = parent
    init["completion_lane"] = dict(status="COMPLETE", usable=True, implementation=m.init.COMPLETION_IMPLEMENTATION,
        hypotheses=[child], parents=[dict(parent_hypothesis_index=parent_index, status="SCHEDULED",
            failed_support_position=2, failed_cad_plane_id=37, failed_scan_plane_id=2, axis_direction=list(axis),
            completion_indices=[0])])
    for group in (init["start_groups"][0], init["all_passing_start_groups"][0], registration["evidence"]["candidates"][0]):
        group["completion_indices"] = [0]
        if not mixed:
            group["hypothesis_indices"] = []
            group["lane"] = m.init.COMPLETION_LANE
    return registration


def _bind(state):
    return state.m.workflow._registration_binding(state.registration, state.scan, state.topology, state.cad,
        state.scan["engine_to_raw_row_indices"], .025, context_index=0,
        expected_two_plane_context_configuration=_expected_configuration(state.m))


@pytest.mark.parametrize("mixed", [False, True])
def test_completion_origins_bind_through_the_full_v16_binding_and_dispatch_to_measurement(plumbing, mixed):
    state = plumbing
    state.registration = _completion_registration(state.m, state.scan, state.topology, state.cad, mixed=mixed)
    binding = _bind(state)
    assert binding["start_origin_lanes"] == (["MAIN", "TRIPLET_COMPLETION"] if mixed else ["TRIPLET_COMPLETION"])
    state.bindings.clear()
    answer = state.run()
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", answer
    assert state.bindings == [0, 1, 2] and len(state.measurement_calls) == len(state.validator_calls) == 1
    assert answer["registration_array_binding"][0]["start_origin_lanes"] == binding["start_origin_lanes"]
    assert answer["registration_array_binding"][1]["start_origin_lanes"] == ["MAIN"]
    assert answer["predictions"] is state.measured["predictions"]
    group = state.registration["evidence"]["initialization"]["all_passing_start_groups"][0]
    assert group["completion_indices"] == [0] and group["hypothesis_indices"] == ([0] if mixed else [])


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fault", ["forged_main_origin", "missing_completion_index", "child_not_passed",
                                   "parent_slot_unmarked", "lane_unusable", "parent_is_a_passing_hypothesis",
                                   "replacement_names_passing_face"])
def test_completion_origin_faults_refuse_the_full_binding_and_the_workflow(plumbing, mixed, fault):
    state = plumbing
    state.registration = _completion_registration(state.m, state.scan, state.topology, state.cad, mixed=mixed)
    init = state.registration["evidence"]["initialization"]
    lane = init["completion_lane"]
    parent_index = lane["parents"][0]["parent_hypothesis_index"]
    if fault == "forged_main_origin":
        if mixed:
            pytest.skip("a mixed group legitimately carries a main origin")
        for group in (init["start_groups"][0], init["all_passing_start_groups"][0], state.registration["evidence"]["candidates"][0]):
            group["hypothesis_indices"] = [0]          # hypothesis 0 is the REJECTED parent, never a passing start
        reason = "SELECTED_START_NOT_BOUND_TO_PASSED_LANE_SUPPORT"
    elif fault == "missing_completion_index":
        for group in (init["start_groups"][0], init["all_passing_start_groups"][0], state.registration["evidence"]["candidates"][0]):
            group["completion_indices"] = []
        reason = "SELECTED_GROUP_COMPLETION_ORIGIN_LEDGER_INCOMPLETE" if mixed else "MISSING_SELECTED_START_SUPPORT_ORIGINS"
    elif fault == "child_not_passed":
        lane["hypotheses"][0]["prefilter_status"] = "REJECTED"
        reason = "SELECTED_GROUP_COMPLETION_ORIGIN_LEDGER_INCOMPLETE"   # a REJECTED child cannot be an origin: the lane ledger refuses first
    elif fault == "parent_slot_unmarked":
        init["hypotheses"][parent_index]["assigned_support"][2]["status"] = "PASSED"
        reason = "COMPLETION_PARENT_NOT_A_COMPLETED_TWO_OF_THREE_TRIPLET"
    elif fault == "lane_unusable":
        lane["usable"] = False
        reason = "INCOMPLETE_COMPLETION_LANE_PUBLICATION"
    elif fault == "parent_is_a_passing_hypothesis":
        lane["hypotheses"][0]["parent_hypothesis_index"] = 1
        lane["parents"][0]["parent_hypothesis_index"] = 1
        reason = "COMPLETION_PARENT_NOT_A_COMPLETED_TWO_OF_THREE_TRIPLET"
    else:
        lane["hypotheses"][0]["replaced_cad_plane_id"] = 13
        reason = "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT"
    with pytest.raises(ValueError, match=reason):
        _bind(state)
    answer = state.run()
    _assert_workflow_refusal(answer)
    assert len(state.measurement_calls) == 0


def test_runner_v10_constructor_policies_reach_the_v16_registration_seam_with_the_allowance_and_checkpoint_printer(plumbing, monkeypatch, capsys):
    """The V10 runner's constructor seam (workflow_entry_and_policies) drives the REAL V17 workflow on the
    plumbing scene: the allowance object and the bounded checkpoint printer must reach the initializer-V16
    registration entry unchanged (no handcrafted kwargs), and the printer must emit bounded stdout lines."""
    import importlib.util
    state, m = plumbing, plumbing.m
    spec = importlib.util.spec_from_file_location("_v11_runner_seam", ROOT/"scripts/run_gold11_mesh_development_v11.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    entry, policies = runner.workflow_entry_and_policies("control")
    assert entry is m.workflow.inspect_cad_guided_mesh_v18 and policies["refinement_allowance"].enabled is True
    printer = runner.stage_checkpoint_printer()
    policies["stage_checkpoint"] = printer
    # The plumbing registration double is bound to the scene's .025 mm error bound and default context budgets;
    # rebind those two scene-specific values (declared here) so the runner's other constructors drive the seam.
    policies["position_error_bound_mm"] = .025
    from dataclasses import replace as _replace
    policies["workflow_policy"] = _replace(policies["workflow_policy"], cad_sample_count=36)   # the scene's CAD double has 36 rows
    published = _publication(m, policies["two_plane_context_budgets"])
    state.registration["evidence"].update(published)
    state.registration["evidence"]["initialization"].update(published)
    def prepare(**kwargs):
        assert kwargs == dict(scan_bytes=SCAN_BYTES, max_engine_rows=runner.CONFIG["workflow"]["max_registration_scan_rows"])
        return state.scan
    monkeypatch.setattr(m.mesh, "prepare_mesh_scan_v1", prepare)
    def register(*args, **kwargs):
        state.registration_calls.append((args, kwargs))
        kwargs["stage_checkpoint"](dict(stage="MOCK_REGISTRATION_CHECKPOINT", candidates=3))
        return state.registration, state.roles
    monkeypatch.setattr(m.init, "register_measurement_contexts_v16", register)
    answer = m.workflow.inspect_cad_guided_mesh_v18(scan_bytes=SCAN_BYTES, cad_bytes=CAD_BYTES, **policies)
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", answer.get("reasons")
    args, kwargs = state.registration_calls[-1]
    assert kwargs["refinement_allowance"] is policies["refinement_allowance"] and kwargs["stage_checkpoint"] is printer
    assert asdict(kwargs["refinement_allowance"]) == runner.CONFIG["refinement_allowance"]
    assert kwargs["clock"] is None and kwargs["two_plane_context_mode"] == "v4"
    assert type(kwargs["completion_policy"]).__name__ == "CompletionLanePolicyV1" and kwargs["max_measurement_contexts"] == 8
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line) == dict(status="STAGE_CHECKPOINT", checkpoint=dict(stage="MOCK_REGISTRATION_CHECKPOINT", candidates=3))
    assert answer["implementation"] == runner.ARM_DEFINITIONS["control"]["implementation"]
