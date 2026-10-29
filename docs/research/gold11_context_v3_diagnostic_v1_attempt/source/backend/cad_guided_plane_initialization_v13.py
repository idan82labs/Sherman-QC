"""NEW unqualified context-V3 initializer adapter; frozen V12 is unchanged.

The V12 registration and prefilter schema names are retained wire contracts for
the unchanged measurement consumer, not historical implementation identities.
Every usable publication additionally binds INITIALIZATION_IMPLEMENTATION and
the explicit context-V3 configuration. No module/global rewriting or injected
caller callable is used. Numeric prefilter/refinement/support helpers are reused.
"""

from dataclasses import asdict
import hashlib
import json

import numpy as np

from . import cad_guided_plane_initialization_v12 as _frozen
from . import cad_guided_two_plane_context_v3 as _context_v3

# Explicitly reused frozen numerical types/helpers, not copied implementations.
_old, _previous, _v3, _surface = _frozen._old, _frozen._previous, _frozen._v3, _frozen._surface
_noisy, _patch, _search, _stages = _frozen._noisy, _frozen._patch, _frozen._search, _frozen._stages
_v1, _context, _pair_schedule = _frozen._v1, _frozen._context, _frozen._pair_schedule
_require, _digest, _pose = _frozen._require, _frozen._digest, _frozen._pose
_LIMITS = _frozen._LIMITS
PlaneInitializationBudgetsV12 = _frozen.PlaneInitializationBudgetsV12
_bounded_report = _frozen._bounded_report
_prepare_triplets, _bound_scan = _frozen._prepare_triplets, _frozen._bound_scan
_prefilter, _registration_rows = _frozen._prefilter, _frozen._registration_rows
_context_support, _publish_contexts = _frozen._context_support, _frozen._publish_contexts
_schedule_equation_digests = _frozen._schedule_equation_digests
_validate_two_plane_schedule_v2 = _frozen._validate_two_plane_schedule_v2
_bind_initialization_schedule_configuration = _frozen._bind_initialization_schedule_configuration

INITIALIZATION_IMPLEMENTATION = "CAD_GUIDED_PLANE_INITIALIZATION_V13_CONTEXT_V3"
CONTEXT_IMPLEMENTATION = "RAW_EXTREMA_CONTEXT_V3_STAGED_SCAN_PROJECTION"
_CONTEXT_DECLARATIONS = (
    "scan_projection_arithmetic", "scan_projection_bound_scope", "cad_projection_arithmetic",
    "scan_projection_chunk_scope", "chunk_rows_scope", "extrema_guard_status",
    "extrema_membership_scope", "extrema_memory_scope",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _context_configuration(mode, budgets):
    """Closed V3 selector; scheduling and scan proposals remain independent."""
    _require(type(mode) is str and mode == "v3", "EXPLICIT_V3_CONTEXT_MODE_REQUIRED")
    owned = _context_v3.TwoPlaneContextBudgetsV3(**_context_v3._configuration(
        _context_v3.TwoPlaneContextBudgetsV3() if budgets is None else budgets))
    base = _context_v3._base()
    declaration = dict(mode="v3", implementation=CONTEXT_IMPLEMENTATION,
        schema_version="two_plane_context_v3", budgets=asdict(owned),
        **{key: base[key] for key in _CONTEXT_DECLARATIONS})
    return owned, declaration


def _context_configuration_snapshot(value):
    """Validate before encoding and return a fresh, bounded exact snapshot."""
    keys = {"mode", "implementation", "schema_version", "budgets", *_CONTEXT_DECLARATIONS}
    _require(type(value) is dict and set(value) == keys and type(value.get("budgets")) is dict,
        "INVALID_CONTEXT_V3_CONFIGURATION")
    cap = value["budgets"]
    expected_keys = {key for key, _ in _context_v3._MAXIMA}
    _require(set(cap) == expected_keys and all(type(item) is int for item in cap.values()),
        "INVALID_CONTEXT_V3_BUDGET_SNAPSHOT")
    _, expected = _context_configuration(value["mode"], _context_v3.TwoPlaneContextBudgetsV3(**cap))
    _require(all(type(value[key]) is type(expected[key]) and value[key] == expected[key]
        for key in keys-{"budgets"}), "CONTEXT_V3_IMPLEMENTATION_OR_ARITHMETIC_MISMATCH")
    return expected, hashlib.sha256(_canonical(expected)).hexdigest()


def _context_publication(configuration):
    snapshot, sha = _context_configuration_snapshot(configuration["two_plane_context_configuration"])
    return dict(initialization_implementation=INITIALIZATION_IMPLEMENTATION,
        two_plane_context_configuration=snapshot, two_plane_context_configuration_sha256=sha)


def _bind_context_publication(publication, expected_configuration):
    """A returned string/digest is evidence to check, never an implementation selector."""
    expected, sha = _context_configuration_snapshot(expected_configuration)
    _require(type(publication) is dict
        and publication.get("initialization_implementation") == INITIALIZATION_IMPLEMENTATION,
        "INITIALIZATION_V13_IMPLEMENTATION_REQUIRED")
    actual, actual_sha = _context_configuration_snapshot(publication.get("two_plane_context_configuration"))
    _require(actual == expected and actual_sha == sha
        and type(publication.get("two_plane_context_configuration_sha256")) is str
        and publication["two_plane_context_configuration_sha256"] == sha,
        "CONTEXT_V3_REQUESTED_RETURNED_CONFIGURATION_MISMATCH")
    return dict(initialization_implementation=INITIALIZATION_IMPLEMENTATION,
        two_plane_context_mode="v3", two_plane_context_implementation=CONTEXT_IMPLEMENTATION,
        two_plane_context_configuration_sha256=sha)


def _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled=True,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v3", two_plane_context_budgets=None):
    _require(two_plane_enabled is True, "CONTEXT_V3_REQUIRES_ENABLED_TWO_PLANE_LANE")
    configuration = _frozen._configuration(registration_policy, noisy_policy, noisy_budgets,
        cad_patch_policy, cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets,
        two_plane_enabled, scan_plane_mode, seeded_plane_budgets, composite_plane_budgets,
        two_plane_schedule_mode, two_plane_schedule_budgets)
    owned, declaration = _context_configuration(two_plane_context_mode, two_plane_context_budgets)
    configuration.update(two_plane_context_mode="v3", two_plane_context_budgets=owned,
        two_plane_context_configuration=declaration)
    return configuration


def _base():
    result = _frozen._base()
    result["initialization_implementation"] = INITIALIZATION_IMPLEMENTATION
    return result


def _validate_two_plane_context_configuration(helper, requested):
    expected, _ = _context_configuration_snapshot(requested)
    _require(type(helper) is dict and helper.get("schema_version") == expected["schema_version"],
        "TWO_PLANE_CONTEXT_V3_SCHEMA_MISMATCH")
    cap = helper.get("budgets")
    _require(type(cap) is dict and set(cap) == set(expected["budgets"])
        and all(type(cap[key]) is int and cap[key] == value for key, value in expected["budgets"].items()),
        "TWO_PLANE_CONTEXT_V3_BUDGET_PROFILE_MISMATCH")
    for key in _CONTEXT_DECLARATIONS:
        _require(type(helper.get(key)) is type(expected[key]) and helper[key] == expected[key],
            "TWO_PLANE_CONTEXT_V3_ARITHMETIC_OR_SCOPE_MISMATCH")
    capacity = _context_v3._base()["scheduling_capacity"]
    actual = helper.get("scheduling_capacity")
    _require(type(actual) is dict and set(actual) == set(capacity)
        and all(type(actual[key]) is type(value) and actual[key] == value for key, value in capacity.items()),
        "TWO_PLANE_CONTEXT_V3_CAPACITY_MISMATCH")
    _require(helper.get("production_use_allowed") is False, "CONTEXT_V3_NOT_PRODUCTION_AUTHORITY")


def _two_plane_records(context, result, configuration):
    """Bind owned plane declarations, then retain every scheduled axial slot."""
    hypotheses = context["hypotheses"]
    for row in hypotheses:
        row["proposal_lane"] = "TRIPLET"
    h3 = len(hypotheses)
    unsigned = {row["unsigned_assignment_id"] for row in hypotheses}
    _require(all(type(x) is int and x >= 0 for x in unsigned),
             "INVALID_TRIPLET_UNSIGNED_NAMESPACE")
    u3 = max(unsigned, default=-1)+1
    budget = configuration["budgets"]
    _require(h3 <= budget.max_hypothesis_records, "TRIPLET_JOINT_PREFLIGHT_EXCEEDED")
    if not configuration["two_plane_enabled"]:
        result["two_plane_schedule"] = dict(status="DISABLED_BY_EXPLICIT_POLICY",
            scheduled_matches=[], selected_policy_evaluation_complete=True)
        result["joint_proposal_accounting"] = dict(triplet_records=h3,
            triplet_unsigned_reservation=u3, ordered_pair_matches=0,
            pair_slot_reservation=0, total_hypothesis_records=h3,
            pair_lane_enabled=False)
        return
    _require(u3 <= budget.max_prefilter_unsigned_assignments,
             "TRIPLET_JOINT_UNSIGNED_RESERVATION_EXCEEDED")
    bank = result["hypothesis_evidence"]["scan_bank"]
    _require(type(bank) is list and len(bank) <= 128 and len(set(bank)) == len(bank)
        and all(type(i) is int and i in context["scan_supports"] for i in bank),
        "INVALID_OWNED_TWO_PLANE_BANK")
    scan_planes, cad_planes, scan_equations, cad_equations = [], [], [], []
    for identifier in bank:
        patch = context["scan_supports"][identifier]
        _bound_scan(context, patch)
        analysis = patch["analysis"]
        _require(analysis["status"] == "PROPOSED", "UNQUALIFIED_TWO_PLANE_SCAN")
        normal = tuple(analysis["normal"])
        anchor = tuple(analysis["witness_anchor_mm"])
        offset, alpha = analysis["witness_offset_from_anchor_mm"], analysis["normal_angle_bound_radians"]
        scan_equations.append(_search.PlaneEquationV6(identifier, normal, anchor, offset, alpha))
        scan_planes.append(_context.ContextPlaneV1(identifier, normal, anchor, offset, alpha, True,
            patch["triangle_rows"], patch["triangle_rows_sha256"],
            patch["raw_vertex_rows"], patch["raw_vertex_rows_sha256"]))
    for patch in context["cad_patches"]["patches"]:
        _old._bound_support(context, cad_patch=patch)
        identifier, normal, anchor = patch["plane_id"], tuple(patch["normal"]), tuple(patch["area_centroid_mm"])
        cad_equations.append(_search.PlaneEquationV6(identifier, normal, anchor, 0., 0.))
        cad_planes.append(_context.ContextPlaneV1(identifier, normal, anchor, 0., 0., True,
            patch["triangle_rows"], patch["triangle_rows_sha256"]))
    if configuration.get("two_plane_schedule_mode", "v1") == "v2":
        from . import cad_guided_two_plane_schedule_v2 as schedule_v2
        expected_digests = _schedule_equation_digests(tuple(scan_equations), tuple(cad_equations))
        schedule = schedule_v2.schedule_two_plane_pairs_v2(
            tuple(scan_equations), tuple(cad_equations),
            retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3,
            max_joint_hypothesis_records=budget.max_hypothesis_records,
            max_joint_unsigned_assignments=budget.max_prefilter_unsigned_assignments,
            budgets=configuration["two_plane_schedule_budgets"])
        result["two_plane_schedule"] = schedule
        _validate_two_plane_schedule_v2(schedule, configuration["two_plane_schedule_configuration"],
            expected_digests, tuple(row.plane_id for row in scan_equations),
            tuple(row.plane_id for row in cad_equations), h3, u3,
            budget.max_hypothesis_records, budget.max_prefilter_unsigned_assignments)
    else:
        schedule = _pair_schedule.schedule_two_plane_pairs_v1(
            tuple(scan_equations), tuple(cad_equations),
            retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3,
            max_joint_hypothesis_records=budget.max_hypothesis_records,
            max_joint_unsigned_assignments=budget.max_prefilter_unsigned_assignments)
        _require(type(schedule) is not dict or (schedule.get("schema_version") != "two_plane_schedule_v2"
            and not any(key in schedule for key in ("breadth_ledger", "residual_windows", "residual_statistics"))),
                 "UNREQUESTED_V2_SCHEDULE_PUBLICATION")
    result["two_plane_schedule"] = schedule
    _require(type(schedule) is dict and schedule.get("status") == "COMPLETE"
        and schedule.get("selected_policy_evaluation_complete") is True,
        "TWO_PLANE_SCHEDULE_UNAVAILABLE")
    matches = schedule["scheduled_matches"]
    _require(type(matches) is list and len(matches) <= 256
        and h3+32*len(matches) <= budget.max_hypothesis_records
        and u3+len(matches) <= budget.max_prefilter_unsigned_assignments,
        "TWO_PLANE_JOINT_PREFLIGHT_EXCEEDED")
    result["joint_proposal_accounting"] = dict(triplet_records=h3,
        triplet_unsigned_reservation=u3, ordered_pair_matches=len(matches),
        pair_slot_reservation=32*len(matches),
        total_hypothesis_records=h3+32*len(matches), pair_lane_enabled=True)
    if not matches:
        result["two_plane_context"] = dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE",
            usable_proposals=False, accepted_registration=False)
        return
    ordered = tuple(_context.OrderedPlanePairV1(
        tuple(row["scan_plane_ids"]), tuple(row["cad_plane_ids"])) for row in matches)
    context_options = dict(budgets=configuration["two_plane_context_budgets"])
    helper = _context_v3.propose_two_plane_context_v3(
        context["scan"], context["raw_triangles"], context["cad"], context["faces"], context["local"],
        tuple(scan_planes), tuple(cad_planes), ordered,
        scan_origin_mm=(0., 0., 0.), cad_origin_mm=(0., 0., 0.),
        unit_assumption=result["hypothesis_evidence"]["unit_assumption"],
        source_frame="OWNED_SCAN_RAW_MM", target_frame="OWNED_CAD_MM", **context_options)
    result["two_plane_context"] = helper
    _require(type(helper) is dict and helper.get("status") == "COMPLETE"
        and helper.get("selected_policy_evaluation_complete") is True
        and helper.get("accepted_registration") is False
        and helper.get("finite_support_match_verified") is False,
        "TWO_PLANE_CONTEXT_UNAVAILABLE")
    _validate_two_plane_context_configuration(helper, configuration["two_plane_context_configuration"])
    _require(helper.get("unit") == "mm"
        and helper.get("unit_assumption") == result["hypothesis_evidence"]["unit_assumption"]
        and helper.get("source_frame") == "OWNED_SCAN_RAW_MM"
        and helper.get("target_frame") == "OWNED_CAD_MM"
        and all(type(helper.get(key)) is list and len(helper[key]) == 3
            and all(type(value) is float and value == 0. for value in helper[key])
            for key in ("source_origin_mm", "target_origin_mm")),
        "TWO_PLANE_CONTEXT_FRAME_OR_UNIT_MISMATCH")
    expected_digests = {key+"_sha256": _digest(context[name]) for key, name in (
        ("scan_points", "scan"), ("scan_triangles", "raw_triangles"),
        ("cad_triangles", "cad"), ("cad_face_ids", "faces"), ("cad_source_triangle_ids", "local"))}
    _require(helper["input_digests"] == expected_digests, "STALE_TWO_PLANE_CONTEXT_BUFFERS")
    planned = helper["statistics"]
    _require(planned["signed_families_completed"] == planned["signed_families_planned"] == 4*len(matches)
        and planned["slot_records_completed"] == planned["candidate_slots_planned"] == 32*len(matches)
        and len(helper["families"]) == 4*len(matches), "INCOMPLETE_TWO_PLANE_SLOT_LEDGER")
    canonical = [(tuple(sorted(row.scan_plane_ids)), row.cad_plane_ids if row.scan_plane_ids[0] < row.scan_plane_ids[1]
                  else row.cad_plane_ids[::-1]) for row in ordered]
    _require(helper["scheduled_pairs"] == [dict(scan_plane_ids=list(source), cad_plane_ids=list(target))
        for source, target in canonical], "TWO_PLANE_CONTEXT_SCHEDULE_MISMATCH")
    seen_slots, seen_signs = set(), set()
    for fi, family in enumerate(helper["families"]):
        ai = family["ordered_assignment_id"]
        _require(type(family["signed_family_id"]) is int and family["signed_family_id"] == fi
            and type(ai) is int and 0 <= ai < len(matches),
                 "STALE_TWO_PLANE_FAMILY_ID")
        signs = tuple(family["normal_signs"])
        _require(len(signs) == 2 and all(type(x) is int and x in (-1, 1) for x in signs)
            and (ai, signs) not in seen_signs, "INVALID_TWO_PLANE_SIGN_LEDGER")
        seen_signs.add((ai, signs))
        source, target = canonical[ai]
        correspondences = [dict(scan_plane_id=a, cad_plane_id=b, normal_sign=s)
                           for a, b, s in zip(source, target, signs)]
        _require(family["correspondences"] == correspondences
            and tuple(row["slot"] for row in family["slots"]) == _context._SLOTS,
            "STALE_TWO_PLANE_ASSIGNMENT_OR_SLOT")
        for slot in family["slots"]:
            identity = (fi, slot["slot"])
            _require(identity not in seen_slots, "DUPLICATE_TWO_PLANE_SLOT")
            seen_slots.add(identity)
            status = slot["status"]
            _require(status in ("PROPOSED_SEED_ONLY", "PAIR_REJECTED", "ABSENT_EMPTY_HEURISTIC_INTERVAL"),
                     "INVALID_TWO_PLANE_SLOT_DISPOSITION")
            row = dict(hypothesis_index=len(hypotheses), unsigned_assignment_id=u3+ai,
                proposal_lane="TWO_PLANE_CONTEXT", signed_family_id=fi, axial_slot=slot["slot"],
                context_slot_disposition=status, context_candidate_index=slot["candidate_index"],
                correspondences=correspondences, status="REJECTED",
                rotation=None, translation_mm=None, reasons=[status])
            if status == "PROPOSED_SEED_ONLY":
                ci = slot["candidate_index"]
                _require(type(ci) is int and 0 <= ci < len(helper["candidates"]),
                         "INVALID_TWO_PLANE_CANDIDATE_INDEX")
                candidate = helper["candidates"][ci]
                _require(type(candidate["candidate_index"]) is int and candidate["candidate_index"] == ci
                    and candidate["accepted_registration"] is False
                    and dict(signed_family_id=fi, slot=slot["slot"]) in candidate["contributing_slots"]
                    and family["status"] == "AVAILABLE", "STALE_TWO_PLANE_CANDIDATE_ORIGIN")
                rotation, translation = _pose(candidate)
                model = family["translation_family"]
                expected_t = np.asarray(model["minimum_norm_local_translation_mm"])+slot["lambda_mm"]*np.asarray(model["target_axial_direction"])
                _require(np.array_equal(rotation, np.asarray(family["rotation"]))
                    and np.array_equal(translation, expected_t), "TWO_PLANE_POSE_NOT_BOUND_TO_AXIAL_SLOT")
                row.update(status="PROPOSED", rotation=rotation.tolist(),
                    translation_mm=translation.tolist(), reasons=[])
            else:
                _require(slot["candidate_index"] is None and slot["lambda_mm"] is None,
                         "UNAVAILABLE_TWO_PLANE_SLOT_HAS_POSE")
                _require((status == "PAIR_REJECTED") == (family["status"] != "AVAILABLE"),
                         "INCONSISTENT_TWO_PLANE_SLOT_FAMILY_STATUS")
                if status == "ABSENT_EMPTY_HEURISTIC_INTERVAL":
                    row["status"] = "ABSENT_CONTEXT_SLOT"
            hypotheses.append(row)
    _require(len(hypotheses) == h3+32*len(matches)
        and len(seen_signs) == 4*len(matches), "INCOMPLETE_JOINT_HYPOTHESIS_LEDGER")


def _prepare(scan_points, raw_triangles, cad_triangles, cad_face_ids, cad_source_triangle_ids,
             epsilon, configuration, result, unit_assumption):
    context = _prepare_triplets(scan_points, raw_triangles, cad_triangles, cad_face_ids,
        cad_source_triangle_ids, epsilon, configuration, result, unit_assumption)
    try:
        _two_plane_records(context, result, configuration)
    except Exception:
        # Preserve the already generated triplet/pair ledger if the additive
        # preparation fails before normal prefilter ledger construction.
        result["hypotheses"] = [dict(row, usable_initialization=False,
            prefilter_status="NOT_ATTEMPTED_ADDITIVE_PREPARATION_FAILURE")
            for row in context["hypotheses"]]
        raise
    return context


def _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, epsilon, configuration, unit_assumption):
    result, context = _base(), None
    result.update(_context_publication(configuration))
    result.update(budgets=asdict(configuration["budgets"]), registration_policy=asdict(configuration["policy"]),
        surface_budgets=asdict(configuration["surface_budgets"]))
    if configuration["scan_plane_mode"] == "composite_v1":
        result["scan_plane_configuration"] = dict(configuration["scan_plane_configuration"])
    if configuration.get("two_plane_schedule_mode", "v1") == "v2":
        result["two_plane_schedule_configuration"] = dict(mode="v2",
            budgets=dict(configuration["two_plane_schedule_configuration"]["budgets"]))
    try:
        context = _prepare(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                           cad_source_triangle_ids, epsilon, configuration, result, unit_assumption)
        _prefilter(context, result, configuration)
    except Exception as error:
        _old._fail_prefilter(result, error)
        result["selected_policy_evaluation_complete"] = False
        result["numerical_consumer_complete"] = False
        result["box_screening_complete"] = False
        result["statistics"].update(exact_passing_start_group_count=len(result["all_passing_start_groups"]),
            exact_passing_start_group_count_scope="ATTEMPTED_PREFILTER_BEFORE_FATAL_NOT_SURVIVING",
            omitted_passing_start_group_count=None)
        for group in result["all_passing_start_groups"]:
            group.update(refinement_disposition="INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE",
                attempted_initialized_rotation=group["initialized_rotation"],
                attempted_initialized_translation_mm=group["initialized_translation_mm"],
                initialized_rotation=None, initialized_translation_mm=None)
        context = None
    result = _bounded_report(result, configuration["budgets"].max_report_bytes)
    return result, context if result["status"] == "COMPLETE" else None


def prefilter_plane_initializations_v13(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm, registration_policy,
        unit_assumption, noisy_policy=_noisy.NoisyMeshPlanePolicyV5(),
        noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(), cad_patch_policy=_patch.CoplanarPatchPolicyV4(),
        cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(), search_policy=_search.PlaneSearchPolicyV6(),
        search_budgets=_search.PlaneSearchBudgetsV6(), budgets=PlaneInitializationBudgetsV12(),
        surface_budgets=None, two_plane_enabled=True, scan_plane_mode="v7",
        seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v3", two_plane_context_budgets=None):
    """Owned-array scheduled finite-face evidence; no accepted registration."""
    result, maximum = _base(), _LIMITS["max_report_bytes"]
    try:
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled,
            scan_plane_mode, seeded_plane_budgets, composite_plane_budgets, **schedule_request,
            two_plane_context_mode=two_plane_context_mode, two_plane_context_budgets=two_plane_context_budgets)
        maximum = configuration["budgets"].max_report_bytes
        _require(type(unit_assumption) is str and unit_assumption == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                 "EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED")
        result, _ = _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
            cad_source_triangle_ids, position_error_bounds_mm, configuration, unit_assumption)
    except Exception as error:
        _old._fail_prefilter(result, error)
        result["selected_policy_evaluation_complete"] = False
        result["numerical_consumer_complete"] = False
        result["box_screening_complete"] = False
    return _bounded_report(result, maximum)


def register_whole_part_planes_v13(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm,
        cad_initialization_points, registration_policy, unit_assumption, registration_raw_rows=None,
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        search_policy=_search.PlaneSearchPolicyV6(), search_budgets=_search.PlaneSearchBudgetsV6(),
        budgets=PlaneInitializationBudgetsV12(), surface_budgets=None, registration_budgets=None, two_plane_enabled=True,
        connected_measurement_contexts=False, _return_owned_measurement_support=False,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v3", two_plane_context_budgets=None):
    """Explicit error-bound declaration and owned supports, no native or file IO.

All original assigned support rows reach bound full-face box checks. Nonexcluded
supports reach unchanged full finite queries; boxes never grant support. The global
registration selection still has V2's <=200k limit and is separately disclosed.
The returned fit never establishes global pose uniqueness or a stable datum.
"""
    result = _old._registration_result(unit_assumption)
    result["schema_version"] = "noisy_plane_initialized_finite_registration_v12"
    evidence, query, measurement_support = result["evidence"], None, None
    evidence["initialization_implementation"] = INITIALIZATION_IMPLEMENTATION
    evidence.update(schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12",
        candidate_search_status="ALL_GENERATED_PROPOSED_SIGNS_FINITE_SCREENED_NOT_GLOBAL_SEARCH",
        search_exhaustive=False, global_pose_uniqueness="NOT_ESTABLISHED",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
        stage_policies=_stages.snapshot_stage_policies_v1(_stages.StagePoliciesV1()),
        proposal_support_policy=_stages.PROXIMITY,
        numerical_consumer_complete=False,
        optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        observability_basis="UNAMBIGUOUS_NONUNCERTAIN_FACE_INTERIORS_ACTUAL_TRIM_ENGINEERING_NORMAL_RANK",
        residual_gate="FINITE_DISTANCE_P95_UPPER_USING_UNCHANGED_LEGACY_NUMERIC_THRESHOLD",
        reverse_coverage_status="ENGINEERING_CAD_SAMPLE_TO_SELECTED_SCAN_ROWS_NOT_CERTIFIED_SURFACE_COVERAGE")
    maximum = _LIMITS["max_report_bytes"]
    try:
        _require(type(connected_measurement_contexts) is bool, "INVALID_CONNECTED_CONTEXT_POLICY")
        _require(type(_return_owned_measurement_support) is bool
            and (not _return_owned_measurement_support or connected_measurement_contexts),
            "INVALID_OWNED_MEASUREMENT_SUPPORT_MODE")
        evidence["connected_measurement_contexts"] = connected_measurement_contexts
        evidence["stage_policies"] = _stages.ablation_stage_policies_v1(
            connected=connected_measurement_contexts)
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled,
            scan_plane_mode, seeded_plane_budgets, composite_plane_budgets, **schedule_request,
            two_plane_context_mode=two_plane_context_mode, two_plane_context_budgets=two_plane_context_budgets)
        evidence.update(_context_publication(configuration))
        if configuration["scan_plane_mode"] == "composite_v1":
            evidence["scan_plane_configuration"] = dict(configuration["scan_plane_configuration"])
        if configuration.get("two_plane_schedule_mode", "v1") == "v2":
            evidence["two_plane_schedule_configuration"] = dict(mode="v2",
                budgets=dict(configuration["two_plane_schedule_configuration"]["budgets"]))
        maximum = configuration["budgets"].max_report_bytes
        execution = _v3._validated_execution_budgets(_v3.SurfaceRegistrationBudgetsV3()
            if registration_budgets is None else registration_budgets)
        execution = _v3.SurfaceRegistrationBudgetsV3(**asdict(execution))
        _require(type(unit_assumption) is str and unit_assumption == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                 "EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED")
        full_scan = _old._array(scan_points, (3,), "fiu", configuration["noisy_budgets"].max_raw_rows,
                               "<f8", "SCAN_POINTS")
        scan = _registration_rows(dict(scan=full_scan), registration_raw_rows, evidence)
        cad = _v1._canonical_scan(cad_initialization_points)
        evidence["cad_initialization_points_sha256"] = _digest(_patch._immutable(cad, "<f8"))
        evidence["cad_sample_provenance"] = "CALLER_MUST_BIND_TO_TRIANGLES_NOT_AUTHENTICATED_BY_ARRAY_API"
        initialization, context = _initialize(full_scan, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
            cad_source_triangle_ids, position_error_bounds_mm, configuration, unit_assumption)
        evidence["initialization"] = initialization
        _bind_initialization_schedule_configuration(initialization, configuration)
        _bind_context_publication(initialization, configuration["two_plane_context_configuration"])
        _require(initialization["status"] == "COMPLETE" and initialization["selected_policy_evaluation_complete"]
            and initialization["numerical_consumer_complete"] is True
            and initialization["box_screening_policy"] == "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
            and initialization["box_screening_complete"] is True,
                 "NOISY_PLANE_INITIALIZATION_UNAVAILABLE")
        _require(bool(initialization["start_groups"]), "NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS")
        if connected_measurement_contexts:
            if configuration["scan_plane_mode"] == "composite_v1":
                from .cad_guided_composite_surface_roles_v1 import prepare_composite_surface_roles_v1 as prepare_roles
            else:
                from .cad_guided_context_surface_roles_v1 import prepare_context_surface_roles_v1 as prepare_roles
            measurement_support = prepare_roles(
                context["scan"], context["raw_triangles"], context["scan_patches"],
                initialization["patch_evidence"]["scan"],
                initialization["input_digests"]["position_error_bounds_mm_sha256"])
            evidence["surface_role_preparation"] = dict(
                points_sha256=measurement_support.points_sha256,
                triangle_raw_rows_sha256=measurement_support.triangle_raw_rows_sha256,
                position_error_bounds_mm_sha256=measurement_support.position_error_bounds_mm_sha256,
                metadata_sha256=measurement_support.metadata_sha256,
                raw_row_count=measurement_support.raw_row_count,
                scope="OWNED_SCAN_PLANE_SUPPORTS_NOT_PHYSICAL_SURFACE_CLASSIFICATION")
        candidates = evidence["candidates"]
        for group in initialization["start_groups"]:
            candidates.append(dict(group, execution_status="NOT_ATTEMPTED", reasons=["NOT_ATTEMPTED_YET"]))
        policy, surface = configuration["policy"], configuration["surface_budgets"]
        evidence.update(policy=asdict(policy), surface_query_budgets=asdict(surface),
            registration_budgets=asdict(execution), input_digests=dict(initialization["input_digests"]))
        coarse = _v1._spatial_sample(scan, policy.coarse_points)
        refine = _v1._spatial_sample(scan, policy.refine_points)
        built = _surface.build_triangle_surface(context["cad"], context["faces"], budgets=surface)
        evidence["surface_index"] = {key: value for key, value in built.items() if key != "index"}
        _require(built["status"] == "COMPLETE", "FINITE_TRIANGLE_SURFACE_UNAVAILABLE")
        query = _v3._Queries(built["index"], execution, surface)
        diagonal = float(_v3._norm(np.ptp(built["index"].triangles.reshape(-1, 3), axis=0)))
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
            coarse_scan_rows=len(coarse), refined_scan_rows=len(refine))
        for row in candidates:
            row.update(execution_status="ATTEMPTED", reasons=[])
            rotation, translation = np.asarray(row["initialized_rotation"]), np.asarray(row["initialized_translation_mm"])
            try:
                rotation, translation, nc, cr, cd = _v3._refine(coarse, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=True)
                rotation, translation, nf, fr, fd = _v3._refine(refine, query, rotation, translation,
                    diagonal=diagonal, policy=policy, coarse=False)
                _pose(dict(rotation=rotation, translation_mm=translation))
                row.update(_v3._diagnose(scan, cad, refine, query, rotation, translation, policy))
                if connected_measurement_contexts:
                    row["conditional_context_support"] = _context_support(
                        scan, query, rotation, translation, policy, row["cad_coverage"])
                row.update(coarse_iterations=nc, refine_iterations=nf, coarse_convergence_state=cr,
                    refine_convergence_state=fr, coarse_last_step=cd, refine_last_step=fd,
                    rotation=rotation.tolist(), translation=translation.tolist())
            except _v3._SurfaceFailure as error:
                row["reasons"] = [str(error)]
                raise
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                row["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
        evidence["candidate_search_complete"] = True
        evidence["numerical_consumer_complete"] = True
        if connected_measurement_contexts:
            _publish_contexts(result)
        else:
            _v3._select(result, refine, policy)
    except Exception as error:
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[_old._reason(error)])
        evidence.pop("selected_candidate_index", None)
        if "two_plane_context_configuration" in evidence or "two_plane_schedule_configuration" in evidence:
            evidence["attempted_measurement_context_indices"] = evidence.get("measurement_context_indices", [])
            evidence["measurement_context_indices"] = []
        evidence["numerical_consumer_complete"] = False
        evidence.update(candidate_search_complete=False, evidence_scope="ATTEMPTED_WORK_ONLY_NO_ACCEPTED_REGISTRATION")
        for row in evidence["candidates"]:
            if row["execution_status"] == "NOT_ATTEMPTED":
                row["reasons"] = ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
            elif not row.get("reasons"):
                row["reasons"] = ["INVALIDATED_BY_FATAL_REGISTRATION_FAILURE"]
    finally:
        evidence["candidate_count"] = len(evidence["candidates"])
        evidence["attempted_candidate_count"] = sum(row["execution_status"] == "ATTEMPTED" for row in evidence["candidates"])
        evidence["candidate_numerical_failure_count"] = sum("CANDIDATE_NUMERICAL_FAILURE" in row["reasons"]
            for row in evidence["candidates"])
        evidence["candidate_evaluation_complete"] = (evidence["candidate_search_complete"]
            and evidence["attempted_candidate_count"] == evidence["candidate_count"]
            and evidence["candidate_numerical_failure_count"] == 0)
        evidence["numerical_consumer_complete"] = bool(evidence["candidate_evaluation_complete"]
            and all(row.get("numerical_consumer_status") == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
                    for row in evidence["candidates"]))
        if query is not None:
            evidence["query_resources"] = dict(query.statistics)
    if result["status"] in ("CONDITIONAL_BEST_FIT", "CONDITIONAL_MEASUREMENT_CONTEXTS") and not (
            evidence["candidate_evaluation_complete"] and evidence["numerical_consumer_complete"]):
        result.update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=["INCOMPLETE_SELECTED_CANDIDATE_EVALUATION"])
        evidence.pop("selected_candidate_index", None)
        evidence["measurement_context_indices"] = []
    bounded = _bounded_report(result, maximum)
    if bounded.get("status") != "CONDITIONAL_MEASUREMENT_CONTEXTS":
        measurement_support = None
    if _return_owned_measurement_support:
        return bounded, measurement_support
    return bounded


def register_measurement_contexts_v13(*args, **kwargs):
    """Return JSON registration evidence and its owned scan-support transport.

    The opaque second value retains existing extraction's exact support arrays;
    descriptor hashes are never inverted or substituted for original row IDs.
    It is not part of the JSON report and supplies no external pose or labels.
    """
    _require("_return_owned_measurement_support" not in kwargs
        and kwargs.get("connected_measurement_contexts", True) is True,
        "CONNECTED_CONTEXT_API_REQUIRED")
    kwargs["connected_measurement_contexts"] = True
    return register_whole_part_planes_v13(*args, **kwargs, _return_owned_measurement_support=True)
