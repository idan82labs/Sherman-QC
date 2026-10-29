"""Unqualified V12 stage-separated successor; V11 remains frozen.

C1 changes proposal proximity only. C2 retains V3 conservative optimization
and rank while admitting explicitly conditional scan measurement contexts from
independent finite-distance evidence. No context is an accepted/global pose.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_plane_initialization_v4 as _old
from . import cad_guided_plane_initialization_v8 as _previous
from . import cad_guided_global_registration_v3 as _v3
from . import cad_guided_triangle_surface_v3 as _surface
from . import cad_guided_face_box_screen_v1 as _box
from . import cad_guided_noisy_mesh_planes_v5 as _noisy
from . import cad_guided_noisy_mesh_planes_v7 as _compact
from .cad_guided_report_envelope_v6 import bound_initialization_report_v6 as _bounded_report
from . import cad_guided_plane_search_v6 as _search
from . import cad_guided_two_plane_context_v1 as _context
from . import cad_guided_two_plane_schedule_v1 as _pair_schedule
from . import cad_guided_stage_policies_v1 as _stages


_v1, _patch = _old._v1, _old._patch
_require, _digest, _pose = _old._require, _old._digest, _old._pose


_LIMITS = dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
    max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
    max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
    chunk_rows=65536, max_node_visits=512_000_000,
    max_triangle_pairs=256_000_000, max_certificate_pairs=1_024_000_000, max_normal_pairs=32_000_000,
    max_work_units=4_000_000_000, max_face_build_nodes=131071,
    max_face_build_triangle_visits=10_000_000, max_refined_start_groups=48,
    max_report_bytes=100_663_296)


@dataclass(frozen=True)
class PlaneInitializationBudgetsV12:
    max_hypothesis_records: int = 8192
    max_prefilter_unsigned_assignments: int = 1024
    max_cached_cad_faces: int = 2048
    max_indexed_face_triangles: int = 500_000
    max_prefilter_query_calls: int = 32768
    max_prefilter_query_rows: int = 64_000_000
    chunk_rows: int = 65536
    max_node_visits: int = 512_000_000
    max_triangle_pairs: int = 256_000_000
    max_certificate_pairs: int = 1_024_000_000
    max_normal_pairs: int = 32_000_000
    max_work_units: int = 4_000_000_000
    max_face_build_nodes: int = 131071
    max_face_build_triangle_visits: int = 10_000_000
    max_refined_start_groups: int = 48
    max_report_bytes: int = 100_663_296


def _two_plane_schedule_configuration(two_plane_enabled, mode, budgets):
    """Explicit V2 opt-in owns a validated snapshot; V1 adds no publication."""
    _require(type(two_plane_enabled) is bool, "INVALID_TWO_PLANE_LANE_POLICY")
    _require(type(mode) is str and mode in ("v1", "v2"), "INVALID_TWO_PLANE_SCHEDULE_MODE")
    if mode == "v1":
        _require(budgets is None, "SCHEDULE_BUDGETS_REQUIRE_EXPLICIT_V2_MODE")
        return {}
    _require(two_plane_enabled, "V2_SCHEDULE_REQUIRES_ENABLED_TWO_PLANE_LANE")
    from . import cad_guided_two_plane_schedule_v2 as schedule_v2
    owned = schedule_v2.TwoPlaneScheduleBudgetsV2(**schedule_v2._configuration(budgets))
    return dict(two_plane_schedule_mode="v2", two_plane_schedule_budgets=owned,
        two_plane_schedule_configuration=dict(mode="v2", budgets=asdict(owned)))


def _schedule_equation_digests(scan_equations, cad_equations):
    """Canonical owned equations, not a digest copied from scheduler output."""
    digests = {}
    for role, equations, maximum in (("scan", scan_equations, 128), ("cad", cad_equations, 2048)):
        _require(type(equations) is tuple and len(equations) <= maximum,
                 "INVALID_BOUND_SCHEDULE_EQUATION_TABLE")
        canonical = [_search._equation(row) for row in equations]
        digests[role+"_bank_equations_sha256"] = hashlib.sha256(json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return digests


def _bind_initialization_schedule_configuration(initialization, configuration):
    """Reject omitted/changed opt-in publication or an unrequested successor."""
    if configuration.get("two_plane_schedule_mode", "v1") == "v1":
        schedule = initialization.get("two_plane_schedule")
        _require("two_plane_schedule_configuration" not in initialization
            and not (type(schedule) is dict and (schedule.get("schema_version") == "two_plane_schedule_v2"
                or any(key in schedule for key in ("breadth_ledger", "residual_windows", "residual_statistics")))),
            "UNREQUESTED_V2_SCHEDULE_PUBLICATION")
        return
    expected = configuration["two_plane_schedule_configuration"]
    actual = initialization.get("two_plane_schedule_configuration")
    _require(type(actual) is dict and set(actual) == {"mode", "budgets"}
        and type(actual["mode"]) is str and actual["mode"] == "v2"
        and type(actual["budgets"]) is dict and set(actual["budgets"]) == set(expected["budgets"])
        and all(type(value) is int and value == expected["budgets"][key]
            for key, value in actual["budgets"].items()), "INITIALIZATION_V2_SCHEDULE_CONFIGURATION_MISMATCH")


def _validate_two_plane_schedule_v2(schedule, expected_configuration, expected_digests,
        scan_plane_ids, cad_plane_ids, h3, u3, max_h, max_u):
    """Bind the new schedule before any match becomes an axial context request."""
    _require(type(expected_configuration) is dict and len(expected_configuration) == 2
        and set(expected_configuration) == {"mode", "budgets"}
        and type(expected_configuration["mode"]) is str and expected_configuration["mode"] == "v2"
        and type(expected_configuration["budgets"]) is dict
        and all(type(value) is int for value in (h3, u3, max_h, max_u))
        and 0 <= h3 <= max_h <= 8192 and 0 <= u3 <= max_u <= 1024,
        "INVALID_EXPECTED_V2_SCHEDULE_BINDING")
    _require(type(schedule) is dict and schedule.get("schema_version") == "two_plane_schedule_v2"
        and schedule.get("status") == "COMPLETE"
        and schedule.get("selected_policy_evaluation_complete") is True
        and schedule.get("accepted_registration") is False
        and schedule.get("production_use_allowed") is False
        and schedule.get("finite_support_match_verified") is False,
        "INVALID_V2_SCHEDULE_PUBLICATION")
    actual = schedule.get("budgets")
    expected = expected_configuration["budgets"]
    _require(type(actual) is dict and len(actual) == len(expected) and set(actual) == set(expected)
        and all(type(actual[key]) is int and actual[key] == expected[key] for key in expected)
        and schedule.get("input_digests") == expected_digests,
        "V2_SCHEDULE_CONFIGURATION_OR_EQUATION_BINDING_MISMATCH")
    breadth = schedule.get("breadth_ledger")
    _require(type(breadth) is dict and breadth.get("schema_version") == "two_plane_schedule_v1"
        and breadth.get("status") == "COMPLETE"
        and breadth.get("selected_policy_evaluation_complete") is True
        and breadth.get("input_digests") == expected_digests,
        "INVALID_V2_BREADTH_BINDING")
    breadth_cap = breadth.get("budgets")
    _require(type(breadth_cap) is dict and set(breadth_cap) == set(_pair_schedule._MAXIMA)
        and all(type(breadth_cap[key]) is int and breadth_cap[key] == expected[key] for key in breadth_cap),
        "V2_BREADTH_BUDGET_BINDING_MISMATCH")
    matches = schedule.get("scheduled_matches")
    prefix = breadth.get("scheduled_matches")
    scalar_fields = {"ordered_match_index", "scan_pair_id", "canonical_cad_pair_index",
        "reserved_hypothesis_records", "reserved_unsigned_assignments"}
    pair_fields = {"scan_plane_ids", "cad_plane_ids"}
    match_fields = scalar_fields | pair_fields
    _require(type(matches) is list and len(matches) <= 256 and len(matches) % 2 == 0
        and type(prefix) is list and len(prefix) <= len(matches) and len(prefix) % 2 == 0
        and all(type(row) is dict and set(row) == match_fields
            and all(type(row[key]) is int for key in scalar_fields)
            and all(type(row[key]) is list and len(row[key]) == 2
                and all(type(value) is int for value in row[key]) for key in pair_fields)
            for table in (matches, prefix) for row in table)
        and all(prefix[index] == matches[index] for index in range(len(prefix)))
        and schedule.get("usable_schedule") is bool(matches), "INVALID_V2_ORDERED_MATCH_LEDGER")
    _require(type(scan_plane_ids) is tuple and len(scan_plane_ids) <= 128
        and type(cad_plane_ids) is tuple and len(cad_plane_ids) <= 2048
        and all(type(identifier) is int and 0 <= identifier < 2**63
                for identifier in (*scan_plane_ids, *cad_plane_ids))
        and len(set(scan_plane_ids)) == len(scan_plane_ids)
        and len(set(cad_plane_ids)) == len(cad_plane_ids), "INVALID_V2_BOUND_EQUATION_IDENTITIES")
    scan_indices = {identifier: index for index, identifier in enumerate(scan_plane_ids)}
    cad_indices = {identifier: index for index, identifier in enumerate(cad_plane_ids)}
    seen = set()
    for index in range(0, len(matches), 2):
        first, reverse = matches[index:index+2]
        for offset, row in enumerate((first, reverse)):
            _require(type(row) is dict and type(row.get("ordered_match_index")) is int
                and row["ordered_match_index"] == index+offset
                and type(row.get("reserved_hypothesis_records")) is int and row["reserved_hypothesis_records"] == 32
                and type(row.get("reserved_unsigned_assignments")) is int and row["reserved_unsigned_assignments"] == 1
                and type(row.get("scan_plane_ids")) is list and len(row["scan_plane_ids"]) == 2
                and type(row.get("cad_plane_ids")) is list and len(row["cad_plane_ids"]) == 2
                and all(type(value) is int for value in row["scan_plane_ids"]+row["cad_plane_ids"]),
                "INVALID_V2_RESERVED_ORDERED_MATCH")
        source, target = first["scan_plane_ids"], first["cad_plane_ids"]
        _require(all(identifier in scan_indices for identifier in source)
            and all(identifier in cad_indices for identifier in target)
            and reverse["scan_plane_ids"] == source and reverse["cad_plane_ids"] == target[::-1],
            "V2_BOTH_ORDER_OR_OWNED_IDENTITY_MISMATCH")
        a, b = (scan_indices[identifier] for identifier in source)
        c, d = (cad_indices[identifier] for identifier in target)
        _require(a < b and c < d, "V2_CANONICAL_PAIR_ORDER_MISMATCH")
        source_pair = a*(2*len(scan_indices)-a-1)//2 + b-a-1
        target_pair = c*(2*len(cad_indices)-c-1)//2 + d-c-1
        _require(all(type(row.get("scan_pair_id")) is int and row["scan_pair_id"] == source_pair
            and type(row.get("canonical_cad_pair_index")) is int
            and row["canonical_cad_pair_index"] == target_pair for row in (first, reverse))
            and (source_pair, target_pair) not in seen, "V2_CANONICAL_PAIR_OR_DUPLICATE_BINDING_MISMATCH")
        seen.add((source_pair, target_pair))
    capacity = min((max_h-h3)//32, max_u-u3)
    expected_joint = dict(retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3,
        max_joint_hypothesis_records=max_h, max_joint_unsigned_assignments=max_u,
        ordered_match_capacity_before_both_order_rounding=capacity,
        both_order_bundle_capacity=capacity//2, unused_odd_ordered_capacity=capacity % 2,
        reserved_two_plane_hypothesis_records=32*len(matches),
        reserved_two_plane_unsigned_assignments=len(matches),
        remaining_ordered_match_capacity=capacity-len(matches),
        joint_hypothesis_records_reserved=h3+32*len(matches),
        joint_unsigned_assignments_reserved=u3+len(matches))
    joint = schedule.get("joint_budget")
    _require(type(joint) is dict and 0 <= len(matches) <= capacity
        and all(type(joint.get(key)) is int and joint[key] == value for key, value in expected_joint.items()),
        "V2_SCHEDULE_JOINT_RESERVATION_BINDING_MISMATCH")


def _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled=True,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    _require(type(two_plane_enabled) is bool, "INVALID_TWO_PLANE_LANE_POLICY")
    schedule_configuration = _two_plane_schedule_configuration(two_plane_enabled,
        two_plane_schedule_mode, two_plane_schedule_budgets)
    _require(type(scan_plane_mode) is str and scan_plane_mode in ("v7", "composite_v1"),
             "INVALID_SCAN_PLANE_MODE")
    scan_configuration = dict(scan_plane_mode=scan_plane_mode)
    if scan_plane_mode == "v7":
        _require(seeded_plane_budgets is None and composite_plane_budgets is None,
                 "COMPOSITE_BUDGETS_REQUIRE_EXPLICIT_COMPOSITE_MODE")
    else:
        from . import cad_guided_scan_plane_preparation_v1 as composite
        seeded = composite._seeded.SeededPlaneBudgetsV1(**composite._seeded._configuration(
            composite._seeded.SeededPlaneBudgetsV1() if seeded_plane_budgets is None else seeded_plane_budgets))
        combined = composite.CompositeScanPlaneBudgetsV1(**composite._configuration(
            composite.CompositeScanPlaneBudgetsV1() if composite_plane_budgets is None else composite_plane_budgets))
        scan_configuration.update(seeded_plane_budgets=seeded, composite_plane_budgets=combined,
            scan_plane_configuration=dict(mode=scan_plane_mode, seeded_budgets=asdict(seeded),
                                          composite_budgets=asdict(combined)))
    _require(type(budgets) is PlaneInitializationBudgetsV12, "INVALID_V12_INITIALIZATION_BUDGET_TYPE")
    cap = {key: getattr(budgets, key) for key in _LIMITS}
    _require(all(type(value) is int and 0 <= value <= _LIMITS[key] for key, value in cap.items())
        and cap["chunk_rows"] > 0, "INVALID_V12_INITIALIZATION_BUDGET")
    ng, nb = _noisy._configuration(noisy_policy, noisy_budgets)
    cg, cb = _patch._configuration(cad_patch_policy, cad_patch_budgets)
    sg, sb = _search._configuration(search_policy, search_budgets)
    surface = _surface._budgets(surface_budgets)
    return dict(**scan_configuration, **schedule_configuration,
        two_plane_enabled=two_plane_enabled, policy=_old._registration_policy(registration_policy),
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(**ng), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(**nb),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(**cg), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(**cb),
        search_policy=_search.PlaneSearchPolicyV6(**sg), search_budgets=_search.PlaneSearchBudgetsV6(**sb),
        budgets=PlaneInitializationBudgetsV12(**cap),
        surface_budgets=_surface.TriangleSurfaceBudgets(**asdict(surface)))


def _base():
    result = _old._base()
    result.update(schema_version="noisy_plane_initialization_prefilter_v12", search_exhaustive=False,
        search_scope="ALL_GENERATED_PROPOSED_SIGNS_SCREENED_UPSTREAM_SEARCH_AND_REFINEMENT_NOT_EXHAUSTIVE",
        noise_bound_status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        global_pose_uniqueness="NOT_ESTABLISHED", selected_unsigned_assignment_ids=[],
        prefilter_schedule="ALL_GENERATED_PROPOSED_SIGNS_HARD_UNSIGNED_BOUND_NO_RANK_OMISSION",
        refinement_schedule="48_STRATIFIED_EXACT_PASSING_START_GROUPS",
        all_passing_start_groups=[], selected_policy_evaluation_complete=False,
        numerical_support_policy=_stages.PROXIMITY,
        proposal_support_policy=_stages.PROXIMITY,
        numerical_consumer_complete=False,
        box_screening_policy="REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1",
        box_screening_complete=False,
        work_unit_formula="BOX_SCREEN_ROWS_PLUS_UNCHANGED_V3_QUERY_WORK",
        counter_semantics=dict(
            support_attempted_box_screen_rows="REQUESTED_DISPATCH_ROWS_INCLUDING_PREFLIGHT_REFUSAL",
            aggregate_box_screen_rows_and_calls="BUDGET_ADMITTED_CHARGED_HELPER_ATTEMPTS",
            completed_box_screen_rows_and_calls="VALIDATED_COMPLETE_HELPER_OUTPUTS",
            completed_query_rows_and_calls="VALIDATED_COMPLETE_V3_SURFACE_OUTPUTS",
            bound_excluded_support_rows="FULL_SUPPORT_ROWS_NOT_FINITE_QUERY_COMPLETIONS"))
    result["statistics"].update(certificate_pairs=0, triangle_box_pairs=0,
        proposed_hypothesis_count=0, scheduled_hypothesis_count=0,
        omitted_proposed_hypothesis_count=0, search_rejected_hypothesis_count=0,
        eligible_unsigned_assignment_count=0, scheduled_unsigned_assignment_count=0,
        omitted_unsigned_assignment_count=0, exact_passing_start_group_count=0,
        omitted_passing_start_group_count=0,
        planned_box_screen_calls=0, planned_box_screen_rows=0,
        box_screen_calls=0, box_screen_rows=0,
        completed_box_screen_calls=0, completed_box_screen_rows=0,
        completed_query_calls=0, completed_query_rows=0,
        assigned_support_count=0, completed_box_support_count=0,
        bound_excluded_support_count=0, bound_excluded_support_rows=0,
        bound_excluded_query_calls=0,
        completed_finite_query_support_count=0, absent_context_slot_count=0)
    result.update(proposal_lanes=["TRIPLET", "TWO_PLANE_CONTEXT"],
        pair_lane_budget_policy="TRIPLET_FIRST_RESIDUAL_NO_OUTCOME_BASED_REFILL",
        pair_lane_starvation_possible=True, two_plane_schedule=None,
        two_plane_context=None, joint_proposal_accounting=None)
    return result


# Default preparation/binding retain V7; composite preparation is explicit.
_prepare_triplets = _previous._prepare
_bound_scan = _previous._bound_scan



def _validate_two_plane_context_configuration(helper, mode):
    """Bind the requested consumer version and complete fixed budget profile."""
    _require(type(mode) is str and mode in ("v1", "v2"), "INVALID_CONTEXT_CONSUMER_MODE")
    if mode == "v2":
        from . import cad_guided_two_plane_context_v2 as context_v2
        expected = context_v2._configuration(context_v2.TwoPlaneContextBudgetsV2())
        capacity = context_v2._base()["scheduling_capacity"]
    else:
        expected = _context._configuration(_context.TwoPlaneContextBudgetsV1())
        capacity = None
    _require(type(helper) is dict and helper.get("schema_version") == "two_plane_context_"+mode,
        "TWO_PLANE_CONTEXT_SCHEMA_MISMATCH")
    actual = helper.get("budgets")
    _require(type(actual) is dict and set(actual) == set(expected)
        and all(type(actual[key]) is int and actual[key] == value for key, value in expected.items()),
        "TWO_PLANE_CONTEXT_BUDGET_PROFILE_MISMATCH")
    if capacity is None:
        _require("scheduling_capacity" not in helper, "UNREQUESTED_V2_CONTEXT_CAPACITY")
    else:
        actual = helper.get("scheduling_capacity")
        _require(type(actual) is dict and set(actual) == set(capacity)
            and all(type(actual[key]) is type(value) and actual[key] == value
                for key, value in capacity.items()), "TWO_PLANE_CONTEXT_CAPACITY_MISMATCH")


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
    context_mode = configuration.get("two_plane_schedule_mode", "v1")
    context_proposer, context_options = _context.propose_two_plane_context_v1, {}
    if context_mode == "v2":
        from . import cad_guided_two_plane_context_v2 as context_v2
        context_proposer = context_v2.propose_two_plane_context_v2
        context_options = dict(budgets=context_v2.TwoPlaneContextBudgetsV2())
    helper = context_proposer(
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
    _validate_two_plane_context_configuration(helper, context_mode)
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


class _FaceQueries:
    """Eager owned V3 face indices; root boxes never grant finite support."""
    def __init__(self, context, face_map, budget, surface_budget, result):
        self.indices, self.boxes = {}, {}
        self.budget, self.surface_budget = budget, surface_budget
        self.statistics = result["statistics"]
        _require(len(face_map) <= budget.max_cached_cad_faces, "CAD_FACE_CACHE_BUDGET_EXCEEDED")
        planned = sum(len(patch["triangle_rows"]) for patch in face_map.values())
        _require(planned <= budget.max_indexed_face_triangles, "CAD_FACE_TRIANGLE_BUDGET_EXCEEDED")
        for face, patch in sorted(face_map.items()):
            values = asdict(surface_budget)
            for name in ("nodes", "triangle_visits"):
                key = "build_"+name
                values["max_"+key] = min(values["max_"+key],
                    getattr(budget, "max_face_"+key)-self.statistics["face_"+key])
            rows = patch["triangle_rows"]
            built = _surface.build_triangle_surface(context["cad"][rows], context["faces"][rows],
                budgets=_surface.TriangleSurfaceBudgets(**values))
            result["face_index_evidence"].append(dict(cad_plane_id=face,
                triangle_rows_sha256=patch["triangle_rows_sha256"], triangle_count=len(rows),
                result={key: value for key, value in built.items() if key != "index"}))
            for name in ("build_nodes", "build_triangle_visits"):
                self.statistics["face_"+name] += built["diagnostics"].get(name, 0)
            _require(built["status"] == "COMPLETE", "ASSIGNED_FACE_SURFACE_UNAVAILABLE")
            index = built["index"]
            self.indices[face] = index
            # The builder snapshots ALL face triangles before root min/max.
            # Immutable extrema enclose every represented triangle convex hull;
            # no caller box or fitted-patch rectangle is trusted.
            self.boxes[face] = (index.lower[0], index.upper[0])
            result["face_index_evidence"][-1]["box_binding"] = dict(
                scope="OWNED_COMPLETE_V3_FACE_INDEX_ROOT_ALL_TRIANGLE_VERTICES",
                index_triangles_sha256=_digest(index.triangles),
                lower_sha256=_digest(index.lower[0]), upper_sha256=_digest(index.upper[0]),
                lower_mm=index.lower[0].tolist(), upper_mm=index.upper[0].tolist(),
                finite_surface_support_admitted=False)
            self.statistics["indexed_face_count"] += 1
            self.statistics["indexed_face_triangles"] += len(rows)

    def screen(self, face, points, gate):
        stats, budget = self.statistics, self.budget
        count = len(points)
        _require(stats["box_screen_calls"] < budget.max_prefilter_query_calls
            and stats["box_screen_rows"]+count <= budget.max_prefilter_query_rows,
            "BOX_SCREEN_BUDGET_EXCEEDED")
        _require(stats["work_units"]+count <= budget.max_work_units,
                 "BOX_SCREEN_WORK_BUDGET_EXCEEDED")
        # Charge the complete attempted call even if admission/arithmetic fails.
        # Completion is a separate validated counter, never an inferred count.
        stats["box_screen_calls"] += 1
        stats["box_screen_rows"] += count
        stats["work_units"] += count
        lower, upper = self.boxes[face]
        answer = _box.screen_face_box_rows(points, lower, upper,
            distance_gate_mm=gate, max_rows=min(budget.chunk_rows, self.surface_budget.max_query_rows))
        _require(type(answer) is dict, "INVALID_FACE_BOX_SCREEN_RESULT")
        if answer.get("status") != "COMPLETE":
            reason = answer.get("reason")
            _require(type(reason) is str and 0 < len(reason) <= 128 and reason.isupper(),
                     "INVALID_FACE_BOX_SCREEN_FAILURE_REASON")
            _require(False, "FACE_BOX_SCREEN_UNAVAILABLE:"+reason)
        _require(all(type(answer.get(key)) is int and answer[key] == count
            for key in ("input_rows", "attempted_rows", "completed_rows")),
            "INCOMPLETE_FACE_BOX_SCREEN_DENOMINATOR")
        bounds, outside = answer.get("distance_lower_mm"), answer.get("definitely_outside_mask")
        _require(type(bounds) is np.ndarray and bounds.dtype == np.dtype(np.float64)
            and bounds.shape == (count,) and np.isfinite(bounds).all() and np.all(bounds >= 0.)
            and type(outside) is np.ndarray and outside.dtype == np.dtype(bool)
            and outside.shape == (count,) and np.array_equal(outside, bounds > gate)
            and type(answer.get("definitely_outside_count")) is int
            and answer["definitely_outside_count"] == int(np.count_nonzero(outside))
            and answer.get("finite_surface_support_admitted") is False,
            "INVALID_FACE_BOX_SCREEN_EVIDENCE")
        stats["completed_box_screen_calls"] += 1
        stats["completed_box_screen_rows"] += count
        return answer

    def __call__(self, face, points):
        stats, budget = self.statistics, self.budget
        _require(stats["query_calls"] < budget.max_prefilter_query_calls
            and stats["query_rows"]+len(points) <= budget.max_prefilter_query_rows,
            "PREFILTER_QUERY_BUDGET_EXCEEDED")
        values = asdict(self.surface_budget)
        charged = ("node_visits", "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units")
        for name in charged:
            values["max_"+name] = min(values["max_"+name], getattr(budget, "max_"+name)-stats[name])
        stats["query_calls"] += 1
        stats["query_rows"] += len(points)
        query = _surface.query_closest_surface(self.indices[face], points,
            budgets=_surface.TriangleSurfaceBudgets(**values))
        _require(type(query) is dict, "INVALID_V3_FACE_QUERY_RESULT")
        telemetry = _v3._diagnostic_snapshot(query, complete=query.get("status") == "COMPLETE")
        for name in charged:
            _require(telemetry[name] <= values["max_"+name], "V3_FACE_QUERY_EXCEEDED_ALLOCATED_BUDGET")
        for name in (*charged, "triangle_box_pairs"):
            stats[name] += telemetry[name]
        _require(query["status"] == "COMPLETE", "ASSIGNED_FACE_QUERY_UNAVAILABLE:"+str(query["reason"]))
        validated = _v3._validate_query_result(query, len(points), index=self.indices[face])
        stats["completed_query_calls"] += 1
        stats["completed_query_rows"] += len(points)
        return validated


def _prefilter(context, result, configuration):
    budget, policy, surface = configuration["budgets"], configuration["policy"], configuration["surface_budgets"]
    stats, hypotheses = result["statistics"], context["hypotheses"]
    _require(len(hypotheses) <= budget.max_hypothesis_records, "HYPOTHESIS_RECORD_BUDGET_EXCEEDED")
    eligible = sorted({row["unsigned_assignment_id"] for row in hypotheses if row["status"] == "PROPOSED"})
    selected = set(eligible)
    result["selected_unsigned_assignment_ids"] = sorted(selected)
    stats.update(hypothesis_count=len(hypotheses), eligible_unsigned_assignment_count=len(eligible),
        scheduled_unsigned_assignment_count=len(selected), omitted_unsigned_assignment_count=len(eligible)-len(selected))
    scans = context["scan_supports"]
    faces = {row["plane_id"]: row for row in context["cad_patches"]["patches"]}
    chunk = min(budget.chunk_rows, surface.max_query_rows)
    _require(chunk > 0, "PREFILTER_QUERY_ROW_CAP_ZERO")
    used_faces, checked_scans, checked_faces, scheduled = {}, set(), set(), []
    # All completed-search identities enter the ledger before any per-support
    # validation can fail. A malformed selected support must not erase later
    # known hypotheses that were never attempted.
    for hypothesis in hypotheses:
        proposed = hypothesis["status"] == "PROPOSED"
        chosen = proposed and hypothesis["unsigned_assignment_id"] in selected
        row = dict(hypothesis, prefilter_status="NOT_ATTEMPTED" if chosen else
            "NOT_SCHEDULED_BY_DECLARED_UNSIGNED_QUOTA" if proposed else "SEARCH_NOMINAL_WITNESS_REFUSAL",
            prefilter_reasons=[], assigned_support=[], prefilter_scheduled=chosen)
        absent = hypothesis["status"] == "ABSENT_CONTEXT_SLOT"
        if absent:
            row["prefilter_status"] = "AXIAL_SLOT_ABSENT_NOT_GEOMETRIC_EXCLUSION"
        result["hypotheses"].append(row)
        stats["proposed_hypothesis_count" if proposed else
              "absent_context_slot_count" if absent else "search_rejected_hypothesis_count"] += 1
        stats["scheduled_hypothesis_count"] += int(chosen)
        stats["omitted_proposed_hypothesis_count"] += int(proposed and not chosen)
    # Retain the whole completed-search ledger before a smaller hard bound can
    # refuse. This bound is NOT a selection quota: no _strata/min/prefix subset
    # may acquire finite support evidence while other proposed signs are omitted.
    _require(len(eligible) <= budget.max_prefilter_unsigned_assignments,
             "ALL_PROPOSED_UNSIGNED_PREFLIGHT_BUDGET_EXCEEDED")
    for index, row in enumerate(result["hypotheses"]):
        _require(row["hypothesis_index"] == index and row["status"] in ("PROPOSED", "REJECTED", "ABSENT_CONTEXT_SLOT"),
                 "INVALID_COMPLETED_SEARCH_HYPOTHESIS")
        chosen = row["prefilter_scheduled"]
        if not chosen:
            continue
        scheduled.append(row)
        _pose(row)
        matches = row["correspondences"]
        lane = row.get("proposal_lane")
        _require(lane in ("TRIPLET", "TWO_PLANE_CONTEXT"), "INVALID_PROPOSAL_LANE")
        count = 3 if lane == "TRIPLET" else 2
        _require(len(matches) == count and len({item["scan_plane_id"] for item in matches}) == count
            and len({item["cad_plane_id"] for item in matches}) == count, "INVALID_LANE_SUPPORT_BINDING")
        for match in matches:
            scan_id, face = match["scan_plane_id"], match["cad_plane_id"]
            _require(scan_id in scans and face in faces and type(match["normal_sign"]) is int
                and match["normal_sign"] in (-1, 1), "UNKNOWN_NOISY_PLANE_SUPPORT")
            patch = scans[scan_id]
            if scan_id not in checked_scans:
                _bound_scan(context, patch)
                checked_scans.add(scan_id)
            if face not in checked_faces:
                _old._bound_support(context, cad_patch=faces[face])
                checked_faces.add(face)
            count = len(patch["raw_vertex_rows"])
            stats["planned_query_rows"] += count
            stats["planned_query_calls"] += (count+chunk-1)//chunk
            used_faces[face] = faces[face]
            row["assigned_support"].append(dict(match, proposal_kind=patch["proposal_kind"],
                source_record_id=patch["source_record_id"], support_count=count,
                raw_vertex_rows_sha256=patch["raw_vertex_rows_sha256"],
                scan_triangle_rows_sha256=patch["triangle_rows_sha256"],
                cad_triangle_rows_sha256=faces[face]["triangle_rows_sha256"],
                query_rows=0, attempted_query_rows=0, finite_unambiguous_inlier_count=0,
                coverage=None, ambiguous_rows=0, boundary_rows=0,
                definite_distance_inlier_rows=0, definite_distance_outside_rows=0,
                threshold_unresolved_rows=0, uncertain_region_rows=0,
                maximum_distance_interval_width_mm=0., maximum_optimizer_error_mm=0.,
                maximum_restoration_error_mm=0.,
                attempted_box_screen_rows=0, completed_box_screen_rows=0,
                definitely_box_outside_count=0, possible_box_inlier_count_upper=None,
                possible_box_coverage_upper=None, box_screening_status="NOT_ATTEMPTED",
                box_exclusion_decision=None, finite_query_disposition="NOT_ATTEMPTED",
                status="NOT_ATTEMPTED", reasons=[]))
            stats["assigned_support_count"] += 1
    _require(stats["planned_query_rows"] <= budget.max_prefilter_query_rows
        and stats["planned_query_calls"] <= budget.max_prefilter_query_calls,
        "PREFILTER_PREFLIGHT_BUDGET_EXCEEDED")
    stats["planned_box_screen_rows"] = stats["planned_query_rows"]
    stats["planned_box_screen_calls"] = stats["planned_query_calls"]
    _require(stats["planned_box_screen_rows"] <= budget.max_work_units,
             "BOX_SCREEN_WORK_PREFLIGHT_BUDGET_EXCEEDED")
    query = _FaceQueries(context, used_faces, budget, surface, result)
    groups, lookup = result["all_passing_start_groups"], {}
    for row in scheduled:
        rotation, translation = _pose(row)
        row["prefilter_status"] = "INCOMPLETE"
        for support in row["assigned_support"]:
            raw = scans[support["scan_plane_id"]]["raw_vertex_rows"]
            support["status"] = "INCOMPLETE"
            support["box_screening_status"] = "INCOMPLETE"
            # Bounded by the admitted original raw-row count. Preserve the
            # original chunkwise transform, then reuse its SAME represented
            # values for screening and any full finite query. No row filtering.
            transformed = np.empty((len(raw), 3), dtype=np.float64)
            for start in range(0, len(raw), chunk):
                transformed[start:start+chunk] = context["scan"][raw[start:start+chunk]]@rotation.T+translation
                points = transformed[start:start+chunk]
                support["attempted_box_screen_rows"] += len(points)
                answer = query.screen(support["cad_plane_id"], points,
                                      policy.max_correspondence_distance_mm)
                support["completed_box_screen_rows"] += len(points)
                support["definitely_box_outside_count"] += answer["definitely_outside_count"]
            _require(support["completed_box_screen_rows"] == support["support_count"],
                     "INCOMPLETE_WHOLE_SUPPORT_BOX_SCREEN")
            decision = _box.decide_face_box_exclusion(
                support_count=support["support_count"],
                definitely_outside_count=support["definitely_box_outside_count"],
                minimum_coverage=policy.min_scan_coverage)
            support.update(box_screening_status="COMPLETE",
                box_exclusion_decision=decision["decision"],
                possible_box_inlier_count_upper=decision["possible_inlier_count_upper"],
                possible_box_coverage_upper=decision["possible_inlier_fraction_upper"])
            stats["completed_box_support_count"] += 1
            if decision["decision"] == "BOUND_EXCLUDED":
                support.update(status="REJECTED", finite_query_disposition="NOT_REQUIRED_BOUND_EXCLUDED",
                    reasons=["ASSIGNED_WHOLE_FACE_BOX_COVERAGE_UPPER_BELOW_MINIMUM"])
                stats["bound_excluded_support_count"] += 1
                stats["bound_excluded_support_rows"] += support["support_count"]
                stats["bound_excluded_query_calls"] += (len(raw)+chunk-1)//chunk
                # Finite coverage remains None. No queried rows does NOT mean
                # zero true support, a missing bend, or a failed inspection.
                continue
            _require(decision["decision"] == "NEEDS_SURFACE_QUERY", "INVALID_FACE_BOX_EXCLUSION_DECISION")
            support["finite_query_disposition"] = "INCOMPLETE"
            for start in range(0, len(raw), chunk):
                points = transformed[start:start+chunk]
                support["attempted_query_rows"] += len(points)
                answer = query(support["cad_plane_id"], points)
                unambiguous = _v3._unambiguous(answer)
                support["query_rows"] += len(points)
                masks = _v3._support_masks(answer, policy.max_correspondence_distance_mm)
                support["finite_unambiguous_inlier_count"] += int(np.count_nonzero(
                    unambiguous & masks["definite_support"]))
                for key, name in (("definite_support", "definite_distance_inlier_rows"),
                                  ("definite_outside", "definite_distance_outside_rows"),
                                  ("unresolved", "threshold_unresolved_rows")):
                    support[name] += int(np.count_nonzero(masks[key]))
                support["uncertain_region_rows"] += int(np.count_nonzero(answer["region_uncertain"]))
                for name, values in (
                    ("maximum_distance_interval_width_mm", np.nextafter(
                        answer["distance_upper"]-answer["distance_lower"], np.inf)),
                    ("maximum_optimizer_error_mm", answer["closest_point_error"]),
                    ("maximum_restoration_error_mm", answer["closest_point_restoration_error"])):
                    support[name] = max(support[name], float(np.max(values)))
                support["ambiguous_rows"] += int(np.count_nonzero(~unambiguous))
                support["boundary_rows"] += int(np.count_nonzero(answer["closest_regions"] != "FACE"))
            _require(support["query_rows"] == support["support_count"],
                     "INCOMPLETE_WHOLE_SUPPORT_FINITE_QUERY")
            support["finite_query_disposition"] = "COMPLETE"
            stats["completed_finite_query_support_count"] += 1
            # Proposal proximity alone is not a correspondence or pose certificate.
            # The legacy ambiguity count is retained, not silently relabelled.
            # A nearest-rounded fraction can cross a represented policy value
            # (for example 5/6). Only a lower enclosure can establish support.
            support["coverage"] = _v3._fraction_bound(
                support["definite_distance_inlier_rows"], support["support_count"], upper=False)
            passed = support["coverage"] >= policy.min_scan_coverage
            support.update(status="PASSED" if passed else "REJECTED",
                reasons=[] if passed else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"])
        passed = all(support["status"] == "PASSED" for support in row["assigned_support"])
        row.update(prefilter_status="PASSED" if passed else "REJECTED",
            prefilter_reasons=[] if passed else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
        stats["hypotheses_evaluated"] += 1
        stats["passing_hypothesis_count" if passed else "rejected_hypothesis_count"] += 1
        if passed:
            key = rotation.tobytes()+translation.tobytes()
            if key not in lookup:
                lookup[key] = len(groups)
                groups.append(dict(exact_group_index=len(groups), initialized_rotation=rotation.tolist(),
                    initialized_translation_mm=translation.tolist(), hypothesis_indices=[],
                    refinement_disposition="NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA"))
            group = groups[lookup[key]]
            row["exact_start_group_index"] = group["exact_group_index"]
            group["hypothesis_indices"].append(row["hypothesis_index"])
    indices = _search._strata(len(groups), budget.max_refined_start_groups)
    for index, group_index in enumerate(indices):
        groups[group_index]["refinement_disposition"] = "SCHEDULED_FOR_V3_REFINEMENT"
        result["start_groups"].append(dict(groups[group_index], index=index))
    stats.update(exact_passing_start_group_count=len(groups), surviving_start_group_count=len(indices),
                 omitted_passing_start_group_count=len(groups)-len(indices))
    _require(stats["hypotheses_evaluated"] == stats["scheduled_hypothesis_count"],
             "PREFILTER_SELECTED_DENOMINATOR_INCOMPLETE")
    _require(stats["completed_box_support_count"] == stats["assigned_support_count"]
        and stats["completed_box_screen_rows"] == stats["planned_box_screen_rows"]
        and stats["completed_box_screen_calls"] == stats["planned_box_screen_calls"]
        and stats["completed_finite_query_support_count"]+stats["bound_excluded_support_count"]
            == stats["assigned_support_count"]
        and stats["completed_query_rows"]+stats["bound_excluded_support_rows"]
            == stats["planned_query_rows"]
        and stats["completed_query_calls"]+stats["bound_excluded_query_calls"]
            == stats["planned_query_calls"], "BOX_AND_FINITE_SUPPORT_DENOMINATOR_INCOMPLETE")
    result.update(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
        numerical_consumer_complete=True, box_screening_complete=True,
        reasons=[] if indices else ["NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"])


def _initialize(scan_points, scan_triangle_raw_rows, cad_triangles, cad_face_ids,
                cad_source_triangle_ids, epsilon, configuration, unit_assumption):
    result, context = _base(), None
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


def prefilter_plane_initializations_v12(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm, registration_policy,
        unit_assumption, noisy_policy=_noisy.NoisyMeshPlanePolicyV5(),
        noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(), cad_patch_policy=_patch.CoplanarPatchPolicyV4(),
        cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(), search_policy=_search.PlaneSearchPolicyV6(),
        search_budgets=_search.PlaneSearchBudgetsV6(), budgets=PlaneInitializationBudgetsV12(),
        surface_budgets=None, two_plane_enabled=True, scan_plane_mode="v7",
        seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    """Owned-array scheduled finite-face evidence; no accepted registration."""
    result, maximum = _base(), _LIMITS["max_report_bytes"]
    try:
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled,
            scan_plane_mode, seeded_plane_budgets, composite_plane_budgets, **schedule_request)
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


_registration_rows = _previous._registration_rows


def register_whole_part_planes_v12(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm,
        cad_initialization_points, registration_policy, unit_assumption, registration_raw_rows=None,
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        search_policy=_search.PlaneSearchPolicyV6(), search_budgets=_search.PlaneSearchBudgetsV6(),
        budgets=PlaneInitializationBudgetsV12(), surface_budgets=None, registration_budgets=None, two_plane_enabled=True,
        connected_measurement_contexts=False, _return_owned_measurement_support=False,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    """Explicit error-bound declaration and owned supports, no native or file IO.

All original assigned support rows reach bound full-face box checks. Nonexcluded
supports reach unchanged full finite queries; boxes never grant support. The global
registration selection still has V2's <=200k limit and is separately disclosed.
The returned fit never establishes global pose uniqueness or a stable datum.
"""
    result = _old._registration_result(unit_assumption)
    result["schema_version"] = "noisy_plane_initialized_finite_registration_v12"
    evidence, query, measurement_support = result["evidence"], None, None
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
            scan_plane_mode, seeded_plane_budgets, composite_plane_budgets, **schedule_request)
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
        if "two_plane_schedule_configuration" in evidence:
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


def register_measurement_contexts_v12(*args, **kwargs):
    """Return JSON registration evidence and its owned scan-support transport.

    The opaque second value retains existing extraction's exact support arrays;
    descriptor hashes are never inverted or substituted for original row IDs.
    It is not part of the JSON report and supplies no external pose or labels.
    """
    _require("_return_owned_measurement_support" not in kwargs
        and kwargs.get("connected_measurement_contexts", True) is True,
        "CONNECTED_CONTEXT_API_REQUIRED")
    kwargs["connected_measurement_contexts"] = True
    return register_whole_part_planes_v12(*args, **kwargs, _return_owned_measurement_support=True)


def _context_support(scan, query_surface, rotation, translation, policy, reverse_coverage):
    """Distance evidence independent of legacy usable-constraint/rank diagnostics.

    All selected registration rows are retained; no ambiguous row is erased.
    These are represented-coordinate intervals, not latent scanner-error bounds.
    """
    answer = query_surface(scan@rotation.T+translation)
    masks = _stages.classify_stage_rows_v1(answer, policy.max_correspondence_distance_mm)
    inside = masks["definite_proximity"]
    count = int(np.count_nonzero(inside))
    coverage = _v3._fraction_bound(count, len(scan), upper=False)
    metrics = _v3._metric_bounds(answer["distance_lower"][inside], answer["distance_upper"][inside])
    reasons = []
    if count < 36 or coverage < policy.min_scan_coverage or reverse_coverage < policy.min_cad_coverage:
        reasons.append("INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY")
    if metrics["p95_upper"] is None or metrics["p95_upper"] > policy.max_inlier_plane_p95_mm:
        reasons.append("CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE")
    return dict(status="SUPPORTED_CONTEXT" if not reasons else "REJECTED_CONTEXT",
        pose_sha256=_digest(np.ascontiguousarray(np.concatenate((rotation.ravel(), translation)), dtype="<f8")),
        reasons=reasons, geometric_scan_coverage_lower=coverage,
        geometric_inlier_count=count, input_row_count=len(scan),
        row_counts=_stages.stage_row_counts_v1(answer, policy.max_correspondence_distance_mm),
        finite_distance_metrics=metrics, reverse_sample_coverage=reverse_coverage,
        proximity_policy=_stages.PROXIMITY, accepted_registration=False,
        scope="REPRESENTED_COORDINATE_PROXIMITY_NOT_POSE_OR_CORRESPONDENCE_CERTIFICATE")


def _publish_contexts(result):
    evidence = result["evidence"]
    candidates = evidence["candidates"]
    _require(evidence["candidate_search_complete"] is True and all(
        row.get("execution_status") == "ATTEMPTED"
        and row.get("numerical_consumer_status") == _v3.CONSUMER_COMPLETE
        and type(row.get("conditional_context_support")) is dict
        and row["conditional_context_support"].get("status") in ("SUPPORTED_CONTEXT", "REJECTED_CONTEXT")
        for row in candidates), "INCOMPLETE_CONDITIONAL_CONTEXT_EVALUATION")
    indices = [row["index"] for row in candidates
        if row["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"]
    evidence.update(measurement_context_indices=indices,
        measurement_context_quantifier="EXISTS_COMPLETED_GENERATED_CONTEXT",
        measurement_context_search_scope="DECLARED_FINITE_CANDIDATE_SET_ONLY",
        joint_configuration_status="NOT_ESTABLISHED",
        continuous_pose_family_coverage="NOT_ESTABLISHED",
        accepted_registration=False)
    result.update(status="CONDITIONAL_MEASUREMENT_CONTEXTS" if indices else "UNAVAILABLE",
        transform_scan_to_cad=None,
        reasons=[] if indices else ["NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS"])
