"""V15 initializer: V14 text plus the bounded two-of-three triplet completion lane (V1).

Frozen V14 stays unchanged. The main triplet/context lanes, prefilter, gates and
counters are verbatim; the completion lane is additive, separately budgeted and
separately published under `completion_lane`, and its passing poses join the
unchanged exact start groups and refinement schedule.

NEW unqualified V14 initializer: V13 orchestration on context V4 and surface V4.

Frozen V12/V13 are unchanged. V14 keeps V13's text and substitutes exactly:
context V4 (kd-partition raw extrema V3), triangle surface V4 with global
registration V4 as the finite-face prefilter and refinement consumer, and
stage policies V2 (identical strings bound to the V4 consumer). Because the
frozen V12 _FaceQueries/_prefilter/_context_support/_configuration bodies bind
their own module's surface/registration modules, V14 carries copies of those
four definitions resolved against this module's V4 bindings: _prefilter and
_context_support are byte-identical to V12; _FaceQueries differs only in the
two TriangleSurfaceBudgetsV4 constructor sites; _configuration is copied as
_frozen_configuration_v4 (renamed def, one added docstring line, the V4
constructor). Every other V12 helper (triplet preparation, scan bounding,
registration rows, schedule configuration/validation, publication) is reused
by attribute. The reused V12 report text keeps V3-era work/counter labels as
wire strings; the additive finite_surface_implementation field below names the
actual V4 surface/registration consumer of this initializer.

The V12 registration and prefilter schema names are retained wire contracts for
the unchanged measurement consumer, not historical implementation identities.
Every usable publication additionally binds INITIALIZATION_IMPLEMENTATION, the
explicit context-V4 configuration and the V4 registration implementation label.
No module/global rewriting or injected caller callable is used.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from . import cad_guided_plane_initialization_v12 as _frozen
from . import cad_guided_two_plane_context_v4 as _context_v4
# The alias names _v3/_surface/_stages/_box are retained from the V12/V13 text so
# the verbatim copied bodies below read identically; they are bound to V4 here.
from . import cad_guided_global_registration_v4 as _v3
from . import cad_guided_triangle_surface_v4 as _surface
from . import cad_guided_stage_policies_v2 as _stages
from . import cad_guided_face_box_screen_v1 as _box

# Explicitly reused frozen numerical types/helpers, not copied implementations.
_old = _frozen._old
_noisy, _patch, _search = _frozen._noisy, _frozen._patch, _frozen._search
_v1, _context, _pair_schedule = _frozen._v1, _frozen._context, _frozen._pair_schedule
_require, _digest, _pose = _frozen._require, _frozen._digest, _frozen._pose
_LIMITS = _frozen._LIMITS
PlaneInitializationBudgetsV12 = _frozen.PlaneInitializationBudgetsV12
from .cad_guided_report_envelope_v7 import bound_initialization_report_v7 as _bounded_report
_prepare_triplets, _bound_scan = _frozen._prepare_triplets, _frozen._bound_scan
_registration_rows = _frozen._registration_rows
_publish_contexts = _frozen._publish_contexts  # string comparisons only; identical strings
_two_plane_schedule_configuration = _frozen._two_plane_schedule_configuration
_schedule_equation_digests = _frozen._schedule_equation_digests
_validate_two_plane_schedule_v2 = _frozen._validate_two_plane_schedule_v2
_bind_initialization_schedule_configuration = _frozen._bind_initialization_schedule_configuration

INITIALIZATION_IMPLEMENTATION = "CAD_GUIDED_PLANE_INITIALIZATION_V15_CONTEXT_V4_COMPLETION_V1"
COMPLETION_IMPLEMENTATION = "TWO_OF_THREE_TRIPLET_THIRD_FACE_COMPLETION_ALONG_PASSING_PAIR_AXIS_V1"
COMPLETION_LANE = "TRIPLET_COMPLETION"
MEASUREMENT_CONTEXT_SCHEDULE = "POSE_CLASS_DEDUP_THEN_RANKED_BUDGET_CAP_V1"
DUPLICATE_CONTEXT = "SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS"
OMITTED_CONTEXT = "SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET"
CONTEXT_IMPLEMENTATION = "RAW_EXTREMA_KD_PARTITION_CONTEXT_V4_STAGED_SCAN_PROJECTION"
REGISTRATION_IMPLEMENTATION = _v3.REGISTRATION_IMPLEMENTATION
_CONTEXT_DECLARATIONS = (
    "scan_projection_arithmetic", "scan_extrema_partition", "scan_projection_bound_scope",
    "cad_projection_arithmetic", "scan_projection_chunk_scope", "chunk_rows_scope",
    "extrema_guard_status", "extrema_membership_scope", "extrema_memory_scope",
)


@dataclass(frozen=True)
class CompletionLanePolicyV1:
    """Bounded third-face completion of two-of-three triplet hypotheses.

    For a scheduled TRIPLET hypothesis whose two supports PASSED the frozen
    finite prefilter and whose third support was REJECTED, the rotation is kept,
    the translation is shifted along the intersection direction of the two
    passing CAD planes, and the third correspondence is re-assigned to every
    CAD face whose plane normal is compatible with the rotated third scan
    normal under the hypothesis's own recorded allowance. Every completion is
    re-screened and re-queried on all three supports with the unchanged rules;
    nothing is admitted on plane arithmetic alone. Unresolvable candidates
    (axis-parallel faces) are counted, not inferred away.
    """
    enabled: bool = True
    max_parents: int = 256
    max_completions_per_parent: int = 64
    max_completions_total: int = 4096
    max_shift_mm: float = 1000.
    min_axis_projection: float = 1e-3
    min_pair_sine: float = 1e-3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("INVALID_COMPLETION_ENABLED")
        for name, low, high in (("max_parents", 1, 8192), ("max_completions_per_parent", 1, 4096),
                                ("max_completions_total", 1, 65536)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("INVALID_COMPLETION_"+name.upper())
        for name, low, high in (("max_shift_mm", 1e-3, 1e6), ("min_axis_projection", 1e-9, .5),
                                ("min_pair_sine", 1e-9, .5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or not low <= float(value) <= high:
                raise ValueError("INVALID_COMPLETION_"+name.upper())
            object.__setattr__(self, name, float(value))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _context_configuration(mode, budgets):
    """Closed V4 selector; scheduling and scan proposals remain independent."""
    _require(type(mode) is str and mode == "v4", "EXPLICIT_V4_CONTEXT_MODE_REQUIRED")
    owned = _context_v4.TwoPlaneContextBudgetsV4(**_context_v4._configuration(
        _context_v4.TwoPlaneContextBudgetsV4() if budgets is None else budgets))
    base = _context_v4._base()
    declaration = dict(mode="v4", implementation=CONTEXT_IMPLEMENTATION,
        schema_version="two_plane_context_v4", budgets=asdict(owned),
        **{key: base[key] for key in _CONTEXT_DECLARATIONS})
    return owned, declaration


def _context_configuration_snapshot(value):
    """Validate before encoding and return a fresh, bounded exact snapshot."""
    keys = {"mode", "implementation", "schema_version", "budgets", *_CONTEXT_DECLARATIONS}
    _require(type(value) is dict and set(value) == keys and type(value.get("budgets")) is dict,
        "INVALID_CONTEXT_V4_CONFIGURATION")
    cap = value["budgets"]
    expected_keys = {key for key, _ in _context_v4._MAXIMA}
    _require(set(cap) == expected_keys and all(type(item) is int for item in cap.values()),
        "INVALID_CONTEXT_V4_BUDGET_SNAPSHOT")
    _, expected = _context_configuration(value["mode"], _context_v4.TwoPlaneContextBudgetsV4(**cap))
    _require(all(type(value[key]) is type(expected[key]) and value[key] == expected[key]
        for key in keys-{"budgets"}), "CONTEXT_V4_IMPLEMENTATION_OR_ARITHMETIC_MISMATCH")
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
        "INITIALIZATION_V14_IMPLEMENTATION_REQUIRED")
    actual, actual_sha = _context_configuration_snapshot(publication.get("two_plane_context_configuration"))
    _require(actual == expected and actual_sha == sha
        and type(publication.get("two_plane_context_configuration_sha256")) is str
        and publication["two_plane_context_configuration_sha256"] == sha,
        "CONTEXT_V4_REQUESTED_RETURNED_CONFIGURATION_MISMATCH")
    return dict(initialization_implementation=INITIALIZATION_IMPLEMENTATION,
        two_plane_context_mode="v4", two_plane_context_implementation=CONTEXT_IMPLEMENTATION,
        two_plane_context_configuration_sha256=sha)


def _frozen_configuration_v4(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled=True,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    """Verbatim V12 configuration body; surface budgets are the V4 type here."""
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
        surface_budgets=_surface.TriangleSurfaceBudgetsV4(**asdict(surface)))




def _completion_configuration(policy, budgets, base_budgets):
    policy = CompletionLanePolicyV1() if policy is None else policy
    _require(type(policy) is CompletionLanePolicyV1, "INVALID_COMPLETION_POLICY_TYPE")
    policy = CompletionLanePolicyV1(**asdict(policy))
    if budgets is None:
        budgets = PlaneInitializationBudgetsV12(**asdict(base_budgets))
    _require(type(budgets) is PlaneInitializationBudgetsV12, "INVALID_COMPLETION_BUDGET_TYPE")
    cap = {key: getattr(budgets, key) for key in _LIMITS}
    _require(all(type(value) is int and 0 <= value <= _LIMITS[key] for key, value in cap.items())
        and cap["chunk_rows"] > 0, "INVALID_COMPLETION_BUDGET")
    return policy, PlaneInitializationBudgetsV12(**cap)


def _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
        cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled=True,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v4", two_plane_context_budgets=None,
        completion_policy=None, completion_budgets=None):
    _require(two_plane_enabled is True, "CONTEXT_V4_REQUIRES_ENABLED_TWO_PLANE_LANE")
    configuration = _frozen_configuration_v4(registration_policy, noisy_policy, noisy_budgets,
        cad_patch_policy, cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets,
        two_plane_enabled, scan_plane_mode, seeded_plane_budgets, composite_plane_budgets,
        two_plane_schedule_mode, two_plane_schedule_budgets)
    owned, declaration = _context_configuration(two_plane_context_mode, two_plane_context_budgets)
    configuration.update(two_plane_context_mode="v4", two_plane_context_budgets=owned,
        two_plane_context_configuration=declaration)
    policy, lane_budgets = _completion_configuration(completion_policy, completion_budgets, configuration["budgets"])
    configuration.update(completion_policy=policy, completion_budgets=lane_budgets)
    return configuration


def _base():
    result = _frozen._base()
    result["initialization_implementation"] = INITIALIZATION_IMPLEMENTATION
    # Additive label: the retained V12 work_unit_formula/counter strings are wire
    # contracts; V4 query work additionally charges 16 units per vertex lookup row.
    result["finite_surface_implementation"] = REGISTRATION_IMPLEMENTATION
    result["finite_surface_work_note"] = "V4_QUERY_WORK_ADDS_16_UNITS_PER_VERTEX_LOOKUP_ROW_TO_RETAINED_V3_FORMULA"
    result["completion_implementation"] = COMPLETION_IMPLEMENTATION
    result["completion_lane"] = dict(status="NOT_ATTEMPTED", implementation=COMPLETION_IMPLEMENTATION)
    return result


def _validate_two_plane_context_configuration(helper, requested):
    expected, _ = _context_configuration_snapshot(requested)
    _require(type(helper) is dict and helper.get("schema_version") == expected["schema_version"],
        "TWO_PLANE_CONTEXT_V4_SCHEMA_MISMATCH")
    cap = helper.get("budgets")
    _require(type(cap) is dict and set(cap) == set(expected["budgets"])
        and all(type(cap[key]) is int and cap[key] == value for key, value in expected["budgets"].items()),
        "TWO_PLANE_CONTEXT_V4_BUDGET_PROFILE_MISMATCH")
    for key in _CONTEXT_DECLARATIONS:
        _require(type(helper.get(key)) is type(expected[key]) and helper[key] == expected[key],
            "TWO_PLANE_CONTEXT_V4_ARITHMETIC_OR_SCOPE_MISMATCH")
    capacity = _context_v4._base()["scheduling_capacity"]
    actual = helper.get("scheduling_capacity")
    _require(type(actual) is dict and set(actual) == set(capacity)
        and all(type(actual[key]) is type(value) and actual[key] == value for key, value in capacity.items()),
        "TWO_PLANE_CONTEXT_V4_CAPACITY_MISMATCH")
    _require(helper.get("production_use_allowed") is False, "CONTEXT_V4_NOT_PRODUCTION_AUTHORITY")


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
    helper = _context_v4.propose_two_plane_context_v4(
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
                budgets=_surface.TriangleSurfaceBudgetsV4(**values))
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
            budgets=_surface.TriangleSurfaceBudgetsV4(**values))
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


def _unit_or_none(vector, floor):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < floor:
        return None
    return vector/length


def _completion_shift(scan_witness_world, scan_normal_world, cad_normal, cad_anchor, axis, policy):
    """Nominal axial shift placing the world scan witness on the candidate CAD plane.

    lambda = (c.a - c.y_s) / (c.u) with unit c the candidate face normal, a its anchor
    (offset zero), y_s the world scan witness and u the unit passing-pair axis. The sign
    of c cancels; the compatibility sign is chosen separately. Returns a record with the
    numerator (mm), denominator (dimensionless), lambda (mm), the post-construction
    residual and an unresolved disposition when the denominator or numerator vanish.
    """
    numerator = float(cad_normal@(cad_anchor-scan_witness_world))
    denominator = float(cad_normal@axis)
    record = dict(numerator_mm=numerator, denominator=denominator, shift_mm=None, residual_mm=None, disposition=None)
    if abs(denominator) < policy.min_axis_projection:
        record["disposition"] = ("AXIAL_FAMILY_UNCONSTRAINED_BY_CANDIDATE" if abs(numerator) <= 1e-9
                                 else "AXIAL_COMPLETION_UNRESOLVED_DENOMINATOR")
        return record
    shift = numerator/denominator
    if not math.isfinite(shift) or abs(shift) > policy.max_shift_mm:
        record["disposition"] = "SHIFT_LIMIT_REJECTED" if math.isfinite(shift) else "NONFINITE_SHIFT"
        return record
    record.update(shift_mm=float(shift), disposition="RESOLVED",
        residual_mm=float(cad_normal@(scan_witness_world+shift*axis-cad_anchor)))
    return record


def _candidate_sign(rotated_scan_normal, cad_normal):
    """Candidate-specific orientation sign and the atan2 alignment angle for that sign."""
    best = None
    for sign in (1, -1):
        target = sign*cad_normal
        angle = math.atan2(float(np.linalg.norm(np.cross(rotated_scan_normal, target))),
                           float(rotated_scan_normal@target))
        if best is None or angle < best[1]:
            best = (sign, angle)
    return best


def _completion_statistics():
    return dict(certificate_pairs=0, triangle_box_pairs=0, node_visits=0, triangle_pairs=0, normal_pairs=0,
        face_normal_tests=0, candidate_records=0, candidate_sort_work=0, pose_keys=0,
        work_units=0, box_screen_calls=0, box_screen_rows=0, completed_box_screen_calls=0,
        completed_box_screen_rows=0, query_calls=0, query_rows=0, completed_query_calls=0,
        completed_query_rows=0, face_build_nodes=0, face_build_triangle_visits=0,
        indexed_face_count=0, indexed_face_triangles=0, planned_query_rows=0, planned_query_calls=0,
        assigned_support_count=0, completed_box_support_count=0, bound_excluded_support_count=0,
        bound_excluded_support_rows=0, bound_excluded_query_calls=0,
        completed_finite_query_support_count=0, hypotheses_evaluated=0,
        passing_hypothesis_count=0, rejected_hypothesis_count=0)


def _evaluate_completion_supports(row, context, scans, query, chunk, policy, stats):
    """The V14 per-support box screen and finite query, applied to one completion row."""
    rotation, translation = _pose(row)
    row["prefilter_status"] = "INCOMPLETE"
    for support in row["assigned_support"]:
        raw = scans[support["scan_plane_id"]]["raw_vertex_rows"]
        support["status"] = "INCOMPLETE"
        support["box_screening_status"] = "INCOMPLETE"
        transformed = np.empty((len(raw), 3), dtype=np.float64)
        for start in range(0, len(raw), chunk):
            transformed[start:start+chunk] = context["scan"][raw[start:start+chunk]]@rotation.T+translation
            points = transformed[start:start+chunk]
            support["attempted_box_screen_rows"] += len(points)
            answer = query.screen(support["cad_plane_id"], points, policy.max_correspondence_distance_mm)
            support["completed_box_screen_rows"] += len(points)
            support["definitely_box_outside_count"] += answer["definitely_outside_count"]
        _require(support["completed_box_screen_rows"] == support["support_count"],
                 "INCOMPLETE_WHOLE_SUPPORT_BOX_SCREEN")
        decision = _box.decide_face_box_exclusion(support_count=support["support_count"],
            definitely_outside_count=support["definitely_box_outside_count"],
            minimum_coverage=policy.min_scan_coverage)
        support.update(box_screening_status="COMPLETE", box_exclusion_decision=decision["decision"],
            possible_box_inlier_count_upper=decision["possible_inlier_count_upper"],
            possible_box_coverage_upper=decision["possible_inlier_fraction_upper"])
        stats["completed_box_support_count"] += 1
        if decision["decision"] == "BOUND_EXCLUDED":
            support.update(status="REJECTED", finite_query_disposition="NOT_REQUIRED_BOUND_EXCLUDED",
                reasons=["ASSIGNED_WHOLE_FACE_BOX_COVERAGE_UPPER_BELOW_MINIMUM"])
            stats["bound_excluded_support_count"] += 1
            stats["bound_excluded_support_rows"] += support["support_count"]
            stats["bound_excluded_query_calls"] += (len(raw)+chunk-1)//chunk
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
            support["finite_unambiguous_inlier_count"] += int(np.count_nonzero(unambiguous & masks["definite_support"]))
            for key, name in (("definite_support", "definite_distance_inlier_rows"),
                              ("definite_outside", "definite_distance_outside_rows"),
                              ("unresolved", "threshold_unresolved_rows")):
                support[name] += int(np.count_nonzero(masks[key]))
            support["uncertain_region_rows"] += int(np.count_nonzero(answer["region_uncertain"]))
            for name, values in (
                ("maximum_distance_interval_width_mm", np.nextafter(answer["distance_upper"]-answer["distance_lower"], np.inf)),
                ("maximum_optimizer_error_mm", answer["closest_point_error"]),
                ("maximum_restoration_error_mm", answer["closest_point_restoration_error"])):
                support[name] = max(support[name], float(np.max(values)))
            support["ambiguous_rows"] += int(np.count_nonzero(~unambiguous))
            support["boundary_rows"] += int(np.count_nonzero(answer["closest_regions"] != "FACE"))
        _require(support["query_rows"] == support["support_count"], "INCOMPLETE_WHOLE_SUPPORT_FINITE_QUERY")
        support["finite_query_disposition"] = "COMPLETE"
        stats["completed_finite_query_support_count"] += 1
        support["coverage"] = _v3._fraction_bound(support["definite_distance_inlier_rows"],
                                                  support["support_count"], upper=False)
        passed = support["coverage"] >= policy.min_scan_coverage
        support.update(status="PASSED" if passed else "REJECTED",
            reasons=[] if passed else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"])
    passed = all(support["status"] == "PASSED" for support in row["assigned_support"])
    row.update(prefilter_status="PASSED" if passed else "REJECTED",
        prefilter_reasons=[] if passed else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
    stats["hypotheses_evaluated"] += 1
    stats["passing_hypothesis_count" if passed else "rejected_hypothesis_count"] += 1
    return passed, rotation, translation


def _fresh_support(match, patch, face):
    count = len(patch["raw_vertex_rows"])
    return dict(match, proposal_kind=patch["proposal_kind"], source_record_id=patch["source_record_id"],
        support_count=count, raw_vertex_rows_sha256=patch["raw_vertex_rows_sha256"],
        scan_triangle_rows_sha256=patch["triangle_rows_sha256"], cad_triangle_rows_sha256=face["triangle_rows_sha256"],
        query_rows=0, attempted_query_rows=0, finite_unambiguous_inlier_count=0, coverage=None,
        ambiguous_rows=0, boundary_rows=0, definite_distance_inlier_rows=0, definite_distance_outside_rows=0,
        threshold_unresolved_rows=0, uncertain_region_rows=0, maximum_distance_interval_width_mm=0.,
        maximum_optimizer_error_mm=0., maximum_restoration_error_mm=0., attempted_box_screen_rows=0,
        completed_box_screen_rows=0, definitely_box_outside_count=0, possible_box_inlier_count_upper=None,
        possible_box_coverage_upper=None, box_screening_status="NOT_ATTEMPTED", box_exclusion_decision=None,
        finite_query_disposition="NOT_ATTEMPTED", status="NOT_ATTEMPTED", reasons=[])


def _complete_two_of_three(context, result, configuration, scheduled, scans, faces, groups, lookup):
    """Bounded completion lane; see CompletionLanePolicyV1. Additive to the frozen main lane."""
    policy, lane_budget = configuration["completion_policy"], configuration["completion_budgets"]
    registration_policy, surface = configuration["policy"], configuration["surface_budgets"]
    lane = dict(status="NOT_ATTEMPTED", implementation=COMPLETION_IMPLEMENTATION, policy=asdict(policy),
        budgets=asdict(lane_budget), parents=[], hypotheses=[], statistics=_completion_statistics(),
        face_index_evidence=[], parent_count=0, scheduled_parent_count=0, omitted_parent_count=0,
        candidate_count=0, scheduled_completion_count=0, omitted_completion_count=0,
        axis_parallel_unresolved_count=0, shift_limit_rejected_count=0, normal_incompatible_count=0,
        passing_completion_count=0, new_start_group_count=0, duplicate_pose_count=0,
        parallel_passing_pair_count=0, axial_unconstrained_count=0, usable=True, ledger_bytes=None,
        ledger_bytes_scope=("ENCODED_LANE_JSON_BYTES_WITH_LEDGER_BYTES_NULL_CHECKED_AGAINST_MAX_REPORT_BYTES_"
                            "AFTER_THE_LAST_MUTATION_ON_EVERY_NON_FATAL_TERMINAL_PATH_INCLUDING_REFINEMENT_SCHEDULE"),
        ranking="NONEXHAUSTIVE_LOCAL_SCHEDULE_ABS_SHIFT_THEN_FACE_ID_NO_UNIQUENESS_OR_COVERAGE_CLAIM",
        combined_allowance="MAIN_INITIALIZATION_BUDGETS_PLUS_THIS_SEPARATE_COMPLETION_BUDGET_UNDER_UNCHANGED_OUTER_LIMITS",
        scope="ADDITIVE_PROPOSALS_RESCREENED_AND_REQUERIED_ON_ALL_THREE_SUPPORTS_NOT_ACCEPTED_REGISTRATION",
        coordinate_note="TRANSLATION_SHIFT_ALONG_UNIT_INTERSECTION_DIRECTION_OF_THE_TWO_PASSING_CAD_PLANES")
    result["completion_lane"] = lane
    if not policy.enabled:
        lane["status"] = "DISABLED_BY_EXPLICIT_POLICY"
        _cap_completion_lane(lane, lane_budget)
        return
    def _completed_parent(row):
        supports = row["assigned_support"]
        return (row.get("proposal_lane") == "TRIPLET" and row["prefilter_status"] == "REJECTED"
                and len(supports) == 3 and len(row["correspondences"]) == 3
                and sum(support["status"] == "PASSED" for support in supports) == 2
                and all(support["box_screening_status"] == "COMPLETE" for support in supports)
                and all(support["status"] in ("PASSED", "REJECTED") for support in supports)
                and all(support["finite_query_disposition"] in ("COMPLETE", "NOT_REQUIRED_BOUND_EXCLUDED")
                        for support in supports))
    parents = [row for row in scheduled if _completed_parent(row)]
    parents.sort(key=lambda row: row["hypothesis_index"])
    lane["parent_count"] = len(parents)
    lane["omitted_parent_count"] = max(0, len(parents)-policy.max_parents)
    lane["omitted_parent_hypothesis_indices"] = [row["hypothesis_index"] for row in parents[policy.max_parents:]]
    parents = parents[:policy.max_parents]
    lane["scheduled_parent_count"] = len(parents)
    lane["status"] = "INCOMPLETE"
    stats = lane["statistics"]
    candidate_rows, needed_faces = [], {}
    for parent in parents:
        rotation, translation = _pose(parent)
        passing = [support for support in parent["assigned_support"] if support["status"] == "PASSED"]
        failed = [support for support in parent["assigned_support"] if support["status"] != "PASSED"][0]
        failed_position = parent["assigned_support"].index(failed)
        record = dict(parent_hypothesis_index=parent["hypothesis_index"], status="EVALUATING",
            passing_support_positions=[parent["assigned_support"].index(support) for support in passing],
            passing_cad_plane_ids=[support["cad_plane_id"] for support in passing],
            failed_support_position=failed_position,
            failed_scan_plane_id=failed["scan_plane_id"], failed_cad_plane_id=failed["cad_plane_id"],
            failed_normal_sign=failed["normal_sign"], axis_direction=None, candidate_count=0,
            scheduled_count=0, omitted_count=0, axis_parallel_unresolved_count=0,
            axial_unconstrained_count=0, normal_incompatible_count=0, shift_limit_rejected_count=0,
            completion_indices=[])
        lane["parents"].append(record)
        m1 = _unit_or_none(faces[passing[0]["cad_plane_id"]]["normal"], 1e-12)
        m2 = _unit_or_none(faces[passing[1]["cad_plane_id"]]["normal"], 1e-12)
        _require(m1 is not None and m2 is not None, "INVALID_COMPLETION_CAD_NORMAL")
        axis = _unit_or_none(np.cross(m1, m2), policy.min_pair_sine)
        if axis is None:
            record["status"] = "PASSING_PAIR_PARALLEL_UNRESOLVED"
            lane["parallel_passing_pair_count"] += 1
            continue
        record["axis_direction"] = axis.tolist()
        analysis = scans[failed["scan_plane_id"]]["analysis"]
        n3 = _unit_or_none(analysis["normal"], 1e-12)
        _require(n3 is not None, "INVALID_COMPLETION_SCAN_NORMAL")
        witness = np.asarray(analysis["witness_anchor_mm"], dtype=np.float64)+analysis["witness_offset_from_anchor_mm"]*n3
        rotated = rotation@n3
        witness_world = rotation@witness+translation
        record.update(scan_witness_world_mm=witness_world.tolist(), rotated_scan_normal=rotated.tolist())
        allowances = parent.get("normal_alignment_allowances_radians")
        _require(type(allowances) is list and len(allowances) == 3 and all(
            type(value) is float and math.isfinite(value) and value >= 0. for value in allowances),
            "MISSING_PARENT_NORMAL_ALLOWANCE")
        allowance = allowances[failed_position]
        candidates = []
        for face_id in sorted(faces):
            if face_id in record["passing_cad_plane_ids"]:
                continue
            stats["face_normal_tests"] += 1
            mj = _unit_or_none(faces[face_id]["normal"], 1e-12)
            _require(mj is not None, "INVALID_COMPLETION_CAD_NORMAL")
            sign, angle = _candidate_sign(rotated, mj)
            if angle > allowance:
                record["normal_incompatible_count"] += 1
                continue
            shift = _completion_shift(witness_world, rotated, mj,
                                      np.asarray(faces[face_id]["area_centroid_mm"], dtype=np.float64), axis, policy)
            if shift["disposition"] == "AXIAL_COMPLETION_UNRESOLVED_DENOMINATOR":
                record["axis_parallel_unresolved_count"] += 1
                continue
            if shift["disposition"] == "AXIAL_FAMILY_UNCONSTRAINED_BY_CANDIDATE":
                record["axial_unconstrained_count"] += 1
                continue
            if shift["disposition"] in ("SHIFT_LIMIT_REJECTED", "NONFINITE_SHIFT"):
                record["shift_limit_rejected_count"] += 1
                continue
            candidates.append((abs(shift["shift_mm"]), face_id, sign, angle, shift))
        stats["candidate_records"] += len(candidates)
        stats["candidate_sort_work"] += len(candidates)*max(1, math.ceil(math.log2(len(candidates)))) if candidates else 0
        candidates.sort(key=lambda item: (item[0], item[1]))
        record["candidate_count"] = len(candidates)
        lane["axis_parallel_unresolved_count"] += record["axis_parallel_unresolved_count"]
        lane["axial_unconstrained_count"] = lane.get("axial_unconstrained_count", 0)+record["axial_unconstrained_count"]
        lane["normal_incompatible_count"] += record["normal_incompatible_count"]
        lane["shift_limit_rejected_count"] += record["shift_limit_rejected_count"]
        lane["candidate_count"] += len(candidates)
        scheduled_here = candidates[:policy.max_completions_per_parent]
        record["omitted_count"] = len(candidates)-len(scheduled_here)
        for _, face_id, sign, angle, shift in scheduled_here:
            if len(candidate_rows) >= policy.max_completions_total:
                record["omitted_count"] += 1
                continue
            new_translation = translation+shift["shift_mm"]*axis
            _require(np.isfinite(new_translation).all(), "NONFINITE_COMPLETION_TRANSLATION")
            correspondences = [dict(item) for item in parent["correspondences"]]
            correspondences[failed_position] = dict(scan_plane_id=failed["scan_plane_id"], cad_plane_id=face_id,
                                                    normal_sign=int(sign))
            signs = list(parent["normal_signs"])
            signs[failed_position] = int(sign)
            _require(len({item["scan_plane_id"] for item in correspondences}) == 3
                and len({item["cad_plane_id"] for item in correspondences}) == 3, "INVALID_COMPLETION_BINDING")
            row = dict(completion_index=len(candidate_rows), parent_hypothesis_index=parent["hypothesis_index"],
                proposal_lane=COMPLETION_LANE, parent_unsigned_assignment_id=parent["unsigned_assignment_id"],
                normal_signs=signs, rotation=rotation.tolist(),
                translation_mm=new_translation.tolist(), correspondences=correspondences,
                replaced_cad_plane_id=failed["cad_plane_id"], completed_cad_plane_id=face_id,
                completed_normal_sign=int(sign), alignment_angle_radians=float(angle),
                alignment_allowance_radians=float(allowance), shift_mm=float(shift["shift_mm"]),
                shift_numerator_mm=shift["numerator_mm"], shift_denominator=shift["denominator"],
                shift_residual_mm=shift["residual_mm"], axis_direction=axis.tolist(),
                cad_anchor_mm=list(faces[face_id]["area_centroid_mm"]),
                status="PROPOSED", prefilter_status="NOT_ATTEMPTED", prefilter_reasons=[],
                prefilter_scheduled=True, assigned_support=[], exact_start_group_index=None)
            candidate_rows.append(row)
            # Publish every selected child BEFORE any fallible work; statuses mutate in place.
            lane["hypotheses"].append(row)
            record["completion_indices"].append(row["completion_index"])
            record["scheduled_count"] += 1
            for match in correspondences:
                needed_faces[match["cad_plane_id"]] = faces[match["cad_plane_id"]]
        if record["scheduled_count"]:
            record["status"] = "SCHEDULED"
        elif candidates:
            record["status"] = "OMITTED_BY_TOTAL_COMPLETION_CAP"
        elif record["axis_parallel_unresolved_count"] or record["axial_unconstrained_count"] or record["shift_limit_rejected_count"]:
            record["status"] = "ALL_COMPATIBLE_CANDIDATES_UNRESOLVED_OR_OUT_OF_RANGE"
        else:
            record["status"] = "NO_NORMAL_COMPATIBLE_THIRD_FACE"
    lane["scheduled_completion_count"] = len(candidate_rows)
    lane["omitted_completion_count"] = lane["candidate_count"]-len(candidate_rows)
    _require(len(candidate_rows) <= lane_budget.max_hypothesis_records, "COMPLETION_HYPOTHESIS_RECORD_BUDGET_EXCEEDED")
    if not candidate_rows:
        lane["status"] = "COMPLETE"
        _cap_completion_lane(lane, lane_budget)
        return
    chunk = min(lane_budget.chunk_rows, surface.max_query_rows)
    _require(chunk > 0, "COMPLETION_QUERY_ROW_CAP_ZERO")
    # Preflight the planned row/call denominators from support sizes before any heavy
    # per-support binding work; then bind each distinct scan patch and face once.
    for row in candidate_rows:
        for match in row["correspondences"]:
            count = len(scans[match["scan_plane_id"]]["raw_vertex_rows"])
            stats["planned_query_rows"] += count
            stats["planned_query_calls"] += (count+chunk-1)//chunk
            stats["assigned_support_count"] += 1
    _require(stats["planned_query_rows"] <= lane_budget.max_prefilter_query_rows
        and stats["planned_query_calls"] <= lane_budget.max_prefilter_query_calls
        and stats["planned_query_rows"] <= lane_budget.max_work_units, "COMPLETION_PREFILTER_PREFLIGHT_BUDGET_EXCEEDED")
    bound_scans = set()
    for row in candidate_rows:
        for match in row["correspondences"]:
            patch = scans[match["scan_plane_id"]]
            if match["scan_plane_id"] not in bound_scans:
                _bound_scan(context, patch)
                bound_scans.add(match["scan_plane_id"])
            row["assigned_support"].append(_fresh_support(match, patch, faces[match["cad_plane_id"]]))
    _require(len(needed_faces) <= lane_budget.max_cached_cad_faces
        and sum(len(face["triangle_rows"]) for face in needed_faces.values()) <= lane_budget.max_indexed_face_triangles,
        "COMPLETION_FACE_INDEX_PREFLIGHT_BUDGET_EXCEEDED")
    for face in needed_faces.values():
        _old._bound_support(context, cad_patch=face)
    shim = dict(statistics=stats, face_index_evidence=lane["face_index_evidence"])
    query = _FaceQueries(context, needed_faces, lane_budget, surface, shim)
    for row in candidate_rows:
        passed, rotation, translation = _evaluate_completion_supports(row, context, scans, query, chunk,
                                                                      registration_policy, stats)
        if not passed:
            continue
        lane["passing_completion_count"] += 1
        key = rotation.tobytes()+translation.tobytes()
        stats["pose_keys"] += 1
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append(dict(exact_group_index=len(groups), initialized_rotation=rotation.tolist(),
                initialized_translation_mm=translation.tolist(), hypothesis_indices=[],
                refinement_disposition="NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA",
                lane=COMPLETION_LANE, completion_indices=[]))
            lane["new_start_group_count"] += 1
        else:
            lane["duplicate_pose_count"] += 1
        group = groups[lookup[key]]
        group.setdefault("completion_indices", []).append(row["completion_index"])
        row["exact_start_group_index"] = group["exact_group_index"]
    _require(stats["hypotheses_evaluated"] == len(candidate_rows)
        and stats["completed_box_support_count"] == stats["assigned_support_count"]
        and stats["completed_finite_query_support_count"]+stats["bound_excluded_support_count"]
            == stats["assigned_support_count"], "COMPLETION_SUPPORT_DENOMINATOR_INCOMPLETE")
    lane["status"] = "COMPLETE"
    _cap_completion_lane(lane, lane_budget)


def _cap_completion_lane(lane, lane_budget):
    """Final completion-lane byte cap, re-applied after every mutation of the published lane.

    `ledger_bytes` is the compact JSON size of the lane measured with this field null; a violation
    is a whole-initialization refusal through the caller's existing fatal path (never a raised cap).
    """
    lane["ledger_bytes"] = None
    encoded = 0
    for piece in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(lane):
        encoded += len(piece.encode())
        _require(encoded <= lane_budget.max_report_bytes, "COMPLETION_LEDGER_REPORT_BUDGET_EXCEEDED")
    lane["ledger_bytes"] = encoded


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
    main_group_count = len(groups)
    _complete_two_of_three(context, result, configuration, scheduled, scans, faces, groups, lookup)
    stats.update(main_passing_start_group_count=main_group_count,
                 completion_passing_start_group_count=len(groups)-main_group_count)
    # The frozen strata schedule applies to the main-lane groups exactly as in V14;
    # completion-only groups fill the remaining refinement slots by their own strata
    # and can never displace a main-lane group.
    indices = list(_search._strata(main_group_count, budget.max_refined_start_groups))
    remaining = budget.max_refined_start_groups-len(indices)
    completion_groups = len(groups)-main_group_count
    completion_indices = (list(_search._strata(completion_groups, remaining)) if remaining > 0 and completion_groups else [])
    indices.extend(main_group_count+index for index in completion_indices)
    result["completion_lane"]["refinement_schedule"] = dict(main_groups=main_group_count,
        main_scheduled=len(indices)-len(completion_indices), completion_groups=completion_groups,
        completion_scheduled=len(completion_indices), completion_omitted=completion_groups-len(completion_indices),
        rule="MAIN_LANE_STRATA_FIRST_THEN_COMPLETION_STRATA_IN_REMAINING_SLOTS")
    _cap_completion_lane(result["completion_lane"], configuration["completion_budgets"])
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
        lane = result.get("completion_lane")
        if type(lane) is dict:
            lane.update(usable=False, status="INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE"
                        if lane.get("status") != "NOT_ATTEMPTED" else "NOT_ATTEMPTED",
                        attempted_child_count=sum(1 for row in lane.get("hypotheses", [])
                                                  if row.get("prefilter_status") not in ("NOT_ATTEMPTED",)),
                        passing_children_invalidated=True)
        for group in result["all_passing_start_groups"]:
            group.update(refinement_disposition="INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE",
                attempted_initialized_rotation=group["initialized_rotation"],
                attempted_initialized_translation_mm=group["initialized_translation_mm"],
                initialized_rotation=None, initialized_translation_mm=None)
        context = None
    result = _bounded_report(result, configuration["budgets"].max_report_bytes)
    return result, context if result["status"] == "COMPLETE" else None


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


def _schedule_measurement_contexts(result, refine, policy, maximum):
    """Bounded, label-free selection among V12-published SUPPORTED contexts.

    After the unchanged V12 publication (every SUPPORTED candidate becomes a
    measurement context), rank the supported contexts by their own retained
    evidence (context p95 upper ascending, geometric coverage lower descending,
    candidate index ascending), collapse near-duplicate poses using the frozen
    V2 engineering action metric over the refinement rows
    (`pose_class_distance_mm`), then keep at most `maximum` contexts. The
    pose-class collapse applies regardless of the cap (also when `maximum` is
    None): near-identical windows would otherwise be merged by the cover
    planner at the cost of their measurement work. Omitted contexts keep their
    complete evidence and receive an explicit status; the retained measurement
    consumer recomputes its index list from the SUPPORTED_CONTEXT status, so
    the published list and that recomputation agree. This is a capacity
    schedule for the frozen measurement preflight budgets, not pose
    acceptance, uniqueness or a correspondence certificate.
    """
    evidence = result["evidence"]
    candidates = evidence["candidates"]
    supported = list(evidence["measurement_context_indices"])
    _require(maximum is None or (type(maximum) is int and 1 <= maximum <= 48),
             "INVALID_MEASUREMENT_CONTEXT_CAP")
    policy = _v3._policy_snapshot(policy)
    refine = _v1._positions(refine, 20000)
    _require(all(type(index) is int and 0 <= index < len(candidates)
        and candidates[index]["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"
        for index in supported), "INVALID_PUBLISHED_SUPPORTED_CONTEXT_LEDGER")
    def rank(index):
        support = candidates[index]["conditional_context_support"]
        p95 = support["finite_distance_metrics"]["p95_upper"]
        coverage = support["geometric_scan_coverage_lower"]
        _require(type(p95) in (int, float) and math.isfinite(p95) and p95 >= 0
            and type(coverage) in (int, float) and math.isfinite(coverage) and 0 <= coverage <= 1,
            "INVALID_SUPPORTED_CONTEXT_RANK_EVIDENCE")
        return (float(p95), -float(coverage), index)
    order = sorted(supported, key=rank)
    actions = {}
    for index in order:
        rotation, translation = _v3._candidate_pose(candidates[index])
        actions[index] = refine@rotation.T+translation
        _require(np.isfinite(actions[index]).all(), "NONFINITE_CONTEXT_POSE_ACTION")
    kept, dispositions = [], []
    for index in order:
        duplicate = next((prior for prior in kept
            if float(np.max(_v3._norm(actions[index]-actions[prior]))) <= policy.pose_class_distance_mm), None)
        if duplicate is not None:
            dispositions.append(dict(index=index, disposition="DUPLICATE_POSE_CLASS", of_index=duplicate))
        elif maximum is not None and len(kept) >= maximum:
            dispositions.append(dict(index=index, disposition="OMITTED_BY_MEASUREMENT_BUDGET"))
        else:
            kept.append(index)
            dispositions.append(dict(index=index, disposition="SCHEDULED"))
    kept = sorted(kept)
    _require(bool(kept) == bool(supported), "MEASUREMENT_CONTEXT_SCHEDULE_LOST_ALL_SUPPORTED")
    # Relabel only after the complete schedule is decided; every supported
    # candidate keeps its evidence and records that it was supported first.
    for row in dispositions:
        support = candidates[row["index"]]["conditional_context_support"]
        support["supported_before_schedule"] = True
        if row["disposition"] == "DUPLICATE_POSE_CLASS":
            support.update(status=DUPLICATE_CONTEXT, duplicate_of_context_index=row["of_index"])
        elif row["disposition"] == "OMITTED_BY_MEASUREMENT_BUDGET":
            support["status"] = OMITTED_CONTEXT
    evidence.update(measurement_context_indices=kept,
        measurement_context_schedule_policy=MEASUREMENT_CONTEXT_SCHEDULE,
        measurement_context_schedule=dict(policy=MEASUREMENT_CONTEXT_SCHEDULE,
            ranking="CONTEXT_P95_UPPER_ASC_GEOMETRIC_COVERAGE_LOWER_DESC_INDEX_ASC",
            pose_class_distance_mm=policy.pose_class_distance_mm, pose_class_rows=len(refine),
            pose_class_collapse="APPLIES_REGARDLESS_OF_CAP",
            max_measurement_contexts=maximum, supported_context_count=len(supported),
            scheduled_context_count=len(kept), dispositions=dispositions,
            scope="DETERMINISTIC_EVIDENCE_RANKED_CAPACITY_SCHEDULE_NOT_POSE_ACCEPTANCE_OR_UNIQUENESS"))


def prefilter_plane_initializations_v15(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm, registration_policy,
        unit_assumption, noisy_policy=_noisy.NoisyMeshPlanePolicyV5(),
        noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(), cad_patch_policy=_patch.CoplanarPatchPolicyV4(),
        cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(), search_policy=_search.PlaneSearchPolicyV6(),
        search_budgets=_search.PlaneSearchBudgetsV6(), budgets=PlaneInitializationBudgetsV12(),
        surface_budgets=None, two_plane_enabled=True, scan_plane_mode="v7",
        seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v4", two_plane_context_budgets=None,
        completion_policy=None, completion_budgets=None):
    """Owned-array scheduled finite-face evidence; no accepted registration."""
    result, maximum = _base(), _LIMITS["max_report_bytes"]
    try:
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        configuration = _configuration(registration_policy, noisy_policy, noisy_budgets, cad_patch_policy,
            cad_patch_budgets, search_policy, search_budgets, budgets, surface_budgets, two_plane_enabled,
            scan_plane_mode, seeded_plane_budgets, composite_plane_budgets, **schedule_request,
            two_plane_context_mode=two_plane_context_mode, two_plane_context_budgets=two_plane_context_budgets,
            completion_policy=completion_policy, completion_budgets=completion_budgets)
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


def register_whole_part_planes_v15(scan_points, scan_triangle_raw_rows, cad_triangles,
        cad_face_ids, cad_source_triangle_ids, *, position_error_bounds_mm,
        cad_initialization_points, registration_policy, unit_assumption, registration_raw_rows=None,
        noisy_policy=_noisy.NoisyMeshPlanePolicyV5(), noisy_budgets=_noisy.NoisyMeshPlaneBudgetsV5(),
        cad_patch_policy=_patch.CoplanarPatchPolicyV4(), cad_patch_budgets=_patch.CoplanarPatchBudgetsV4(),
        search_policy=_search.PlaneSearchPolicyV6(), search_budgets=_search.PlaneSearchBudgetsV6(),
        budgets=PlaneInitializationBudgetsV12(), surface_budgets=None, registration_budgets=None, two_plane_enabled=True,
        connected_measurement_contexts=False, _return_owned_measurement_support=False,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v4", two_plane_context_budgets=None,
        max_measurement_contexts=None, completion_policy=None, completion_budgets=None):
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
    evidence["registration_implementation"] = REGISTRATION_IMPLEMENTATION
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
        residual_gate_value_scope="POLICY_FIELD_max_inlier_plane_p95_mm_CALLER_DECLARED_SEE_policy_SNAPSHOT",
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
            two_plane_context_mode=two_plane_context_mode, two_plane_context_budgets=two_plane_context_budgets,
            completion_policy=completion_policy, completion_budgets=completion_budgets)
        evidence.update(_context_publication(configuration))
        evidence["completion_implementation"] = COMPLETION_IMPLEMENTATION
        evidence["completion_policy"] = asdict(configuration["completion_policy"])
        if configuration["scan_plane_mode"] == "composite_v1":
            evidence["scan_plane_configuration"] = dict(configuration["scan_plane_configuration"])
        if configuration.get("two_plane_schedule_mode", "v1") == "v2":
            evidence["two_plane_schedule_configuration"] = dict(mode="v2",
                budgets=dict(configuration["two_plane_schedule_configuration"]["budgets"]))
        maximum = configuration["budgets"].max_report_bytes
        execution = _v3._validated_execution_budgets(_v3.SurfaceRegistrationBudgetsV4()
            if registration_budgets is None else registration_budgets)
        execution = _v3.SurfaceRegistrationBudgetsV4(**asdict(execution))
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
            if result["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS":
                _schedule_measurement_contexts(result, refine, policy, max_measurement_contexts)
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


def register_measurement_contexts_v15(*args, **kwargs):
    """Return JSON registration evidence and its owned scan-support transport.

    The opaque second value retains existing extraction's exact support arrays;
    descriptor hashes are never inverted or substituted for original row IDs.
    It is not part of the JSON report and supplies no external pose or labels.
    """
    _require("_return_owned_measurement_support" not in kwargs
        and kwargs.get("connected_measurement_contexts", True) is True,
        "CONNECTED_CONTEXT_API_REQUIRED")
    kwargs["connected_measurement_contexts"] = True
    return register_whole_part_planes_v15(*args, **kwargs, _return_owned_measurement_support=True)
