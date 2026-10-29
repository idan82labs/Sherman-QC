"""NEW unqualified V17 implementation: V16 text plus the native feature bridge V1 (explicit lossless
kernel-V1 -> kernel-V2 feature reconstruction immediately after the UNCHANGED native factory, before any
costly registration; both r0 and v1 factory branches). Frozen V16 stays unchanged.

Original V16 header: V15 text on initializer V15 (completion lane V1),
measurement kernel V2 (declared layer selection, diagnostics) via context measurement V2 and
support-cover measurement V3. Frozen V15 stays unchanged.

Original V15 header: V14 text on initializer V14 (context V4, surface V4).

Both prospective successor arms explicitly use context V4 and the V4 finite
surface/registration consumer; the comparator is new, not the historical paired
control. Frozen V12/V13/V14 modules stay unchanged. The retained V13 wire schema
is a contract for the unchanged measurement and contact projector. Workflow,
initialization and registration implementation identities and the context
configuration are mandatory separate bindings; this adapter does not claim
historical execution.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_mesh_offline_v12 as _v12
from . import cad_guided_plane_initialization_v16 as _initialize_v15   # alias kept for the verbatim V16 text; bound to V16 here
from . import cad_guided_context_measurement_v3 as _context3
from .cad_guided_measurement_v2 import LayerSelectionPolicyV1
from . import cad_guided_stage_policies_v2 as _stages
from . import cad_guided_native_feature_bridge_v1 as _bridge

_v2 = _v12._v2
MAX_WORKFLOW_REPORT_BYTES = _v12.MAX_WORKFLOW_REPORT_BYTES
_triangle_scan_binding = _v12._triangle_scan_binding
_registration_selection = _v12._registration_selection
_previous = _v12._previous
_PreparedScanPlaneBinding = _v12._PreparedScanPlaneBinding
_SCAN_PLANE_BINDING_TOKEN = _v12._SCAN_PLANE_BINDING_TOKEN
_scan_plane_configuration_snapshot = _v12._scan_plane_configuration_snapshot
_scan_plane_binding = _v12._scan_plane_binding

WORKFLOW_IMPLEMENTATION = "CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1"
ALLOWANCE_IMPLEMENTATION = _initialize_v15.ALLOWANCE_IMPLEMENTATION
PUBLICATION_CONTRACT = _initialize_v15.PUBLICATION_CONTRACT
NATIVE_FEATURE_BRIDGE_IMPLEMENTATION = _bridge.BRIDGE_IMPLEMENTATION
KERNEL_IMPLEMENTATION = "CAD_GUIDED_MEASUREMENT_V2_LAYER_SELECTION_AND_DIAGNOSTICS"
COMPLETION_IMPLEMENTATION = _initialize_v15.COMPLETION_IMPLEMENTATION
REGISTRATION_IMPLEMENTATION = _initialize_v15.REGISTRATION_IMPLEMENTATION
SUCCESSOR_COMPARATOR = "FRESH_V17_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2"
CONTEXT_REQUEST_SCOPE = "BOTH_SUCCESSOR_ARMS_CONTEXT_V4_NOT_HISTORICAL_PAIRED_CONTROL"
# Declarative comparison only; future runner freezes complete budgets/policies.
# Fields: arm, scan-plane mode, schedule mode, context mode, native adapter.
SUCCESSOR_ARM_MECHANISMS = (
    ("control", "v7", "v1", "v4", "v1"),
    ("challenger", "composite_v1", "v2", "v4", "v1"),
)


@dataclass(frozen=True)
class MeshWorkflowPolicyV17(_v2.MeshWorkflowPolicyV2):
    """V2 policy fields plus the declared measurement-context schedule maximum."""
    max_measurement_contexts: int = 8

    def __post_init__(self):
        from .cad_guided_mesh_offline_v1 import MeshWorkflowPolicyV1
        values = asdict(self)
        maximum = values.pop("max_measurement_contexts")
        MeshWorkflowPolicyV1(**values)
        if type(maximum) is not int or not 1 <= maximum <= 48:
            raise ValueError("INVALID_MAX_MEASUREMENT_CONTEXTS")


def _two_plane_schedule_binding(registration, expected_configuration):
    # V4 context capacity is not a V2 scheduler marker. Inspect scheduling
    # metadata directly; the separate mandatory context binding handles V4.
    if expected_configuration is not None:
        return _v12._two_plane_schedule_binding(registration, expected_configuration)
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    schedule = initialization.get("two_plane_schedule")
    _v2._require("two_plane_schedule_configuration" not in evidence
        and "two_plane_schedule_configuration" not in initialization
        and not (type(schedule) is dict and (schedule.get("schema_version") == "two_plane_schedule_v2"
            or any(key in schedule for key in ("breadth_ledger", "residual_windows", "residual_statistics")))),
        "UNREQUESTED_V2_TWO_PLANE_SCHEDULE_PUBLICATION")
    return None


def _validate_completion_origins(initialization, candidate, gi, completion_indices):
    """Bind every completion origin of a selected group to its published child and parent.

    Each child must be a PASSED TRIPLET_COMPLETION row whose exact start group is this group,
    whose rotation equals the candidate's initialized rotation and whose translation equals
    parent translation + shift * axis, whose three refreshed supports all PASSED with three
    distinct scan and CAD ids, and whose parent is a completed REJECTED main TRIPLET row with
    exactly two PASSED supports and the same rotation. Nothing is inferred from an empty
    main-origin list; a missing, stale or spoofed origin refuses.
    """
    lane = initialization.get("completion_lane")
    _v2._require(type(lane) is dict and lane.get("status") == "COMPLETE" and lane.get("usable") is True
        and lane.get("implementation") == _initialize_v15.COMPLETION_IMPLEMENTATION
        and type(lane.get("hypotheses")) is list, "INCOMPLETE_COMPLETION_LANE_PUBLICATION")
    children = lane["hypotheses"]
    hypotheses = initialization["hypotheses"]
    _v2._require(type(completion_indices) is list and completion_indices
        and len(set(completion_indices)) == len(completion_indices), "MISSING_SELECTED_COMPLETION_ORIGINS")
    for index in completion_indices:
        _v2._require(type(index) is int and 0 <= index < len(children), "INVALID_SELECTED_COMPLETION_ORIGIN")
        child = children[index]
        _v2._require(child["completion_index"] == index and child["proposal_lane"] == _initialize_v15.COMPLETION_LANE
            and child["prefilter_status"] == "PASSED" and child["exact_start_group_index"] == gi
            and len(child["assigned_support"]) == 3 and len(child["correspondences"]) == 3
            and len({row["scan_plane_id"] for row in child["correspondences"]}) == 3
            and len({row["cad_plane_id"] for row in child["correspondences"]}) == 3
            and all(all(type(a.get(key)) is int and a[key] == b[key] for key in ("scan_plane_id", "cad_plane_id", "normal_sign"))
                    for a, b in zip(child["assigned_support"], child["correspondences"]))
            and all(row["status"] == "PASSED" for row in child["assigned_support"])
            and np.array_equal(child["rotation"], candidate["initialized_rotation"])
            and np.array_equal(child["translation_mm"], candidate["initialized_translation_mm"]),
            "SELECTED_START_NOT_BOUND_TO_PASSED_COMPLETION_SUPPORT")
        parent_index = child["parent_hypothesis_index"]
        _v2._require(type(parent_index) is int and 0 <= parent_index < len(hypotheses), "INVALID_COMPLETION_PARENT_ORIGIN")
        parent = hypotheses[parent_index]
        rejected = [position for position, row in enumerate(parent["assigned_support"]) if row["status"] != "PASSED"]
        _v2._require(parent["hypothesis_index"] == parent_index and parent.get("proposal_lane") == "TRIPLET"
            and parent["prefilter_status"] == "REJECTED" and len(parent["assigned_support"]) == 3
            and len(parent["correspondences"]) == 3 and len(parent["normal_signs"]) == 3 and len(rejected) == 1
            and all(a["scan_plane_id"] == b["scan_plane_id"] and a["cad_plane_id"] == b["cad_plane_id"]
                    for a, b in zip(parent["assigned_support"], parent["correspondences"]))
            and np.array_equal(parent["rotation"], child["rotation"]),
            "COMPLETION_PARENT_NOT_A_COMPLETED_TWO_OF_THREE_TRIPLET")
        # Exact replacement mapping: the unique rejected parent slot is the only slot that changes;
        # its scan plane is kept, its CAD face/sign become the child's completed face/sign, and the
        # two passing correspondences (and their signs) are carried over unchanged.
        position = rejected[0]
        failed = parent["assigned_support"][position]
        passing_ids = {row["cad_plane_id"] for row in parent["assigned_support"] if row["status"] == "PASSED"}
        _v2._require(failed["status"] == "REJECTED" and len(child["normal_signs"]) == 3
            and child["replaced_cad_plane_id"] == failed["cad_plane_id"]
            and child["correspondences"][position]["scan_plane_id"] == failed["scan_plane_id"]
            and child["correspondences"][position]["cad_plane_id"] == child["completed_cad_plane_id"]
            and child["correspondences"][position]["normal_sign"] == child["completed_normal_sign"]
                == child["normal_signs"][position]
            and child["completed_cad_plane_id"] not in passing_ids
            and all(child["correspondences"][slot] == parent["correspondences"][slot]
                    and child["normal_signs"][slot] == parent["normal_signs"][slot]
                    for slot in range(3) if slot != position),
            "COMPLETION_REPLACEMENT_NOT_BOUND_TO_PARENT_REJECTED_SLOT")
        parent_records = [row for row in lane["parents"] if row["parent_hypothesis_index"] == parent_index]
        _v2._require(len(parent_records) == 1 and parent_records[0]["failed_support_position"] == position
            and parent_records[0]["failed_cad_plane_id"] == failed["cad_plane_id"]
            and parent_records[0]["failed_scan_plane_id"] == failed["scan_plane_id"]
            and parent_records[0]["axis_direction"] == child["axis_direction"]
            and index in parent_records[0]["completion_indices"],
            "COMPLETION_PARENT_LEDGER_NOT_BOUND_TO_CHILD")
        expected = np.asarray(parent["translation_mm"], dtype=np.float64) \
            + float(child["shift_mm"])*np.asarray(child["axis_direction"], dtype=np.float64)
        _v2._require(np.array_equal(expected, np.asarray(child["translation_mm"], dtype=np.float64)),
                     "COMPLETION_POSE_NOT_PARENT_PLUS_SHIFT")
    _v2._require(completion_indices == [row["completion_index"] for row in children
        if row.get("exact_start_group_index") == gi and row.get("prefilter_status") == "PASSED"],
        "SELECTED_GROUP_COMPLETION_ORIGIN_LEDGER_INCOMPLETE")


def _array_binding_v17(registration, scan, topology, cad, selection, position_error_bound_mm):
    """The V8 array-binding seam text with ONE change: the initializer-V16 registration is accepted either
    with strict whole-set completeness or with the explicit finite-subset contract (validated by the
    context-V3 rule); every other binding check is verbatim."""
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    _v2._require(initialization["status"] == "COMPLETE" and initialization["search_complete"] is True
        and initialization["selected_policy_evaluation_complete"] is True
        and initialization["search_exhaustive"] is False
        and _context3._registration_evaluation_acceptable(evidence),
                 "INCOMPLETE_INITIALIZATION_CANNOT_REGISTER")
    expected = dict(position_error_bounds_mm_sha256=_v2._digest(np.full(len(scan["raw_points"]),
            position_error_bound_mm, dtype="<f8")), scan_points_sha256=_v2._digest(scan["raw_points"]),
        scan_triangle_raw_rows_sha256=_v2._digest(topology["triangle_raw_rows"]),
        cad_triangles_sha256=_v2._digest(cad["triangles"]),
        cad_face_ids_sha256=_v2._digest(cad["source_face_indices"]),
        cad_source_triangle_ids_sha256=_v2._digest(cad["source_triangle_indices"]))
    _v2._require(initialization["input_digests"] == expected
        and evidence["input_digests"] == expected, "REGISTRATION_GEOMETRY_BINDING_MISMATCH")
    selected = evidence["registration_raw_rows"]
    _v2._require(evidence["prefilter_full_scan_raw_rows"] == len(scan["raw_points"])
        and evidence["registration_input_rows"] == len(scan["points"])
        and evidence["registration_input_points_sha256"] == _v2._digest(scan["points"])
        and selected["count"] == len(scan["points"]), "REGISTRATION_SCAN_SELECTION_BINDING_MISMATCH")
    if selection is None:
        _v2._require(selected["selection"] == "ALL_RAW_ROWS" and selected["sha256"] is None,
                     "REGISTRATION_FULL_ROW_MODE_MISMATCH")
    else:
        _v2._require(selected["selection"] == "EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING"
            and selected["sha256"] == _v2._digest(selection), "REGISTRATION_SELECTED_ROW_MODE_MISMATCH")
    canonical_cad = np.unique(cad["points"], axis=0)
    canonical_digest = _v2._digest(np.ascontiguousarray(canonical_cad, dtype="<f8"))
    _v2._require(evidence["cad_initialization_points_sha256"] == canonical_digest,
                 "REGISTRATION_CAD_SAMPLE_BINDING_MISMATCH")
    return dict(status="CONSISTENT_REGISTRATION_ARRAY_BINDING",
        raw_scan_points_sha256=expected["scan_points_sha256"],
        raw_triangle_rows_sha256=expected["scan_triangle_raw_rows_sha256"],
        position_error_bounds_mm_sha256=expected["position_error_bounds_mm_sha256"],
        cad_triangles_sha256=expected["cad_triangles_sha256"],
        cad_sampler_points_sha256=_v2._digest(cad["points"]),
        registration_canonical_cad_points_sha256=canonical_digest,
        registration_selection="ALL_RAW_ROWS" if selection is None else "BOUND_V1_RAW_ROW_SELECTION",
        registration_evaluation=("STRICT_WHOLE_SET" if evidence.get("candidate_evaluation_complete") is True
                                 else "EVALUATED_ALLOWANCE_SUBSET_EXPIRED_AND_OMITTED_UNKNOWN"),
        scope="EXACT_IN_PROCESS_ARRAY_BINDING_NOT_CALIBRATION_OR_NATIVE_AUTHORIZATION")


def _registration_binding_with_completion_origins(registration, scan, topology, cad, selection,
        position_error_bound_mm, *, context_index):
    """The predecessor array binding (unchanged), plus the selected group's origin-lane label.

    V16's own origin block below validates main origins exactly as V12 did and, for groups
    with completion origins, validates every completion child against its published parent.
    A completion-only group has `hypothesis_indices == []` and is never given a fabricated
    main origin.
    """
    binding = _array_binding_v17(registration, scan, topology, cad, selection, position_error_bound_mm)
    evidence = registration["evidence"]
    selected = evidence["selected_candidate_index"] if context_index is None else context_index
    _v2._require(type(selected) is int and 0 <= selected < len(evidence["candidates"]),
                 "INVALID_BOUND_AWARE_SELECTED_CANDIDATE")
    candidate = evidence["candidates"][selected]
    lanes = (["MAIN"] if candidate.get("hypothesis_indices") else []) + \
            (["TRIPLET_COMPLETION"] if candidate.get("completion_indices") else [])
    binding["start_origin_lanes"] = lanes
    return binding


def _registration_binding(registration, scan, topology, cad, selection, position_error_bound_mm, *, context_index=None,
        expected_scan_plane_configuration=None, _prepared_scan_plane_binding=None,
        expected_two_plane_schedule_configuration=None, expected_two_plane_context_configuration=None):
    context_binding = _initialize_v15._bind_context_publication(
        registration["evidence"], expected_two_plane_context_configuration)
    _initialize_v15._bind_context_publication(
        registration["evidence"]["initialization"], expected_two_plane_context_configuration)
    _v2._require(registration["evidence"].get("registration_implementation") == REGISTRATION_IMPLEMENTATION,
        "REGISTRATION_V4_IMPLEMENTATION_REQUIRED")
    schedule_binding = _two_plane_schedule_binding(registration, expected_two_plane_schedule_configuration)
    if _prepared_scan_plane_binding is None:
        prepared_scan = _scan_plane_binding(registration, scan, topology, position_error_bound_mm,
                                           expected_scan_plane_configuration)
    else:
        prepared_scan = _prepared_scan_plane_binding
        _, canonical = _scan_plane_configuration_snapshot(expected_scan_plane_configuration)
        _v2._require(type(prepared_scan) is _PreparedScanPlaneBinding
            and prepared_scan._token is _SCAN_PLANE_BINDING_TOKEN
            and prepared_scan.registration is registration and prepared_scan.evidence is registration["evidence"]
            and prepared_scan.initialization is registration["evidence"]["initialization"]
            and prepared_scan.encoded_scan is prepared_scan.initialization["patch_evidence"]["scan"]
            and prepared_scan.scan is scan and prepared_scan.topology is topology
            and type(prepared_scan.position_error_bound_mm) is type(position_error_bound_mm)
            and prepared_scan.position_error_bound_mm == position_error_bound_mm
            and prepared_scan.configuration == canonical, "INVALID_PREPARED_SCAN_PLANE_BINDING")
        for value in (prepared_scan.evidence, prepared_scan.initialization):
            _, actual = _scan_plane_configuration_snapshot(value.get("scan_plane_configuration"))
            _v2._require(actual == canonical, "STALE_PREPARED_SCAN_PLANE_CONFIGURATION")
        if prepared_scan.evidence["connected_measurement_contexts"]:
            _v2._require(prepared_scan.evidence["surface_role_preparation"]["metadata_sha256"]
                == prepared_scan.metadata_sha256, "STALE_PREPARED_SURFACE_ROLE_METADATA_BINDING")
    binding = _registration_binding_with_completion_origins(
        registration, scan, topology, cad, selection, position_error_bound_mm, context_index=context_index)
    if schedule_binding is not None:
        binding.update(schedule_binding)
    if prepared_scan is not None:
        if registration["evidence"]["connected_measurement_contexts"]:
            role = registration["evidence"]["surface_role_preparation"]
            _v2._require(all(role.get(key) == binding[value] for key, value in (
                ("points_sha256", "raw_scan_points_sha256"),
                ("triangle_raw_rows_sha256", "raw_triangle_rows_sha256"),
                ("position_error_bounds_mm_sha256", "position_error_bounds_mm_sha256")))
                and type(role.get("raw_row_count")) is int and role["raw_row_count"] == len(scan["raw_points"]),
                "STALE_PREPARED_SURFACE_ROLE_ARRAY_BINDING")
        binding.update(scan_plane_mode="composite_v1", scan_plane_metadata_sha256=prepared_scan.metadata_sha256)
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    _v2._require(registration["schema_version"] == "noisy_plane_initialized_finite_registration_v12"
        and evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12"
        and initialization["schema_version"] == "noisy_plane_initialization_prefilter_v12"
        and initialization["numerical_consumer_complete"] is True
        and initialization["box_screening_complete"] is True
        and initialization["box_screening_policy"] == "REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1"
        and evidence["numerical_consumer_complete"] is True
        and initialization["numerical_support_policy"] == _stages.PROXIMITY
        and evidence["numerical_support_policy"] == "DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION"
        and evidence["stage_policies"] == _stages.ablation_stage_policies_v1(
            connected=evidence["connected_measurement_contexts"])
        and evidence["distance_model"] == "CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL"
        and evidence["optimizer_objective"] ==
            "MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS",
        "MISSING_BOUND_AWARE_REGISTRATION_CONSUMER_CONTRACT")
    selected = evidence["selected_candidate_index"] if context_index is None else context_index
    if context_index is not None:
        _v2._require(evidence["connected_measurement_contexts"] is True
            and evidence["accepted_registration"] is False
            and registration["transform_scan_to_cad"] is None
            and context_index in evidence["measurement_context_indices"], "INVALID_MEASUREMENT_CONTEXT_PUBLICATION")
    _v2._require(type(selected) is int and 0 <= selected < len(evidence["candidates"]),
                 "INVALID_BOUND_AWARE_SELECTED_CANDIDATE")
    candidate = evidence["candidates"][selected]
    _v2._require(type(candidate) is dict and type(candidate["index"]) is int
        and candidate["index"] == selected and candidate["execution_status"] == "ATTEMPTED"
        and (candidate["reasons"] == [] if context_index is None else
             candidate["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT")
        and candidate["numerical_consumer_status"] == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
        and all(candidate[name] == evidence[name] for name in
            ("numerical_support_policy", "distance_model", "optimizer_objective")),
        "UNQUALIFIED_BOUND_AWARE_SELECTED_CANDIDATE")

    joint = initialization["joint_proposal_accounting"]
    schedule = initialization["two_plane_schedule"]
    hypotheses = initialization["hypotheses"]
    _v2._require(type(joint) is dict and type(joint.get("pair_lane_enabled")) is bool
        and all(type(joint.get(key)) is int and 0 <= joint[key] <= maximum for key, maximum in (
            ("triplet_records", 8192), ("triplet_unsigned_reservation", 1024),
            ("ordered_pair_matches", 256), ("pair_slot_reservation", 8192),
            ("total_hypothesis_records", 8192)))
        and type(hypotheses) is list and joint["total_hypothesis_records"] == len(hypotheses)
        and joint["pair_slot_reservation"] == 32*joint["ordered_pair_matches"]
        and joint["triplet_records"]+joint["pair_slot_reservation"] == len(hypotheses),
        "INCOMPLETE_JOINT_PROPOSAL_PUBLICATION")
    _v2._require(type(schedule) is dict and schedule.get("selected_policy_evaluation_complete") is True
        and schedule.get("status") == ("COMPLETE" if joint["pair_lane_enabled"] else "DISABLED_BY_EXPLICIT_POLICY"),
        "INCOMPLETE_TWO_PLANE_SCHEDULE_PUBLICATION")
    if joint["pair_lane_enabled"] and joint["ordered_pair_matches"]:
        helper = initialization["two_plane_context"]
        initialize = _initialize_v15
        initialize._validate_two_plane_context_configuration(helper,
            expected_two_plane_context_configuration)
        _v2._require(helper["status"] == "COMPLETE" and helper["selected_policy_evaluation_complete"] is True
            and helper["accepted_registration"] is False
            and helper["finite_support_match_verified"] is False
            and len(schedule["scheduled_matches"]) == joint["ordered_pair_matches"]
            and helper["statistics"]["slot_records_completed"] == joint["pair_slot_reservation"],
            "INCOMPLETE_TWO_PLANE_CONTEXT_PUBLICATION")
        expected_context = {key+"_sha256": initialization["input_digests"][name+"_sha256"]
            for key, name in (("scan_points", "scan_points"),
                ("scan_triangles", "scan_triangle_raw_rows"), ("cad_triangles", "cad_triangles"),
                ("cad_face_ids", "cad_face_ids"), ("cad_source_triangle_ids", "cad_source_triangle_ids"))}
        _v2._require(helper["input_digests"] == expected_context,
                     "PUBLISHED_TWO_PLANE_CONTEXT_BUFFER_MISMATCH")
    selected_groups = initialization["start_groups"]
    all_groups = initialization["all_passing_start_groups"]
    _v2._require(type(selected_groups) is list and len(selected_groups) == len(evidence["candidates"])
        and type(all_groups) is list and selected < len(selected_groups),
        "INCOMPLETE_SELECTED_START_GROUP_LEDGER")
    group = selected_groups[selected]
    gi = candidate["exact_group_index"]
    _v2._require(type(gi) is int and 0 <= gi < len(all_groups)
        and group["index"] == selected and group["exact_group_index"] == gi
        and all_groups[gi]["exact_group_index"] == gi
        and group["refinement_disposition"] == all_groups[gi]["refinement_disposition"]
            == "SCHEDULED_FOR_V3_REFINEMENT"
        and candidate["hypothesis_indices"] == group["hypothesis_indices"] == all_groups[gi]["hypothesis_indices"]
        and all(np.array_equal(candidate[key], group[key]) and np.array_equal(group[key], all_groups[gi][key])
                for key in ("initialized_rotation", "initialized_translation_mm")),
        "SELECTED_CANDIDATE_NOT_COMPLETE_SCHEDULED_GROUP")
    origins = candidate["hypothesis_indices"]
    completion_origins = candidate.get("completion_indices", [])
    _v2._require(completion_origins == group.get("completion_indices", []) == all_groups[gi].get("completion_indices", []),
                 "SELECTED_CANDIDATE_COMPLETION_ORIGINS_MISMATCH")
    _v2._require(type(origins) is list and len(set(origins)) == len(origins)
        and (origins or (all_groups[gi].get("lane") == _initialize_v15.COMPLETION_LANE and completion_origins)),
        "MISSING_SELECTED_START_SUPPORT_ORIGINS")
    # Whenever a completion lane is published, the selected group's completion origins must equal
    # the PASSED children that claim this group, even when that list is empty: a mixed group may
    # not silently drop a passed child, and no group may claim completion origins without a lane.
    lane = initialization.get("completion_lane")
    if type(lane) is dict and type(lane.get("hypotheses")) is list:
        _v2._require(completion_origins == [row.get("completion_index") for row in lane["hypotheses"]
            if type(row) is dict and row.get("exact_start_group_index") == gi and row.get("prefilter_status") == "PASSED"],
            "SELECTED_GROUP_COMPLETION_ORIGIN_LEDGER_INCOMPLETE")
    else:
        _v2._require(completion_origins == [], "COMPLETION_ORIGINS_WITHOUT_PUBLISHED_LANE")
    if completion_origins:
        _validate_completion_origins(initialization, candidate, gi, completion_origins)
    for index in origins:
        _v2._require(type(index) is int and 0 <= index < len(hypotheses),
                     "INVALID_SELECTED_START_SUPPORT_ORIGIN")
        origin = hypotheses[index]
        count = {"TRIPLET": 3, "TWO_PLANE_CONTEXT": 2}.get(origin.get("proposal_lane"))
        _v2._require(origin["hypothesis_index"] == index and origin["prefilter_status"] == "PASSED"
            and count is not None and len(origin["assigned_support"]) == count
            and len(origin["correspondences"]) == count
            and all(all(type(a.get(key)) is int and type(b.get(key)) is int and a[key] == b[key]
                    for key in ("scan_plane_id", "cad_plane_id", "normal_sign"))
                for a, b in zip(origin["assigned_support"], origin["correspondences"]))
            and all(row["status"] == "PASSED" for row in origin["assigned_support"])
            and np.array_equal(origin["rotation"], candidate["initialized_rotation"])
            and np.array_equal(origin["translation_mm"], candidate["initialized_translation_mm"]),
            "SELECTED_START_NOT_BOUND_TO_PASSED_LANE_SUPPORT")
    _v2._require(origins == [index for index, row in enumerate(hypotheses)
        if row.get("exact_start_group_index") == gi and row.get("prefilter_status") == "PASSED"],
        "SELECTED_GROUP_ORIGIN_LEDGER_INCOMPLETE")

    # Bind publication to the pose whose diagnostics actually passed, not just
    # any well-formed rigid transform accompanying the same input digests.
    selected_rotation = np.asarray(candidate["rotation"])
    selected_translation = np.asarray(candidate["translation"])
    _v2._require(selected_rotation.shape == (3, 3) and selected_rotation.dtype.kind in "fiu"
        and selected_translation.shape == (3,) and selected_translation.dtype.kind in "fiu",
        "INVALID_SELECTED_CANDIDATE_POSE")
    selected_transform = np.eye(4)
    selected_transform[:3, :3] = selected_rotation
    selected_transform[:3, 3] = selected_translation
    sr, st = _v2._rigid_transform(selected_transform)
    if context_index is None:
        pr, pt = _v2._rigid_transform(registration["transform_scan_to_cad"])
        _v2._require(np.array_equal(sr, pr) and np.array_equal(st, pt),
                     "PUBLISHED_TRANSFORM_NOT_SELECTED_CANDIDATE_POSE")
    return dict(binding, **context_binding, registration_implementation=REGISTRATION_IMPLEMENTATION,
        numerical_support_policy=evidence["numerical_support_policy"],
        numerical_consumer_status=candidate["numerical_consumer_status"],
        distance_model=evidence["distance_model"], optimizer_objective=evidence["optimizer_objective"])


def _bounded_workflow(output, maximum=MAX_WORKFLOW_REPORT_BYTES):
    size = 0
    try:
        _v2._require(type(maximum) is int and 0 <= maximum <= MAX_WORKFLOW_REPORT_BYTES,
                     "INVALID_WORKFLOW_REPORT_BUDGET")
        for piece in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(output):
            size += len(piece.encode("utf-8"))
            _v2._require(size <= maximum, "WORKFLOW_REPORT_BUDGET_EXCEEDED")
    except (ValueError, TypeError, OverflowError, MemoryError):
        # No successful prefix or malformed later measurement survives. The
        # small explicit refusal envelope is outside the detailed-report cap.
        return dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", implementation=WORKFLOW_IMPLEMENTATION,
            status="UNAVAILABLE",
            phase="REPORT_PUBLICATION", reasons=["WORKFLOW_DETAILED_REPORT_UNAVAILABLE"],
            predictions=[], measurements=None, registration=None, observed_bend_count=None,
            production_use_allowed=False, conformance="INDETERMINATE",
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size)
    return output


def inspect_cad_guided_mesh_v17(*, scan_bytes, cad_bytes, position_error_bound_mm, workflow_policy,
        registration_policy, measurement_policy, instance_policy,
        noisy_policy=None, noisy_budgets=None, cad_patch_policy=None, cad_patch_budgets=None,
        search_policy=None, search_budgets=None, initialization_budgets=None,
        surface_budgets=None, registration_budgets=None, allow_ocp=False, two_plane_enabled=True,
        connected_measurement_contexts=True, scan_plane_mode="v7", seeded_plane_budgets=None,
        composite_plane_budgets=None, two_plane_schedule_mode="v1", two_plane_schedule_budgets=None,
        two_plane_context_mode="v4", two_plane_context_budgets=None,
        completion_policy=None, completion_budgets=None, layer_policy=None,
        refinement_allowance=None, clock=None, stage_checkpoint=None):
    """Byte/native workflow with conditional contexts, never an accepted pose.

    Complete selected-context binding precedes fresh support-cover measurement.
    Position bounds guide initialization and context surface-role exclusions;
    they are not scanner calibration or propagated measurement uncertainty.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", implementation=WORKFLOW_IMPLEMENTATION,
        status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V12",
        initialization_implementation=_initialize_v15.INITIALIZATION_IMPLEMENTATION,
        registration_implementation=REGISTRATION_IMPLEMENTATION,
        kernel_implementation=KERNEL_IMPLEMENTATION, completion_implementation=COMPLETION_IMPLEMENTATION,
        context_request_scope=CONTEXT_REQUEST_SCOPE,
        wire_schema_scope="V13_TRANSPORT_WITH_EXPLICIT_V17_IMPLEMENTATION_NOT_HISTORICAL_EXECUTION",
        measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if connected_measurement_contexts is not True:
        return {**output, "reasons": ["V17_CONNECTED_CONTEXTS_REQUIRED"]}
    output["connected_measurement_contexts"] = connected_measurement_contexts
    if type(two_plane_enabled) is not bool:
        return {**output, "reasons": ["INVALID_TWO_PLANE_LANE_POLICY"]}
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV17:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    if type(position_error_bound_mm) not in (int, float) or not 0 <= position_error_bound_mm <= 1000.:
        return {**output, "reasons": ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]}
    position_error_bound_mm = float(position_error_bound_mm)
    output["position_error_model"] = dict(uniform_euclidean_bound_mm=position_error_bound_mm,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        scope="PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY")
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4
        initialize = _initialize_v15
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy

        _v2._require(type(measurement_policy) is NativeGuidedPolicy
                     and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        _v2._require(measurement_policy.fit_model == "free_circle",
                     "CONNECTED_CONTEXT_FREE_CIRCLE_REQUIRED")
        workflow_policy = MeshWorkflowPolicyV17(**asdict(workflow_policy))
        measurement_policy = NativeGuidedPolicy(**asdict(measurement_policy))
        instance_policy = SupportPolicy(**asdict(instance_policy))
        layer_policy = LayerSelectionPolicyV1() if layer_policy is None else layer_policy
        _v2._require(type(layer_policy) is LayerSelectionPolicyV1, "VALIDATED_LAYER_POLICY_REQUIRED")
        layer_policy = LayerSelectionPolicyV1(**asdict(layer_policy))
        schedule_request = {} if (type(two_plane_schedule_mode) is str and two_plane_schedule_mode == "v1"
            and two_plane_schedule_budgets is None) else dict(two_plane_schedule_mode=two_plane_schedule_mode,
                two_plane_schedule_budgets=two_plane_schedule_budgets)
        config = initialize._configuration(registration_policy,
            initialize._noisy.NoisyMeshPlanePolicyV5() if noisy_policy is None else noisy_policy,
            initialize._noisy.NoisyMeshPlaneBudgetsV5() if noisy_budgets is None else noisy_budgets,
            initialize._patch.CoplanarPatchPolicyV4() if cad_patch_policy is None else cad_patch_policy,
            initialize._patch.CoplanarPatchBudgetsV4() if cad_patch_budgets is None else cad_patch_budgets,
            initialize._search.PlaneSearchPolicyV6() if search_policy is None else search_policy,
            initialize._search.PlaneSearchBudgetsV6() if search_budgets is None else search_budgets,
            initialize.PlaneInitializationBudgetsV12() if initialization_budgets is None else initialization_budgets,
            surface_budgets, two_plane_enabled, scan_plane_mode, seeded_plane_budgets, composite_plane_budgets,
            **schedule_request, two_plane_context_mode=two_plane_context_mode,
            two_plane_context_budgets=two_plane_context_budgets,
            completion_policy=completion_policy, completion_budgets=completion_budgets)
        scan_plane_arguments = {} if config["scan_plane_mode"] == "v7" else dict(
            scan_plane_mode=config["scan_plane_mode"], seeded_plane_budgets=config["seeded_plane_budgets"],
            composite_plane_budgets=config["composite_plane_budgets"])
        schedule_arguments = {} if config.get("two_plane_schedule_mode", "v1") == "v1" else dict(
            two_plane_schedule_mode="v2", two_plane_schedule_budgets=config["two_plane_schedule_budgets"])
        context_arguments = dict(two_plane_context_mode="v4",
            two_plane_context_budgets=config["two_plane_context_budgets"])
        context_expected, _ = initialize._context_configuration_snapshot(config["two_plane_context_configuration"])
        registration_policy = config["policy"]
        registration_budgets = initialize._v3._validated_execution_budgets(
            initialize._v3.SurfaceRegistrationBudgetsV4() if registration_budgets is None else registration_budgets)
        registration_budgets = initialize._v3.SurfaceRegistrationBudgetsV4(**asdict(registration_budgets))
        output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
            measurement=asdict(measurement_policy), support=asdict(instance_policy),
            noisy=asdict(config["noisy_policy"]), noisy_budgets=asdict(config["noisy_budgets"]),
            cad_patch=asdict(config["cad_patch_policy"]), cad_patch_budgets=asdict(config["cad_patch_budgets"]),
            search=asdict(config["search_policy"]), search_budgets=asdict(config["search_budgets"]),
            initialization_budgets=asdict(config["budgets"]), surface_budgets=asdict(config["surface_budgets"]),
            registration_budgets=asdict(registration_budgets), two_plane_enabled=two_plane_enabled,
            stage_policies=_stages.ablation_stage_policies_v1(connected=connected_measurement_contexts),
            completion_policy=asdict(config["completion_policy"]), completion_budgets=asdict(config["completion_budgets"]),
            layer_policy=asdict(layer_policy), kernel_implementation=KERNEL_IMPLEMENTATION,
            completion_implementation=COMPLETION_IMPLEMENTATION)
        output["policies"].update(initialize._context_publication(config))
        output.update(initialize._context_publication(config))
        if scan_plane_arguments:
            output["policies"]["scan_plane_configuration"] = _v12._scan_plane_configuration_snapshot(
                config["scan_plane_configuration"])[0]
        if schedule_arguments:
            output["policies"]["two_plane_schedule_configuration"] = _v12._two_plane_schedule_configuration_snapshot(
                config["two_plane_schedule_configuration"])[0]
        scan = prepare_mesh_scan_v1(scan_bytes=scan_bytes,
                                    max_engine_rows=workflow_policy.max_registration_scan_rows)
        output["input_preparation"] = {key: scan[key] for key in (
            "status", "reasons", "source_bindings", "coverage", "measurement_cloud_contract",
            "sample_cloud_contract", "units_status", "coordinate_frame_status")}
        if scan["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["MESH_INPUT_UNAVAILABLE", *scan["reasons"]]})
        _v2._scan_binding(scan, scan_bytes, workflow_policy.max_registration_scan_rows)
        topology = parse_triangle_mesh_ply_v4(scan_bytes)
        output["scan_topology"] = {key: value for key, value in topology.items()
                                   if key not in ("points", "triangle_raw_rows")}
        if topology["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["TRIANGLE_SCAN_UNAVAILABLE", *topology["reasons"]]})
        output["scan_triangle_binding"] = _triangle_scan_binding(scan, topology, scan_bytes)
        selection = _registration_selection(scan)
    except Exception as error:
        return _bounded_workflow({**output, "reasons": ["INPUT_PREPARATION_EXCEPTION:"+type(error).__name__]})
    output["cad_sha256"] = hashlib.sha256(cad_bytes).hexdigest()
    try:
        output["phase"] = "NATIVE_IMPORT"
        shape, receipt = _v2._read_step_owned_snapshot(cad_bytes)
        _v2._require(receipt["target_unit"] == "mm"
                     and receipt["physical_calibration"] == "NOT_ESTABLISHED", "INVALID_NATIVE_UNIT_RECEIPT")
        output["cad_units"] = receipt
        output["phase"] = "NATIVE_FEATURES"
        if workflow_policy.native_adapter_version == "r0":
            from .cad_guided_native_features import extract_ocp_guided_features as extract
        else:
            from .cad_guided_native_features_v1 import extract_ocp_guided_features_v1 as extract
        features = extract(shape, policy=measurement_policy, unit_status="DECLARED_MM", allow_ocp=True)
        output["native_features"] = features["metadata"]
        output["phase"] = "NATIVE_FEATURE_BRIDGE"
        try:
            bridged_features, output["native_feature_bridge"] = _bridge.bridge_native_features_v1(features)
        except _bridge.BridgeRefusal as error:
            # Wrong class/schema, duplicate ids or an over-limit set refuse BEFORE any costly work,
            # with the factory's own diagnostics retained and no partial measurement prefix.
            output["native_feature_bridge"] = dict(implementation=_bridge.BRIDGE_IMPLEMENTATION,
                status="REFUSED", reason=str(error), native_status=features.get("status"))
            return _bounded_workflow({**output, "reasons": ["NATIVE_FEATURE_BRIDGE_REFUSED:"+str(error)]})
        output["phase"] = "CAD_SURFACE_SAMPLING"
        from .cad_guided_cad_surface_v2 import sample_native_cad_surface_v2
        cad = sample_native_cad_surface_v2(shape, allow_ocp=True, native_binding="ocp",
            sample_count=workflow_policy.cad_sample_count,
            linear_deflection_mm=workflow_policy.cad_linear_deflection_mm,
            angular_deflection_radians=workflow_policy.cad_angular_deflection_radians)
        array_keys = {"triangles", "source_face_indices", "source_triangle_indices", "points", "normals",
                      "sample_triangle_rows", "sample_source_face_indices",
                      "sample_source_triangle_indices", "sample_barycentric_weights"}
        output["cad_surface_sampling"] = {key: value for key, value in cad.items() if key not in array_keys}
        if cad["status"] != "AVAILABLE":
            return _bounded_workflow({**output, "reasons": ["CAD_SURFACE_SAMPLING_UNAVAILABLE", *cad["reasons"]]})
        output["triangle_sample_binding"] = _v2._surface_binding(cad, workflow_policy.cad_sample_count)
        # Measurement-context cap from the frozen consumer's fixed preflight
        # budgets (contexts x features <= windows; contexts x features x raw
        # rows <= crop rows) and the declared policy maximum. A feature count of
        # zero cannot measure anyway; the cap then defaults to one context.
        output["phase"] = "MEASUREMENT_CONTEXT_CAP"
        from .cad_guided_context_measurement_v3 import MAX_CONTEXT_WINDOWS, MAX_CROP_ROW_VISITS
        feature_count = max(1, len(features["features"]))
        raw_rows = len(scan["raw_points"])
        context_cap = max(1, min(workflow_policy.max_measurement_contexts,
            MAX_CONTEXT_WINDOWS//feature_count, MAX_CROP_ROW_VISITS//(feature_count*raw_rows)))
        output["measurement_context_cap"] = dict(value=context_cap,
            policy_maximum=workflow_policy.max_measurement_contexts, feature_count=feature_count,
            raw_rows=raw_rows, window_budget=MAX_CONTEXT_WINDOWS, crop_row_budget=MAX_CROP_ROW_VISITS,
            schedule=_initialize_v15.MEASUREMENT_CONTEXT_SCHEDULE)
        output["phase"] = "WHOLE_PART_REGISTRATION"
        registration, owned_surface_roles = initialize.register_measurement_contexts_v16(
            scan["raw_points"], topology["triangle_raw_rows"],
            cad["triangles"], cad["source_face_indices"], cad["source_triangle_indices"],
            position_error_bounds_mm=position_error_bound_mm,
            cad_initialization_points=cad["points"], registration_policy=registration_policy,
            registration_raw_rows=selection, unit_assumption=workflow_policy.unit_assumption,
            noisy_policy=config["noisy_policy"], noisy_budgets=config["noisy_budgets"],
            cad_patch_policy=config["cad_patch_policy"], cad_patch_budgets=config["cad_patch_budgets"],
            search_policy=config["search_policy"], search_budgets=config["search_budgets"],
            budgets=config["budgets"], surface_budgets=config["surface_budgets"],
            registration_budgets=registration_budgets, two_plane_enabled=two_plane_enabled,
            connected_measurement_contexts=connected_measurement_contexts, **scan_plane_arguments,
            **schedule_arguments, **context_arguments, max_measurement_contexts=context_cap,
            completion_policy=config["completion_policy"], completion_budgets=config["completion_budgets"],
            refinement_allowance=refinement_allowance, clock=clock, stage_checkpoint=stage_checkpoint)
        output["registration"] = registration
        if registration["status"] != "CONDITIONAL_MEASUREMENT_CONTEXTS":
            return _bounded_workflow({**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]})
        try:
            _v2._require(registration["evidence"]["initialization"]["joint_proposal_accounting"]["pair_lane_enabled"]
                is two_plane_enabled, "REGISTRATION_PAIR_LANE_POLICY_MISMATCH")
            indices = registration["evidence"]["measurement_context_indices"]
            _v2._require(type(indices) is list and indices and len(indices) <= 48
                and len(set(indices)) == len(indices), "INVALID_CONTEXT_INDEX_LEDGER")
            # Mandatory even for a triplet-origin context or an empty pair schedule.
            initialize._bind_context_publication(output, context_expected)
            initialize._bind_context_publication(output["policies"], context_expected)
            initialize._bind_context_publication(registration["evidence"], context_expected)
            initialize._bind_context_publication(registration["evidence"]["initialization"], context_expected)
            scan_binding_arguments = dict(expected_two_plane_context_configuration=context_expected)
            if schedule_arguments:
                published_budget = registration["evidence"]["initialization"]["budgets"]
                requested_budget = asdict(config["budgets"])
                _v2._require(type(published_budget) is dict and set(published_budget) == set(requested_budget)
                    and all(type(published_budget[key]) is int and published_budget[key] == value
                        for key, value in requested_budget.items()),
                    "V2_SCHEDULE_INITIALIZATION_BUDGET_REQUEST_MISMATCH")
                scan_binding_arguments["expected_two_plane_schedule_configuration"] = (
                    output["policies"]["two_plane_schedule_configuration"])
            if scan_plane_arguments:
                expected = config["scan_plane_configuration"]
                scan_binding_arguments.update(expected_scan_plane_configuration=expected,
                    _prepared_scan_plane_binding=_v12._scan_plane_binding(registration, scan, topology,
                        position_error_bound_mm, expected))
            output["registration_array_binding"] = [
                _registration_binding(registration, scan, topology, cad, selection,
                    position_error_bound_mm, context_index=index, **scan_binding_arguments) for index in indices]
        except Exception:
            output["registration"] = dict(registration, status="UNAVAILABLE", transform_scan_to_cad=None,
                                            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
            attempted = registration["evidence"].get("measurement_context_indices", [])
            output["registration"]["evidence"] = dict(registration["evidence"],
                attempted_measurement_context_indices=attempted, measurement_context_indices=[],
                evidence_scope="ATTEMPTED_WORK_ONLY_INVALID_V15_CONTEXT_OR_SCHEDULE_BINDING")
            output["registration_array_binding"] = []
            owned_surface_roles = None
            raise
        output["phase"] = "NATIVE_FEATURES"
        if features["status"] not in ("AVAILABLE", "PARTIAL") or not features["features"]:
            return _bounded_workflow({**output, "reasons": ["NATIVE_FEATURES_UNAVAILABLE"]})
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        from .cad_guided_support_cover_measurement_v4 import (
            inspect_support_cover_context_windows_v4, validate_support_cover_publication_v4)
        measured = inspect_support_cover_context_windows_v4(scan["raw_points"], bridged_features,
            registration=registration, scan_sha256=scan["source_bindings"]["scan_sha256"],
            instance_policy=instance_policy, surface_roles=owned_surface_roles, layer_policy=layer_policy)
        output["support_cover_publication_validation"] = validate_support_cover_publication_v4(
            measured, raw_points=scan["raw_points"], scan_sha256=scan["source_bindings"]["scan_sha256"])
        output.update(status=measured["status"],
            phase="MEASUREMENT_REFUSAL" if measured.get("computation_status") != "COMPLETE" else "COMPLETE",
            measurements=measured, predictions=measured["predictions"], reasons=measured["reasons"],
            prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS")
    except Exception as error:
        output.update(status="UNAVAILABLE", measurements=None, predictions=[],
                      reasons=["WORKFLOW_EXCEPTION:"+type(error).__name__])
    return _bounded_workflow(output)
