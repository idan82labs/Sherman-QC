"""Unqualified V12 byte-to-conditional-scan-measurement challenger.

C1 is proposal-only; C2 does not require or assert an accepted global pose.
Frozen V11/R0 paths remain unchanged. Controlled explicit opt-in only.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from . import cad_guided_mesh_offline_v2 as _v2
from . import cad_guided_mesh_offline_v8 as _previous
from . import cad_guided_stage_policies_v1 as _stages


MAX_WORKFLOW_REPORT_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class MeshWorkflowPolicyV12(_v2.MeshWorkflowPolicyV2):
    """Exactly the unchanged V2 fields/admission, a distinct opt-in type."""


# Frozen pure byte/array custody helpers, no caller-supplied geometry certificates.
_triangle_scan_binding = _previous._triangle_scan_binding
_registration_selection = _previous._registration_selection


_SCAN_PLANE_BINDING_TOKEN = object()


@dataclass(frozen=True)
class _PreparedScanPlaneBinding:
    """Workflow-local reuse, not authentication against mutable Python code.

    The workflow owns the report and invokes no external callback between this
    preparation and its context-binding loop. Direct binding calls validate the
    complete evidence anew; the private token cannot be supplied as JSON input.
    """
    registration: object
    evidence: object
    initialization: object
    encoded_scan: object
    scan: object
    topology: object
    position_error_bound_mm: float
    configuration: str
    metadata_sha256: str
    _token: object


def _scan_plane_configuration_snapshot(value):
    from . import cad_guided_seeded_plane_proposals_v1 as seeded
    from . import cad_guided_scan_plane_preparation_v1 as composite
    _v2._require(type(value) is dict and set(value) == {"mode", "seeded_budgets", "composite_budgets"}
        and type(value["mode"]) is str and value["mode"] == "composite_v1",
        "INVALID_EXPECTED_SCAN_PLANE_CONFIGURATION")
    for key, maximum in (("seeded_budgets", asdict(seeded.SeededPlaneBudgetsV1())),
                         ("composite_budgets", asdict(composite.CompositeScanPlaneBudgetsV1()))):
        row = value[key]
        _v2._require(type(row) is dict and set(row) == set(maximum)
            and all(type(v) is int and 0 <= v <= maximum[k] for k, v in row.items()),
            "INVALID_SCAN_PLANE_CONFIGURATION_BUDGETS")
    copied = dict(mode="composite_v1", seeded_budgets=dict(value["seeded_budgets"]),
                  composite_budgets=dict(value["composite_budgets"]))
    return copied, json.dumps(copied, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _scan_plane_binding(registration, scan, topology, position_error_bound_mm, expected_configuration):
    """Once per composite workflow: strict mode, full codec and source binding."""
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    if expected_configuration is None:
        _v2._require("scan_plane_configuration" not in evidence
            and "scan_plane_configuration" not in initialization, "UNREQUESTED_COMPOSITE_SCAN_PLANE_CONFIGURATION")
        # Legacy test/report records may omit patch evidence. If it is present,
        # inspect its exact root schema; an unlabelled composite is not V7.
        # Full composite grammar/hashes are admitted only in the explicit arm.
        if "patch_evidence" in initialization:
            envelope = initialization["patch_evidence"]["scan"]
            _v2._require(type(envelope) is dict and set(envelope) == {"format", "schemas", "root"}
                and envelope["format"] == "noisy_mesh_metadata_schema_v6"
                and type(envelope["schemas"]) is list, "INVALID_LEGACY_SCAN_PLANE_ENVELOPE")
            root, schemas = envelope["root"], envelope["schemas"]
            _v2._require(type(root) is list and len(root) == 3 and root[0] == "D"
                and type(root[1]) is int and 0 <= root[1] < len(schemas)
                and type(root[2]) is list, "INVALID_LEGACY_SCAN_PLANE_ROOT")
            keys = schemas[root[1]]
            _v2._require(type(keys) is list and len(keys) <= 128 and len(keys) == len(root[2])
                and all(type(k) is str for k in keys) and len(set(keys)) == len(keys)
                and "schema_version" in keys and root[2][keys.index("schema_version")] == "noisy_mesh_planes_v7",
                "UNREQUESTED_OR_INVALID_SCAN_PLANE_SCHEMA")
        return None
    expected, canonical = _scan_plane_configuration_snapshot(expected_configuration)
    _v2._require(type(position_error_bound_mm) in (int, float)
        and 0 <= position_error_bound_mm <= 1000., "INVALID_COMPOSITE_POSITION_ERROR_BINDING")
    for value in (evidence, initialization):
        _, actual = _scan_plane_configuration_snapshot(value.get("scan_plane_configuration"))
        _v2._require(actual == canonical, "REGISTRATION_SCAN_PLANE_CONFIGURATION_MISMATCH")
    from . import cad_guided_context_surface_roles_v1 as roles
    envelope = initialization["patch_evidence"]["scan"]
    cap = roles.ContextSurfaceRoleBudgetsV1()
    meta, digest, encoded_bytes, expanded_bytes = roles._encoded_snapshot(envelope, cap)
    binding = dict(points_sha256=_v2._digest(scan["raw_points"]),
        triangle_raw_rows_sha256=_v2._digest(topology["triangle_raw_rows"]),
        position_error_bounds_mm_sha256=_v2._digest(np.full(len(scan["raw_points"]),
            position_error_bound_mm, dtype="<f8")))
    _v2._require(type(meta) is dict and meta.get("schema_version") == "composite_scan_planes_v1"
        and meta.get("status") == "COMPLETE" and meta.get("input_digests") == binding
        and meta.get("unit") == "mm" and meta.get("coordinate_frame") == "RAW_SCAN_UNREGISTERED"
        and meta.get("raw_coordinate_welding") is False and meta.get("residual_trimming") is False
        and meta.get("accepted_registration") is False and meta.get("production_use_allowed") is False
        and meta.get("search_exhaustive") is False and meta.get("physical_plane_count", False) is None
        and meta.get("observed_bend_count", False) is None, "INVALID_COMPOSITE_SCAN_PLANE_PUBLICATION")
    _, actual = _scan_plane_configuration_snapshot(dict(mode="composite_v1",
        seeded_budgets=meta["requested_lane_budgets"]["seeded_v1"], composite_budgets=meta["budgets"]))
    _v2._require(actual == canonical and encoded_bytes <= expected["composite_budgets"]["max_metadata_bytes"],
        "COMPOSITE_SCAN_PLANE_REQUEST_BINDING_MISMATCH")
    stats = meta["statistics"]
    _v2._require(type(stats) is dict and stats.get("selected_policy_evaluation_complete") is True
        and all(type(stats.get(key)) is int and stats[key] == value for key, value in (
            ("input_raw_rows", len(scan["raw_points"])), ("input_triangles", len(topology["triangle_raw_rows"])),
            ("metadata_bytes", encoded_bytes), ("metadata_expanded_bytes", expanded_bytes))),
        "INCOMPLETE_COMPOSITE_SCAN_PLANE_PUBLICATION")
    lanes = meta.get("lanes")
    _v2._require(type(lanes) is dict and tuple(lanes) == ("v7", "seeded_v1")
        and meta.get("lane_execution") == dict(v7="COMPLETE", seeded_v1="COMPLETE"),
        "INCOMPLETE_COMPOSITE_SCAN_PLANE_LANES")
    for lane, schema in (("v7", "noisy_mesh_planes_v7"), ("seeded_v1", "seeded_plane_proposals_v1")):
        row = lanes[lane]
        _v2._require(type(row) is dict and row.get("schema_version") == schema
            and row.get("status") == "COMPLETE" and row.get("input_digests") == binding
            and row.get("unit") == "mm" and row.get("coordinate_frame") == "RAW_SCAN_UNREGISTERED"
            and row.get("raw_coordinate_welding") is False and row.get("residual_trimming") is False,
            "UNAVAILABLE_OR_STALE_COMPOSITE_SCAN_PLANE_LANE")
        complete = ("core_construction_complete", "core_evaluation_complete", "scheduled_closure_evaluation_complete") \
            if lane == "v7" else ("selected_policy_evaluation_complete",)
        _v2._require(type(row.get("statistics")) is dict and all(row["statistics"].get(k) is True for k in complete),
            "INCOMPLETE_COMPOSITE_LANE_EVALUATION")
    if evidence["connected_measurement_contexts"]:
        role = evidence["surface_role_preparation"]
        _v2._require(type(role) is dict and all(role.get(k) == v for k, v in binding.items())
            and role.get("metadata_sha256") == digest
            and type(role.get("raw_row_count")) is int and role["raw_row_count"] == len(scan["raw_points"]),
            "COMPOSITE_SURFACE_ROLE_METADATA_BINDING_MISMATCH")
    return _PreparedScanPlaneBinding(registration, evidence, initialization, envelope, scan, topology,
        position_error_bound_mm, canonical, digest, _SCAN_PLANE_BINDING_TOKEN)


def _two_plane_schedule_configuration_snapshot(value):
    """Validate an independently requested V2 configuration, never infer one."""
    from . import cad_guided_two_plane_schedule_v2 as schedule_v2
    _v2._require(type(value) is dict and len(value) == 2 and set(value) == {"mode", "budgets"}
        and type(value["mode"]) is str and value["mode"] == "v2"
        and type(value["budgets"]) is dict and len(value["budgets"]) == len(schedule_v2._MAXIMA)
        and set(value["budgets"]) == set(schedule_v2._MAXIMA),
        "INVALID_EXPECTED_TWO_PLANE_SCHEDULE_CONFIGURATION")
    cap = schedule_v2._configuration(schedule_v2.TwoPlaneScheduleBudgetsV2(**value["budgets"]))
    copied = dict(mode="v2", budgets=dict(cap))
    return copied, json.dumps(copied, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _has_two_plane_schedule_v2_publication(evidence):
    """Recognize even malformed/unrequested V2 markers for failure clearing."""
    if type(evidence) is not dict:
        return False
    if "two_plane_schedule_configuration" in evidence:
        return True
    initialization = evidence.get("initialization")
    if type(initialization) is not dict:
        return False
    if "two_plane_schedule_configuration" in initialization:
        return True
    schedule = initialization.get("two_plane_schedule")
    context = initialization.get("two_plane_context")
    return (type(schedule) is dict and (schedule.get("schema_version") == "two_plane_schedule_v2"
        or any(key in schedule for key in ("breadth_ledger", "residual_windows", "residual_statistics")))) \
        or (type(context) is dict and (context.get("schema_version") == "two_plane_context_v2"
            or "scheduling_capacity" in context))


def _two_plane_schedule_binding(registration, expected_configuration):
    """Bind configuration and equations before any connected context is exposed."""
    evidence, initialization = registration["evidence"], registration["evidence"]["initialization"]
    schedule = initialization.get("two_plane_schedule")
    if expected_configuration is None:
        _v2._require(not _has_two_plane_schedule_v2_publication(evidence),
            "UNREQUESTED_V2_TWO_PLANE_SCHEDULE_PUBLICATION")
        return None
    expected, canonical = _two_plane_schedule_configuration_snapshot(expected_configuration)
    for publication in (evidence, initialization):
        _, actual = _two_plane_schedule_configuration_snapshot(publication.get("two_plane_schedule_configuration"))
        _v2._require(actual == canonical, "REGISTRATION_TWO_PLANE_SCHEDULE_CONFIGURATION_MISMATCH")
    joint = initialization["joint_proposal_accounting"]
    _v2._require(type(joint) is dict and joint.get("pair_lane_enabled") is True,
        "V2_SCHEDULE_REQUIRES_ENABLED_PUBLISHED_PAIR_LANE")
    from . import cad_guided_plane_initialization_v12 as initialize
    hypothesis = initialization["hypothesis_evidence"]
    bank, all_scan = hypothesis["scan_bank"], hypothesis["scan_planes"]
    _v2._require(type(bank) is list and len(bank) <= 128 and len(set(bank)) == len(bank)
        and type(all_scan) is list and len(all_scan) <= 65552
        and all(type(identifier) is int and 0 <= identifier < len(all_scan) for identifier in bank),
        "INVALID_PUBLISHED_SCHEDULE_SOURCE_BANK")
    # V12 creates scan proposal IDs as the original contiguous proposal index;
    # the search publishes all proposals sorted by those IDs. Direct bank lookups
    # avoid rescanning a possibly large proposal table for every context.
    fields = set(initialize._pair_schedule._FIELDS)
    scan_rows = []
    for identifier in bank:
        record = all_scan[identifier]
        _v2._require(type(record) is dict and type(record.get("equation")) is dict,
            "INVALID_PUBLISHED_SCHEDULE_SOURCE_EQUATION")
        equation = record["equation"]
        _v2._require(set(equation) == fields and type(equation["plane_id"]) is int
            and equation["plane_id"] == identifier, "STALE_PUBLISHED_SCHEDULE_SOURCE_IDENTITY")
        typed = initialize._search.PlaneEquationV6(identifier, tuple(equation["normal"]),
            tuple(equation["anchor_mm"]), equation["offset_from_anchor_mm"], equation["normal_angle_bound_radians"])
        initialize._search._equation(typed)  # Validate only; do not normalize canonical coefficients twice.
        _v2._require(all(type(value) is float for value in (*equation["normal"], *equation["anchor_mm"],
            equation["offset_from_anchor_mm"], equation["normal_angle_bound_radians"])),
            "NONCANONICAL_PUBLISHED_SCHEDULE_SOURCE_EQUATION")
        scan_rows.append({key: equation[key] for key in initialize._pair_schedule._FIELDS})
    cad_patches = initialization["patch_evidence"]["cad"]["patches"]
    _v2._require(type(cad_patches) is list and len(cad_patches) <= 2048,
        "INVALID_PUBLISHED_SCHEDULE_CAD_TABLE")
    cad_rows = [initialize._search._equation(initialize._search.PlaneEquationV6(
        row["plane_id"], tuple(row["normal"]), tuple(row["area_centroid_mm"]), 0., 0.)) for row in cad_patches]
    digests = {role+"_bank_equations_sha256": hashlib.sha256(json.dumps(rows,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for role, rows in (("scan", scan_rows), ("cad", cad_rows))}
    budget = initialization["budgets"]
    _v2._require(type(budget) is dict and all(type(budget.get(key)) is int and 0 <= budget[key] <= maximum
        for key, maximum in (("max_hypothesis_records", 8192), ("max_prefilter_unsigned_assignments", 1024)))
        and all(type(joint.get(key)) is int and 0 <= joint[key] <= maximum
        for key, maximum in (("triplet_records", 8192), ("triplet_unsigned_reservation", 1024))),
        "INVALID_PUBLISHED_V2_JOINT_BUDGET_BINDING")
    initialize._validate_two_plane_schedule_v2(schedule, expected, digests,
        tuple(bank), tuple(row["plane_id"] for row in cad_rows),
        joint["triplet_records"], joint["triplet_unsigned_reservation"],
        budget["max_hypothesis_records"], budget["max_prefilter_unsigned_assignments"])
    _v2._require(len(schedule["scheduled_matches"]) == joint["ordered_pair_matches"],
        "V2_SCHEDULE_JOINT_PUBLICATION_COUNT_MISMATCH")
    return dict(two_plane_schedule_mode="v2",
        two_plane_schedule_configuration_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        two_plane_schedule_input_digests=digests)


def _registration_binding(registration, scan, topology, cad, selection, position_error_bound_mm, *, context_index=None,
        expected_scan_plane_configuration=None, _prepared_scan_plane_binding=None,
        expected_two_plane_schedule_configuration=None):
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
    binding = _previous._registration_binding(
        registration, scan, topology, cad, selection, position_error_bound_mm)
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
        from . import cad_guided_plane_initialization_v12 as initialize
        initialize._validate_two_plane_context_configuration(helper,
            "v1" if expected_two_plane_schedule_configuration is None else "v2")
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
    _v2._require(type(origins) is list and origins and len(set(origins)) == len(origins),
                 "MISSING_SELECTED_START_SUPPORT_ORIGINS")
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
    return dict(binding, numerical_support_policy=evidence["numerical_support_policy"],
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
        return dict(schema="CAD_GUIDED_MESH_OFFLINE_V12", status="UNAVAILABLE",
            phase="REPORT_PUBLICATION", reasons=["WORKFLOW_DETAILED_REPORT_UNAVAILABLE"],
            predictions=[], measurements=None, registration=None, observed_bend_count=None,
            production_use_allowed=False, conformance="INDETERMINATE",
            detailed_report_discarded_whole=True, detailed_report_limit_bytes=maximum,
            encoded_bytes_examined=size)
    return output


def inspect_cad_guided_mesh_v12(*, scan_bytes, cad_bytes, position_error_bound_mm, workflow_policy,
        registration_policy, measurement_policy, instance_policy,
        noisy_policy=None, noisy_budgets=None, cad_patch_policy=None, cad_patch_budgets=None,
        search_policy=None, search_budgets=None, initialization_budgets=None,
        surface_budgets=None, registration_budgets=None, allow_ocp=False, two_plane_enabled=True,
        connected_measurement_contexts=True, scan_plane_mode="v7", seeded_plane_budgets=None,
        composite_plane_budgets=None, two_plane_schedule_mode="v1", two_plane_schedule_budgets=None):
    """Same byte/native/measurement workflow, declared-noise full-raw initialization.

    A successful global fit remains provisional and may absorb deformation.
    Selected-candidate completeness is enforced at publication; it is not a
    global-search or production verdict. Noise bounds are explicit declarations,
    not a scanner error estimate or propagated measurement uncertainty.
    """
    output = dict(schema="CAD_GUIDED_MESH_OFFLINE_V12", status="UNAVAILABLE",
        phase="ADMISSION", reasons=[], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V12",
        measurement_geometry_frame="CAD_ALIGNED_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE")
    if type(connected_measurement_contexts) is not bool:
        return {**output, "reasons": ["INVALID_CONNECTED_CONTEXT_POLICY"]}
    output["connected_measurement_contexts"] = connected_measurement_contexts
    if connected_measurement_contexts:
        output.update(registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE",
            measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM")
    if type(two_plane_enabled) is not bool:
        return {**output, "reasons": ["INVALID_TWO_PLANE_LANE_POLICY"]}
    if type(allow_ocp) is not bool or not allow_ocp:
        return {**output, "reasons": ["EXPLICIT_OCP_OPT_IN_REQUIRED"]}
    if type(workflow_policy) is not MeshWorkflowPolicyV12:
        return {**output, "reasons": ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]}
    if type(cad_bytes) is not bytes or not 1 <= len(cad_bytes) <= 8 * 1024**2:
        return {**output, "reasons": ["CAD_BYTE_BUDGET"]}
    if type(scan_bytes) is not bytes or not 1 <= len(scan_bytes) <= 64 * 1024**2:
        return {**output, "reasons": ["SCAN_BYTE_BUDGET"]}
    if type(position_error_bound_mm) not in (int, float) or not 0 <= position_error_bound_mm <= 1000.:
        return {**output, "reasons": ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]}
    position_error_bound_mm = float(position_error_bound_mm)
    output["position_error_model"] = dict(uniform_euclidean_bound_mm=position_error_bound_mm,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION", scope="PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY")
    output["phase"] = "INPUT_PREPARATION"
    try:
        from .cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
        from .cad_guided_mesh_topology_v4 import parse_triangle_mesh_ply_v4
        from . import cad_guided_plane_initialization_v12 as initialize
        from .cad_guided_native_features import NativeGuidedPolicy
        from .cad_guided_instances import SupportPolicy

        _v2._require(type(measurement_policy) is NativeGuidedPolicy
                     and type(instance_policy) is SupportPolicy, "VALIDATED_INFERENCE_POLICIES_REQUIRED")
        _v2._require(not connected_measurement_contexts or measurement_policy.fit_model == "free_circle",
                     "CONNECTED_CONTEXT_FREE_CIRCLE_REQUIRED")
        workflow_policy = MeshWorkflowPolicyV12(**asdict(workflow_policy))
        measurement_policy = NativeGuidedPolicy(**asdict(measurement_policy))
        instance_policy = SupportPolicy(**asdict(instance_policy))
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
            **schedule_request)
        scan_plane_arguments = {} if config["scan_plane_mode"] == "v7" else dict(
            scan_plane_mode=config["scan_plane_mode"], seeded_plane_budgets=config["seeded_plane_budgets"],
            composite_plane_budgets=config["composite_plane_budgets"])
        schedule_arguments = {} if config.get("two_plane_schedule_mode", "v1") == "v1" else dict(
            two_plane_schedule_mode="v2", two_plane_schedule_budgets=config["two_plane_schedule_budgets"])
        registration_policy = config["policy"]
        registration_budgets = initialize._v3._validated_execution_budgets(
            initialize._v3.SurfaceRegistrationBudgetsV3() if registration_budgets is None else registration_budgets)
        registration_budgets = initialize._v3.SurfaceRegistrationBudgetsV3(**asdict(registration_budgets))
        output["policies"] = dict(workflow=asdict(workflow_policy), registration=asdict(registration_policy),
            measurement=asdict(measurement_policy), support=asdict(instance_policy),
            noisy=asdict(config["noisy_policy"]), noisy_budgets=asdict(config["noisy_budgets"]),
            cad_patch=asdict(config["cad_patch_policy"]), cad_patch_budgets=asdict(config["cad_patch_budgets"]),
            search=asdict(config["search_policy"]), search_budgets=asdict(config["search_budgets"]),
            initialization_budgets=asdict(config["budgets"]), surface_budgets=asdict(config["surface_budgets"]),
            registration_budgets=asdict(registration_budgets), two_plane_enabled=two_plane_enabled,
            stage_policies=_stages.ablation_stage_policies_v1(connected=connected_measurement_contexts))
        if scan_plane_arguments:
            output["policies"]["scan_plane_configuration"] = _scan_plane_configuration_snapshot(
                config["scan_plane_configuration"])[0]
        if schedule_arguments:
            output["policies"]["two_plane_schedule_configuration"] = _two_plane_schedule_configuration_snapshot(
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
        output["phase"] = "WHOLE_PART_REGISTRATION"
        register = (initialize.register_measurement_contexts_v12 if connected_measurement_contexts
                    else initialize.register_whole_part_planes_v12)
        registration = register(scan["raw_points"], topology["triangle_raw_rows"],
            cad["triangles"], cad["source_face_indices"], cad["source_triangle_indices"],
            position_error_bounds_mm=position_error_bound_mm,
            cad_initialization_points=cad["points"], registration_policy=registration_policy,
            registration_raw_rows=selection, unit_assumption=workflow_policy.unit_assumption,
            noisy_policy=config["noisy_policy"], noisy_budgets=config["noisy_budgets"],
            cad_patch_policy=config["cad_patch_policy"], cad_patch_budgets=config["cad_patch_budgets"],
            search_policy=config["search_policy"], search_budgets=config["search_budgets"],
            budgets=config["budgets"], surface_budgets=config["surface_budgets"],
            registration_budgets=registration_budgets, two_plane_enabled=two_plane_enabled,
            connected_measurement_contexts=connected_measurement_contexts, **scan_plane_arguments, **schedule_arguments)
        owned_surface_roles = None
        if connected_measurement_contexts:
            registration, owned_surface_roles = registration
        output["registration"] = registration
        expected_status = "CONDITIONAL_MEASUREMENT_CONTEXTS" if connected_measurement_contexts else "CONDITIONAL_BEST_FIT"
        if registration["status"] != expected_status:
            return _bounded_workflow({**output, "reasons": ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]]})
        try:
            _v2._require(registration["evidence"]["initialization"]["joint_proposal_accounting"]["pair_lane_enabled"]
                is two_plane_enabled, "REGISTRATION_PAIR_LANE_POLICY_MISMATCH")
            scan_binding_arguments = {}
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
                    _prepared_scan_plane_binding=_scan_plane_binding(registration, scan, topology,
                        position_error_bound_mm, expected))
            if connected_measurement_contexts:
                indices = registration["evidence"]["measurement_context_indices"]
                _v2._require(type(indices) is list and indices and len(indices) <= 48
                    and len(set(indices)) == len(indices), "INVALID_CONTEXT_INDEX_LEDGER")
                output["registration_array_binding"] = [
                    _registration_binding(registration, scan, topology, cad, selection,
                        position_error_bound_mm, context_index=index, **scan_binding_arguments) for index in indices]
            else:
                output["registration_array_binding"] = _registration_binding(registration, scan, topology, cad, selection,
                    position_error_bound_mm, **scan_binding_arguments)
                rotation, translation = _v2._rigid_transform(registration["transform_scan_to_cad"])
        except Exception:
            output["registration"] = dict(registration, status="UNAVAILABLE", transform_scan_to_cad=None,
                                            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
            if schedule_arguments or _has_two_plane_schedule_v2_publication(registration["evidence"]):
                attempted = registration["evidence"].get("measurement_context_indices", [])
                output["registration"]["evidence"] = dict(registration["evidence"],
                    attempted_measurement_context_indices=attempted, measurement_context_indices=[],
                    evidence_scope="ATTEMPTED_WORK_ONLY_INVALID_V2_SCHEDULE_BINDING")
            raise
        output["phase"] = "NATIVE_FEATURES"
        if features["status"] not in ("AVAILABLE", "PARTIAL") or not features["features"]:
            return _bounded_workflow({**output, "reasons": ["NATIVE_FEATURES_UNAVAILABLE"]})
        output["phase"] = "FULL_RAW_LOCAL_MEASUREMENT"
        if connected_measurement_contexts:
            from .cad_guided_context_measurement_v1 import (
                inspect_generated_context_windows_v1, validate_context_measurement_publication_v1)
            measured = inspect_generated_context_windows_v1(scan["raw_points"], features["features"],
                registration=registration, scan_sha256=scan["source_bindings"]["scan_sha256"],
                instance_policy=instance_policy, surface_roles=owned_surface_roles)
            validate_context_measurement_publication_v1(measured, raw_count=len(scan["raw_points"]),
                scan_sha256=scan["source_bindings"]["scan_sha256"], raw_points_sha256=_v2._digest(scan["raw_points"]))
            output.update(status=measured["status"],
                phase="MEASUREMENT_REFUSAL" if measured.get("discarded_measurement_prefix") else "COMPLETE",
                measurements=measured,
                predictions=measured["predictions"], reasons=measured["reasons"],
                prediction_collection_status="CONDITIONAL_SUPPORT_RECORDS_NOT_JOINT_BEND_INSTANCES")
            return _bounded_workflow(output)
        aligned = scan["raw_points"]@rotation.T+translation
        _v2._require(np.isfinite(aligned).all(), "NONFINITE_ALIGNED_RAW_POINTS")
        from .cad_guided_local_measurement_v1 import inspect_full_raw_windows_v1
        measured = inspect_full_raw_windows_v1(aligned, features["features"], instance_policy=instance_policy,
            scan_sha256=scan["source_bindings"]["scan_sha256"], registration_status="CONDITIONAL_BEST_FIT")
        published, predictions = _v2._publish_measurements(measured, len(aligned), rotation,
                                                           translation, output["prediction_frame"])
        output.update(status="PARTIAL" if predictions else "UNAVAILABLE", phase="COMPLETE",
            measurements=published, predictions=predictions, observed_support_instance_count=len(predictions),
            reasons=[] if predictions else ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"])
    except Exception as error:
        output.update(status="UNAVAILABLE", measurements=None, predictions=[],
                      reasons=["WORKFLOW_EXCEPTION:"+type(error).__name__])
    return _bounded_workflow(output)
