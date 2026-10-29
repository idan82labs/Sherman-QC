"""Unqualified V4 same-input fresh V10-versus-V11 workflow candidate.

Both arms retain native V1, full-raw measurement, the fixed measured carrier,
V3 bound-aware refinement, full-face box screening and the unchanged scorer.
V11 adds a bounded two-plane/context proposal lane beside unchanged triplets.
The qualified V4 terminal helper replaces retained publication and terminal fill;
new complete campaign-tree custody remains independently unqualified integration.
The uniform .05 mm proposal bound is an exploratory uncalibrated stipulation,
not a scanner bound, manufacturing tolerance or a choice made from Gold.
Freeze is source-only. Execution requires an independent audit of that exact
bundle. Every case/arm is attempted once; no resume/retry CLI. Gold is opened
only after all 22 inference attempts are durable and terminal.
This is consumed development, not historical replay or production qualification.
The separately released cached UNION is the primary nonzero retained control;
a fresh V10 comparison cannot establish that V11 beats that control. Joining the
separate geometry-only ledgers requires the exact common71 reference snapshot
and unchanged matching; never disguise cached lines as modern measured carriers.
Killed-before-result internal stages are UNKNOWN, not inferred never reached.
Completion remains pending observed parent exit and independent retained audit.
"""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
from importlib import metadata as import_metadata
import json
import math
import os
from pathlib import Path
import resource
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUN = ROOT / "docs/research/gold11_mesh_development_v4_attempt"
PROTOCOL = "docs/research/2026-09-10_gold11_mesh_development_v4_protocol.md"
REVIEW = "scripts/review_gold11_inputs_v1.py"
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
CATALOG_SHA = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_SHA = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
SOURCES = (
    "scripts/run_gold11_mesh_development_v4.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/gold11_owned_process_supervision_v1.py",
    "scripts/gold11_terminal_artifacts_v4.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py",
    "backend/cad_guided_mesh_offline_v1.py", "backend/cad_guided_mesh_input_v1.py",
    "backend/cad_guided_global_registration_v1.py", "backend/cad_guided_cad_sampling_v1.py",
    "backend/cad_guided_native_features_v1.py", "backend/cad_guided_native_features.py",
    "backend/cad_guided_local_measurement_v1.py", "backend/cad_guided_measurement.py",
    "backend/cad_guided_instances.py",
    "backend/cad_guided_mesh_offline_v2.py", "backend/cad_guided_mesh_offline_v8.py",
    "backend/cad_guided_cad_surface_v2.py", "backend/cad_guided_global_registration_v2.py",
    "backend/cad_guided_triangle_surface_v2.py", "backend/cad_guided_mesh_topology_v4.py",
    "backend/cad_guided_plane_initialization_v4.py", "backend/cad_guided_plane_initialization_v8.py",
    "backend/cad_guided_mesh_plane_proposals_v4.py", "backend/cad_guided_plane_hypotheses_v4.py",
    "backend/cad_guided_noisy_mesh_planes_v5.py", "backend/cad_guided_noisy_mesh_planes_v7.py",
    "backend/cad_guided_plane_search_v6.py", "backend/cad_guided_report_envelope_v6.py",
    "backend/cad_guided_mesh_offline_v10.py", "backend/cad_guided_plane_initialization_v10.py",
    "backend/cad_guided_global_registration_v3.py", "backend/cad_guided_triangle_surface_v3.py",
    "backend/cad_guided_triangle_distance_bounds_v3.py", "backend/cad_guided_face_box_screen_v1.py",
    "backend/cad_guided_mesh_offline_v11.py", "backend/cad_guided_plane_initialization_v11.py",
    "backend/cad_guided_two_plane_context_v1.py", "backend/cad_guided_two_plane_schedule_v1.py",
    "scripts/run_gold11_mesh_development_v3.py",
    # Prior runners are source-only preservation oracles, never imported/run.
    "scripts/run_gold11_mesh_development_v2.py", "tests/test_gold11_mesh_development_runner_v4.py",
    REVIEW, PROTOCOL,
)
ARMS = ["control", "challenger"]
ARM_DEFINITIONS = dict(
    control=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V10", native_adapter_version="v1",
        registration="V10_FULL_FACE_BOX_REJECTION_AND_V3_BOUND_AWARE_FINITE_SURFACE_CONSUMERS",
        inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"),
    challenger=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V11", native_adapter_version="v1",
        registration="V11_TRIPLET_AND_TWO_PLANE_CONTEXT_SHARED_V3_FINITE_SUPPORT_REFINEMENT",
        inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"),
)
COMPARATOR = "FRESH_V10_VS_V11_FIXED_NATIVE_V1_NOT_CACHED_UNION_COMPARISON"
POSITION_ERROR_DECLARATION = dict(
    status="EXPLORATORY_STIPULATED_UNCALIBRATED_PROPOSAL_BOUND",
    scope="BOTH_ARMS_PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY",
    uniform_across_all_cases=True, physical_scanner_bound_established=False,
    selected_from_gold=False, constructed_point005_bound_is_not_physical_evidence=True,
)
LIMITS = dict(case_wall_seconds=180, case_cpu_seconds=150,
              sampled_worker_tree_rss_bytes=8*1024**3,
              case_log_bytes=1024**2, case_result_bytes=128*1024**2,
              total_wall_seconds=4200, max_scan_bytes=64*1024**2,
              max_cad_bytes=8*1024**2, scorer_wall_seconds=60,
              scorer_cpu_seconds=50, scorer_rss_bytes=2*1024**3,
              scorer_request_bytes=192*1024**2, scorer_result_bytes=32*1024**2,
              compact_case_bytes=8*1024**2, coordinator_rss_bytes=4*1024**3,
              global_tree_rss_bytes=12*1024**3, parent_read_seconds=20,
              coordinator_log_bytes=2*1024**2, terminalization_reserve_seconds=30)
# Explicit V3 numerical certificate accounting only; CONFIG and all its existing
# work/geometry quotas remain byte-for-byte the predecessor source expression.
CERTIFICATE_CAPS = dict(initialization_budgets=1_024_000_000,
    surface_budgets=128_000_000, registration_budgets=8_000_000_000)
RETAINED_CONTROL_POLICY = dict(primary_nonzero_control="SEPARATELY_FROZEN_CACHED_UNION",
    cached_control_value="NOT_ASSUMED_OR_LOADED_BY_THIS_RUNNER",
    comparison="SEPARATE_PROVISIONAL_RAW_ENDPOINT_GEOMETRY_LEDGER_ONLY",
    requires="INDEPENDENT_CONTROL_RELEASE_COMMON71_REFERENCE_SHA_UNCHANGED_MATCHER",
    cached_payload_as_modern_measured_carrier=False, historical_replay_claim_allowed=False,
    fresh_v10_gain_is_not_cached_union_superiority=True)
CONFIG = dict(
    workflow=dict(max_registration_scan_rows=50000, cad_sample_count=32768,
                  cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
                  unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED"),
    registration=dict(coarse_points=2048, refine_points=8192, coarse_iterations=30,
                      refine_iterations=30, trim_fraction=.85, huber_delta_mm=.25,
                      coarse_distance_fraction=.15, max_correspondence_distance_mm=2.,
                      max_inlier_plane_p95_mm=.5, min_scan_coverage=.75,
                      min_cad_coverage=.25, min_observability_ratio=.005,
                      pose_class_distance_mm=.1, ambiguity_plane_rms_margin_mm=.05,
                      ambiguity_coverage_margin=.03, max_rotation_step_radians=.15,
                      max_translation_step_mm=5.),
    measurement=dict(flange_window_mm=[2.5, 12.5], search_margin_mm=3.,
                     max_fit_residual_mm=.05, plane_exclusion_mm=.075,
                     max_tangency_error_mm=.1, fit_model="free_circle"),
    support=dict(max_axial_gap_mm=4., max_angular_gap_deg=5., min_arc_sweep_deg=20.,
                 min_local_points=3),
    # JSON-stable lists are restored to the exact tuple required by the policy
    # only inside the guarded worker; freeze never imports backend classes.
    position_error_bound_mm=.05,
    noisy_policy=dict(tile_triangles=[128, 1024], halo_rings=1, min_support_rows=3,
        max_normal_angle_radians=math.pi/18., min_inplane_covariance_ratio=1e-8),
    noisy_budgets=dict(max_raw_rows=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_core_records=65_536, max_core_triangle_rows=4096,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_full_support_row_tests=64_000_000, max_cone_row_tests=16_000_000,
        max_closure_triangle_tests=32_000_000, max_closure_edge_tests=96_000_000,
        max_fit_evaluations=65_552, max_closure_schedules=16,
        max_retained_array_bytes=536_870_912, max_metadata_bytes=67_108_864,
        chunk_rows=65_536),
    cad_patch_policy=dict(max_pair_normal_sine=1e-8, max_pair_anchored_distance_mm=1e-8,
        max_full_support_distance_mm=1e-8, min_support_vertices=12,
        min_second_extent_mm=.25, min_inplane_covariance_ratio=1e-8),
    cad_patch_budgets=dict(max_vertices=1_000_000, max_triangles=2_000_000,
        max_edge_records=6_000_000, max_components=65_536, max_output_patches=4096,
        max_support_memberships=6_000_000, max_full_vertex_tests=12_000_000,
        chunk_triangles=65_536),
    search_policy=dict(min_scan_support_count=3, max_scan_support_rms_mm=.1,
        min_normal_condition_ratio=.02, max_normal_alignment_extra_radians=.02,
        max_intersection_backward_residual_mm=1e-8, pair_angle_numeric_guard_radians=1e-10),
    search_budgets=dict(max_scan_planes=65_552, max_cad_planes=2048, max_bank_planes=128,
        max_scan_triplets=256, max_scan_partner_tests=1_048_576, max_cad_pair_records=2_096_128,
        max_pair_join_tests=32_000_000, max_cad_pair_samples_per_triplet=12,
        max_unsigned_per_scan_triplet=4, max_compatible_rows_per_triplet=49_104,
        max_unsigned_assignments=1024, max_signed_assignments=8192,
        max_report_bytes=67_108_864, chunk_pair_rows=65_536),
    initialization_budgets=dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
        max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
        max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
        chunk_rows=65_536, max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
        max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
        max_face_build_nodes=131_071, max_face_build_triangle_visits=10_000_000,
        max_refined_start_groups=48, max_report_bytes=100_663_296),
    surface_budgets=dict(max_triangles=500_000, max_query_rows=200_000, leaf_size=16,
        max_build_nodes=131_071, max_build_triangle_visits=10_000_000,
        max_node_visits=32_000_000, max_triangle_pairs=32_000_000,
        max_normal_pairs=32_000_000, max_work_units=512_000_000,
        max_pair_batch=65_536, max_ties_per_query=32),
    registration_budgets=dict(max_query_calls=4096, max_query_rows=64_000_000,
        max_node_visits=2_000_000_000, max_triangle_pairs=2_000_000_000,
        max_normal_pairs=500_000_000, max_work_units=20_000_000_000),
)

# New lane defaults are separately frozen, not modifications to scientific
# CONFIG or permission to override the public V11 default-only helper seam.
TWO_PLANE_DECLARATION = dict(enabled_arms=["challenger"],
    quota="TRIPLET_FIRST_RESIDUAL_JOINT8192_RECORDS_1024_UNSIGNED_NO_REFUND",
    pair_lane_starvation_possible=True, global_search_exhaustive=False,
    schedule_budgets=dict(max_scan_planes=128, max_cad_planes=2048,
        max_scan_pair_records=8128, max_selected_scan_pairs=64,
        max_scan_selection_tests=520192, max_cad_pair_records=2096128,
        max_cad_sort_work=44018688, max_window_pair_tests=134152192,
        max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
        max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536),
    context_budgets=dict(max_scan_rows=1000000, max_scan_triangles=2000000,
        max_cad_triangles=500000, max_scan_planes=128, max_cad_planes=2048,
        max_scan_pairs=64, max_ordered_cad_matches_per_scan_pair=4,
        max_signed_families=1024, max_candidate_slots=8192,
        max_scan_row_visits=64000000, max_cad_vertex_visits=128000000,
        max_index_visits=64000000, max_index_sort_work=256000000,
        max_hash_bytes=1073741824, max_projection_cache_entries=6144,
        max_working_bytes=268435456, max_report_bytes=33554432, chunk_rows=65536))
ARTIFACT_SOURCE = "scripts/gold11_terminal_artifacts_v4.py"
ARTIFACT_SOURCE_SHA = "3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838"
_VERIFIED_ARTIFACT_SHA = None
_ARTIFACT_MODULE = None
_ARTIFACT_DIRECTORY_WALK = False
ARTIFACT_POLICY = dict(helper_sha256=ARTIFACT_SOURCE_SHA,
    scientific_manifest_entries=128, custody_envelope_bytes=256*1024,
    complete_tree="ALL_SCHEMA_BRANCHES_ALL_NAMES_NO_UNKNOWN_CONTENT_READS",
    extra_entry_policy="VETO_AND_CHARGE_EACH_UNKNOWN_NAME_TO_SCIENTIFIC128",
    derived_metadata_bound="SCHEMA_PATH_SLOTS_PLUS128_EXTRA_NAMES_PLUS_ONE_OVERFLOW_WITNESS",
    publication="QUALIFIED_HELPER_EXCEPT_PRIVATE192MIB_SCORING_REQUEST",
    final_phase="PRE_COMPLETED_THEN_COMPLETED_THEN_FINAL_GATE_EXACT_RESCAN",
    concurrency="QUIESCENT_WRITERS_CALLER_ABSOLUTE_DEADLINE_NOT_ATOMIC_ADVERSARIAL_SNAPSHOT")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def decode(blob):
    def reject(_):
        raise ValueError("NONFINITE_JSON")
    return json.loads(blob, object_pairs_hook=unique_object, parse_constant=reject)


@contextmanager
def deadline(seconds):
    """Bound parent-side regular-file preparation as well as native workers."""
    def expired(_signum, _frame):
        raise TimeoutError("PARENT_IO_WALL_LIMIT")
    started = time.monotonic()
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    effective = min(seconds, previous_timer[0]) if previous_timer[0] > 0 else seconds
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, effective)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] or previous_timer[1]:
            remaining = max(1e-6, previous_timer[0]-(time.monotonic()-started)) if previous_timer[0] else 0
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def unavailable(reason, phase="SUPERVISOR"):
    return dict(status="UNAVAILABLE", phase=phase, reasons=[reason], predictions=[],
                observed_bend_count=None, production_use_allowed=False)


SCORING_FIELDS = ("status", "phase", "predictions", "observed_bend_count",
                  "production_use_allowed", "conformance", "uncertainty_status",
                  "prediction_frame", "units_status", "reference_frame_compatibility",
                  "reasons", "case_id", "family_id")


def compact_result(payload):
    # Preserve optional identity conflicts and every prediction record: the
    # scorer rejects the whole malformed case, never a filtered valid prefix.
    if type(payload) is not dict or type(payload.get("status")) is not str:
        return unavailable("MALFORMED_WORKFLOW_RESULT")
    result = {key: payload[key] for key in SCORING_FIELDS if key in payload}
    if len(encoded(result)) > LIMITS["compact_case_bytes"]:
        return unavailable("SCORING_PROJECTION_BYTE_BUDGET")
    return result


def workflow_stage_diagnostics(payload, arm):
    """Bounded descriptive status projection, never a scoring/geometry repair.

    A missing stage is UNKNOWN, not proof it never ran. Preserve returned phase,
    nested statuses and work counters separately; the summary is not a geometry
    gate or a substitute for the complete retained workflow. Never decode/re-fit
    compressed scan-plane geometry or classify a killed worker's internal stage.
    """
    if type(payload) is not dict:
        return dict(evidence="NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN", stages={},
                    reported_workflow_phase=None, arm=arm,
                    missing_stage_policy="UNKNOWN_NEVER_INFER_NOT_REACHED_OR_COMPLETED")
    phase = payload.get("phase")
    phase = phase if type(phase) is str and len(phase) <= 128 else None
    stages = {}
    for name in ("input_preparation", "scan_topology", "native_features",
                 "cad_surface_sampling", "registration", "measurements"):
        if name not in payload:
            stages[name] = dict(evidence="MISSING_UNKNOWN", status=None, reasons=[])
            continue
        row = payload[name]
        if type(row) is not dict:
            stages[name] = dict(evidence="NULL_UNKNOWN" if row is None else "MALFORMED_UNKNOWN",
                                status=None, reasons=[])
            continue
        status, reasons = row.get("status"), row.get("reasons", [])
        status = status if type(status) is str and len(status) <= 128 else None
        complete = (type(reasons) is list and len(reasons) <= 32
                    and all(type(reason) is str and len(reason) <= 256 for reason in reasons))
        stages[name] = dict(evidence="PRESENT", status=status,
            reasons=list(reasons) if complete else [], reasons_complete=complete,
            detailed_report_discarded_whole=row.get("detailed_report_discarded_whole") is True)
    def projection(row, names):
        if type(row) is not dict:
            return dict(evidence="MISSING_OR_NULL_UNKNOWN")
        answer, omitted = {}, []
        for name in names:
            if name not in row:
                continue
            value = row[name]
            if (value is None or type(value) is bool or type(value) is int and abs(value) <= 2**53
                    or type(value) is float and math.isfinite(value)
                    or type(value) is str and len(value) <= 256):
                answer[name] = value
            elif (type(value) is list and len(value) <= 32
                    and all(type(item) is str and len(item) <= 256 for item in value)):
                answer[name] = list(value)
            else:
                omitted.append(name)
        return dict(evidence="RETURNED_FIELDS_ONLY", values=answer, omitted_fields=omitted)
    registration = payload.get("registration")
    evidence = registration.get("evidence") if type(registration) is dict else None
    initialization = evidence.get("initialization") if type(evidence) is dict else None
    statistics = initialization.get("statistics") if type(initialization) is dict else None
    patches = initialization.get("patch_evidence") if type(initialization) is dict else None
    scan_patches = patches.get("scan") if type(patches) is dict else None
    cad_patches = patches.get("cad") if type(patches) is dict else None
    search = initialization.get("hypothesis_evidence") if type(initialization) is dict else None
    scan_format = scan_patches.get("format") if type(scan_patches) is dict else None
    scan_format = scan_format if type(scan_format) is str and len(scan_format) <= 256 else None
    stages["scan_plane_extraction"] = dict(
        evidence="RETURNED_ENCODED_LEDGER_NOT_DECODED_HERE" if type(scan_patches) is dict else "MISSING_OR_NULL_UNKNOWN",
        format=scan_format,
        ledger_path="registration.evidence.initialization.patch_evidence.scan")
    stages["cad_plane_extraction"] = projection(cad_patches, ("status", "reasons", "row_map_representation"))
    stages["plane_search"] = projection(search, ("status", "reasons", "selected_policy_evaluation_complete",
        "usable_hypotheses", "bank_policy", "search_exhaustive"))
    stages["plane_search_work"] = projection(search.get("statistics") if type(search) is dict else None, (
        "input_scan_planes", "input_cad_planes", "eligible_scan_planes", "bank_plane_count",
        "unscheduled_eligible_scan_planes", "scan_triplets_scheduled", "scan_triplets_evaluated",
        "signed_assignments_planned", "signed_assignments_evaluated", "proposed_hypotheses", "rejected_hypotheses"))
    stages["registration_funnel"] = projection(evidence, (
        "candidate_search_complete", "candidate_evaluation_complete", "numerical_consumer_complete",
        "candidate_count", "attempted_candidate_count", "candidate_numerical_failure_count",
        "selected_candidate_index", "scan_unique_rows", "coarse_scan_rows", "refined_scan_rows",
        "distance_model", "numerical_support_policy", "optimizer_objective"))
    stages["initialization_funnel"] = projection(initialization, (
        "status", "reasons", "search_complete", "selected_policy_evaluation_complete",
        "numerical_consumer_complete", "box_screening_complete", "box_screening_policy",
        "detailed_report_discarded_whole", "search_scope", "noise_bound_status"))
    stages["initialization_work"] = projection(statistics, (
        "proposed_hypothesis_count", "scheduled_hypothesis_count", "omitted_proposed_hypothesis_count",
        "search_rejected_hypothesis_count", "eligible_unsigned_assignment_count",
        "scheduled_unsigned_assignment_count", "omitted_unsigned_assignment_count",
        "exact_passing_start_group_count", "omitted_passing_start_group_count",
        "planned_box_screen_calls", "planned_box_screen_rows", "box_screen_calls", "box_screen_rows",
        "completed_box_screen_calls", "completed_box_screen_rows", "completed_query_calls",
        "completed_query_rows", "assigned_support_count", "completed_box_support_count",
        "bound_excluded_support_count", "bound_excluded_support_rows", "bound_excluded_query_calls",
        "completed_finite_query_support_count", "query_calls", "query_rows", "node_visits",
        "triangle_pairs", "certificate_pairs", "normal_pairs", "work_units"))
    summary = dict(evidence="DESCRIPTIVE_RETURNED_LEDGER_NOT_SCORING_OR_STAGE_RECLASSIFICATION",
        arm=arm, reported_workflow_phase=phase, stages=stages,
        missing_stage_policy="UNKNOWN_NEVER_INFER_NOT_REACHED_OR_COMPLETED",
        workflow_detailed_report_discarded_whole=payload.get("detailed_report_discarded_whole") is True,
        detailed_ledger_location="RESULT.json_RETURNED_WORKFLOW_WITHOUT_FILTERING",
        full_internal_ledger_recoverable_after_worker_kill=False)
    if len(encoded(summary)) > 32*1024:
        return dict(evidence="STAGE_SUMMARY_BYTE_BUDGET_FULL_RETURNED_RESULT_RETAINED",
                    reported_workflow_phase=phase, arm=arm, stages={})
    return summary


def _bounded_read(path, cap, expected=None, *, allow_empty=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not int(not allow_empty) <= before.st_size <= cap:
            raise ValueError("NONREGULAR_OR_BYTE_BUDGET")
        chunks, total = [], 0
        while True:
            block = os.read(fd, min(1024**2, cap+1-total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > cap:
                raise ValueError("BYTE_BUDGET")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError("SOURCE_CHANGED_DURING_READ")
        blob = b"".join(chunks)
        if expected is not None and digest(blob) != expected:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return blob
    finally:
        os.close(fd)


def bounded_read(path, cap, expected=None):
    with deadline(LIMITS["parent_read_seconds"]):
        return _bounded_read(path, cap, expected)


def owned_read(relative, cap, expected=None, *, allow_empty=False):
    """Read only a fixed owned receipt/artifact, with no symlink components."""
    if type(relative) is not str:
        raise ValueError("OWNED_RELATIVE_PATH_REQUIRED")
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or relative != "/".join(parts) or ".." in parts:
        raise ValueError("OWNED_RELATIVE_PATH_REQUIRED")
    directory = os.open(RUN, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not int(not allow_empty) <= before.st_size <= cap:
            raise ValueError("OWNED_NONREGULAR_OR_BYTE_BUDGET")
        blob = bytearray()
        while True:
            block = os.read(fd, min(1024**2, cap+1-len(blob)))
            if not block:
                break
            blob.extend(block)
            if len(blob) > cap:
                raise ValueError("OWNED_BYTE_BUDGET")
        after = os.fstat(fd)
        if any(getattr(before, key) != getattr(after, key)
               for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("OWNED_ARTIFACT_CHANGED_DURING_READ")
        blob = bytes(blob)
        if expected is not None and digest(blob) != expected:
            raise ValueError("OWNED_ARTIFACT_HASH_MISMATCH")
        return blob
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def marker_present(path):
    """A dangling link is still spent evidence; other lstat errors refuse."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def artifact_module():
    """Compile the exact qualified buffer, not a package/loader reopening it.

    Freeze uses the pinned live helper. Execution/authorized leaves use the
    same hash from their frozen source. No filesystem imports occur at startup.
    """
    global _ARTIFACT_MODULE
    expected = _VERIFIED_ARTIFACT_SHA or ARTIFACT_SOURCE_SHA
    if expected != ARTIFACT_SOURCE_SHA:
        raise ValueError("UNQUALIFIED_TERMINAL_ARTIFACT_HELPER")
    if _ARTIFACT_MODULE is None:
        base = RUN/"source" if _VERIFIED_ARTIFACT_SHA is not None else ROOT
        blob = _bounded_read(base/ARTIFACT_SOURCE, 2*1024**2, expected)
        module = ModuleType("_gold11_terminal_artifacts_v4")
        exec(compile(blob, str(base/ARTIFACT_SOURCE), "exec"), module.__dict__)
        module.__verified_source_sha256 = expected
        _ARTIFACT_MODULE = module
    if _ARTIFACT_MODULE.__verified_source_sha256 != expected:
        raise ValueError("LOADED_TERMINAL_ARTIFACT_SOURCE_CHANGED")
    return _ARTIFACT_MODULE


def publication_cap(path):
    path = Path(path)
    try:
        name = path.relative_to(RUN/"source").as_posix()
    except ValueError:
        name = None
    if name in SOURCES:
        return 2*1024**2
    if path.parent == RUN/"leaf_dispatch":
        return 16384
    special = {"scan.ply": LIMITS["max_scan_bytes"], "cad.step": LIMITS["max_cad_bytes"],
        "RESULT.json": LIMITS["case_result_bytes"], "RESULT_PARTIAL.bin": LIMITS["case_result_bytes"],
        "SCORING_INPUT.json": LIMITS["compact_case_bytes"],
        "SCORER_RESULT.json": LIMITS["scorer_result_bytes"],
        "SCORER_RESULT_PARTIAL.bin": LIMITS["scorer_result_bytes"],
        "SCORED.json": LIMITS["scorer_result_bytes"], "SELECTED_REFERENCES.json": 8*1024**2,
        "COORDINATOR_VERIFIED.json": 32768, "COORDINATOR_DISPATCH.json": 4096}
    if path.name == "SCORING_REQUEST.json":
        raise ValueError("PRIVATE_SCORER_REQUEST_REQUIRES_SEPARATE_WRITER")
    if path.name == "COMPLETED.json" and path.parent != RUN:
        return LIMITS["compact_case_bytes"]
    return special.get(path.name, 256*1024)


@contextmanager
def artifact_directory_walk():
    global _ARTIFACT_DIRECTORY_WALK
    previous = _ARTIFACT_DIRECTORY_WALK
    _ARTIFACT_DIRECTORY_WALK = True
    try:
        yield
    finally:
        _ARTIFACT_DIRECTORY_WALK = previous


class _ArtifactDeadlineExpired(BaseException):
    """Private alarm only: bypass helper Exception handlers until it unwinds."""


@contextmanager
def artifact_alarm_guard():
    """Preserve the active absolute timer; convert expiry outside helper code.

    The qualified helper reports ordinary Exception failures and may continue
    its bounded inventory after them. An ordinary TimeoutError alarm would be
    swallowed there. This private BaseException instead unwinds its descriptor
    finally blocks, then becomes TimeoutError here. No timer is reset or extended.
    Caller code must stop filesystem work when that TimeoutError is returned.
    """
    if signal.getitimer(signal.ITIMER_REAL)[0] <= 0:
        raise TimeoutError("ARTIFACT_ABSOLUTE_DEADLINE_NOT_ACTIVE")
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise _ArtifactDeadlineExpired()
    try:
        try:
            # Establish restoration/conversion before installing the handler:
            # the alarm can arrive immediately as signal.signal returns.
            signal.signal(signal.SIGALRM, expired)
            yield
        except _ArtifactDeadlineExpired:
            raise TimeoutError("ARTIFACT_ABSOLUTE_DEADLINE_EXPIRED") from None
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


def publish_bytes(path, blob):
    """Qualified no-write preexistence and exclusive publication, no retries."""
    path = Path(path)
    # Nested timers preserve any stricter active case/campaign deadline.
    with deadline(LIMITS["parent_read_seconds"]), artifact_directory_walk(), artifact_alarm_guard():
        return artifact_module().publish_bytes_exclusive_v4(
            path.parent, path.name, blob, max_bytes=publication_cap(path))


def publish_private_scoring_request(cache, blob):
    """Only the transient192MiB request uses this preserved separate writer.

    It is NOT qualified-helper publication. The fresh private cache is checked,
    the exact path is constructed here, and the caller's scorer deadline stays
    active. Never used for retained outputs, controls or terminal fill.
    """
    cache_identity(cache)
    if signal.getitimer(signal.ITIMER_REAL)[0] <= 0:
        raise ValueError("PRIVATE_SCORER_REQUEST_ABSOLUTE_DEADLINE_REQUIRED")
    if type(blob) is not bytes or len(blob) > LIMITS["scorer_request_bytes"]:
        raise ValueError("SCORER_REQUEST_BYTE_BUDGET")
    path = Path(cache)/"SCORING_REQUEST.json"
    if marker_present(path):
        raise FileExistsError("PRIVATE_SCORER_REQUEST_ALREADY_EXISTS_NO_WRITE")
    path = Path(path)
    temporary = path.with_name("."+path.name+".partial-"+secrets.token_hex(8))
    with temporary.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path, value):
    publish_bytes(path, encoded(value)+b"\n")


def approved_cases_from_review(blob):
    if digest(blob) != REVIEW_SHA:
        raise ValueError("REVIEW_SOURCE_BINDING_MISMATCH")
    assignments = [node for node in ast.parse(blob).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "CASES"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("CASE_CONSTANT_SCHEMA")
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
            and call.func.attr == "loads" and len(call.args) == 1 and not call.keywords):
        raise ValueError("CASE_CONSTANT_SCHEMA")
    cases = decode(ast.literal_eval(call.args[0]))
    if len(cases) != 11 or len({case["scan_sha256"] for case in cases}) != 11:
        raise ValueError("CASE_SCOPE_MISMATCH")
    if len({case["cad_sha256"] for case in cases}) != 6:
        raise ValueError("CAD_SCOPE_MISMATCH")
    return cases


def runtime_metadata():
    versions = {}
    for name in ("numpy", "scipy", "cadquery-ocp"):
        distributions = list(import_metadata.Distribution.discover(name=name, path=[runtime_roots()[0]]))
        if len(distributions) != 1:
            raise ValueError("RUNTIME_DISTRIBUTION_METADATA_AMBIGUOUS:" + name)
        versions[name] = distributions[0].version
    return dict(python_executable=str(Path(sys.executable).resolve()), python_version=sys.version,
                platform=sys.platform, machine=os.uname().machine,
                distributions=versions, attestation="VERSION_METADATA_NOT_BINARY_ATTESTATION")


def freeze():
    # No input/reference paths are touched by this operation.
    blobs = {name: bounded_read(ROOT/name, 2*1024**2) for name in SOURCES}
    cases = approved_cases_from_review(blobs[REVIEW])
    bundle = dict(schema="GOLD11_MESH_DEVELOPMENT_V4", config=CONFIG, limits=LIMITS,
                  cases=cases, source_sha256={name: digest(blob) for name, blob in blobs.items()},
                  catalog_sha256=CATALOG_SHA, selected_reference_sha256=SELECTED_SHA,
                  runtime_metadata=runtime_metadata(),
                  arms=ARMS, arm_definitions=ARM_DEFINITIONS, comparator=COMPARATOR,
                  position_error_declaration=POSITION_ERROR_DECLARATION,
                  certificate_caps=CERTIFICATE_CAPS, retained_control_policy=RETAINED_CONTROL_POLICY,
                  two_plane_declaration=TWO_PLANE_DECLARATION, artifact_policy=ARTIFACT_POLICY,
                  historical_comparability="UNESTABLISHED", production_use_allowed=False)
    RUN.mkdir(exist_ok=False)
    snapshot = RUN/"source"
    for name, blob in blobs.items():
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_bytes(destination, blob)
    exclusive_json(RUN/"FROZEN.json", bundle)
    print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", bundle_sha256=digest(encoded(bundle)),
                          attempt=str(RUN))), flush=True)


def verify_bundle():
    global _VERIFIED_SUPERVISION_SHA, _VERIFIED_ARTIFACT_SHA
    bundle = decode(bounded_read(RUN/"FROZEN.json", 256*1024))
    if (bundle.get("schema") != "GOLD11_MESH_DEVELOPMENT_V4"
            or bundle["config"] != CONFIG or bundle["limits"] != LIMITS or bundle["arms"] != ARMS
            or bundle.get("arm_definitions") != ARM_DEFINITIONS
            or bundle.get("comparator") != COMPARATOR
            or bundle.get("position_error_declaration") != POSITION_ERROR_DECLARATION
            or bundle.get("certificate_caps") != CERTIFICATE_CAPS
            or bundle.get("retained_control_policy") != RETAINED_CONTROL_POLICY
            or bundle.get("two_plane_declaration") != TWO_PLANE_DECLARATION
            or bundle.get("artifact_policy") != ARTIFACT_POLICY):
        raise ValueError("FROZEN_PROTOCOL_CONSTANTS_CHANGED")
    if bundle.get("runtime_metadata") != runtime_metadata():
        raise ValueError("FROZEN_RUNTIME_METADATA_CHANGED")
    if set(bundle["source_sha256"]) != set(SOURCES):
        raise ValueError("SOURCE_DENOMINATOR_CHANGED")
    for name, sha in bundle["source_sha256"].items():
        bounded_read(ROOT/name, 2*1024**2, sha)
        bounded_read(RUN/"source"/name, 2*1024**2, sha)
    cases = approved_cases_from_review(bounded_read(RUN/"source"/REVIEW, 2*1024**2))
    if (bundle["cases"] != cases or bundle["catalog_sha256"] != CATALOG_SHA
            or bundle["selected_reference_sha256"] != SELECTED_SHA):
        raise ValueError("FROZEN_INPUT_OR_REFERENCE_SCOPE_CHANGED")
    _VERIFIED_SUPERVISION_SHA = bundle["source_sha256"][SUPERVISION_SOURCE]
    if bundle["source_sha256"][ARTIFACT_SOURCE] != ARTIFACT_SOURCE_SHA:
        raise ValueError("FROZEN_TERMINAL_ARTIFACT_SOURCE_UNQUALIFIED")
    _VERIFIED_ARTIFACT_SHA = ARTIFACT_SOURCE_SHA
    return bundle


def contained(path, roots):
    return any(path == root or path.startswith(root+os.sep) for root in roots)


def runtime_roots():
    return (str((ROOT/".venv/lib/python3.11/site-packages").resolve()),
            str((Path(sys.base_prefix)/"lib/python3.11").resolve()))


def allowed_native_library(value, roots):
    # Importing ctypes itself initializes pythonapi with PyDLL(None).
    if value is None:
        return True
    if not isinstance(value, (str, bytes)):
        return False
    value = os.fsdecode(value)
    if not os.path.isabs(value):
        return False
    path = os.path.realpath(value)
    system = {"/usr/lib/libSystem.B.dylib", "/usr/lib/libc.dylib"}
    return path in system or (contained(path, roots) and path.endswith((".so", ".dylib")))


def worker_audit_hook(snapshot, cache, package="backend"):
    runtime = runtime_roots()
    source = str((snapshot/package).resolve())
    cache = str(cache.resolve())
    allowed_roots, writable = (source, cache, *runtime), (cache,)

    def descriptor_path(fd):
        if sys.platform != "darwin":
            raise PermissionError("WORKER_DESCRIPTOR_PLATFORM_UNSUPPORTED")
        # Darwin F_GETPATH=50, PATH_MAX=1024. Needed for Python's fd-safe
        # TemporaryDirectory/rmtree cleanup; never admit a descriptor blindly.
        value = fcntl.fcntl(fd, 50, bytes(1024)).split(b"\0", 1)[0]
        if not value or not os.path.isabs(value):
            raise PermissionError("WORKER_DESCRIPTOR_PATH_UNAVAILABLE")
        return os.path.realpath(os.fsdecode(value))

    def checked_path(value, roots, dir_fd=None):
        if isinstance(value, int):
            path = descriptor_path(value)
        else:
            value = os.fsdecode(value)
            if dir_fd not in (None, -1) and not os.path.isabs(value):
                value = os.path.join(descriptor_path(dir_fd), value)
            path = os.path.realpath(value)
        if not contained(path, roots):
            raise PermissionError("WORKER_PATH_OUTSIDE_FROZEN_SCOPE")
        return path

    def audit(event, args):
        if event.startswith("socket.") or event in (
                "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.exec",
                "os.posix_spawn", "os.posix_spawnp", "os.setsid", "os.setpgid"):
            raise PermissionError("WORKER_EXTERNAL_ACTION_DENIED")
        if event == "ctypes.dlopen":
            if not allowed_native_library(args[0], runtime):
                raise PermissionError("WORKER_NATIVE_LIBRARY_OUTSIDE_RUNTIME")
        if event == "open" and not isinstance(args[0], int):
            mode, flags = args[1], args[2]
            writes = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                      or isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            path = os.path.realpath(os.fsdecode(args[0]))
            if not writes and path in ("/dev/null", "/dev/urandom"):
                return
            # Qualified publication walks no-follow directory components from
            # '/'. Admit only read-only DIRECTORY opens of ancestors of the
            # already bound source/cache; never files or ancestor enumeration.
            # Relative dir_fd components are checked by their pinned helper;
            # realpath alone has no dir_fd context, so resolve them as for other
            # fd-safe operations using the descriptor audit below.
            if (not writes and type(flags) is int and flags & os.O_DIRECTORY
                    and flags & os.O_NOFOLLOW and os.path.isabs(os.fsdecode(args[0]))
                    and any(root == path or root.startswith(path.rstrip(os.sep)+os.sep)
                            for root in (source, cache))):
                return
            if (not writes and _ARTIFACT_DIRECTORY_WALK and type(flags) is int
                    and flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW
                    and type(args[0]) is str and args[0] not in ("", ".", "..")
                    and "/" not in args[0]
                    and args[0] in set(Path(source).parts+Path(cache).parts)):
                # open's audit event omits dir_fd. Only the trusted helper's
                # literal ancestor components get this directory-only exception.
                return
            checked_path(args[0], writable if writes else allowed_roots)
        if event in ("os.listdir", "os.scandir"):
            checked_path(args[0], allowed_roots)
        if event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"):
            position = {"os.remove": 1, "os.unlink": 1, "os.rmdir": 1,
                        "os.mkdir": 2, "os.chmod": 2, "os.utime": 3, "os.truncate": 99}[event]
            checked_path(args[0], writable, args[position] if len(args) > position else None)
        if event in ("os.rename", "os.link"):
            checked_path(args[0], writable, args[2] if len(args) > 2 else None)
            checked_path(args[1], writable, args[3] if len(args) > 3 else None)
        if event == "os.symlink":
            # No symlink creation is part of input/STEP/cache preparation.
            raise PermissionError("WORKER_SYMLINK_DENIED")
    return audit


def install_namespace(name, directory):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory.resolve())]
    sys.modules[name] = module


def initialize_worker(snapshot, cache, package, cpu_seconds):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds,)*2)
    sys.path.insert(0, runtime_roots()[0])
    install_namespace(package, snapshot/package)
    # This guards Python-level accidental I/O, not native OS syscalls or hostile
    # extension code. The frozen native import/reader source is reviewed too.
    sys.addaudithook(worker_audit_hook(snapshot, cache, package))


_COORDINATOR_CONTEXT = None
_VERIFIED_SUPERVISION_SHA = None
_SUPERVISION_MODULE = None
SUPERVISION_SOURCE = "scripts/gold11_owned_process_supervision_v1.py"
LEAF_SCHEMA = "ONE_USE_PARENT_BOUND_LEAF_V4"
RUNNER_SOURCE = "scripts/run_gold11_mesh_development_v4.py"


def supervision_module():
    """Compile the exact verified frozen bytes; never reopen via a loader."""
    global _SUPERVISION_MODULE
    if not is_hex(_VERIFIED_SUPERVISION_SHA, 64):
        raise ValueError("VERIFIED_SUPERVISION_SOURCE_REQUIRED")
    if _SUPERVISION_MODULE is None:
        blob = _bounded_read(RUN/"source"/SUPERVISION_SOURCE, 2*1024**2, _VERIFIED_SUPERVISION_SHA)
        module = ModuleType("_gold11_owned_process_supervision_v1")
        exec(compile(blob, str(RUN/"source"/SUPERVISION_SOURCE), "exec"), module.__dict__)
        module.__verified_source_sha256 = _VERIFIED_SUPERVISION_SHA
        _SUPERVISION_MODULE = module
    if _SUPERVISION_MODULE.__verified_source_sha256 != _VERIFIED_SUPERVISION_SHA:
        raise ValueError("LOADED_SUPERVISION_SOURCE_BINDING_CHANGED")
    return _SUPERVISION_MODULE


def is_hex(value, length):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def leaf_sources(role):
    if role == "INFERENCE":
        return (RUNNER_SOURCE, ARTIFACT_SOURCE,
                *(name for name in SOURCES if name.startswith("backend/")))
    if role == "SCORER":
        return (RUNNER_SOURCE, ARTIFACT_SOURCE, "scripts/gold11_mesh_scoring_v1.py",
                "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py")
    raise ValueError("UNKNOWN_LEAF_ROLE")


def activate_coordinator(bundle, audit, expected, start):
    """Publish only neutral proof after the full coordinator gate succeeds.

    Leaves never open FROZEN/review source: those include case identities. The
    actual parent is the coordinator that verified them, not a caller-selected
    source of authority. Exclusive context creation also makes dispatch one-use.
    This is a fail-closed workflow gate, not protection from a hostile local user
    rewriting controls or replacing the Python executable.
    """
    global _COORDINATOR_CONTEXT
    context = dict(schema=LEAF_SCHEMA, coordinator_pid=os.getpid(), outer_pid=os.getppid(),
        bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
        started_sha256=digest(encoded(start)), coordinator_dispatch_sha256=digest(encoded(expected)),
        source_sha256=bundle["source_sha256"], runtime_metadata=bundle["runtime_metadata"])
    exclusive_json(RUN/"COORDINATOR_VERIFIED.json", context)
    (RUN/"leaf_dispatch").mkdir(exist_ok=False)
    _COORDINATOR_CONTEXT = context


def cache_identity(cache):
    cache = Path(cache)
    info = cache.lstat()
    if (not cache.is_absolute() or cache != cache.resolve() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError("LEAF_CACHE_NOT_PRIVATE_OWNED_DIRECTORY")
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid)


def issue_leaf_dispatch(role, cache, **binding):
    context = _COORDINATOR_CONTEXT
    if (type(context) is not dict or context.get("coordinator_pid") != os.getpid()
            or context.get("outer_pid") != os.getppid() or os.getpgrp() != os.getpid()):
        raise ValueError("VERIFIED_COORDINATOR_REQUIRED_TO_ISSUE_LEAF")
    validate_leaf_binding(role, binding)
    # This neutral digest is unique per scan/CAD/arm, not a case or family ID.
    # Exclusive issuance prevents retries even when a worker never starts.
    dispatch_id = digest(encoded(dict(role=role, binding=binding)))
    token = secrets.token_hex(32)
    record = dict(schema=LEAF_SCHEMA, role=role, binding=binding, cache=str(cache),
        cache_identity=cache_identity(cache), coordinator_pid=os.getpid(),
        context_sha256=digest(encoded(context)), token_sha256=digest(token.encode()))
    exclusive_json(RUN/"leaf_dispatch"/(dispatch_id+".json"), record)
    return dict(dispatch_id=dispatch_id, token=token)


def validate_leaf_binding(role, binding):
    expected = {"arm", "scan_sha256", "cad_sha256"} if role == "INFERENCE" else {"request_sha256"}
    if role not in ("INFERENCE", "SCORER") or type(binding) is not dict or set(binding) != expected:
        raise ValueError("LEAF_BINDING_SCHEMA")
    if role == "INFERENCE" and binding["arm"] not in ARMS:
        raise ValueError("LEAF_ARM_SCHEMA")
    if any(not is_hex(value, 64) for key, value in binding.items() if key != "arm"):
        raise ValueError("LEAF_HASH_SCHEMA")


def authorize_leaf(request, role):
    global _VERIFIED_ARTIFACT_SHA
    # Reject the former arbitrary snapshot/cache/arm API before any file read.
    if (type(request) is not dict or set(request) != {"dispatch_id", "token"}
            or not is_hex(request["dispatch_id"], 64) or not is_hex(request["token"], 64)):
        raise ValueError("COORDINATOR_LEAF_DISPATCH_REQUIRED")
    paths = leaf_sources(role)
    record = decode(bounded_read(RUN/"leaf_dispatch"/(request["dispatch_id"]+".json"), 16384))
    context = decode(bounded_read(RUN/"COORDINATOR_VERIFIED.json", 32768))
    expected = decode(bounded_read(RUN/"COORDINATOR_DISPATCH.json", 4096))
    start = decode(bounded_read(RUN/"STARTED.json", 4096))
    if (record.get("schema") != LEAF_SCHEMA or context.get("schema") != LEAF_SCHEMA
            or record.get("role") != role or record.get("coordinator_pid") != os.getppid()
            or context.get("coordinator_pid") != os.getppid() or os.getpgrp() != os.getppid()
            or context.get("outer_pid") != expected.get("parent_pid")
            or record.get("context_sha256") != digest(encoded(context))
            or context.get("coordinator_dispatch_sha256") != digest(encoded(expected))
            or context.get("started_sha256") != digest(encoded(start))
            or start.get("status") != "STARTED_NO_RETRY" or start.get("limits") != LIMITS
            or context.get("bundle_sha256") != start.get("bundle_sha256")
            or context.get("audit_sha256") != start.get("audit_sha256")
            or not is_hex(start.get("bundle_sha256"), 64) or not is_hex(start.get("audit_sha256"), 64)
            or not secrets.compare_digest(record.get("token_sha256", ""), digest(request["token"].encode()))):
        raise ValueError("LEAF_DISPATCH_PARENT_OR_CONTROL_BINDING")
    binding = record.get("binding")
    validate_leaf_binding(role, binding)
    if request["dispatch_id"] != digest(encoded(dict(role=role, binding=binding))):
        raise ValueError("LEAF_DISPATCH_IDENTITY_BINDING")
    # Terminal runs cannot authorize a later process even if PIDs are reused.
    for name in ("COMPLETED.json", "COORDINATOR_COMPLETED.json", "SCORING_UNAVAILABLE.json",
                 "FINAL_GATE.json", "FINALIZATION_FAILED.json"):
        if marker_present(RUN/name):
            raise ValueError("LEAF_RUN_ALREADY_TERMINAL")
    if role == "INFERENCE" and marker_present(RUN/"INFERENCE_TERMINAL.json"):
        raise ValueError("INFERENCE_ALREADY_TERMINAL_NO_RETRY")
    if role == "SCORER":
        terminal = decode(bounded_read(RUN/"INFERENCE_TERMINAL.json", 32768))
        if any(type(terminal.get(key)) is not int or terminal[key] != count
               for key, count in (("case_count", 11), ("arm_count", 2), ("terminal_record_count", 22))):
            raise ValueError("SCORER_REQUIRES_ALL_22_TERMINALS")
    # One neutral bootstrap source is needed to publish the exclusive claim.
    # Verify its pinned qualified bytes after capability authentication but
    # before claim; no backend/runtime/scientific input is accessed here.
    sources = context.get("source_sha256")
    if (type(sources) is not dict or set(sources) != set(SOURCES)
            or sources.get(ARTIFACT_SOURCE) != ARTIFACT_SOURCE_SHA):
        raise ValueError("LEAF_TERMINAL_HELPER_BINDING")
    _VERIFIED_ARTIFACT_SHA = ARTIFACT_SOURCE_SHA
    # Burn before backend/runtime/input access. Any later failure is terminal.
    exclusive_json(RUN/"leaf_dispatch"/(request["dispatch_id"]+".CLAIMED.json"),
        dict(worker_pid=os.getpid(), coordinator_pid=os.getppid(), role=role,
             dispatch_sha256=digest(encoded(record)), no_retry=True))
    snapshot, cache = RUN/"source", Path(record["cache"])
    if (cache_identity(cache) != record.get("cache_identity") or Path.cwd() != cache
            or os.environ.get("TMPDIR") != str(cache)):
        raise ValueError("LEAF_CACHE_BINDING_CHANGED")
    if Path(__file__).resolve() != (snapshot/RUNNER_SOURCE).resolve():
        raise ValueError("LEAF_MUST_EXECUTE_FROZEN_RUNNER")
    sources = context.get("source_sha256")
    if type(sources) is not dict or set(sources) != set(SOURCES):
        raise ValueError("LEAF_FROZEN_SOURCE_DENOMINATOR")
    for name in paths:
        if not is_hex(sources[name], 64):
            raise ValueError("LEAF_FROZEN_SOURCE_HASH_SCHEMA")
        bounded_read(snapshot/name, 2*1024**2, sources[name])
    if context.get("runtime_metadata") != runtime_metadata():
        raise ValueError("LEAF_FROZEN_RUNTIME_METADATA_CHANGED")
    return snapshot, cache, binding


def workflow_entry_and_policies(arm):
    """Constructor-only qualification seam; no input read/native inspection."""
    if arm not in ARMS:
        raise ValueError("LEAF_ARM_SCHEMA")
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_instances import SupportPolicy
    from backend.cad_guided_noisy_mesh_planes_v5 import NoisyMeshPlanePolicyV5, NoisyMeshPlaneBudgetsV5
    from backend.cad_guided_mesh_plane_proposals_v4 import CoplanarPatchPolicyV4, CoplanarPatchBudgetsV4
    from backend.cad_guided_plane_search_v6 import PlaneSearchPolicyV6, PlaneSearchBudgetsV6
    common = dict(
        registration_policy=WholePartRegistrationPolicyV1(**CONFIG["registration"]),
        measurement_policy=NativeGuidedPolicy(**CONFIG["measurement"]),
        instance_policy=SupportPolicy(**CONFIG["support"]), allow_ocp=True)
    noisy_policy = dict(CONFIG["noisy_policy"])
    noisy_policy["tile_triangles"] = tuple(noisy_policy["tile_triangles"])
    common.update(position_error_bound_mm=CONFIG["position_error_bound_mm"],
        noisy_policy=NoisyMeshPlanePolicyV5(**noisy_policy),
        noisy_budgets=NoisyMeshPlaneBudgetsV5(**CONFIG["noisy_budgets"]),
        cad_patch_policy=CoplanarPatchPolicyV4(**CONFIG["cad_patch_policy"]),
        cad_patch_budgets=CoplanarPatchBudgetsV4(**CONFIG["cad_patch_budgets"]),
        search_policy=PlaneSearchPolicyV6(**CONFIG["search_policy"]),
        search_budgets=PlaneSearchBudgetsV6(**CONFIG["search_budgets"]))
    from backend.cad_guided_triangle_surface_v3 import TriangleSurfaceBudgets
    from backend.cad_guided_global_registration_v3 import SurfaceRegistrationBudgetsV3
    common.update(
        surface_budgets=TriangleSurfaceBudgets(**CONFIG["surface_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["surface_budgets"]),
        registration_budgets=SurfaceRegistrationBudgetsV3(**CONFIG["registration_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["registration_budgets"]))
    if arm == "control":
        from backend.cad_guided_mesh_offline_v10 import MeshWorkflowPolicyV10, inspect_cad_guided_mesh_v10
        from backend.cad_guided_plane_initialization_v10 import PlaneInitializationBudgetsV10
        common.update(
            workflow_policy=MeshWorkflowPolicyV10(**CONFIG["workflow"], native_adapter_version="v1"),
            initialization_budgets=PlaneInitializationBudgetsV10(**CONFIG["initialization_budgets"],
                max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]))
        return inspect_cad_guided_mesh_v10, common
    from backend.cad_guided_mesh_offline_v11 import MeshWorkflowPolicyV11, inspect_cad_guided_mesh_v11
    from backend.cad_guided_plane_initialization_v11 import PlaneInitializationBudgetsV11
    from backend.cad_guided_two_plane_context_v1 import TwoPlaneContextBudgetsV1
    from backend.cad_guided_two_plane_schedule_v1 import TwoPlaneScheduleBudgetsV1
    if (asdict(TwoPlaneContextBudgetsV1()) != TWO_PLANE_DECLARATION["context_budgets"]
            or asdict(TwoPlaneScheduleBudgetsV1()) != TWO_PLANE_DECLARATION["schedule_budgets"]):
        raise ValueError("FROZEN_TWO_PLANE_DEFAULT_BUDGETS_CHANGED")
    common.update(workflow_policy=MeshWorkflowPolicyV11(**CONFIG["workflow"], native_adapter_version="v1"),
        initialization_budgets=PlaneInitializationBudgetsV11(**CONFIG["initialization_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        two_plane_enabled=True)
    return inspect_cad_guided_mesh_v11, common


def worker():
    # Parent sends no Gold, case/family identity or original source path.
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["case_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "INFERENCE")
    initialize_worker(snapshot, cache, "backend", LIMITS["case_cpu_seconds"])
    entry, policies = workflow_entry_and_policies(request["arm"])
    scan = bounded_read(cache/"scan.ply", LIMITS["max_scan_bytes"], request["scan_sha256"])
    cad = bounded_read(cache/"cad.step", LIMITS["max_cad_bytes"], request["cad_sha256"])
    print(json.dumps(dict(status="PUBLIC_WORKFLOW_CALL_BEGIN", arm=request["arm"],
                          internal_stage="UNKNOWN_UNTIL_RETURNED_RESULT")), flush=True)
    result = entry(scan_bytes=scan, cad_bytes=cad, **policies)
    blob = encoded(result)
    if len(blob) > LIMITS["case_result_bytes"]:
        raise ValueError("RESULT_BYTE_BUDGET")
    publish_bytes(cache/"RESULT.json", blob)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(status="INFERENCE_TERMINAL", result_sha256=digest(blob),
                          worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
                          worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)


def process_table():
    return supervision_module().process_table()


def descendants(table, root_pid):
    return supervision_module().descendants(table, root_pid)


def clean_environment(cache):
    return supervision_module().clean_environment(cache)


def runner_command(mode):
    return [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
            str(RUN/"source/scripts/run_gold11_mesh_development_v4.py"), mode]


def memory_budget(tree_rss, parent_rss, rss_limit, *, include_parent=False, parent_rss_limit=None):
    return supervision_module().memory_budget(tree_rss, parent_rss, rss_limit,
        include_parent=include_parent, parent_rss_limit=parent_rss_limit)


def supervise_process(command, request, *, cache, log_path, wall_seconds, rss_limit,
                      log_limit, new_session=False, parent_rss_limit=None):
    return supervision_module().supervise_process(command, request, cache=cache,
        log_path=log_path, wall_seconds=wall_seconds, rss_limit=rss_limit,
        log_limit=log_limit, new_session=new_session, parent_rss_limit=parent_rss_limit)


def coordinator_memory_check():
    table = process_table()
    rss = table.get(os.getpid(), {}).get("rss")
    if (rss is None or rss > LIMITS["coordinator_rss_bytes"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMITS["coordinator_rss_bytes"]):
        raise RuntimeError("COORDINATOR_MEMORY_UNAVAILABLE_OR_OVER_LIMIT")


def retain_worker_result(cache, destination, *, scorer=False):
    """Retain the one complete/partial derived result, never raw input caches.

    Called after supervision ends, before private raw-input cache cleanup. A
    linked complete name and its leftover partial share bytes; retain them once
    and record that identity. No partial is parsed/scored as a successful result.
    """
    filename = "SCORER_RESULT.json" if scorer else "RESULT.json"
    cap = LIMITS["scorer_result_bytes"] if scorer else LIMITS["case_result_bytes"]
    complete = cache/filename
    prefix = "."+filename+".partial-"
    partials = []
    for path in cache.glob(prefix+"*"):
        if partials or not is_hex(path.name.removeprefix(prefix), 16):
            raise ValueError("UNEXPECTED_WORKER_RESULT_PARTIAL_SET")
        partials.append(path)
    try:
        complete.lstat()
        has_complete = True
    except FileNotFoundError:
        has_complete = False
    if not has_complete and not partials:
        return dict(status="NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN", artifacts=[])
    source = complete if has_complete else partials[0]
    if has_complete and partials:
        left, right = complete.lstat(), partials[0].lstat()
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            raise ValueError("UNBOUND_COMPLETE_AND_PARTIAL_WORKER_RESULTS")
    blob = _bounded_read(source, cap, allow_empty=not has_complete)
    name = filename if has_complete else filename.removesuffix(".json")+"_PARTIAL.bin"
    publish_bytes(destination/name, blob)
    return dict(status="COMPLETE_RESULT_BYTES_RETAINED" if has_complete else "PARTIAL_BYTES_NOT_SCORABLE",
        artifacts=[dict(path=name, sha256=digest(blob), bytes=len(blob))],
        leftover_partial_names=[path.name for path in partials], partial_alias_bytes_retained_once=bool(has_complete and partials))


def supervise_case(case, arm, destination, remaining_wall):
    started = time.monotonic()
    outcome = unavailable("INFERENCE_NOT_STARTED")
    receipt = dict(status="TERMINAL", budget_stop=None, source_binding_valid=False,
                   derived_result_retention_complete=None,
                   inference_started=False, workflow_stage_diagnostics=workflow_stage_diagnostics(None, arm))
    try:
        destination.mkdir(parents=True, exist_ok=False)
        exclusive_json(destination/"STARTED.json", dict(case_id=case["case_token"], arm=arm,
                       limits=LIMITS, no_resume=True))
        coordinator_memory_check()
        if remaining_wall < LIMITS["case_wall_seconds"]:
            raise ValueError("TOTAL_WALL_RESERVE_UNAVAILABLE")
        with deadline(min(LIMITS["parent_read_seconds"], remaining_wall)):
            scan = _bounded_read(case["scan_path"], LIMITS["max_scan_bytes"], case["scan_sha256"])
            cad = _bounded_read(case["cad_path"], LIMITS["max_cad_bytes"], case["cad_sha256"])
        receipt["source_binding_valid"] = True
        with tempfile.TemporaryDirectory(prefix="gold11-mesh-v4-") as temporary:
            cache = Path(temporary).resolve()
            with deadline(LIMITS["parent_read_seconds"]):
                for name, blob in (("scan.ply", scan), ("cad.step", cad)):
                    publish_bytes(cache/name, blob)
            del scan, cad
            receipt["inference_started"] = "UNKNOWN_UNTIL_RETURNED_RESULT"
            receipt["worker_dispatch_attempted"] = True
            request = issue_leaf_dispatch("INFERENCE", cache, arm=arm,
                           scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"])
            receipt["leaf_dispatch_id"] = request["dispatch_id"]
            try:
                with deadline(max(.001, LIMITS["case_wall_seconds"]-(time.monotonic()-started))):
                    worker_receipt = supervise_process(runner_command("--worker"), request, cache=cache,
                        log_path=destination/"OUTPUT.txt", wall_seconds=max(.01, LIMITS["case_wall_seconds"]-(time.monotonic()-started)),
                        rss_limit=LIMITS["sampled_worker_tree_rss_bytes"], log_limit=LIMITS["case_log_bytes"],
                        parent_rss_limit=LIMITS["coordinator_rss_bytes"])
                    receipt.update(worker_receipt)
            finally:
                with deadline(LIMITS["parent_read_seconds"]):
                    try:
                        receipt["retained_result_artifacts"] = retain_worker_result(cache, destination)
                        receipt["derived_result_retention_complete"] = True
                    except Exception as error:
                        # Raw-input temp cleanup remains mandatory. If derived
                        # retention fails, do not hide its possible loss or use
                        # that case to release a completed campaign score.
                        receipt["derived_result_retention_complete"] = False
                        receipt["retained_result_artifacts"] = dict(
                            status="RETENTION_FAILED_DERIVED_BYTES_MAY_BE_LOST_WITH_RAW_CACHE_CLEANUP",
                            error_type=type(error).__name__, artifacts=[])
                        raise
            if receipt["exit_code"] != 0 or receipt["budget_stop"]:
                raise ValueError("INFERENCE_WORKER_FAILED")
            result_blob = bounded_read(destination/"RESULT.json", LIMITS["case_result_bytes"],
                receipt["retained_result_artifacts"]["artifacts"][0]["sha256"])
            receipt["result_sha256"] = digest(result_blob)
            receipt["inference_started"] = True
            with deadline(LIMITS["parent_read_seconds"]):
                payload = decode(result_blob)
                outcome = compact_result(payload)
                receipt["workflow_stage_diagnostics"] = workflow_stage_diagnostics(payload, arm)
                del payload
            del result_blob
            coordinator_memory_check()
            if time.monotonic()-started > LIMITS["case_wall_seconds"]:
                receipt["budget_stop"] = "CASE_PARENT_WALL_LIMIT_AFTER_RESULT_RETENTION"
                raise ValueError("CASE_PARENT_WALL_LIMIT_AFTER_RESULT_RETENTION")
    except Exception as error:
        outcome = unavailable(str(error)[:160] if isinstance(error, (ValueError, TimeoutError))
                              else "SUPERVISOR_EXCEPTION:"+type(error).__name__)
        receipt["failure_type"] = type(error).__name__
    receipt["parent_wall_seconds"] = time.monotonic()-started
    # Storage failures propagate to the coordinator/outer terminal-fill path;
    # they must not claim a durable outcome that could not actually be written.
    destination.mkdir(parents=True, exist_ok=True)
    receipt["scoring_input_sha256"] = digest(encoded(outcome)+b"\n")
    exclusive_json(destination/"SCORING_INPUT.json", outcome)
    exclusive_json(destination/"COMPLETED.json", receipt)
    print(json.dumps(dict(case_id=case["case_token"], arm=arm, status=outcome.get("status", "MALFORMED"),
                          phase=outcome.get("phase"), wall_seconds=receipt["parent_wall_seconds"],
                          budget_stop=receipt["budget_stop"])), flush=True)
    return outcome, receipt


def selected_references(cases):
    blob = bounded_read(CATALOG, 8*1024**2, CATALOG_SHA)
    payload = decode(blob)
    by_sha = {case["scan_sha256"]: case for case in cases}
    selected = {}
    for row in payload["cases"]:
        case = by_sha.get(row.get("scan_sha256"))
        if case is None:
            continue
        key = case["case_token"]
        if (key in selected or row.get("case_id") != case["reference_id"]
                or row.get("scan_path") != case["scan_path"]
                or str(row.get("part_id")) != case["part_key"]):
            raise ValueError("REFERENCE_IDENTITY_OR_DUPLICATE")
        selected[key] = row
    if set(selected) != {case["case_token"] for case in cases}:
        raise ValueError("REFERENCE_CASE_DENOMINATOR")
    if sum(len(row["bends"]) for row in selected.values()) != 71 or digest(encoded(selected)) != SELECTED_SHA:
        raise ValueError("REFERENCE_SNAPSHOT_MISMATCH")
    return selected


def verify_pre_run_audit(bundle):
    audit = decode(bounded_read(RUN/"PRE_RUN_AUDIT.json", 256*1024))
    if not (audit.get("decision") == "GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN"
            and audit.get("bundle_sha256") == digest(encoded(bundle))
            and isinstance(audit.get("reviewer"), str) and audit["reviewer"].strip()
            and audit.get("blocking_findings") == []):
        raise ValueError("EXACT_INDEPENDENT_PRE_RUN_AUDIT_REQUIRED")
    return audit


def terminalize_case_records(bundle, reason):
    """Fill missing terminal records only; never rerun/replace an existing arm."""
    records = []
    deadline_exhausted = False
    for case in bundle["cases"]:
        for arm in bundle["arms"]:
            destination = RUN/"cases"/case["case_token"]/arm
            row = dict(case_id=case["case_token"], arm=arm, durable=False)
            if deadline_exhausted:
                row.update(storage_error="TimeoutError", disposition="NOT_ATTEMPTED_AFTER_TERMINAL_FILL_DEADLINE")
                records.append(row)
                continue
            try:
                with deadline(LIMITS["parent_read_seconds"]), artifact_directory_walk(), artifact_alarm_guard():
                    destination.mkdir(parents=True, exist_ok=True)
                    module = artifact_module()
                    def validate(blob, *, completed=False):
                        prior = decode(blob)
                        return (type(prior) is dict and type(prior.get("status")) is str
                            and (not completed or prior["status"] == "TERMINAL"))
                    scoring = module.ensure_terminal_record_v4(destination, "SCORING_INPUT.json",
                        lambda: encoded(unavailable(reason, "GLOBAL_TERMINALIZATION"))+b"\n",
                        validate, max_bytes=LIMITS["compact_case_bytes"])
                    completion = module.ensure_terminal_record_v4(destination, "COMPLETED.json",
                        lambda: encoded(dict(status="TERMINAL", reason=reason, no_resume=True,
                            scoring_input_sha256=scoring["sha256"],
                            inference_started="UNKNOWN_IF_STARTED_WITHOUT_RECEIPT"))+b"\n",
                        lambda blob: validate(blob, completed=True), max_bytes=LIMITS["compact_case_bytes"])
                    row["records"] = {name: {key: record[key] for key in
                        ("disposition", "sha256", "bytes", "identity")}
                        for name, record in (("SCORING_INPUT.json", scoring), ("COMPLETED.json", completion))}
                row["durable"] = True
            except TimeoutError:
                row.update(storage_error="TimeoutError", disposition="INTERRUPTED_BY_TERMINAL_FILL_DEADLINE")
                deadline_exhausted = True
            except Exception as error:
                row["storage_error"] = type(error).__name__
            records.append(row)
    return records


def scorer_worker():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["scorer_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "SCORER")
    initialize_worker(snapshot, cache, "scripts", LIMITS["scorer_cpu_seconds"])
    payload = decode(bounded_read(cache/"SCORING_REQUEST.json", LIMITS["scorer_request_bytes"],
                                 request["request_sha256"]))
    from scripts.gold11_mesh_scoring_v1 import score_gold11_mesh_workflows_v1
    result = score_gold11_mesh_workflows_v1(**payload)
    blob = encoded(result)
    if len(blob) > LIMITS["scorer_result_bytes"]:
        raise ValueError("SCORER_RESULT_BYTE_BUDGET")
    publish_bytes(cache/"SCORER_RESULT.json", blob)
    print(json.dumps(dict(status="SCORER_TERMINAL", result_sha256=digest(blob))), flush=True)


def score_after_terminal(bundle, outcomes, receipts):
    # Only this post-terminal coordinator function can open the fixed catalog.
    verify_bundle()
    selected = selected_references(bundle["cases"])
    # New durable selected71 snapshot for later cached-control report joining.
    publish_bytes(RUN/"SELECTED_REFERENCES.json", encoded(selected))
    request = dict(reference_records_by_case={key: row["bends"] for key, row in selected.items()},
                   control_results_by_case=outcomes["control"], challenger_results_by_case=outcomes["challenger"],
                   case_family_by_id={case["case_token"]: case["part_key"] for case in bundle["cases"]})
    blob = encoded(request)
    if len(blob) > LIMITS["scorer_request_bytes"]:
        raise ValueError("SCORER_REQUEST_BYTE_BUDGET")
    with tempfile.TemporaryDirectory(prefix="gold11-score-v4-") as temporary:
        cache = Path(temporary).resolve()
        with deadline(LIMITS["parent_read_seconds"]):
            publish_private_scoring_request(cache, blob)
        dispatch = issue_leaf_dispatch("SCORER", cache, request_sha256=digest(blob))
        try:
            with deadline(LIMITS["scorer_wall_seconds"]):
                receipt = supervise_process(runner_command("--score-worker"), dispatch,
                    cache=cache, log_path=RUN/"SCORER_OUTPUT.txt", wall_seconds=LIMITS["scorer_wall_seconds"],
                    rss_limit=LIMITS["scorer_rss_bytes"], log_limit=LIMITS["case_log_bytes"],
                    parent_rss_limit=LIMITS["coordinator_rss_bytes"])
            exclusive_json(RUN/"SCORER_RECEIPT.json", receipt)
        finally:
            with deadline(LIMITS["parent_read_seconds"]):
                retained = retain_worker_result(cache, RUN, scorer=True)
                exclusive_json(RUN/"SCORER_ARTIFACT_RETENTION.json", retained)
        if receipt["exit_code"] != 0 or receipt["budget_stop"]:
            raise ValueError("SCORER_FAILED_OR_BUDGET_STOP")
        score_blob = bounded_read(RUN/"SCORER_RESULT.json", LIMITS["scorer_result_bytes"],
            retained["artifacts"][0]["sha256"])
        scorer_result_sha = digest(score_blob)
        report = decode(score_blob)
    # Bind the source/runtime AFTER scorer execution, not only before import.
    verify_bundle()
    bounded_read(CATALOG, 8*1024**2, CATALOG_SHA)
    # Correct descriptive experiment/arm identity only. The scorer module,
    # thresholds, complete denominator, numeric metrics and undefined values
    # remain unchanged; neither .458 nor an undefined control F1 is beaten here.
    report.update(experiment_schema="GOLD11_MESH_DEVELOPMENT_V4",
                  control_identity="FRESH_V10_WORKFLOW_NATIVE_V1_NOT_HISTORICAL_INCUMBENT",
                  challenger_identity="FRESH_V11_TWO_PLANE_REGISTRATION_WORKFLOW_NATIVE_V1",
                  arm_definitions=ARM_DEFINITIONS, comparator=COMPARATOR,
                  retained_control_policy=RETAINED_CONTROL_POLICY, certificate_caps=CERTIFICATE_CAPS,
                  two_plane_declaration=TWO_PLANE_DECLARATION, artifact_policy=ARTIFACT_POLICY,
                  position_error_bound_mm=CONFIG["position_error_bound_mm"],
                  position_error_declaration=POSITION_ERROR_DECLARATION,
                  workflow_stage_attribution={
                      arm: {case: row.get("workflow_stage_diagnostics")
                            for case, row in arm_rows.items()}
                      for arm, arm_rows in receipts.items()},
                  phase_attribution_caveat="MISSING_OR_KILLED_INTERNAL_STAGE_UNKNOWN_USE_RETURNED_NESTED_LEDGERS",
                  bundle_sha256=digest(encoded(bundle)), reference_snapshot_sha256=SELECTED_SHA,
                  scorer_result_sha256=scorer_result_sha, scoring_request_sha256=digest(blob),
                  all_declared_input_bytes_verified=all(row.get("source_binding_valid") is True
                      for rows in receipts.values() for row in rows.values()),
                  historical_score_beaten_claim_allowed=False, source_before_after_consistent=True,
                  case_declared_coordinate_frames={key: row.get("coordinate_frame")
                                                   for key, row in selected.items()},
                  declared_frame_interpretation="PROVISIONAL_COMMON_RAW_FRAME_NOT_VERIFIED")
    if len(encoded(report))+1 > LIMITS["scorer_result_bytes"]:
        raise ValueError("SCORING_REPORT_WITH_STAGE_LEDGER_BYTE_BUDGET")
    exclusive_json(RUN/"SCORED.json", report)


def internal_coordinator():
    dispatch = decode(sys.stdin.buffer.read(4096))
    bundle = verify_bundle()
    audit = verify_pre_run_audit(bundle)
    expected = decode(bounded_read(RUN/"COORDINATOR_DISPATCH.json", 4096))
    start = decode(bounded_read(RUN/"STARTED.json", 4096))
    if (type(dispatch.get("token")) is not str or len(dispatch["token"]) != 64
            or digest(dispatch["token"].encode()) != expected.get("token_sha256")
            or dispatch.get("parent_pid") != os.getppid() or expected.get("parent_pid") != os.getppid()
            or os.getpgrp() != os.getpid() or start.get("bundle_sha256") != digest(encoded(bundle))
            or start.get("audit_sha256") != digest(encoded(audit))
            or start.get("status") != "STARTED_NO_RETRY" or start.get("limits") != LIMITS):
        raise ValueError("INTERNAL_COORDINATOR_DISPATCH_REQUIRED")
    for name in ("COMPLETED.json", "COORDINATOR_COMPLETED.json", "SCORING_UNAVAILABLE.json",
                 "FINAL_GATE.json", "FINALIZATION_FAILED.json", "INFERENCE_TERMINAL.json"):
        if marker_present(RUN/name):
            raise ValueError("COORDINATOR_RUN_ALREADY_TERMINAL")
    activate_coordinator(bundle, audit, expected, start)
    started = time.monotonic()
    outcomes = {arm: {} for arm in bundle["arms"]}
    receipts = {arm: {} for arm in bundle["arms"]}
    try:
        for case in bundle["cases"]:
            for arm in bundle["arms"]:
                outcome, receipt = supervise_case(case, arm, RUN/"cases"/case["case_token"]/arm,
                    LIMITS["total_wall_seconds"]-(time.monotonic()-started)-LIMITS["terminalization_reserve_seconds"])
                outcomes[arm][case["case_token"]] = outcome
                receipts[arm][case["case_token"]] = receipt
                coordinator_memory_check()
        if any(row.get("derived_result_retention_complete") is False
               for rows in receipts.values() for row in rows.values()):
            raise ValueError("DERIVED_RESULT_RETENTION_INCOMPLETE_NO_SCORING")
        exclusive_json(RUN/"INFERENCE_TERMINAL.json", dict(case_count=11, arm_count=2,
            terminal_record_count=sum(map(len, receipts.values())), outcomes_sha256=digest(encoded(outcomes)),
            result_bindings={arm: {case: row.get("result_sha256") for case, row in rows.items()}
                             for arm, rows in receipts.items()}, parent_wall_seconds=time.monotonic()-started))
        score_after_terminal(bundle, outcomes, receipts)
        status = "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"
    except Exception as error:
        terminal = terminalize_case_records(bundle, "COORDINATOR_ABORT_NO_RETRY")
        exclusive_json(RUN/"SCORING_UNAVAILABLE.json", dict(error_type=type(error).__name__,
                       reason=str(error)[:160], no_inference_retry=True, terminal_case_records=terminal))
        status = "TERMINAL_SCORING_UNAVAILABLE"
    exclusive_json(RUN/"COORDINATOR_COMPLETED.json", dict(status=status,
                   parent_wall_seconds=time.monotonic()-started, no_resume=True,
                   historical_comparability="UNESTABLISHED", production_use_allowed=False))
    print(json.dumps(dict(status=status, attempt=str(RUN))), flush=True)


def verify_parent_artifacts(bundle, supervisor_receipt, evidence):
    """Fixed owned files only. Never parse raw geometry or reopen input/Gold.

    A failed inference may legitimately have no result/log, but that absence is
    explicit. Any claimed retained complete/partial bytes must still match.
    A partial manifest is retained on failure and cannot qualify the campaign.
    """
    evidence.update(schema="GOLD11_OWNED_ARTIFACT_CUSTODY_V4", complete=False, files={}, logs={})
    def read(relative, cap, expected=None, *, allow_empty=False):
        blob = owned_read(relative, cap, expected, allow_empty=allow_empty)
        evidence["files"][relative] = dict(sha256=digest(blob), bytes=len(blob))
        if len(evidence["files"]) > 128:
            raise ValueError("OWNED_ARTIFACT_MANIFEST_ENTRY_BUDGET")
        return blob
    def obj(relative, cap=256*1024, expected=None):
        value = decode(read(relative, cap, expected))
        if type(value) is not dict:
            raise ValueError("OWNED_OBJECT_RECORD_REQUIRED")
        return value
    def log(relative, row, cap, *, clean=False):
        created = row.get("log_created")
        present = marker_present(RUN/relative)
        if clean and created is not True:
            raise ValueError("CLEAN_WORKER_LOG_CREATION_PROOF_REQUIRED")
        if not present:
            if created is True or clean or row.get("forwarded_log_bytes", 0) != 0:
                raise ValueError("CLAIMED_WORKER_LOG_MISSING")
            evidence["logs"][relative] = dict(status="NOT_CREATED_OR_UNOBSERVED_FAILED_ATTEMPT", bytes=None)
            return
        blob = read(relative, cap, allow_empty=True)
        if created is True:
            observed, stored = row.get("forwarded_log_bytes"), row.get("stored_log_bytes")
            if (type(observed) is not int or observed < 0 or type(stored) is not int
                    or stored != min(observed, cap) or len(blob) != stored
                    or observed > cap and not row.get("budget_stop")):
                raise ValueError("WORKER_LOG_BYTE_BINDING_MISMATCH")
            status = "COMPLETE_LOG_BYTES_BOUND" if observed <= cap else "EXPLICIT_BUDGET_TRUNCATED_LOG"
        else:
            if clean:
                raise ValueError("UNBOUND_CLEAN_WORKER_LOG")
            status = "RETURNED_FAILED_LOG_WITH_UNKNOWN_OBSERVED_BYTE_COUNT"
        evidence["logs"][relative] = dict(status=status, sha256=digest(blob), bytes=len(blob))
    def retained(row, prefix, *, scorer=False):
        full = "SCORER_RESULT.json" if scorer else "RESULT.json"
        partial = "SCORER_RESULT_PARTIAL.bin" if scorer else "RESULT_PARTIAL.bin"
        cap = LIMITS["scorer_result_bytes"] if scorer else LIMITS["case_result_bytes"]
        record = row if scorer else row.get("retained_result_artifacts")
        if record is None:
            if scorer or row.get("inference_started") is True:
                raise ValueError("RETAINED_RESULT_RECORD_REQUIRED")
            artifacts = []
        else:
            if type(record) is not dict or type(record.get("artifacts")) is not list:
                raise ValueError("RETAINED_RESULT_RECORD_SCHEMA")
            artifacts = record["artifacts"]
            status = record.get("status")
            if status == "NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN":
                if artifacts:
                    raise ValueError("UNKNOWN_RESULT_HAS_ARTIFACTS")
            elif status in ("COMPLETE_RESULT_BYTES_RETAINED", "PARTIAL_BYTES_NOT_SCORABLE"):
                name = full if status == "COMPLETE_RESULT_BYTES_RETAINED" else partial
                if (len(artifacts) != 1 or type(artifacts[0]) is not dict
                        or artifacts[0].get("path") != name or not is_hex(artifacts[0].get("sha256"), 64)
                        or type(artifacts[0].get("bytes")) is not int
                        or not int(name == full) <= artifacts[0]["bytes"] <= cap):
                    raise ValueError("RETAINED_RESULT_ARTIFACT_BINDING")
                blob = read(prefix+name, cap, artifacts[0]["sha256"], allow_empty=name == partial)
                if len(blob) != artifacts[0]["bytes"]:
                    raise ValueError("RETAINED_RESULT_ARTIFACT_LENGTH")
            else:
                raise ValueError("DERIVED_RESULT_RETENTION_NOT_COMPLETE")
        names = {entry["path"] for entry in artifacts}
        if any(marker_present(RUN/(prefix+name)) for name in (full, partial) if name not in names):
            raise ValueError("UNDECLARED_RETURNED_RESULT_ARTIFACT")
        return artifacts
    expected_cases = {case["case_token"] for case in bundle["cases"]}
    if (len(bundle["cases"]) != 11 or len(expected_cases) != 11 or bundle["arms"] != ARMS
            or any(type(case) is not str or not case or len(case) > 128
                   or "/" in case or case in (".", "..") for case in expected_cases)):
        raise ValueError("OWNED_CASE_PATH_DENOMINATOR")
    coordinator = obj("COORDINATOR_COMPLETED.json", 4096)
    if coordinator.get("status") != "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT":
        raise ValueError("COORDINATOR_DID_NOT_COMPUTE_COMPLETE_DIAGNOSTIC")
    outcomes, result_bindings = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    for case in bundle["cases"]:
        for arm in ARMS:
            key = case["case_token"]
            prefix = f"cases/{key}/{arm}/"
            receipt = obj(prefix+"COMPLETED.json", LIMITS["compact_case_bytes"])
            if receipt.get("status") != "TERMINAL" or not is_hex(receipt.get("scoring_input_sha256"), 64):
                raise ValueError("CASE_TERMINAL_SCORING_INPUT_BINDING")
            outcome = obj(prefix+"SCORING_INPUT.json", LIMITS["compact_case_bytes"], receipt["scoring_input_sha256"])
            if type(outcome.get("status")) is not str:
                raise ValueError("CASE_SCORING_INPUT_SCHEMA")
            if marker_present(RUN/(prefix+"STARTED.json")):
                case_start = obj(prefix+"STARTED.json")
                if (case_start.get("case_id") != key or case_start.get("arm") != arm
                        or case_start.get("limits") != LIMITS or case_start.get("no_resume") is not True):
                    raise ValueError("CASE_START_IDENTITY_OR_LIMIT_BINDING")
            elif (receipt.get("inference_started") is not False or receipt.get("source_binding_valid") is not False
                  or outcome.get("status") != "UNAVAILABLE"):
                raise ValueError("STARTED_CASE_RECORD_MISSING")
            artifacts = retained(receipt, prefix)
            result_sha = receipt.get("result_sha256")
            if result_sha is not None and (not is_hex(result_sha, 64) or len(artifacts) != 1
                    or artifacts[0]["path"] != "RESULT.json" or artifacts[0]["sha256"] != result_sha):
                raise ValueError("CASE_SELECTED_RESULT_HASH_BINDING")
            clean = receipt.get("exit_code") == 0 and receipt.get("budget_stop") is None and "failure_type" not in receipt
            if "exit_code" in receipt and receipt["exit_code"] is not None and type(receipt["exit_code"]) is not int:
                raise ValueError("CASE_EXIT_CODE_SCHEMA")
            if not clean and outcome.get("status") != "UNAVAILABLE":
                raise ValueError("FAILED_CASE_MUST_NOT_SUPPLY_USABLE_SCORE_PROJECTION")
            expected_dispatch = digest(encoded(dict(role="INFERENCE", binding=dict(arm=arm,
                scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"]))))
            if (receipt.get("worker_dispatch_attempted") is True
                    and receipt.get("leaf_dispatch_id") != expected_dispatch):
                raise ValueError("CASE_LEAF_DISPATCH_IDENTITY_BINDING")
            if clean:
                if receipt.get("worker_dispatch_attempted") is not True:
                    raise ValueError("CLEAN_CASE_REQUIRES_BOUND_DISPATCH")
                evidence.setdefault("required_leaf_dispatch_ids", []).append(expected_dispatch)
            log(prefix+"OUTPUT.txt", receipt, LIMITS["case_log_bytes"], clean=clean)
            outcomes[arm][key], result_bindings[arm][key] = outcome, result_sha
    terminal = obj("INFERENCE_TERMINAL.json")
    if (any(type(terminal.get(key)) is not int or terminal[key] != count
            for key, count in (("case_count", 11), ("arm_count", 2), ("terminal_record_count", 22)))
            or terminal.get("outcomes_sha256") != digest(encoded(outcomes))
            or terminal.get("result_bindings") != result_bindings):
        raise ValueError("ALL22_TERMINAL_OUTCOME_RESULT_BINDING")
    del outcomes, result_bindings
    log("COORDINATOR_OUTPUT.txt", supervisor_receipt, LIMITS["coordinator_log_bytes"], clean=True)
    scorer_receipt = obj("SCORER_RECEIPT.json")
    if type(scorer_receipt.get("exit_code")) is not int or scorer_receipt["exit_code"] != 0 or scorer_receipt.get("budget_stop") is not None:
        raise ValueError("SCORER_NOT_CLEANLY_COMPLETED")
    log("SCORER_OUTPUT.txt", scorer_receipt, LIMITS["case_log_bytes"], clean=True)
    retention = obj("SCORER_ARTIFACT_RETENTION.json")
    scorer_artifacts = retained(retention, "", scorer=True)
    if len(scorer_artifacts) != 1 or scorer_artifacts[0]["path"] != "SCORER_RESULT.json":
        raise ValueError("COMPLETE_ORIGINAL_SCORER_RESULT_REQUIRED")
    original = obj("SCORER_RESULT.json", LIMITS["scorer_result_bytes"], scorer_artifacts[0]["sha256"])
    score = obj("SCORED.json", LIMITS["scorer_result_bytes"])
    families = {case["case_token"]: case["part_key"] for case in bundle["cases"]}
    if (score.get("experiment_schema") != "GOLD11_MESH_DEVELOPMENT_V4"
            or score.get("schema") != "GOLD11_MESH_GEOMETRIC_DEVELOPMENT_COMPARISON_V1"
            or score.get("status") != "COMPUTED_DEVELOPMENT_DIAGNOSTIC"
            or any(type(score.get(key)) is not int or score[key] != count
                for key, count in (("case_count", 11), ("family_count", 6), ("reference_line_count", 71)))
            or score.get("bundle_sha256") != digest(encoded(bundle))
            or score.get("reference_snapshot_sha256") != SELECTED_SHA
            or score.get("retained_control_policy") != RETAINED_CONTROL_POLICY
            or score.get("arm_definitions") != ARM_DEFINITIONS or score.get("comparator") != COMPARATOR
            or score.get("certificate_caps") != CERTIFICATE_CAPS
            or score.get("two_plane_declaration") != TWO_PLANE_DECLARATION
            or score.get("artifact_policy") != ARTIFACT_POLICY
            or score.get("scorer_result_sha256") != scorer_artifacts[0]["sha256"]
            or any(score.get(key) != value for key, value in original.items() if key != "control_identity")):
        raise ValueError("ORIGINAL_SCORER_OR_ENRICHED_REPORT_BINDING")
    for arm in ARMS:
        lane = score.get(arm)
        if (type(lane) is not dict or type(lane.get("case_dispositions")) is not dict
                or set(lane["case_dispositions"]) != expected_cases
                or type(lane.get("families")) is not dict or set(lane["families"]) != set(families.values())
                or any(type(row) is not dict or row.get("family_id") != families[key]
                       for key, row in lane["case_dispositions"].items())):
            raise ValueError("COMPLETE_CASE_FAMILY_SCORE_LANE_REQUIRED")
    read("SELECTED_REFERENCES.json", 8*1024**2, SELECTED_SHA)
    evidence.update(complete=True, case_count=11, arm_count=2,
        scorer_result_sha256=scorer_artifacts[0]["sha256"],
        scoring_request_sha256=score["scoring_request_sha256"],
        scored_report_sha256=evidence["files"]["SCORED.json"]["sha256"], reference_snapshot_sha256=SELECTED_SHA)
    if len(encoded(evidence)) > 256*1024:
        evidence["complete"] = False
        raise ValueError("OWNED_ARTIFACT_MANIFEST_BYTE_BUDGET")


def _tree_fingerprint(info):
    return dict(device=info.st_dev, inode=info.st_ino, mode=info.st_mode,
        links=info.st_nlink, bytes=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def _open_tree_directory(relative):
    """New mixed-directory adapter: pinned/no-follow components, no recursion discovery."""
    target = RUN if relative == "" else RUN/relative
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in target.parts[1:]:
            following = os.open(part, flags, dir_fd=fd)
            previous, fd = fd, following
            os.close(previous)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _tree_names(fd, maximum):
    """Complete bounded names or explicit one-name overflow; never glob-filter."""
    names = []
    scan = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        with os.scandir(scan) as rows:
            for row in rows:
                if (type(row.name) is not str or not row.name or row.name in (".", "..")
                        or "/" in row.name or "\x00" in row.name
                        or len(os.fsencode(row.name)) > 255):
                    raise ValueError("TREE_ENTRY_NAME_SCHEMA")
                if len(names) >= maximum:
                    return sorted(names), row.name
                if row.name in names:
                    raise ValueError("TREE_DUPLICATE_DIRECTORY_ENTRY")
                names.append(row.name)
    finally:
        os.close(scan)
    return sorted(names), None


class _CampaignInventoryStop(ValueError):
    """Stop the whole tree on a global admission failure, not only one branch."""


def _campaign_schema(bundle, artifact_evidence, control_bindings, published):
    """Fixed namespace only. Optional names still require type/size inspection.

    Scientific128 is the original V3 files-map scope, not a new all-source cap.
    Source/control/dispatch paths have separate schema-derived slot bounds.
    Every unrecognized entry, wherever found, additionally charges scientific128.
    """
    files, directories = {}, {"": {}}
    def add(relative, cap, category, *, required=False, sha=None, size=None):
        parts = Path(relative).parts
        if (not parts or Path(relative).is_absolute() or relative != "/".join(parts)
                or any(part in (".", "..") for part in parts)):
            raise ValueError("CAMPAIGN_SCHEMA_PATH_INVALID")
        if relative in files:
            raise ValueError("CAMPAIGN_SCHEMA_PATH_DUPLICATE")
        for i in range(len(parts)):
            parent, name = "/".join(parts[:i]), parts[i]
            kind = "file" if i == len(parts)-1 else "directory"
            prior = directories.setdefault(parent, {}).setdefault(name, kind)
            if prior != kind:
                raise ValueError("CAMPAIGN_SCHEMA_FILE_DIRECTORY_CONFLICT")
            if kind == "directory":
                directories.setdefault("/".join(parts[:i+1]), {})
        files[relative] = dict(max_bytes=cap, sha256=sha, bytes=size,
                               required=required, category=category)
    for name in SOURCES:
        add("source/"+name, 2*1024**2, "SOURCE", required=True, sha=bundle["source_sha256"][name])
    root_controls = {"FROZEN.json": 256*1024, "PRE_RUN_AUDIT.json": 256*1024,
        "STARTED.json": 4096, "COORDINATOR_DISPATCH.json": 4096,
        "COORDINATOR_VERIFIED.json": 32768, "SCORING_UNAVAILABLE.json": 256*1024,
        "FINALIZATION_FAILED.json": 256*1024}
    root_science = {"COORDINATOR_COMPLETED.json": 4096,
        "COORDINATOR_OUTPUT.txt": LIMITS["coordinator_log_bytes"],
        "INFERENCE_TERMINAL.json": 256*1024, "SCORER_RECEIPT.json": 256*1024,
        "SCORER_OUTPUT.txt": LIMITS["case_log_bytes"], "SCORER_ARTIFACT_RETENTION.json": 256*1024,
        "SCORER_RESULT.json": LIMITS["scorer_result_bytes"],
        "SCORER_RESULT_PARTIAL.bin": LIMITS["scorer_result_bytes"],
        "SCORED.json": LIMITS["scorer_result_bytes"], "SELECTED_REFERENCES.json": 8*1024**2}
    for name, cap in root_controls.items():
        add(name, cap, "CONTROL", required=name in control_bindings, sha=control_bindings.get(name))
    for name, cap in root_science.items():
        row = artifact_evidence.get("files", {}).get(name, {})
        add(name, cap, "SCIENTIFIC", required=bool(row), sha=row.get("sha256"), size=row.get("bytes"))
    case_names = {"STARTED.json": 256*1024, "COMPLETED.json": LIMITS["compact_case_bytes"],
        "SCORING_INPUT.json": LIMITS["compact_case_bytes"], "OUTPUT.txt": LIMITS["case_log_bytes"],
        "RESULT.json": LIMITS["case_result_bytes"], "RESULT_PARTIAL.bin": LIMITS["case_result_bytes"]}
    bindings = []
    for case in bundle["cases"]:
        key = case["case_token"]
        if type(key) is not str or not key or len(key) > 128 or "/" in key or key in (".", ".."):
            raise ValueError("CAMPAIGN_CASE_PATH_SCHEMA")
        for arm in ARMS:
            for name, cap in case_names.items():
                relative = f"cases/{key}/{arm}/{name}"
                row = artifact_evidence.get("files", {}).get(relative, {})
                add(relative, cap, "SCIENTIFIC", required=name in ("COMPLETED.json", "SCORING_INPUT.json"),
                    sha=row.get("sha256"), size=row.get("bytes"))
            bindings.append(("INFERENCE", dict(arm=arm, scan_sha256=case["scan_sha256"],
                                               cad_sha256=case["cad_sha256"])))
    request_sha = artifact_evidence.get("scoring_request_sha256")
    if request_sha is not None:
        if not is_hex(request_sha, 64):
            raise ValueError("CAMPAIGN_SCORER_REQUEST_BINDING")
        bindings.append(("SCORER", dict(request_sha256=request_sha)))
    # No discovered name is promoted into the allowed set. A failed scorer for
    # which no complete request binding survives stays an unknown/vetoed name.
    for role, binding in bindings:
        identifier = digest(encoded(dict(role=role, binding=binding)))
        for suffix in (".json", ".CLAIMED.json"):
            name = "leaf_dispatch/"+identifier+suffix
            add(name, 16384, "DISPATCH", required=role == "SCORER" or
                identifier in artifact_evidence.get("required_leaf_dispatch_ids", []))
            files[name]["leaf"] = dict(role=role, binding=binding, identifier=identifier,
                                       claim=suffix == ".CLAIMED.json")
    for name, row in published.items():
        if name not in ("COMPLETED.json", "FINAL_GATE.json"):
            raise ValueError("UNKNOWN_FINAL_PUBLICATION_PHASE")
        add(name, 256*1024, "CONTROL", required=True, sha=row["sha256"], size=row["bytes"])
    for name in ("coordinator_work", "leaf_dispatch"):
        directories[""][name] = "directory"
        directories.setdefault(name, {})
    return files, directories


def _verify_tree_leaf_bindings(files, result, control_bindings):
    """Authenticate neutral issued/claimed records; no cache paths are followed."""
    present = {path: spec for path, spec in files.items() if "leaf" in spec and path in result["entries"]}
    if not present:
        return
    context_sha = control_bindings.get("COORDINATOR_VERIFIED.json")
    if not is_hex(context_sha, 64):
        raise ValueError("LEAF_TREE_REQUIRES_BOUND_COORDINATOR_CONTEXT")
    blob = owned_read("COORDINATOR_VERIFIED.json", 32768, context_sha)
    result["read_bytes"] += len(blob)
    context = decode(blob)
    expected_context = digest(encoded(context))
    records = {}
    for path, spec in sorted(present.items(), key=lambda row: row[1]["leaf"]["claim"]):
        observed = result["entries"][path]
        sha = observed.get("sha256")
        if not is_hex(sha, 64):
            raise ValueError("LEAF_TREE_BYTES_NOT_COMPLETE")
        blob = owned_read(path, 16384, sha)
        result["read_bytes"] += len(blob)
        row, leaf = decode(blob), spec["leaf"]
        if type(row) is not dict:
            raise ValueError("LEAF_TREE_OBJECT_REQUIRED")
        if not leaf["claim"]:
            if (set(row) != {"schema", "role", "binding", "cache", "cache_identity", "coordinator_pid",
                            "context_sha256", "token_sha256"}
                    or row["schema"] != LEAF_SCHEMA or row["role"] != leaf["role"]
                    or row["binding"] != leaf["binding"] or row["context_sha256"] != expected_context
                    or type(row["coordinator_pid"]) is not int or row["coordinator_pid"] != context["coordinator_pid"]
                    or not is_hex(row["token_sha256"], 64)
                    or type(row["cache"]) is not str or not row["cache"].startswith("/") or len(row["cache"]) > 4096
                    or type(row["cache_identity"]) is not dict or set(row["cache_identity"]) != {"device", "inode", "uid"}
                    or any(type(value) is not int or value < 0 for value in row["cache_identity"].values())):
                raise ValueError("ISSUED_LEAF_TREE_BINDING")
            records[leaf["identifier"]] = row
        else:
            prior = records.get(leaf["identifier"])
            if (prior is None or set(row) != {"worker_pid", "coordinator_pid", "role", "dispatch_sha256", "no_retry"}
                    or type(row["worker_pid"]) is not int or row["worker_pid"] <= 0
                    or type(row["coordinator_pid"]) is not int or row["coordinator_pid"] != context["coordinator_pid"]
                    or row["role"] != leaf["role"] or row["dispatch_sha256"] != digest(encoded(prior))
                    or row["no_retry"] is not True):
                raise ValueError("CLAIMED_LEAF_TREE_BINDING")
        observed["neutral_control_binding"] = "VERIFIED_DECLARED_SCOPE_NOT_LIVE_CACHE_OR_PROCESS_ATTESTATION"


def verify_campaign_inventory(bundle, artifact_evidence, control_bindings, *, published=None, previous=None):
    """New unqualified complete-tree adapter, only after all writers stop.

    Exact schema-derived branches; unknown directories are metadata-only vetoes,
    never recursed. Leaf inventories use the qualified helper, while mixed
    branches add pinned two-pass enumeration and read-only existing-file checks.
    This returns a bounded partial ledger on ordinary failures, never a favorable
    prefix. The caller supplies the still-active absolute campaign deadline.
    """
    result = dict(schema="COMPLETE_CAMPAIGN_TREE_CUSTODY_V4", complete=False, entries={},
        directories={}, errors=[], scientific_entries=0, observed_entries=0, read_bytes=0,
        phase="VALIDATION", inventory_scope="SCHEMA_TREE_WITH_ALL_NAMES_NO_UNKNOWN_CONTENT_READ",
        new_adapter_qualification="SEPARATE_FROM_QUALIFIED_SINGLE_DIRECTORY_HELPER",
        published_control_names=sorted(published or {}), first_unrecorded_path=None,
        schema_directories_not_attempted=[], deadline_expired=False,
        read_bytes_exact=True,
        read_byte_semantics="RETURNED_COUNTERS_INFLIGHT_TIMEOUT_BYTES_UNKNOWN_NOT_ASSUMED_ZERO",
        entry_count_semantics="DISTINCT_RETAINED_PATHS_ACROSS_ALL_PASSES_PLUS_AT_MOST_ONE_UNRECORDED_WITNESS")
    def issue(code, path):
        result["errors"].append(dict(code=code, path=path))
    def missing_factory():
        raise ValueError("INVENTORY_NEVER_CREATES_MISSING_RECORD")
    def check_time():
        if (signal.getitimer(signal.ITIMER_REAL)[0] <= 0
                or time.monotonic() >= stop_at):
            raise TimeoutError("CAMPAIGN_INVENTORY_ABSOLUTE_DEADLINE_EXPIRED")
    def overflow_at(relative, name):
        # The caller exits every nested branch before another scan. Never
        # replace or append a second per-directory overflow witness.
        path = relative+"/"+name if relative else name
        if result["first_unrecorded_path"] is None:
            result["first_unrecorded_path"] = path
        issue("CAMPAIGN_METADATA_ENTRY_LIMIT", path)
        raise _CampaignInventoryStop("CAMPAIGN_METADATA_ENTRY_LIMIT")
    def observe(relative, names, expected):
        # Rechecks/helper passes can introduce distinct late names. Charge their
        # UNION with prior paths, never just each pass's list length.
        for name in names:
            path = relative+"/"+name if relative else name
            if path in result["entries"]:
                continue
            if result["observed_entries"] >= result["metadata_entry_limit"]:
                overflow_at(relative, name)
            spec, unknown = files.get(path), name not in expected
            result["observed_entries"] += 1
            if unknown or spec is not None and spec["category"] == "SCIENTIFIC":
                result["scientific_entries"] += 1
            if result["scientific_entries"] > 128:
                issue("SCIENTIFIC_MANIFEST_ENTRY_BUDGET", path)
            result["entries"][path] = dict(status="NAME_OBSERVED_NOT_YET_VALIDATED",
                category="UNKNOWN" if unknown else spec["category"] if spec is not None else "DIRECTORY")
            if unknown:
                issue("UNDECLARED_CAMPAIGN_NAME", path)
    try:
        initial_remaining = signal.getitimer(signal.ITIMER_REAL)[0]
        stop_at = time.monotonic()+initial_remaining
        check_time()
        files, directories = _campaign_schema(bundle, artifact_evidence, control_bindings, published or {})
        order = sorted(directories.items(), key=lambda row: (row[0].count("/"), row[0]))
        result["schema_directories_not_attempted"] = [name for name, _ in order]
        # These are derived metadata/read ceilings, not larger scientific limits.
        slots = len(files)+len(directories)-1
        result.update(schema_entry_slots=slots, metadata_entry_limit=slots+128,
            schema_directory_limit=len(directories), scientific_entry_limit=128,
            schema_file_byte_sum=sum(row["max_bytes"] for row in files.values()))
        result["schema_read_byte_bound"] = result["schema_file_byte_sum"]+32768+sum(
            row["max_bytes"] for row in files.values() if "leaf" in row)
        if result["schema_read_byte_bound"] > 8*1024**3:
            raise ValueError("CAMPAIGN_DERIVED_READ_BOUND_EXCEEDS_HELPER_CEILING")
        with artifact_alarm_guard():
            module = artifact_module()
        check_time()
        # Tree root itself is pinned again for each operation; no symlink stat
        # acceptance or recursive discovery can introduce an unbound branch.
        for relative, expected in order:
            check_time()
            fd = None
            local = dict(complete=False, expected_names=sorted(expected), errors=[], name_enumeration_complete=False)
            result["directories"][relative] = local
            result["schema_directories_not_attempted"].remove(relative)
            try:
                fd = _open_tree_directory(relative)
                before = os.fstat(fd)
                local["identity"] = dict(device=before.st_dev, inode=before.st_ino)
                remaining = result["metadata_entry_limit"]-result["observed_entries"]
                cap = min(1024, max(0, remaining))
                names, overflow = _tree_names(fd, cap)
                check_time()
                observe(relative, names, expected)
                local["initial_names"] = names
                if overflow is not None:
                    overflow_at(relative, overflow)
                local["name_enumeration_complete"] = True
                fingerprints = {}
                for name in names:
                    check_time()
                    path = relative+"/"+name if relative else name
                    unknown = name not in expected
                    row = result["entries"][path]
                    try:
                        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                        row["identity"] = _tree_fingerprint(info)
                        fingerprints[name] = row["identity"]
                        if unknown:
                            row["status"] = "UNDECLARED_METADATA_ONLY_NO_READ"
                        elif expected[name] == "directory":
                            row["status"] = "DECLARED_DIRECTORY" if stat.S_ISDIR(info.st_mode) else "INVALID_DIRECTORY_TYPE"
                            if row["status"] != "DECLARED_DIRECTORY":
                                issue(row["status"], path)
                        elif not stat.S_ISREG(info.st_mode):
                            row["status"] = "NONREGULAR_OR_SYMLINK_NO_READ"
                            issue(row["status"], path)
                    except TimeoutError:
                        raise
                    except Exception as error:
                        row.update(status="ENTRY_STAT_FAILED", error_type=type(error).__name__)
                        issue(row["status"], path)
                for name, kind in expected.items():
                    path = relative+"/"+name if relative else name
                    if name not in names and (kind == "directory" or files[path]["required"]):
                        issue("REQUIRED_CAMPAIGN_NAME_MISSING", path)
                if not any(kind == "directory" for kind in expected.values()):
                    admitted = {name: {key: files[(relative+"/" if relative else "")+name][key]
                        for key in ("max_bytes", "sha256", "bytes")}
                        for name in names if name in expected}
                    check_time()
                    with artifact_alarm_guard():
                        bound = module.inventory_directory_v4(RUN/relative, admitted, max_entries=cap,
                            max_total_bytes=result["schema_read_byte_bound"]-result["read_bytes"])
                    result["read_bytes"] += bound["read_bytes"]
                    check_time()
                    observe(relative, bound.get("initial_names", []), expected)
                    observe(relative, bound.get("final_names", []), expected)
                    if bound.get("first_unrecorded_name") is not None:
                        overflow_at(relative, bound["first_unrecorded_name"])
                    if bound.get("complete") is not True or bound.get("initial_names") != names:
                        issue("QUALIFIED_LEAF_INVENTORY_UNAVAILABLE", relative)
                    local["helper_complete"] = bound.get("complete") is True
                    local["helper_errors"] = bound.get("errors", [])
                    for name, observed in bound.get("entries", {}).items():
                        path = relative+"/"+name if relative else name
                        if path in result["entries"]:
                            result["entries"][path].update(status=observed.get("status"),
                                sha256=observed.get("sha256"), bytes=observed.get("bytes"))
                else:
                    # The public helper inventories files only. This new mixed
                    # branch adapter validates existing files individually, and
                    # supplies the complete branch name/identity checks itself.
                    for name in names:
                        check_time()
                        path = relative+"/"+name if relative else name
                        if name not in expected or expected[name] != "file":
                            continue
                        spec = files[path]
                        row = result["entries"][path]
                        size = row.get("identity", {}).get("bytes")
                        if (type(size) is not int or not 0 <= size <= spec["max_bytes"]
                                or size > result["schema_read_byte_bound"]-result["read_bytes"]
                                or spec["bytes"] is not None and size != spec["bytes"]):
                            issue("MIXED_BRANCH_FILE_SIZE_ADMISSION", path)
                            continue
                        try:
                            def validate(blob, spec=spec):
                                return (len(blob) <= spec["max_bytes"]
                                    and (spec["bytes"] is None or len(blob) == spec["bytes"])
                                    and (spec["sha256"] is None or digest(blob) == spec["sha256"]))
                            with artifact_directory_walk(), artifact_alarm_guard():
                                observed = module.ensure_terminal_record_v4(RUN/relative, name,
                                    missing_factory, validate, max_bytes=spec["max_bytes"])
                            result["read_bytes"] += observed["read_bytes"]
                            check_time()
                            row.update(status="REGULAR_FILE_BYTES_BOUND", sha256=observed["sha256"], bytes=observed["bytes"])
                        except TimeoutError:
                            raise
                        except Exception as error:
                            result["read_bytes"] += getattr(error, "evidence", {}).get("read_bytes", 0)
                            row.update(status="MIXED_BRANCH_FILE_UNAVAILABLE", error_type=type(error).__name__)
                            issue(row["status"], path)
                check_time()
                final_names, final_overflow = _tree_names(fd, cap)
                check_time()
                observe(relative, final_names, expected)
                local["final_names"] = final_names
                if final_overflow is not None:
                    overflow_at(relative, final_overflow)
                if final_names != names or final_overflow is not None:
                    issue("CAMPAIGN_DIRECTORY_NAMES_CHANGED", relative)
                for name, identity in fingerprints.items():
                    check_time()
                    if _tree_fingerprint(os.stat(name, dir_fd=fd, follow_symlinks=False)) != identity:
                        issue("CAMPAIGN_ENTRY_CHANGED_DURING_BRANCH", (relative+"/" if relative else "")+name)
                if _tree_fingerprint(os.fstat(fd)) != _tree_fingerprint(before):
                    issue("CAMPAIGN_DIRECTORY_CHANGED_DURING_BRANCH", relative)
                following = _open_tree_directory(relative)
                try:
                    after = os.fstat(following)
                    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                        issue("CAMPAIGN_DIRECTORY_PATH_REBOUND", relative)
                finally:
                    os.close(following)
                local["complete"] = (overflow is None and final_overflow is None and names == final_names
                    and not any(relative == "" or error["path"] == relative or error["path"].startswith(relative+"/")
                                for error in result["errors"]))
            except (TimeoutError, _CampaignInventoryStop):
                raise
            except Exception as error:
                local["error_type"] = type(error).__name__
                issue("CAMPAIGN_BRANCH_UNAVAILABLE", relative)
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception as error:
                        local["close_error_type"] = type(error).__name__
                        issue("CAMPAIGN_BRANCH_CLOSE_FAILURE", relative)
            if len(encoded(result)) > 256*1024:
                raise ValueError("CAMPAIGN_CUSTODY_ENVELOPE_BYTE_BUDGET")
        check_time()
        _verify_tree_leaf_bindings(files, result, control_bindings)
        if result["read_bytes"] > result["schema_read_byte_bound"]:
            raise ValueError("CAMPAIGN_DERIVED_READ_BOUND")
        # A branch scan alone misses a later change while another branch is
        # read. Recheck every observed entry at the end via fresh pinned parents.
        # This is bounded change detection, not an atomic adversarial snapshot.
        for path, row in result["entries"].items():
            check_time()
            if "identity" not in row:
                continue
            parts = Path(path).parts
            fd = _open_tree_directory("/".join(parts[:-1]))
            try:
                current = _tree_fingerprint(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False))
                if current != row["identity"]:
                    issue("CAMPAIGN_ENTRY_CHANGED_BEFORE_TREE_END", path)
            finally:
                os.close(fd)
        if previous is not None:
            for path, row in previous["entries"].items():
                if result["entries"].get(path) != row:
                    issue("PRIOR_INVENTORY_ENTRY_CHANGED_AFTER_PUBLICATION", path)
            for path, row in previous["directories"].items():
                if result["directories"].get(path, {}).get("identity") != row.get("identity"):
                    issue("PRIOR_DIRECTORY_REBOUND_AFTER_PUBLICATION", path)
        check_time()
        result.update(phase="ALL_SCHEMA_BRANCHES_VISITED", complete=not result["errors"])
    except TimeoutError:
        result.update(complete=False, phase="INTERRUPTED_BY_ABSOLUTE_DEADLINE", deadline_expired=True,
            read_bytes_exact=False,
            failure_type="TimeoutError", failure_reason="CAMPAIGN_INVENTORY_ABSOLUTE_DEADLINE_EXPIRED")
    except _CampaignInventoryStop as error:
        result.update(complete=False, phase="STOPPED_GLOBAL_METADATA_BUDGET",
            failure_type=type(error).__name__, failure_reason=str(error))
    except Exception as error:
        result.update(complete=False, failure_type=type(error).__name__,
            failure_reason=str(error)[:128] if isinstance(error, ValueError) else type(error).__name__)
    if len(encoded(result)) > 256*1024:
        # Do not publish a successful prefix or silently drop an unbounded ledger.
        return dict(schema=result["schema"], complete=False, phase="REPORT_BUDGET",
            reason="CAMPAIGN_CUSTODY_ENVELOPE_BYTE_BUDGET", detailed_inventory_discarded_whole=True,
            observed_entries=result["observed_entries"], scientific_entries=result["scientific_entries"],
            read_bytes=result["read_bytes"], read_bytes_exact=result["read_bytes_exact"],
            deadline_expired=result["deadline_expired"], first_unrecorded_path=result["first_unrecorded_path"],
            unattempted_schema_directory_count=len(result["schema_directories_not_attempted"]))
    return result


def final_source_audit(bundle, evidence, *, control_bindings):
    """Named source-only checks on success AND failure; partial checks stay so."""
    evidence.update(current={}, frozen={}, errors={}, complete=False, unchanged=False)
    for label, base in (("current", ROOT), ("frozen", RUN/"source")):
        for name in SOURCES:
            try:
                evidence[label][name] = digest(_bounded_read(base/name, 2*1024**2))
            except TimeoutError:
                raise
            except Exception as error:
                evidence["errors"][label+":"+name] = type(error).__name__
    evidence["runtime_metadata_unchanged"] = runtime_metadata() == bundle["runtime_metadata"]
    evidence["controls"] = {}
    for name, sha in control_bindings.items():
        try:
            evidence["controls"][name] = digest(owned_read(name, 256*1024, sha))
        except TimeoutError:
            raise
        except Exception as error:
            evidence["errors"]["control:"+name] = type(error).__name__
    evidence["complete"] = True
    evidence["unchanged"] = (not evidence["errors"] and evidence["runtime_metadata_unchanged"]
        and evidence["current"] == evidence["frozen"] == bundle["source_sha256"]
        and evidence["controls"] == control_bindings)


def execute():
    # The unchanged20s source preflight precedes the4200s campaign. No
    # automatic coordinator-cache cleanup may run after an expired alarm.
    started_attempted = False
    try:
        with deadline(LIMITS["parent_read_seconds"]):
            bundle = verify_bundle()
            audit = verify_pre_run_audit(bundle)
            control_bindings = {}
            for name, value in (("FROZEN.json", bundle), ("PRE_RUN_AUDIT.json", audit)):
                blob = owned_read(name, 256*1024)
                if decode(blob) != value:
                    raise ValueError("PREFLIGHT_CONTROL_CHANGED_DURING_BINDING")
                control_bindings[name] = digest(blob)
            for name in ("STARTED.json", "COMPLETED.json", "FINAL_GATE.json", "FINALIZATION_FAILED.json",
                         "COORDINATOR_DISPATCH.json", "COORDINATOR_COMPLETED.json", "COORDINATOR_VERIFIED.json",
                         "SCORING_UNAVAILABLE.json", "INFERENCE_TERMINAL.json", "SCORED.json",
                         "SCORER_RECEIPT.json", "SCORER_RESULT.json", "SELECTED_REFERENCES.json",
                         "COORDINATOR_OUTPUT.txt", "cases", "leaf_dispatch", "coordinator_work"):
                if marker_present(RUN/name):
                    raise ValueError("STARTED_OR_TERMINAL_ATTEMPT_NO_RETRY")
            start = dict(bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
                status="STARTED_NO_RETRY", limits=LIMITS,
                timing_scope="SOURCE_PREFLIGHT_20S_THEN_CAMPAIGN4200S_INCLUDING_TERMINAL_RESERVE30S")
            control_bindings["STARTED.json"] = digest(encoded(start)+b"\n")
            started_attempted = True
            exclusive_json(RUN/"STARTED.json", start)
    except BaseException as error:
        if (not started_attempted or isinstance(error, FileExistsError)
                or getattr(error, "code", None) == "TARGET_ALREADY_EXISTS_NO_WRITE"):
            raise
        try:
            with deadline(1):
                exclusive_json(RUN/"FINALIZATION_FAILED.json", dict(status="TERMINAL_VETO_NO_RELEASE",
                    phase="STARTED_PUBLICATION", error_type=type(error).__name__, worker_dispatched=False))
        except BaseException:
            pass
        return 1
    started = time.monotonic()
    receipt, terminal, source_evidence, artifact_evidence, failures = {}, [], {}, {}, []
    tree_inventory = None
    writers_stopped = False
    status = "TERMINAL_GLOBAL_BUDGET_OR_COORDINATOR_FAILURE"
    def remaining():
        return max(.001, LIMITS["total_wall_seconds"]-(time.monotonic()-started))
    def failed(phase, error):
        failures.append(dict(phase=phase, error_type=type(error).__name__,
            reason=str(error)[:160] if isinstance(error, (ValueError, TimeoutError)) else type(error).__name__))
    try:
        with deadline(max(.001, remaining()-LIMITS["terminalization_reserve_seconds"])):
            token = secrets.token_hex(32)
            dispatch = dict(parent_pid=os.getpid(), token_sha256=digest(token.encode()))
            control_bindings["COORDINATOR_DISPATCH.json"] = digest(encoded(dispatch)+b"\n")
            exclusive_json(RUN/"COORDINATOR_DISPATCH.json", dispatch)
            cache = RUN/"coordinator_work"
            cache.mkdir(mode=0o700, exist_ok=False)
            # This owned, raw-input-free working directory remains retained.
            # Per-case scan/CAD TemporaryDirectory cleanup is unchanged.
            receipt = supervise_process(runner_command("--internal-coordinator"),
                dict(token=token, parent_pid=os.getpid()), cache=cache.resolve(),
                log_path=RUN/"COORDINATOR_OUTPUT.txt",
                wall_seconds=max(.001, remaining()-LIMITS["terminalization_reserve_seconds"]),
                rss_limit=LIMITS["global_tree_rss_bytes"], log_limit=LIMITS["coordinator_log_bytes"], new_session=True)
    except BaseException as error:
        failed("COORDINATOR_SUPERVISION", error)
    try:
        with deadline(min(LIMITS["parent_read_seconds"], remaining())):
            if "COORDINATOR_DISPATCH.json" in control_bindings:
                context = dict(schema=LEAF_SCHEMA, coordinator_pid=receipt.get("child_pid"), outer_pid=os.getpid(),
                    bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
                    started_sha256=digest(encoded(start)), coordinator_dispatch_sha256=digest(encoded(dispatch)),
                    source_sha256=bundle["source_sha256"], runtime_metadata=bundle["runtime_metadata"])
                control_bindings["COORDINATOR_VERIFIED.json"] = digest(encoded(context)+b"\n")
            final_source_audit(bundle, source_evidence, control_bindings=control_bindings)
            if not source_evidence["unchanged"]:
                raise ValueError("FINAL_SOURCE_SNAPSHOT_OR_CONTROL_DRIFT")
    except BaseException as error:
        failed("FINAL_SOURCE_AUDIT", error)
    try:
        with deadline(remaining()):
            # Do not terminal-fill or inventory while an unconfirmed group may
            # still write. A killed but reaped/empty group may receive fallback
            # terminals; it still cannot become a successful comparison.
            writers_stopped = (receipt.get("direct_child_reaped") is True
                and receipt.get("owned_group_empty_confirmed") is True
                and receipt.get("observed_foreign_group_escape_detected") is not True
                and receipt.get("observed_foreign_group_escape_unconfirmed_after_cleanup") is not True
                and receipt.get("cleanup_errors") == [] and receipt.get("control_close_errors") == [])
            if not writers_stopped:
                terminal = [dict(case_id=case["case_token"], arm=arm, durable=False,
                    reason="WRITER_QUIESCENCE_UNCONFIRMED_NO_TERMINAL_FILL")
                    for case in bundle["cases"] for arm in ARMS]
                raise ValueError("WRITER_QUIESCENCE_REQUIRED_FOR_TERMINAL_FILL")
            terminal = terminalize_case_records(bundle, "GLOBAL_ABORT_OR_MISSING_TERMINAL_RECORD")
            if not (len(terminal) == 22 and all(row["durable"] for row in terminal)):
                raise ValueError("TERMINAL_STORAGE_FAILURE")
            if receipt.get("exit_code") != 0 or receipt.get("budget_stop") is not None:
                raise ValueError("COORDINATOR_EXIT_RESOURCE_OR_GROUP_CUSTODY_FAILURE")
            if marker_present(RUN/"SCORING_UNAVAILABLE.json") or marker_present(RUN/"FINALIZATION_FAILED.json"):
                raise ValueError("TERMINAL_VETO_MARKER_PRESENT")
            verify_parent_artifacts(bundle, receipt, artifact_evidence)
            if not failures:
                status = "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"
    except BaseException as error:
        failed("TERMINAL_EVIDENCE", error)
    try:
        with deadline(remaining()):
            if not writers_stopped:
                raise ValueError("WRITER_QUIESCENCE_REQUIRED_FOR_CAMPAIGN_INVENTORY")
            # Inventory even a semantic failure, retaining all discoverable
            # metadata. It never converts such a failure into usable scoring.
            tree_inventory = verify_campaign_inventory(bundle, artifact_evidence, control_bindings)
            artifact_evidence["campaign_tree_inventory"] = tree_inventory
            if tree_inventory.get("complete") is not True:
                raise ValueError("COMPLETE_CAMPAIGN_TREE_CUSTODY_REQUIRED")
            if len(encoded(artifact_evidence)) > 256*1024:
                raise ValueError("OWNED_ARTIFACT_MANIFEST_BYTE_BUDGET")
    except BaseException as error:
        artifact_evidence["complete"] = False
        failed("COMPLETE_CAMPAIGN_INVENTORY", error)
    try:
        with deadline(remaining()):
            elapsed = time.monotonic()-started
            parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if elapsed > LIMITS["total_wall_seconds"] or parent_peak > LIMITS["global_tree_rss_bytes"]:
                failures.append(dict(phase="FINAL_RESOURCE_GATE", reason="WALL_OR_PARENT_RSS_LIMIT"))
            if failures:
                status = "TERMINAL_EVIDENCE_OR_RESOURCE_FAILURE"
            completed = dict(status=status, coordinator_receipt=receipt, failures=failures,
                terminal_case_records=terminal, final_source_audit=source_evidence,
                owned_artifact_custody=artifact_evidence,
                parent_wall_seconds=elapsed, parent_peak_rss_bytes=parent_peak,
                retained_coordinator_work_directory="coordinator_work",
                release_requires="OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT",
                no_resume=True, production_use_allowed=False)
            if len(encoded(completed))+1 > 256*1024:
                raise ValueError("PARENT_TERMINAL_CONTROL_BYTE_BUDGET")
            exclusive_json(RUN/"COMPLETED.json", completed)
            if failures or time.monotonic()-started > LIMITS["total_wall_seconds"]:
                raise ValueError("TERMINAL_FAILURE_OR_POST_PUBLICATION_WALL_LIMIT")
            published = {"COMPLETED.json": dict(sha256=digest(encoded(completed)+b"\n"),
                                                bytes=len(encoded(completed))+1)}
            completed_inventory = verify_campaign_inventory(bundle, artifact_evidence, control_bindings,
                published=published, previous=tree_inventory)
            if completed_inventory.get("complete") is not True:
                raise ValueError("POST_COMPLETED_CAMPAIGN_INVENTORY_FAILED")
            final_gate = dict(status="PENDING_INDEPENDENT_POST_AUDIT",
                completed_canonical_sha256=digest(encoded(completed)),
                source_and_control_audit_sha256=digest(encoded(source_evidence)),
                owned_artifact_custody_sha256=digest(encoded(artifact_evidence)),
                scored_report_sha256=artifact_evidence["scored_report_sha256"],
                scorer_result_sha256=artifact_evidence["scorer_result_sha256"],
                reference_snapshot_sha256=SELECTED_SHA, parent_wall_seconds=time.monotonic()-started,
                completed_publication_inventory_sha256=digest(encoded(completed_inventory)),
                inventory_boundary="AFTER_COMPLETED_BEFORE_FINAL_GATE_WITH_EXACT_COMPLETED_BYTES",
                final_gate_self_hash_claim=False,
                final_release_requires="POST_FINAL_GATE_FULL_TREE_RESCAN_OBSERVED_EXIT0_AND_INDEPENDENT_POST",
                measurement_boundary="BEFORE_FINAL_GATE_FSYNC_RECHECKED_BEFORE_PARENT_EXIT")
            exclusive_json(RUN/"FINAL_GATE.json", final_gate)
            published["FINAL_GATE.json"] = dict(sha256=digest(encoded(final_gate)+b"\n"),
                                                 bytes=len(encoded(final_gate))+1)
            final_inventory = verify_campaign_inventory(bundle, artifact_evidence, control_bindings,
                published=published, previous=completed_inventory)
            if final_inventory.get("complete") is not True:
                raise ValueError("POST_FINAL_GATE_CAMPAIGN_INVENTORY_FAILED")
            print(json.dumps(dict(status=status, attempt=str(RUN), release_allowed=False,
                durable_terminal_case_records=sum(row["durable"] for row in terminal),
                final_campaign_inventory_sha256=digest(encoded(final_inventory)),
                final_campaign_inventory_entries=final_inventory["observed_entries"],
                final_scientific_manifest_entries=final_inventory["scientific_entries"],
                final_inventory_stdout_is_not_durable_runner_artifact=True)), flush=True)
        if (time.monotonic()-started > LIMITS["total_wall_seconds"]
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMITS["global_tree_rss_bytes"]):
            raise ValueError("POST_FINAL_GATE_WALL_OR_PARENT_RSS_LIMIT")
        return 0
    except BaseException as error:
        try:
            with deadline(1):
                exclusive_json(RUN/"FINALIZATION_FAILED.json", dict(status="TERMINAL_VETO_NO_RELEASE",
                    error_type=type(error).__name__, failures=failures, parent_wall_seconds=time.monotonic()-started))
        except BaseException:
            pass
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--score-worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--internal-coordinator", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        worker()
    elif args.score_worker:
        scorer_worker()
    elif args.internal_coordinator:
        internal_coordinator()
    else:
        raise SystemExit(execute())


if __name__ == "__main__":
    main()
