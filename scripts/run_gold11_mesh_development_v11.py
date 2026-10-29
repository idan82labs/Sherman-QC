"""V11 full11 successor: V10 runner text on the V18 engine = the V10 engine with ONE changed mechanism, measurement
kernel V3 (two-mode band-membership layer separation of the flange crops with published bridging rows, reached
through native feature bridge V2, context measurement V4 and support cover V5, scored by contact V5 / schedule
projection V6). Registration, allowance, admission gates, δ/e measurement policy and every cap are the V10 values.

Original V10 header: V9 runner text on the V17/V16/V5 engine (native feature bridge V1, refinement
allowance V1 with deterministic lane quotas and the declared CPU guard, exact bound query reuse, bounded
stage checkpoints and cost receipts, the explicit evaluated-subset registration contract consumed by
context V3, support cover V4, contact scoring V4 and schedule projection V5). Both fresh arms use the
same V17 engine; the arm comparison is scan-plane lane/schedule only. No cap is raised.

Original V9 header: V8 runner text on the V16/V15 engine (completion lane V1, measurement
kernel V2 with declared layer selection and diagnostics, derived measurement policy).

Original V8 header: V7 runner text on the V15/V14/context-V4/surface-V4 engine.

V7 and every earlier attempt remain terminal and unchanged. Both fresh arms use
mesh offline V15 (initializer V14, context V4 on the kd-partition raw extrema
V3, finite surface V4 with registration V4, stage policies V2), schedule
projection V3 and the unchanged contact scorer with raw17/reference71 and
all22 terminal scope. Declared engine-configuration changes relative to V7:
registration max_inlier_plane_p95_mm 0.5 -> 1.0 (pose admission separated from
shape conformance; a documented provisional development assumption), surface
max_ties_per_query 32 -> 256, context extrema coordinate cap 256M -> 512M for
the new construction model, and a NEW deterministic measurement-context
schedule (pose-class dedup, evidence-ranked cap of at most 8 contexts bounded
by the frozen consumer's preflight budgets). Runner custody/supervision/scoring
transport is the V7 text. Physical execution still requires the exact frozen
protocol and PRE.
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
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUN = ROOT / "docs/research/gold11_mesh_development_v11_attempt"
PROTOCOL = "docs/research/2026-09-14_gold11_mesh_development_v11_protocol.md"
REVIEW = "scripts/review_gold11_inputs_v1.py"
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
CATALOG_SHA = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_SHA = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
SOURCE_BASIS = "scripts/run_gold11_mesh_development_v10.py"
SOURCE_BASIS_SHA = "f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92"
SOURCE_PROFILE = dict(implementation="GOLD11_MESH_DEVELOPMENT_V11",
    unchanged_source_basis=SOURCE_BASIS, unchanged_source_basis_sha256=SOURCE_BASIS_SHA,
    source_delta=["VERSIONED_IDENTITY_AND_RELEASE_DECLARATIONS",
        "ENGINE_V18_MEASUREMENT_KERNEL_V3_TWO_MODE_BAND_MEMBERSHIP_LAYER_SEPARATION_BRIDGE_V2_CONTEXT_V4_COVER_V5",
        "SCHEDULE_PROJECTION_V6_CONTACT_V5_V18_IDENTITY_BINDING_THIRTEEN_FIELD_LAYER_POLICY_TWO_NEW_UNRESOLVED_STATUSES",
        "CONFIG_LAYER_POLICY_FIVE_NEW_DECLARED_FIELDS_ALL_OTHER_VALUES_V10"],
    scientific_config_or_scorer_change=True,
    scorer_change=True,
    scorer_change_scope="CONTACT_V5_IS_THE_V4_TEXT_RECOGNIZING_COVER_V5_CONTEXT_V4_THE_THIRTEEN_FIELD_LAYER_POLICY_AND_TWO_"
                        "NEW_UNRESOLVED_STATUSES_NEITHER_INNER_FACE_ELIGIBLE_MATCHER_GATES_AND_CARRIER_UNCHANGED",
    retained_scorer_request_transport="GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6")
SOURCES = (
    SOURCE_BASIS,
    "scripts/run_gold11_mesh_development_v11.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/gold11_owned_process_supervision_v3.py",
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
    "scripts/run_gold11_mesh_development_v2.py", "tests/test_gold11_mesh_development_runner_v5.py",
    "scripts/run_gold11_mesh_development_v4.py",
    "scripts/gold11_contact_scoring_v2.py", "scripts/gold11_schedule_projection_v1.py",
    "backend/cad_guided_composite_surface_roles_v1.py",
    "backend/cad_guided_context_measurement_v1.py", "backend/cad_guided_context_surface_roles_v1.py",
    "backend/cad_guided_exact_support_cover_v1.py", "backend/cad_guided_mesh_offline_v12.py",
    "backend/cad_guided_mesh_offline_v13.py", "backend/cad_guided_plane_initialization_v12.py",
    "backend/cad_guided_scan_plane_preparation_v1.py", "backend/cad_guided_seeded_plane_proposals_v1.py",
    "backend/cad_guided_stage_policies_v1.py", "backend/cad_guided_support_cover_measurement_v2.py",
    "backend/cad_guided_two_plane_schedule_v2.py", "tests/__init__.py",
    "backend/cad_guided_two_plane_context_v2.py",
    "scripts/run_gold11_mesh_development_v5.py",
    "scripts/run_gold11_mesh_development_v6.py",
    "backend/cad_guided_raw_extrema_index_v3.py", "backend/cad_guided_two_plane_context_v4.py",
    "backend/cad_guided_triangle_surface_v4.py", "backend/cad_guided_global_registration_v4.py",
    "backend/cad_guided_stage_policies_v2.py", "backend/cad_guided_plane_initialization_v14.py",
    "backend/cad_guided_mesh_offline_v15.py",
    "scripts/gold11_schedule_projection_v3.py",
    "scripts/gold11_engine_stage_sampler_v1.py",
    "docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md",
    # V9 additions: the V8 runner is SOURCE_BASIS; V7 runner retained as a preservation oracle.
    "scripts/run_gold11_mesh_development_v7.py",
    "backend/cad_guided_measurement_v2.py", "backend/cad_guided_context_measurement_v2.py",
    "backend/cad_guided_support_cover_measurement_v3.py",
    "backend/cad_guided_plane_initialization_v15.py", "backend/cad_guided_mesh_offline_v16.py",
    "backend/cad_guided_report_envelope_v7.py",
    "scripts/gold11_schedule_projection_v4.py", "scripts/gold11_contact_scoring_v3.py",
    "docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md",
    "scripts/gold11_v9_execute_waiter.py",
    # V10 additions: the V9 runner is SOURCE_BASIS; the V8 runner is retained as a preservation oracle.
    "scripts/run_gold11_mesh_development_v8.py",
    "backend/cad_guided_native_feature_bridge_v1.py", "backend/cad_guided_global_registration_v5.py",
    "backend/cad_guided_plane_initialization_v16.py", "backend/cad_guided_context_measurement_v3.py",
    "backend/cad_guided_support_cover_measurement_v4.py", "backend/cad_guided_mesh_offline_v17.py",
    "scripts/gold11_contact_scoring_v4.py", "scripts/gold11_schedule_projection_v5.py",
    "docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md",
    "scripts/gold11_v10_execute_waiter.py",
    # V11 additions: the V10 runner is SOURCE_BASIS; the V9 runner and V10 protocol are retained as preservation oracles.
    "scripts/run_gold11_mesh_development_v9.py", "docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md",
    "backend/cad_guided_measurement_v3.py", "backend/cad_guided_native_feature_bridge_v2.py",
    "backend/cad_guided_context_measurement_v4.py", "backend/cad_guided_support_cover_measurement_v5.py",
    "backend/cad_guided_mesh_offline_v18.py",
    "scripts/gold11_contact_scoring_v5.py", "scripts/gold11_schedule_projection_v6.py",
    "docs/research/2026-09-14_gold11_mesh_development_v11_delta_design.md",
    "scripts/gold11_v11_execute_waiter.py",
    REVIEW, PROTOCOL,
)
# Final candidate declaration, not independent physical execution authority.
# Exact source qualification, frozen bundle and external PRE remain mandatory.
RELEASE_AUTHORIZED = True
TIMING_RELEASE_STATUS = "REVIEWED_V11_EXACT_TIMING_RELEASE"
STAGE_TELEMETRY_STATUS = "UNCHANGED_SAMPLER_WIRING_V8_RUNNER_PATH_NAMED"
STAGE_SAMPLER_SOURCE = "scripts/gold11_engine_stage_sampler_v1.py"
STAGE_SAMPLER_SHA = "80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0"
_STAGE_SAMPLER_MODULE = None
STAGE_SAMPLE_SOURCE_PATHS = tuple(name for name in SOURCES
    if name.startswith("backend/") and name.endswith(".py")) + ("scripts/run_gold11_mesh_development_v11.py",)
STAGE_LOG_DECLARATION = dict(sample_stream_bytes=524288, receipt_bytes=4096,
    checkpoint_stdout_bytes=65536, checkpoint_cap_notice_bytes=128,
    other_output_remaining_bytes=454528, whole_leaf_log_bytes=1048576,
    enforcement="UNCHANGED_WHOLE_LOG_CAP_NOT_SEPARATE_NATIVE_OUTPUT_QUOTA",
    checkpoint_scope="V10_BOUNDED_STAGE_CHECKPOINT_STDOUT_LINES_EMITTED_EVIDENCE_OF_A_KILLED_LEAF_NEVER_A_RESULT",
    overflow="TERMINAL_LOG_BUDGET_FAILURE_NO_CAP_RAISE_OR_RETRY")
# Preserved initial draft checklist, not the current release/custody decision.
# Accepted external audits and require_release/verify_pre_run_audit govern it.
DRAFT_RELEASE_BLOCKERS = (
    "INDEPENDENT_REVIEW_OF_EVERY_V11_MODULE_REQUIRED_BEFORE_FREEZE",
    "FOCUSED_WORKFLOW_AND_RUNNER_SOURCE_TESTS_REQUIRED_BEFORE_FREEZE",
    "EXACT_INPUT_REFERENCE_AUTHORITY_PRE_AND_ARCHIVE_REQUIRED_BEFORE_EXECUTE",
    "SCORING_WRAPPER_AND_REQUEST_CAPACITY_MUST_BE_FROZEN_TOGETHER",
)
POSE_ADMISSION_DECLARATION = dict(
    max_inlier_plane_p95_mm=1.,
    status="PROVISIONAL_DEVELOPMENT_ADMISSION_GATE_NOT_CALIBRATED_CONFORMANCE",
    scope="BOTH_ARMS_CONTEXT_SUPPORT_P95_UPPER_OVER_2MM_PROXIMITY_ROWS",
    false_pose_risk="ADMITS_CONTEXTS_WITH_P95_UPPER_IN_0P5_TO_1P0_MM_NO_CROSS_CONTEXT_AMBIGUITY_VETO",
    expected_new_outcome_class="MORE_ADMITTED_CONTEXTS_CAN_REFUSE_AT_FIXED_MEASUREMENT_PREFLIGHT_BUDGETS")

ARMS = ["control", "challenger"]
ARM_DEFINITIONS = dict(
    control=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V13", implementation="CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1",
        initialization_implementation="CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1",
        registration_implementation="SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1",
        two_plane_context_mode="v4", native_adapter_version="v1",
        registration="V16_ALLOWANCE_V1_COMPLETION_V1_CONTEXT_V4_SURFACE_V4_REGISTRATION_V5_REUSE_V1_CONNECTED_CONTEXTS_V7_SCAN_PLANES_SCHEDULE_V1",
        scan_plane_mode="v7", two_plane_schedule_mode="v1",
        measurement="EXACT_SUPPORT_COVER_V5_CONTEXT_V4_SUBSET_CONTRACT_NATIVE_BRIDGE_V2_KERNEL_V3_TWO_MODE_LAYER_SEPARATION_DIAGNOSTICS_FREE_CIRCLE_RAW_SCAN_CONTACT_CARRIER",
        inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"),
    challenger=dict(workflow="CAD_GUIDED_MESH_OFFLINE_V13", implementation="CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1",
        initialization_implementation="CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1",
        registration_implementation="SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1",
        two_plane_context_mode="v4", native_adapter_version="v1",
        registration="V16_ALLOWANCE_V1_COMPLETION_V1_CONTEXT_V4_SURFACE_V4_REGISTRATION_V5_REUSE_V1_CONNECTED_CONTEXTS_COMPOSITE_SCAN_PLANES_SCHEDULE_V2",
        scan_plane_mode="composite_v1", two_plane_schedule_mode="v2",
        measurement="EXACT_SUPPORT_COVER_V5_CONTEXT_V4_SUBSET_CONTRACT_NATIVE_BRIDGE_V2_KERNEL_V3_TWO_MODE_LAYER_SEPARATION_DIAGNOSTICS_FREE_CIRCLE_RAW_SCAN_CONTACT_CARRIER",
        inference="FRESH_ONCE_FOR_EACH_OF_THE_FIXED_ELEVEN_CASES"),
)
COMPARATOR = "FRESH_V18_KERNEL_V3_ALLOWANCE_V1_BRIDGE_V2_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1"
# Declared engine-configuration deltas relative to the terminal V10 bundle. They
# are new candidate declarations with recorded reasons, not retroactive edits.
# The V9->V10, V8->V9 and V7->V8 deltas are inherited unchanged and restated for continuity.
CONFIG_DELTA_FROM_V10 = dict(
    measurement_layer_separation=dict(
        v10="QUALIFYING_GAP_PARTITION_ON_MATERIAL_SIDE_COORDINATE_KEEP_SMALLER_COMPATIBLE_PAIR_V1",
        v11="TWO_MODE_BAND_MEMBERSHIP_ON_MATERIAL_SIDE_COORDINATE_KEEP_SMALLER_COMPATIBLE_PAIR_V2",
        new_layer_policy_fields=dict(layer_mode_bin_mm=.05, layer_membership_band_mm=.1, max_bridging_row_fraction=.35,
                                     layer_refit_iterations=6, layer_mode_ambiguity_fraction=.25),
        reason="V10: the three slots that reached measurement evaluated 43 physical flange windows and every one was "
               "refused at the first flange plane fit; the retained both-flange diagnostics are CONSISTENT WITH "
               "two-population (two-skin) crops plus between-skin rows continuous in the material-side coordinate "
               "(an inferred mixture hypothesis, not a proved sheet identity), which the V1 empty-gap partition did "
               "not separate: role 0 = 34 fits unresolved as a single population (32 with no qualifying gap, largest "
               "gaps 0.028-0.230 mm; 2 the duplicated CG1E0 singleton-tail gap 0.433 mm failing the 12-row minimum), "
               "7 selected pairs and 2 incompatible pairs; where a pair was selected the kept group still held rows "
               "beyond the residual policy (24 selected fits failing the 0.03 mm maximum residual: 7 role 0, 17 role 1). "
               "Kernel V3 detects two modes on the same coordinate (fixed 0.05 mm bins; EXACT inclusive coordinate-band "
               "row masses around each bin centre; the secondary mode must reach 25 percent of the primary's band mass; "
               "a third separated population with band mass >= 12 rows and >= 25 percent of the smaller skin's is "
               "ambiguous), assigns rows under a published membership rule (nearest refitted skin plane, equal distances "
               "to the primary-mode plane, inclusive 0.1 mm band; rows outside both bands and strictly BETWEEN the "
               "planes are UNASSIGNED rows published as a distinct excluded quantity of unverified cut-edge identity "
               "with a declared maximum fraction; rows outside both bands and NOT between the planes stay in their "
               "nearest group: in the selected population they meet the unchanged gates, in the opposite population "
               "only the pair planarity checks and that surface's conformance stays unknown), requires a membership "
               "fixed point within 6 completed "
               "refits (completed refits and loop entries published separately; no unbudgeted refit), validates the "
               "projection premise, keeps the unchanged pair compatibility and the unchanged smaller-coordinate "
               "choice, and applies the unchanged flange gate forms to the retained skin rows (the band must exceed "
               "the residual policy: joint feature/policy contract enforced at measurement, so the gate is never "
               "satisfied by construction). A measured flange is therefore a skin-membership subset with explicit "
               "excluded-support semantics and coverage NOT_ESTABLISHED: a displaced population of the selected skin "
               "outside the sandwich is refused by the unchanged gate, a deviation of the never-measured opposite skin "
               "faces only the pair planarity checks, a between-skin displacement is an explicit counted exclusion bounded "
               "by the declared fraction, never a conformance claim. No status is inner-face eligible; the arc gates "
               "were never exercised in V10 and V11 makes no claim about them; a layer-only successor targets a first "
               "non-empty output on the already-admitted slots, not a broad F1 result",
        unchanged="REGISTRATION_ALLOWANCE_ADMISSION_GATES_DELTA_E_MEASUREMENT_POLICY_SUPPORT_POLICY_COMPLETION_LANE_AND_"
                  "EVERY_CAP_ARE_THE_V10_VALUES_THE_EIGHT_V1_LAYER_POLICY_FIELDS_KEEP_THEIR_V10_VALUES"),
    inherited_v9_to_v10=dict(
        refinement_allowance=dict(v9=None,
            v10=dict(enabled=True, main_refined_groups=12, completion_refined_groups=4, main_query_rows=6_000_000,
                     completion_query_rows=3_000_000, cpu_budget_seconds=1860., cpu_guard_seconds=1500.,
                     cooperative_chunk_rows=32_768),
            reason="V9: 14 of 22 leaves were killed by the unchanged 1860 s CPU cap inside surface-V4 candidate "
                   "refinement with nothing published (all refinements precede publication in V15/V16). The allowance "
                   "is a declared work allocation INSIDE the unchanged outer caps: deterministic per-lane group and "
                   "row quotas within the frozen 48-group schedule (main strata first, ONE completion allocation for "
                   "the whole completion lane) plus a cooperative CPU guard at 1500 s of the 1860 s absolute process "
                   "CPU (imports and preparation included) declared timing dependent. Expired/omitted candidates are "
                   "UNKNOWN, never rejected; the registration publishes the evaluated subset under an explicit disjoint "
                   "disposition ledger validated by every new consumer; strict whole-set completeness stays FALSE with "
                   "any omission. Headroom 360 s is for publication/measurement/serialization, not a guarantee. "
                   "Corruption and unclassified errors stay fatal. This is not a raised cap and not a speedup claim",
            timing_dependence="CPU_GUARD_STOPS_ARE_TIMING_DEPENDENT_ROW_AND_GROUP_QUOTA_STOPS_ARE_DETERMINISTIC_THE_REALIZED_"
                              "TRACE_IS_PUBLISHED_IN_THE_DISPOSITION_LEDGER_COST_RECEIPTS_AND_STAGE_CHECKPOINTS"),
        native_feature_bridge=dict(v9=None,
            v10="LOSSLESS_KERNEL_V1_TO_KERNEL_V2_FEATURE_RECONSTRUCTION_CAP_64_FAIL_CLOSED_BEFORE_CAD_SAMPLING",
            reason="V9: only ONE of the 22 slots (CG4BDD control) reached measurement, and its support cover was refused "
                   "as a whole BEFORE any window was evaluated (INVALID_SUPPORT_COVER_FEATURES: cover V3 exact-type guard "
                   "versus the kernel-V1 objects both frozen native factories return); the other 21 slots failed earlier (CPU kills inside "
                   "refinement, no passing start, membership/residual refusals), so the V9 measurement side was never "
                   "exercised on physical data. The bridge reconstructs kernel-V2 objects field for field (order, "
                   "provenance and unknown semantics preserved, 64 = the cover/context grammar boundary, not the "
                   "standalone kernel entry cap) and refuses before any costly work; frozen factories untouched"),
        query_reuse=dict(v9=None,
            v10="EXACT_BOUND_REUSE_OF_THE_DIAGNOSIS_FULL_SCAN_QUERY_IMMUTABLE_PAYLOAD_INTEGRITY_FAILURE_FATAL",
            reason="V9 cost postmortem: the diagnosis full-scan query was performed twice per candidate (diagnosis and "
                   "context support) on identical rows, pose, index and policy. V5 hands the same owned immutable answer "
                   "to context support under exact bindings (rows, point bytes, pose bytes, query/index identity, policy "
                   "and implementation snapshot); a legitimate binding change requeries, corrupted retained evidence is "
                   "fatal. Values are unchanged by reuse"),
        completion_budget_declaration=dict(
            v9="max_node_visits_max_triangle_pairs_max_normal_pairs_DECLARED_INACTIVE",
            v10="DECLARED_ACTIVE_FOR_THE_COMPLETION_LANE_FINITE_FACE_QUERIES_VALUES_UNCHANGED",
            reason="V9 cost postmortem: the completion lane's finite face queries allocate and charge node visits, "
                   "triangle pairs, normal pairs and work units against these envelope fields and certificate pairs "
                   "against the certificate caps (the V15 _FaceQueries text); the V9 prose declaring them inactive "
                   "was false. Budget values are byte-identical to V9"),
        worker_stage_checkpoints=dict(v9=None, v10="BOUNDED_STAGE_CHECKPOINT_STDOUT_LINES_65536_BYTES_EMITTED_NOT_DURABLE",
            reason="V9: a killed leaf left no internal-stage evidence; V10 prints bounded engine checkpoints inside the "
                   "unchanged whole-leaf log cap. They are evidence of what a killed leaf reached, never a result"),
    ),
    inherited_v8_to_v9=dict(
        measurement_max_fit_residual_mm=dict(v8=.05, v9=.03),
        measurement_plane_exclusion_mm=dict(v8=.075, v9=.04),
        support_max_angular_gap_deg=dict(v8=5., v9=30.),
        initialization_completion_lane=dict(v8=None, v9="TWO_OF_THREE_TRIPLET_THIRD_FACE_COMPLETION_ALONG_PASSING_PAIR_AXIS_V1"),
        measurement_layer_selection=dict(v8=None, v9="QUALIFYING_GAP_PARTITION_ON_MATERIAL_SIDE_COORDINATE_KEEP_SMALLER_COMPATIBLE_PAIR_V1")),
    inherited_v7_to_v8=dict(
        registration_max_inlier_plane_p95_mm=dict(v7=.5, v8=1.),
        surface_max_ties_per_query=dict(v7=32, v8=256),
        context_max_extrema_coordinate_visits=dict(v7=256_000_000, v8=512_000_000),
        workflow_max_measurement_contexts=dict(v7=None, v8=8)),
)
MEASUREMENT_POLICY_DECLARATION = dict(
    status="SEPARATELY_DERIVED_DEVELOPMENT_POLICY_FOR_NOMINAL_R0P8_90DEG_GATE_FORMS_UNCHANGED",
    derivation="USABLE_ARC_PHI_MINUS_2ACOS(1-E/R)_SAGITTA_MUST_EXCEED_2DELTA_PLUS_GUARD_WITH_E_GREATER_THAN_DELTA",
    frozen_v8_policy_feasible_for_cohort_geometry=False,
    chosen_from_gold=False, tuned_to_case_identities=False,
    noise_sensitivity="SYNTHETIC_SIGMA_0P005_MEASURES_SIGMA_0P010_REFUSES_AT_FLANGE_RESIDUAL_GATE",
    expected_negative_class="WINDOWS_ABOVE_0P03_MM_MAX_FLANGE_RESIDUAL_REFUSE_NOT_PLANAR",
    finite_sampling_dependence="CONTINUOUS_HALF_WIDTH_HEADROOM_ONLY_0P011_TO_0P013_MM_AT_R0P78_TO_0P8_SPARSE_0P3_MM_"
                               "SAMPLING_CAN_REFUSE_FLAT_WITNESS_BY_PHASE_RETAINED_AS_NEGATIVES",
    coupled_effects="DELTA_0P05_TO_0P03_TIGHTENS_PLANE_AND_CIRCLE_MAX_RESIDUAL_BUT_EASES_MINOR_SPREAD_GATE_0P5_TO_0P3_MM_"
                    "AND_FLAT_ALTERNATIVE_REJECTION_E_0P075_TO_0P04_ADMITS_MORE_NEAR_CONTACT_ROWS_"
                    "SUPPORT_GAP_5_TO_30_DEG_ADMITS_SPARSER_ANGULAR_STATIONS",
    uniqueness="EXPLICIT_PROSPECTIVE_DEVELOPMENT_CANDIDATE_NOT_A_UNIQUELY_FORCED_OR_CALIBRATED_CHOICE",
    physical_claim="EXACT_NOMINAL_CLEAN_MODEL_INFEASIBILITY_ONLY_NOT_A_UNIVERSAL_PHYSICAL_IMPOSSIBILITY")
POSITION_ERROR_DECLARATION = dict(
    status="EXPLORATORY_STIPULATED_UNCALIBRATED_PROPOSAL_BOUND",
    scope="BOTH_ARMS_PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY",
    uniform_across_all_cases=True, physical_scanner_bound_established=False,
    selected_from_gold=False, constructed_point005_bound_is_not_physical_evidence=True,
)
LIMITS = dict(case_wall_seconds=3540, case_cpu_seconds=1860,
              sampled_worker_tree_rss_bytes=8*1024**3,
              case_log_bytes=1024**2, case_result_bytes=128*1024**2,
              total_wall_seconds=104400, max_scan_bytes=64*1024**2,
              max_cad_bytes=8*1024**2, scorer_wall_seconds=180,
              scorer_cpu_seconds=120, scorer_rss_bytes=2*1024**3,
              scorer_request_bytes=192*1024**2, scorer_result_bytes=32*1024**2,
              compact_case_bytes=8*1024**2, coordinator_rss_bytes=4*1024**3,
              global_tree_rss_bytes=12*1024**3, parent_read_seconds=180,
              coordinator_log_bytes=2*1024**2, terminalization_reserve_seconds=2400)
# Explicit V3 numerical certificate accounting only; CONFIG and all its existing
# work/geometry quotas remain byte-for-byte the predecessor source expression.
CERTIFICATE_CAPS = dict(initialization_budgets=1_024_000_000,
    surface_budgets=128_000_000, registration_budgets=8_000_000_000)
RETAINED_CONTROL_POLICY = dict(primary_nonzero_control="SEPARATELY_FROZEN_CACHED_UNION",
    cached_control_value="NOT_ASSUMED_OR_LOADED_BY_THIS_RUNNER",
    comparison="SEPARATE_PROVISIONAL_RAW_ENDPOINT_GEOMETRY_LEDGER_ONLY",
    requires="INDEPENDENT_CONTROL_RELEASE_COMMON71_REFERENCE_SHA_UNCHANGED_MATCHER",
    cached_payload_as_modern_measured_carrier=False, historical_replay_claim_allowed=False,
    fresh_v13_gain_is_not_cached_union_superiority=True)
CONFIG = dict(
    workflow=dict(max_registration_scan_rows=50000, cad_sample_count=32768,
                  cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
                  unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
                  max_measurement_contexts=8),
    registration=dict(coarse_points=2048, refine_points=8192, coarse_iterations=30,
                      refine_iterations=30, trim_fraction=.85, huber_delta_mm=.25,
                      coarse_distance_fraction=.15, max_correspondence_distance_mm=2.,
                      max_inlier_plane_p95_mm=1., min_scan_coverage=.75,
                      min_cad_coverage=.25, min_observability_ratio=.005,
                      pose_class_distance_mm=.1, ambiguity_plane_rms_margin_mm=.05,
                      ambiguity_coverage_margin=.03, max_rotation_step_radians=.15,
                      max_translation_step_mm=5.),
    measurement=dict(flange_window_mm=[2.5, 12.5], search_margin_mm=3.,
                     max_fit_residual_mm=.03, plane_exclusion_mm=.04,
                     max_tangency_error_mm=.1, fit_model="free_circle"),
    support=dict(max_axial_gap_mm=4., max_angular_gap_deg=30., min_arc_sweep_deg=20.,
                 min_local_points=3),
    # V10: declared work allocation INSIDE the unchanged outer leaf caps (see CONFIG_DELTA_FROM_V9).
    refinement_allowance=dict(enabled=True, main_refined_groups=12, completion_refined_groups=4,
                              main_query_rows=6_000_000, completion_query_rows=3_000_000,
                              cpu_budget_seconds=1860., cpu_guard_seconds=1500., cooperative_chunk_rows=32_768),
    layer_policy=dict(enabled=True, layer_split_gap_mm=.3, min_layer_rows=12, max_layers_reported=8,
                      max_layer_normal_angle_deg=3., max_layer_gap_mm=6., min_projected_interval_overlap=.5,
                      max_layer_rms_mm=.1,
                      # V11 (kernel V3): declared two-mode band-membership fields; see CONFIG_DELTA_FROM_V10.
                      layer_mode_bin_mm=.05, layer_membership_band_mm=.1, max_bridging_row_fraction=.35,
                      layer_refit_iterations=6, layer_mode_ambiguity_fraction=.25),
    completion_policy=dict(enabled=True, max_parents=256, max_completions_per_parent=64,
                           max_completions_total=4096, max_shift_mm=1000., min_axis_projection=1e-3,
                           min_pair_sine=1e-3),
    completion_budgets=dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
        max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
        max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
        chunk_rows=65_536, max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
        max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
        max_face_build_nodes=131_071, max_face_build_triangle_visits=10_000_000,
        max_refined_start_groups=48, max_report_bytes=100_663_296),
    completion_budget_declaration=dict(
        schema="INHERITED_PLANE_INITIALIZATION_BUDGETS_V12_TYPE_AS_A_SEPARATE_ADDITIONAL_ENVELOPE",
        active_fields=["max_hypothesis_records", "chunk_rows", "max_prefilter_query_calls",
                       "max_prefilter_query_rows", "max_work_units", "max_cached_cad_faces",
                       "max_indexed_face_triangles", "max_face_build_nodes", "max_face_build_triangle_visits",
                       "max_report_bytes", "max_node_visits", "max_triangle_pairs", "max_normal_pairs"],
        inactive_fields=["max_prefilter_unsigned_assignments", "max_refined_start_groups"],
        inactive_scope="DECLARED_FOR_TYPE_COMPATIBILITY_ONLY_NOT_ENFORCED_BY_THE_COMPLETION_LANE_"
                       "ENUMERATION_IS_CAPPED_BY_COMPLETION_POLICY_AND_MAX_HYPOTHESIS_RECORDS_"
                       "REFINEMENT_SLOTS_BY_THE_DECLARED_LANE_QUOTAS_INSIDE_THE_MAIN_LANE_MAX_REFINED_START_GROUPS",
        finite_query_charging="V10_CORRECTION_THE_COMPLETION_LANE_FINITE_FACE_QUERIES_ALLOCATE_AND_CHARGE_NODE_VISITS_"
                              "TRIANGLE_PAIRS_NORMAL_PAIRS_AND_WORK_UNITS_AGAINST_THIS_ENVELOPE_AND_CERTIFICATE_PAIRS_"
                              "AGAINST_CERTIFICATE_CAPS_THE_V9_INACTIVE_DECLARATION_WAS_FALSE_VALUES_UNCHANGED",
        work_units_formula="BOX_SCREEN_ROW_VISITS_PLUS_FINITE_FACE_QUERY_SURFACE_WORK_UNITS_FACE_NORMAL_TESTS_"
                           "CANDIDATE_SORT_WORK_AND_POSE_KEYS_ARE_REPORTED_COUNTERS_NOT_CHARGED_WORK",
        allowance_interaction="THE_REFINEMENT_ALLOWANCE_COMPLETION_ROW_QUOTA_IS_ONE_ALLOCATION_FOR_THE_WHOLE_LANE_"
                              "QUERY_ROWS_ONLY_ATTEMPTED_ROWS_CHARGED_BEFORE_EACH_SCREEN_OR_QUERY_COMPLETED_ROWS_AFTER_A_"
                              "COMPLETE_ANSWER_ENUMERATION_BINDING_AND_INDEX_BUILDS_CHARGE_NO_ROWS_CPU_PHASE_CHECKS_ONLY_"
                              "TYPED_EXPIRY_ROLLS_BACK_COMPLETION_GROUPS_ONLY_AND_RESTORES_MAIN_GROUPS",
        combined_allowance="MAIN_INITIALIZATION_BUDGETS_PLUS_THIS_SEPARATE_ENVELOPE_UNDER_THE_UNCHANGED_"
                           "OUTER_WALL_AND_RSS_LIMITS_NOT_AN_UNCHANGED_TOTAL_WORK_EXPERIMENT",
        ledger_cap="COMPACT_JSON_BYTES_OF_THE_PUBLISHED_LANE_RECHECKED_AFTER_ITS_LAST_MUTATION"),
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
        max_pair_batch=65_536, max_ties_per_query=256),
    registration_budgets=dict(max_query_calls=4096, max_query_rows=64_000_000,
        max_node_visits=2_000_000_000, max_triangle_pairs=2_000_000_000,
        max_normal_pairs=500_000_000, max_work_units=20_000_000_000),
)

# Lane declarations are separate from unchanged scientific CONFIG. Both newly
# declared successor arms explicitly request context V3, independently of their
# unchanged scheduler choice. This is not historical paired-control continuity.
TWO_PLANE_DECLARATION = dict(enabled_arms=["control", "challenger"],
    quota="TRIPLET_FIRST_RESIDUAL_JOINT8192_RECORDS_1024_UNSIGNED_NO_REFUND",
    pair_lane_starvation_possible=True, global_search_exhaustive=False,
    schedule_budgets=dict(max_scan_planes=128, max_cad_planes=2048,
        max_scan_pair_records=8128, max_selected_scan_pairs=64,
        max_scan_selection_tests=520192, max_cad_pair_records=2096128,
        max_cad_sort_work=44018688, max_window_pair_tests=134152192,
        max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
        max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536),
    context_selection="BOTH_SUCCESSOR_ARMS_EXPLICIT_CONTEXT_V4",
    context_request_scope="BOTH_SUCCESSOR_ARMS_CONTEXT_V4_NOT_HISTORICAL_PAIRED_CONTROL")
CONTEXT_V4_REQUEST = dict(mode="v4",
    implementation="RAW_EXTREMA_KD_PARTITION_CONTEXT_V4_STAGED_SCAN_PROJECTION",
    schema_version="two_plane_context_v4",
    budgets=dict(max_scan_rows=1000000, max_scan_triangles=2000000,
        max_cad_triangles=500000, max_scan_planes=128, max_cad_planes=2048,
        max_scan_pairs=64, max_ordered_cad_matches_per_scan_pair=256,
        max_signed_families=1024, max_candidate_slots=8192,
        max_scan_row_visits=64000000, max_cad_vertex_visits=128000000,
        max_index_visits=64000000, max_index_sort_work=256000000,
        max_hash_bytes=1073741824, max_projection_cache_entries=6144,
        max_working_bytes=268435456, max_report_bytes=33554432, chunk_rows=65536,
        max_extrema_indexes=129, max_extrema_indexed_rows=4000000,
        max_extrema_node_visits=16000000, max_extrema_coordinate_visits=512000000,
        max_extrema_queries=3072),
    scan_projection_arithmetic="STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1",
    scan_extrema_partition="RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3",
    scan_projection_bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
    cad_projection_arithmetic="UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
    scan_projection_chunk_scope="FIXED_EXTREMA_LEAF64_NOT_CHUNK_ROWS",
    chunk_rows_scope="INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
    extrema_guard_status="NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_V3_INVARIANTS",
    extrema_membership_scope="SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN",
    extrema_memory_scope="SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS")
CONTEXT_REQUESTS_BY_ARM = {arm: json.loads(json.dumps(CONTEXT_V4_REQUEST)) for arm in ARMS}
# Literal default maps are independently copied from the named producer and
# scheduler sources, never discovered by importing backend code during freeze.
SEEDED_PLANE_BUDGETS = dict(max_seeds=32, max_seed_selection_triangle_visits=64_000_000,
    max_proposal_raw_row_visits=32_000_000, max_proposal_triangle_visits=64_000_000,
    max_proposal_edge_visits=96_000_000, max_full_support_row_tests=32_000_000,
    max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_fit_evaluations=32, max_retained_array_bytes=268_435_456,
    max_metadata_bytes=67_108_864, max_working_array_bytes=1_073_741_824)
COMPOSITE_PLANE_BUDGETS = dict(max_qualified_planes=65_552,
    max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
    max_full_support_row_tests=64_000_000, max_raw_predicate_visits=48_000_000,
    max_triangle_predicate_visits=160_000_000, max_edge_predicate_visits=192_000_000,
    max_fit_evaluations=65_584, max_retained_array_bytes=536_870_912,
    max_metadata_bytes=67_108_864)
SCHEDULE_V2_BUDGETS = dict(TWO_PLANE_DECLARATION["schedule_budgets"],
    max_residual_priority_work=384, max_residual_pair_visits=64,
    max_residual_breadth_position_tests=128, max_residual_rank_selections=128,
    max_residual_skip_position_tests=512, max_residual_index_lookups=128,
    max_residual_cache_reference_records=64, max_residual_cache_reference_bytes=32768,
    max_residual_retained_bytes=1048576, max_residual_report_records=512,
    max_residual_report_reserved_bytes=1048576, max_final_ledger_tests=320)
SCORING_SOURCES = (
    "scripts/f1_physical_evaluation_contract_v2.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/gold11_contact_scoring_v2.py", "scripts/gold11_schedule_projection_v1.py",
    "scripts/gold11_contact_scoring_v3.py", "scripts/gold11_schedule_projection_v4.py",
    "scripts/gold11_contact_scoring_v4.py", "scripts/gold11_schedule_projection_v5.py",
    "scripts/gold11_contact_scoring_v5.py", "scripts/gold11_schedule_projection_v6.py",
)
_VERIFIED_SCORING_SOURCES = None
_SCORING_MODULES = None
SCORING_TRANSPORT = dict(
    retained_slot="SCORING_INPUT.json",
    returned_result_bytes=128*1024**2, inner_projection_bytes=8*1024**2,
    retained_schedule_wrapper_bytes=8*1024**2,
    wrapper_schema="GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V6",
    parent_reconstruction="ORIGINAL_RESULT_BYTES_EXPECTED_SHA_AND_FROZEN_FULL_ARM_REQUEST",
    result_reencoding_allowed=False, supervisor_refusal_keys=6,
    final_reference_access="ONLY_AFTER_ALL22_DURABLE_RECONSTRUCTED_TERMINALS",
    carrier="MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR",
    historical_comparability="UNESTABLISHED")

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



def require_release():
    """Fail before process/source/control/input access; no environment bypass."""
    if RELEASE_AUTHORIZED is not True:
        raise ValueError("V11_DRAFT_BLOCKED_NOT_FROZEN_OR_AUTHORIZED")
    if TIMING_RELEASE_STATUS != "REVIEWED_V11_EXACT_TIMING_RELEASE":
        raise ValueError("V11_TIMING_DECLARATION_NOT_RELEASED")
    for name in ("case_wall_seconds", "case_cpu_seconds", "total_wall_seconds",
                 "scorer_wall_seconds", "scorer_cpu_seconds", "parent_read_seconds",
                 "terminalization_reserve_seconds"):
        value = LIMITS[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("V11_CONCRETE_TIMING_LIMITS_REQUIRED:" + name)
    if (type(LIMITS["case_cpu_seconds"]) is not int
            or type(LIMITS["scorer_cpu_seconds"]) is not int
            or LIMITS["terminalization_reserve_seconds"] >= LIMITS["total_wall_seconds"]):
        raise ValueError("V11_INVALID_RELEASE_TIMING_LIMITS")


def arm_request(arm):
    """Complete JSON-stable frozen declaration, including explicit default maps."""
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("LEAF_ARM_SCHEMA")
    composite = arm == "challenger"
    scan = dict(mode="composite_v1", seeded_budgets=SEEDED_PLANE_BUDGETS,
                composite_budgets=COMPOSITE_PLANE_BUDGETS) if composite else None
    schedule = dict(mode="v2" if composite else "v1",
        budgets=SCHEDULE_V2_BUDGETS if composite else TWO_PLANE_DECLARATION["schedule_budgets"],
        initialization_budgets=dict(CONFIG["initialization_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]))
    return decode(encoded(dict(workflow="CAD_GUIDED_MESH_OFFLINE_V13", implementation="CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1",
        initialization_implementation="CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1",
        registration_implementation="SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1",
        two_plane_context_mode="v4", native_adapter_version="v1",
        common_config=CONFIG, certificate_caps=CERTIFICATE_CAPS, two_plane_enabled=True,
        connected_measurement_contexts=True, scan_plane_mode="composite_v1" if composite else "v7",
        scan_plane_configuration=scan, schedule_request=schedule,
        context_request=CONTEXT_REQUESTS_BY_ARM[arm])))


def arm_request_sha256(arm):
    return digest(encoded(arm_request(arm)))


def all_arm_requests():
    return {arm: arm_request(arm) for arm in ARMS}


def scan_plane_requests():
    return {arm: dict(mode=arm_request(arm)["scan_plane_mode"],
        configuration=arm_request(arm)["scan_plane_configuration"]) for arm in ARMS}


def projection_module():
    """Load named verified frozen buffers in a private, inert scripts namespace.

    The exact matcher imports NumPy/SciPy; it is not invoked during projection.
    No backend namespace, original input, default output path or CLI is called.
    All module bytes are verified before the first compiled module executes.
    Coordinator/parent calls remain inside their absolute wall/RSS supervision;
    they do not acquire a separate hard CPU limit merely by using pure helpers.
    """
    global _SCORING_MODULES
    require_release()
    if (type(_VERIFIED_SCORING_SOURCES) is not dict
            or set(_VERIFIED_SCORING_SOURCES) != set(SCORING_SOURCES)
            or any(not is_hex(value, 64) for value in _VERIFIED_SCORING_SOURCES.values())):
        raise ValueError("VERIFIED_FROZEN_SCORING_BUFFERS_REQUIRED")
    if _SCORING_MODULES is not None:
        if _SCORING_MODULES["hashes"] != _VERIFIED_SCORING_SOURCES:
            raise ValueError("LOADED_SCORING_SOURCE_BINDING_CHANGED")
        return _SCORING_MODULES["projection"]
    blobs = {name: _bounded_read(RUN/"source"/name, 2*1024**2, sha)
             for name, sha in _VERIFIED_SCORING_SOURCES.items()}
    package = "_gold11_v11_exact_scoring"
    if any(name == package or name.startswith(package+".") for name in sys.modules):
        raise ValueError("UNEXPECTED_EXISTING_PRIVATE_SCORING_NAMESPACE")
    namespace = ModuleType(package)
    namespace.__package__, namespace.__path__ = package, []
    sys.modules[package] = namespace
    installed = [package]
    try:
        sys.path.insert(0, runtime_roots()[0])
        modules = {}
        for relative in SCORING_SOURCES:
            leaf = Path(relative).stem
            name = package+"."+leaf
            module = ModuleType(name)
            module.__package__ = package
            module.__file__ = str(RUN/"source"/relative)
            sys.modules[name] = module
            installed.append(name)
            exec(compile(blobs[relative], module.__file__, "exec"), module.__dict__)
            setattr(namespace, leaf, module)
            modules[leaf] = module
        _SCORING_MODULES = dict(hashes=dict(_VERIFIED_SCORING_SOURCES),
            projection=modules["gold11_schedule_projection_v6"],
            contact=modules["gold11_contact_scoring_v5"])
        return _SCORING_MODULES["projection"]
    except BaseException:
        for name in reversed(installed):
            sys.modules.pop(name, None)
        raise


def projection_request(arm, expected_sha):
    declaration = arm_request(arm)
    return dict(expected_result_sha256=expected_sha,
        expected_scan_plane_mode=declaration["scan_plane_mode"],
        expected_scan_plane_configuration=declaration["scan_plane_configuration"],
        expected_schedule_request=declaration["schedule_request"],
        expected_context_request=declaration["context_request"],
        max_result_bytes=LIMITS["case_result_bytes"],
        # Both APIs must admit the same retained byte cap, including its newline.
        # compact_result separately checks encoded body + publication newline.
        max_wrapper_bytes=LIMITS["compact_case_bytes"])


def compact_result(result_bytes, *, expected_result_sha256, arm):
    """Original bytes -> qualified schedule wrapper; never reserialize RESULT."""
    result = projection_module().project_gold11_schedule_workflow_v6(
        result_bytes, **projection_request(arm, expected_result_sha256))
    if len(encoded(result))+1 > LIMITS["compact_case_bytes"]:
        raise ValueError("SCORING_PROJECTION_BYTE_BUDGET")
    return result


def valid_supervisor_refusal(value):
    return (type(value) is dict and set(value) == {"status", "phase", "reasons",
        "predictions", "observed_bend_count", "production_use_allowed"}
        and value["status"] == "UNAVAILABLE" and value["phase"] in ("SUPERVISOR", "GLOBAL_TERMINALIZATION")
        and type(value["reasons"]) is list and 1 <= len(value["reasons"]) <= 64
        and all(type(reason) is str and 1 <= len(reason) <= 512 for reason in value["reasons"])
        and type(value["predictions"]) is list and value["predictions"] == []
        and value["observed_bend_count"] is None and value["production_use_allowed"] is False)


def reconstruct_scoring_input(projection_bytes, receipt, arm, *, result_bytes=None):
    """Rebuild every usable inner projection, or exact retained supervisor veto."""
    if (type(receipt) is not dict or receipt.get("status") != "TERMINAL"
            or receipt.get("arm_request_sha256") != arm_request_sha256(arm)
            or not is_hex(receipt.get("scoring_input_sha256"), 64)
            or digest(projection_bytes) != receipt["scoring_input_sha256"]):
        raise ValueError("TERMINAL_ARM_OR_PROJECTION_BYTE_BINDING")
    clean = (type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
        and receipt.get("budget_stop") is None and "failure_type" not in receipt)
    if clean:
        sha = receipt.get("result_sha256")
        if (not is_hex(sha, 64) or type(result_bytes) is not bytes
                or receipt.get("projection_kind") != "EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER"):
            raise ValueError("CLEAN_CASE_ORIGINAL_RESULT_REQUIRED")
        return projection_module().revalidate_gold11_schedule_projection_v6(
            projection_bytes, expected_projection_sha256=receipt["scoring_input_sha256"],
            result_bytes=result_bytes, **projection_request(arm, sha))["scorer_input"]
    refusal = decode(projection_bytes)
    if (not valid_supervisor_refusal(refusal)
            or not valid_supervisor_refusal(receipt.get("supervisor_refusal"))
            or encoded(refusal) != encoded(receipt["supervisor_refusal"])
            or receipt.get("projection_kind") != "SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW"):
        raise ValueError("FAILED_CASE_EXACT_SUPERVISOR_REFUSAL_REQUIRED")
    return refusal


def reconstruct_all_terminal_inputs(bundle):
    """Read only22 fixed owned terminals/results; must finish BEFORE Gold."""
    expected = {case["case_token"] for case in bundle["cases"]}
    if len(expected) != 11 or len(bundle["cases"]) != 11 or bundle["arms"] != ARMS:
        raise ValueError("ALL22_TERMINAL_DENOMINATOR_REQUIRED")
    outer, inner, receipts = ({arm: {} for arm in ARMS} for _ in range(3))
    for case in bundle["cases"]:
        for arm in ARMS:
            key = case["case_token"]
            prefix = f"cases/{key}/{arm}/"
            receipt = decode(owned_read(prefix+"COMPLETED.json", LIMITS["compact_case_bytes"]))
            projection_bytes = owned_read(prefix+"SCORING_INPUT.json", LIMITS["compact_case_bytes"],
                receipt.get("scoring_input_sha256"))
            clean = (type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
                and receipt.get("budget_stop") is None and "failure_type" not in receipt)
            result = owned_read(prefix+"RESULT.json", LIMITS["case_result_bytes"],
                receipt.get("result_sha256")) if clean else None
            inner[arm][key] = reconstruct_scoring_input(projection_bytes, receipt, arm, result_bytes=result)
            outer[arm][key], receipts[arm][key] = decode(projection_bytes), receipt
    return outer, inner, receipts


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
    require_release()
    # No input/reference paths are touched by this operation.
    blobs = {name: bounded_read(ROOT/name, 2*1024**2) for name in SOURCES}
    if digest(blobs[SOURCE_BASIS]) != SOURCE_BASIS_SHA:
        raise ValueError("V11_FROZEN_V10_SOURCE_BASIS_CHANGED")
    cases = approved_cases_from_review(blobs[REVIEW])
    bundle = dict(schema="GOLD11_MESH_DEVELOPMENT_V11", source_profile=SOURCE_PROFILE,
                  config_delta_from_v10=CONFIG_DELTA_FROM_V10,
                  pose_admission_declaration=POSE_ADMISSION_DECLARATION,
                  measurement_policy_declaration=MEASUREMENT_POLICY_DECLARATION,
                  config=CONFIG, limits=LIMITS,
                  cases=cases, source_sha256={name: digest(blob) for name, blob in blobs.items()},
                  catalog_sha256=CATALOG_SHA, selected_reference_sha256=SELECTED_SHA,
                  runtime_metadata=runtime_metadata(),
                  arms=ARMS, arm_definitions=ARM_DEFINITIONS, comparator=COMPARATOR,
                  position_error_declaration=POSITION_ERROR_DECLARATION,
                  certificate_caps=CERTIFICATE_CAPS, retained_control_policy=RETAINED_CONTROL_POLICY,
                  arm_requests=all_arm_requests(), scoring_transport=SCORING_TRANSPORT,
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
    require_release()
    global _VERIFIED_SUPERVISION_SHA, _VERIFIED_ARTIFACT_SHA, _VERIFIED_SCORING_SOURCES
    bundle = decode(bounded_read(RUN/"FROZEN.json", 256*1024))
    if (bundle.get("schema") != "GOLD11_MESH_DEVELOPMENT_V11"
            or bundle.get("source_profile") != SOURCE_PROFILE
            or bundle.get("config_delta_from_v10") != CONFIG_DELTA_FROM_V10
            or bundle.get("pose_admission_declaration") != POSE_ADMISSION_DECLARATION
            or bundle.get("measurement_policy_declaration") != MEASUREMENT_POLICY_DECLARATION
            or bundle["config"] != CONFIG or bundle["limits"] != LIMITS or bundle["arms"] != ARMS
            or bundle.get("arm_definitions") != ARM_DEFINITIONS
            or bundle.get("comparator") != COMPARATOR
            or bundle.get("position_error_declaration") != POSITION_ERROR_DECLARATION
            or bundle.get("certificate_caps") != CERTIFICATE_CAPS
            or bundle.get("retained_control_policy") != RETAINED_CONTROL_POLICY
            or bundle.get("two_plane_declaration") != TWO_PLANE_DECLARATION
            or bundle.get("artifact_policy") != ARTIFACT_POLICY
            or encoded(bundle.get("arm_requests")) != encoded(all_arm_requests())
            or encoded(bundle.get("scoring_transport")) != encoded(SCORING_TRANSPORT)):
        raise ValueError("FROZEN_PROTOCOL_CONSTANTS_CHANGED")
    if bundle.get("runtime_metadata") != runtime_metadata():
        raise ValueError("FROZEN_RUNTIME_METADATA_CHANGED")
    if set(bundle["source_sha256"]) != set(SOURCES):
        raise ValueError("SOURCE_DENOMINATOR_CHANGED")
    if bundle["source_sha256"][SOURCE_BASIS] != SOURCE_BASIS_SHA:
        raise ValueError("V11_FROZEN_V10_SOURCE_BASIS_CHANGED")
    for name, sha in bundle["source_sha256"].items():
        bounded_read(ROOT/name, 2*1024**2, sha)
        bounded_read(RUN/"source"/name, 2*1024**2, sha)
    cases = approved_cases_from_review(bounded_read(RUN/"source"/REVIEW, 2*1024**2))
    if (bundle["cases"] != cases or bundle["catalog_sha256"] != CATALOG_SHA
            or bundle["selected_reference_sha256"] != SELECTED_SHA):
        raise ValueError("FROZEN_INPUT_OR_REFERENCE_SCOPE_CHANGED")
    _VERIFIED_SCORING_SOURCES = {name: bundle["source_sha256"][name] for name in SCORING_SOURCES}
    if bundle["source_sha256"][SUPERVISION_SOURCE] != SUPERVISION_SOURCE_SHA:
        raise ValueError("FROZEN_V3_SUPERVISION_SOURCE_UNQUALIFIED")
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
SUPERVISION_SOURCE = "scripts/gold11_owned_process_supervision_v3.py"
SUPERVISION_SOURCE_SHA = "9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea"
_COORDINATOR_WRITERS_QUIESCENT = True
_COORDINATOR_ABORTED = False
LEAF_SCHEMA = "ONE_USE_PARENT_BOUND_LEAF_V11"
RUNNER_SOURCE = "scripts/run_gold11_mesh_development_v11.py"


def supervision_module():
    """Compile the exact verified frozen bytes; never reopen via a loader."""
    global _SUPERVISION_MODULE
    if _VERIFIED_SUPERVISION_SHA != SUPERVISION_SOURCE_SHA:
        raise ValueError("VERIFIED_SUPERVISION_SOURCE_REQUIRED")
    if _SUPERVISION_MODULE is None:
        blob = _bounded_read(RUN/"source"/SUPERVISION_SOURCE, 2*1024**2, _VERIFIED_SUPERVISION_SHA)
        module = ModuleType("_gold11_owned_process_supervision_v3")
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
        return (RUNNER_SOURCE, ARTIFACT_SOURCE, STAGE_SAMPLER_SOURCE,
                *(name for name in SOURCES if name.startswith("backend/")))
    if role == "SCORER":
        return (RUNNER_SOURCE, ARTIFACT_SOURCE, *SCORING_SOURCES)
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


def writers_quiescent(receipt, *, new_session=False):
    """Strict V3 receipt predicate, not fresh process or signal authorization."""
    return (type(receipt) is dict
        and receipt.get("supervisor_schema") == "GOLD11_OWNED_PROCESS_SUPERVISION_V3"
        and receipt.get("direct_child_reaped") is True
        and (not new_session or receipt.get("owned_group_empty_confirmed") is True)
        and receipt.get("requires_process_attention") is False
        and receipt.get("observed_foreign_group_escape_detected") is False
        and receipt.get("observed_foreign_group_escape_unconfirmed_after_cleanup") is False
        and receipt.get("observed_identity_unknown_still_present") is False
        and receipt.get("cleanup_errors") == []
        and type(receipt.get("cleanup_errors_discarded")) is int
        and receipt["cleanup_errors_discarded"] == 0
        and receipt.get("control_close_errors") == [])


def fresh_private_cache(prefix):
    """Create only a new private Darwin cache outside the retained attempt."""
    require_release()
    if sys.platform != "darwin":
        raise ValueError("PRIVATE_CACHE_PLATFORM_UNSUPPORTED")
    cache = Path(tempfile.mkdtemp(prefix=prefix, dir="/private/tmp")).resolve()
    identity = cache_identity(cache)
    if contained(str(cache), (str(RUN),)):
        raise ValueError("PRIVATE_CACHE_MUST_BE_OUTSIDE_ATTEMPT")
    return cache, identity


@contextmanager
def owned_leaf_cache(role, receipt):
    """No finalizer: unknown writers or failed retention keep opaque bytes."""
    if not _COORDINATOR_WRITERS_QUIESCENT or _COORDINATOR_ABORTED:
        raise ValueError("COORDINATOR_CUSTODY_OR_ABORT_VETO")
    cache, identity = fresh_private_cache("gold11-v11-"+role.lower()+"-")
    custody = dict(path=str(cache), identity=identity,
        status="FRESH_PRIVATE_DIRECTORY_BOUND_BEFORE_DISPATCH",
        contents_read_for_custody=False)
    receipt["private_cache_custody"] = custody
    receipt["private_cache_cleanup_complete"] = None
    try:
        yield cache
    finally:
        may_have_writer = receipt.get("worker_dispatch_attempted") is True
        safe = (not may_have_writer or writers_quiescent(receipt))
        retained = not may_have_writer or receipt.get("derived_result_retention_complete") is True
        if not safe or not retained:
            receipt["private_cache_cleanup_complete"] = False
            custody["status"] = ("OPAQUE_RETAINED_WRITER_QUIESCENCE_UNCONFIRMED"
                if not safe else "OPAQUE_RETAINED_DERIVED_RETENTION_FAILED")
        else:
            try:
                with deadline(LIMITS["parent_read_seconds"]):
                    if cache_identity(cache) != identity:
                        raise ValueError("PRIVATE_CACHE_IDENTITY_CHANGED_NO_CLEANUP")
                    # Only this exact fresh owned directory; no glob discovery,
                    # reused cache paths or automatic TemporaryDirectory finalizer.
                    shutil.rmtree(cache)
                custody["status"] = "REMOVED_AFTER_QUIESCENCE_AND_DERIVED_RETENTION"
                receipt["private_cache_cleanup_complete"] = True
            except BaseException:
                receipt["private_cache_cleanup_complete"] = False
                custody["status"] = "CLEANUP_FAILED_POSSIBLY_PARTIAL_PRIVATE_CONTENTS_RETAINED"
                raise


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
    expected = {"arm", "scan_sha256", "cad_sha256", "arm_request_sha256"} if role == "INFERENCE" else {"request_sha256"}
    if role not in ("INFERENCE", "SCORER") or type(binding) is not dict or set(binding) != expected:
        raise ValueError("LEAF_BINDING_SCHEMA")
    if role == "INFERENCE" and (type(binding["arm"]) is not str or binding["arm"] not in ARMS
            or binding["arm_request_sha256"] != arm_request_sha256(binding["arm"])):
        raise ValueError("LEAF_ARM_OR_FROZEN_REQUEST_SCHEMA")
    if any(not is_hex(value, 64) for key, value in binding.items() if key != "arm"):
        raise ValueError("LEAF_HASH_SCHEMA")


def authorize_leaf(request, role):
    require_release()
    global _VERIFIED_ARTIFACT_SHA, _VERIFIED_SCORING_SOURCES
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
            or start.get("arm_requests_sha256") != digest(encoded(all_arm_requests()))
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
    if role == "SCORER":
        _VERIFIED_SCORING_SOURCES = {name: sources[name] for name in SCORING_SOURCES}
    return snapshot, cache, binding


def workflow_entry_and_policies(arm):
    """Constructor-only qualification seam; no input read/native inspection."""
    require_release()
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
    from backend.cad_guided_triangle_surface_v4 import TriangleSurfaceBudgetsV4
    from backend.cad_guided_global_registration_v4 import SurfaceRegistrationBudgetsV4
    from backend.cad_guided_global_registration_v5 import REGISTRATION_IMPLEMENTATION, RefinementAllowanceV1
    common.update(
        surface_budgets=TriangleSurfaceBudgetsV4(**CONFIG["surface_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["surface_budgets"]),
        registration_budgets=SurfaceRegistrationBudgetsV4(**CONFIG["registration_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["registration_budgets"]))
    from backend.cad_guided_mesh_offline_v18 import MeshWorkflowPolicyV18, inspect_cad_guided_mesh_v18
    from backend.cad_guided_plane_initialization_v16 import _context_configuration, CompletionLanePolicyV1
    from backend.cad_guided_measurement_v3 import LayerSelectionPolicyV2
    from backend.cad_guided_plane_initialization_v12 import PlaneInitializationBudgetsV12
    from backend.cad_guided_two_plane_schedule_v1 import TwoPlaneScheduleBudgetsV1
    from backend.cad_guided_two_plane_context_v4 import TwoPlaneContextBudgetsV4
    expected_context = CONTEXT_REQUESTS_BY_ARM[arm]
    context_budgets = TwoPlaneContextBudgetsV4(**expected_context["budgets"])
    _, actual_context = _context_configuration("v4", context_budgets)
    if (encoded(actual_context) != encoded(expected_context)
            or encoded(asdict(TwoPlaneScheduleBudgetsV1())) != encoded(TWO_PLANE_DECLARATION["schedule_budgets"])
            or REGISTRATION_IMPLEMENTATION != ARM_DEFINITIONS[arm]["registration_implementation"]):
        raise ValueError("FROZEN_CONTEXT_V4_OR_SCHEDULE_DEFAULT_BINDING")
    allowance = RefinementAllowanceV1(**CONFIG["refinement_allowance"])
    if (encoded(asdict(allowance)) != encoded(CONFIG["refinement_allowance"])
            or allowance.cpu_budget_seconds != LIMITS["case_cpu_seconds"] or allowance.enabled is not True):
        raise ValueError("FROZEN_REFINEMENT_ALLOWANCE_BINDING")
    common.update(workflow_policy=MeshWorkflowPolicyV18(**CONFIG["workflow"], native_adapter_version="v1"),
        refinement_allowance=allowance,
        initialization_budgets=PlaneInitializationBudgetsV12(**CONFIG["initialization_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        completion_policy=CompletionLanePolicyV1(**CONFIG["completion_policy"]),
        completion_budgets=PlaneInitializationBudgetsV12(**CONFIG["completion_budgets"],
            max_certificate_pairs=CERTIFICATE_CAPS["initialization_budgets"]),
        layer_policy=LayerSelectionPolicyV2(**CONFIG["layer_policy"]),
        two_plane_enabled=True, connected_measurement_contexts=True,
        two_plane_context_mode="v4", two_plane_context_budgets=context_budgets,
        scan_plane_mode="v7", seeded_plane_budgets=None, composite_plane_budgets=None,
        two_plane_schedule_mode="v1", two_plane_schedule_budgets=None)
    if arm == "challenger":
        from backend.cad_guided_seeded_plane_proposals_v1 import SeededPlaneBudgetsV1
        from backend.cad_guided_scan_plane_preparation_v1 import CompositeScanPlaneBudgetsV1
        from backend.cad_guided_two_plane_schedule_v2 import TwoPlaneScheduleBudgetsV2
        common.update(scan_plane_mode="composite_v1",
            seeded_plane_budgets=SeededPlaneBudgetsV1(**SEEDED_PLANE_BUDGETS),
            composite_plane_budgets=CompositeScanPlaneBudgetsV1(**COMPOSITE_PLANE_BUDGETS),
            two_plane_schedule_mode="v2", two_plane_schedule_budgets=TwoPlaneScheduleBudgetsV2(**SCHEDULE_V2_BUDGETS))
        for value, expected in ((common["seeded_plane_budgets"], SEEDED_PLANE_BUDGETS),
                (common["composite_plane_budgets"], COMPOSITE_PLANE_BUDGETS),
                (common["two_plane_schedule_budgets"], SCHEDULE_V2_BUDGETS)):
            if encoded(asdict(value)) != encoded(expected):
                raise ValueError("EXPLICIT_COMPOSITE_OR_SCHEDULE_CONSTRUCTOR_BINDING")
    return inspect_cad_guided_mesh_v18, common


def stage_sampler_module(snapshot):
    """Exact frozen source read is mandatory custody, not optional telemetry."""
    require_release()
    global _STAGE_SAMPLER_MODULE
    blob = bounded_read(Path(snapshot)/STAGE_SAMPLER_SOURCE, 2*1024**2, STAGE_SAMPLER_SHA)
    module = ModuleType("_gold11_v11_stage_sampler_v1")
    exec(compile(blob, str(Path(snapshot)/STAGE_SAMPLER_SOURCE), "exec"), module.__dict__)
    if module.POLICY.get("maximum_sample_stream_bytes") != STAGE_LOG_DECLARATION["sample_stream_bytes"]:
        raise ValueError("FROZEN_STAGE_SAMPLER_LOG_POLICY_MISMATCH")
    module.__verified_source_sha256 = STAGE_SAMPLER_SHA
    _STAGE_SAMPLER_MODULE = module
    return module


def engine_call_with_stage_sampler(snapshot, entry, scan, cad, policies):
    """Observe one unchanged call; diagnostic failures cannot replace its result."""
    require_release()
    module = _STAGE_SAMPLER_MODULE
    if module is None or module.__verified_source_sha256 != STAGE_SAMPLER_SHA:
        raise ValueError("VERIFIED_STAGE_SAMPLER_PRELOAD_REQUIRED")
    sampler = None
    startup_failure_type = None
    diagnostic = dict(status="START_NOT_COMPLETED", inference_result_modified=False)
    try:
        sampler = module.MainThreadStageSamplerV1(source_root=str(snapshot),
            source_paths=STAGE_SAMPLE_SOURCE_PATHS, output_fd=2)
        sampler.start()
    except Exception as error:
        startup_failure_type = type(error).__name__
        diagnostic = dict(status="START_FAILED", failure_type=type(error).__name__,
            inference_result_modified=False)
    try:
        return entry(scan_bytes=scan, cad_bytes=cad, **policies)
    finally:
        if sampler is not None:
            try:
                diagnostic = sampler.finish()
            except Exception as error:
                diagnostic = dict(status="FINISH_FAILED", failure_type=type(error).__name__,
                    sampler_thread_stopped="UNKNOWN", inference_result_modified=False)
        # Separate log evidence only. Prefix newline separates any preceding
        # partial diagnostic line. Framing and both newlines count in4096 bytes.
        try:
            record = dict(status="ENGINE_STAGE_SAMPLER_TERMINAL", receipt=diagnostic,
                startup_failure_type=startup_failure_type,
                log_declaration=STAGE_LOG_DECLARATION)
            line = b"\n"+encoded(record)+b"\n"
            if len(line) > STAGE_LOG_DECLARATION["receipt_bytes"]:
                line = b'\n{"status":"STAGE_SAMPLER_RECEIPT_TOO_LARGE_OMITTED_WHOLE"}\n'
            sys.stderr.write(line.decode("utf-8"))
            sys.stderr.flush()
        except Exception:
            # A missing receipt cannot be called complete telemetry by POST.
            # Do not mask an engine exception or rewrite a valid engine result.
            pass


def stage_checkpoint_printer():
    """Bounded STAGE_CHECKPOINT stdout lines: emitted evidence of what a killed leaf reached, never a
    result, a measurement or a success claim. Beyond the declared byte cap nothing more is printed and
    one notice line records the cap; an unencodable record is omitted whole."""
    state = dict(bytes=0, truncated=False)
    cap = STAGE_LOG_DECLARATION["checkpoint_stdout_bytes"]
    def emit(record):
        if state["truncated"]:
            return
        try:
            line = json.dumps(dict(status="STAGE_CHECKPOINT", checkpoint=record), sort_keys=True, allow_nan=False)+"\n"
        except Exception:
            line = '{"status":"STAGE_CHECKPOINT_UNENCODABLE_OMITTED"}\n'
        if state["bytes"]+len(line.encode("utf-8")) > cap:
            state["truncated"] = True
            line = '{"status":"STAGE_CHECKPOINT_STDOUT_CAP_REACHED_FURTHER_CHECKPOINTS_OMITTED"}\n'
        else:
            state["bytes"] += len(line.encode("utf-8"))
        sys.stdout.write(line)
        sys.stdout.flush()
    emit.state = state
    return emit


def worker():
    require_release()
    # Parent sends no Gold, case/family identity or original source path.
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["case_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "INFERENCE")
    # Named source is verified by authorize_leaf, then compiled from its exact
    # pinned buffer before the backend-only filesystem hook is installed.
    stage_sampler_module(snapshot)
    initialize_worker(snapshot, cache, "backend", LIMITS["case_cpu_seconds"])
    entry, policies = workflow_entry_and_policies(request["arm"])
    policies["stage_checkpoint"] = stage_checkpoint_printer()
    scan = bounded_read(cache/"scan.ply", LIMITS["max_scan_bytes"], request["scan_sha256"])
    cad = bounded_read(cache/"cad.step", LIMITS["max_cad_bytes"], request["cad_sha256"])
    print(json.dumps(dict(status="PUBLIC_WORKFLOW_CALL_BEGIN", arm=request["arm"],
                          internal_stage="UNKNOWN_UNTIL_RETURNED_RESULT")), flush=True)
    result = engine_call_with_stage_sampler(snapshot, entry, scan, cad, policies)
    blob = encoded(result)
    if len(blob) > LIMITS["case_result_bytes"]:
        raise ValueError("RESULT_BYTE_BUDGET")
    publish_bytes(cache/"RESULT.json", blob)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(status="INFERENCE_TERMINAL", result_sha256=digest(blob),
                          worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
                          worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)


def clean_environment(cache):
    return supervision_module().clean_environment(cache)


def runner_command(mode):
    return [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
            str(RUN/"source/scripts/run_gold11_mesh_development_v11.py"), mode]


def memory_budget(tree_rss, parent_rss, rss_limit, *, include_parent=False, parent_rss_limit=None):
    return supervision_module().memory_budget(tree_rss, parent_rss, rss_limit,
        include_parent=include_parent, parent_rss_limit=parent_rss_limit)


def supervise_process(command, request, *, cache, log_path, wall_seconds, rss_limit,
                      log_limit, new_session=False, parent_rss_limit=None):
    global _COORDINATOR_WRITERS_QUIESCENT
    require_release()
    if not new_session:
        if not _COORDINATOR_WRITERS_QUIESCENT or _COORDINATOR_ABORTED:
            raise ValueError("NO_LEAF_AFTER_CUSTODY_LOSS_OR_COORDINATOR_ABORT")
        _COORDINATOR_WRITERS_QUIESCENT = False
    receipt = supervision_module().supervise_process(command, request, cache=cache,
        log_path=log_path, wall_seconds=wall_seconds, rss_limit=rss_limit,
        log_limit=log_limit, new_session=new_session, parent_rss_limit=parent_rss_limit)
    if not new_session:
        _COORDINATOR_WRITERS_QUIESCENT = writers_quiescent(receipt)
    return receipt


def coordinator_memory_check():
    """Darwin lifetime high-water bytes: conservative, never called current RSS."""
    if sys.platform != "darwin":
        raise RuntimeError("COORDINATOR_HIGHWATER_PLATFORM_UNSUPPORTED")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak < 0:
        raise RuntimeError("COORDINATOR_HIGHWATER_UNAVAILABLE")
    if peak > LIMITS["coordinator_rss_bytes"]:
        raise RuntimeError("COORDINATOR_LIFETIME_HIGHWATER_OVER_LIMIT")
    return dict(bytes=peak, scope="DARWIN_RU_MAXRSS_LIFETIME_HIGHWATER_NOT_CURRENT_RSS")


def retain_worker_result(cache, destination, receipt, *, scorer=False):
    """Retain the one complete/partial derived result, never raw input caches.

    Called after supervision ends, before private raw-input cache cleanup. A
    linked complete name and its leftover partial share bytes; retain them once
    and record that identity. No partial is parsed/scored as a successful result.
    """
    if not writers_quiescent(receipt):
        raise ValueError("WRITER_QUIESCENCE_REQUIRED_BEFORE_ANY_PRIVATE_RESULT_READ")
    custody = receipt.get("private_cache_custody")
    if (type(custody) is not dict or custody.get("path") != str(cache)
            or cache_identity(cache) != custody.get("identity")):
        raise ValueError("FRESH_PRIVATE_CACHE_IDENTITY_REQUIRED_BEFORE_RESULT_RETENTION")
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
    require_release()
    started = time.monotonic()
    outcome = unavailable("INFERENCE_NOT_STARTED")
    receipt = dict(status="TERMINAL", budget_stop=None, source_binding_valid=False,
                   derived_result_retention_complete=None,
                   inference_started=False, arm_request_sha256=arm_request_sha256(arm),
                   workflow_stage_diagnostics=workflow_stage_diagnostics(None, arm))
    try:
        destination.mkdir(parents=True, exist_ok=False)
        exclusive_json(destination/"STARTED.json", dict(case_id=case["case_token"], arm=arm,
                       arm_request_sha256=arm_request_sha256(arm), limits=LIMITS, no_resume=True))
        coordinator_memory_check()
        if remaining_wall < LIMITS["case_wall_seconds"]:
            raise ValueError("TOTAL_WALL_RESERVE_UNAVAILABLE")
        with deadline(min(LIMITS["parent_read_seconds"], remaining_wall)):
            scan = _bounded_read(case["scan_path"], LIMITS["max_scan_bytes"], case["scan_sha256"])
            cad = _bounded_read(case["cad_path"], LIMITS["max_cad_bytes"], case["cad_sha256"])
        receipt["source_binding_valid"] = True
        with owned_leaf_cache("INFERENCE", receipt) as cache:
            with deadline(LIMITS["parent_read_seconds"]):
                for name, blob in (("scan.ply", scan), ("cad.step", cad)):
                    publish_bytes(cache/name, blob)
            del scan, cad
            receipt["inference_started"] = "UNKNOWN_UNTIL_RETURNED_RESULT"
            receipt["worker_dispatch_attempted"] = True
            request = issue_leaf_dispatch("INFERENCE", cache, arm=arm,
                           scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"],
                           arm_request_sha256=arm_request_sha256(arm))
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
                        receipt["retained_result_artifacts"] = retain_worker_result(cache, destination, receipt)
                        receipt["derived_result_retention_complete"] = True
                    except Exception as error:
                        # Unknown writers are never inspected; failed derived
                        # retention keeps the exact fresh private cache opaque.
                        # Either failure prevents further inference/scoring.
                        receipt["derived_result_retention_complete"] = False
                        receipt["retained_result_artifacts"] = dict(
                            status="RETENTION_UNAVAILABLE_PRIVATE_CACHE_RETAINED_OPAQUE",
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
                outcome = compact_result(result_blob, expected_result_sha256=receipt["result_sha256"], arm=arm)
                receipt["projection_kind"] = "EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER"
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
        receipt["supervisor_refusal"] = outcome
        receipt["projection_kind"] = "SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW"
    receipt["parent_wall_seconds"] = time.monotonic()-started
    # Storage failures propagate to the coordinator/outer terminal-fill path;
    # they must not claim a durable outcome that could not actually be written.
    destination.mkdir(parents=True, exist_ok=True)
    receipt["scoring_input_sha256"] = digest(encoded(outcome)+b"\n")
    exclusive_json(destination/"SCORING_INPUT.json", outcome)
    exclusive_json(destination/"COMPLETED.json", receipt)
    print(json.dumps(dict(case_id=case["case_token"], arm=arm, status=outcome.get("status", outcome.get("disposition", "MALFORMED")),
                          phase=outcome.get("phase", "SEE_EXACT_RESULT_AND_SCHEDULE_PROJECTION"), wall_seconds=receipt["parent_wall_seconds"],
                          budget_stop=receipt["budget_stop"])), flush=True)
    return outcome, receipt


def selected_references(cases):
    require_release()
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
                        return (type(prior) is dict and (prior.get("status") == "TERMINAL" if completed else
                            valid_supervisor_refusal(prior) or prior.get("schema") == SCORING_TRANSPORT["wrapper_schema"]))
                    scoring = module.ensure_terminal_record_v4(destination, "SCORING_INPUT.json",
                        lambda: encoded(unavailable(reason, "GLOBAL_TERMINALIZATION"))+b"\n",
                        validate, max_bytes=LIMITS["compact_case_bytes"])
                    completion = module.ensure_terminal_record_v4(destination, "COMPLETED.json",
                        lambda: encoded(dict(status="TERMINAL", reason=reason, no_resume=True,
                            arm_request_sha256=arm_request_sha256(arm),
                            projection_kind="SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW",
                            supervisor_refusal=unavailable(reason, "GLOBAL_TERMINALIZATION"),
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



def make_scoring_request(bundle, selected, inner, receipts):
    expected = {case["case_token"] for case in bundle["cases"]}
    if (len(expected) != 11 or set(selected) != expected or set(inner) != set(ARMS)
            or set(receipts) != set(ARMS)
            or any(set(inner[arm]) != expected or set(receipts[arm]) != expected for arm in ARMS)):
        raise ValueError("SCORING_RECONSTRUCTED_ALL22_REQUIRED")
    bindings = {arm: {key: dict(arm_request_sha256=arm_request_sha256(arm),
        scoring_input_sha256=receipts[arm][key]["scoring_input_sha256"],
        result_sha256=receipts[arm][key].get("result_sha256"),
        projection_kind=receipts[arm][key]["projection_kind"],
        scorer_input_canonical_sha256=digest(encoded(inner[arm][key])))
        for key in expected} for arm in ARMS}
    return dict(schema="GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6",
        reference_records_by_case={key: row["bends"] for key, row in selected.items()},
        control_results_by_case=inner["control"], challenger_results_by_case=inner["challenger"],
        case_family_by_id={case["case_token"]: case["part_key"] for case in bundle["cases"]},
        scan_plane_requests_by_arm=scan_plane_requests(), arm_requests=all_arm_requests(),
        projection_bindings=bindings, selected_reference_sha256=SELECTED_SHA)


def scorer_arguments(request):
    """No arbitrary wrapper unwrap: inputs came from bound parent reconstruction."""
    expected_keys = {"schema", "reference_records_by_case", "control_results_by_case",
        "challenger_results_by_case", "case_family_by_id", "scan_plane_requests_by_arm",
        "arm_requests", "projection_bindings", "selected_reference_sha256"}
    if (type(request) is not dict or set(request) != expected_keys
            or request["schema"] != "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6"
            or request["selected_reference_sha256"] != SELECTED_SHA
            or encoded(request["arm_requests"]) != encoded(all_arm_requests())
            or encoded(request["scan_plane_requests_by_arm"]) != encoded(scan_plane_requests())):
        raise ValueError("SCORER_FROZEN_ARM_REQUEST_BINDING")
    cases = request["case_family_by_id"]
    if type(cases) is not dict or len(cases) != 11:
        raise ValueError("SCORER_CASE_DENOMINATOR")
    bindings = request["projection_bindings"]
    if type(bindings) is not dict or set(bindings) != set(ARMS):
        raise ValueError("SCORER_PROJECTION_BINDING_DENOMINATOR")
    for arm in ARMS:
        rows, declared = request[arm+"_results_by_case"], bindings[arm]
        if type(rows) is not dict or type(declared) is not dict or set(rows) != set(cases) or set(declared) != set(cases):
            raise ValueError("SCORER_ALL22_PROJECTION_BINDINGS_REQUIRED")
        for key, item in rows.items():
            row = declared[key]
            if (type(row) is not dict or set(row) != {"arm_request_sha256", "scoring_input_sha256",
                    "result_sha256", "projection_kind", "scorer_input_canonical_sha256"}
                    or row["arm_request_sha256"] != arm_request_sha256(arm)
                    or not is_hex(row["scoring_input_sha256"], 64)
                    or row["scorer_input_canonical_sha256"] != digest(encoded(item))):
                raise ValueError("SCORER_INNER_PROJECTION_TRANSPORT_BINDING")
            if row["projection_kind"] == "EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER":
                if not is_hex(row["result_sha256"], 64):
                    raise ValueError("SCORER_RETURNED_RESULT_HASH_REQUIRED")
            elif row["projection_kind"] == "SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW":
                if not valid_supervisor_refusal(item):
                    raise ValueError("SCORER_FAILED_WORKER_HAS_USABLE_PROJECTION")
            else:
                raise ValueError("SCORER_PROJECTION_KIND")
    return {key: request[key] for key in ("reference_records_by_case", "control_results_by_case",
        "challenger_results_by_case", "case_family_by_id", "scan_plane_requests_by_arm")}


def scorer_worker():
    require_release()
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["scorer_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "SCORER")
    initialize_worker(snapshot, cache, "scripts", LIMITS["scorer_cpu_seconds"])
    payload = decode(bounded_read(cache/"SCORING_REQUEST.json", LIMITS["scorer_request_bytes"],
                                 request["request_sha256"]))
    projection_module()  # exact verified private module graph, no backend imports
    result = _SCORING_MODULES["contact"].score_gold11_contact_workflows_v5(**scorer_arguments(payload))
    blob = encoded(result)
    if len(blob) > LIMITS["scorer_result_bytes"]:
        raise ValueError("SCORER_RESULT_BYTE_BUDGET")
    publish_bytes(cache/"SCORER_RESULT.json", blob)
    print(json.dumps(dict(status="SCORER_TERMINAL", result_sha256=digest(blob))), flush=True)


def score_after_terminal(bundle, outcomes, receipts):
    require_release()
    if _COORDINATOR_ABORTED or not _COORDINATOR_WRITERS_QUIESCENT:
        raise ValueError("NO_SCORING_OR_REFERENCE_ACCESS_AFTER_ABORT_OR_CUSTODY_LOSS")
    # Only this post-terminal coordinator function can open the fixed catalog.
    verify_bundle()
    with deadline(LIMITS["parent_read_seconds"]):
        stored, inner, durable_receipts = reconstruct_all_terminal_inputs(bundle)
        terminal = decode(owned_read("INFERENCE_TERMINAL.json", 256*1024))
        if (encoded(stored) != encoded(outcomes) or encoded(durable_receipts) != encoded(receipts)
                or any(type(terminal.get(key)) is not int or terminal[key] != count
                    for key, count in (("case_count", 11), ("arm_count", 2), ("terminal_record_count", 22)))
                or terminal.get("outcomes_sha256") != digest(encoded(stored))
                or terminal.get("arm_requests_sha256") != digest(encoded(all_arm_requests()))):
            raise ValueError("ALL22_DURABLE_TERMINALS_BEFORE_REFERENCE_ACCESS")
    selected = selected_references(bundle["cases"])
    # New durable selected71 snapshot for later cached-control report joining.
    publish_bytes(RUN/"SELECTED_REFERENCES.json", encoded(selected))
    request = make_scoring_request(bundle, selected, inner, durable_receipts)
    scorer_arguments(request)  # preflight only; matching remains in the scorer leaf
    blob = encoded(request)
    if len(blob) > LIMITS["scorer_request_bytes"]:
        raise ValueError("SCORER_REQUEST_BYTE_BUDGET")
    receipt = dict(worker_dispatch_attempted=False, derived_result_retention_complete=False)
    try:
        with owned_leaf_cache("SCORER", receipt) as cache:
            with deadline(LIMITS["parent_read_seconds"]):
                publish_private_scoring_request(cache, blob)
            dispatch = issue_leaf_dispatch("SCORER", cache, request_sha256=digest(blob))
            receipt["worker_dispatch_attempted"] = True
            try:
                with deadline(LIMITS["scorer_wall_seconds"]):
                    receipt.update(supervise_process(runner_command("--score-worker"), dispatch,
                        cache=cache, log_path=RUN/"SCORER_OUTPUT.txt", wall_seconds=LIMITS["scorer_wall_seconds"],
                        rss_limit=LIMITS["scorer_rss_bytes"], log_limit=LIMITS["case_log_bytes"],
                        parent_rss_limit=LIMITS["coordinator_rss_bytes"]))
            finally:
                if writers_quiescent(receipt):
                    with deadline(LIMITS["parent_read_seconds"]):
                        retained = retain_worker_result(cache, RUN, receipt, scorer=True)
                        exclusive_json(RUN/"SCORER_ARTIFACT_RETENTION.json", retained)
                        receipt["derived_result_retention_complete"] = True
                else:
                    receipt["derived_result_retention_complete"] = False
                    receipt["private_cache_custody"]["status"] = "OPAQUE_RETAINED_WRITER_QUIESCENCE_UNCONFIRMED"
            if not writers_quiescent(receipt):
                raise ValueError("SCORER_WRITER_CUSTODY_UNCONFIRMED_NO_RESULT_READ")
            if receipt["exit_code"] != 0 or receipt["budget_stop"]:
                raise ValueError("SCORER_FAILED_OR_BUDGET_STOP")
            score_blob = bounded_read(RUN/"SCORER_RESULT.json", LIMITS["scorer_result_bytes"],
                retained["artifacts"][0]["sha256"])
            scorer_result_sha = digest(score_blob)
            report = decode(score_blob)
    except BaseException as error:
        receipt["failure_type"] = type(error).__name__
        receipt["failure_reason"] = str(error)[:160]
        raise
    finally:
        exclusive_json(RUN/"SCORER_RECEIPT.json", receipt)
    # Bind the source/runtime AFTER scorer execution, not only before import.
    verify_bundle()
    bounded_read(CATALOG, 8*1024**2, CATALOG_SHA)
    # Enrich without modifying any exact qualified contact scorer field/metric.
    # This new contact comparison cannot assert historical .458 comparability.
    report.update(experiment_schema="GOLD11_MESH_DEVELOPMENT_V11",
                  control_identity="FRESH_V18_KERNEL_V3_ALLOWANCE_V1_BRIDGE_V2_COMPLETION_V1_V7_SCAN_PLANES_SCHEDULE_V1_NATIVE_V1",
                  challenger_identity="FRESH_V18_KERNEL_V3_ALLOWANCE_V1_BRIDGE_V2_COMPLETION_V1_COMPOSITE_SCAN_PLANES_SCHEDULE_V2_NATIVE_V1",
                  config_delta_from_v10=CONFIG_DELTA_FROM_V10,
                  pose_admission_declaration=POSE_ADMISSION_DECLARATION,
                  measurement_policy_declaration=MEASUREMENT_POLICY_DECLARATION,
                  result_interpretation="COMBINED_CONSUMED_DEVELOPMENT_CHANGES_NOT_MECHANISM_ONLY_SPEEDUP_OR_CALIBRATED_ACCURACY",
                  arm_comparison_scope="SCAN_PLANE_LANE_AND_SCHEDULE_ONLY_SAME_V18_ENGINE_NOT_OLD_VS_NEW_ENGINE",
                  comparison_scope_note=("BOTH_ARMS_RUN_THE_SAME_V18_ENGINE_UNDER_THE_SAME_ALLOWANCE_ANY_DIFFERENCE_IS_THE_"
                                         "SCAN_PLANE_LANE_OR_SCHEDULE_PLUS_TIMING_DEPENDENT_CPU_GUARD_STOPS_NOT_AN_ENGINE_"
                                         "OR_ACCURACY_COMPARISON_HISTORICAL_0P4580_NOT_COMPARABLE_NEVER_CLAIMED_BEATEN"),
                  arm_requests=all_arm_requests(), scoring_transport=SCORING_TRANSPORT,
                  projection_bindings=request["projection_bindings"],
                  combined_change_attribution=True, accepted_registration_claim_allowed=False,
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
    global _COORDINATOR_ABORTED
    require_release()
    dispatch = decode(sys.stdin.buffer.read(4096))
    bundle = verify_bundle()
    audit = verify_pre_run_audit(bundle)
    expected = decode(bounded_read(RUN/"COORDINATOR_DISPATCH.json", 4096))
    start = decode(bounded_read(RUN/"STARTED.json", 4096))
    if (set(expected) != {"parent_pid", "token_sha256", "cache", "cache_identity"}
            or type(expected["cache"]) is not str):
        raise ValueError("COORDINATOR_PRIVATE_CACHE_DISPATCH_SCHEMA")
    coordinator_cache = Path(expected["cache"])
    if (cache_identity(coordinator_cache) != expected["cache_identity"]
            or contained(str(coordinator_cache), (str(RUN),))
            or Path.cwd() != coordinator_cache
            or os.environ.get("TMPDIR") != str(coordinator_cache)
            or os.environ.get("HOME") != str(coordinator_cache)):
        raise ValueError("COORDINATOR_PRIVATE_CACHE_IDENTITY_OR_ENVIRONMENT_MISMATCH")
    if (type(dispatch.get("token")) is not str or len(dispatch["token"]) != 64
            or digest(dispatch["token"].encode()) != expected.get("token_sha256")
            or dispatch.get("parent_pid") != os.getppid() or expected.get("parent_pid") != os.getppid()
            or os.getpgrp() != os.getpid() or start.get("bundle_sha256") != digest(encoded(bundle))
            or start.get("audit_sha256") != digest(encoded(audit))
            or start.get("status") != "STARTED_NO_RETRY" or start.get("limits") != LIMITS
            or start.get("arm_requests_sha256") != digest(encoded(all_arm_requests()))):
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
                if receipt.get("source_binding_valid") is not True:
                    raise ValueError("INPUT_BINDING_NOT_ESTABLISHED_ABORT_NO_FURTHER_DISPATCH_OR_SCORING")
                if (not _COORDINATOR_WRITERS_QUIESCENT
                        or receipt.get("worker_dispatch_attempted") is True and not writers_quiescent(receipt)):
                    raise ValueError("LEAF_CUSTODY_UNKNOWN_ABORT_NO_FURTHER_DISPATCH_OR_SCORING")
                if receipt.get("derived_result_retention_complete") is False:
                    raise ValueError("DERIVED_RETENTION_FAILED_ABORT_NO_FURTHER_DISPATCH_OR_SCORING")
                if receipt.get("private_cache_cleanup_complete") is False:
                    raise ValueError("PRIVATE_CACHE_CUSTODY_FAILURE_ABORT_NO_FURTHER_DISPATCH_OR_SCORING")
                coordinator_memory_check()
        if any(row.get("derived_result_retention_complete") is False
               for rows in receipts.values() for row in rows.values()):
            raise ValueError("DERIVED_RESULT_RETENTION_INCOMPLETE_NO_SCORING")
        exclusive_json(RUN/"INFERENCE_TERMINAL.json", dict(case_count=11, arm_count=2,
            terminal_record_count=sum(map(len, receipts.values())), outcomes_sha256=digest(encoded(outcomes)),
            arm_requests_sha256=digest(encoded(all_arm_requests())),
            result_bindings={arm: {case: row.get("result_sha256") for case, row in rows.items()}
                             for arm, rows in receipts.items()}, parent_wall_seconds=time.monotonic()-started))
        score_after_terminal(bundle, outcomes, receipts)
        status = "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT"
    except Exception as error:
        _COORDINATOR_ABORTED = True
        terminal = (terminalize_case_records(bundle, "COORDINATOR_ABORT_NO_RETRY")
            if _COORDINATOR_WRITERS_QUIESCENT else
            [dict(case_id=case["case_token"], arm=arm, durable=False,
                reason="UNCONFIRMED_LEAF_WRITERS_DEFER_TERMINAL_FILL_TO_OUTER")
                for case in bundle["cases"] for arm in ARMS])
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
    evidence.update(schema="GOLD11_OWNED_ARTIFACT_CUSTODY_V11", complete=False, files={}, logs={})
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
    inner_inputs, case_receipts = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    for case in bundle["cases"]:
        for arm in ARMS:
            key = case["case_token"]
            prefix = f"cases/{key}/{arm}/"
            receipt = obj(prefix+"COMPLETED.json", LIMITS["compact_case_bytes"])
            if receipt.get("status") != "TERMINAL" or not is_hex(receipt.get("scoring_input_sha256"), 64):
                raise ValueError("CASE_TERMINAL_SCORING_INPUT_BINDING")
            outcome = obj(prefix+"SCORING_INPUT.json", LIMITS["compact_case_bytes"], receipt["scoring_input_sha256"])
            if not (valid_supervisor_refusal(outcome)
                    or outcome.get("schema") == SCORING_TRANSPORT["wrapper_schema"]):
                raise ValueError("CASE_SCORING_INPUT_SCHEMA")
            if marker_present(RUN/(prefix+"STARTED.json")):
                case_start = obj(prefix+"STARTED.json")
                if (case_start.get("case_id") != key or case_start.get("arm") != arm
                        or case_start.get("limits") != LIMITS or case_start.get("no_resume") is not True
                        or case_start.get("arm_request_sha256") != arm_request_sha256(arm)):
                    raise ValueError("CASE_START_IDENTITY_OR_LIMIT_BINDING")
            elif (receipt.get("inference_started") is not False or receipt.get("source_binding_valid") is not False
                  or outcome.get("status") != "UNAVAILABLE"):
                raise ValueError("STARTED_CASE_RECORD_MISSING")
            artifacts = retained(receipt, prefix)
            result_sha = receipt.get("result_sha256")
            if result_sha is not None and (not is_hex(result_sha, 64) or len(artifacts) != 1
                    or artifacts[0]["path"] != "RESULT.json" or artifacts[0]["sha256"] != result_sha):
                raise ValueError("CASE_SELECTED_RESULT_HASH_BINDING")
            clean = type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0 and receipt.get("budget_stop") is None and "failure_type" not in receipt
            if "exit_code" in receipt and receipt["exit_code"] is not None and type(receipt["exit_code"]) is not int:
                raise ValueError("CASE_EXIT_CODE_SCHEMA")
            if not clean and outcome.get("status") != "UNAVAILABLE":
                raise ValueError("FAILED_CASE_MUST_NOT_SUPPLY_USABLE_SCORE_PROJECTION")
            expected_dispatch = digest(encoded(dict(role="INFERENCE", binding=dict(arm=arm,
                scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"],
                arm_request_sha256=arm_request_sha256(arm)))))
            if (receipt.get("worker_dispatch_attempted") is True
                    and receipt.get("leaf_dispatch_id") != expected_dispatch):
                raise ValueError("CASE_LEAF_DISPATCH_IDENTITY_BINDING")
            if clean:
                if receipt.get("worker_dispatch_attempted") is not True:
                    raise ValueError("CLEAN_CASE_REQUIRES_BOUND_DISPATCH")
                evidence.setdefault("required_leaf_dispatch_ids", []).append(expected_dispatch)
            log(prefix+"OUTPUT.txt", receipt, LIMITS["case_log_bytes"], clean=clean)
            # Independently rebuild from the retained original bytes, never
            # accept mutually consistent stored wrapper/hash claims alone.
            original_result = read(prefix+"RESULT.json", LIMITS["case_result_bytes"], result_sha) if clean else None
            projection_blob = read(prefix+"SCORING_INPUT.json", LIMITS["compact_case_bytes"],
                receipt["scoring_input_sha256"])
            inner_inputs[arm][key] = reconstruct_scoring_input(
                projection_blob, receipt, arm, result_bytes=original_result)
            case_receipts[arm][key] = receipt
            outcomes[arm][key], result_bindings[arm][key] = outcome, result_sha
    terminal = obj("INFERENCE_TERMINAL.json")
    if (any(type(terminal.get(key)) is not int or terminal[key] != count
            for key, count in (("case_count", 11), ("arm_count", 2), ("terminal_record_count", 22)))
            or terminal.get("outcomes_sha256") != digest(encoded(outcomes))
            or terminal.get("result_bindings") != result_bindings
            or terminal.get("arm_requests_sha256") != digest(encoded(all_arm_requests()))):
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
    if (score.get("experiment_schema") != "GOLD11_MESH_DEVELOPMENT_V11"
            or score.get("schema") != "GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V5"
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
            or encoded(score.get("arm_requests")) != encoded(all_arm_requests())
            or encoded(score.get("scoring_transport")) != encoded(SCORING_TRANSPORT)
            or score.get("combined_change_attribution") is not True
            or score.get("accepted_registration_claim_allowed") is not False
            or any(encoded(score.get(key)) != encoded(value) for key, value in original.items())):
        raise ValueError("ORIGINAL_SCORER_OR_ENRICHED_REPORT_BINDING")
    for arm in ARMS:
        lane = score.get(arm)
        if (type(lane) is not dict or type(lane.get("case_dispositions")) is not dict
                or set(lane["case_dispositions"]) != expected_cases
                or type(lane.get("families")) is not dict or set(lane["families"]) != set(families.values())
                or any(type(row) is not dict or row.get("family_id") != families[key]
                       for key, row in lane["case_dispositions"].items())):
            raise ValueError("COMPLETE_CASE_FAMILY_SCORE_LANE_REQUIRED")
    selected = decode(read("SELECTED_REFERENCES.json", 8*1024**2, SELECTED_SHA))
    expected_request = make_scoring_request(bundle, selected, inner_inputs, case_receipts)
    scorer_arguments(expected_request)
    if (score.get("scoring_request_sha256") != digest(encoded(expected_request))
            or encoded(score.get("projection_bindings")) != encoded(expected_request["projection_bindings"])):
        raise ValueError("SCORER_REQUEST_NOT_RECONSTRUCTIBLE_FROM_OWNED_RESULT_BYTES")
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
                cad_sha256=case["cad_sha256"], arm_request_sha256=arm_request_sha256(arm))))
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
    for name in ("leaf_dispatch",):
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
    result = dict(schema="COMPLETE_CAMPAIGN_TREE_CUSTODY_V11", complete=False, entries={},
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


def prestart_tree_admission():
    """Exact named root/source metadata only; unknown contents are never read."""
    require_release()
    expected = {"": {"FROZEN.json": "file", "PRE_RUN_AUDIT.json": "file", "source": "directory"},
        "source": {}}
    for name in SOURCES:
        parts = ("source", *Path(name).parts)
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            kind = "file" if index == len(parts)-1 else "directory"
            prior = expected.setdefault(parent, {}).setdefault(parts[index], kind)
            if prior != kind:
                raise ValueError("V11_PRESTART_SOURCE_TREE_DECLARATION_CONFLICT")
            if kind == "directory":
                expected.setdefault("/".join(parts[:index+1]), {})
    for relative, names in sorted(expected.items()):
        fd = _open_tree_directory(relative)
        try:
            before = os.fstat(fd)
            actual, overflow = _tree_names(fd, len(names))
            if overflow is not None or actual != sorted(names):
                raise ValueError("V11_PRESTART_UNDECLARED_MISSING_OR_SPENT_TREE_NAME")
            for name, kind in names.items():
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                valid = stat.S_ISDIR(info.st_mode) if kind == "directory" else stat.S_ISREG(info.st_mode)
                if not valid:
                    raise ValueError("V11_PRESTART_TREE_TYPE")
            after_names, after_overflow = _tree_names(fd, len(names))
            if (after_names != actual or after_overflow is not None
                    or _tree_fingerprint(before) != _tree_fingerprint(os.fstat(fd))):
                raise ValueError("V11_PRESTART_TREE_DIRECTORY_CHANGED")
        finally:
            os.close(fd)


def execute():
    require_release()
    # Concrete source/campaign timings remain unset until a new reviewed freeze.
    # No automatic coordinator-cache cleanup may run after an expired alarm.
    started_attempted = False
    try:
        with deadline(LIMITS["parent_read_seconds"]):
            bundle = verify_bundle()
            audit = verify_pre_run_audit(bundle)
            prestart_tree_admission()
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
                arm_requests_sha256=digest(encoded(all_arm_requests())),
                timing_scope="SOURCE_PREFLIGHT_THEN_CAMPAIGN_INCLUDING_FROZEN_TERMINAL_RESERVE")
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
    coordinator_cache_custody = None
    writers_stopped = False
    status = "TERMINAL_GLOBAL_BUDGET_OR_COORDINATOR_FAILURE"
    def remaining():
        return max(.001, LIMITS["total_wall_seconds"]-(time.monotonic()-started))
    def failed(phase, error):
        failures.append(dict(phase=phase, error_type=type(error).__name__,
            reason=str(error)[:160] if isinstance(error, (ValueError, TimeoutError)) else type(error).__name__))
    try:
        with deadline(max(.001, remaining()-LIMITS["terminalization_reserve_seconds"])):
            cache, coordinator_cache_identity = fresh_private_cache("gold11-v11-coordinator-")
            coordinator_cache_custody = dict(path=str(cache), identity=coordinator_cache_identity,
                status="OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT",
                contents_enumerated_or_read=False)
            token = secrets.token_hex(32)
            dispatch = dict(parent_pid=os.getpid(), token_sha256=digest(token.encode()),
                cache=str(cache), cache_identity=coordinator_cache_identity)
            control_bindings["COORDINATOR_DISPATCH.json"] = digest(encoded(dispatch)+b"\n")
            exclusive_json(RUN/"COORDINATOR_DISPATCH.json", dispatch)
            if cache_identity(cache) != coordinator_cache_identity:
                raise ValueError("COORDINATOR_PRIVATE_CACHE_CHANGED_BEFORE_DISPATCH")
            receipt = supervise_process(runner_command("--internal-coordinator"),
                dict(token=token, parent_pid=os.getpid()), cache=cache.resolve(),
                log_path=RUN/"COORDINATOR_OUTPUT.txt",
                wall_seconds=max(.001, remaining()-LIMITS["terminalization_reserve_seconds"]),
                rss_limit=LIMITS["global_tree_rss_bytes"], log_limit=LIMITS["coordinator_log_bytes"], new_session=True)
    except BaseException as error:
        failed("COORDINATOR_SUPERVISION", error)
    try:
        with deadline(remaining()):
            # Do not terminal-fill or inventory while an unconfirmed group may
            # still write. A killed but reaped/empty group may receive fallback
            # terminals; it still cannot become a successful comparison.
            writers_stopped = writers_quiescent(receipt, new_session=True)
            if not writers_stopped:
                terminal = [dict(case_id=case["case_token"], arm=arm, durable=False,
                    reason="WRITER_QUIESCENCE_UNCONFIRMED_NO_TERMINAL_FILL")
                    for case in bundle["cases"] for arm in ARMS]
                raise ValueError("WRITER_QUIESCENCE_REQUIRED_FOR_TERMINAL_FILL")
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
                failed("FINAL_SOURCE_AUDIT_AFTER_QUIESCENCE", error)
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
                coordinator_private_cache_custody=coordinator_cache_custody,
                coordinator_memory_observation="DARWIN_RU_MAXRSS_LIFETIME_HIGHWATER_NOT_CURRENT_RSS",
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
    require_release()
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
