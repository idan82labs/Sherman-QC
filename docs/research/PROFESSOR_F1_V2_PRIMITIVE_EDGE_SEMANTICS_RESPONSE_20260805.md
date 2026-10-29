NEXT_TRANCHE: V2_PRIMITIVE_REPLACEMENT_PROBE
SELECTED_UPSTREAM_HYPOTHESIS: EDGE_SEMANTICS

1. Decision and causal rationale

OBSERVED. The locked V2 lane is valid as an experiment and is a formal empirical STOP: all 25 pre-gold criteria passed, all 37 jobs and deterministic rebuilds succeeded, the designated replays were byte-identical, and post-lock attribution recovered 0/67 stage-1 objects, 0 independent groups, and certified 0/33 frozen A0 false predictions. Sixty-one of the 67 primary rows have no matching finite component; the remaining six fail unchanged continuity, side-bin, or branching gates. 

01_LOCKED_STOP_REPORT

OBSERVED. Of 35,813 V2 components, 15,609 are zero-span and every one has exactly one physical support parent. The more diagnostic subset is the 4,788 components that already contain multiple modes, both represented sides, and a valid same-parent CROSS_MODE_LOCAL rung, but still have no second physical parent position. 

01_LOCKED_STOP_REPORT

OBSERVED. V2 constructs CROSS_MODE_LOCAL edges only between mode children of the same parent. In contrast, cross-parent continuity is supplied by SAME_SURFACE edges, for which all geometrically eligible candidates are first generated and then reduced to mutual top-eight neighbors per mode. 

06_patch_graph_latent_decomposi…

 

06_patch_graph_latent_decomposi…

OBSERVED. LPF component construction subsequently accepts only the surviving inherited edges. It does not recreate a geometrically eligible edge that was removed by the mutual-top-eight operation. 

08_f1_local_plane_fragment_probe

INFERENCE. The smallest discriminating hypothesis is therefore:

Existing V2 parents and projective-normal modes contain enough local two-sided transition evidence, but the mutual-top-eight SAME_SURFACE edge semantics discard enough cross-parent continuation to fragment that evidence into one-parent or discontinuous components.

INFERENCE. The replacement will make one categorical change only: replace the mutual-top-eight SAME_SURFACE graph with the complete bounded graph of all mode pairs satisfying the already frozen 15-degree and bilateral surface-gap eligibility predicates.

INFERENCE. This is not a sweep from 8 to 12, 16, or another degree. No k remains. It is one preregistered semantic alternative: MUTUAL_TOP8 versus COMPLETE_ELIGIBLE.

INFERENCE. All spatial parents, mode children, CROSS_MODE_LOCAL rungs, local-plane fitting, orphan handling, line construction, binning, rejection order, and hard geometry gates remain frozen.

INFERENCE. Any newly finite axial span must still arise from at least two existing physical parent centroids connected through the replacement graph. A one-parent component remains a point. The probe therefore does not perform the prohibited endpoint-only repair.

UNKNOWN. The compact professor package does not contain the complete 37 per-case parent and mode tables or the omitted locked bundle, so the exact number of edges removed by mutual-top-eight cannot be determined from this upload alone. That uncertainty is handled by a mandatory gold-free intervention check before lock, not by assumption.

2. Why this hypothesis is smaller than the other three

INFERENCE. EDGE_SEMANTICS is smaller than VOXEL_SCALE_ALIASING because it leaves all 49,212 frozen parent records, their centroids, voxel keys, masses, mode partitions, and IDs unchanged. A voxel experiment would rebuild the physical support substrate and would change several upstream variables simultaneously.

INFERENCE. EDGE_SEMANTICS is smaller than PROJECTIVE_NORMAL_MODES because it leaves all 60,697 frozen mode records and all 20,846 inherited local rungs unchanged. The 4,788 two-sided same-parent components are direct evidence that the existing mode representation can, in at least a substantial population, preserve two distinct surface orientations and a transition relation.

INFERENCE. EDGE_SEMANTICS is smaller than EXPLICIT_TRANSITION_STRIP because it consumes no new point-derived strip, corridor, trace, ridge, or finite-span primitive. An explicit strip would add a new evidence class; the selected probe instead asks whether evidence already in the frozen tables was disconnected by graph pruning.

INFERENCE. The result is more discriminating because a GO would show that existing parents, modes, and local rungs were sufficient once cross-parent continuity was completed. A STOP would eliminate the exact mutual-top-eight subhypothesis while leaving voxel aliasing, mode construction, and absent explicit strip evidence as the remaining upstream explanations.

3. Exact experiment identities

INFERENCE. Use these exact identities:

Python
Run
PRIMITIVE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_complete_same_surface_edge_probe_v1"

BATCH_SCHEMA_VERSION = 1
BATCH_KIND = "f1_v2_complete_same_surface_edge_probe_batch_v1"

REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = (
    "f1_v2_complete_same_surface_edge_probe_replay_attestation_v1"
)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_KIND = "f1_v2_complete_same_surface_edge_probe_bundle_v1"

LOCK_ATTESTATION_KIND = (
    "f1_v2_complete_same_surface_edge_probe_lock_attestation_v1"
)

POSTLOCK_SCHEMA_VERSION = 1
POSTLOCK_KIND = (
    "postlock_gold_v2_complete_same_surface_edge_attribution_v1"
)

INPUT_PROJECTION_PROFILE = "V2_COMPLETE_EDGE_INPUT_PROJECTION_V1"

EDGE_RULE_ID = (
    "COMPLETE_ELIGIBLE_SAME_SURFACE_NO_MUTUAL_TOPK_V1"
)

CONTROL_EDGE_RULE_ID = (
    "TRANSITION_ATOM_V2_MUTUAL_TOP8_SAME_SURFACE"
)

COUNTERFACTUAL_PROFILE = (
    "DELETE_ADDED_SAME_SURFACE_EDGES_RESTORES_LPF_V2_V1"
)

POSTLOCK_MATCHING_PROFILE = (
    "V2_COMPLETE_EDGE_POSTLOCK_EXACT_MATCH_V1"
)

INFERENCE. No V2 LPF probe, manifest, replay, bundle, lock, or post-lock artifact may satisfy a replacement-probe validator merely because its downstream field shapes resemble the new lane.

INFERENCE. The old LPF files and their hashes remain frozen controls. They are never overwritten or relabeled as replacement artifacts.

4. Exact inference input contract

INFERENCE. The per-case replacement core receives exactly the same deep-copied allowlisted projection as LPF V2:

Python
Run
ALLOWED_EDGE_PROBE_INPUT_FIELDS = (
    "support_representation_profile",
    "source_point_serialization",

    "support_parent_table",
    "support_mode_table",
    "support_mode_edges",
    "plane_table",

    "transition_parent_table_sha256",
    "transition_mode_table_sha256",
    "transition_mode_edge_table_sha256",

    "transition_local_spacing_mm",
    "transition_atom_caps",
    "working_point_budget",
    "transient_sampled_point_count",
)

INFERENCE. The projection is constructed immediately after parsing:

Python
Run
def project_edge_probe_input(
    raw_pool: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        field
        for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS
        if field not in raw_pool
    ]
    if missing:
        raise ValueError(
            f"edge-probe input projection missing fields: {missing}"
        )

    projected = {
        field: deepcopy(raw_pool[field])
        for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS
    }

    if tuple(projected) != ALLOWED_EDGE_PROBE_INPUT_FIELDS:
        raise AssertionError("edge-probe projection order mismatch")

    return projected

INFERENCE. The core rejects any extra or missing top-level field rather than ignoring it.

INFERENCE. Validation must recompute and prove:

support_representation_profile == "TRANSITION_ATOM_V2"
source_point_serialization is false
transition_atom_caps equals the frozen V2 cap dictionary
working_point_budget == 200000
0 <= transient_sampled_point_count <= 200000

parent IDs, mode IDs, and edge IDs are unique
parent voxel keys are unique
all parent voxel levels are identical within a case
all parent voxel sizes are identical within a case
every parent mode_child_ids set exactly matches the mode table
every mode references one existing parent
every current edge references two existing modes
every current edge kind is SAME_SURFACE or CROSS_MODE_LOCAL

claimed parent, mode, and edge hashes recompute exactly
the global-plane table hash recomputes exactly
all inherited V2 caps hold

INFERENCE. The scan and frozen A0 prediction are opened only as byte streams for SHA-256. Neither is parsed by the inference core.

INFERENCE. The old LPF control output is not an inference input. It is regenerated separately from the same projected input and compared with the frozen batch hash.

5. Forbidden inputs and operations

INFERENCE. The recursive forbidden-key set remains the LPF V2 set and is extended explicitly to cover alternative strip or learned evidence:

Python
Run
FORBIDDEN_KEYS = {
    # Source samples or membership.
    "point_indices",
    "source_indices",
    "source_point_indices",
    "vertex_indices",
    "points",
    "source_points",
    "point_coordinates",
    "coordinates",
    "bbox_points",

    # Identity or truth.
    "filename",
    "scan_path",
    "part_id",
    "part_key",
    "family_id",
    "split_group_id",
    "gold",
    "manual_truth",
    "expected_count",
    "requested_count",
    "drawing",
    "cad",
    "cad_metadata",

    # Prediction, ownership, optimization, or learning.
    "atom_owners",
    "parent_owners",
    "mode_side_assignments",
    "ownership",
    "solver",
    "objective",
    "reduced_cost",
    "potential",
    "fitted_potential",
    "learned_score",
    "model_score",
    "birth_columns",
    "emitted_births",

    # Alternative new evidence classes.
    "transition_strip_table",
    "support_strip",
    "corridor",
    "evidence_trace",
    "ridge_candidate",
    "junction_path",
}

INFERENCE. A1 ownership, emitted births, candidate emission, learning, fitted potentials, expected-count logic, family-aware routing, threshold tuning, and runtime/main promotion remain false and prohibited.

6. Exact replacement construction
6.1 Complete SAME_SURFACE eligibility table

OBSERVED. The frozen V2 edge generator already defines a full candidate predicate before mutual-top-eight pruning: distinct neighboring parent voxels, projective-normal angle at most 15 degrees, and bilateral normal-direction gap at most tau_surface. 

06_patch_graph_latent_decomposi…

INFERENCE. Recompute that complete candidate set from the frozen compact tables, using this exact pseudocode:

Python
Run
def build_complete_same_surface_edges(projected_input):
    parents = validate_and_index_parents(
        projected_input["support_parent_table"]
    )
    modes = validate_and_index_modes(
        projected_input["support_mode_table"],
        parents=parents,
    )
    current_edges = validate_current_edges(
        projected_input["support_mode_edges"],
        modes=modes,
    )

    spacing = binary64(
        projected_input["transition_local_spacing_mm"]
    )
    tau_surface = min(
        3.0,
        max(1.25, 1.5 * spacing),
    )

    voxel_levels = {
        int(parent["voxel_level"])
        for parent in parents.values()
    }
    voxel_sizes = {
        binary64(parent["voxel_size_mm"])
        for parent in parents.values()
    }

    if len(voxel_levels) != 1:
        raise ValueError("parent voxel level is not case-global")
    if len(voxel_sizes) != 1:
        raise ValueError("parent voxel size is not case-global")

    parent_voxel_size = only(voxel_sizes)

    parent_id_by_key = {
        tuple(int(v) for v in parent["voxel_key"]):
        parent_id
        for parent_id, parent in parents.items()
    }

    modes_by_parent = group_modes_by_parent(modes)

    complete = {}

    for left_parent_id in sorted(parents):
        left_parent = parents[left_parent_id]
        left_key = tuple(left_parent["voxel_key"])

        for right_key in sorted(neighbor_26_keys(left_key)):
            right_parent_id = parent_id_by_key.get(right_key)

            if right_parent_id is None:
                continue

            # Count each physical parent pair exactly once.
            if right_parent_id <= left_parent_id:
                continue

            adjacent_parent_pair_count += 1

            for left_mode_id in sorted(
                modes_by_parent[left_parent_id]
            ):
                for right_mode_id in sorted(
                    modes_by_parent[right_parent_id]
                ):
                    eligible_mode_pair_evaluation_count += 1

                    left_mode = modes[left_mode_id]
                    right_mode = modes[right_mode_id]

                    angle_deg = projective_normal_angle_deg(
                        left_mode["normal_projective"],
                        right_mode["normal_projective"],
                    )

                    # Strict binary64 >, no epsilon.
                    # Equality at 15 degrees is eligible.
                    if angle_deg > 15.0:
                        continue

                    delta = (
                        vector(right_mode["centroid"])
                        - vector(left_mode["centroid"])
                    )

                    left_normal = unit(
                        left_mode["normal_projective"]
                    )
                    right_normal = unit(
                        right_mode["normal_projective"]
                    )

                    gap_mm = max(
                        abs(dot(left_normal, delta)),
                        abs(dot(right_normal, delta)),
                    )

                    # Strict binary64 >, no epsilon.
                    # Equality at tau_surface is eligible.
                    if gap_mm > tau_surface:
                        continue

                    mode_pair = tuple(sorted((
                        left_mode_id,
                        right_mode_id,
                    )))

                    edge_id = (
                        "E_"
                        + sha256_json([
                            list(mode_pair),
                            "SAME_SURFACE",
                        ])[:16]
                    )

                    record = {
                        "mode_edge_id": edge_id,
                        "mode_ids": list(mode_pair),
                        "edge_kind": "SAME_SURFACE",
                        "spatial_residual": (
                            norm(delta) / parent_voxel_size
                        ),
                        "normal_residual": (
                            angle_deg / 15.0
                        ),
                        "plane_residual": (
                            gap_mm / tau_surface
                        ),
                    }

                    if edge_id in complete:
                        if complete[edge_id] != record:
                            raise ValueError(
                                "same-surface edge-ID collision"
                            )
                    else:
                        complete[edge_id] = record

    complete_table = sorted(
        complete.values(),
        key=lambda row: row["mode_edge_id"],
    )

    return complete_table

INFERENCE. There is no neighbor ranking, no mutual-neighbor condition, no degree limit, no output truncation, and no retry at another scale.

6.2 Exact current-edge subset proof

INFERENCE. Partition the inherited edge table into:

Python
Run
current_same_surface_edges = sorted(
    edge for edge in current_edges
    if edge["edge_kind"] == "SAME_SURFACE"
)

inherited_cross_mode_edges = sorted(
    edge for edge in current_edges
    if edge["edge_kind"] == "CROSS_MODE_LOCAL"
)

INFERENCE. Every inherited SAME_SURFACE record must occur in the complete table with the same ID, mode IDs, edge kind, and exact serialized residual values.

INFERENCE. The complete table is invalid before downstream construction if even one inherited SAME_SURFACE edge is absent or differs.

INFERENCE. Every inherited CROSS_MODE_LOCAL record is copied byte-for-byte into the effective edge table. No cross-parent rung and no new rung is generated.

INFERENCE. The effective table is:

Python
Run
effective_mode_edge_table = sorted(
    [
        *complete_same_surface_edge_table,
        *inherited_cross_mode_edge_table,
    ],
    key=lambda row: row["mode_edge_id"],
)

INFERENCE. Duplicate IDs or duplicate (mode_ids, edge_kind) records fail the case closed.

6.3 Downstream geometry

INFERENCE. Run the same deterministic LPF construction on the effective edge table, retaining all of these exact rules:

fragment normal:
  point-count-weighted projective moment

fragment offset:
  point-count-weighted lower median

fragment fit:
  weighted P90(
    abs(n dot mode_centroid + d)
    + mode_surface_fit_p90_mm
  )

fragment union order:
  normal_residual,
  plane_residual,
  spatial_residual,
  mode_edge_id

fragment normal diameter:
  <= 15 degrees

plane fit tolerance:
  min(3.0, max(1.25, 1.5 * spacing))

orphan rule:
  INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1

pair seed:
  inherited CROSS_MODE_LOCAL edges only

pair angle:
  20–90 projective degrees

line tube tolerance:
  min(3.0, max(1.5, 1.25 * spacing))

axial bin width:
  max(3.0, 2.0 * spacing)

physical support:
  unique frozen parent IDs and their frozen centroids

INFERENCE. The unchanged component gates remain, in this order:

Python
Run
REJECTION_ORDER = (
    ("minimum_parent_count", "COMPONENT_TOO_SMALL"),
    ("minimum_axial_span", "INSUFFICIENT_AXIAL_SPAN"),
    ("minimum_occupied_bins", "INSUFFICIENT_OCCUPIED_BINS"),
    ("minimum_left_side_bins", "INSUFFICIENT_LEFT_SIDE_BINS"),
    ("minimum_right_side_bins", "INSUFFICIENT_RIGHT_SIDE_BINS"),
    (
        "minimum_transition_rung_bins",
        "INSUFFICIENT_TRANSITION_RUNG_BINS",
    ),
    (
        "minimum_consecutive_occupied_run",
        "INSUFFICIENT_CONSECUTIVE_BIN_RUN",
    ),
    ("maximum_empty_bin_run", "EXCESSIVE_EMPTY_BIN_RUN"),
    ("no_internal_branching", "INTERNAL_BRANCHING"),
    ("tls_axis", "TLS_AXIS_MISMATCH"),
    ("line_residual", "LINE_RESIDUAL_EXCEEDED"),
    (
        "transition_certificate_complete",
        "MODE_OVERFLOW_CERTIFICATE_DISABLED",
    ),
    (
        "certified_three_bin_window",
        "NO_CERTIFIED_THREE_BIN_WINDOW",
    ),
)

INFERENCE. Their exact thresholds remain:

minimum parent count                 3
minimum axial span                   2 bins
minimum occupied bins                3
minimum left-side bins               3
minimum right-side bins              3
minimum transition-rung bins         2
minimum consecutive occupied run     3
maximum empty-bin run                1
maximum internal branches            0
maximum TLS-axis mismatch             4 degrees
certified three-bin windows          at least 1

INFERENCE. The implementation may index edges by fragment and fragment pair to avoid repeatedly scanning the complete table, but the indexed enumeration must be proven byte-equivalent to brute-force restricted-edge enumeration on bounded fixtures.

6.4 No endpoint repair

INFERENCE. Component endpoints remain:

Python
Run
parent_projection = dot(
    frozen_parent_centroid,
    pair_intersection_axis,
)

lower = min(parent_projection values)
upper = max(parent_projection values)

endpoints = (
    pair_anchor + lower * pair_axis,
    pair_anchor + upper * pair_axis,
)

INFERENCE. For every component with support_parent_count == 1, validation requires lower == upper before rounding and identical serialized endpoints afterward.

INFERENCE. one_parent_finite_span_count must equal zero in every case and in the aggregate.

7. Canonical IDs and hashes
7.1 Edge IDs

INFERENCE. Complete SAME_SURFACE edges reuse the exact inherited V2 edge identity:

Python
Run
mode_edge_id = (
    "E_"
    + sha256_json([
        sorted_mode_ids,
        "SAME_SURFACE",
    ])[:16]
)

INFERENCE. Reusing the identity makes the current-edge subset proof exact rather than heuristic.

7.2 Replacement downstream IDs

INFERENCE. Replacement downstream IDs are profile-separated:

Python
Run
local_plane_fragment_id = (
    "VEF_"
    + sha256_json([
        PRIMITIVE_PROFILE,
        sorted_full_member_mode_ids,
    ])[:16]
)

local_plane_orphan_id = (
    "VEO_"
    + sha256_json([
        PRIMITIVE_PROFILE,
        "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1",
        mode_atom_id,
        parent_atom_id,
    ])[:16]
)

local_plane_pair_id = (
    "VEP_"
    + sha256_json(
        sorted_fragment_ids
    )[:16]
)

local_transition_component_id = (
    "VEC_"
    + sha256_json([
        local_plane_pair_id,
        sorted_parent_ids,
        sorted_mode_ids,
    ])[:16]
)
7.3 Canonical serialization

INFERENCE. All IDs and hashes use:

Python
Run
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")

INFERENCE. Every table is sorted by its canonical ID before hashing.

INFERENCE. Every per-record SHA-256 is computed over the complete record excluding only its own hash field.

INFERENCE. No rounded value is used in an eligibility comparison. Downstream serialized rounding remains exactly:

normal / axis / tangent                 12 places
centroid / offset / endpoints / extent  6 places
dimensionless diagnostic residual       8 places
8. Exact inference-safe records
8.1 Complete edge table

INFERENCE. Each complete_same_surface_edge_table record contains exactly:

Python
Run
COMPLETE_EDGE_FIELDS = {
    "mode_edge_id",
    "mode_ids",
    "edge_kind",
    "spatial_residual",
    "normal_residual",
    "plane_residual",
}
8.2 Added-edge delta table

INFERENCE. same_surface_edge_delta_table contains exactly the complete-table edges absent from the inherited edge table.

INFERENCE. Each delta record contains exactly:

Python
Run
EDGE_DELTA_FIELDS = {
    "mode_edge_id",
    "mode_ids",

    # Aligned with mode_ids, not independently sorted.
    "parent_atom_ids",
    "parent_voxel_keys",

    "spatial_residual",
    "normal_residual",
    "plane_residual",

    "delta_kind",
    "edge_delta_sha256",
}

INFERENCE. delta_kind is always:

ADDED_BY_COMPLETE_ELIGIBILITY
8.3 Edge-semantics attestation

INFERENCE. Each per-case edge_semantics_attestation contains exactly:

Python
Run
EDGE_SEMANTICS_ATTESTATION_FIELDS = {
    "edge_rule_id",
    "control_edge_rule_id",

    "same_surface_angle_deg",
    "surface_gap_tolerance_mm",
    "parent_voxel_level",
    "parent_voxel_size_mm",

    "adjacent_parent_pair_count",
    "eligible_mode_pair_evaluation_count",

    "current_same_surface_edge_count",
    "current_cross_mode_edge_count",
    "complete_same_surface_edge_count",
    "added_same_surface_edge_count",
    "retained_current_same_surface_edge_count",
    "missing_current_same_surface_edge_count",

    "current_same_surface_edge_ids_sha256",
    "current_cross_mode_edge_table_sha256",
    "complete_same_surface_edge_table_sha256",
    "added_same_surface_edge_ids_sha256",
    "effective_mode_edge_table_sha256",

    "all_current_same_surface_edges_reproduced",
    "current_same_surface_records_byte_identical",
    "cross_mode_edges_byte_identical",

    "all_added_edges_connect_distinct_parents",
    "all_added_edges_connect_26_neighbor_voxels",
    "no_neighbor_degree_truncation",
    "strict_binary64_comparisons",

    "edge_semantics_attestation_sha256",
}
8.4 Fragment-level causal attribution

INFERENCE. Log every SAME_SURFACE edge that actually merges two union-find roots. Redundant edges that connect an already unified root are not accepted union edges.

INFERENCE. Each fragment_edge_attribution_table record contains exactly:

Python
Run
FRAGMENT_EDGE_ATTRIBUTION_FIELDS = {
    "local_plane_fragment_id",
    "member_mode_ids_sha256",

    "accepted_same_surface_union_edge_count",
    "accepted_current_same_surface_union_edge_count",
    "accepted_added_same_surface_union_edge_count",

    "accepted_added_same_surface_union_edge_ids_sha256",
    "witness_accepted_added_same_surface_union_edge_ids",
    "witness_ids_truncated",

    "edge_semantics_dependent",
    "fragment_edge_attribution_sha256",
}

INFERENCE. edge_semantics_dependent is true exactly when at least one added edge caused an accepted union used by that fragment.

8.5 Component-level causal attribution

INFERENCE. Each component_edge_attribution_table record contains exactly:

Python
Run
COMPONENT_EDGE_ATTRIBUTION_FIELDS = {
    "local_transition_component_id",
    "local_plane_pair_id",
    "fragment_ids",

    "component_added_same_surface_edge_count",
    "component_added_same_surface_edge_ids_sha256",

    "incident_fragment_added_union_edge_count",
    "incident_fragment_added_union_edge_ids_sha256",

    "causal_added_same_surface_edge_count",
    "causal_added_same_surface_edge_ids_sha256",

    "witness_causal_added_same_surface_edge_ids",
    "witness_ids_truncated",

    "edge_semantics_dependent",
    "geometry_certificate_valid",

    "component_edge_attribution_sha256",
}

INFERENCE. The complete causal edge set is the union of:

added SAME_SURFACE edges contained in the component graph; and

accepted added union edges used to construct either incident fragment.

INFERENCE. A component is edge_semantics_dependent exactly when that complete causal set is nonempty.

8.6 Reused V2 records

INFERENCE. The fragment, orphan, global-plane coverage, pair, global-pair comparison, component, geometry-gate, input-contract, and orphan-role field sets remain exactly those of LPF V2. Only their profile-separated IDs, hashes, edge counts, and geometry produced by the replacement table can change. 

08_f1_local_plane_fragment_probe

 

08_f1_local_plane_fragment_probe

8.7 Exact top-level per-case schema

INFERENCE. Each canonical per-case probe contains exactly:

Python
Run
TOP_LEVEL_FIELDS = {
    "schema_version",
    "kind",
    "profile",

    "research_only",
    "inference_safe",
    "prediction_mutated",
    "birth_columns_enabled",
    "ownership_consumed",
    "solver_consumed",
    "truth_inputs_consumed",
    "source_points_consumed",
    "source_points_serialized",

    "scan_sha256",
    "candidate_pool_sha256",
    "a0_prediction_sha256",
    "frozen_control_probe_sha256",

    "input_table_hashes",
    "replacement_edge_table_hashes",
    "output_table_hashes",

    "engineering_profile_sha256",
    "inference_semantic_sha256",

    "input_contract_attestation",
    "edge_semantics_attestation",

    "thresholds",
    "hard_caps",
    "case_counters",

    "complete_same_surface_edge_table",
    "same_surface_edge_delta_table",

    "fragment_edge_attribution_table",
    "component_edge_attribution_table",

    "local_plane_fragment_table",
    "local_plane_orphan_table",
    "local_plane_pair_records",
}
8.8 Output hashes

INFERENCE. output_table_hashes contains exactly:

Python
Run
{
    "complete_same_surface_edge_table_sha256": "...",
    "same_surface_edge_delta_table_sha256": "...",
    "fragment_edge_attribution_table_sha256": "...",
    "component_edge_attribution_table_sha256": "...",
    "local_plane_fragment_table_sha256": "...",
    "local_plane_orphan_table_sha256": "...",
    "local_plane_pair_records_sha256": "...",
}

INFERENCE. The per-case semantic hash is:

Python
Run
inference_semantic_sha256 = sha256_json({
    "profile": PRIMITIVE_PROFILE,
    "engineering_profile_sha256":
        engineering_profile_sha256,
    "projected_input_sha256":
        input_contract_attestation[
            "projected_input_sha256"
        ],
    "frozen_control_probe_sha256":
        frozen_control_probe_sha256,
    "edge_semantics_attestation_sha256":
        edge_semantics_attestation[
            "edge_semantics_attestation_sha256"
        ],
    "output_table_hashes":
        output_table_hashes,
})
9. Hard caps

INFERENCE. Preserve all inherited and LPF caps and add these mathematically bounded edge caps:

Python
Run
MAX_INPUT_WORKING_POINTS = 200_000

MAX_TRANSITION_PARENTS = 4_000
MAX_TRANSITION_MODES = 16_000
MAX_MODES_PER_PARENT = 4
MAX_PARENT_PLUS_MODE_RECORDS = 20_000
MAX_INHERITED_MODE_EDGES = 200_000

# 4,000 parents × 26 neighboring cells / 2.
MAX_ADJACENT_PARENT_PAIRS = 52_000

# At most 4 modes on each side of a parent pair.
MAX_COMPLETE_EDGE_EVALUATIONS = 52_000 * 16
# 832,000
MAX_COMPLETE_EDGE_EVALUATIONS = 832_000

MAX_COMPLETE_SAME_SURFACE_EDGES = 832_000

# At most C(4,2) local rungs per physical parent.
MAX_INHERITED_CROSS_MODE_EDGES = 4_000 * 6
# 24,000
MAX_INHERITED_CROSS_MODE_EDGES = 24_000

MAX_EFFECTIVE_MODE_EDGES = (
    MAX_COMPLETE_SAME_SURFACE_EDGES
    + MAX_INHERITED_CROSS_MODE_EDGES
)
# 856,000

MAX_LOCAL_PLANE_FRAGMENTS = 16_000
MAX_LOCAL_PLANE_ORPHANS = 16_000
MAX_FRAGMENT_PLUS_ORPHAN_RECORDS = 16_000

MAX_LOCAL_PLANE_PAIRS = 32_768
MAX_LOCAL_TRANSITION_COMPONENTS = 65_536
MAX_COMPONENTS_PER_LOCAL_PAIR = 4_096

MAX_EDGE_ID_WITNESSES = 8
MAX_FRAGMENT_ID_WITNESSES = 8
MAX_COMPONENT_ID_WITNESSES = 8

MAX_CONTROL_CERTIFICATES_PER_CASE = 32

PEAK_RSS_LIMIT_BYTES = 16 * 1024**3

INFERENCE. Structural tables are never truncated. A cap breach aborts the case and causes pre-lock STOP.

INFERENCE. Only explicitly labeled witness arrays may be truncated. Every witness array retains the SHA-256 of the complete untruncated ID set.

10. Gold-free counterfactual and preregistered falsifier
10.1 Frozen-control reproduction

INFERENCE. Before accepting a replacement case, regenerate LOCAL_PLANE_FRAGMENT_PROBE_V2 from the original projected input using the frozen control core.

INFERENCE. Its canonical SHA-256 must equal the exact per-case probe_sha256 in the frozen batch manifest.

INFERENCE. Across all 37 controls, the following aggregate census must reproduce exactly:

13,413 fragments
39 orphans
17,034 pairs
35,813 components
31 certificates
15,609 zero-span components

INFERENCE. The certificate distribution must also reproduce exactly:

C017  2
C018  3
C019  1
C022 11
C032  1
C035  8
C036  5

INFERENCE. A mismatch is an implementation or environment failure. Gold remains unopened.

10.2 Exact edge-deletion reference

INFERENCE. Let:

Python
Run
added_edge_ids = (
    complete_same_surface_edge_ids
    - current_same_surface_edge_ids
)

INFERENCE. Delete exactly added_edge_ids from the replacement effective table.

INFERENCE. The resulting edge table must canonical-equal the original inherited support_mode_edges table, including the complete CROSS_MODE_LOCAL table.

INFERENCE. The downstream control build from that deletion table must reproduce the frozen control probe hash.

10.3 One-parent invariant

INFERENCE. Every replacement component with one physical support parent must remain zero-span.

INFERENCE. Any one-parent finite component falsifies the implementation before lock because it would imply invented extent.

10.4 Intervention-exercise criteria

INFERENCE. All three of the following must be true before gold:

aggregate added_same_surface_edge_count > 0

at least one of the frozen 4,788
multi-mode / same-parent / cross-rung / both-side
zero-span components is contained in a replacement component
with at least two physical parents and at least one causal added edge

at least one valid replacement certificate is
edge-semantics-dependent and has no line-equivalent
frozen control certificate

INFERENCE. These are identifiability requirements, not bend-count targets. They prove that the selected intervention is present, reaches the locked failure population, and creates at least one novel testable certificate.

INFERENCE. Failure of any one causes a pre-gold STOP. Opening gold when the intervention is absent or produces no novel certificate would be uninformative.

10.5 Counterfactual falsifier

INFERENCE. The preregistered counterfactual falsifier is:

Delete all added SAME_SURFACE edges. Every credited recovery must disappear, and the complete frozen LPF V2 control hashes and census must be restored exactly.

INFERENCE. A purported recovered object is not attributable if either:

its replacement certificate has no causal added edge; or

a frozen control certificate matches the same line under the unchanged line contract.

INFERENCE. Such a row is excluded from the recovery numerator even if the replacement certificate is otherwise valid.

11. Deterministic batch, replay, and memory attestations
11.1 Ordinary batch

INFERENCE. Run all 37 cases in fresh isolated processes with at most 18 simultaneous workers.

INFERENCE. Each ordinary row records:

Python
Run
{
    "case_token",
    "status",

    "replacement_probe_sha256",
    "frozen_control_probe_sha256",

    "scan_sha256",
    "candidate_pool_sha256",
    "a0_prediction_sha256",

    "input_table_hashes",
    "replacement_edge_table_hashes",
    "output_table_hashes",

    "input_contract_attestation_sha256",
    "edge_semantics_attestation_sha256",
    "engineering_profile_sha256",
    "inference_semantic_sha256",

    "case_counters",

    "wall_seconds",
    "peak_rss_bytes",
    "input_json_bytes",
    "serialized_output_bytes",
}

INFERENCE. All 37 rows must report success and remain below 16 GiB.

11.2 Designated replays

INFERENCE. The replay set contains exactly four distinct cases:

C001, reason INHERITED_200000_POINT_REFERENCE;

C003, reason FROZEN_MAXIMUM_TRANSITION_MODE_EDGE_COUNT;

C030, reason FROZEN_MAXIMUM_TRANSITION_MODE_COUNT;

the lexicographically smallest case outside {C001,C003,C030} having the maximum ordinary-batch complete_same_surface_edge_count, reason MAXIMUM_COMPLETE_SAME_SURFACE_EDGE_COUNT_OUTSIDE_FROZEN_SET.

INFERENCE. The fourth selection is recomputed from the ordinary inference-safe batch. It is not supplied manually.

INFERENCE. Each selected case runs exactly three new isolated replays, for exactly 12 replay processes.

INFERENCE. Within each case, all three runs must have identical:

replacement probe SHA-256
frozen control probe SHA-256
complete same-surface table SHA-256
effective edge-table SHA-256
inference-semantic SHA-256
input bytes
serialized output bytes

INFERENCE. Every replay hash must equal its ordinary-batch hash.

INFERENCE. Every replay and ordinary process must remain strictly below 16 GiB.

INFERENCE. Solver randomness is absent at inference. Any post-lock CP-SAT operation retains one worker, seed zero, fixed search, no wall-clock cutoff, and accepts only proven OPTIMAL or INFEASIBLE statuses.

11.3 Synthetic worst-cap resource test

INFERENCE. Add one isolated synthetic resource test with:

4,000 parent records
4 modes per parent
832,000 complete same-surface candidate evaluations
the maximum constructible eligible edge population
24,000 or fewer inherited local rungs

INFERENCE. Run it three times. The three canonical hashes must be identical, and every run must remain below 16 GiB.

12. Lock bundle

INFERENCE. The new bundle is a compact hash-and-attestation bundle. It does not embed every complete edge table.

INFERENCE. Each bundle case row contains the canonical probe SHA, all table hashes, all semantic and engineering hashes, all structural counters, resource measurements, and the compact certificate novelty records.

INFERENCE. Post-lock attribution receives a separate --probe-root. Before reading any gold-bearing source, it resolves C001.json through C037.json, validates each file against its locked SHA, validates canonical serialization, and validates all exact nested schemas.

12.1 Certificate novelty record

INFERENCE. Before gold, compare each valid replacement certificate against the 31 frozen valid control certificates using the unchanged 5-degree / 4-mm / 60%-60% line contract.

INFERENCE. Each bundle certificate_novelty_table record contains exactly:

Python
Run
CERTIFICATE_NOVELTY_FIELDS = {
    "case_token",
    "replacement_certificate_id",

    "edge_semantics_dependent",
    "causal_added_same_surface_edge_ids_sha256",

    "control_equivalent_certificate_ids",
    "control_equivalent_certificate_ids_sha256",

    "novel_edge_attributable",
    "certificate_novelty_sha256",
}

INFERENCE. novel_edge_attributable is true exactly when:

Python
Run
(
    edge_semantics_dependent is True
    and geometry_certificate_valid is True
    and control_equivalent_certificate_ids == []
)
13. Complete pre-gold lock criteria

INFERENCE. Preserve the existing 25 LPF criteria and add the exact edge-specific criteria below:

Python
Run
LOCK_CRITERION_KEYS = (
    # Existing V2 criteria.
    "profile_identity_valid",
    "engineering_profile_hashes_recomputed",
    "exact_nested_schemas_valid",

    "case_count_is_37",
    "successful_case_count_is_37",
    "canonical_rebuilds_identical",

    "scan_hashes_unchanged",
    "candidate_pool_file_hashes_unchanged",
    "a0_prediction_hashes_unchanged",
    "input_table_hashes_unchanged",

    "input_projection_attestations_valid",
    "inherited_v2_caps_valid",
    "no_structural_cap_breach",

    "forbidden_data_absent",
    "source_points_not_consumed",
    "source_points_not_serialized",

    "research_only",
    "inference_safe",
    "prediction_mutated_false",
    "birth_columns_disabled",
    "ownership_not_consumed",
    "solver_not_consumed",
    "truth_not_consumed",

    "designated_replay_attestation_valid",
    "all_measured_peak_rss_below_16_gib",

    # New exact criteria.
    "frozen_source_hashes_unchanged",
    "frozen_lpf_control_probe_hashes_reproduced",
    "frozen_lpf_control_census_reproduced",

    "complete_same_surface_edges_recomputed",
    "current_same_surface_edges_exact_subset",
    "current_same_surface_records_byte_identical",
    "inherited_cross_mode_edges_byte_identical",

    "added_edges_only_connect_distinct_neighbor_parents",
    "no_neighbor_degree_truncation",

    "edge_deletion_counterfactual_valid",
    "one_parent_components_remain_zero_span",

    "intervention_nonempty",
    "two_sided_single_parent_support_extended",
    "novel_edge_attributable_certificate_exists",
)

INFERENCE. Missing or extra criterion keys make the lock invalid.

INFERENCE. ready_for_gold is true only when every criterion is exactly true.

14. Exact post-lock attribution
14.1 Opening order

INFERENCE. The post-lock command requires:

expected replacement bundle SHA-256
expected replacement lock-attestation SHA-256
probe root containing the 37 locked per-case payloads
frozen prior stage attribution
frozen grouped split
sealed gold
frozen A0 predictions

INFERENCE. Validation order is mandatory:

replacement bundle SHA;

replacement lock SHA;

complete lock criterion mapping;

all 37 replacement probe hashes and schemas;

replay-attestation SHA;

frozen source hashes;

only then gold, grouped split, prior attribution, and A0 geometry.

14.2 Primary population

INFERENCE. The primary population remains exactly the frozen 67 transition stage-1 rows.

INFERENCE. Stage-4 and stage-7 rows remain excluded from the recovery numerator.

14.3 Exact matching

INFERENCE. Reuse the bounded exact CP-SAT matching sequence:

maximum cardinality;

minimum bounded integer geometric cost;

lexicographically smallest ordered assignment.

INFERENCE. Retain the exact line contract:

axis angle                     <= 5 degrees
transverse offset              <= 4 mm
gold-span overlap              >= 60%
certificate-span overlap       >= 60%

INFERENCE. Retain the exact matching caps:

Python
Run
MAX_MATCH_LEFT_VERTICES = 256
MAX_MATCH_RIGHT_VERTICES = 65_536
MAX_MATCH_EDGES = 32_768
MAX_MATCH_EDGE_COST_INT = 10_000_000

INFERENCE. Gold recovery matching is run only against certificates with novel_edge_attributable == true.

INFERENCE. Frozen A0 false-prediction matching is run independently against all valid replacement certificates, not merely the attributable subset. This prevents a non-attributable replacement certificate from escaping the false-certificate denominator.

14.4 Exact post-lock row

INFERENCE. Each primary row contains exactly:

Python
Run
POSTLOCK_ROW_FIELDS = {
    "case_token",
    "split_group_id",
    "gold_id",
    "primary_population",

    "recovered",
    "reason_code",

    "replacement_certificate_id",
    "selected_component_id",
    "selected_geometry_gate_depth",

    "edge_semantics_dependent",
    "novel_edge_attributable",
    "causal_added_same_surface_edge_ids_sha256",

    "control_equivalent_certificate_ids",
    "integer_geometric_cost",
}

INFERENCE. Exact success reason:

VALID_NOVEL_EDGE_ATTRIBUTABLE_CERTIFICATE

INFERENCE. Exact attribution-specific failure reasons:

MATCHED_CERTIFICATE_NOT_EDGE_DEPENDENT
MATCHED_CONTROL_EQUIVALENT_CERTIFICATE

INFERENCE. Otherwise use:

NO_MATCHING_REPLACEMENT_COMPONENT

or the unchanged first failed geometry-gate code.

14.5 Exact summary records

INFERENCE. The post-lock summary exposes at least:

Python
Run
{
    "primary_stage1_count": 67,

    "replacement_recovered_stage1_count": ...,
    "edge_attributable_recovered_stage1_count": ...,
    "control_recovered_stage1_count": 0,

    "independent_edge_attributable_group_count": ...,

    "replacement_valid_certificate_count": ...,
    "control_valid_certificate_count": 31,
    "novel_edge_attributable_certificate_count": ...,

    "control_equivalent_recovery_count": ...,

    "observed_frozen_a0_false_prediction_count": 33,
    "a0_false_predictions_certified": ...,

    "new_failure_reason_counts": {...},
}
15. Exact GO/STOP thresholds

INFERENCE. A GO requires all pre-lock criteria and all three unchanged empirical criteria:

Python
Run
GO = (
    lock_attestation["ready_for_gold"] is True
    and edge_attributable_recovered_stage1_count >= 14
    and independent_edge_attributable_group_count >= 8
    and a0_false_predictions_certified <= 8
)

INFERENCE. The exact boundaries remain:

14 recovered / 8 groups / 8 false  -> GO
13 recovered / 8 groups / 8 false  -> STOP
14 recovered / 7 groups / 8 false  -> STOP
14 recovered / 8 groups / 9 false  -> STOP

INFERENCE. A result at or above 14/8/8 remains STOP if any lock, counterfactual, source-hash, replay, cap, or attribution criterion is false.

INFERENCE. A GO authorizes only preparation of another bounded professor research packet.

INFERENCE. A GO does not authorize A1 ownership, emitted births, learning, fitted potentials, threshold changes, expected-count logic, family routing, runtime integration, or main-branch promotion.

INFERENCE. A STOP falsifies this exact operational edge hypothesis—removal of mutual-top-eight SAME_SURFACE pruning—not every imaginable alternative edge construction.

16. Ordered file-by-file patch plan

OBSERVED. 00_GPT_PRO_PROMPT.md, 01_LOCKED_STOP_REPORT.md, and 02_PRIOR_PROFESSOR_RULING_B.md remain immutable prior-decision evidence. They are copied into the next professor package without alteration.

OBSERVED. 03_BATCH_MANIFEST.json, 04_REPLAY_ATTESTATION.json, 05_POSTLOCK_ATTRIBUTION.json, 14_LOCKED_RESULT_SUMMARY.json, 15_PER_CASE_CERTIFICATE_CENSUS.json, and 16_COMBINED_LPF_V2_TESTS.txt remain frozen controls. They are never republished as new results.

INFERENCE. backend/patch_graph_latent_decomposition.py remains byte-unchanged. It is the authoritative frozen definition of parent construction, projective modes, the current complete-candidate predicate, mutual-top-eight pruning, and same-parent local rungs.

INFERENCE. scripts/build_f1_scan_only_candidate_pool.py remains byte-unchanged. No candidate pool, source cloud, plane table, or transition table is regenerated for this tranche.

INFERENCE. backend/f1_local_plane_fragment_probe.py remains byte-unchanged and is invoked only to regenerate the frozen control outputs.

INFERENCE. Existing diagnose, batch, replay, bundle, and attribution scripts—files 09 through 13—remain byte-unchanged as the complete control lane.

INFERENCE. Add backend/f1_v2_complete_same_surface_edge_probe.py containing:

exact projection validation;

complete edge construction;

inherited-edge subset proof;

unchanged downstream LPF geometry under the new profile;

fragment/component causal-edge logging;

exact schema and hash validation;

deletion counterfactual helpers.

INFERENCE. Add scripts/diagnose_f1_v2_complete_same_surface_edge_probe.py with the exact CLI:

--scan
--candidate-pool
--a0-prediction
--expected-control-probe-sha256
--output

It hashes scan and A0 as byte streams, projects the pool, regenerates the control, validates the expected control hash, and writes one canonical replacement payload.

INFERENCE. Add scripts/run_f1_v2_complete_same_surface_edge_probe_batch.py for 37 isolated jobs, at most 18 workers, complete resource rows, and no replay claims.

INFERENCE. Add scripts/run_f1_v2_complete_same_surface_edge_probe_replays.py for the exact four-case, twelve-run replay attestation.

INFERENCE. Add scripts/build_f1_v2_complete_same_surface_edge_probe_bundle.py to:

validate all 37 probes;

regenerate all 37 controls;

validate the complete counterfactual;

compute the certificate novelty table;

emit every lock criterion;

produce a compact non-embedding bundle.

INFERENCE. Add scripts/attribute_f1_v2_complete_same_surface_edge_probe_gold.py with the unchanged exact CP-SAT matcher, strict opening order, causal recovery numerator, and conservative all-certificate false matching.

INFERENCE. Add:

tests/test_f1_v2_complete_same_surface_edge_probe.py
tests/test_f1_v2_complete_same_surface_edge_probe_memory.py
tests/test_f1_v2_complete_same_surface_edge_probe_replays.py
tests/test_build_f1_v2_complete_same_surface_edge_probe_bundle.py
tests/test_attribute_f1_v2_complete_same_surface_edge_probe_gold.py

INFERENCE. Mandatory core tests include:

test_complete_builder_reproduces_every_current_same_surface_edge
test_current_same_surface_records_are_byte_identical
test_cross_mode_edges_are_byte_identical
test_added_edges_only_join_distinct_26_neighbor_parents
test_complete_builder_has_no_degree_truncation

test_angle_equal_to_15_is_eligible
test_angle_nextafter_above_15_is_ineligible
test_gap_equal_to_tau_is_eligible
test_gap_nextafter_above_tau_is_ineligible

test_edge_input_order_is_byte_invariant
test_projective_normal_sign_flip_is_output_table_invariant

test_control_mode_reproduces_frozen_lpf_bytes
test_deleting_added_edges_restores_control
test_one_parent_component_cannot_gain_finite_span

test_causal_fragment_requires_accepted_added_union_edge
test_causal_component_requires_added_component_or_fragment_edge
test_noncausal_certificate_cannot_enter_recovery_numerator
test_control_equivalent_certificate_cannot_enter_recovery_numerator

test_complete_edge_cap_fails_closed
test_effective_edge_cap_fails_closed
test_no_structural_table_is_truncated

INFERENCE. Regenerate a new compact result summary, per-case edge/certificate census, combined focused-test evidence, and package manifest only after the mandatory pre-gold stop milestone is reached.

17. Mandatory stop milestone
V2 COMPLETE-SAME-SURFACE EDGE PRE-GOLD LOCK

INFERENCE. The milestone is reached only after all of the following are complete:

all frozen source hashes validate

all new focused tests pass

all 37 replacement jobs succeed in isolated processes

all 37 canonical replacement rebuilds match

all 37 regenerated LPF V2 control hashes match
the frozen batch manifest

the frozen LPF census and certificate distribution
reproduce exactly

every current SAME_SURFACE edge is reproduced
byte-identically by complete eligibility

every CROSS_MODE_LOCAL edge is preserved byte-identically

all added edges connect distinct 26-neighbor parents

the deletion counterfactual restores the frozen control

every one-parent replacement component remains zero-span

the intervention is nonempty

at least one locked two-sided one-parent component
extends to multi-parent support through a causal added edge

at least one novel edge-attributable certificate exists

the exact four designated cases complete three fresh
byte-identical replays each

all ordinary, replay, and synthetic-cap processes remain
strictly below 16 GiB

the compact bundle is externally hashed

the complete lock attestation reports ready_for_gold=true

INFERENCE. At that exact point, stop all implementation and architecture work.

INFERENCE. Gold may be opened only by the post-lock command receiving the externally frozen replacement bundle SHA-256 and lock-attestation SHA-256.

INFERENCE. After post-lock attribution, stop again regardless of GO or STOP.

INFERENCE. No runtime/main file may be changed during this tranche.

18. Final ruling

INFERENCE. Authorize exactly one research implementation: V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1.

INFERENCE. Do not alter the 15-degree angle predicate, the surface-gap predicate, any LPF geometry gate, or the 14/8/8 empirical gate.

INFERENCE. Do not add cross-parent rungs, new modes, finer voxels, point-derived strips, finite endpoint padding, ownership, births, learning, fitted potentials, or family identity.

INFERENCE. The tranche asks one clean question: whether the frozen transition evidence becomes physically localizable when every already-eligible cross-parent same-surface continuation is retained instead of being subjected to mutual-top-eight pruning.

UNKNOWN. Until the complete Gold37 parent/mode tables are processed under this exact specification, whether the intervention produces a GO is unresolved.

OBSERVED. The submission basis included the prompt, locked report, prior ruling, batch manifest, replay attestation, post-lock attribution, primitive and candidate-pool builders, LPF core, diagnose/batch/replay/bundle/attribution scripts, locked summary, per-case census, combined tests, and package manifest. The old lane remains frozen and the ruling above adds only a separate research lane. 

00_GPT_PRO_PROMPT

 

01_LOCKED_STOP_REPORT

 

02_PRIOR_PROFESSOR_RULING_B

 

03_BATCH_MANIFEST

 

04_REPLAY_ATTESTATION

 

05_POSTLOCK_ATTRIBUTION

 

06_patch_graph_latent_decomposi…

 

07_build_f1_scan_only_candidate…

 

08_f1_local_plane_fragment_probe

 

09_diagnose_f1_local_plane_frag…

 

10_run_f1_local_plane_fragment_…

 

11_run_f1_local_plane_fragment_…

 

12_build_f1_local_plane_fragmen…

 

13_attribute_f1_local_plane_fra…

 

14_LOCKED_RESULT_SUMMARY

 

15_PER_CASE_CERTIFICATE_CENSUS

 

16_COMBINED_LPF_V2_TESTS

 

manifest
