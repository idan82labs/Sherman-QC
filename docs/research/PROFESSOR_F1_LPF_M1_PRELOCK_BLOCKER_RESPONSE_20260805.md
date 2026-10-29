PRIMARY_RULING: B

1. Exact rationale

OBSERVED. LOCAL_PLANE_FRAGMENT_PROBE_V1 is internally contradictory on the frozen C001 input. The written profile requires every mode to belong to exactly one retained fragment, every singleton to be retained, and every fragment to satisfy the unchanged plane-fit tolerance. C001 contains six final singleton modes above the fixed 3.0 mm limit; the largest raw value is 3.5759435084334337 mm. No 37-case LPF batch was frozen, no LPF bundle was locked, and gold was not opened. 

01_PRELOCK_BLOCKER_REPORT

OBSERVED. The provisional code resolves the contradiction by applying the final fit assertion only when len(mode_ids) > 1, retaining over-tolerance singletons and setting them certificate-ineligible. The test suite explicitly codifies that provisional exception. The exception is absent from the V1 engineering-profile descriptor and therefore changes the experiment without changing its profile identity. 

04_f1_local_plane_fragment_probe

 

09_test_f1_local_plane_fragment…

INFERENCE. V1 must be retired as an invalid pre-lock specification. It must never be labeled a completed or empirically failed Gold37 experiment.

INFERENCE. The broader local-plane hypothesis is not falsified by this contradiction. The failure occurs before inference and before gold: the profile lacks a total classification rule for an inherited mode that is not a valid fitted plane fragment. Treating such a mode as a separate, bounded, non-participating orphan is a representation-domain completion, not a threshold rescue.

INFERENCE. Ruling B is preferable to terminating the tranche because the orphan rule is strictly conservative:

it preserves every inherited V2 mode for accounting;

it does not increase the 3.0 mm tolerance;

it does not allow an orphan to create or strengthen a certificate;

it removes invalid plane evidence from pair and component construction;

it is chosen without opening gold;

it forces a new profile, new hashes, and complete replay.

UNKNOWN. The number and spatial distribution of high-fit singleton orphans across all 37 cases have not been locked under the new profile. The V2 implementation must therefore fail closed on every schema, cap, partition, or non-finite-value violation rather than extrapolating from C001.

2. Versioned bounded orphan/singleton specification
2.1 Exact identities

INFERENCE. Use these exact identities:

Python
Run
LOCAL_PLANE_FRAGMENT_PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
PROBE_SCHEMA_VERSION = 2
PROBE_KIND = "f1_local_plane_fragment_probe_v2"

BATCH_SCHEMA_VERSION = 2
BATCH_KIND = "f1_local_plane_fragment_probe_batch_v2"

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "f1_local_plane_fragment_probe_bundle_v2"

POSTLOCK_SCHEMA_VERSION = 2
POSTLOCK_KIND = "postlock_gold_local_plane_fragment_attribution_v2"

ORPHAN_RULE_ID = "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1"
ORPHAN_KIND = "INHERITED_HIGH_FIT_SINGLETON"

INPUT_PROJECTION_PROFILE = "LPF_INPUT_PROJECTION_V2"
REPLAY_ATTESTATION_KIND = (
    "f1_local_plane_fragment_probe_replay_attestation_v2"
)
LOCK_ATTESTATION_KIND = (
    "f1_local_plane_fragment_probe_lock_attestation_v2"
)
POSTLOCK_MATCHING_PROFILE = "LPF_POSTLOCK_EXACT_MATCH_V2"

INFERENCE. No V1 artifact, V1 engineering-profile hash, V1 replay record, or V1 bundle may satisfy a V2 validator.

2.2 Exact classification predicate

INFERENCE. Perform the existing deterministic SAME_SURFACE union procedure unchanged up to its final connected components. Compute each final component’s fit with the unchanged formula:

fit(C)=P
90,w
	​

(∣n
C
	​

⋅c
m
	​

+d
C
	​

∣+surface_fit_p90_mm(m))

where weights remain the inherited mode point counts.

INFERENCE. Classify each final component exactly as follows:

Python
Run
fit_mm = float(_fit_fragment(component_mode_ids)["surface_fit_p90_mm"])
tolerance_mm = float(plane_fit_tolerance_mm)

if not math.isfinite(fit_mm):
    raise ValueError("non-finite final fragment fit")

if len(component_mode_ids) > 1:
    if fit_mm > tolerance_mm:
        raise ValueError(
            "multi-mode final component exceeds plane-fit tolerance"
        )
    classification = "VALID_LOCAL_PLANE_FRAGMENT"

else:
    if fit_mm > tolerance_mm:
        classification = "INHERITED_HIGH_FIT_SINGLETON_ORPHAN"
    else:
        classification = "VALID_LOCAL_PLANE_FRAGMENT"

INFERENCE. The comparison is binary64 strict > with no epsilon and no rounded-value comparison. Equality is valid:

fit_mm == tolerance_mm  -> valid singleton fragment
fit_mm >  tolerance_mm  -> orphan

INFERENCE. The tolerance formula remains unchanged:

Python
Run
plane_fit_tolerance_mm = min(
    3.0,
    max(1.25, 1.5 * transition_local_spacing_mm),
)

INFERENCE. An over-tolerance component with two or more modes is not converted into multiple orphans, split, retried, or re-clustered. It fails the case closed. Only an already final singleton can become an orphan.

INFERENCE. Existing mode-overflow semantics remain independent. A fit-valid singleton whose parent has mode_overflow=true remains a normal fragment with its existing transition-certificate prohibition. It is not an orphan unless the exact fit predicate above is also true.

2.3 Representation decision

INFERENCE. A high-fit singleton is recorded only in a separate local_plane_orphan_table.

INFERENCE. It is not retained in local_plane_fragment_table and does not receive a local_plane_fragment_id.

INFERENCE. This preserves the invariant:

Every record called a local plane fragment satisfies
surface_fit_p90_mm <= plane_fit_tolerance_mm.

INFERENCE. Every inherited mode still has exactly one representation role:

VALID_FRAGMENT_MEMBER
or
HIGH_FIT_SINGLETON_ORPHAN
2.4 Exact orphan role contract

INFERENCE. Every orphan record contains this exact nested role contract:

JSON
{
  "local_pair_seed_allowed": false,
  "component_membership_allowed": false,
  "parent_mass_contribution_allowed": false,
  "global_plane_coverage_allowed": false,
  "transition_certificate_allowed": false
}

INFERENCE. An orphan:

may not seed a local plane pair;

may not enter a local transition component;

contributes zero normalized parent mass;

receives no global-plane coverage classification;

cannot complete or partially satisfy a geometry certificate;

cannot supply left-side, right-side, occupied-bin, transition-rung, TLS, or span evidence.

INFERENCE. A parent containing both an orphan child and another valid fragment child may still contribute physical mass through the valid child. The parent is counted at most once in any component. The orphan adds no second mass contribution.

2.5 Hard structural exclusion

INFERENCE. fragment_of_mode must contain valid-fragment modes only:

Python
Run
fragment_of_mode: dict[str, str] = {}
orphan_mode_ids: set[str] = set()

for component in final_components:
    if classify(component) == "VALID_LOCAL_PLANE_FRAGMENT":
        for mode_id in component.mode_ids:
            fragment_of_mode[mode_id] = component.fragment_id
    else:
        orphan_mode_ids.add(component.only_mode_id)

INFERENCE. Local pair enumeration must exclude every edge incident to an orphan structurally, before angle or geometry evaluation:

Python
Run
for edge in canonical_cross_mode_edges:
    left_mode, right_mode = edge.mode_ids

    if (
        left_mode not in fragment_of_mode
        or right_mode not in fragment_of_mode
    ):
        excluded_cross_mode_seed_edge_count += 1
        continue

    left_fragment = fragment_of_mode[left_mode]
    right_fragment = fragment_of_mode[right_mode]

    if left_fragment != right_fragment:
        pair_seed_edges[(left_fragment, right_fragment)].append(edge)

INFERENCE. Component construction accepts only modes belonging to its two valid fragment records. The validator must reject a pair or component referencing any orphan mode or orphan ID.

INFERENCE. No additional geometry gate such as orphan_free is needed. Orphan participation must be impossible in the graph, rather than allowed and later rejected by a Boolean gate.

2.6 Exact orphan record schema

INFERENCE. Every record in local_plane_orphan_table must have exactly these fields and no others:

Python
Run
ORPHAN_RECORD_FIELDS = {
    "local_plane_orphan_id",
    "orphan_kind",
    "orphan_rule_id",

    "mode_atom_id",
    "parent_atom_id",
    "mode_point_count",

    "inherited_mode_surface_fit_p90_mm",
    "computed_singleton_surface_fit_p90_mm",
    "plane_fit_tolerance_mm",
    "excess_over_tolerance_mm",

    "parent_mode_overflow",
    "parent_transition_certificate_allowed",

    "incident_same_surface_edge_count",
    "incident_cross_mode_edge_count",
    "incident_mode_edge_ids_sha256",

    "witness_mode_edge_ids",
    "witness_ids_truncated",

    "member_mode_ids_sha256",
    "member_parent_ids_sha256",

    "role_contract",
    "orphan_sha256",
}

INFERENCE. Field semantics are:

JSON
{
  "orphan_kind": "INHERITED_HIGH_FIT_SINGLETON",
  "orphan_rule_id": "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1",
  "mode_point_count": "inherited nonnegative integer",
  "excess_over_tolerance_mm":
    "computed_singleton_surface_fit_p90_mm - plane_fit_tolerance_mm",
  "witness_mode_edge_ids":
    "first eight canonical incident mode-edge IDs",
  "witness_ids_truncated":
    "true exactly when the full incident-edge set contains more than eight IDs"
}

INFERENCE. No centroid, normal, source point, source-point index, filename, case identity, part identity, family identity, truth, ownership, or solver field is serialized in the orphan record. The frozen mode table already contains its bounded geometric statistics.

2.7 IDs and hashes

INFERENCE. Construct the orphan ID from stable inherited identities only:

Python
Run
local_plane_orphan_id = (
    "LPO_"
    + sha256_json(
        [
            "LOCAL_PLANE_FRAGMENT_PROBE_V2",
            "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1",
            mode_atom_id,
            parent_atom_id,
        ]
    )[:16]
)

INFERENCE. Compute these hashes from complete sorted sets, not bounded witness arrays:

Python
Run
member_mode_ids_sha256 = sha256_json([mode_atom_id])
member_parent_ids_sha256 = sha256_json([parent_atom_id])

incident_mode_edge_ids_sha256 = sha256_json(
    sorted(full_incident_mode_edge_ids)
)

INFERENCE. Compute orphan_sha256 over the complete orphan record excluding orphan_sha256 itself, using the existing canonical JSON encoder.

INFERENCE. Canonical orphan-table ordering is ascending local_plane_orphan_id.

2.8 Top-level V2 payload changes

INFERENCE. The V2 top-level payload adds:

inference_semantic_sha256
input_contract_attestation
local_plane_orphan_table

INFERENCE. Its output table hashes are exactly:

JSON
{
  "local_plane_fragment_table_sha256": "...",
  "local_plane_orphan_table_sha256": "...",
  "local_plane_pair_records_sha256": "..."
}

INFERENCE. inference_semantic_sha256 is:

Python
Run
sha256_json(
    {
        "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
        "engineering_profile_sha256": engineering_profile_sha256,
        "projected_input_sha256":
            input_contract_attestation["projected_input_sha256"],
        "output_table_hashes": output_table_hashes,
    }
)

INFERENCE. Full candidate-pool file SHA-256 remains provenance. It is not part of inference_semantic_sha256.

2.9 Exact partition invariants

INFERENCE. The validator must prove:

Python
Run
input_modes = set(mode_by_id)

fragment_modes = {
    mode_id
    for fragment in fragment_table
    for mode_id in full_fragment_membership[fragment.id]
}

orphan_modes = {
    orphan["mode_atom_id"]
    for orphan in orphan_table
}

assert fragment_modes.isdisjoint(orphan_modes)
assert fragment_modes | orphan_modes == input_modes
assert len(fragment_modes) + len(orphan_modes) == len(input_modes)

INFERENCE. It must also prove:

Every orphan has exactly one mode.
Every over-tolerance singleton is an orphan.
Every orphan is over tolerance.
Every valid fragment is at or below tolerance.
No multi-mode orphan exists.
No orphan mode is referenced by any valid fragment.
No pair or component references an orphan.
2.10 Caps and counters

INFERENCE. Preserve all existing V1 caps and add:

Python
Run
MAX_LOCAL_PLANE_ORPHANS = 16_000
MAX_FRAGMENT_PLUS_ORPHAN_RECORDS = 16_000
MAX_ORPHAN_WITNESS_MODE_EDGE_IDS = 8

INFERENCE. Enforce:

Python
Run
len(fragment_table) <= 16_000
len(orphan_table) <= 16_000
len(fragment_table) + len(orphan_table) <= 16_000

INFERENCE. Structural tables may not be truncated. Only orphan witness edge IDs may be truncated, with the complete edge-set hash retained.

INFERENCE. Add these exact counters:

valid_fragment_mode_count
orphan_mode_count
local_plane_orphan_count
high_fit_singleton_orphan_count
valid_singleton_fragment_count
orphan_incident_same_surface_edge_count
orphan_incident_cross_mode_edge_count
excluded_cross_mode_seed_edge_count_due_to_orphan
mode_partition_record_count

INFERENCE. Remove singleton_fit_ineligible_fragment_count. A high-fit singleton is no longer a fragment.

INFERENCE. Require:

orphan_mode_count == local_plane_orphan_count
orphan_mode_count == high_fit_singleton_orphan_count
valid_fragment_mode_count + orphan_mode_count == input_mode_count
mode_partition_record_count ==
    local_plane_fragment_count + local_plane_orphan_count
2.11 Required engineering-profile hash additions

OBSERVED. The current engineering descriptor hashes the profile, existing formulas, thresholds, caps, rejection order, and rounding, but not the provisional singleton exception. 

04_f1_local_plane_fragment_probe

INFERENCE. V2 must hash this complete descriptor:

Python
Run
{
    "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V2",
    "schema_version": 2,

    "formulas": {
        # all unchanged V1 formula identifiers,
        ...
    },

    "final_component_classification": {
        "rule_id": "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1",
        "fit_source": "unrounded_binary64_fit_fragment_result",
        "comparison": "STRICT_GT_NO_EPSILON",
        "equality_result": "VALID_LOCAL_PLANE_FRAGMENT",
        "singleton_over_tolerance_result":
            "INHERITED_HIGH_FIT_SINGLETON_ORPHAN",
        "multimode_over_tolerance_result": "FAIL_CLOSED",
        "nonfinite_result": "FAIL_CLOSED"
    },

    "orphan_role_contract": {
        "local_pair_seed_allowed": False,
        "component_membership_allowed": False,
        "parent_mass_contribution_allowed": False,
        "global_plane_coverage_allowed": False,
        "transition_certificate_allowed": False
    },

    "mode_partition_invariant":
        "EXACT_DISJOINT_UNION_OF_VALID_FRAGMENT_MODES_AND_ORPHAN_MODES",

    "input_projection_profile": "LPF_INPUT_PROJECTION_V2",

    "exact_schema_field_sets": {
        "top_level": [...],
        "fragment": [...],
        "global_plane_coverage": [...],
        "orphan": [...],
        "orphan_role_contract": [...],
        "pair": [...],
        "global_pair_comparison": [...],
        "component": [...],
        "input_contract_attestation": [...]
    },

    "canonical_ids": {
        "fragment": "...",
        "orphan": "...",
        "pair": "...",
        "component": "..."
    },

    "hard_caps": {...},
    "first_rejection_order": [...],
    "rounding": {...}
}

INFERENCE. The hash is recomputed as:

Python
Run
engineering_profile_sha256 = sha256_json(
    {
        "descriptor": engineering_profile_descriptor_v2(),
        "case_thresholds": thresholds,
    }
)
2.12 Formal counterfactual

INFERENCE. The required gold-free counterfactual is:

Build the V2 result normally.

Record the set of V2 orphan mode IDs.

Build a reference result using the same V2 code and profile, but remove exactly those mode IDs and their incident edges immediately after input validation.

Compare valid-fragment, local-pair, component, and certificate tables.

INFERENCE. The following must be byte-identical between the normal and deletion-reference runs:

local_plane_fragment_table
local_plane_pair_records
geometry certificate set
all non-orphan counters after orphan-edge exclusion

INFERENCE. Only these may differ:

local_plane_orphan_table
orphan counters
input-mode partition counters
inference_semantic_sha256

INFERENCE. The implementation is falsified before lock if:

an orphan appears in a pair or component;

an orphan changes a valid fragment or certificate relative to the deletion reference;

an orphan contributes mass or coverage;

a fit-valid singleton becomes an orphan;

an over-tolerance singleton remains a fragment;

a multi-mode component is converted into an orphan;

the mode partition is incomplete or overlapping.

2.13 Thresholds and existing artifacts

INFERENCE. The preregistered empirical thresholds remain exactly:

minimum recovered stage-1 gold objects: 14
minimum independent family groups:      8
maximum frozen A0 false certificates:   8

INFERENCE. No threshold is weakened because of the orphan rule.

INFERENCE. All provisional V1 per-case outputs, replay outputs, manifests, profile hashes, and prospective bundle artifacts must be discarded and regenerated under V2.

OBSERVED. The existing nine designated replays demonstrate bounded operation of the provisional V1 implementation, but their profile was not approved for lock. They cannot serve as the V2 replay attestation. 

01_PRELOCK_BLOCKER_REPORT

3. Mandatory audit dispositions
Audit item	Disposition	Adjudication and exact mandatory change
1. P0 designated three-run replay/RSS evidence	MANDATORY BEFORE LOCK	OBSERVED. The current bundle code substitutes [probe_sha] when replay hashes are absent and can therefore lock one-run rows. 

07_build_f1_local_plane_fragmen…

 INFERENCE. Require a separate V2 replay attestation containing exactly C001, C030, and C003; exactly three isolated runs per case; all three hashes equal the corresponding V2 batch probe hash; per-run RSS, time, input bytes, and output bytes; and verified case-selection reasons. The bundle builder must require this attestation as an input. No fallback is permitted. Post-lock validation must verify its SHA-256 before any gold file is read.
2. P1 incomplete final GO result	MANDATORY BEFORE LOCK	OBSERVED. _evaluate_gate() currently exposes only 14/8/8 and successful-job criteria. 

08_attribute_f1_local_plane_fra…

 INFERENCE. The V2 result must expose every pre-lock safety, hash, schema, replay, cap, forbidden-data, source-point, and memory criterion as a named Boolean. A missing criterion is false. passed is the conjunction of the complete pre-lock lock attestation and the unchanged 14/8/8 empirical criteria.
3. P1 incomplete engineering/nested-schema/gate-depth validation	MANDATORY BEFORE LOCK	OBSERVED. The current validator checks the top-level field set and selected nested invariants, but treats the engineering profile as a supplied digest rather than recomputing the complete descriptor. 

04_f1_local_plane_fragment_probe

 INFERENCE. Recompute the engineering descriptor and hash at CLI, batch, replay, bundle, deterministic rebuild, and post-lock boundaries. Enforce exact nested field sets. Recompute geometry_gate_depth as the number of leading true gates in REJECTION_ORDER, or len(REJECTION_ORDER) when all pass.
4. P1 unbounded lexical tie encoding	MANDATORY BEFORE LOCK	OBSERVED. The post-lock matcher uses radix = 1 << len(edges) to encode lexical preference. 

08_attribute_f1_local_plane_fra…

 INFERENCE. Replace it with the bounded three-stage exact CP-SAT procedure specified below: maximum cardinality, minimum bounded integer geometry cost, then successive exact feasibility fixing by sorted left ID and smallest feasible right ID. No exponential integer is allowed.
5. P2 full candidate-pool mapping reaches core	MANDATORY BEFORE LOCK	OBSERVED. The CLI parses the complete candidate-pool object and passes it directly to the probe core. 

05_diagnose_f1_local_plane_frag…

 INFERENCE. Construct an explicit deep-copied allowlisted projection immediately after parsing. The core accepts only that exact projection schema. Extra pool fields remain file-level provenance only and cannot affect the projected input or inference-semantic hash.
6. P2 inherited V2 caps and working-cloud attestation	MANDATORY BEFORE LOCK	OBSERVED. The core checks the cap dictionary and point counts, but the current lock contract does not carry a complete independently recomputed per-case cap attestation through bundle and post-lock validation. 

04_f1_local_plane_fragment_probe

 

01_PRELOCK_BLOCKER_REPORT

 INFERENCE. Add the exact input_contract_attestation specified below, hash it per case, aggregate its hashes in the bundle, and validate it before gold.
4. Exact mandatory audit implementations
4.1 Allowed inference-input projection

INFERENCE. The CLI must project exactly these keys, in this canonical order:

Python
Run
ALLOWED_POOL_PROJECTION_FIELDS = (
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

INFERENCE. Projection code:

Python
Run
def project_lpf_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        field
        for field in ALLOWED_POOL_PROJECTION_FIELDS
        if field not in raw_pool
    ]
    if missing:
        raise ValueError(f"LPF input projection missing fields: {missing}")

    projection = {
        field: deepcopy(raw_pool[field])
        for field in ALLOWED_POOL_PROJECTION_FIELDS
    }

    if set(projection) != set(ALLOWED_POOL_PROJECTION_FIELDS):
        raise AssertionError("LPF input projection schema mismatch")

    return projection

INFERENCE. Extra top-level candidate-pool fields are excluded, not forwarded and not silently interpreted.

INFERENCE. The projection validator requires:

support_representation_profile == "TRANSITION_ATOM_V2"
source_point_serialization is false
transition_atom_caps equals the exact frozen cap dictionary
working_point_budget == 200000
0 <= transient_sampled_point_count <= working_point_budget
4.2 Exact input-contract attestation

INFERENCE. Every per-case probe must include this exact nested record:

Python
Run
INPUT_CONTRACT_ATTESTATION_FIELDS = {
    "projection_profile",
    "support_representation_profile",
    "source_point_serialization",

    "transition_atom_caps",
    "working_point_budget",
    "transient_sampled_point_count",

    "transition_parent_count",
    "transition_mode_count",
    "transition_mode_edge_count",
    "transition_parent_plus_mode_count",
    "maximum_observed_modes_per_parent",

    "table_hashes",
    "projected_input_sha256",

    "all_inherited_caps_satisfied",
    "input_contract_attestation_sha256",
}

INFERENCE. Its exact values are:

JSON
{
  "projection_profile": "LPF_INPUT_PROJECTION_V2",
  "support_representation_profile": "TRANSITION_ATOM_V2",
  "source_point_serialization": false,

  "transition_atom_caps": {
    "input_point_cap": 200000,
    "parent_cap": 4000,
    "mode_cap": 16000,
    "mode_children_per_parent_cap": 4,
    "parent_plus_mode_cap": 20000,
    "mode_edge_cap": 200000
  },

  "working_point_budget": 200000,
  "transient_sampled_point_count": 0,

  "transition_parent_count": 0,
  "transition_mode_count": 0,
  "transition_mode_edge_count": 0,
  "transition_parent_plus_mode_count": 0,
  "maximum_observed_modes_per_parent": 0,

  "table_hashes": {
    "parent_table_sha256": "...",
    "mode_table_sha256": "...",
    "mode_edge_table_sha256": "...",
    "global_plane_table_sha256": "..."
  },

  "projected_input_sha256": "...",
  "all_inherited_caps_satisfied": true,
  "input_contract_attestation_sha256": "..."
}

INFERENCE. The validator recomputes all counts and hashes from the projected tables. It does not trust the Boolean all_inherited_caps_satisfied.

4.3 Exact replay attestation

INFERENCE. Create:

kind:    f1_local_plane_fragment_probe_replay_attestation_v2
profile: LOCAL_PLANE_FRAGMENT_PROBE_V2
schema:  2

INFERENCE. It must contain exactly these designated cases:

C001 — inherited 200,000-point reference
C030 — maximum transition-mode count
C003 — maximum transition-mode-edge count

INFERENCE. Selection must be recomputed from the frozen V2 pool manifest. A tied maximum uses the lexicographically smallest case token.

INFERENCE. Each designated case must contain:

Python
Run
{
    "case_token",
    "selection_reason",

    "working_point_count",
    "transition_mode_count",
    "transition_mode_edge_count",

    "batch_probe_sha256",
    "replay_count",
    "replay_probe_sha256s",
    "maximum_peak_rss_bytes",

    "runs": [
        {
            "run_index",
            "probe_sha256",
            "resource_profile_sha256",
            "peak_rss_bytes",
            "wall_seconds",
            "input_json_bytes",
            "serialized_output_bytes",
        }
    ],
}

INFERENCE. Lock requirements are:

designated case set exactly {C001, C003, C030}
replay_count exactly 3 per case
total isolated runs exactly 9
all three probe hashes equal one another
all three probe hashes equal the V2 batch probe hash
all per-run RSS values < 16 GiB
all output byte counts equal within case
all input byte counts equal within case
4.4 Complete lock and GO criteria

INFERENCE. The V2 bundle must contain a lock_attestation with these exact criterion keys:

profile_identity_valid
engineering_profile_hashes_recomputed
exact_nested_schemas_valid

case_count_is_37
successful_case_count_is_37
canonical_rebuilds_identical

scan_hashes_unchanged
candidate_pool_file_hashes_unchanged
a0_prediction_hashes_unchanged
input_table_hashes_unchanged

input_projection_attestations_valid
inherited_v2_caps_valid
no_structural_cap_breach

forbidden_data_absent
source_points_not_consumed
source_points_not_serialized

research_only
inference_safe
prediction_mutated_false
birth_columns_disabled
ownership_not_consumed
solver_not_consumed
truth_not_consumed

designated_replay_attestation_valid
all_measured_peak_rss_below_16_gib

INFERENCE. ready_for_gold is true only when every key above is true.

INFERENCE. The post-lock result then adds:

recovered_stage1_objects_at_least_14
independent_family_groups_at_least_8
a0_false_predictions_certified_at_most_8

INFERENCE. Final passed is:

Python
Run
passed = (
    lock_attestation["ready_for_gold"]
    and recovered_stage1_objects >= 14
    and independent_family_groups >= 8
    and a0_false_predictions_certified <= 8
)

INFERENCE. A missing metric, criterion, attestation, or hash is false rather than “not applicable.”

4.5 Bounded exact post-lock matching

INFERENCE. Use these hard caps:

Python
Run
MAX_MATCH_LEFT_VERTICES = 256
MAX_MATCH_RIGHT_VERTICES = 65_536
MAX_MATCH_EDGES = 32_768
MAX_MATCH_EDGE_COST_INT = 10_000_000

INFERENCE. Exceeding a cap aborts attribution. No ranked edge subset or geometric truncation is allowed.

INFERENCE. Build one Boolean x
e
	​

 per eligible line-contract edge:

Python
Run
for left_id:
    model.Add(sum(x[e] for e incident to left_id) <= 1)

for right_id:
    model.Add(sum(x[e] for e incident to right_id) <= 1)

INFERENCE. Stage 1:

Python
Run
model.Maximize(sum(x.values()))
solve_to_OPTIMAL()
target_cardinality = solver.Value(sum(x.values()))
model.Add(sum(x.values()) == target_cardinality)

INFERENCE. Stage 2:

Python
Run
bounded_cost = sum(edge.cost * x[edge] for edge in edges)
model.Minimize(bounded_cost)
solve_to_OPTIMAL()
target_cost = solver.Value(bounded_cost)
model.Add(bounded_cost == target_cost)

INFERENCE. Stage 3 obtains the lexicographically smallest sorted assignment without a radix:

Python
Run
for left_id in sorted(left_ids):
    row = sorted(
        edges_by_left[left_id],
        key=lambda edge: edge.right_id,
    )

    if not row:
        continue

    # Prefer matching this earliest left vertex whenever a K,C solution allows it.
    if not prove_feasible(sum(x[e] for e in row) == 1):
        model.Add(sum(x[e] for e in row) == 0)
        continue

    # Find the smallest feasible right ID by prefix feasibility.
    lo = 0
    hi = len(row) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        prefix = row[: mid + 1]

        if prove_feasible(sum(x[e] for e in prefix) == 1):
            hi = mid
        else:
            lo = mid + 1

    model.Add(x[row[lo]] == 1)

INFERENCE. Every feasibility solve uses:

num_search_workers = 1
random_seed = 0
search_branching = FIXED_SEARCH
no wall-clock cutoff
accepted statuses = OPTIMAL or INFEASIBLE only

INFERENCE. The procedure is exact by induction. At each sorted left vertex, a feasible match produces a lexicographically earlier pair than leaving that vertex unmatched. Given that vertex is matched, the smallest feasible right ID produces the earliest pair. Previous decisions, maximum cardinality, and minimum geometric cost remain fixed.

5. Ordered file-by-file patch plan
1. backend/f1_local_plane_fragment_probe.py

INFERENCE. Change all profile, kind, and schema constants to V2.

INFERENCE. Replace the provisional singleton exception with the exact final-component classifier.

INFERENCE. Add:

LocalPlaneOrphanRecord
local_plane_orphan_table
orphan-table hashing
orphan counters
mode-partition validation
input-contract attestation
inference-semantic hash
engineering_profile_descriptor_v2()

INFERENCE. Remove the V1 behavior:

Python
Run
if len(mode_ids) > 1 and fit > tolerance:
    ...

INFERENCE. Enforce exact fragment, orphan, coverage, pair, global-pair, component, and attestation field sets.

INFERENCE. Recompute geometry_gate_depth:

Python
Run
expected_depth = next(
    (
        index
        for index, (gate_name, _code)
        in enumerate(REJECTION_ORDER)
        if gates[gate_name] is False
    ),
    len(REJECTION_ORDER),
)
2. scripts/diagnose_f1_local_plane_fragment_probe.py

INFERENCE. Change imports and identity checks to V2.

INFERENCE. Add project_lpf_input() and pass only the allowlisted projection to the core.

INFERENCE. Continue to open the scan and A0 prediction as byte streams for hashing only.

INFERENCE. Keep the CLI option set exactly:

--scan
--candidate-pool
--a0-prediction
--output
3. scripts/run_f1_local_plane_fragment_probe_batch.py

INFERENCE. Change batch identity to V2.

INFERENCE. Keep one isolated run per ordinary Gold37 case.

INFERENCE. Remove any implication that a one-run case row constitutes replay evidence.

INFERENCE. Each row must include:

input_contract_attestation_sha256
inference_semantic_sha256
engineering_profile_sha256
all exact table hashes
peak RSS and serialized-byte measurements
4. New scripts/run_f1_local_plane_fragment_probe_replays.py

INFERENCE. Select and verify C001/C030/C003 from the frozen V2 pool manifest.

INFERENCE. Run exactly three isolated V2 probe processes per designated case.

INFERENCE. Write the exact V2 replay attestation and its canonical SHA-256.

5. scripts/build_f1_local_plane_fragment_probe_bundle.py

INFERENCE. Require:

--probe-manifest
--replay-attestation
--output

INFERENCE. Delete:

Python
Run
row.get("replay_probe_sha256s") or [probe_sha]

INFERENCE. Rebuild and validate every V2 probe from its allowlisted projected input.

INFERENCE. Verify the replay attestation, all per-case input-contract attestations, all engineering-profile hashes, all structural caps, and all V2 output-table hashes.

INFERENCE. Emit the complete lock attestation and set locked=true only when ready_for_gold=true.

6. scripts/attribute_f1_local_plane_fragment_probe_gold.py

INFERENCE. Require the V2 bundle kind/profile and an externally supplied expected bundle SHA-256.

INFERENCE. Validate the complete lock attestation and replay-attestation hash before reading gold, grouped split, prior attribution, or A0 line geometry.

INFERENCE. Replace the 1 << E matcher with LPF_POSTLOCK_EXACT_MATCH_V2.

INFERENCE. Expand _evaluate_gate() to include every pre-lock criterion plus the unchanged 14/8/8 empirical criteria.

7. tests/test_f1_local_plane_fragment_probe.py

INFERENCE. Delete:

test_singleton_fit_exception_is_retained_but_cannot_certify

INFERENCE. Replace it with the exact V2 orphan and counterfactual tests listed below.

8. tests/test_f1_local_plane_fragment_probe_memory.py

INFERENCE. Update fixtures to V2 schema and include the orphan table and input-contract attestation.

INFERENCE. Keep the 16 GiB failure boundary and byte-identical replay requirements.

9. tests/test_build_f1_local_plane_fragment_probe_bundle.py

INFERENCE. Update the fixture so C001, C030, and C003 each have exactly three V2 runs and every other case has one ordinary batch run.

INFERENCE. Add replay-attestation and lock-attestation hash tests.

INFERENCE. Remove the expected 39-run generic replay summary; ordinary one-run case rows are not replay attestations.

10. tests/test_attribute_f1_local_plane_fragment_probe_gold.py

INFERENCE. Replace matcher tests with bounded CP-SAT cardinality/cost/lex tests.

INFERENCE. Parameterize every pre-lock and post-lock gate criterion so that one missing or false criterion causes STOP.

11. New tests/test_f1_local_plane_fragment_probe_replays.py

INFERENCE. Test designated-case selection, exact run count, byte identity, per-run RSS, V2 identity, and replay-attestation hashing.

12. Package manifest and generated evidence

INFERENCE. Regenerate the submission manifest only after the complete V2 pre-lock suite passes.

INFERENCE. Do not copy V1 profile hashes or V1 probe payloads into the V2 evidence directory.

6. Exact new and changed tests
6.1 Core orphan tests

INFERENCE. Add:

test_high_fit_singleton_is_orphan_not_fragment
test_singleton_equal_to_tolerance_is_valid_fragment
test_singleton_strictly_above_tolerance_is_orphan
test_nonfinite_singleton_fit_fails_closed
test_overfit_multimode_component_fails_closed

test_mode_partition_is_complete_and_disjoint
test_orphan_never_seeds_local_pair
test_orphan_never_enters_component
test_orphan_has_zero_parent_mass_contribution
test_orphan_has_no_global_plane_coverage

test_valid_sibling_mode_can_use_shared_parent_without_orphan_mass
test_orphan_incident_edges_are_counted_but_excluded
test_orphan_witness_cap_and_full_edge_hash
test_orphan_record_hash_recomputes_exactly

test_orphan_counterfactual_matches_mode_deletion_reference
test_orphan_order_permutation_is_byte_identical
test_projective_normal_sign_flip_is_byte_identical
6.2 Profile and schema tests

INFERENCE. Add:

test_engineering_hash_recomputes_from_complete_v2_descriptor
test_engineering_hash_changes_when_orphan_comparator_changes
test_engineering_hash_changes_when_orphan_role_changes
test_engineering_hash_changes_when_orphan_cap_changes
test_v2_rejects_v1_profile_and_schema

test_exact_fragment_nested_schema
test_exact_orphan_nested_schema
test_exact_coverage_nested_schema
test_exact_pair_nested_schema
test_exact_component_nested_schema
test_exact_input_contract_attestation_schema

test_geometry_gate_depth_is_recomputed
test_first_rejection_matches_rejection_order
test_valid_certificate_requires_all_gates_true
6.3 Input projection and inherited-cap tests

INFERENCE. Add:

test_cli_projects_exact_allowed_pool_keys
test_core_rejects_unprojected_input_mapping
test_missing_projection_field_fails_closed

test_excluded_identity_fields_do_not_change_projected_input_hash
test_excluded_graph_fields_do_not_change_projected_input_hash
test_excluded_solver_fields_do_not_change_projected_input_hash
test_excluded_truth_fields_do_not_change_projected_input_hash

test_input_contract_recomputes_table_counts
test_input_contract_recomputes_maximum_modes_per_parent
test_input_contract_recomputes_all_table_hashes
test_input_contract_rejects_wrong_working_budget
test_input_contract_rejects_sample_count_above_budget
test_input_contract_rejects_parent_plus_mode_cap_breach
test_input_contract_rejects_source_point_serialization
6.4 Replay and bundle tests

INFERENCE. Add:

test_lock_requires_exact_designated_case_set
test_lock_requires_three_runs_for_each_designated_case
test_lock_rejects_one_run_fallback
test_lock_rejects_extra_designated_case
test_lock_rejects_wrong_max_mode_case
test_lock_rejects_wrong_max_edge_case

test_lock_rejects_nonidentical_replay_hash
test_lock_rejects_replay_hash_different_from_batch_probe
test_lock_rejects_replay_rss_at_or_above_16_gib
test_lock_rejects_v1_replay_attestation

test_lock_attestation_hash_recomputes
test_bundle_rejects_mutated_lock_attestation
test_bundle_ready_for_gold_requires_every_prelock_criterion
6.5 Exact matching tests

INFERENCE. Add:

test_matching_maximizes_cardinality_before_cost
test_matching_minimizes_cost_before_lex
test_matching_prefers_earliest_feasible_left
test_matching_prefers_smallest_feasible_right
test_matching_is_invariant_to_edge_input_order
test_matching_matches_bruteforce_on_small_bipartite_graphs
test_matching_rejects_duplicate_edges
test_matching_rejects_negative_or_oversized_cost
test_matching_rejects_vertex_cap_breach
test_matching_rejects_edge_cap_breach
test_matching_three_replays_are_identical

INFERENCE. Add an isolated worst-cap synthetic resource test for the configured matcher caps. It must record wall time and peak RSS and must remain below 16 GiB.

6.6 Complete GO/STOP tests

INFERENCE. Add a parameterized test that flips each lock criterion individually and requires:

ready_for_gold == false
passed == false

INFERENCE. Preserve the exact empirical boundary tests:

14 recovered / 8 groups / 8 false -> empirical pass
13 recovered / 8 groups / 8 false -> STOP
14 recovered / 7 groups / 8 false -> STOP
14 recovered / 8 groups / 9 false -> STOP

INFERENCE. Even 14/8/8 must remain STOP when any lock criterion is false.

6.7 Existing test evidence

OBSERVED. The current 170 tests passed, but the evidence file correctly states that they validate only the provisional implementation’s internal consistency and do not authorize lock or gold opening. 

03_FOCUSED_TEST_EVIDENCE

INFERENCE. All existing focused tests must pass after their V2 updates, together with all new V2 tests. The old count of 170 is not a V2 acceptance target.

7. Required lock attestations
7.1 Per-case probe attestation

INFERENCE. Each V2 probe carries:

input_contract_attestation_sha256
engineering_profile_sha256
inference_semantic_sha256
input and output table hashes
all safety flags
all structural counters
7.2 Designated replay attestation

INFERENCE. The separate replay artifact carries the exact nine V2 runs and is hashed independently.

7.3 Bundle lock attestation

INFERENCE. The bundle carries:

probe_manifest_sha256
replay_attestation_sha256
probe_hash_table_sha256
inference_semantic_hash_table_sha256
engineering_profile_hash_table_sha256
input_contract_attestation_hash_table_sha256
a0_prediction_hash_table_sha256
maximum_peak_rss_bytes
complete criterion mapping
ready_for_gold
lock_attestation_sha256
7.4 Post-lock opening requirement

INFERENCE. The attribution command must require both:

expected_bundle_sha256
expected_lock_attestation_sha256

INFERENCE. It must validate those hashes and ready_for_gold=true before opening any gold-bearing file.

8. Single mandatory stop milestone
LPF-M1-V2 PRE-GOLD LOCK

INFERENCE. The milestone is reached only after:

the V2 orphan rule and all mandatory audit patches are complete;

all focused and new V2 tests pass;

all provisional V1 outputs are discarded;

all 37 V2 inference-safe jobs succeed in isolated processes;

all 37 deterministic rebuilds match their canonical outputs;

every per-case input, schema, cap, and engineering attestation validates;

C001, C030, and C003 each complete three new V2 isolated replays;

all nine designated replay outputs are byte-identical within case;

every measured process remains below 16 GiB;

the V2 bundle is written and externally hashed;

the complete lock attestation reports ready_for_gold=true.

INFERENCE. At that exact point, stop all implementation work.

INFERENCE. Gold may be opened only by the unchanged post-lock evaluation procedure receiving the frozen V2 bundle SHA-256 and lock-attestation SHA-256.

INFERENCE. After gold attribution, apply the unchanged 14/8/8 gate and stop again, regardless of GO or STOP.

INFERENCE. This ruling does not authorize A1 ownership integration, any birth column, learning, fitted potentials, threshold tuning, or runtime/main promotion.
