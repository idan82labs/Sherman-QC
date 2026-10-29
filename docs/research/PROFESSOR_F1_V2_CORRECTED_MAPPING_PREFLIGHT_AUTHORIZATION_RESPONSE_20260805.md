AUTHORIZE_ORDINARY_V2_BATCH_SPECIFICATION
Ruling

The corrected mapping-only preflight now establishes the amended target-to-baseline relation contract and closes the implementation defects that caused the prior STOP.

I found no remaining violated row, hash, count, mapping predicate, safety predicate, or attached implementation defect. Authorization is limited to one bounded, gold-free ordinary V2 batch under the specification below. It does not authorize designated replays, the full synthetic-resource envelope, gold/truth access, a final empirical family decision, or any runtime or production action.

The decisive correction is genuine rather than cosmetic:

The complete fixed-tube membership relation now contains all 16,988 baseline pairs, not merely the 4,783 pairs reached by frozen targets.

The 12,205 non-target pairs are therefore covered by the invariance comparison.

Exact frozen-target rows, target-ID sets, and mode-hash sets are compared to independently reconstructed source projections.

All 37 threshold hashes are compared exactly.

The independent validator reconstructs the complete-pair, target, threshold, and safety evidence instead of accepting asserted Booleans.

All five required equal-count and non-target tamper regressions are present and reject the specified modifications. 

01_INVALID_PREFLIGHT_RESPONSE

 

05_CORRECTED_V2_MAPPING_CORE

 

06_CORRECTED_MAPPING_RUNNER

 

07_INDEPENDENT_MAPPING_VALIDATOR

 

08_REQUIRED_TAMPER_TESTS

Attachment integrity
Attachment	Independently computed SHA-256	Result
00_GPT_PRO_PROMPT(9).md 

00_GPT_PRO_PROMPT

	08ba556a03c5d340ac9ea555527bc8459f61e0bea0d8d237836648b8311d2445	No reference hash supplied
01_INVALID_PREFLIGHT_RESPONSE.md 

01_INVALID_PREFLIGHT_RESPONSE

	9253b6dfcd2a2fdc98d4a0f51dab223ea06f88cc13a54cb8df40f1674ed75a18	MATCH
02_AMENDED_MAPPING_RULING.md 

02_AMENDED_MAPPING_RULING

	bf49709a1835ddefd95cdf175fa4bca089c067b0ed7afe4e10c47c7692bb2dd3	MATCH
03_CORRECTED_AUTHORITATIVE_MAPPING_PREFLIGHT.json 

03_CORRECTED_AUTHORITATIVE_MAPP…

	9d314e7ebe5900f65c79039d8f729add3b8f1ad5cc4e190cc330103c9eec60cd	MATCH
04_INDEPENDENT_COMPACT_V2_EVIDENCE.json 

04_INDEPENDENT_COMPACT_V2_EVIDE…

	869f733f48344d3a8cb3a1bb8aae7a28c6e6e10a60e16c43819affc9bccd3954	MATCH
05_CORRECTED_V2_MAPPING_CORE.py 

05_CORRECTED_V2_MAPPING_CORE

	3ca00c8dc5e837288c29a531f067141d3fef2f24152fa365b1a24e025f1b7f72	MATCH
06_CORRECTED_MAPPING_RUNNER.py 

06_CORRECTED_MAPPING_RUNNER

	0463e8d85fd89952db0383c37c54fc7832ab1cb953ea1e4deeea1baab3c8e11c	MATCH
07_INDEPENDENT_MAPPING_VALIDATOR.py 

07_INDEPENDENT_MAPPING_VALIDATOR

	ab2f01e7b6528e3ca9397edceaab2615ddf431534ade42852387bda8c503834e	MATCH
08_REQUIRED_TAMPER_TESTS.py 

08_REQUIRED_TAMPER_TESTS

	8d4e001268ba91d7ca649fa617988e7f8991dbb37364caf7603273e4093030e5	MATCH
09_COMPLETE_EDGE_BASELINE_MANIFEST.json 

09_COMPLETE_EDGE_BASELINE_MANIF…

	4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5	MATCH

All four Python attachments pass syntax compilation. I also executed all 11 supplied tests in 08_REQUIRED_TAMPER_TESTS.py using a minimal import harness for the repository modules omitted from this packet; 11/11 passed. That test execution covers the attached relation, outcome, identity, CLI-surface, and tamper-helper logic, but it is not a substitute for rerunning the 37 source-dependent cases from the omitted repository.

OBSERVED
1. The complete mapping artifact is internally exact

I independently traversed every attached authoritative row. The following recompute without contradiction:

Frozen targets:                     4,788
Relation rows:                      4,788
Outcome rows:                       4,788
Complete all-pair membership rows: 16,988

PAIR_UNIQUE_COMPONENT_UNIQUE:                         4,772
PAIR_UNIQUE_COMPONENT_NONE_PARENT_OUTSIDE_FROZEN_TUBE:   16

The complete internal table hashes are:

frozen_target_projection_sha256
04d964eb0d855948d5522fc0338dfbaa83c7e5b0bd51c87ebf303aeedc7d634b

relation_table_sha256
dd57cfe2787fc04eaf49ed28c878514e44988fa852935f127df8b2397f876042

outcome_table_sha256
d7d9c422ecd861cb2684e9e4e749418308c57cc3eb96b32796f6a410c4a9d3e8

all_pair_membership_table_sha256
cd957ef0c6b83f302fb6a5e19e42426d018e65b8b178229b7460d4806ecbbcb1

threshold_hash_table_sha256
7abd75a9a89efdd66eb71b9828e6bcf3145b9b5c3d3c55c0d64f258f2ca0800f

Every relation and outcome record hash recomputes. Relation and outcome keys are canonically ordered and identical. Every target mode-set, frozen-target, mode-to-fragment, fragment-set, transverse-parent, and complete-table hash recomputes. All 37 per-case counts and hashes agree between the authoritative artifact, compact evidence, and complete-edge manifest. The compact evidence binds the exact external authoritative file SHA. 

03_CORRECTED_AUTHORITATIVE_MAPP…

 

04_INDEPENDENT_COMPACT_V2_EVIDE…

 

09_COMPLETE_EDGE_BASELINE_MANIF…

2. The prior all-pair scope defect is closed

The corrected authoritative table includes:

Complete baseline pair memberships:       16,988
Distinct target-lineage pairs:              4,783
Pairs not referenced by a frozen target:   12,205

The canonical 4,783-pair target-eligible set derived from the relation table has:

target_eligible_pair_ids_sha256
de7af6f03457a2a5f45b6121fc2dd4ece74fe00a1a83459aac1d72c9b83ac08d

The corrected core constructs one row for every pair context. The runner separately builds the complete membership relation from the frozen baseline and compares it against the relation regenerated from the newly reproduced baseline. The independent validator reconstructs the same relation directly from the candidate pool and per-case frozen baseline files, verifies the exact 16,988-row census, and rejects any row or table mismatch. 

06_CORRECTED_MAPPING_RUNNER

 

07_INDEPENDENT_MAPPING_VALIDATOR

This directly corrects the prior implementation, in which the purported “all-pair” digest covered only target-lineage pairs. 

01_INVALID_PREFLIGHT_RESPONSE

3. Target preservation and threshold preservation are exact comparisons

The corrected runner derives:

byte equality of the relation target projection, frozen-control projection, and regenerated target projection;

exact equality of target-ID sets;

exact equality of mode-hash sets;

exact threshold-object equality and matching frozen/regenerated threshold hashes.

The independent validator performs the same target projection, ID-set, mode-hash-set, and threshold-table comparisons from source-bound files before accepting the asserted criteria. Counts are supplementary invariants rather than substitutes for equality. 

06_CORRECTED_MAPPING_RUNNER

 

07_INDEPENDENT_MAPPING_VALIDATOR

4. All 16 structurally unreachable targets remain conservative negatives

The 16 absent component images occur in:

C001: 1
C002: 1
C005: 2
C007: 1
C020: 1
C024: 4
C027: 1
C030: 4
C036: 1

None is removed from the frozen population. For every such row:

the pair lineage remains unique;

the immutable parent is not in the pair’s frozen transverse-parent set;

component mapping count is zero;

baseline and replacement component IDs are null;

all unavailable before/after geometry remains null rather than zero;

added-rung count is zero;

reached, side-gate-advanced, novel-attributable, and after-certificate flags are false.

All 4,772 in-tube rows have one component image. Every out-of-tube diagnostic distance is greater than its frozen tolerance; every in-tube diagnostic distance is at or below its tolerance. The membership Boolean, rather than a newly rounded distance calculation, remains authoritative. This is exactly the two-state accounting required by the amendment. 

02_AMENDED_MAPPING_RULING

 

03_CORRECTED_AUTHORITATIVE_MAPP…

5. C001 remains exactly locked

The generic relation rule produces:

case_token                              C001
frozen_target_component_id              LTC_32d3609a79a9c314
frozen_target_pair_id                   LPP_daf852ef51c06429
frozen_target_parent_id                 P_47b0d4bd56e0f8ce

frozen_target_mode_ids
  M_73e7f7dd53e69d0f
  M_d5d9b9a524d05ccc

baseline_fragment_ids
  VEF_6fa481eb4e1f5bc8
  VEF_ca3d2d1302651a00

baseline_local_plane_pair_id            VEP_7610cbf5af1e3558
baseline_pair_sha256                     e48e7f2d53e2885a6bbc5e5af9256c81
                                         e1761b3b0e0bfe9efd104f9b65f805b2
target_parent_line_distance_mm           7.421545400596292
line_tube_tolerance_mm                   3.0
target_parent_in_frozen_transverse_set   false
baseline_component_mapping_count         0
complete_edge_baseline_component_id      null
relation_status                          PAIR_UNIQUE_COMPONENT_NONE_
                                         PARENT_OUTSIDE_FROZEN_TUBE

Its complete all-pair membership row contains eight transverse parents, has hash:

f8e662b8ad272a52e40afd3103fec2ff86baf111eb78948ea492e4a95d1397d8

and does not contain P_47b0d4bd56e0f8ce. The compact C001 row is byte-equivalent to the authoritative relation row. No C001-specific mapping branch appears in the corrected core. 

02_AMENDED_MAPPING_RULING

 

03_CORRECTED_AUTHORITATIVE_MAPP…

 

05_CORRECTED_V2_MAPPING_CORE

6. The five mandated tamper regressions are present

The attached tests reject:

A transverse-parent change to a pair not reached by a frozen target.

A target-ID exchange preserving row count.

A target-mode-hash exchange preserving row count.

A syntactically valid but incorrect threshold SHA-256.

An asserted true all-pair criterion over mismatching underlying evidence.

The test file additionally verifies the exact 20-criterion mapping surface, structural-negative outcome semantics, rejection of zero-imputed geometry, third-state rejection, V1 profile rejection, and absence of unauthorized validator CLI lanes. 

08_REQUIRED_TAMPER_TESTS

7. Safety remained mapping-only

The authoritative artifact contains:

added cross-parent rungs:          0
designated replays:                0
synthetic-resource executions:     0
truth inputs consumed:             false
ordinary batch authorized:         false
gold open authorized:              false

No target outcome claims a replacement component, added-rung effect, reached target, side-gate advancement, novel attribution, or valid after-certificate. The independent validator walks the artifact for forbidden keys, verifies the false safety flags, and reconstructs the safety criterion before producing compact evidence. 

03_CORRECTED_AUTHORITATIVE_MAPP…

 

07_INDEPENDENT_MAPPING_VALIDATOR

INFERENCE

The corrected packet satisfies the amended mapping contract.

The previous STOP did not question the two-state geometry represented by the 4,788 relation rows; it stopped because several required invariance predicates were not actually derived. Those deficiencies are now corrected at three levels:

source construction in the corrected mapping core;

frozen-versus-regenerated comparisons in the runner;

independent source reconstruction and mismatch rejection in the compact validator.

The mapping correction gives no experimental advantage to the 16 out-of-tube targets. They remain in the denominator and contribute false to every positive empirical target metric.

This authorization is not evidence that cross-parent rungs exist or work. No added rung has yet been constructed, so no inference follows concerning target reach, bilateral support, novel certificates, or the empirical viability of the LOCAL_PLANE_FRAGMENT family.

Exact next bounded ordinary V2 execution specification
1. Frozen identity

The ordinary batch must use exactly:

Python
Run
PRIMITIVE_PROFILE = (
    "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
)

PROBE_SCHEMA_VERSION = 2

PROBE_KIND = (
    "f1_v2_cross_parent_opposite_fragment_rung_probe_v2"
)

MAPPING_PREFLIGHT_KIND = (
    "f1_v2_cross_parent_rung_target_baseline_relation_preflight_v1"
)

TARGET_BASELINE_RELATION_RULE_ID = (
    "EXACT_FRAGMENT_PAIR_LINEAGE_PARTIAL_COMPONENT_IMAGE_V1"
)

BASELINE_PROFILE = (
    "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
)

RUNG_RULE_ID = (
    "COMPLETE_ADJACENT_PARENT_OPPOSITE_FRAGMENT_NO_TOPK_V1"
)

CONTROL_RUNG_RULE_ID = (
    "INHERITED_SAME_PARENT_CROSS_MODE_LOCAL_ONLY_V1"
)

V1 profile, schema, engineering, inference-semantic, or result artifacts must fail before execution. The original rung thresholds, geometry-gate order, line-equivalence rules, axial-bin semantics, hard caps, and thirteen-field input projection remain unchanged. 

02_AMENDED_MAPPING_RULING

2. Frozen evidence bindings

Before constructing any rung, the process must bind and verify:

inference_manifest_sha256
bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66

candidate_pool_manifest_sha256
c3ed8f6356c49c28ca56670cc9d061fe25dc4f1dea71625886e989330f8df3c7

frozen_control_manifest_sha256
3b9dfa156746a01cb38195f7d75c92e23bff839228de96e425b8902edf262510

complete_edge_baseline_manifest_sha256
4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5

mapping_preflight_external_sha256
9d314e7ebe5900f65c79039d8f729add3b8f1ad5cc4e190cc330103c9eec60cd

mapping_compact_evidence_external_sha256
869f733f48344d3a8cb3a1bb8aae7a28c6e6e10a60e16c43819affc9bccd3954

frozen_target_projection_sha256
04d964eb0d855948d5522fc0338dfbaa83c7e5b0bd51c87ebf303aeedc7d634b

mapping_relation_table_sha256
dd57cfe2787fc04eaf49ed28c878514e44988fa852935f127df8b2397f876042

mapping_outcome_table_sha256
d7d9c422ecd861cb2684e9e4e749418308c57cc3eb96b32796f6a410c4a9d3e8

all_pair_membership_table_sha256
cd957ef0c6b83f302fb6a5e19e42426d018e65b8b178229b7460d4806ecbbcb1

threshold_hash_table_sha256
7abd75a9a89efdd66eb71b9828e6bcf3145b9b5c3d3c55c0d64f258f2ca0800f

target_eligible_pair_count
4,783

target_eligible_pair_ids_sha256
de7af6f03457a2a5f45b6121fc2dd4ece74fe00a1a83459aac1d72c9b83ac08d

The target-eligible-pair hash is the SHA-256 of canonical JSON for the sorted unique baseline_local_plane_pair_id values in the authoritative 4,788-row relation table.

Any mismatch produces an integrity failure before rung construction.

3. Execution envelope

The authorized execution is exactly:

Cases:                         C001–C037
Case count:                    37
Frozen target count:           4,788
Target-eligible pair count:    4,783
Workers:                       1–8
Isolation:                     one fresh process per case
Peak RSS cap:                  strictly below 16 GiB per process
Gold/truth access:             0
Solver use:                    0
Ownership consumption:        0
Birth-column use:              0
Learning or threshold fitting: 0
Runtime integration:           0

This authorization does not include a designated replay or full synthetic-resource process.

4. Mandatory pre-rung gate

For each case, in an isolated process:

Hash the scan, candidate pool, and A0 prediction bytes.

Regenerate the frozen LPF control.

Regenerate the complete-SAME_SURFACE baseline.

Compare each regenerated payload byte-for-byte against its frozen source.

Reconstruct the exact frozen target projection.

Reconstruct the complete all-pair fixed-tube membership relation.

Reconstruct the exact threshold object.

Re-run the exact two-state relation mapping.

Reproduce the locked C001 row.

Derive every criterion from underlying objects with all criteria initialized to false.

Required pre-rung invariants:

Python
Run
len(frozen_targets) == 4_788
len(relation_rows) == 4_788
len(mapping_outcome_rows) == 4_788
len(all_pair_membership_rows) == 16_988

relation_target_projection == frozen_target_projection

set(relation_target_ids) == set(frozen_target_ids)
set(relation_mode_hashes) == set(frozen_target_mode_hashes)

all_pair_membership_rows == regenerated_all_pair_membership_rows
threshold_hash_table == regenerated_threshold_hash_table

component_image_present_count == 4_772
component_image_absent_outside_tube_count == 16

Any false invariant ends the execution as an integrity failure. No partial case result may enter aggregation.

5. Eligible-pair population

Rung construction is limited to the exact 4,783 unique pair IDs derived from all 4,788 pair-lineage rows.

The 16 out-of-tube targets still contribute their unique pair lineages to eligibility. They do not enter the pair-specific component graph, and their immutable parents may not be inserted into the frozen transverse-parent set.

The other 12,205 complete-edge pairs remain part of the invariance audit but are not target-eligible intervention pairs in this ordinary batch.

6. Only permitted intervention

For every target-eligible pair, the only permitted delta is construction of candidate rungs under:

COMPLETE_ADJACENT_PARENT_OPPOSITE_FRAGMENT_NO_TOPK_V1

The execution must retain exactly:

the complete SAME_SURFACE edge table;

the inherited SAME_PARENT CROSS_MODE_LOCAL seeds;

the two fixed complete-edge fragments;

the fixed pair anchor and axis;

the exact frozen transverse-parent set;

the frozen line-tube tolerance;

the 26-neighbor rule;

the opposite-fragment requirement;

the frozen axial-bin-width comparison;

all existing geometry and line-equivalence thresholds.

It may not introduce:

ranking
top-k selection
nearest-pair fallback
line-distance fallback
degree caps
new epsilon values
threshold fitting
scale retries
pair births
target births
a C001-specific branch
a structurally unreachable-target branch in rung construction

An out-of-tube target remains outside the graph even though its pair is processed.

7. Replacement components and target outcomes

The ordinary result must contain exactly 4,788 target outcomes.

For the 4,772 component-image-present targets

Retain the original V1 before/after component, geometry-gate, bilateral-support, line-equivalence, and causal-attribution semantics.

After adding edges, exactly one replacement component must contain the exact frozen target mode set. Zero or multiple replacement component matches fail closed.

For the 16 structurally unreachable targets

The following values remain mandatory regardless of all batch results:

Python
Run
complete_edge_baseline_component_id = None
replacement_component_id = None

causal_added_cross_parent_rung_count = 0
causal_added_cross_parent_rung_ids_sha256 = sha256_json([])

support_parent_count_before = None
support_parent_count_after = None
occupied_bin_count_before = None
occupied_bin_count_after = None
left_side_bin_count_before = None
left_side_bin_count_after = None
right_side_bin_count_before = None
right_side_bin_count_after = None
transition_rung_bin_count_before = None
transition_rung_bin_count_after = None
geometry_gate_depth_before = None
geometry_gate_depth_after = None
first_rejection_code_before = None
first_rejection_code_after = None

geometry_certificate_valid_after = False
target_reached_by_added_rung = False
target_side_gate_advanced = False
target_novel_rung_attributable = False

target_outcome_code = (
    "TARGET_STRUCTURALLY_UNREACHABLE_"
    "PARENT_OUTSIDE_FROZEN_PAIR_TUBE"
)

Zero may not replace null. These targets remain in every 4,788-target denominator.

8. Executed global deletion counterfactual

The ordinary batch must perform, rather than assert, the global deletion test:

Save the ordinary result and added-rung table.

Delete the entire added cross-parent rung table.

Invoke an independent rebuild function from the same frozen inputs.

Canonicalize all required baseline tables.

Compare the rebuilt result against the frozen complete-edge baseline.

At minimum, exact restoration is required for:

effective mode-edge table;

fragments and orphan table;

pair records;

pair transverse-parent memberships;

component records;

component geometry-gate records;

table counts and hashes.

The deletion attestation may become true only from exact equality. A constructed Boolean containing expected baseline hashes is prohibited.

9. Derived attestations and fail-closed criteria

Every ordinary-batch lock or integrity criterion must begin as false:

Python
Run
criteria = {
    key: False
    for key in ORDINARY_V2_CRITERION_KEYS
}

Each field may become true only after its specific executable predicate succeeds.

No literal true may substitute for checking:

opposite-fragment validity;

parent adjacency;

frozen-tube membership;

no pair births;

no orphan consumption;

target preservation;

baseline restoration;

causal-rung attribution;

input projection;

safety flags;

profile separation.

10. Required authoritative ordinary-batch artifact

The ordinary run must emit one full, untruncated authoritative artifact containing at least:

37 per-case source and reproduction records
complete added cross-parent rung table
complete target-eligible pair table
complete before-component table
complete replacement-component table
complete component/rung attribution tables
complete deletion-counterfactual evidence
exact 4,788-row target outcome table
per-case and aggregate censuses
all row and table SHA-256 values
maximum per-process RSS
all derived integrity criteria
all safety and authorization flags

No table contributing to a decision may be truncated or represented only by a count.

An independently executed validator must then reconstruct source-bound target, pair, threshold, safety, and deletion evidence and emit a compact gold-free professor evidence packet bound to the authoritative artifact’s externally computed SHA.

11. Ordinary-batch funnel to report

The packet must report, without fitting an expected value:

aggregate_added_cross_parent_rung_count

target_reached_by_added_rung_count
    denominator = 4,788

target_side_gate_advanced_count
    denominator = 4,788

target_novel_rung_attributable_certificate_count
    denominator = 4,788

target_component_image_present_count = 4,772
target_component_image_absent_outside_tube_count = 16

The 16 structural negatives contribute false to all three positive target metrics.

12. Result handling

During this execution:

Python
Run
if any_integrity_schema_cap_safety_or_deletion_criterion_is_false:
    packet_status = "INVALID_EXPERIMENT_STOP"
    empirical_family_conclusion = None

else:
    packet_status = (
        "ORDINARY_V2_BATCH_COMPLETE_NEXT_PROFESSOR_PACKET"
    )
    empirical_family_conclusion = None

The four empirical rung STOP codes are not yet applied in this ordinary-only execution, because designated replays and the full preregistered synthetic-resource envelope are not authorized here. The ordinary packet must preserve the exact funnel counts for the next professor decision.

No result, including a seemingly positive one, may open gold or authorize runtime work.

UNKNOWN

The attachment set does not include the raw scans, candidate-pool files, A0 prediction tree, frozen-control manifest and per-case control files, per-case complete-edge baseline JSON files, or the complete backend repository. I therefore did not independently rerun the 37 source-dependent cases or the full validator CLI from raw inputs. The attached artifacts, hashes, complete tables, manifest, implementation, and compact evidence are mutually consistent, and the corrected validator would require those omitted source files when executed.

It remains unknown whether the ordinary V2 batch will:

construct any eligible cross-parent rung;

reach any frozen target;

advance bilateral support;

produce any novel attributable certificate;

pass the executed deletion counterfactual;

pass later designated replays;

pass the eventual 4,000-parent, 16,000-mode, 1,688,000-edge resource process.

No conclusion follows regarding production applicability, calibration traceability, optical-material behavior, publication priority, legal novelty, patentability, runtime promotion, or main-branch suitability.

Authorization state
Python
Run
selected_decision = (
    "AUTHORIZE_ORDINARY_V2_BATCH_SPECIFICATION"
)

mapping_preflight_status = (
    "VALIDATED_UNDER_AMENDED_MAPPING_CONTRACT"
)

authorized_profile = (
    "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
)

ordinary_v2_batch_authorized = True
ordinary_batch_authoritative_artifact_authorized = True
ordinary_batch_independent_compact_evidence_authorized = True
next_professor_packet_authorized = True

designated_replays_authorized = False
full_synthetic_resource_run_authorized = False

truth_parsing_authorized = False
gold_open_authorized = False
ownership_consumption_authorized = False
birth_columns_authorized = False
learning_authorized = False
expected_count_logic_authorized = False
solver_authorized = False

runtime_integration_authorized = False
main_branch_promotion_authorized = False
production_inspection_authorized = False

local_plane_fragment_family_empirical_stop = False

The corrected mapping preflight passes. Exactly one bounded, gold-free ordinary V2 batch may now execute under the specification above.
