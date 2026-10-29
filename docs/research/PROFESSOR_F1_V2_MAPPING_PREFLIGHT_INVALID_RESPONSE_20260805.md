INVALID_MAPPING_PREFLIGHT_STOP

I choose option 2.

The 4,788-row relation table and 4,788-row outcome table are internally coherent, and I found no row-level contradiction in the two-state mapping census or the locked C001 record. The preflight nevertheless fails closed because the implementation does not actually establish all 20 required predicates.

The decisive defect is in:

mapping_preflight_criteria.all_pair_line_tube_memberships_are_unchanged

That criterion is propagated from an unconditional case attestation, and the live runner hashes only pair information appearing in target-relation rows—not every complete-edge baseline pair. The validator then trusts the Boolean rather than independently reconstructing and comparing the complete all-pair membership relation. This violates the amended ruling’s explicit requirement that every Boolean attestation be recomputed from its underlying rows. 

01_AMENDED_PROFESSOR_RULING

No ordinary V2 batch is authorized.

Attachment integrity
Attachment	Independently computed SHA-256	Result
00_GPT_PRO_PROMPT(8).md 

00_GPT_PRO_PROMPT

	5ab4801610fef90527a827dc71530d89d8d69fd0bea651853a6431450fe0eeff	No expected hash supplied
01_AMENDED_PROFESSOR_RULING.md 

01_AMENDED_PROFESSOR_RULING

	bf49709a1835ddefd95cdf175fa4bca089c067b0ed7afe4e10c47c7692bb2dd3	MATCH
02_AUTHORITATIVE_MAPPING_PREFLIGHT.json 

02_AUTHORITATIVE_MAPPING_PREFLI…

	a297ac88c508beffab59adaa8e85cd66a61fb85700072da953a27df9de105c1a	MATCH
03_COMPACT_VALIDATOR_EVIDENCE.json 

03_COMPACT_VALIDATOR_EVIDENCE

	6991fb7afb48da5825da159c8f0eef657198c55addbef5d00a74cd84bebf7b62	MATCH
04_V2_MAPPING_CORE.py 

04_V2_MAPPING_CORE

	2b48d6e721315a552fdbcc4933ee2ad8565fd40a789b210358f0359646365b12	MATCH
05_MAPPING_PREFLIGHT_RUNNER.py 

05_MAPPING_PREFLIGHT_RUNNER

	eb1822f0cd4f7b5f337964561aaff98d1ad93482fc2f1dbbb280c4a18f24ac77	MATCH
06_MAPPING_PREFLIGHT_VALIDATOR.py 

06_MAPPING_PREFLIGHT_VALIDATOR

	a51fac91daa2372f799880795aab6890c3aaf7b41747a8003377603b4e3eb4e7	MATCH
07_MAPPING_PREFLIGHT_VALIDATOR_TEST.py 

07_MAPPING_PREFLIGHT_VALIDATOR_…

	ac7e8358b18d8fb724ecd3ee2980459bf48784cac7317e2d196673a1dec36885	MATCH
08_COMPLETE_EDGE_BASELINE_MANIFEST.json 

08_COMPLETE_EDGE_BASELINE_MANIF…

	4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5	MATCH
09_FeatureExtractionFromPointClouds(2).pdf 

09_FeatureExtractionFromPointCl…

	21e948aff638d8c09883d132117ea37e95bd8afa47a21d80aeb268666b61b65d	MATCH to declared PDF hash

All four Python attachments pass syntax compilation.

OBSERVED
1. The authoritative tables themselves are internally consistent

I independently parsed and traversed every authoritative relation and outcome row.

The following recompute exactly:

Relation rows: 4,788

Outcome rows: 4,788

Relation-table SHA-256:
dd57cfe2787fc04eaf49ed28c878514e44988fa852935f127df8b2397f876042

Outcome-table SHA-256:
d7d9c422ecd861cb2684e9e4e749418308c57cc3eb96b32796f6a410c4a9d3e8

Relation-state census:

PAIR_UNIQUE_COMPONENT_UNIQUE: 4,772

PAIR_UNIQUE_COMPONENT_NONE_PARENT_OUTSIDE_FROZEN_TUBE: 16

Every relation and outcome row has the exact declared schema.

Every per-row record hash recomputes.

Every frozen mode-set hash, frozen-target record hash, mode-to-fragment hash, and fragment-set hash recomputes.

Relation and outcome tables have identical, canonically ordered target keys.

All 16 structurally unreachable outcomes preserve null geometry rather than imputing zero.

No outcome claims a rung, replacement component, reached target, side-gate advancement, novelty, or after-certificate.

All 37 per-case counts and table hashes agree between the authoritative artifact and compact evidence.

Compact cases_sha256 and compact_evidence_sha256 recompute.

The compact evidence binds the exact authoritative file SHA. 

02_AUTHORITATIVE_MAPPING_PREFLI…

 

03_COMPACT_VALIDATOR_EVIDENCE

The 16 structurally unreachable rows occur in:

C001: 1
C002: 1
C005: 2
C007: 1
C020: 1
C024: 4
C027: 1
C030: 4
C036: 1

None is removed from the 4,788-target population.

2. C001 satisfies the amended locked regression

Exactly one row exists for:

C001/LTC_32d3609a79a9c314

It has:

frozen_target_pair_id                 = LPP_daf852ef51c06429
frozen_target_parent_id               = P_47b0d4bd56e0f8ce
frozen_target_mode_ids                = [
                                          M_73e7f7dd53e69d0f,
                                          M_d5d9b9a524d05ccc
                                        ]
baseline_fragment_ids                 = [
                                          VEF_6fa481eb4e1f5bc8,
                                          VEF_ca3d2d1302651a00
                                        ]
baseline_pair_mapping_count           = 1
baseline_local_plane_pair_id          = VEP_7610cbf5af1e3558
baseline_pair_sha256                  = e48e7f2d53e2885a6bbc5e5af9256c81
                                         e1761b3b0e0bfe9efd104f9b65f805b2
target_parent_line_distance_mm        = 7.421545400596292
line_tube_tolerance_mm                = 3.0
target_parent_in_frozen_transverse_set = false
baseline_component_mapping_count      = 0
complete_edge_baseline_component_id   = null
target_baseline_relation_status       =
  PAIR_UNIQUE_COMPONENT_NONE_PARENT_OUTSIDE_FROZEN_TUBE

The compact evidence’s C001 row is byte-equivalent to the authoritative relation row, including its record hash. This is produced by the generic relation rule, not by an evident C001-specific branch. 

01_AMENDED_PROFESSOR_RULING

3. The fatal all-pair membership predicate is not computed

In the mapping core, the following case attestation is assigned directly:

Python
Run
"pair_line_tube_membership_unchanged": True

There is no preceding exact comparison of the complete baseline pair-membership population supporting that Boolean. 

04_V2_MAPPING_CORE

The live runner then constructs pair_membership_rows from the target relation rows:

Python
Run
[
    row["baseline_local_plane_pair_id"],
    row["baseline_transverse_parent_count"],
    row["baseline_transverse_parent_ids_sha256"],
]
for row in relations

It does not iterate over the complete baseline pair population. It subsequently defines:

Python
Run
"all_pair_line_tube_memberships_unchanged":
    attestation["pair_line_tube_membership_unchanged"] is True

Thus the Boolean is not a comparison result; it is a round trip from the literal in the core. 

05_MAPPING_PREFLIGHT_RUNNER

My complete-file census shows the scope error:

Complete-edge baseline pair records across C001–C037: 16,988
Distinct case/pair lineages referenced by targets:       4,783
Baseline pairs absent from the purported all-pair audit: 12,205

Examples:

C001:
  complete-edge baseline pairs = 842
  target-lineage pair keys      = 260

C030:
  complete-edge baseline pairs = 1,497
  target-lineage pair keys      = 469

The baseline manifest reports the complete pair counts, while the mapping artifact contains only target-lineage rows. 

08_COMPLETE_EDGE_BASELINE_MANIF…

 

08_COMPLETE_EDGE_BASELINE_MANIF…

 

02_AUTHORITATIVE_MAPPING_PREFLI…

The stronger all-pair calculation visible earlier in the runner does not cure this defect. The file explicitly redefines the function and states that the later, weaker definition is the callable binding used by the worker and tests. 

05_MAPPING_PREFLIGHT_RUNNER

4. The validator does not repair the defect

The validator checks only:

Python
Run
case["all_pair_line_tube_memberships_unchanged"] is True

It does not:

regenerate the complete all-pair membership table;

compare pair_tube_membership_sha256 against an independently computed expected value;

verify that the digest covers local_plane_pair_count pairs;

verify that the serialized transverse-parent hash corresponds to a frozen complete-edge membership set.

It then accepts the already-true criteria object and copies the case digest into compact evidence. 

06_MAPPING_PREFLIGHT_VALIDATOR

 

06_MAPPING_PREFLIGHT_VALIDATOR

Consequently, a changed membership hash for a non-target pair cannot be detected because that pair is absent from the digest entirely. Even for a target-lineage pair, the validator checks that baseline_transverse_parent_ids_sha256 has SHA syntax, but it has no complete parent-ID row from which to recompute that value.

5. Related predicate implementations are also weaker than their names

These are secondary defects; criterion 19 alone is sufficient to invalidate the packet.

The live runner implements:

Python
Run
target_ids_equal =
    len(relations) == len(unique_relation_ids)

That establishes uniqueness, not equality to the frozen-control target-ID set.

It implements:

Python
Run
target_mode_hashes_equal =
    len(relations) == frozen_target_count

That establishes a count, not equality of mode-hash sets.

It implements:

Python
Run
target_rows_byte_identical =
    attestation["frozen_target_rows_regenerated_from_control"] is True

That trusts another literal rather than comparing the frozen target projection row by row.

It implements:

Python
Run
all_threshold_hashes_are_unchanged =
    all(bool(case["thresholds_sha256"]) for case in cases)

That establishes only that each string is nonempty, not that it equals the frozen threshold hash.

Finally:

Python
Run
gold_open_authorized_false = True

is assigned directly in the criteria builder. The top-level artifact and validator separately establish the safety flag as false, so I found no gold opening, but the criterion itself is still not derived in the required fail-closed manner. 

05_MAPPING_PREFLIGHT_RUNNER

 

05_MAPPING_PREFLIGHT_RUNNER

The attached test file contains six tests. It verifies the two relation states, null rather than zero geometry, third-state rejection, V1 identity rejection, and CLI surface restrictions. It does not test complete all-pair membership recomputation, exact target-set equality, exact mode-hash-set equality, threshold-hash comparison, or rejection of a falsely asserted attestation. 

07_MAPPING_PREFLIGHT_VALIDATOR_…

Audit of the 20 mapping-preflight criteria

“Pass” below means directly supported by the attached complete tables. “Packet-bound” means internally hash-consistent but not independently rerunnable because the underlying manifests, candidate pools, A0 files, backend modules, and scans are not attached. “Not derived” means the actual predicate implementation is weaker than the required predicate.

#	Criterion	Audit result
1	all_declared_source_hashes_match	PACKET-BOUND. Declared hashes are mutually consistent; only the complete-edge baseline manifest is attached.
2	all_frozen_control_probe_hashes_reproduce	PACKET-BOUND. Per-case control hashes cross-link consistently, but the frozen-control manifest and generated controls are absent.
3	all_complete_edge_baseline_hashes_reproduce	PACKET-BOUND / CONSISTENT. All 37 case hashes match the attached baseline manifest; no independent baseline rerun was possible.
4	frozen_target_count_is_4788	PASS.
5	frozen_target_rows_are_byte_identical	NOT DERIVED by the live audit. Row hashes are internally valid, but the predicate trusts a Boolean instead of comparing to the frozen-control rows.
6	frozen_target_id_sets_are_exactly_equal	NOT DERIVED. The live predicate checks uniqueness, not exact set equality.
7	frozen_target_mode_hash_sets_are_exactly_equal	NOT DERIVED. The live predicate checks count, not exact set equality.
8	every_target_mode_has_one_baseline_fragment	PASS for all 4,788 rows.
9	every_target_maps_to_exactly_two_fragments	PASS for all 4,788 rows.
10	every_target_has_exactly_one_pair_lineage	PASS for all 4,788 rows.
11	every_in_tube_target_has_one_component_image	PASS for 4,772 rows.
12	every_out_of_tube_target_has_zero_component_images	PASS for 16 rows.
13	no_unrecognised_relation_status_exists	PASS. Exactly two states occur.
14	relation_row_count_is_4788	PASS.
15	relation_table_is_untruncated	PASS. Canonical order and 4,788 unique target keys.
16	outcome_row_count_is_4788	PASS. Exact identity and order match the relation table.
17	c001_locked_regression_matches	PASS.
18	all_threshold_hashes_are_unchanged	NOT DERIVED by the criterion; it checks only nonempty hashes. Exact baseline reproduction may indirectly constrain the values, but the mandated comparison is absent.
19	all_pair_line_tube_memberships_are_unchanged	FAIL-CLOSED. Literal attestation, partial target-lineage digest, and no independent validator recomputation.
20	gold_open_authorized_false	PASS in the artifact. Top-level flags are false, no forbidden gold/truth payload was found, and no positive outcome is claimed; the criteria-builder assignment itself remains unconditional.

The amended ruling requires all 20, not merely the row-derived subset. A falsely or incompletely established criterion must stop the amended experiment before batch. 

01_AMENDED_PROFESSOR_RULING

INFERENCE

The two-state mapping rule itself appears to have been implemented correctly for the 4,788 attached targets:

Pair lineage is total.

Component image is unique for 4,772 targets.

Component image is absent for 16 targets whose immutable parent is outside the frozen pair tube.

Those 16 targets remain fully represented.

Their geometry fields remain null.

They receive no rung credit and are not counted as successful.

C001 remains outside the exact 3 mm tube.

The defect is therefore not evidence that the relation table is geometrically wrong. It is evidence that the required invariance proof and fail-closed decision machinery are incomplete. Under the amended contract, “probably unchanged” or “unchanged for the target-used pairs” cannot substitute for exact all-pair verification.

No empirical LOCAL_PLANE_FRAGMENT conclusion follows. No cross-parent rung was built, so this is an integrity STOP for the mapping preflight, not an empirical family-ending STOP.

Smallest mapping-only correction

No target, pair, tube, threshold, C001 value, relation status, or outcome semantic needs to be changed.

1. Correct the live per-case audit

After regenerating the complete-edge baseline, construct one canonical row for every baseline pair:

Python
Run
all_pair_membership_rows = [
    {
        "baseline_local_plane_pair_id": pair_id,
        "transverse_parent_ids":
            sorted(pair_context["transverse_parent_ids"]),
        "transverse_parent_ids_sha256":
            sha256_json(sorted(pair_context["transverse_parent_ids"])),
    }
    for pair_id, pair_context in sorted(all_pair_contexts.items())
]

Required invariants:

Python
Run
len(all_pair_membership_rows) == baseline_case_counters[
    "local_plane_pair_count"
]

len({
    row["baseline_local_plane_pair_id"]
    for row in all_pair_membership_rows
}) == len(all_pair_membership_rows)

pair_tube_membership_sha256 must hash this complete table—not the 4,788 target-relation projection.

2. Make the comparison independent

Regenerate the expected all-pair membership table in an independent mapping-only function call from the frozen complete-edge inputs and compare the canonical rows exactly:

Python
Run
pair_line_tube_membership_unchanged = (
    regenerated_all_pair_membership_rows
    == frozen_all_pair_membership_rows
)

The Boolean may become true only from this equality.

3. Replace the weak target predicates

Derive the three target-preservation predicates directly:

Python
Run
relation_target_projection == frozen_target_projection

set(relation_target_ids) == set(frozen_target_ids)

set(relation_mode_hashes) == set(frozen_target_mode_hashes)

Counts and uniqueness are additional checks, not substitutes for equality.

4. Derive threshold and safety predicates
Python
Run
all_threshold_hashes_are_unchanged = (
    regenerated_threshold_hashes
    == frozen_complete_edge_threshold_hashes
)

gold_open_authorized_false = (
    artifact["gold_open_authorized"] is False
    and artifact["truth_inputs_consumed"] is False
    and no_forbidden_truth_or_gold_keys
)
5. Make the validator recompute rather than trust

The validator must independently reconstruct:

exact frozen target projection;

exact target-ID set;

exact target-mode-hash set;

complete all-pair membership table and hash;

exact threshold hash;

safety predicates.

It must reject a packet whose criteria object says True when any reconstructed predicate is false.

At minimum, add tamper tests proving rejection when:

A transverse parent is changed for a baseline pair not reached by any frozen target.

A target ID is exchanged while total count remains 4,788.

A target mode hash is exchanged while total count remains 4,788.

A valid-looking but wrong 64-character threshold hash is supplied.

all_pair_line_tube_memberships_are_unchanged=True is supplied over a mismatching complete-pair table.

6. Rerun only the mapping preflight

The corrected execution remains bounded to:

37 cases
4,788 exact frozen targets
4,788 untruncated relation rows
4,788 untruncated outcome rows
0 added cross-parent rungs
0 designated replays
0 synthetic-resource execution
0 gold/truth access
0 ownership consumption
0 solver use
0 runtime or production integration

The corrected packet must carry a new external authoritative SHA and newly derived compact evidence. The existing relation and outcome rows may remain byte-identical if the corrected audit reproduces them; only the attestation and case-level membership hashes necessarily need to change.

UNKNOWN

Whether all 16,988 complete-edge baseline pair memberships were in fact unchanged. The packet’s current evidence does not establish that proposition.

Whether the exact frozen target and mode-hash sets match the omitted frozen-control manifest, notwithstanding the internally valid target hashes.

Whether the ordinary cross-parent intervention produces any eligible rung, reaches any in-tube target, advances bilateral side support, or produces an attributable novel certificate.

Whether the eventual deletion counterfactual, designated replays, and full 4,000-parent / 16,000-mode / 1,688,000-edge resource envelope will pass.

Raw source geometry and the backend modules are not attached, so I did not independently regenerate the 37 baselines or recalculate C001 from source coordinates.

The historical paper remains only a historical anchor. Its page 6 wedge diagrams describe grouping points around two planes; page 7 describes fitting two planes to the two groups and using their intersection as the local crease; page 8 extends the grouping and plane-fitting treatment to junctions. Those observations neither prove applicability to this scan regime nor support a publication-priority, legal-novelty, or patentability conclusion. 

09_FeatureExtractionFromPointCl…

 

09_FeatureExtractionFromPointCl…

Authorization state
Python
Run
selected_decision = "INVALID_MAPPING_PREFLIGHT_STOP"

ordinary_batch_authorized = False
designated_replays_authorized = False
full_synthetic_resource_run_authorized = False
compact_result_bundle_authorized = False

truth_parsing_authorized = False
gold_open_authorized = False
ownership_consumption_authorized = False
birth_columns_authorized = False
learning_authorized = False
expected_count_logic_authorized = False

runtime_integration_authorized = False
main_branch_promotion_authorized = False
production_inspection_authorized = False

local_plane_fragment_family_empirical_stop = False

The current authoritative mapping artifact is stopped. The smallest acceptable next submission is a corrected mapping-only preflight that derives and independently validates the exact predicates above.
