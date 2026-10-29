# V9 material identity chain: source-only delta audit

Date: 2026-09-13. Reviewer: Codex/Astra `v9_material_identity_chain`.
This bounded review follows the initial material-identity chain audit. Fable
remains the sole editor; these are changing, unfrozen sources, not a declared
finished candidate. No imports, tests, geometry jobs, physical inputs, Gold,
attempts, controls or caches were accessed. Only this new report was written.
AGENTS and current status were reread. The initial layer-design and chain
audits remain unchanged.

## Decision

The original policy-grammar and contact-schema blockers are **closed in this
source snapshot**, but connected execution is not yet verified. Two actionable
issues remain: ordinary canonical-fit refusal lacks a field required by its
publication validator, and unverified/disabled layer modes still grant
`inner_face_claim_eligible=true`. The latter is not required to run a generic
conditional contact-generator F1 and should not block that diagnostic once
the overclaim is removed and the contract remains explicit.

## Snapshot binding

These five hashes were stable before and after the read. Findings apply only
to this snapshot; later edits require a bounded delta check.

| Source | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `9a2e7da3a58838c5de74085b197547550bcbcd1b53804754b609bba5302c9baa` |
| `backend/cad_guided_context_measurement_v2.py` | `b04affdb627bcf68586c9017337febc2cef76e18c6a7a86564c7b5a9e66c7edc` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `a298b123218d931911e064a2f35ab52e5553f2c1ad5b25a71e3c149c2d7d9234` |
| `scripts/gold11_contact_scoring_v3.py` | `f4fe9b4d70f6eda9fc32912deb80d2541f8d6a050e37812579b3bf489c73cbd4` |
| `scripts/gold11_schedule_projection_v4.py` | `32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708` |

The reused exact-cover planner V1 remains
`70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47`;
frozen contact scorer V2 remains
`9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff`,
both matching the initial review. Named test snapshots inspected:

- context V2: `8a2ae1da718c01a4e66016713adca87b446c11d464bfe6a87a16fcdc3fe574a2`;
- cover V3: `2f2880006a401aae855e42b8224634afec19598c562af11609232a5b78dd7843`;
- schedule projection V4: `2849158aad364c4282f309fb9fcf35ed9844b91c404ca2e38834f8dd9e853192`;
- kernel V2: `9e814c7eeb518757512b1984aa79b8a51032dc6e9552daea2cb3411f221bac27`.

## Closed at source: scalar policy and layer policy are distinct, bound records

Context V2 `_fit_policy` now returns the original seven-field scalar fit
grammar (`:72–75`). `_layer_policy_record` separately serializes the eight
layer-selection fields (`:78–81`). Each support identity includes both
records (`:353–356`), each support record retains both (`:361–364`), and the
context publication validator checks the layer policy and equality of every
record to the common source policy (`:480–485`, `:541–543`).

The original exact-cover planner therefore receives the grammar it expects.
Its policy equality remains meaningful because the context validator first
establishes that all source records have the same separately declared layer
policy. Cover V3 now checks scalar fit policy and separate layer-policy
equality before refit (`:181–188`), and rechecks group/witness/source equality
on publication (`:306–315`). The earlier guaranteed planner rejection and
masked scalar-versus-extended-dictionary comparison are no longer present.

This is static closure, not proof an actual discovery-to-cover positive has
executed. Retaining `layer_policy` separately is not evidence erasure: it is
included in the new raw-support record identity and retained publication.

## Closed at source: V4 actually uses the new contact V3 grammar

Contact V3 recognizes cover V3 and source context V2, binds the kernel
implementation, retains all material enums and carrier roles, and validates
source/group layer-policy equality. Its scalar fit-policy grammar remains the
original seven fields. The V4 wrapper now calls its new
`_project_schedule_with_contact_v3` helper, which uses the existing schedule
request/evidence helpers around the new contact V3 projector. It no longer
delegates the actual result to frozen contact V2 (`:186–227`, `:257`).

A narrow read of the V9 runner source also finds contact V3 selected in its
module map and `score_gold11_contact_workflows_v3` called by the scorer
entrypoint (`scripts/run_gold11_mesh_development_v9.py:541`, `:1877`). This is
wiring inspection only, not a runner/protocol/launcher audit.

The V2-to-V3 contact source diff shows no changes to the endpoint-copying
computation, geometric matcher imports, thresholds (5 degrees, 4 mm, 0.60
overlap in both directions), one-to-one matching, selected-case denominator,
reference-count requirement, FN retention for failures, or geometric metric
aggregation. The V4 wrapper preserves original result bytes and reconstructs
the returned projection before exposing it. New identity fields and diagnostic
category counts are added; endpoint values are not adjusted to compensate for
identity or schema differences.

These are source-comparison findings; no numerical regression was executed.
The new grammar deliberately changes which payload schema is recognized, not
the geometric matching rule. No historical comparability follows from it.

## P1 — Expected canonical-fit refusal becomes a publication failure

In context V2, a new association is constructed without
`inner_face_claim_eligible` (`:370–376`). That field is added only inside the
successful canonical-fit path (`:393–402`). The explicit `_Unavailable`
handler marks the record unavailable (`:403–404`) without filling the field.

The publication validator loops over all associations before it branches on
record status. At `:562–564` it requires
`association.get("inner_face_claim_eligible") == record["inner_face_claim_eligible"]`.
For an expected canonical-fit refusal, the left side is `None` and the right
side is a boolean. Validation therefore fails even though the producer
deliberately retained the record as an ordinary unsupported geometry outcome.
When called from cover V3, that exception becomes a fatal source-publication
failure instead of a retained per-record refusal with other valid records.

Bind the already-known identity on association creation, independently of
whether the later canonical numerical fit succeeds. Keep absent canonical
quantities explicitly absent/unknown; do not fabricate a radius deviation to
satisfy the validator. A focused test must force one normal canonical
`_Unavailable` after successful per-context support discovery, retain another
successful disjoint record, and validate the entire publication and transport.
This is a source proof of a reachable exception-path mismatch, not a claim
that the real corpus necessarily reaches that branch.

## P1 — Disabling selection still grants an inner-specific claim

The kernel and contact V3 both list these two statuses in
`INNER_FACE_ELIGIBLE_STATUSES`:

- `CONTEXT_ASSUMED_NOT_VERIFIED`;
- `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED`.

Kernel `_measure_arc` explicitly states that these modes may carry
inner-face-specific claims, derives the boolean from this enum, and chooses
`INNER_FACE_CONDITIONAL_CONTACT_LINE_UNVERIFIED` as the carrier role. Context
and cover now use that boolean to decide whether to compute a canonical
radius-to-nominal-inner-radius difference. Contact V3 counts such carriers as
`inner_face_eligible_carrier_count`.

The disabled-mode counterexample does not require novel physics: use a clean
outer-only surface at 2.3 mm radius against nominal inner radius 0.8 mm.
Enabled selection produces singleton/unresolved, a generic carrier and no
inner radius deviation. Disabling the selection changes the status to
`CONTEXT_ASSUMED_NOT_VERIFIED`, sets eligibility true and permits a +1.5 mm
inner-radius deviation from the same outer-only observation. A feature toggle
does not supply evidence of which material side was observed.

The new two-group compatibility checks improve the search hypothesis, but
their result explicitly remains `UNVERIFIED_MATERIAL`. The nominal outer-arc
bound is now correctly labelled `certificate=false`, with unverified
assumptions. It is not used as a side-isolation certificate. Those facts do
not justify eligibility for an actual inner-surface answer either.

Minimum closure: keep `inner_face_claim_eligible=false` for every currently
unverified status, or rename the field and all its consumers to a strictly
conditional hypothesis that cannot be interpreted as eligibility for actual
inner metrology. Do not allow an unverified conditional hypothesis to trigger
an asserted inner-radius deviation. Auxiliary surface radius and generic
contact-generator geometry can remain available. A future true eligibility
path requires separately bound material-side evidence.

This is **not** a request to withhold all generic conditional contact lines.
Contact V3 expressly declares generic contact localization, no manufacturing
event metric, unknown observed counts, indeterminate conformance and no
production use. That diagnostic may proceed under its frozen, explicit scope
without solving material-side identification first. Removing the false
inner-eligibility branch does not require changing the matcher or endpoints.

## Identity propagation: improved, with two bounded omissions

The current source checks window-to-record identity, canonical measurement
and instance identity, full-cover witness-to-group identity, and
group/measurement/carrier-to-source witness identity. Contact V3 additionally
requires a real boolean (`is eligible`) and preserves the status in compact
carriers. This closes the prior missing cover-witness equality.

Two structural gaps remain:

1. Context publication never compares the retained
   `association.per_context_measurements` identity fields to its record or
   window; only the association's separate eligibility bit is checked. A
   contradictory original kernel-side status can remain embedded in the
   evidence. Validate that propagation link as well.
2. Context `_bind_identity` uses ordinary `==` for every field, so 0/1 are
   accepted as false/true. Contact V3 rejects non-boolean eligibility on its
   checked source/measurement/carrier rows, but that does not close every
   context/window publication boundary. Use type-strict equality for the
   boolean claim.

These are P2 structural consistency findings, not allegations that the
current producer actively falsifies its own record. They should be covered by
the small tamper matrix rather than a separate research phase.

## Connected-test evidence remains incomplete in the inspected WIP

The revised context tests still mock `_measure`; the revised cover tests
still mock actual context discovery. The cover fixture now explicitly fills
`inner_face_claim_eligible` on its unavailable associations, so it cannot
catch the real producer's missing-field refusal path. Its fixture record ID
also still omits `layer_policy`, unlike actual new discovery.

The inspected V4 wrapper test source remains the same hash as the initial
review and still embeds the old contact payload before changing only
workflow/context metadata. No new `test_gold11_contact_scoring_v3.py` appeared
in the bounded named test inventory. This is an unfinished-test observation,
not a statement about what Fable will complete next or a claimed test failure.

The minimum connected acceptance remains: one analytically constructed
positive using the real measurement/discovery/cover seam; one ordinary
canonical refusal coexisting with a positive; finite-support refusal; and
exact-byte V3/V4 projection/scoring with unchanged endpoints. Cover enabled,
disabled and nondefault layer policy, singleton/outer-only generic output,
identity tampering and late whole-case failure. Keep physical observed count
unknown and conformance indeterminate throughout. No physical rerun is needed
to expose these interface defects.

## Handoff

The two P1s and bounded equality/test gaps were sent to the root promptly.
Next is the sole editor's source closure and actual focused integration
checks, followed by independent exact-hash delta review. No frozen attempt
was altered; no score or promotion is claimed by this report.

## Publication-time WIP addendum: later closures observed

Fable edited sources after the main snapshot read while this report was being
written. This addendum supersedes the corresponding open-state statements
above, without rewriting the findings for the earlier exact hashes. Only the
named changed branches below were reread; this is not a complete review of
every later source edit.

- Kernel became
  `cb63853d45d71a54aae337d4e59c0092d1bd0e8fc939625b0917c04356676328`.
  Contact V3 became
  `a38c4b02db48b4163866a0e202d1aa1063bff664d82c1c5dd642df025324ee35`.
  Both remove `CONTEXT_ASSUMED_NOT_VERIFIED` from inner eligibility. The
  disabled-mode overclaim is therefore **closed at source**. Both still list
  `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED` as inner eligible;
  that narrower semantic issue remains. Generic contact scoring remains
  allowed without granting that inner-specific claim.
- Context V2 first became
  `91bf7cb7da3b3784ec80a65e8b2f1424315dda7fb02c3398ff4d1f9f6d4cc708`.
  Association construction now sets eligibility and initializes canonical
  deviations to `None` before refitting. This source-closes the missing-field
  ordinary-refusal P1. Its tightened validator initially introduced a second
  mismatch: successful records later invalidated by overlapping support
  retained nonnull canonical deviations while unavailable status prohibited
  them. This transient regression was sent immediately to the coordinator.
- The next context V2 snapshot was
  `e1e93847ada9f4f1658bdba7369f2deeda71e8ff99eaf4b70fcbb5aabf4c75ad`.
  `_invalidate_overlapping_records` now clears both active canonical
  association deviations before marking a record unavailable (`:463–467`).
  This source-closes that transient overlap/publication incompatibility.
  Original per-context measurements and raw supports remain in the records;
  no unavailable record is relabeled independently measured.

At this stopping point the actionable open semantics are the still-unverified
two-layer inner eligibility, missing per-context identity revalidation and
non-type-strict `_bind_identity`. The connected positive/refusal/overlap
transport tests remain unverified by this reviewer. Do not report the two
earlier transient WIP refusal mismatches as currently open after these
observed corrections, or treat their source closure as an executed test pass.
