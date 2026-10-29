# V9 material-side identity and connected publication audit

Date: 2026-09-13. Reviewer: Codex/Astra `v9_material_identity_chain`.
Scope: independent, source-only review of the new measurement kernel, context
consumer, support-cover consumer and schedule/contact publication chain. The
layer-design audit was read completely. No imports, tests, geometry execution,
physical input, Gold, attempt, control, private cache or credentials were
accessed. Only this new report was written. Fable remains the sole engine/test
editor. This is an unfrozen source-snapshot audit, not a PRE or physical POST.

## Decision

**Not green for the connected V9 run at this snapshot.** Two deterministic
interface incompatibilities prevent an actual new measurement from reaching
the scorer. The new material-side strings are carried into measurements and
contact carriers, but are not an inner-identity eligibility gate. Physical
observed-bend count and conformance nevertheless remain appropriately unknown
and indeterminate in the reviewed V16 path.

An important distinction: the existing contact scorer explicitly evaluates
conditional raw-coordinate contact-generator geometry, not verified inner
arcs. This review does not impose verified material identity as a prerequisite
to every exploratory contact-only diagnostic. It does require preserving the
actual ambiguity, keeping inner-specific claims ineligible, and resolving the
declared contract before adapting the new bytes. Accidental rejection of every
new schema is not an intentional material-side safeguard.

## Exact reviewed snapshot

The first four sources were hashed before and after inspection with matching
hashes. Directly required sources and inspected test seams were also hashed.
Line references below apply only to these snapshots, not later Fable edits.

| Source | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d` |
| `backend/cad_guided_context_measurement_v2.py` | `cc10db40e376ed0786f27babcac40b8261633955a8c9926331f8c9d9414e4ce0` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `3f53d054165468d9197f7e0e0b3c6b564267880906377c1fb95f964457b2b74d` |
| `backend/cad_guided_mesh_offline_v16.py` | `441c618d4998edaca0d401dd22f14d1444f248750a807386fecef19741708769` |
| `scripts/gold11_schedule_projection_v4.py` | `06869ff7bb8761eae976f294e733713c3f2d94a83ee0c8d756c90d50afccfa90` |
| `scripts/gold11_schedule_projection_v1.py` | `a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245` |
| `scripts/gold11_contact_scoring_v2.py` | `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff` |
| `backend/cad_guided_exact_support_cover_v1.py` | `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47` |
| `tests/test_cad_guided_measurement_v2.py` | `13e2fbf913313cef2ce78a61e2dc4d3a7f16afd75f405bb46188d9347a66f67b` |
| `tests/test_cad_guided_context_measurement_v2.py` | `720c6209654767911aebc254ba1f0ee3253c2f6ff8bccabd7aca9083d71ae2fd` |
| `tests/test_cad_guided_support_cover_measurement_v3.py` | `68ee5051fb0d2d394cb5bdb5f0ac4ab634d3ab8c1168d6459029d3122947b809` |
| `tests/test_gold11_schedule_projection_v4.py` | `2849158aad364c4282f309fb9fcf35ed9844b91c404ca2e38834f8dd9e853192` |

## P1 — Every nonempty real V2 support ledger fails the old cover policy grammar

`cad_guided_context_measurement_v2._fit_policy` adds `layer_selection` whenever
the layer-policy argument is present (`:59–63`). Its public discovery function
always constructs and validates a concrete `LayerSelectionPolicyV1`, including
when the caller passes no policy (`:200–205`). Every real support record and
its identity therefore include that extra policy object (`:332–342`).

Support-cover V3 passes those records unchanged to the frozen exact-support
planner V1 (`cad_guided_support_cover_measurement_v3.py:157`). That planner
requires exactly the seven original numeric/fit-model keys
(`cad_guided_exact_support_cover_v1.py:36–40`, called at `:94`). Consequently
the first nonempty record produces `INVALID_COVER_FIT_POLICY_FIELDS`; the
planner returns an unavailable whole plan, and the consumer returns
`SUPPORT_COVER_PLAN_REFUSED`. Zero records may pass this point, but cannot
produce a useful carrier.

There is a second, masked incompatibility in the same chain. After a cover is
found, support-cover V3 compares `_v1._fit_policy(feature)` without the layer
argument against the witness's extended policy (`:179`). That also cannot be
equal for a real V2 record. Fixing only the planner reveals
`SUPPORT_COVER_FIT_POLICY_BINDING` and whole-report fatal clearing instead.

This is a source proof from unconditional dictionary construction and exact
key/equality checks, not an executed reproducer. The minimum repair is a
versioned, bounded policy representation understood by discovery, set-cover
planning, canonical refit, publication and transport. Preserve the original
layer selection declaration and record identity; do not drop it merely to make
the old grammar accept a new mechanism. Do not edit frozen predecessor modules.

### Why the inspected tests can miss this

The new cover test suite mocks the discovery function and constructs its
source ledger with `_fit_policy(feature)` without a layer policy
(`tests/test_cad_guided_support_cover_measurement_v3.py:142`, mock installed at
`:197`). It thus supplies the old seven-key shape that real V2 discovery never
emits. Its passing refit tests would not establish compatibility of the actual
discovery-to-cover seam. No test was run by this reviewer, and the reported
648-test total was not independently reproduced.

## P1 — The V4 wrapper still admits only the old contact payload grammar

The actual chain is:

```text
mesh offline V16
  -> context measurements V2 -> support-cover measurements V3
  -> schedule projection V4
       -> schedule projection V1
            -> contact projection/scoring V2
```

Mesh V16 calls the new cover consumer and validator, then copies its
predictions into the workflow (`cad_guided_mesh_offline_v16.py:506–515`).
Schedule projection V4 calls V1 on the original bytes and passes through its
contact projection (`gold11_schedule_projection_v4.py:204–218`). V1 directly
calls `gold11_contact_scoring_v2.project_gold11_contact_workflow_v2`
(`gold11_schedule_projection_v1.py:250–254`).

The frozen contact module still requires:

- cover schema `CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2` (`:31`, `:66`);
- nested context schema `CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V1`
  (`:390`);
- exactly the old seven-key fit policy (`:86–87`, `:272–274`);
- quantity and carrier material status exactly `CONTEXT_ASSUMED_NOT_VERIFIED`
  (`:79–95`, checked by `:270` and `:341`).

The actual V9 producer supplies cover V3, nested context V2, an extended
policy, and enabled-layer statuses such as
`SINGLE_LAYER_IDENTITY_UNRESOLVED`. Thus a complete V9 result is rejected as
malformed even after the cover-policy incompatibility is repaired. A
measurement-phase refusal retaining the new cover object is also incompatible
with `_MEASUREMENT_CLAIMS` at contact scorer `:545`.

On rejection, no successful prefix survives; V4 reconstruction provides the
supervisor empty-prediction input. That is correct fail-closed behavior for an
unrecognized payload, but it makes a new numeric score impossible from this
chain. The versioned successor adapter must understand and bind the new
evidence without changing matching thresholds, dropping cases, remapping
coordinates, stripping side ambiguity, or mutating the old scorer.

### Why the inspected wrapper tests can miss this

`tests/test_gold11_schedule_projection_v4.py:17–18` imports the old V1
schedule/contact fixture. `_payload` updates only workflow, initializer,
registration and context metadata (`:116–139`); it does not replace the old
measurement payload with actual V3/V2 output. Those tests qualify the context
metadata wrapper, not transport of the newly emitted measurement schema.

## Material identity is propagated, but it is not eligibility

The four allowed statuses in context V2 (`:45–46`) are all unverified:

| Kernel observation | Published material status | Current geometry action |
| --- | --- | --- |
| No flange split | `SINGLE_LAYER_IDENTITY_UNRESOLVED` | Continue free-circle fit |
| Both flanges split | `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED` | Continue free-circle fit |
| Only one flange split | `MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED` | Continue free-circle fit |
| Selection disabled | `CONTEXT_ASSUMED_NOT_VERIFIED` | Continue free-circle fit |

The kernel chooses the enum at `cad_guided_measurement_v2.py:554–562`, then
returns radius, deviation and raw support at `:599–637`; it does not branch
eligibility on that enum. The singleton counterexample is already explicit in
the test source: an outer-only radius 2.3 mm observation against nominal 0.8 mm
is expected to produce radius 2.3 mm with the unresolved string
(`tests/test_cad_guided_measurement_v2.py:138–143`). That test checks wording,
not exclusion from an inner-specific output.

Context V2 stores the enum on its window and record (`:326–342`) and passes it
to `_canonical_fit` (`:367–369`). That function computes a scan-derived circle
and offsets the finite axis-support line by
`radius * unit(radial0 + radial1)`, forming the measured contact-sector
mid-angle generator (`:149–159`). It copies the status as metadata. Every
measured record with supported finite rows enters `predictions` (`:379–381`),
regardless of material identity. Canonical radius-to-CAD differences are also
computed for all these records (`:371–374`).

Support-cover V3 likewise passes the full-cover witness's status to the same
canonical fitter (`:184–186`) and publishes any finite-supported refit
(`:193–194`). Its publication validator permits any of the four enums
(`:304–309`) and requires the carrier to equal the measurement enum
(`:354–355`). It does not require a resolved inner side.

### P1 for an inner-specific claim; conditional contact diagnostics are different

None of the reviewed paths establishes that an ambiguous selected surface is
the actual inner skin. An outer-only circle cannot be relabeled as an inner
radius or an inner reference-line carrier solely because CAD proposed the
window. The nominal formula's `COMPLETE_FOR_NOMINAL_SWEEP` metadata is also
not a material-side certificate; see the independent layer audit's exact
nominal-radius counterexample. Neither context nor cover reads that status as
an eligibility gate.

The existing contact scorer, however, expressly disclaims inner-arc identity
in its header (`gold11_contact_scoring_v2.py:8–11`). It declares direct
endpoint-matching, unverified carrier compatibility and unavailable physical
counts (`:749–764`). A separately declared exploratory contact-generator
score can retain unverified geometry without pretending this closes inner
metrology. This is not historical Gold-score comparability or physical target
achievement. The corrected V9 protocol/adapter must make this distinction
explicit and retain the individual side enums, not weaken every enum to the
old generic value merely to cross a schema check.

A minimal inner-specific path should keep point estimates as auxiliary
evidence while expressing `inner_carrier_eligible=false` (or an equivalently
strict declared refusal) when identity is unresolved. Eligibility requires
separately bound side evidence, not a nominal-radius-nearest tie break. If the
immediate run remains contact-only, say so in the score and enforce that it
cannot count as the locked physical inner/count/conformance target.

## P2 — Material-status equality is not revalidated throughout the evidence chain

Context V2's publication validator checks quantity geometry/uncertainty
(`:548–553`) and carrier semantics (`:570–575`), but does not validate the
material enum or equality among window, per-context measurement, support
record, canonical measurement and finite instance. Cover V3 checks allowed
measurement/instance enums and their mutual equality, but not equality to
the original full-cover witness's status (`:293–309`, `:354–355`).

Therefore mutating a carrier and its canonical measurement together from one
allowed unverified enum to another would not be caught by that cover check.
Changing the cover output does not change the retained source-context digest.
This is not proof that the implementation currently mutates them, nor an
execution-attestation requirement. It is a missing structural provenance check
for a field that now controls scientific interpretation and may later control
eligibility. Revalidate every propagation link; reject missing, invalid and
contradictory values instead of supplying a default.

## What remains correctly unavailable

Context V2's base (`:165–174`) and validator (`:455–464`) require
`observed_bend_count=None`, `production_use_allowed=False`, conformance
`INDETERMINATE`, uncalibrated uncertainty and no joint configuration. Cover V3
inherits these and revalidates their exact values (`:225–234`). V16 retains
unknown observed count. The contact scorer separately names its count metric
`EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS` (`:36`) and forces all
eleven physical observed counts unknown (`:706–707`).

Thus this review found no current conversion of side-ambiguous contact
carriers into a physical observed-bend count or a PASS/FAIL conformance
decision on the reviewed V16 route. The public standalone kernel has a
separate sample-support count (`cad_guided_measurement_v2.py:733–740`); that
must not be substituted for the absent manufacturing count. It is not the
active V16 consumer.

## Minimum closure tests, source-proposed only

1. **Real discovery-to-cover seam:** analytically constructed broad tangent
   flanges and a finite circular arc, both layer-policy enabled and disabled.
   Use actual V2 discovery, actual planner, refit and publication validators;
   do not mock `_measure`, `_fit_policy` or the discovery return ledger. At
   least one declared conditional contact-positive fixture must pass. A
   nondefault valid layer policy must also be exactly bound. A changed or
   missing layer policy must be rejected, with complete denominators retained.
2. **Actual emitted-byte transport:** serialize that real new cover/context
   output inside the independently constructed workflow envelope. Run the
   new adapter and exact-byte reconstruction without downgrading schemas or
   side enums. Assert original endpoints unchanged. Cover both arms, a finite
   refusal, an early refusal, and a later malformed record that clears the
   whole case. The old frozen adapter should still reject the new schema.
3. **Every material enum across every boundary:** singleton inner-only and
   outer-only, mixed flange evidence, genuine paired skins and disabled mode.
   Assert equality from kernel diagnostics/window through support record,
   cover witness, canonical geometry, carrier and compact projection. Assert
   no inferred material approval and no physical observed count/conformance.
   For any inner-specific path, unresolved inputs must produce zero eligible
   inner carriers while preserving the auxiliary observation and reason.
4. **Paired skin identity is not a split count:** use the independent layer
   audit's R5/t1.5/120-degree outer-arc survivor and nominal-R4/actual-R5/
   t1.5/85-degree counterexample; include unrelated parallel surfaces and
   three layers. A nominal exclusion heuristic must not grant verified inner
   status or remove real deviations. Both sweep signs and an oblique rigid
   pose are required.
5. **Tamper matrix:** remove or substitute each side status independently,
   substitute a different allowed enum only on canonical measurement and
   carrier, alter layer policy, change the original witness status, or inject
   an asserted verified-inner/observed-count/conformance value. Every broken
   equality or unauthorized claim must fail closed before scoring.

These are bounded integration/contract tests, not a demand for a new broad
research phase or physical positive evidence before freezing. Actual budgets
must be declared by the coordinating owner before executing them. Frozen
predecessors and terminal attempts must remain untouched.

## Handoff

Actionable P1s were sent promptly to the root coordinator while this report was
being written, including the earlier planner failure that masks the later
cover comparison and contact schema failures. Next: sole editor repairs the
new-version seams and contract; an independent delta check verifies exact new
hashes and actual connected tests before a candidate freeze/PRE. No numerical
score, physical root-cause claim or promotion follows from this source audit.
