# Source-only delta review: surface V4 uncertainty repair

Checkpoint: 2026-09-12 22:42 UTC. Independent reviewer: Codex
`fable_surface_v4_review`. Fable worker `1519fdc2` remains the implementation
owner. Only this new report was edited by the reviewer.

## Disposition

**P2 CLOSED AT SOURCE-REVIEW LEVEL.** The necessary counter/flag guard and the
one actual mixed-row surface-to-consumer/stage test requested in the preliminary
review are correctly implemented in the inspected source. No new concrete
blocker was found in this bounded delta. The associated stale width-policy
comments are repaired. Fixture-specific byte comparison and timing statements
are appropriately limited.

This is not independently verified test success, a full import/wiring/runner
audit, a frozen PRE, execution authorization, physical evidence or promotion.
The coordinator relayed an author report of 283 total checks; none was rerun,
collected or independently audited here. Closing this P2 source finding does
not establish those passes or a clean F1 result. Separate wiring/runner reviewers
own the broader integration review; this report does not duplicate their scope.

## Exact inspected bindings

The five source/test hashes matched at the beginning and end of this review.
Total: 141,573 bytes. These observations bind the examined delta, not a live
repository freeze or attestation against an intervening reverted edit.

| Repository-relative file | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_triangle_surface_v4.py` | 37248 | `f79b73b37be2684f73f7d467d5aa1074f8a27d968cdf52e9fb1d652dae02422c` |
| `backend/cad_guided_global_registration_v4.py` | 37673 | `5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273` |
| `backend/cad_guided_stage_policies_v2.py` | 3328 | `699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d` |
| `tests/test_cad_guided_triangle_surface_v4.py` | 26622 | `cc80b5c7f3f676176804d1d5c15855469ca2c17485672acc0f6549fd46c24db8` |
| `tests/test_cad_guided_global_registration_v4.py` | 36702 | `1cf31b12f373f1c0b8d72eecf5c6bd5ae86f704feeaaf0d14e5dda8031735204` |

The original preliminary report remains unchanged:
`docs/research/2026-09-12_fable51_surface_v4_codex_preliminary_review.md`,
13,834 bytes, SHA-256
`fb3a3597727230b9dcb43445db2d3b4768f675d2e4e66966510b4e80dbd82e53`.
Its earlier source bindings and findings remain historical evidence; this
document supplies a new disposition for this specific delta, not a rewrite.

The author response was fully read:
`docs/research/2026-09-12_fable51_review_response_v1.md`, 3,980 bytes,
SHA-256 `99458961d11b507a11a1a33e9640d42b7ca0d8f5669be01c981d664827ad1663`.
Only its surface/registration items were assessed. Its extrema/context numbers
and other claims were not validated here.

Method: compared the new source against the preliminary report and retained
earlier source observations, using the same narrow V3 source comparisons to
identify the V4 changes; inspected the changed validator, comments, test bodies
and unchanged stage classifier. No archived source bundle was accessed. The
stage-policy file has exactly the original reviewed hash, so it is an unchanged
consumer seam, not a newly repaired engine file.

## P2 closure evidence

### Necessary counter/flag consistency

`backend/cad_guided_global_registration_v4.py:205-213` computes the intersection
of `normal_ambiguity_uncertain` and `closest_point_ambiguity_uncertain` and
requires a positive wide-pair counter to have a nonempty intersection. It also
requires that counter not exceed the sum of `near_tie_counts` on those jointly
uncertain rows. The earlier boolean, row-shape, counter-type and tie-count
validation runs first (169-178 and 201-204).

This is the requested necessary check: each actual counted wide competitor sets
both flags and occupies a tie slot, so its count cannot exceed all available
slots on jointly uncertain rows. With at most 200,000 rows and 256 ties per row,
the sum is at most 51,200,000, safely inside the validated int64 summation range.
An empty query with a positive counter fails the joint-row requirement. A zero
counter may legitimately coexist with uncertainty caused by other comparisons,
so the guard correctly avoids asserting the converse.

The guard does not identify exactly which competitor was wide; the code says
so explicitly. It does not recertify geometry or substitute aggregate telemetry
for the producer's per-row flags. It changes validation of inconsistent complete
records, not the admitted geometry or pruning arithmetic. The producer still
sets both flags for every wide competitive pair before storing and comparing it.

`tests/test_cad_guided_global_registration_v4.py:624-638` now makes the positive
fixture carry both flags and two tie slots on the affected row, verifies that
row is not unambiguous, and expects refusal when the wide count exceeds those
slots. Lines 641-651 separately require refusal for no flags, normal-only and
point-only flags. This removes the original test's explicit acceptance of the
all-false-uncertainty contradiction.

The mocked positive record is a necessary-field consistency seam, not a complete
reconstruction of all triangle-pair telemetry. The isolated 256-count test is now
explicitly labelled a field-maximum test rather than an execution with 256 real
competitors (654-662). Neither is presented here as runtime geometry evidence.

### One actual mixed-row path

The new test at `tests/test_cad_guided_global_registration_v4.py:665-698` builds
the actual two-triangle needle/sound index, constructs `_Queries` with the V4
budgets, and sends three invented point rows through the real query adapter.
It imports the real `cad_guided_stage_policies_v2` classifier in the same isolated
package namespace; there is no monkeypatched query, fabricated uncertainty output
or mocked stage mask in this test.

Its asserted classifications are appropriate to the requested promise:

- The affected needle row has both uncertainty flags and retains in-gate
  distance-only proximity, but no proposal target, legacy unambiguous support
  or tangent constraint.
- A sound in-gate interior row retains proximity, proposal support and tangent
  support.
- A sound out-of-gate row is definite-outside and contributes no support.
- `_chosen` retains only the sound in-gate row, index 1 (698).

These expectations are independently understandable from the invented geometry:
the second row lies inside the z=0 sound triangle at distance 0.05, and the third
lies inside it at distance 3, on opposite sides of the declared 0.5 gate. The
first row reproduces the previously reviewed wide-competitor needle location.
The actual occurrence of its wide interval still requires execution evidence;
the test explicitly asserts that counter and the joint-flag pattern rather than
assuming them silently.

The unchanged stage classifier at
`backend/cad_guided_stage_policies_v2.py:61-75` first validates through
`_support_masks`, then intersects proximity with `_unambiguous` for proposal
targets and with nonuncertain face-interior status for tangent constraints.
Thus the new test reaches precisely the seam implicated by the original P2;
it is not merely another fabricated consumer-contract record.

## Documentation and comparison delta

`backend/cad_guided_triangle_surface_v4.py:406-414` now accurately describes
pass 1 as minimizing the certificates actually evaluated and explains why it
need not minimize every triangle's certificate upper endpoint. It also states
the V4 wide-competitor row-uncertainty policy. The corresponding deferred-admission
comment at 562-564 no longer incorrectly calls interval width fatal. The inspected
surface delta does not change the earlier pruning or certificate implementation.

`tests/test_cad_guided_triangle_surface_v4.py:460-492` labels the soup comparison
as one fixture, checks dtype/shape and C-order bytes for every result array, and
retains separate source-face tuple equality. This is stronger than numeric array
equality for represented-byte claims, including signed zero. Both timing windows
now encompass index construction and the query; printed process-CPU values are
explicitly consumed observations, not a frozen efficacy gate. No timing value,
performance improvement or general bitwise-V3 equality is independently verified
by this source review.

## Boundary and next action

The requested P2 source/control repair is closed, with no additional repair or
broad rerun requested by this reviewer. Root may accept this bounded disposition
after reading the report and checking the bindings. Any later source changes
require the appropriate delta review, and a physical successor still needs its
separate exact-source protocol/PRE and resource gate.

No target import, collection, test execution, raw/Gold/new-data access, artifact
or cache inspection, credential access, engine/test edit, worker interference or
broader integration review occurred. This report establishes source-review
closure only.
