# Independent source review — residual-depth context V2 integration

Date: 2026-09-11. Reviewer: `/root/cad_blocker_closure_review`.

## Decision and scope

**No source blocker found in the reviewed correction. GREEN to finish the
invented source tests and freeze their protocol, not an exact PRE or execution
release.** The terminal connected V1 attempt remains RED at 6 PASS/1 FAIL and
0/4 curved positives. This review establishes no recovered measurement, speed
gain, new F1 score, native-byte qualification or production readiness.

The correction addresses the observed interface mismatch without changing the
original V1 context module: V2 can concentrate the already bounded global
ordered-match denominator into fewer source pairs. It does not improve or relax
the underlying geometry, support identity, normal-cone, finite-distance,
measurement or acceptance conditions.

I fully read the new context module and the changed initializer/byte-binding
seams with their surrounding consumers. I compared the complete existing-file
diffs against only their named members in the approved frozen source archive,
in memory without extraction. I independently performed AST and static-symbol
analysis; no module import, collection, test, geometry, native function or
scientific-artifact read occurred. Candidate tests and a final supervisor are
still separate work, not reviewed or qualified by this document.

## Exact reviewed source

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_two_plane_context_v2.py` | 11130 | `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663` |
| `backend/cad_guided_plane_initialization_v12.py` | 73393 | `57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87` |
| `backend/cad_guided_mesh_offline_v12.py` | 46206 | `7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e` |
| `backend/cad_guided_two_plane_context_v1.py` — unchanged | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |

All supplied hashes matched. The comparison archive is the exact previously
qualified source archive `two_plane_depth_v1_connected_frozen.vel94oib/SOURCE.tar`
under `docs/research/cad_engine_v1_synthetic_checks/`, SHA256
`f0b9102ad68a4943f2f58cf51f2dfc69105e0b8a20c672a2dadb81dc017c59a7`.
Only the two changed predecessor members and unchanged context V1 member were
read for this comparison. The old failed packet and its archive stay unchanged.
These four rows are a review snapshot, not a complete future test-import closure.

## Mathematical equivalence and admission change

The V2 orchestration function is mechanically AST-identical to V1 after exactly
four declared substitutions: function name, default budget constructor, leading
docstring and the one added global preflight call. No other statement in the
proposal loop, exact-pose deduplication, slot enumeration, accounting, success
publication or whole-refusal path changed.

The new module imports the same immutable plane/match types and the actual V1
`_inputs`, `_planes`, `_schedule`, `_family`, `_slots`, `_Work`, final-report and
JSON helpers. It does not copy or override their mathematics, mutate V1 globals,
monkeypatch a module, split the match table into batches or reset counters.
Static symbol analysis found no unresolved global in the new functions. NumPy
and `itertools` used by the copied loop are explicitly imported. The new module's
own `_base` and `_configuration` resolve locally, so the V1 exact-type validator
does not inadvertently reject the V2 budget instance.

V1 remains byte-identical, with four ordered CAD assignments per source pair.
The V2 exact budget type inherits all fields and changes only that allowance to
256. Its strict validator rejects bool/non-int values, over-maxima fields and
invalid chunk size. Other maxima, visit/cache/report caps, 64 source pairs,
1,024 signed families and 8,192 slots remain unchanged.

The preflight limit is explicitly:

`N <= min(256, max_scan_pairs * max_ordered_matches_per_pair,
          floor(max_signed_families / 4), floor(max_candidate_slots / 32))`.

Here N is the complete ordered-match table length. The exact-tuple/count check
runs after fixed work-object setup but **before origin conversion, raw-buffer
validation or support traversal**. Thus 257 matches cannot trigger a scan/support
read merely because the pair-product cap is 64*256. Lower custom sign/slot/pair
budgets tighten this preflight; no cap is inferred from a successful prefix.

The later unchanged `_schedule` still checks every typed item, canonical source
reversal, distinct/known plane identities, duplicate ordered assignments, per-pair
limits and the complete sign/slot denominator. Full raw-support/CAD-face identity
checks remain unchanged. Four sign choices and eight slots per sign still cost
32 slots per ordered assignment; the added depth does not establish independent
physical bends or additional unique poses.

Concentrated depth can change cache reuse, report growth and actual runtime.
Unchanged total logical caps are not proof of unchanged allocation/RSS/time, nor
proof that the subsequent 48-start selection will recover all four measurements.
No old noisy/cone/finite-support or measurement threshold was relaxed.

## Publication, failure and integration binding

The V2 `_base` explicitly publishes `two_plane_context_v2` and a fixed
`scheduling_capacity` declaration. Both success and every early/late caught
failure use that local base. No V2 result is labeled V1. The capacity describes
the implementation's maximum profile; a smaller direct-call requested budget
is separately represented by `budgets` on complete reports, not falsely raised
by the fixed capacity declaration. Refusals retain the bounded attempted-work
envelope and no usable family, lambda or candidate prefix.

The initializer selects the new proposer only after an explicit, validated V2
schedule. Its V1 branch still calls the original proposer with its unchanged
default signature. The V2 branch supplies the exact default V2 context budget;
there is no new public caller override of the initializer's context profile.

The new `_validate_two_plane_context_configuration` binds the exact expected
schema and every budget field with strict integer types/equality. For V2 it
also binds all capacity keys, scalar types and values. For V1 it rejects any
capacity marker. It runs after the helper's COMPLETE/status guard and before
frame, input, sign/slot, candidate and schedule-origin evidence can produce
initializer hypotheses. Those existing validations remain unchanged.

The byte adapter calls the same validator for every nonempty used pair lane,
selecting V1 versus V2 from the independent expected schedule request—not from
the returned helper's claim. It retains original array digests, complete slot
counts, selected-start origins, exact pose and numerical-support gates.
The existing V13 workflow delegates to this shared binding seam.

The unrequested-V2 marker detector now recognizes either a V2 context schema or
any `scheduling_capacity` key, in addition to the existing schedule/configuration
markers. Therefore V1 mode does not silently consume or preserve a disguised
V2 capacity publication during failure handling. Empty scheduled pair lanes
still use the prior NOT_CALLED sentinel and do not require a nonexistent complete
helper budget; this path must stay covered separately from a nonempty helper.

There is no legitimate default-V1 source regression apparent: actual complete
V1 helpers already publish the required V1 schema and exact default budget.
The stricter binding intentionally rejects older incomplete/mocked helper
publications that omit those fields; tests must use truthful complete envelopes
where their target path consumes a nonempty pair lane. Such a fixture correction
cannot itself substitute for exercising real residual-depth context generation.

The binding is a process-owned publication contract, not authentication against
arbitrary mutable Python modules. Public byte-workflow fail-closed handling must
still be tested, including a failure after an earlier context was processed.

## Required tests before a source PRE

The next independently frozen source gate must include these concrete checks;
their existence or pass count is not asserted here:

- Real immutable invented inputs with one source pair and 5, 42 and 256 distinct
  ordered targets: complete signs/slots, exact input/support identities, no
  unexpected quota refusal, and actual unchanged geometry/origin assertions.
  The 42 case represents the observed failing depth; 256 tests the fixed global
  edge rather than merely the smallest >4 success.
- Global 257-match refusal before raw/support access; smaller signed-family,
  slot, pair-count and per-pair budgets; malformed types and bool values.
- Canonical source-reversal duplicate, repeated/unknown plane IDs, malformed
  late match/support and stale support digest; all fail wholly without poses.
- Low report/visit/cache/work budgets and late injected helper refusal after
  at least one attempted family, retaining counters but clearing usable output.
- V1 source hash and <=4-input behavior remain equivalent; V1 still refuses
  five per-pair assignments. Compare V2/V1 geometry while excluding only their
  deliberate schema/capacity/budget declarations, not measurement coordinates.
- A real scheduler-V2-to-context-V2 handoff with >4 matches per source pair,
  with full initializer provenance and global sign/slot accounting. Testing
  the scheduler and context separately would repeat the prior coverage gap.
- Initializer and shared byte binder reject missing/wrong schema, altered
  budget keys/values/types, missing or stale capacity, unrequested V2 markers,
  and V1-labelled residual-depth output. Include zero-pair and disabled-pair
  controls, mismatched request publications and a late whole-workflow refusal.
- AST equivalence of the copied orchestration and an explicit import/global
  closure including the new V2 module and pytest package initializer.

Tests/harness/protocol must be read before execution, with fixed outcome count,
logical and outer budgets, exact source/archive custody and independent PRE/POST.
Passing a focused source gate would qualify that source contract only. A new
connected packet still needs its own unchanged four-case measurement oracles,
separately frozen authority and one bounded launch. The prior terminal negative
must not be rerun or relabeled.

Only this new review document is written. No engine, test, old audit, archive or
scientific artifact was modified by this reviewer.
