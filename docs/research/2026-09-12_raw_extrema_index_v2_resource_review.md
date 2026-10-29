# Raw-extrema index V2: independent resource and failure review

Date: 2026-09-12. Reviewer: `process_v2_qualification`, independent of the
candidate/test authors. SOURCE_ONLY; no target import, compile, collection or
execution, no physical/raw/Gold/attempt/cache access, and no existing-file edits.

## Decision and exact source

NO_CONCRETE_RESOURCE_BLOCKER_FOR_A_SEPARATELY_FROZEN_INVENTED_QUALIFICATION.
The new numeric payload and bound work are admitted before their operations,
using V1's unchanged cumulative budgets. No efficacy or runtime pass is claimed;
this is not an execution PRE or integration release.

Full source reads and hashes:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| backend/cad_guided_raw_extrema_index_v2.py | 16616 | `862b6926d1eeb8a2cd05656b5a0a69e501cfcdfda69f27a1036ef6c361e74057` |
| tests/test_cad_guided_raw_extrema_index_v2.py | 23252 | `799f3295d3a2c560557045d1e83e944cb6167b2d869139ee47a5552907b0007b` |
| backend/cad_guided_raw_extrema_index_v1.py | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |
| backend/cad_guided_sheared_bounds_v1.py | 11420 | `36b87ea58c920774602d7f7cf53e817d5d130bf2b98bcc02dd8b8cce947201b6` |

Line references below are to this V2 candidate unless stated otherwise.

## Work accounting

Lines61–68 increment the selected phase and `sheared_bound_steps`, then call
inherited `_charge("coordinate_visits", amount)`. An over-cap attempt remains
in all three counters and raises before the charged operation; it is not
refunded or converted into a cheaper completed result. Constructor and query
share the same ledger/caps. No override of V1 `_build`, `_charge`, `_memory`,
`_evaluate`, fixed maxima or staged arithmetic exists.

Let N be raw rows, S admitted support rows, L=ceil(S/256), B the next power of
two at least L, K=2B node slots, M the completed two-child merges, C single-child
copies, r seed rows, T attempted seed triples, and I=1 if a best cross reaches
the ratio block. In a successful enabled construction M=L−1. Exact new charges:

| Phase | Additional coordinate-budget charge |
| --- | --- |
| Chooser, lines72–120 | 18 + 4r + 32T + 32I; at most1198 since r≤7,T≤35 |
| Leaf primitive, lines160–161 | 61S + 78L |
| Merge primitive, lines173–174 | 75M |
| Packing during construction, lines130–188 | 8K + 3S + 8L + 8(B−1) + 36M + 4C |
| Each successful enabled node query, lines197–203 | 6 vector packing +16 box packing +390 primitive =412 |

Thus added enabled construction is chooser +8K+64S+86L+8(B−1)+111M+4C.
This is in addition to unchanged V1 coordinate construction6N+19S, not a
replacement. On whole-buffer S=N the old construction is25N. Hashing the new
packed payload separately charges32K hash bytes; seed and leaf gathering add
r+S to both index and row visits; the new internal-node pass adds B−1 node
visits. Completed counters increment only after successful primitive/storage
operations and are distinct from attempted counters.

The chooser's32-per-triple reserve covers6 differences,6 products,3 cross
differences,3 finite inputs plus all-reduction inputs,3 abs and3 max inputs and the
strength comparison, with space for the three scalar-to-array values. The
ratio block's32 reserve covers abs/argmax, two negations/divisions and bounded
finite/magnitude admission. Packing charges precede each gather, represented
conversion, zero-fill, four-value store and immutable-copy operation. Helper
callbacks charge every helper step before execution; that primitive's previously
reviewed local65536 limit is not substituted for the parent's aggregate cap.

Zero shear retains the chooser charge and4096-byte metadata allowance, but
skips all packed-array/leaf/merge/primitive-query work. Lines192–196 deliberately
use unchanged V1 `_bounds`; its old uncharged numeric corner work is not silently
represented as zero in the external common-work comparison.

For each32768-row matrix population, L=B=128,K=256,M=127,C=0. Added enabled
construction is at most2,126,519 declared steps before any query saving. This
is substantial overhead, not free preprocessing. No source-only claim is made
that the12 queries amortize it, especially on folded or repeated-extrema data.

The model is explicitly attempted abstract work, not every Python operation.
Scalar loop/count/address/statistics bookkeeping, callback overhead, dictionary
creation and JSON serialization are not counted as coordinate arithmetic.
Legacy traversal comparisons/reductions are not retroactively made into a
hardware-instruction ledger. Such overhead must remain visible in the separate
instrumented CPU result; no claim of universal "all operations" fit follows.

## Common V1 world-bound supplement

The test's WORLD_BOUND_STEPS=37 matches the declared successful V1 `_bounds`
body:6 sign comparisons +6 coordinate selections +2×(9 staged arithmetic +1
zero comparison +1 zero selection) +2 finite checks +1 ordering check =37.
It is a fixed source ledger, not measured hardware/allocator work. The old
18-arithmetic-only number is retained solely as a labelled partial diagnostic.

Instrumentation counts every actual V1 world-bound call, and V2 world-bound
calls only when shear is disabled. Enabled V2's390 helper steps already include
its own original-world/error computation, so adding37 there would double count.
The V1 monkeypatch context ends before V2 runs: the inherited V1 corner function
is not accidentally observed twice on V2's disabled branch.

`common_steps=coordinate_visits+37*old_world_bound_calls` is cumulative from
construction, including at query1/3/12 checkpoints, not leaf-only work. Each
population emits both arms' dispositions/counters before requiring COMPLETE;
a refusal cannot win by comparing only a short prefix. The common-step ledger
still excludes separately itemized hash/sort/index work and Python overhead.
Build+query CPU includes the observed constructors/queries and instrumentation,
but excludes oracle/report encoding. It is one serial instrumented comparison,
not a statistical or universal speed guarantee. No threshold is asserted in
these tests; root's prospective independent efficacy decision remains separate.

## Memory admission and immutable ownership

V1 owns byte-backed snapshots of the whole point buffer and original support
identities, then a byte-backed sorted row permutation. V2 calls that constructor
first; extra metadata derives only from those owned rows/seeds. Mutation of the
caller's point/support arrays after construction cannot change these snapshots.
The exact same V1 tree, seeds, support/hash definitions and evaluator remain.

Lines48–49 reserve4096 metadata bytes before chooser construction; the small
counter initialization immediately preceding it fits inherited bookkeeping.
The at-most-seven-row chooser
reserves32r for gathered24r coordinates and8r indices. Tiny cross vectors and
Python metadata fit the declared fixed scratch model; this is not an allocator
or foreign-object memory proof.

The packed table is32K bytes. Lines145–148 reserve it before allocating and then
add it to owned bytes. Each leaf reserves131072+24*rows_in_this_leaf temporary bytes
before advanced-index gathering and the helper's local snapshot/work; at most
137216 bytes beyond live owned/report payload. Internal merge and enabled
query calls reserve131072 before box construction/helper calls. For freezing,
lines184–187 reserve a second32K while the old mutable table is still live;
the final array is from an immutable bytes buffer, not merely a writable owner
with its write flag lowered. Hashing uses a memoryview and the hash-byte budget.

With S≤1M, B≤4096,K≤8192: added persistent payload is at most266240 bytes
(4096+262144). New-phase temporary peak above retained payload is no more than
max(137216,32K), plus V1's fixed65536 allowance. The inherited constructor's
sort/raw/support peak is still accounted separately before these additions.
The retained report-byte estimate is included by `_memory`; it is not an exact
Python object-graph measurement. Existing process-level sampled RSS controls
remain necessary in a future qualified invocation.

Only raw/row snapshots and new packed shear storage are immutable backing
buffers. Inherited world-tree arrays, owner attributes, counters and chooser
dictionary are still mutable internal state. The class expressly does not
authenticate an externally altered index/ShearedBox; do not describe the entire
object as tamper-proof or safely shared with arbitrary mutators.

## Failure boundary and fixture adequacy

Constructor/query catch BaseException to set valid=false, then rethrow. The
batch retains the partly initialized object via `__new__` before `__init__`,
so failed constructor charges survive. Its ordinary exception boundary builds
a fresh UNAVAILABLE response with projections=[], attempted statistics and
discarded prefix count, without raw/support bindings or COMPLETE metadata
(lines271–307). Internal budget/helper reasons survive by exact exception type;
unexpected exceptions use EXTREMA_V2_EXCEPTION. SystemExit/KeyboardInterrupt
are not converted into successful records. Report accumulation/envelope/final
encoding errors also invalidate and discard the whole successful prefix.

This is logical invalidation/output clearing, not explicit zeroization or
immediate reclamation of externally referenced owner arrays. Direct private
`_node_bounds` calls bypass the public query failure wrapper; future callers
must use the owned query/batch boundary and aggregate their own budgets.
The inherited refusal envelope is not universally guaranteed to fit an
arbitrarily lowered report cap, e.g. zero. Success is encoded-cap checked;
an outer publisher still needs its independent bounded failure envelope.

Static AST expansion finds10 test functions/16 prospective items:4 full fixed
populations,4 small enclosure/oracle controls and8 individual identity/order/
charge/refusal/snapshot/zero controls. No collection was performed. Tests use
an inert backend namespace and only the three named modules; no initializer,
native reader, geometry process or physical fixture is invoked by source review.
Specific controls retain constructor refusal charges, inject first leaf failure
after charging, discard a later query prefix and prohibit reuse, check raw
mutation isolation, packed read-only storage and exact aggregate charge formulas.
They do not exhaust every memory-cap edge, late merge failure or allocator path;
those are covered here by source ordering, not claimed as executed tests.

No repair or broad additional suite is required by this review. Freeze the exact
focused source/runtime/budget command before any invocation; retain all four
efficacy populations and negative outcomes. Integration remains conditional on
the separately declared efficacy result, not this source review.
