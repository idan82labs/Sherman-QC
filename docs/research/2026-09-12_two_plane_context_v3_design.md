# Two-plane context V3: shared exact raw extrema indexes

Date: 2026-09-12. Status: source-only implementation; not an execution release,
integration qualification, physical improvement or production promotion.
Only the new V3 source and this design note are owned by this change. V1, V2,
the standalone extrema helper, active/frozen protocols and all data remain
unchanged. No target module was imported, collected or executed for this work.

## Contract and changed seam

`propose_two_plane_context_v3` has the same arguments and typed plane/match
records as V2, with `TwoPlaneContextBudgetsV3`. It retains V2's complete ordered
schedule: at most 256 total assignments, four signed families and eight explicit
slots per family, at most 1024 families / 8192 slots. No pair, sign, support,
failed family or ambiguous candidate is selectively removed. The inherited
global denominator preflight still runs before any input buffer is read.

The change is narrowly scan-side: both the whole raw scan and each complete
assigned scan support obtain exact extrema of a separately declared rounded
program from the frozen hierarchy. CAD projections still call V1 `_projection`
directly, including its triangle/vertex witnesses and chunked NumPy matmul.
The normal cones, central rotation, translation family, backward-residual and
rank gates, eight slot equations, pose duplicate rule, and downstream finite
admission requirements are unchanged. This module produces seeds only:
`accepted_registration=False`, unknown count and no production authorization
remain explicit. It does not itself integrate an initializer or workflow.

The V3 `_slots` body is AST-identical to V1. The entire numerical `with` block
of the public orchestrator is AST-identical to V2, including `_inputs`, `_planes`,
`_schedule`, every sign/slot loop and the final complete-ledger assertions.
The local `_slots` name is necessary to dispatch scan projections without
mutating a frozen module's global function. Numeric helpers are reused directly.
These are source comparisons, not claims of identical returned poses: changed
scan arithmetic can alter extrema, slots, exact duplicate counts and all later
nonmonotonic candidate/admission decisions.

## Shared owner and private adapter

V1 `_inputs` already captures each input using `tobytes` and reconstructs an
immutable array over the admitted bytes, with captured shape/dtype/size,
finite/range validation and a fresh SHA-256. V3 reuses that exact scan array
object. A bounded base-chain check requires immutable `bytes` backing, not just
a mutable ndarray with its writeable flag disabled. No index recopies,
revalidates every coordinate of, or rehashes the full raw point snapshot.

`_ContextIndex` is a private subclass of the frozen `_Index`. It overrides only
its constructor, `_charge` and `_memory`; it inherits `_build`, `_digest`,
`_bounds`, `_evaluate` and `query` unchanged. Its support-only constructor:

- receives the private context owner, never an arbitrary caller array;
- captures/copies assigned row bytes, or constructs all raw IDs for whole scope;
- checks sorted unique in-range int64 IDs and the fresh support digest;
- calls the unchanged build on every member of that support.

The frozen owning `_Index.__init__` is never called. Its standalone caps are not
silently reset per index: all inherited work charges go to a shared helper
ledger and the authoritative parent context ledger. None of the inherited
methods used by V3 reads `self.cap` except through the two overridden admission
methods. This private cross-module contract must stay source-bound in future
qualification; the adapter is not a new public general-purpose index API.

A context-local registry lazily builds one index per exact scoped support.
The key includes the freshly owned raw-buffer hash, scope, support hash and
row count. A new immutable assigned-support alias with a matching key must
also pass exact array equality against the original owned row sequence.
The alias retains that immutable array, so object-ID reuse cannot establish
identity. Every scan projection resolves the registry before a projection-cache
hit; digest-only cache lookup cannot bypass the first exact alias comparison.
Repeated calls on the same verified immutable alias need no repeated full-row
comparison. A hash collision with unequal assigned rows refuses the context.

`WHOLE` and `ROWS` are intentionally distinct scope keys. An assigned support
containing all raw IDs may therefore build a second index; it is not silently
coalesced with whole scope. Identical assigned supports do share after the exact
check. The membership counter counts these distinct scoped index populations,
not all old input-support aliases. All input plane records and raw support
bindings are still captured and validated by unchanged V1 `_planes` first.

There is no eviction, refund, approximate direction key or selected-support
trimming. The public `finally` disposes the private registry on success or
failure: it invalidates indexes, clears index/alias ownership and drops raw,
work and result references. Large arrays need not wait for cyclic garbage
collection. Returned primitive reports and attempted/peak statistics survive;
the owned-byte ledger records in-call admitted ownership, not post-return RSS.

## Rounded arithmetic and witnesses

For each scan row, the frozen program separately evaluates float64 operations:

`delta = p - origin; terms = delta * direction;`
`value = ((terms[0] + terms[1]) + terms[2]) + constant`, then positive-zero
normalization. NumPy ufunc results are separated; no BLAS matmul, FMA or
reassociation is substituted. Direction and constant come from the unchanged
two-plane family, not from a CAD radius or a known physical answer. Origins
and geometry remain explicitly declared millimetres and frames.

For fixed finite origin/direction/constant, evaluate the same staged program at
the AABB's sign-selected lower and upper coordinate corners. Under consistent
binary64 round-to-nearest, gradual underflow and no reassociation, subtraction,
signed multiplication and addition have the needed monotonicity. These two
rounded outputs enclose every rounded row output in the node. They are not
outward enclosures of the exact-real dot product and provide no physical error
interval. The inherited arithmetic sentinels run once per registry after its
scratch-memory admission; they are necessary checks, not complete platform
attestation. All input coordinates and query vector/origin/constant magnitudes
must satisfy the inherited finite `<= 1e6` domain. The extra query-domain guard
is prospective V3 behavior, not proof that every legacy input remains admitted.

Hierarchy bounds only skip rows that cannot improve either extremum or its
tie witness. A tie is the smallest original raw-row ID, independently for min
and max. Each node retains its smallest raw ID; equality pruning is legal only
when that ID cannot improve the current witness. Stable internal sorting never
renumbers raw identities. Both extrema and raw witnesses are actual evaluated
rows; bounds are not emitted as fictitious witnesses. Correctness does not
depend on tree quality. Cancellation, signed zeros, subnormals and direction
changes do not justify relaxed keys or tolerance-based ties.

Scan leaf capacity is fixed at 256, explicitly versioned in the report.
`chunk_rows` now controls only input validation and CAD projection chunks, not
the scan query. The scan result is not claimed bitwise equal to legacy chunked
matmul; exact equivalence is to the declared staged program over the entire
eligible raw row set.

## Cumulative guards and payload accounting

All V2 default ceilings are retained, including 64M scan-row visits, 64M index
visits, 256M abstract sort work, 1 GiB hashed bytes, 256 MiB accounted working
payload, 32 MiB complete report and 6144 joint scan/CAD cache entries. Config is
an exact V3 dataclass snapshot; bools, subclasses, negatives and literal-cap
increases refuse. No mutable default supplies the maxima.

Five new, prospective engineering guards are added:

| Field | Default maximum | Meaning |
| --- | ---: | --- |
| `max_extrema_indexes` | 129 | Attempted whole plus distinct assigned indexes |
| `max_extrema_indexed_rows` | 4,000,000 | Sum of attempted scoped support populations, including whole |
| `max_extrema_node_visits` | 16,000,000 | Build-parent and bound-query node charges |
| `max_extrema_coordinate_visits` | 256,000,000 | Declared build/query coordinate-work proxy |
| `max_extrema_queries` | 3072 | Actual uncached scan index queries |

The query configuration must additionally satisfy
`max_extrema_queries <= 3 * max_signed_families`; lowering the latter requires
an explicitly compatible query cap, not an implicit clamp. Three scan supports
per available signed family give the 3072 default opportunity ceiling. At most
128 input scan planes plus whole give 129 possible scoped indexes. The 4M
membership ceiling is a new bounded allocation policy (four times the default
1M raw-row ceiling), **not** an invariant implied by old support caps and not a
claim that arbitrary 128 overlapping supports fit. Excess overlap refuses
whole-context. CAD work and old full-support binding work consume their old
caps independently and can still refuse before or after index construction.

Let `I` be built index count, `S_i` its support size, `M = sum(S_i)`, and `E`
the total query row evaluations including seed repeats. Beyond unchanged input
and binding work, construction charges `2M` scan rows, `5M` index visits,
`19M` coordinate proxy units, `8M` hashed support bytes and
`sum(S_i * max(1, ceil(log2(S_i))))` abstract stable-sort units. New distinct
equal-support aliases add their full row counts to the parent index ledger;
these comparisons are separately reported, not hidden in helper counters.
Queries add `E` scan and index visits, `9E` coordinate units and actual node
charges. Root raw-coordinate validation is not repeated or counted again as
new index-coordinate work. This is an explicit work model, not all scalar
instructions or an elapsed-time guarantee.

At the 4M/1M ceilings the extra sort charge is at most 80M; it competes with
unchanged input/CAD/support sort work under the existing 256M total. Seed rows
and evaluated leaf rows both consume real visit budget. No counter is reset at
a later support. Charges precede operations; an attempted overflow can exceed a
cap in refusal statistics without claiming the operation completed.

For an index of `S` rows, `K = ceil(S/256)`,
`V = 2 * 2**ceil(log2(K))` node slots are allocated, including padding and slot
zero. The implementation is **not** a `2K-1` compact tree. Each slot costs 72
numeric bytes, so retained index payload is `8S + 72V`. Since
`V < S/64 + 4`, all 129 indexes under 4M memberships use fewer than 63,016
slots: at most the conservative 4,537,152 node bytes plus 32,000,000 row bytes.
This bound excludes already owned context raw/support arrays, CAD inputs and
all report/cache payload, which remain charged separately.

One extra 64 KiB index scratch reserve is admitted when the registry starts,
plus 1024 bytes per attempted index descriptor/private bookkeeping and 256
bytes per owned support alias. These are declared conservative payload
allowances, not CPython allocator/RSS measurements. Assigned support capture
precharges `8S`; the helper precharges `64S` sort/gather/replacement scratch,
then `72V + 8S` at node allocation and `8S + 64*leaf_rows` while old and sorted
support buffers overlap during leaf construction. Query scratch is precharged
as `128*evaluated_rows` (at most 256 leaf rows or seven seed rows per call).

The crucial integration invariant is the override of parent `_Work.memory`:
**every** memory check includes all registry owned bytes, metadata and scratch,
including later inherited CAD projections, input/support copies, cache records,
pose records and final publication. It is insufficient to check only inside an
individual index. Parent temporary/owned/cache/proposal counters are added once;
registry payload is added once. The helper's nested peak describes only the
index registry; the parent `accounted_working_peak_bytes` is the combined
admission ledger. It is not process RSS, caller memory or a universal bound on
allocator/runtime overhead. Normal process qualification must add those limits.

## Reporting and whole-context failure

Successful output adds `scan_extrema_indexes`, the arithmetic/chunk/memory
scope declarations, and cumulative `statistics.extrema_work` plus the five
flat guard counters, completed-index and exact-alias counters. Each scan
projection records its index ID, complete support population, original min/max
raw witnesses, declared arithmetic and query work. Its `row_visits` is actual
query evaluation attempts including seeds; it no longer mislabels the support
population as evaluated work. CAD projection records are unchanged.

Index descriptors use the parent record-byte ledger; scan projection/cache
records use the original parent cache-payload accounting. The final complete
report retains the inherited streaming size check. The nested helper report
counter remains zero because the parent, not the standalone batch wrapper,
owns publication. Index payload accounting is retained on late build/query
refusal even when no completed descriptor exists. Completed counters advance
only after successful operations; attempt and completion counts are distinct.

Any input, schedule, support, arithmetic, alias, build, query, cache, memory or
report error rebuilds the base refusal: no input binding, index descriptor,
projection, family, candidate or usable flag survives. Only bounded attempt
statistics and discarded family/candidate counts remain. A registry/query
failure also poisons the private object; no later subset can continue. V3's
refusal envelope is explicitly 8192 bytes, increased from V1/V2's 4096 solely
for the new fixed schema and nested counters. The complete-report cap and
scientific gates are not increased. Process-level interruption/OOM is not
converted into fabricated completion by this Python failure boundary.

## Source checks and remaining gates

Source-only checks performed: complete source reads, stdlib AST parsing,
AST equality of the slot body and full numerical orchestrator block described
above, inherited-method attribute inspection, and exact byte/hash inspection.
There was no target import, collection, test, geometry, scientific artifact or
physical-data access. The only local imports introduced are V1, V2 and the
frozen extrema helper; their external imports are NumPy and stdlib. The known
`backend/__init__.py` path is absent. Future exact PRE must bind its actual
source/test/harness closure and recheck implicit package boundaries.

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `backend/cad_guided_two_plane_context_v1.py` | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_context_v2.py` | 11130 | `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663` |
| `backend/cad_guided_raw_extrema_index_v1.py` | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |
| `backend/cad_guided_two_plane_context_v3.py` | 25845 | `9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d` |

Independent prospective tests/review must bind full scan and assigned-support
brute staged oracles, smallest raw-ID ties, duplicate/overlapping exact support
reuse, shared raw ownership, cross-index cumulative cap/memory pressure,
malformed and over-budget late failure with whole-prefix clearing, retained
V2 depth/sign/slot denominators, unchanged CAD projection, and final report
failure. Large analytic axis cases may establish one bounded work reduction,
not general pruning quality, registration speedup or a physical score. Future
initializer/workflow integration must explicitly select and bind V3/schema,
budget and arithmetic publications; no old V2 result may be relabeled.

Implementation is held for those independent gates. This document grants no
run authority and changes no terminal experiment or physical candidate.
