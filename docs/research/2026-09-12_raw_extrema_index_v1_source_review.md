# Raw extrema index V1: independent source review

Date: 2026-09-12. Disposition: no remaining blocking finding in the reviewed
standalone source under its declared arithmetic and payload-accounting premises.
This is a source review, NOT exact PRE approval, an executed test result,
integration qualification, a speedup measurement or physical run authority.

## Reviewed scope and exact revision

Fully read canonical `AGENTS.md`, the current-status top, the relevant complete
`backend/cad_guided_two_plane_context_v1.py` source, and the complete new helper.
The final helper was read completely and rehashed after the corrections:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_raw_extrema_index_v1.py` | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |

The preceding 18473-byte revision, SHA256
`14e9738bb3d8a7176b4995111a5d48343f4f540f947d5f195b3d7ae2b9544fc7`,
was also fully read. The final disposition applies to the corrected revision
above, not an assumption that the original revision already contained the fixes.
The helper imports only dataclasses, hashlib, json, math and NumPy. Its imports
and definitions were inspected as source; no project or helper import occurred.

No test source, harness or prospective denominator is qualified by this review.
No import, collection, test, simulation, geometry execution, native call,
physical/Gold input, active artifact or private cache was accessed. Only this new
review document was written. The terminal V5 attempt remains terminal; this
candidate does not establish that a projection cache-key defect existed.

## Findings and verified closures

1. **Captured metadata and row caps — closed.** The earlier code checked live
   input shape/count, then recaptured those values afterward. A concurrent
   metadata change between these reads could admit a larger owned snapshot than
   the checked raw/support row cap. The final source captures shape, dtype and
   byte count, validates their compatibility and limits, checks captured byte
   length, and revalidates the immutable owned reconstruction. This prevents the
   identified cap-bypass seam. It does not claim atomic capture of a caller that
   writes individual data values concurrently: the result binds the actual owned
   bytes, not an authenticated external acquisition transaction.
2. **Initial scratch precharge — closed.** `_memory()` now runs immediately
   after the typed cap snapshot, before arithmetic sentinels allocate arrays.
   An insufficient fixed working allowance is refused before those operations.
3. **Attempted versus completed work — closed.** `query_rows_evaluated` now
   increments after projection, finiteness and extrema/tie extraction succeed;
   `leaves_evaluated` increments after the leaf evaluation succeeds. Attempted
   row/index/coordinate charges still precede operations and survive refusal.
4. **Coordinate-counter scope — closed.** The final successful report explicitly
   describes its counters as attempted abstract work and publishes the formula
   `6N_RAW_VALIDATION_12S_ROOT_EXTREMA_S_SORT_6S_LEAF_BOXES_9E_QUERY_ARITHMETIC`.
   This is not an exhaustive count of scalar comparisons, zero canonicalization,
   finiteness passes, extrema reductions, tie masks, allocator work or hardware
   instructions. Completed-counter semantics are separately stated.

No additional large-array payload undercharge or incorrect pruning/tie rule was
found. The integration requirements below remain open by design; they are not
claims that the standalone helper is already an integrated registration repair.

## Conditional rounded-arithmetic proof

For admitted binary64 point `p`, origin `o`, direction `d` and constant `k`, the
helper separately evaluates:

    s_j = RN(p_j - o_j)
    q_j = RN(s_j * d_j)
    f(p) = RN(RN(RN(q_0 + q_1) + q_2) + k)

Each subtraction, multiplication and addition is a separate NumPy ufunc call.
The final numeric zero is canonicalized to positive zero. This is explicitly
NOT the legacy `(points-origin) @ direction + constant` arithmetic. The source
does not claim identical legacy BLAS accumulation, chunk behavior or candidate
trajectories.

For node box coordinate bounds `[l_j,u_j]`, select the lower corner coordinate
as `l_j` when `d_j >= 0`, otherwise `u_j`; select the opposite upper coordinate.
Under the declared consistent binary64 rounding behavior, subtraction is
monotone, multiplication has the fixed direction's sign, and each ordered
addition is monotone in its operands. Therefore evaluating the same staged
program at those two corners encloses every rounded row output in the node.
The box need not contain a raw row at either corner. Correlation between
coordinates can loosen the bound, but cannot invalidate it.

No outward `nextafter` is required for this rounded-program enclosure. It would
be conservative but would needlessly obstruct equality pruning. The resulting
bounds do NOT enclose the exact-real linear functional or physical measurement
error. Cancellation does not invalidate monotonicity of the staged program.
Signed zero does not change numeric ordering; final zero canonicalization makes
its output representation explicit.

The source declares round-to-nearest, gradual underflow, consistent behavior
between node and row evaluations, and no reassociation/fusion. The sentinel
checks are correctly labeled necessary, not complete platform attestation.
They do not prove all rounding modes or every vector-kernel behavior. A frozen
runtime and independent arithmetic controls remain required; `np.errstate`
controls exception reporting, not the floating-point environment itself.

Unlike a generic unrestricted interval API, this helper admits raw coordinates,
origins, directions and constant magnitudes only up to 1e6. Subtractions are
bounded by 2e6, products by 2e12, and the completed ordered expression by
6e12+1e6. Thus admitted inputs are far from overflow, including at artificial
box corners. Refusing a nonfinite node bound is consistent with this restricted
domain. If a successor enlarges that domain, a corner overflow alone cannot be
treated as proof of a nonfinite raw row; it would require descent or a separately
declared domain refusal, never unsafe pruning.

## Raw witnesses, ties and padded traversal

The build starts from strictly increasing original raw IDs. Its six coordinate
extremum seeds therefore choose the smallest raw ID on each coordinate tie;
the smallest support raw ID is also seeded. Seed IDs are deduplicated, yielding
at most seven actually evaluated raw rows. A box corner is never returned as a
witness.

Every node stores its smallest original raw ID `m`. With current minimum
`(vmin,imin)` and maximum `(vmax,imax)`, the code prunes only if BOTH hold:

    L > vmin or (L == vmin and m >= imin)
    U < vmax or (U == vmax and m >= imax)

This preserves an unvisited smaller-ID tie and avoids the incorrect OR of the
two extremum predicates. Leaf extrema select the smallest actual raw ID among
equal rounded values, rather than the first internal sorted position. Bound
priority changes work order only; final values and raw-ID ties do not depend on
that traversal order.

Let `S` be support rows, `K = ceil(S/256)`, `B = 2**ceil(log2(K))` with `B=1`
for `K=1`, and `V = 2*B`. The actual code allocates **V node slots**, including
unused slot zero and padded leaves. The earlier abstract `2K-1` full-tree bound
does not describe this implementation's storage. For S=1,000,000, K=3907,
B=4096 and V=8192.

Actual leaves occupy a prefix. Empty padded leaves have empty start/end ranges;
their infinite sentinel boxes are merged as neutral values and are never sent
to `_bounds`. A left-empty/right-nonempty node cannot arise from this prefix
construction. Nonempty parent ranges, boxes and minimum IDs are consequently
valid. Build merges charge B-1 node visits, including empty internal merges.
Query bound evaluations visit at most V-1 slots; successful leaf row work is at
most S plus the at-most-seven seed evaluations. With current limits, depth is at
most 12 and the 32-entry DFS stack reserve is sufficient.

## Complete standalone payload audit

Let `N` be raw rows, `S` support rows, `V` padded slots, `C=65536` fixed scratch,
`R` accumulated encoded-record allowance, and `m<=256` a leaf's row count.
Raw storage is 24N bytes; the retained immutable sorted support is 8S bytes;
the two three-coordinate box arrays and three int64 arrays require 72V bytes.
The final permanent numeric ownership ledger is therefore `24N+8S+72V`.

The source admits these peak reservations before the relevant large operations:

| Stage | Charged working amount |
| --- | --- |
| Raw numeric validation | `56N + C` |
| Support identity validation | `24N + 12S + C` |
| Root gather, extrema, stable sort and row replacement | `24N + 72S + C` |
| Padded node allocation while original support remains live | `24N + 16S + 72V + C` |
| Leaf boxes while original support remains live | `24N + 16S + 72V + 64m + C` |
| Query row evaluation | `24N + 8S + 72V + R + 128m + C` |

Seed evaluation uses the same query expression with at most seven rows.
The 32N validation temporary covers the explicit full-size finiteness/magnitude
arrays. The 64S build reservation covers gathered XYZ, sort/permutation arrays,
immutable row replacement and conservative numeric scratch. The explicit extra
8S during node/leaf construction correctly accounts for the original support
buffer still live in the constructor. Query's 128m allowance exceeds the visible
gathered points, subtraction/products, ordered-add arrays, final zero mask/output
and separate extrema/tie temporaries. Fixed-schema vectors, seed tuples, bounded
stack and small JSON fragments use the fixed reserve.

For N=S=1,000,000, permanent numeric ownership is 32,589,824 bytes. The largest
declared construction reservation is 96,065,536 bytes, below the standalone
134,217,728-byte cap. This is allocation arithmetic, NOT an executed maximum
fixture, measured peak RSS, guarantee of NumPy allocator internals, or admission
of every possible query batch. The model intentionally excludes caller memory
and total Python-object/allocator/process RSS; a process supervisor remains
necessary.

Successful constructor work, before queries, is:

    row_visits        = N + 2S
    index_visits      = 5S
    coordinate_visits = 6N + 19S       (declared proxy)
    sort_work         = S * max(1, ceil(log2(S)))
    hash_bytes        = 24N + 8S
    node_visits       = B - 1

Queries add their charged row/index visits, nine coordinate-work units per row
evaluation, and bound-node visits. A later refusal may leave attempted charges
above a cap, but no positive batch survives. This is not a promise of sublinear
work: an adversarial transverse direction or poorly separated support can force
all leaves to be evaluated and honestly exhaust the query cap.

## Failure and report boundary

The public batch retains the constructor object via `__new__` before invoking
`__init__`, so a late construction refusal does not erase attempted statistics.
An internal query failure invalidates that index. The public wrapper rebuilds
its refusal envelope on ordinary exceptions, clearing projections, successful
status and raw-binding fields even if failure occurred during final report
encoding after the COMPLETE dictionary had been populated.

Each completed record is sized and charged before adding it to the retained
list. The additional 8192-byte report reserve covers the surrounding fixed
schema/list framing allowance; final streamed size is checked independently.
If either admission fails, the whole batch is refused. `discarded_projection_count`
counts previously retained records, not a claim that an unappended failing
record completed publication. Process interrupts may propagate without returning
any result; they do not return a successful prefix and still require the outer
supervisor's terminal-failure handling.

## Mandatory gates before context integration

- Keep V1 legacy arithmetic/default evidence unchanged; introduce an explicit
  new context/arithmetic configuration. Small numeric differences can change
  axial votes, exact pose deduplication and downstream search. Neither bitwise
  legacy equivalence nor unchanged trajectory is claimed.
- Optimize both whole raw buffer and complete assigned scan supports. Preserve
  the context's existing triangle-to-raw support rederivation and exact hashes.
  This standalone helper only validates the supplied support, not plane or
  triangle ownership.
- Add the planned shared immutable raw-buffer owner and lazy distinct-support
  index registry before integration. `_Index` currently owns one raw copy per
  instance; multiplying standalone instances is not an aggregate memory design.
- Precharge aggregate index rows/nodes, construction sorts/row work, live owned
  bytes, query work and reports alongside existing workflow costs. Independent
  128MiB caps per index do not bound their combined process. Do not reclassify
  skipped evaluations as removed support or inflate completed-work counters.
- Bind exact support/buffer identities, arithmetic/build configuration and query
  direction/origin/constant to context publication and cache keys. The standalone
  result does not itself echo every query parameter; it must not be treated as
  self-authenticating serialized query evidence. Publicly supplied or mutated
  private tree internals are outside its trust contract.
- Preserve whole-context failure clearing when the private class is reused;
  earlier successful private query returns cannot by themselves enforce a later
  batch/context refusal. Preserve all scheduled signs/slots and downstream gates.
- Independently review and freeze tests, harness, complete import closure,
  resource limits and exact PRE before execution. Tests must include a separate
  staged scalar/rational brute oracle, all-node enclosure controls, ties across
  traversal order, signed-zero/cancellation/subnormal cases, whole/subset and
  mutation controls, padded capacities around 256 and power-of-two boundaries,
  exact/one-over budgets, failed-evaluation counters and late report clearing.

No current source change or test count supplies a measured speedup, successful
registration, new score, metrological confidence or physical generalization.
