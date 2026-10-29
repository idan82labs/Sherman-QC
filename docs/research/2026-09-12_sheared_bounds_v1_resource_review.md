# Sheared bounds V1 — independent source-only resource review

Date: 2026-09-12. Reviewer: `process_v2_qualification`.

Decision: no remaining resource-accounting blocker for the declared internal,
trusted-box primitive. Mathematical enclosure correctness is a separate review.
No target imports, compilation, execution, tests, physical inputs, Gold,
attempts or private caches were accessed. This new document is the only write.

Reviewed source: `backend/cad_guided_sheared_bounds_v1.py`, 11420 bytes, SHA256
`36b87ea58c920774602d7f7cf53e817d5d130bf2b98bcc02dd8b8cce947201b6`.
The complete initial/final source and the final row-limit/zero-canonicalization
delta were read. This is not an integrated index, runtime qualification or speedup.

## Charges precede operations

`_Ops.binary` charges the maximum scalar/one-dimensional operand population
before its ufunc. All internal broadcasting is scalar-to-scalar/vector or equal
length, at most256 rows. Each nextafter is another charged ufunc, not free rounding.
`reduce` charges every input before min/max. `checked` reserves6L: two isfinite
passes, their two all reductions, one endpoint comparison and its all reduction.
Short-circuit failures only reduce actual work. Build coordinate admission's18N
charge conservatively covers isfinite/all/abs/comparison/all over3N coordinates.

For length L, iadd/isub cost4L; scale costs1+4L including its sign test; imul
costs12L (four products, six min/max passes, two nextafter passes). Scalar gap
costs17 including abs/abs/max and both checks. Explicit charges cover direction
signs, propagated abs, final interval intersection, and the two world-zero tests.
The initial64 reserve covers fixed finite/range/axis/slope admission comparisons.
Shape/type metadata, initial callback-callability validation, scalar bookkeeping,
encoding and callback internals are explicitly outside the numeric-work model.
These are source-defined abstract element charges, not hardware instruction counts.

One finding was repaired before this final pin: merge checked numeric axis/slope
compatibility before any charge. It now reserves64 before that predicate. Query
box admission was also moved after its initial charge. The final merge checks
exact positive integer row counts and `left <= MAX_NODE_ROWS - right` before the
sum; those fixed comparisons/subtraction fit the admission reserve. Repeated
legitimate merges cannot grow a row-count integer without limit: MAX_NODE_ROWS=1M.

## Exact successful-path charge totals

| Public operation | Derivation | Charged steps |
| --- | --- | ---: |
| build_leaf_box, N rows | 64 admission +3N snapshot +18N coordinate admission +4N z +8N+2 scales +12N residual subtraction +6N residual check +10N reductions +12 scalar checks | 61N+78; at N=256,15694 |
| merge_boxes | 64 admission +6 world min/max +2 z min/max +2 residual min/max +1 row sum | 75 |
| bound_sheared_box | 64 admission +101 affine enclosure/check +207 original-stage/error enclosure including zero tests +18 final expansion/check/intersection | 390 |

All are below MAX_LOCAL_STEPS=65536. These totals assume successful trusted
fixed-shape inputs; malformed inputs/overflow may stop earlier. No traversal,
global query count or amortized speed is claimed. A future index must charge
every build/merge/query cumulatively in its parent ledger, in addition to its
existing evaluation, node and index charges; each fresh `_Ops` resets only the
local guard, not the caller's aggregate budget.

`take` invokes the parent callback first, increments local steps only if that
returns, and checks the local cap before the covered operation. Thus a parent
exception propagates unchanged and no covered operation follows it. Local-cap
failure has already charged the attempted amount to the parent. There is no
catch that converts such failures into a successful box/bound. Numeric overflow,
nonfinite/reversed bounds, allocation failure and ordinary callback failure also
propagate. No partial returned prefix is manufactured. There is no numeric_steps
receipt on failure; attempted work is preserved by the parent's callback ledger,
which must increment/check before doing the operation it authorizes. The caller
must invalidate/clear its whole index or proposal output after any failure.

## 128 KiB scratch: plausible bounded reservation, not RSS theorem

The source requires the caller to reserve WORKING_BYTES=131072 before entry,
in addition to all live parent arrays and boxes. The primitive has no memory
callback and cannot enforce that outer reservation itself.

For a256-row build, the immutable3N snapshot is6144 bytes. Conservatively retaining
even all source-visible float-vector results ever created by z/residual work
would cost24N*8=49152 bytes; coordinate-admission abs adds3N*8=6144. All source-
visible vector Boolean outputs together add9N=2304 bytes. This deliberately
overstates their simultaneous lifetimes: total63744 bytes before scalar outputs,
small arrays/views/tuples, call frames and NumPy internal scratch. The remainder
67328 bytes is substantial for this fixed shallow routine's small objects and
bounded buffers. Query and merge operate on fixed scalar tuples, not N-row arrays.
The128KiB numeric/small-Python reservation is therefore plausible for the intended
owned execution shape; it is not a byte-exact CPython/NumPy allocator attestation,
OS RSS limit, or budget for arbitrary callback allocations, retained parent boxes,
input gathering, concurrent calls or exception tracebacks held by a caller.

`ShearedBox` is explicitly trusted internal data, not foreign-record validation.
Exact dataclass type alone does not authenticate field shapes, finite contents,
raw lineage or nested immutability. The resource argument assumes boxes produced
by this builder and compatible merges, not arbitrary externally constructed or
mutated instances. A later index must bind original raw IDs/support, keep original
row evaluation and charge persistent per-node storage separately. No new runner,
source-version gate or broad prerequisite is required by this primitive review.
