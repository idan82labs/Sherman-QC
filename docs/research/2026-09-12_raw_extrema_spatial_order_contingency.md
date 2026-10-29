# Raw-extrema spatial-order contingency — 2026-09-12

Disposition: plausible bounded NEW-version hypothesis, not selected, implemented,
qualified or released. This memo does not assert that V2 failed. Await the frozen
V2 efficacy decision; do not alter its source, tests, budget or decision rule.
Read-only source/math review only: no target import, execution, collection,
physical inputs, Gold, attempt artifacts or caches were accessed.

## Source basis and proposed difference

V1 (`backend/cad_guided_raw_extrema_index_v1.py`, SHA256
`668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121`)
gathers the complete support and derives root extrema/raw-ID seeds at lines
176–190, sorts one longest root axis stably at 191–197, then builds fixed
256-row leaves and bottom-up boxes at 203–236. It need not change the raw
snapshot, support set, arithmetic or geometric model to change this permutation.

The contingency is ONE raw-only three-coordinate key and ONE stable sort,
followed by the existing leaf/parent world-box construction. Keep the same
support-order-derived seeds, original staged evaluator (250–262), minimum raw
ID per node (228), tie-aware pruning (280–281), and world-bound child priority
(293–301). Give the ordering, quantization and counter model explicit NEW
identities; do not label the resulting tree the old single-axis tree.

A concrete inexpensive option for later specification is a fixed 10-bit grid
per axis, packed into a 30-bit unsigned key. For each represented root interval
`[lo_j, hi_j]`, compute a separately staged coordinate:

`s_j = RN(hi_j-lo_j)`, `u_j = RN(p_j-lo_j)`,
`v_j = RN(u_j/s_j)`,
`q_j = uint32(clamp(floor(RN(v_j*1023)), 0, 1023))`.

Here subtraction and division also mean separate binary64 operations. If
`hi_j == lo_j`, set that axis key to zero without division. All represented
input coordinates are finite and bounded by 1e6; the represented positive
span is at most 2e6. Explicitly check intermediate finiteness before conversion,
allow gradual underflow, and precharge every attempted operation. The clamp
prevents a rounding endpoint from producing an out-of-range integer. This is
an ordering recipe, NOT an enclosure of an exact coordinate or a geometry fit.

Interleave x/y/z bits in a fixed documented order. A compact unsigned spread
can use shifts 16, 8, 4, 2 with masks `0x030000FF`, `0x0300F00F`,
`0x030C30C3`, `0x09249249`, then OR `spread(x)`, `spread(y)<<1`,
`spread(z)<<2`. The bit identities require independent review/tests before use;
this memo does not execute or certify code. Stable sort equal keys in the
original increasing support-row order. Do not use CAD, case names, labels,
query directions or physical outcomes to choose grid resolution/orientation.

## Correctness is independent of key fidelity

Any permutation containing every supplied raw ID exactly once is sufficient.
Quantization collisions, ties, subnormal rounding, bin-boundary movement and
even a poor key do not change the original staged values. Each leaf box must
still be reduced from its ORIGINAL represented points, never grid-cell corners
or reconstructed coordinates. Parent unions then enclose their descendants;
V1's sign-selected rounded corner program still bounds every descendant value.
Keeping each node's actual minimum original raw ID preserves tie pruning, and
leaf ties must continue to select the minimum raw ID, not first key position.
Positive-zero canonicalization remains unchanged.

Do not sort the raw point buffer itself or change the support hash. Reject
invalid/overflowing construction rather than publish a partial permutation.
All attempted charges survive whole-batch refusal. Deterministic ordering is
desirable for cost reproducibility, but exact extrema correctness does not
depend on matching an ideal real-number quantizer. A new tree loses V2's
same-tree leaf-visit dominance argument: it may evaluate MORE rows than V1.

## Why it might help, and where it cannot

Single-axis slabs can contain points far apart along both other coordinates.
Grouping spatially adjacent rows can instead produce smaller world boxes on
locally separated folded panels, without finding one plane for the whole part.
This attacks node overlap directly and adds no sheared bound to each query.
It is a plausible improvement for a folded whole-buffer support, not proof
about the unseen spatial distribution or query directions of any physical run.

Worst cases remain: a Morton interval can cross coarse cell boundaries; points
in one cell retain an unhelpful raw-ID order; interleaved panels or coincident
rows cannot be separated spatially; outliers waste grid resolution. Independent
axis normalization can overweight tiny noise in a nearly degenerate dimension.
A common-scale grid avoids that particular effect but wastes bits in thin axes;
choose neither adaptively after seeing results. More bits cost wider keys and
do not remove all discontinuities. All-equal keys are a legitimate deterministic
fallback, not a license to discard rows or refund attempted work.

For an exact oblique plane and its normal, smaller axis boxes can still have
strictly wider bounds than every staged output at EVERY leaf. For example,
points `(t,t,0)`, direction `(1,-1,0)`, zero origin/constant have staged
output +0, while any
leaf with a nonzero t range has world bounds straddling zero. Spatial ordering
does not solve this cancellation case. No universal acceleration or recovery
from flat/plateau worst cases follows.

## Construction, query and memory ledger

Let N be raw rows, S support memberships, B the existing tree's node slots.
V1's declared coordinate construction is `6N+19S` (line350), sorting is
`S*ceil(log2(max(2,S)))`, and persistent numeric payload is `24N+8S+72B`
bytes. V1's sort-stage temporary allowance is 64S bytes (181), not an unlimited
workspace. Keep its cumulative caps (35–39); sorting integer keys is not grounds
to claim a measured linear-time sort. Key computation alone is bounded O(S).

A chunked compact 10-bit key construction could conservatively reserve about
80S additional abstract scalar steps, including normalization, conversion,
spreading and key writes; this is a DESIGN allowance, not a source-verified
exact charge. A literal per-bit implementation costs materially more. Freeze
the eventual per-operation ledger before running it. At S=1M, 80M extra steps
already consume substantial room in the unchanged 256M coordinate cap.
Retain old sort charges, and separately count key generation, scatter/copy,
hashing if retained, and attempted sorting/index work; no free key work.

By comparison, enabled V2 (16616B, SHA256
`862b6926d1eeb8a2cd05656b5a0a69e501cfcdfda69f27a1036ef6c361e74057`)
adds at least 64S leaf gather/primitive steps, leaf/merge/packing/chooser work,
32 times B persistent bytes, and 390 primitive steps plus adapters per node bound
(lines130–203). The key hypothesis need not be cheaper to BUILD; its attraction
is avoiding that new repeated query cost. For fair comparisons retain the
declared common supplement for old world-bound work; do not compare only leaf
visits or legacy coordinate counters.

A uint32 key array costs 4S bytes transiently; discard it after permutation.
Use bounded chunks for normalized coordinates and bit-spreading scratch.
Explicitly reserve coexistence of root gather (24S), keys (4S), argsort indices
(8S on the pinned runtime), ordered rows (8S), immutable replacement (8S),
old support, sorting scratch and chunk buffers before allocating. A prospective
96S temporary reservation plus 128KiB chunk reserve gives at N=S=1M
`24N+8S+96S+131072+65536 = 128196608` bytes, below the unchanged 128MiB
payload cap. This is a feasible envelope, NOT certification of NumPy's native
sort workspace or process RSS; source/runtime review must justify the workspace
allowance. Drop root gather/keys/sort arrays before allocating tree arrays and
preserve the original support/replacement overlap accounting. No per-row Python
integer list, second full point snapshot, or persistent grid is needed.

## Bounded next decision, only if requested

If V2 fails its frozen efficacy rule, this is a reasonable next standalone
candidate to compare on the SAME predeclared folded/resolved/flat/plateau
discriminators, retaining full constructor plus 1/3/12-query work and CPU.
Before any new execution, require exact permutation/set preservation, original
staged scalar/witness equality, real descendant bounds, collision/degenerate
controls, whole-refusal accounting and declared peak payload. Freeze the new
source and decision independently. Do not quietly replace or rerun the V2 gate,
expand into workflow/physical qualification, or tune spatial parameters on Gold.
