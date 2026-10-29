# Standalone sheared index V2: root source review

2026-09-12. Source-only review; no V2 import, collection, fixture execution,
physical input or retained cache access. This is not a pre-run authorization,
performance result, context integration or engine accuracy result.

## Reviewed source and scope

- `backend/cad_guided_raw_extrema_index_v2.py`: 16,616 bytes, SHA256
  `862b6926d1eeb8a2cd05656b5a0a69e501cfcdfda69f27a1036ef6c361e74057`.
- `tests/test_cad_guided_raw_extrema_index_v2.py`: 23,252 bytes, SHA256
  `799f3295d3a2c560557045d1e83e944cb6167b2d869139ee47a5552907b0007b`.
- Unchanged V1 index and the separately qualified sheared-bound primitive are
  the only candidate dependencies beyond NumPy/standard library. The exact
  source closure and ordinary runtime metadata must be bound by a new PRE.

Root read the complete V2 source in c9f04e. The combined test/V1 display
b302d9 was output-truncated; the complete test tail was separately read in
62db79. Root also read the primitive completely in e80616. The previous
primitive proof and 40-item result do not qualify this index or its efficiency.

## Correctness argument and limitations

V2 inherits the V1 constructor, raw/support byte snapshots, strictly increasing
support admission, original raw-ID seed selection, single stable coordinate
sort, tree membership, world-axis boxes, hash function and staged row evaluator.
The added shear is chosen only from at most seven existing raw seed rows and
at most 35 lexicographic triples. Dominant cross-product component division
produces represented slopes with magnitude at most one; unsupported or
degenerate selection falls back to zero shear. This is a bounded heuristic,
not a fitted plane, selected datum, CAD carrier or registration result.

For each occupied leaf, the primitive receives exactly the immutable raw rows
selected by that V1 leaf. Its represented sheared-coordinate interval and
exact-real reconstruction-residual interval are packed in four binary64
numbers. Internal intervals are child unions, with single-child copying for
the incomplete last subtree. World boxes, support spans and minimum raw IDs
remain the inherited ones. The trusted transient `ShearedBox` therefore
describes the same complete node membership, without substituting sampled
points for the source population.

The bound primitive encloses the ORIGINAL staged rounded row program; query
results still come only from that unchanged evaluator. V2 uses the intersection
to prune, while its child priority uses the returned original world interval.
The extrema and original-ID tie predicates are unchanged. An enclosing bound
cannot exclude a better value or a smaller-ID equal-value witness. The DFS
stack cap, per-query admission and all-or-nothing invalidation remain explicit.
The correctness argument relies on the previously reviewed IEEE binary64,
gradual-underflow and separate-operation assumptions, not real-arithmetic
reassociation or an exactly orthogonal floating-point rotation.

For zero slopes the optional local box cannot improve the original world-corner
enclosure; V2 deterministically disables it and retains attempted chooser
costs. This is a pre-execution source choice, not a result-dependent escape.
The internal owner does not authenticate arbitrary externally mutated objects.
Its raw rows and added packed array are bytes-backed; the inherited world-tree
arrays are unchanged, not newly made immutable.

## Work, failures and reporting

Every new declared phase is added to both its own attempted counter and the
unchanged cumulative coordinate-work limit before execution. Primitive charges
are forwarded through that same boundary. New gathers consume row/index
budgets; the packed array consumes hash and payload budgets. The constructor
reserves its metadata, packed storage, leaf gather plus primitive scratch, and
old/new packed-copy overlap. This is the declared owned-payload accounting,
not a claim of continuous total RSS enforcement.

The public wrapper allocates the object before calling its constructor so a
late construction failure retains attempted counters. Construction, query or
report failure returns no completed projection prefix or raw binding and
invalidates the index. A second-query failure records the discarded first
projection count. No refusal is a bend defect or a physical observed count.

Root accepts the fixture's common-ledger supplement `WORLD_BOUND_STEPS=37`:
six sign comparisons + six coordinate selections + two times (nine original
arithmetic operations + zero comparison and selection) + three finite/order
checks. Apply it only to actual calls to uncharged V1 world-bound code. Active
sheared bounds already charge their 390-step primitive plus packing. This is
a fixed source-level abstract comparison, not Python/hardware instruction
count, and does not claim every old operation is counted. The unchanged V1
construction charges cancel only when the same full input and query workload
is compared; failures must not be compared as cheaper completed prefixes.

## Fixture review

The source defines sixteen prospective items: four complete 32,768-row,
twelve-query comparisons; four small all-node enclosures; and eight narrow
controls. The module fixture creates an inert backend package and loads only
the three named files. Neither collection nor fixture authoring executes the
backend initializer, server, CAD imports or physical data access.

The full-population independent blocked oracle uses the original stage order,
positive-zero convention and first original row ID on ties. Small controls
cross-check it against scalar Python binary64 stages and inspect every occupied
node. Other controls cover cross-leaf ties, complete supplied support,
world-bound rather than intersection ordering, the added-work partition,
pre-work cap refusal, constructor failure, late-query whole clearing, owned
raw snapshots and disabled-shear positive zero. Tests never promote nominal
geometry to measurement output.

Four fixed populations include exact flatness, resolved thickness with unique
extrema, repeated extremal plateaus, and three folded panels. The last receives
fixed ideal proper Rx(5/13,12/13) then Rz(3/5,4/5), implemented in separate
represented operations. The oracle uses the actual represented rows, not
ideal-rotation coordinates. One raw-row permutation and the same twelve
literal queries are declared before execution. No outcome-selected query,
orientation, prefix or omitted negative population is allowed.

The index is built once per arm/population and receives all twelve queries.
One/three/twelve-query checkpoints are cumulative observations within that
same call. The instrumented CPU measure covers construction and queries, not
oracle generation or report serialization; V1 runs before V2 in each case.
This fixed small comparison is an exploratory discriminator, not a repeatable
hardware speed claim. Its measurement/order limitations remain visible.

## Decision

No source-level correctness blocker was found in this root review. Independent
resource review and exact-source PRE remain required before the one bounded
test invocation. Passing correctness tests alone will NOT earn integration.

Prospective efficacy requires complete exact endpoints/IDs on all four cases,
lower full-twelve-query common work on BOTH the resolved-thickness and folded
populations, and no aggregate construction-plus-query CPU regression over all
four populations. Flat/plateau overheads remain in that aggregate. Otherwise
retain the failure and revise the mechanism; do not integrate or rerun Gold11
on this basis. The standalone result cannot by itself authorize a physical
successor or prove the historical F1 was beaten.
