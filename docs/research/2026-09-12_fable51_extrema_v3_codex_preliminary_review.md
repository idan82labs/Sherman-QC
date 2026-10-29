# Preliminary independent source review: extrema V3 / context V4

## Decision and limits

No concrete P1 pruning, raw-ID tie, or completed-answer preservation blocker was
found in the bytes below. This is a PRELIMINARY SOURCE REVIEW while Fable 5.1 / max
worker `1519fdc2` owns implementation. It is not a frozen PRE, test result,
resource qualification, integration approval, speedup result, physical result,
or promotion. The reviewed candidate remains subject to change.

The two evidence/accounting observations below should be resolved in the new
prospective efficacy protocol before a passing test suite is used to justify
net improvement. They do not identify a wrong returned extremum. No particular
earlier rejected V2 efficacy rule is imposed on this new mechanism.

The reviewer read `AGENTS.md`, the newest `docs/CURRENT_STATUS.md` checkpoint,
`docs/REPO_MAP.md`, `docs/operations/astra-workflow.md`, and the released
`docs/research/2026-09-12_fable51_clean_f1_handoff.md`. Only the two new engine
modules, their two new test sources, and narrow predecessor source seams were
inspected. No target import, collection, test, compilation, geometry job,
physical input, Gold, artifact, cache, credential, Claude log or process was
accessed or executed. No engine/test file was edited. This report is the only
reviewer-created file.

## Examined source bindings

The four candidate files were read completely. Their hashes and sizes matched
between the beginning and end of the source inspection; this is not an atomic
freeze or a promise that they remain unchanged after this report.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_raw_extrema_index_v3.py` | 23208 | `2b0f4f9c9ed81d564b1846669b42531471590adb6b57d52dceb557ab6f67ca6e` |
| `backend/cad_guided_two_plane_context_v4.py` | 26196 | `e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25` |
| `tests/test_cad_guided_raw_extrema_index_v3.py` | 32860 | `018ddead1870500b096fa831e23c602166320a89aa226a4c5f10f42b6b455be1` |
| `tests/test_cad_guided_two_plane_context_v4.py` | 37574 | `0a193a608ecb21b19a71f41130989c5a519379faf6a54d8a4ed69d6e8555478b` |

Narrow predecessor comparisons, not independent full requalification:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_raw_extrema_index_v1.py` | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |
| `backend/cad_guided_two_plane_context_v1.py` | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_context_v2.py` | 11130 | `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663` |
| `backend/cad_guided_two_plane_context_v3.py` | 25845 | `9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d` |

All following line numbers refer to these bytes.

## Evidence/accounting observations and minimal repairs

### P2: the checked-in efficacy assertion is not a net-cost decision

`tests/test_cad_guided_raw_extrema_index_v3.py:378-404` requires fewer evaluated
rows over the first sixteen invented bend-axis queries and fewer coordinate
charges over the complete nineteen-query batch. It prints the changed sort
work but merely checks it against the new absolute cap. It does not require
lower total common work or compare build-plus-query CPU. The four consumed
32768-row populations at lines 359-375 assert correctness and print selected
costs, without an efficacy decision. The context comparison at
`tests/test_cad_guided_two_plane_context_v4.py:724-747` correctly makes no
directional cost assertion.

This matters because recursive construction can save coordinate evaluations
while adding materially more sorting and nodes. For an exact 32768-row support,
the source-only formula is nine partition depths: V3 sort charges are
`S * (15 + 14 + ... + 7) = 99S`, versus V1's `15S`. V3 standalone construction
coordinate charges are `33S`, versus V1's `25S`; V3 constructs 511 internal
nodes versus 127. These are deductions from the declared code model, not
measurements or a prediction that a particular complete workload regresses.

Minimal repair: label these assertions as coordinate/row-evaluation evidence,
then freeze a separate successor decision using complete construction plus all
queries, with a declared comparable work model and build-plus-query CPU.
Retain cheap-query, plateau and folded outcomes, including regressions and
refusals. Previously explored fixtures or leaf sizes are consumed development,
not prospective confirmation. Do not turn a future pass of the current source
tests alone into a net-speedup or physical-readiness claim.

### Accounting clarification: index-visit units changed during partitioning

V1 charges `2*count` index visits before its sort/permutation
(`backend/cad_guided_raw_extrema_index_v1.py:193-199`). V3 charges only
`segment_count` before each partition's coordinate gather, sort and
`segment[order]` permutation
(`backend/cad_guided_raw_extrema_index_v3.py:260-266`), and introduces a working
permutation copy at lines 250-251. The new tests deliberately assert this
convention: `3S + partitioned-row sum` in the single/two-leaf cases
(`tests/test_cad_guided_raw_extrema_index_v3.py:541-563`). Thus this is not a
missing cumulative-counter update: the chosen abstract per-partition unit is
different from the predecessor's pre-sort/permutation charge.

The counters are expressly abstract, not hardware-operation counts. This
review does not call inherited omissions (such as unexpanded world-bound scalar
arithmetic) new bugs, nor establish an RSS or scientific-answer defect from
this difference. Nevertheless, a raw sum of old/new counters should not be
called the same operation ledger without explaining the mapping.

Minimal repair: explicitly document what one index-visit unit covers in each
implementation and account the incremental gather/reorder/copy work consistently
in the prospective common-work comparison. If the intended unit remains the
old sort/permutation convention, adjust the NEW candidate and NEW literal tests
accordingly before freezing; never edit the old qualified source to harmonize
the evidence. CPU remains an independent necessary observation, not a substitute
for explaining the declared ledger.

## Numerical and identity reasoning

1. **The tree remains a partition of original identities.** V3 begins with
   strictly increasing, range-checked immutable support rows (lines 168-190).
   Each stable sort writes a permutation of one existing contiguous node segment
   (lines 250-268); child boundaries come from fixed leaf slots (lines 210-219).
   Skipping a node with no occupied right child does not discard its rows.
   Original coordinates are never replaced by representative or nominal points.

2. **Pruning still encloses the original rounded program.** Each nonempty leaf
   stores coordinate minima/maxima of its original members, and parents take
   unions (lines 281-301). For a fixed finite origin, rounded subtraction is
   monotone in the point coordinate; multiplication is monotone or antitone
   according to the direction sign. Sign-selected corners therefore bound each
   term. The unchanged staged additions and constant addition are monotone in
   their inputs. V3 evaluates those SAME operations at the corners
   (lines 82-89 and 304-312), not an exact-affine surrogate or a rearranged dot
   product. This argument assumes consistent binary64 rounding, gradual underflow
   and no reassociation; the sentinels at lines 92-100 are necessary checks, not
   full platform attestation. Source `diff` shows `_staged`, `_bounds`, `_evaluate`,
   `query` and `_arithmetic_sentinels` unchanged from V1.

3. **Raw-ID ties survive different traversal.** Every node stores the smallest
   raw ID among its members (lines 292 and 301). A bound equal to the current
   extremum may prune only when the entire node cannot improve the current
   witness ID (lines 343-347). Leaf extrema choose the smallest ORIGINAL raw ID
   among equal represented values (lines 322-324). The partition stability is
   helpful for determinism but is not relied upon as the final tie rule.

4. **Finite and degenerate cases fail closed or remain ordinary extrema.** Raw
   points, vectors, origins and constants are finite and magnitude-bounded
   (lines 75-79, 145-160 and 331-334). Under these limits, the three products and
   staged sums cannot approach binary64 overflow. Empty raw/support populations
   are refused, as are duplicate/decreasing/out-of-range support IDs. Zero spans
   merely choose a deterministic axis; they do not remove rows. Single-leaf,
   repeated-coordinate, cancellation and all-zero-direction cases still reach
   the same rounded program and smallest-ID tie rule. Final signed zeros are
   canonicalized to positive zero at line 89. Empty padded tree nodes are not
   queried because child admission checks their occupied interval at line 360.

## Integration, failure boundary and resources

The V4/V3 source diff changes the helper import, version/partition metadata,
leaf-size descriptions, construction-counter description and the declared
coordinate cap, not `_slots`, pose construction, family order, CAD projection
arithmetic or downstream admission rules. In V4, `rotation.T @ u` and
`dot(u, t0)` remain literal at line 342; exact represented direction/origin/
constant values remain in the projection-cache key at lines 306-308. There is
no coefficient snapping, CAD-derived measured answer, support culling, changed
slot denominator, or observed-count substitution in this reviewed seam.

The equivalence conclusion is conditional on both versions completing under
their declared policies. V3 standalone coordinates rise from 256M to 512M and
sort work from 64M to 256M; V4 context coordinates rise from 256M to 512M.
New tree shape also changes row/node/sort/memory consumption. COMPLETE versus
UNAVAILABLE can therefore legitimately differ. A future result must distinguish
mechanism improvement from a changed cap and must not claim bitwise equivalence
to the older BLAS scan program or equivalent admission outcomes at every budget.

The index snapshots raw/support bytes and exposes an immutable permutation;
construction precharges root scratch, partition scratch, persistent tree arrays
and live original support (V3 lines 148-190, 226, 250-287). The dominant
`64*S` root reservation and partition `8*S + 32*n` reservation are conservative
numeric-payload allowances for the visible allocations; no concrete newly
introduced payload under-admission was found. They are not measurements of
NumPy/native/Python total process RSS.

The connected path shares the context-owned raw snapshot, rather than copying
and rehashing it per support (V4 lines 137-175, 192-209). Index counters are a
shared dictionary and every charge delegates to the parent guard (lines 142,
177-189); inherited parent memory calls dynamically include all live registry
payload and metadata (lines 113-118). Exact support aliases are authenticated
before projection-cache lookup (lines 230-247 and 303-305). No per-support reset,
cache-hit bypass of support equality, or favorable-prefix return was found.

The standalone batch rebuilds a refusal shell and drops prior projection records
after a late failure (V3 lines 425-433). V4 similarly clears all geometry/pose
evidence after any ordinary exception and releases registry ownership in
`finally` (lines 459-471). Query/registry failures mark internal state invalid;
BaseException interrupts propagate instead of masquerading as completed
scientific refusals. Changed capacity may prevent a previous refusal, but it
does not silently keep a partial supported pose after a later fatal query.

## Test-source coverage and next gate

The authored standalone tests include scalar staged-rounding oracles, subnormal
and cancellation cases, every-node member enclosures, permutation bijections,
original-ID ties across leaves, immutable snapshots, literal counters, exact-cap
and one-below refusals, late-prefix discard and interrupt handling. The connected
tests include whole-buffer versus assigned-support semantics, 900000-row capacity
with only tiny supports, exact alias collision checks, cumulative parent caps,
late failures, owner-cycle release and V3/V4 completed geometry comparison.
These are source coverage observations, NOT passed tests or full-workflow capacity.

One narrow test wording improvement: the connected test named
`test_v4_geometry_evidence_is_byte_identical_to_v3` compares decoded dictionaries
and numeric lists (lines 712-739). Python float equality does not distinguish
`-0.0` from `+0.0`. The standalone helper uses packed binary64 assertions, and
scan output is explicitly positive-zero canonicalized, so no concrete changed
zero was found. To test the advertised byte-level full-geometry claim, compare
canonical serialized bytes or packed float fields as well as IDs, rather than
calling ordinary object equality a byte comparison.

Authorized next action: Fable may resolve the bounded accounting/evidence items
and hand off final source/test/harness/protocol bindings. Root then needs an
independent exact frozen PRE and relevant resource gate before the sole scoped
execution. This report authorizes no execution, old-attempt retry, physical
successor, cap adjustment to a started run, or promotion. Any subsequent source
delta requires reinspection; no numeric F1 or physical improvement is asserted.
