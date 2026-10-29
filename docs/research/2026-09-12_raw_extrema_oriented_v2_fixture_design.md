# Raw-extrema oriented V2: finite adversarial fixture design

Date: 2026-09-12. SOURCE-ONLY PROPOSAL; no candidate, tests, collection, imports, simulations or performance experiments were executed. This document does not authorize a run. Read canonical AGENTS.md, current-status top, and the complete 367-line V1 source and 631-line V1 test source. No raw, Gold, cache, attempt, authorization or terminal-control artifacts were read. Only this new document is authored.

## Question and invariant

Can a bounded oriented hierarchy reduce the total work of exact raw-row extrema queries, while producing precisely the original staged binary64 values and minimum original-row-ID witnesses? The mechanism hypothesis concerns lost coordinate correlation in world-axis boxes. It is not a claim that the orientations or covariance of the physical failure queries are known.

The baseline is `backend/cad_guided_raw_extrema_index_v1.py`, SHA256 `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121`. Its complete original tests are source context, not a suite to recollect here. Useful existing seams are scalar_projection/brute/assert_exact, all-member node enclosure, anticorrelated-box negative, cross-leaf ties, snapshot mutation, exact/one-below accounting, late whole-clear and the three-axis 900,000-row control. That axis-only control is not evidence about oblique queries. A future V2 test module should carry independently authored fixture/oracle code, not import the old test module or backend package initializer.

Preserve `STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1`: subtract each origin coordinate; multiply each direction coordinate; add terms 0+1, then term 2, then constant, each separately rounded binary64; canonicalize final zero to positive zero. Do not substitute BLAS, FMA, reassociated dot products, exact-real extrema or the value computed in a fitted local basis. Compare the two returned endpoint **bytes**, minimum and maximum original raw IDs, complete support population, point/support hashes and all-or-nothing refusal envelope. Duplicate positions retain distinct raw identities.

Orientation is only an acceleration index. Every supplied support row remains eligible; no plane fit, residual trim, welding, coordinate replacement, scanner calibration, pose acceptance or CAD measurement is introduced. Orientation degeneracy must have a reviewed conservative fallback or explicit whole refusal, never an uncertified tight bound. Refusal remains negative evidence.

## Proof seams the fixtures must exercise

For every nonempty node and admissible query, the proposed lower/upper bound must enclose the original staged output of every original member. An oriented-coordinate range alone is insufficient. Construction must bind a basis, translation, local coordinate ranges and a conservative reconstruction residual for the **stored** original coordinates; query bounds must also enclose rounding of the original subtraction/multiplication/addition sequence. Rounded basis orthogonality is not exact orthogonality. Converting back to a loose world box may lose the intended benefit; an algebraically transformed affine value without residual/roundoff bounds changes the target program.

Independently test the proof components and final enclosure. For small fixtures, inspect every nonempty node and every member, using a separate scalar oracle, not only the final extrema. Verify node membership coverage, original minimum raw ID and the correlation of the claimed reconstruction residual with original coordinates. The test author must derive these checks independently from the math specification rather than call the candidate's proof helpers. Omitted-residual, inward-rounding and ignored-original-rounding fault controls must be constructed to produce a real witnessed bound violation; a mutation that is accidentally harmless is not a successful adversarial control.

Existing pruning logic distinguishes strict improvement from tied value with a smaller raw ID (V1:283–300). A bound equal to an existing witness is insufficient to prune if the node contains a smaller eligible original ID. Test each min/max side independently, including a node that can improve only one side. Reordering traversal may change work but not endpoint bytes or tie witnesses.

## Independent deterministic fixture families

Use integer/dyadic formulas and literal hexadecimal binary64 values; no physical coordinates, CAD-derived dimensions, learned direction set or random search for favorable cases. Freeze all row-order permutations, support masks, directions, origins and constants in source before PRE.

P, oblique plate: enumerate `(u,v)` on a rectangular integer grid, center both integer coordinates, and form original stored rows `(u,v,u+v+epsilon*h)`, with bounded deterministic `h=((17*i+13*j) mod 33)-16`. Construct the third coordinate by its declared scalar operation order and retain its rounded bytes; a real-plane formula is not an exact-output oracle. Query normals `(1,1,-1)` and its signed dyadic multiples, plus fixed near-normal perturbations. V1's sign-selected world-box corners admit artificial normal variation although rows are correlated.

L, oblique line: stored rows `(t,-t,epsilon*h)` with a modular dyadic `h`, and queries `(1,1,1)`, `(1,1,0)` and fixed near-normal perturbations. Rank-one covariance/degenerate secondary orientation must not create an arbitrary uncertified frame.

For both, epsilon=0 is a mandatory **negative efficacy control**. A nonzero reconstruction/roundoff enclosure can straddle tied extrema at every node and require all leaves. Even an exactly real-planar set can have only original-program rounding variation. No universal pruning or speedup is required for this condition. The fixed nonzero thickness sweep tests whether geometric spread dominates arithmetic enclosure enough to help. Do not remove the epsilon=0 or near-roundoff cases from the final performance denominator.

Controls also include near-axis dyadic shear, a full-dimensional integer lattice with no preferred plane, and duplicate/cross-leaf extrema. Small special values include signed zero, smallest subnormal, smallest normal, adjacent floats, half-ULP addition ties, mixed signs and large cancellation. Row permutation must alter array positions only as declared; raw IDs always refer to positions in that particular original supplied array. Expected minimum IDs are recalculated from its full eligible support, never copied from a geometric position label.

## Small matrix: 32 planned controls

Each numbered variant below is a distinct retained control. Counts are fixture-control counts, not a claim of pytest node count; exact test nodes must be frozen after source authoring. Default maximum population is 2,049 rows, at most 12 queries per control; scalar-rounding microcases have at most 17 rows. All-member/node enclosure checks use the original immutable membership map. No broad inherited validation suite is repeated.

| Group | Controls | Fixed content and required observation |
|---|---:|---|
| Exact oblique plate | 2 | Whole support and noncontiguous sorted support; normal/cancellation query, complete endpoint/tie correctness; no required pruning gain. |
| Exact oblique line | 2 | Rank-one set and repeated-coordinate line; truthful orientation/fallback, every-node enclosure. |
| Dyadic thickness | 2 | Plate epsilon=2^-20 and 2^-6, fixed near-normal queries; measure construction+query work, not a hardcoded leaf-only victory. |
| Near-roundoff thickness | 3 | Plate epsilon=2^-40, adjacent-float perturbation of third coordinate, and one-sided residual sign; independently certify actual original rounding values. |
| Near-axis/volume controls | 3 | Exactly axis-aligned plate, dyadic shear 2^-20, full-dimensional lattice; correctness regardless of overhead/regression. |
| Ties and one-sided improvement | 3 | Equal extrema in separated leaves with smallest raw IDs late in traversal; min-only improvement; max-only improvement. Include duplicate positions and zero-direction query in tie control. |
| Immutable lineage | 2 | Mutate caller points/support after snapshot; mismatched/incomplete node membership or residual-binding metadata must not authenticate a bound. Avoid pretending arbitrary private-internal tampering is an existing public API. |
| Large translated cancellation | 2 | Coordinates/origins near opposite +/-1e6 bounds, directions with mixed +/-1e6 components; nearby representable values requiring original-stage rounding. |
| Underflow and signed zero | 3 | Smallest subnormal arithmetic; normal-to-subnormal products; cancellation to signed zeros with positive-zero output canonicalization. |
| Proof fault witnesses | 3 | Actual witnessed failures for removed reconstruction residual, inward endpoint rounding, omitted original-program rounding. Each has a correct positive premise and independently located violating member. |
| Numerical refusal/fallback | 3 | Nonfinite/out-of-domain public input; valid degenerate-basis data; injected nonfinite bound/intermediate after a valid first query. Verify no successful prefix survives the actual refusal branch. |
| Resource/late-failure seams | 4 | New construction reservation exact/one-below; new query-bound reservation exact/one-below; later query exhaustion invalidates reusable index and whole batch; final report failure removes bindings and successful prefix. Each paired boundary is one declared control with both subobservations retained. |
| Total | 32 | All remain in the final ledger. |

The last group is scoped to costs/seams newly added by oriented construction, not a re-enumeration of every old V1 scalar policy. After the implementation exists, independently derive exact required charges for a tiny literal fixture and pin them before execution; do not obtain favorable cap values by running the candidate first. If a proof-fault witness or required charge cannot be established source-only, mark that specific control unresolved before PRE rather than silently weakening it.

Overflow distinction: the admitted V1 domain has absolute points/origins/directions/constants <=1e6. Thus original subtraction <=2e6, each product <=2e12, and the final magnitude is bounded by approximately6e12+1e6, far below binary64 overflow. Do not fabricate a valid-input V1 overflow case. Out-of-domain infinities/overflowing conversions test admission; injected overflow/nonfinite intermediates test V2 whole-clear behavior; valid tiny/degenerate basis data test avoidance of unsafe normalization/reciprocal overflow. Underflow is genuinely inside the public domain and must preserve gradual-underflow semantics or explicitly refuse its arithmetic environment.

## Scale matrix: eight 900,000-row controls

Each control constructs exactly 900,000 original rows and evaluates a frozen batch of 12 projections in V1 and V2 sequentially, under identical old public budgets. No support reduction after a timeout. Only one population/index/oracle block is live at a time except the explicitly accounted caller plus owned copies. Array order uses a fixed bijection `k -> (899999*k) mod900000`; grid shape is1000x900. All point/direction/origin/constant values must pass the original1e6 admission domain by an independent formula check.

| Scale control | Population |
|---|---|
| S1 | P with epsilon=0: exact oblique planar negative efficacy control. |
| S2 | P with epsilon=2^-40: arithmetic-scale thickness. |
| S3 | P with epsilon=2^-20: small resolved thickness. |
| S4 | P with epsilon=2^-6: clearly resolved thickness. |
| S5 | L with epsilon=0: rank-one/tied negative efficacy control. |
| S6 | L with epsilon=2^-6: resolved line thickness. |
| S7 | Axis-aligned/near-axis plate control using fixed shear2^-20. |
| S8 | Full-dimensional lattice control, no planarity assumption. |

Freeze 12 queries per population: six signed dyadic multiples (+/-1, +/-2, +/-1/2) of its declared correlation-testing direction, and six literal near-direction/translated-origin/constant variants. All 12 coefficients and operation orders must be enumerated in the eventual test source, not selected after seeing pruning. Keep translation magnitudes safely within admitted bounds. Repeat queries only if explicitly present in this fixed list; do not disguise warmed-cache reruns as independent trials.

Large oracle: stream complete original eligible rows in blocks <=65,536, execute independently written separated scalar-stage array operations, and update exact min/max and smallest original-ID witnesses. It must never use V1/V2 _staged, their indexes/bounds, BLAS or candidate-selected rows. Its implementation is cross-checked against the independent scalar oracle on the small controls; small adversarial rounding microcases additionally use a separately specified exact-rational, round-to-binary64 stage oracle with explicit tie-to-even/subnormal handling. The large vector oracle is not itself a formal arithmetic proof. Account oracle time/memory separately; do not count it as V1 or V2 speed. No900k Python object list or dense query-by-point matrix is needed.

With whole support, V1 public construction charges25N coordinate visits and a worst-case12 full-query batch adds108N, totaling119,700,000 before node-bound operations (which V1 currently charges as node visits rather than coordinate visits). This fits its256M coordinate cap and makes a complete-baseline comparison plausible, not proven. V2 must retain the same old public maxima and charge its added work; refusal is a recorded negative, not permission to increase maxima. Each proposed scale input is21,600,000 XYZ bytes, not a claim that total process memory is21.6MB.

## Total-work decision, including unsuccessful cases

Keep correctness and efficacy decisions separate. A byte/tie/lineage/enclosure mismatch fails correctness regardless of performance. Candidate whole refusal where V1 completes is retained and prevents a broad supported-domain acceleration claim; any baseline or candidate budget timeout remains in the fixed ledger and is not assigned an invented work ratio. No ratio is computed between a completed workload and a refused prefix.

For every control retain construction, per-query and cumulative charges: original row/index/hash/sort/coordinate/node categories, query rows/leaves, added orientation estimation, residual construction, basis storage, bound arithmetic, extra sorts, temporary peak and report bytes. V1 node-bound evaluation performs two original staged corner programs; counting only legacy coordinate_visits would conceal that work. Define a common **source-derived** arithmetic ledger before PRE: count actual V1 corner-program operations and actual V2 enclosure operations, including every construction operation. The independent source reviewer must fix each coefficient/increment site before results exist. Report the whole work vector; an arbitrary fitted conversion to CPU-instruction units is forbidden.

Proposed efficacy rule for root to accept/freeze before execution: all40 controls retain their prescribed denominator and correctness disposition; all eight large comparisons complete and match; the common total arithmetic ledger over the eight equal-query workloads is at least20% smaller for V2, with separate per-control deltas; total measured V2 build+query CPU is no greater than V1 over that fixed matrix. S1/S2/S5 regressions remain visible and enter the aggregate. Failure of these proposed thresholds means efficacy not demonstrated, not permission to discard an inconvenient thickness, change the metric or rerun unchanged code. These thresholds are prospective engineering choices, not inferred physical-effect sizes. If the source-derived work model cannot fairly compare both implementations, do not substitute leaf counts: report efficacy unresolved and retain wall/CPU plus uncombined vectors.

This is not a requirement that every fixture accelerate, nor proof of performance on arbitrary orientations. A benefit on S3/S4/S6 but none on S1/S2/S5 would support a limited conclusion: acceleration when true variation exceeds conservative arithmetic width. Exact-normal data with only roundoff variation may still visit every leaf; those are deliberate retained negatives.

## Proposed bounded execution envelope and PRE

No run is authorized by this document. Source authorship, exact independent arithmetic/source review, fixed test-node ledger and explicit independent pre-run audit precede any invocation. Proposal: two named phases within one consumed qualification, small then scale, no concurrent heavy job and numerical threads one. Only the candidate/baseline pure modules, explicit synthetic test/oracle/harness files, protocol and declared Python/NumPy/pytest runtime may load; no backend package initializer, native CAD/Open3D, scipy, network or project data access. No autoplugins, conftest, pytest cache or bytecode creation.

- Small phase:32 controls, <=2,049 rows each,30CPU/45wall seconds,1GiB process tree,1MiB log and8MiB report.
- Scale phase:eight controls,900,000 rows each,12 queries each, V1 then V2 in each fixed slot,1,200CPU/1,500wall seconds total,8GiB process tree (stricter than repository16GiB),2MiB log and16MiB report. These are finite proposed caps, not runtime promises or approval to raise a started limit.
- Outer envelope:1,620wall seconds, including75seconds terminal/quiescent publication reserve beyond the two phase wall caps; one heavy child at a time. Keep process/RSS supervision and writer-quiescence responsibilities explicit; sampledRSS is not an OS allocation guarantee.

The independent PRE must bind source hashes/full module-load closure, exact candidate schema/arithmetic literal, operation-accounting coefficients, all40 fixture controls and actual collected-node denominator, literal query lists, scalar/rational/blocked oracle independence, input generator byte formulas, work/memory/report maxima, dependency hashes, fixed outcome categories and efficacy rule. It must verify no physical paths/artifacts are reachable and no other heavy job overlaps. A future harness must preserve every failed/unknown slot, logs and attempted counters, stop on exhausted global allowance, and perform retention/cleanup only after owned-writer quiescence. No retries, cap increases or reduced populations under the same consumed attempt. Independent POST must reconcile exact terminal outputs and negative cases before claiming qualification.

Scope stop: this document is a bounded fixture proposal only. It does not implement oriented bounds, certify their proof, authorize synthetic execution, modify the frozen engine, or create a new physical benchmark prerequisite. Parent and independent math reviewer own the candidate/proof decisions.
