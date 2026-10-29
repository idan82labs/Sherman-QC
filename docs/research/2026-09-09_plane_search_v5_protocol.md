# V5 bounded declared-plane search: first source protocol

Date: 2026-09-09. This additive source is not an execution result or authorization.
No noisy extractor, V4 source, native reader, finite registration gate, physical
input, Gold, held-out, email artifact or product-default route is changed.

## Boundary and typed API

Owned source: `backend/cad_guided_plane_search_v5.py`. Dependencies are standard
Python and NumPy only. No IO, native geometry, optimizer service or application
initialization is imported. The API is:

```text
PlaneEquationV5(
    plane_id, normal, anchor_mm, offset_from_anchor_mm,
    normal_angle_bound_radians=0.0)
ObservedPlaneV5(
    equation, support_count, support_rms_mm, support_centroid_mm,
    proposal_kind='DECLARED')
propose_plane_search_v5(
    scan_planes, cad_planes, *, unit_assumption,
    policy=PlaneSearchPolicyV5(), budgets=PlaneSearchBudgetsV5())
```

Both tables must be exact tuples of the exact dataclass types. Inner vectors
must be three-element tuples of finite Python numbers; IDs are unique in each
table and nonnegative signed-int64-range Python integers. Booleans and bypass-
mutated invalid fields are refused. Normals must be unit within 1e-8; both normal
and offset are divided by the same admitted length. Normal cones are in [0,pi/4].
Position/anchor magnitudes are at most 1e6 mm, support count at most 1e6, support
RMS at most 1,000 mm. All fields are copied and revalidated; caller objects are
not retained as authoritative evidence.

The equation is **n dot (x-anchor) = offset**, not n dot x = offset. The geometric
anchor need not lie on the plane. `support_centroid_mm` is separate and used only
for within-scan exploration priority. An explicit nonblank unit-assumption string
of at most 128 characters is required, but does not establish physical units.

Every table and support summary remains `UNVERIFIED_DECLARATIONS`. Normal cones
are stipulated and uncalibrated; no spatial/offset-error model is supplied, so
translation uncertainty remains unavailable. `CORE` and `CLOSURE` are optional
proposal-origin scheduling categories, never flange/part labels. No result is
accepted registration or production-authorized, and physical plane/bend counts
remain unknown. The future wrapper must bind exact raw-row/triangle support.

## Frozen initial search schedule

1. **Scan admission and bank.** Default declared support minimum is three rows
   and maximum declared RMS is 0.1 mm. Every excluded or unscheduled witness and
   its identity remains in `scan_planes`. Select at most 128 eligible witnesses,
   alternating present proposal-origin categories; prefer unoriented normal
   novelty, then within-frame centroid spread, then smaller cone, then stable
   import-local ID. There is no largest-area ranking or raw-row removal.
2. **Scan triplets.** Every bank member is an anchor. Rank all other-member pairs
   by absolute normal determinant, then summed cone width and bank indices.
   Predeclare the first two partner pairs, or the only pair for a three-member
   bank. Exact same three scan IDs share a triplet record while retaining every
   originating anchor. At most 256 distinct triplets are scheduled. No closure,
   CAD match, pose residual or evaluator answer changes this selection.
3. **CAD angle index.** Build only the unordered CAD normal-pair table, not CAD
   triplets or a scan-by-CAD-by-CAD tensor. Pairs are lexicographic in sorted CAD
   IDs; their unoriented angles are stably sorted for an index. Chunked geometry
   evaluation avoids a full all-pairs vector tensor.
4. **Fixed pair-window strata.** For each available scan triplet's first normal
   pair, query a conservative indexed-angle window using the largest CAD cone
   radius. Select at most **12** evenly spaced integer rank positions, including
   endpoints (one slot uses the middle). Exact interval compatibility is tested
   only for those selected pairs. The other window entries are **unsearched**,
   not rejected or assumed compatible.
5. **Third-plane join.** For each passing selected CAD pair, test both ordered
   assignments against every other CAD plane. Test both remaining normal-pair
   interval overlaps, retaining all searched decisions. This is a bounded join,
   not all CAD triplets or all feasible poses.
6. **Unsigned and signed schedules.** Within each scan triplet, compatible
   assignments are ordered by scheduled pair-window rank, pair orientation, then
   ascending third CAD ID. Select at most **four** evenly spaced compatible
   assignment positions with the same endpoint/middle rule. Every scheduled
   unsigned assignment produces all eight normal-sign records, including
   geometric refusals. At most 1,024 unsigned and 8,192 signed assignments exist.

These quotas are a prospective, explicitly incomplete hypothesis policy, not
selection from a failed computational prefix. All unscheduled categories retain
separate denominators. Tuple-order permutation with unchanged IDs is canonicalized;
renumbering IDs can change tie/strata choices. No ID carries a part identity,
Gold label or learned correctness score. Independence from physical labels does
not imply every true association or pose is retained.

## Pair and pose mathematics

The unoriented pair angle is

```text
beta(i,j) = atan2(norm(cross(n_i,n_j)), abs(dot(n_i,n_j)))
interval = [max(0, beta-alpha_i-alpha_j-g),
            min(pi/2, beta+alpha_i+alpha_j+g)]
```

The engineering arithmetic guard `g` defaults to 1e-10 radians; it is separate
from the declared physical cone values. Overlapping intervals are only necessary
pair compatibility, not simultaneous joint correspondence. Parallel/repeated and
orthogonal planes can leave many indistinguishable choices. The coarse index
window is deliberately broader than exact per-pair compatibility when CAD cone
widths differ; selected false positives are explicitly rejected by the intervals.

Nominal triplet condition ratio must be at least 0.02. Additionally, with
`eta = sqrt(sum((2*sin(alpha_i/2))^2))`, its smallest nominal singular value must
exceed eta before using its intersection. These are conditional engineering
independence gates, not a spatial intersection-error certificate.

Each triplet solves its own intersection relative to its first anchor:

```text
N local = offsets + [n_i dot (anchor_i-anchor_0)]
x = anchor_0 + local
```

Every nominal equation is re-evaluated at the computed world intersection;
maximum backward residual must be at most 1e-8 mm. Floating-point intersections
are not assumed exact. For each signed assignment, proper Wahba/Kabsch normal
alignment uses the determinant correction, validates SO(3), and retains the
per-normal residual. The residual allowance is alpha_scan + alpha_CAD + 0.02
radians (the last term is an explicitly exploratory alignment gate, not confidence).
This allowance does not expand the separate pair-index schedule into an exhaustive
search over every alignment accepted by that gate.
NOMINAL_WAHBA_WITNESS_OUTSIDE_DECLARED_GATE rejects only this equal-weight
nominal-normal witness; it does not establish infeasibility within heterogeneous
normal cones. For example, let scan normals be e1/e2/e3, rotate only CAD e1 by
0.1 radians in the xy plane, and assign CAD cone widths (0.1, 0, 0). Identity is
feasible within those cones, but nominal Wahba splits the angle and violates the
second normal's 0.02-radian extra allowance. No constrained cone solver is claimed.

Translation is **t = x_CAD - R x_scan**. For imperfect normal congruence this
preserves origin covariance of the proposed intrinsic-intersection mapping;
direct nominal-offset matching is not substituted. No partial-patch centroid or
independently recomputed closest-to-origin plane anchor is treated as a matched
point. No intersection uncertainty is fabricated from normal cones alone.

## Evidence and completion semantics

Every scan plane records support exclusion, bank scheduling or omission. Every
CAD plane records selected-pair, searched-third and scheduled-unsigned participation.
Every scan triplet retains its solve/refusal and search denominators. Each searched
ordered pair retains two lossless little-bit-order packed rejection masks for its
remaining third-plane interval tests. Their order is the full ascending CAD-ID
table excluding the two assigned IDs; padding after `tested_third_count` is ignored.
Passing rows are the complement of the union of both rejection masks. Thus searched
incompatible identities remain reconstructable without millions of JSON rows.

Unsearched index-window occurrences, compatible-but-unscheduled unsigned
occurrences and scheduled signed refusals are different fields. No field implies
that an unsearched candidate was geometrically impossible. Signed attempted and
fully evaluated counts are distinct. Every selected sign is recorded before its
numerical work, so an unexpected failure can invalidate its EVALUATING record.
SCAN_TRIPLET_UNAVAILABLE_NO_CAD_JOIN covers unavailable nominal intersections,
including a numerical backward-residual gate; it is not geometric infeasibility.
CAD participation is exported from the charged usage ledger on ordinary failure
as well as success; a late solve/SVD failure cannot erase already attempted
participation or relabel a reached CAD plane as NOT_REACHED. A scan triplet is
EVALUATING before its join and becomes INTERRUPTED_BY_FATAL_SEARCH_FAILURE if
that join or its assignment work fails, retaining its prior status. Completed
earlier triplets remain completed and genuinely unstarted ones remain
NOT_EVALUATED; none of these partial states qualifies the whole search as complete.

`selected_policy_evaluation_complete` means the selected bounded policy finished;
`search_exhaustive` is always false and global pose uniqueness is NOT_ESTABLISHED.
Even an empty completed search is not observed absence. A posed candidate still
needs every original row of each admitted assigned support checked against its
complete finite CAD face union. The later wrapper may prospectively schedule at
most 48 refinements; it must disclose all omitted starts and keep the maintained
V2 global/local gates unchanged. That wrapper is not implemented here.

For example, three orthogonal planes have six interval-compatible unsigned
bijections. This schedule selects four and records the other two as omitted;
it does not claim all 24 proper cube-orientation alternatives were enumerated.
Symmetry remains an explicit negative/ambiguity control, not a uniqueness claim.

## Hard resource caps

| Quantity | Maximum |
| --- | ---: |
| Declared scan planes / CAD planes | 65,552 / 2,048 |
| Scheduled scan bank / scan triplets | 128 / 256 |
| Scan partner-pair tests | 1,048,576 |
| CAD unordered-pair records | 2,096,128 |
| Pair-join interval tests | 32,000,000 |
| Scheduled CAD pairs per scan triplet | 12 |
| Compatible temporary rows per scan triplet | 49,104 |
| Unsigned per scan triplet / total unsigned | 4 / 1,024 |
| Total signed assignments | 8,192 |
| Detailed report / pair geometry chunk | 64 MiB / 65,536 pairs |

Exact worst-case bounds are checked before their corresponding loops/buffers.
For 128 bank planes the partner total is `128*C(127,2)=1,024,128`.
For 2,048 CAD planes the index has `C(2048,2)=2,096,128` pairs. The initial join
maximum is `256*12*(1+4*(2048-2))=25,144,320` interval tests, below 32 million.
The compatible buffer is at most `12*2*(2048-2)=49,104` three-index rows, reused
one scan triplet at a time. No three-dimensional combinatorial tensor is formed.
Smaller hard budgets cause preflight refusal, not a best available prefix.

The outer boundary catches `Exception`, not `BaseException`. Unexpected input,
numeric, graph-free array, resource or bookkeeping failures invalidate every
usable pose; generated records retain prior status and become INVALIDATED.
Counters explicitly label attempted work. A report-boundary failure discards the
whole detailed report and emits only a small refusal envelope with denominators;
this error envelope is not a usable report prefix and is exempt from a zero
requested detailed-report budget.

First source/helper gate: one serial job, **60 seconds wall, 45 seconds worker
CPU, 1 GiB process-tree RSS, 1 MiB stdout** under the root-owned supervisor. This
source document does not execute or authorize a test. Any later connected or
physical protocol must independently freeze exact closures, schedules, gates,
artifacts, all-case denominators and budgets before execution.

## Prospective independent controls

- Strict typed snapshots, scalar/ID/normal/anchor and unit-string boundaries;
  bypass-mutated policies, immutable detachment and all allocation preflights.
- Tiny exhaustive reference for interval overlap, pair indices and both ordered
  joins; decode every rejection mask against explicit scalar comparisons.
- Full signed-permutation reference for the **scheduled unsigned subset**;
  retain exhaustive-reference alternatives omitted by the declared schedule.
- Independent nonsymmetric plane triad and independently posed/translated frame;
  oracle pose evaluated after inference only. Zero numerical failures and real
  correct candidates are required; abstention does not pass a positive case.
- Imperfect congruent normals, anchor reparameterization of the same plane,
  separate partial centroids, proper/reflection signs, length rescaling and
  origin translation. No matching-centroid shortcut.
- Parallel/near-dependent triads, uncertain independence and repeated orthogonal
  symmetry; no complete-search or global uniqueness claim.
- Unequal cone intervals creating conservative-window false positives;
  unsearched entries distinct from explicitly incompatible mask decisions.
- Endpoint/middle integer strata, both proposal-origin families, all unscheduled
  witness IDs, two partner choices per anchor and exact triplet provenance aliases.
- Late arithmetic/resource/report failures after a proposed pose, attempted vs
  evaluated counts, complete whole-report failure rather than qualified prefix.

These cases remain source plans until the independent test author/root freezes
and audits their exact fixtures and executable harness. No result is claimed.

## Research boundary

[GraffMatch](https://arxiv.org/html/2212.12745v1) motivates pairwise consistency
before pose estimation, not this specific bounded schedule or physical sheet-metal
performance. Its shifted affine-anchor argument must not be transplanted using
unmatched partial centroids or independently canonicalized closest-point anchors.
For x=0 and x+y=0, translating by (1,1,0) and re-canonicalizing such anchors changes
the shifted lifted angle from pi/4 to acos(1/sqrt(3)); this is a counterexample
to that proposed adaptation, not to the paper's transported-anchor theorem.
This source uses only normal-angle invariants and intrinsic triplet intersections.
