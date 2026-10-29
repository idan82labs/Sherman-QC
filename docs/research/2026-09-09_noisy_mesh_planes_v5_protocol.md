# V5 noisy mesh CORE + CLOSURE: source-only protocol

Date: 2026-09-09. This is an additive extractor proposal, not a result, execution
authorization, noisy-scanner qualification or promotion. Existing V4 source and
all retained receipts remain unchanged. No correspondence, CAD matching,
registration, native geometry, physical inputs or product-default routing is
implemented here.

## Owned source and API

`backend/cad_guided_noisy_mesh_planes_v5.py` imports only Python standard-library
modules, NumPy and SciPy sparse graph operations. It does not import the legacy
application or the older quadratic research discovery modules.

```text
extract_noisy_scan_plane_proposals_v5(
    points, triangle_raw_rows, *, position_error_bounds_mm,
    policy=NoisyMeshPlanePolicyV5(), budgets=NoisyMeshPlaneBudgetsV5())
```

Inputs are numeric NumPy arrays: original `N x 3` positions and integer `M x 3`
original vertex-row triangle incidence. A position bound is either an explicit
finite Python scalar or an `N`-element numeric NumPy array; it is nonnegative and
at most 1,000 mm. Booleans, malformed shapes, object arrays and invalid indices
are refused. Policies and budgets are copied and fully revalidated, including
objects whose frozen dataclass fields were bypass-mutated. Input snapshots and
returned lineage arrays are immutable byte-backed copies with little-endian
digests. Caller mutation cannot change admitted support.
The cheap pre-snapshot range checks are repeated on the owned triangle indices
before any indexing; owned point/triangle/error-array dimensions and value
invariants are also revalidated. Thus a caller mutation during snapshot
preparation cannot silently introduce NumPy negative-index aliases into the
admitted lineage. This is not a security claim against native writes to memory.

The unit is only `PROVISIONAL_ARRAY_MM_UNVERIFIED`. The supplied error is a
simultaneous Euclidean position-error **stipulation**, never a fitted residual,
CAD tolerance, point-density estimate, scanner calibration or confidence level.
Future physical workflow admission must separately bind units and provenance.

The result is an internal **array-native** payload, not a JSON report. `cores`
and `closures` own exact triangle/raw-ID support arrays and an `analysis`.
`proposals` reference those records by `kind` and `record_id`; support arrays
are not copied into every alias. `partitions` retain a complete original-triangle
to base-core map at each scale. Unreferenced input raw rows are retained as
`INPUT_RETAINED_NOT_TRIANGLE_SUPPORT`, never silently counted as plane support.
Successful extraction is `COMPLETE`, not accepted registration. Physical plane
and bend counts remain unknown; production use and physical flatness are false.

## Frozen initial geometric and scheduling policy

- Two prospective topology block sizes: **128 and 1,024 original triangles**.
- A complete one-ring halo surrounds each base block.
- At least three original raw identities per support; equal coordinates with
  distinct identities are not welded or reduced to one sample.
- Conditional normal-cone half-angle at most **pi/18 radians (10 degrees)**.
- Conservative in-plane covariance ratio lower bound at least **1e-8**.
- At most **16 scheduled closures**. Every constructed core is evaluated first;
  every individually qualified core is retained regardless of its closure.

These are exploratory algorithm parameters, not calibrated scanner limits. The
API permits supported exploratory scalar/size policies and stricter budgets, but a future
evaluated protocol must pin the exact values. Tiling and one-ring count cannot
be changed through this V5 policy.

The graph joins triangles only on complete shared ORIGINAL raw-row edges.
Coincident coordinates with different row IDs cannot create adjacency. Degenerate
or numerically unstable triangles and edge fans larger than two are refused;
there is no mesh repair or material-topology certification. Duplicate/overlapping
geometry is not geometrically unioned or interpreted as material area.

For each scale, lowest unassigned original triangle row starts a BFS block;
sorted original neighbor IDs break frontier ties. On reaching the fixed block
quota, remaining unassigned rows belong to later blocks. This is a prospective
topology partition, not a fit-driven crop. Its complete one-ring support is
subsequently fitted without residual removal. A disconnected component smaller
than the quota becomes a complete core. Holes and disconnected support remain
represented by the original triangles, not a hull or filled polygon.

Tiling is deterministic for fixed incidence/order and rigid transformations of
its coordinates; it is not invariant to arbitrary triangle-row permutations or
retriangulation. Those are explicit robustness tests, not false invariance claims.
A connected small cap can fall across tile boundaries. This finite policy cannot
guarantee every physical patch has a valid core; missed support remains unknown.

The closure scheduler alternates tile-size families. Within the selected family
it prefers unoriented normal novelty relative to already scheduled cores, then
observed-centroid spread, then smaller conditional cone, then original core ID.
It uses no area, CAD, part identity, measurement answer or registration residual.
Centroids rank exploration in a SINGLE frame; they are not cross-frame anchors.
The schedule is constructed before any closure result. Qualified unscheduled
cores explicitly say `NOT_SCHEDULED_BY_POLICY` and remain CORE proposals.

## Whole-support witness and conditional cone

Fits use uniform original raw identities, not triangle area or unique XYZ.
Anchored/scaled covariance is computed before a symmetric eigensolve. The chosen
TLS normal is a proposal; no optimizer over all feasible orientations is claimed.

For that fixed unit normal and anchor `a`, every original support row contributes
the interval `[n dot (p_i-a)-epsilon_i, n dot (p_i-a)+epsilon_i]`. A common interval
provides an all-row offset witness. A midpoint of the computed common strip is
tested against every original error ball. Keep this strip offset distinct from
the observed centroid used in the cone proposal. RMS is diagnostic only.

Numerical projection guards do **not** increase epsilon. Robustly disjoint
computed intervals return `WITNESS_NOT_FOUND`: failure of this TLS orientation,
not an infeasibility proof. Touching/uncertain intervals return
`NUMERICALLY_UNRESOLVED_OFFSET_INTERVAL`. A strictly interior witness must have
minimum original error-ball margin greater than its separate projection guard.
The explicit algebraic exception is a represented exact coordinate plane with
an exact coordinate-axis normal and identical represented coordinate values;
this can witness an epsilon-zero support without inventing slack.

The conditional cone initially uses the conservative model in
[the working design note](2026-09-09_noisy_plane_registration_design_notes.md).
With `e_bar=mean(epsilon_i)` and
`B_squared=mean((epsilon_i+e_bar)^2)`, any simultaneous planar counterpart has
`n dot C dot n <= B_squared`. For a resolved smallest eigengap, its normal lies
within the declared covariance-to-cone relaxation. There is no independent-noise
assumption and no `1/sqrt(N)` confidence improvement.

The source separately records coordinate/centroid/projection arithmetic guards, covariance
guard, eigensystem residual and orthogonality residual. The cone uses a lower
eigengap and upper numerator including that separate covariance uncertainty.
The mean-epsilon and positive-sum B-squared calculations have separate
length-dependent rounding allowances; the epsilon array itself is unchanged.
The length-dependent gamma also appears in covariance accumulation and observed
centroid arithmetic; it is not limited to the uncertainty-bound mean.
An additional residual/separation sine bound accounts for the angle between the
COMPUTED normal and the minimum eigendirection. Specifically, projecting
`(C-lambda_0 I)n_computed` into the other eigenspaces gives an angle bound from
the residual upper bound and second-eigenvalue separation lower bound. This
angle is added to the covariance-to-cone angle by the spherical triangle
inequality. An unresolved separation is refused, not assumed to be an exact
eigenvector. Covariance arithmetic and eigensystem errors are distinguished in
the output; all these error-enclosure formulas require independent review.
An absent gap, ratio at least one, negative upper numerator or unresolved support
cannot become a favorable zero-angle bound. These allowances are disclosed
engineering error estimates, **not directed-rounding or metrology certificates**.
Very poor arithmetic resolution is a fatal numerical refusal. This numerical
design requires independent review; no statistical coverage is claimed.

Given fixed seed centroid `c`, normal `n` and cone half-angle `alpha`, a row's
necessary pointwise cone distance is

```text
v = p-c
lower = max(0, abs(n dot v)*cos(alpha)
               - norm(v-(n dot v)*n)*sin(alpha))
```

Compare it to `epsilon_row + mean(seed_epsilon)`. Arithmetically uncertain rows
are conservatively kept as POSSIBLE, with their boundary-sensitive count and
arithmetic allowance recorded. This is not simultaneous feasibility. All three
vertices must be possible before a triangle is eligible. The original-edge
eligible component containing the entire core is the complete proposed closure.
There is no iterative cone update or seed-dependent new topology.

Every scheduled closure receives one all-row refit at the original epsilon.
Only exact original triangle-ID equality deduplicates complete closures; a hash
narrows lookup but never replaces the equality check. All source core IDs and
their individual cone diagnostics remain retained. A rejected whole closure
leaves its independently qualified CORE proposal intact. Curvature within the
stipulated position balls remains non-identifiable; even a found witness is not
evidence of a physically planar flange.

## Hard budgets and failure semantics

| Quantity | Hard maximum |
| --- | ---: |
| Original raw rows / triangles / raw-edge records | 1,000,000 / 2,000,000 / 6,000,000 |
| Core records / triangle rows per halo core | 65,536 / 4,096 |
| Retained base/core/closure triangle memberships | 20,000,000 |
| Retained raw-ID memberships | 32,000,000 |
| Full-support fit row tests | 64,000,000 |
| Cone row tests / closure triangle tests | 16,000,000 / 32,000,000 |
| Closure original-edge pair tests | 96,000,000 |
| Fit evaluations / scheduled closures | 65,552 / 16 |
| Retained array bytes / metadata bytes | 536,870,912 / 67,108,864 |
| Row-processing chunk | 65,536 |

Every counter includes rejected work. Shape/index/cardinality checks precede
unbounded coercion; the core-record cap is checked before another BFS block is
constructed. Fits, cone batches, closure topology tests and retained memberships
are charged against their distinct budgets. Closure selection is a prospective
quota, not a computational overflow or an area-ranked prefix rescue.

All unexpected input, numeric, graph, fit, membership, resource or report-boundary
failures clear usable proposals AND core/closure/partition records; completion
flags become false. Attempted-work counts and discarded-record denominators remain
as negative evidence. No retained prefix becomes success or absence. On ordinary
execution, the outer boundary catches `Exception`, including unexpected graph or
fit `RuntimeError`/`IndexError`; it does not catch `BaseException` process-control
signals. On ordinary
completion, `scheduled_closure_evaluation_complete` and
`all_qualified_core_closures_evaluated` are separate. `search_exhaustive` is always
false even if all declared closures happened to be scheduled.

The first independently reviewed **source/helper** gate is 60 seconds wall,
45 seconds worker CPU, 1 GiB process-tree RSS and 1 MiB stdout, one serial job.
No execution is authorized by this document itself. A later capacity-only gate
would require its own exact fixture/harness audit before 120/90 seconds and
4 GiB; the host's available RAM never relaxes those limits.

## Prospective independent control groups

1. **Input/custody:** malformed/scalar/array bounds, dataclass bypass mutation,
   row and unsigned-index boundaries, immutable detachment, unused raw rows,
   degenerate triangles, nonmanifold fans and no application/native import.
2. **Fixed-normal witness:** heteroscedastic feasible/empty/touching intervals;
   original epsilon unchanged; `(+-1,0,0),(0,+-1,0),(1,0,1)` with errors
   `0,0,0,0,1` has a feasible z=0 plane although its TLS proposal can fail.
   Require WITNESS_NOT_FOUND, not infeasibility or false qualification.
3. **Conditional cone:** independent covariance/eigengap arithmetic, absent gap,
   negative upper numerator, broad cone, large translated coordinates and
   separate numerical guards; no statistical confidence interpretation.
4. **Pointwise is not joint:** corners `(+-10,+-10,0)` with error 1 and additional
   `(0,0,+-1)` with error 0 can each satisfy the necessary cone relaxation but
   cannot share a plane consistent with all rows. Refuse the complete closure
   while retaining the prospectively validated core; no residual trimming.
   For this private math counterexample explicitly allow a 0.25-radian cone:
   the conservative corner-only model gives `asin(0.2)` (about 0.20136 radians),
   which exceeds the initial default 10-degree admission policy. This fixed
   analytic control is not a change to a subsequently frozen noisy matrix.
5. **Topology/coverage:** both complete original-triangle partition denominators,
   one-ring halos, holes, disconnected tabs, equal XYZ/different IDs, tiny
   disconnected and tile-boundary caps, and complete curved-chain rejection.
6. **Scheduling:** no more than 16 closures; every core analyzed first, both
   families represented where eligible, unscheduled cores retained, deterministic
   source ordering, exact closure support aliases and all origin IDs retained.
7. **Frames/noise:** independent proper rotations/translations/unit rescaling
   away from numerical boundaries; row-order/retriangulation robustness without
   claiming identical tiles. Zero, bounded alternating and correlated
   heterogeneous constructed perturbations; shallow bow remains unclassified.
8. **Failure ledger:** early/late tight caps, numeric failure after good cores,
   metadata overflow, no qualified-prefix output, no accidental physical count,
   no false all-closures completion and all counter limits respected.

Private `_analyze_support(points, epsilon, gate)` is an exact-source mathematical
test seam on already validated finite arrays, not a public provenance claim.
The 108-case noisy geometry matrix in the working note remains prospective;
expected recoveries/refusals and exact noise/epsilon values must be frozen before
it runs. No success is reported merely because this source/protocol exists.
Its predeclared disposition categories must separate required core recovery,
required geometric refusal, bounded-search omission and numerical-resolution
refusal. In particular, represented rotated coordinates with epsilon exactly
zero can be numerically unresolved; they must not be silently rerouted through
the positive-epsilon class or counted as successful recovery. A required recovery
that abstains still fails its case. This document does not authorize that matrix.

## Research grounding and implementation boundary

[CGAL mesh region growing](https://doc.cgal.org/latest/Shape_detection/index.html)
provides relevant original-edge neighborhoods and whole-region furthest-vertex
checks, but its ownership/refit/seed-order policy is not copied as a completeness
or scanner certificate. [Polylidar](https://arxiv.org/abs/2002.01886) motivates
preserving finite mesh holes rather than filling a hull; this source preserves
the triangle union directly and does not implement polygon extraction.

The retained research `_fit_plane` and fixed `_cone_membership` are mathematical
references only. Their 256-row exhaustive searches, NxN arrays, XYZ deduplication
and loss of valid cores after failed closure are not imported into this path.
No GraffMatch anchor metric or cross-frame partial-centroid correspondence is
implemented. The next separately owned seam is bounded plane correspondence,
only after independent extractor review and constructed controls.
