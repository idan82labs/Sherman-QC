# V3 local triangle-distance enclosure — first pure gate

Date: 2026-09-09. Source-only successor to the terminal physical V2 arithmetic
admission failure; not a physical retry or new benchmark. Freeze exact source,
test, protocol and harness hashes in a separate pre-run audit before execution.
Once executed, keep this revision and its result terminal, including any failure.

## Hypothesis and scope

Soup-wide absolute-coordinate/maximum-edge/minimum-quality/minimum-edge coupling
rejected nine physical challenger cases before their first finite query. A
per-pair local feasible-point distance enclosure can avoid that coupling without
dropping triangles or declaring arbitrary numerical guards certified. This first
gate verifies the enclosure helper, not a completed surface search, registration,
observed count, physical measurement or F1 improvement. Frozen V2 is unchanged.

Candidate: `backend/cad_guided_triangle_distance_bounds_v3.py`.
Independent test: `tests/test_cad_guided_triangle_distance_bounds_v3.py`.
Harness: `scripts/check_cad_guided_measurement.py`, exact named suite
`triangle_distance_bounds_v3`; no blanket pytest, native imports, CAD, scans,
Gold, held-outs, email or other experiment reads. The prior exact used harness
4f2a7781 is archived under `source_snapshots/guard_smoke_v2_4f2a7781/`.

The scalar mathematical baseline is exact finite convex-hull distance, calculated
by an independently written rational KKT active-set oracle from the represented
binary64 inputs. Seven nonempty simplex subsets and global convex-projection
inequalities avoid duplicating the candidate's local-chart implementation.
Poor feasible candidates must still yield sound (possibly wide) distance bounds.
The numerical constants below are prospective engineering accuracy requirements,
not claimed universal error theorems or scanner specifications.

## Mathematical contract

For exact represented coefficients a,b>=0 with exact a+b<=1, define the feasible
mathematical candidate q=v0+a(v1-v0)+b(v2-v0), even when its materialized world
float differs. Let r=p-q and h=max_i r dot(v_i-q). For every x in the triangle
convex hull, r dot(p-x)>=||r||^2-h; Cauchy bounds this by ||r|| ||p-x||.
Consequently a positive lower enclosure of ||r||^2-h divided by an upper
enclosure of ||r|| yields a valid distance lower bound. The residual norm upper
bound is a valid upper bound because q is feasible. A nonpositive/unresolved
numerator yields lower0; no division through an unresolved zero norm occurs.
This proof applies to degenerate convex hulls too; the future surface caller
must independently retain its existing triangle edge/shape admission.

Each pair is anchored at v0 and uses its own positive power-of-two scale.
Subtract before scaling; outward intervals retain subtraction, scaling and every
certificate operation. Nominal differences choose the chart only. TwoSum checks
exact feasibility, including the case where an infeasible sum rounds to1.
The represented-coordinate model assumes binary64 round-to-nearest elementary
operations, correctly rounded sqrt, gradual underflow and no reassociation or
fast-math. Outward nextafter includes rounding/underflow under that model.
Small runtime sentinels are necessary checks, not full runtime attestation.
Overflow/nonfinite arithmetic, bad inputs and exhausted pair budgets clear all
result rows. Original nonzero subnormal inputs are outside this helper's declared
input regime. Intermediate gradual underflow is enclosed, not treated as exact.

Distance bounds concern the finite represented triangle, not analytic/native CAD.
`closest_points` is the supplied candidate materialized in world coordinates,
not a proved optimizer. Its Euclidean restoration error is bounded separately.
The original local-distance and world-point accuracies must not be conflated.
Normal/region labels, BVH pruning, full-surface ties and threshold decisions are
not implemented by this helper. `maximum_distance_width` is descriptive rounded
telemetry; later certified-width admission requires an outward subtraction.

Primary references: [Eberly's finite-simplex minimization](https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)
and [Shewchuk's exact-predicate/error-free arithmetic work](https://www.cs.cmu.edu/~quake/robust.html).
The support/Cauchy lower-bound derivation above is separately reviewed here;
neither reference is presented as certifying this implementation or metrology.

## Fixed acceptance and resource rules

The exact test revision bound by pre-run audit determines all enumerated cases;
do not add/remove/retune cases after seeing its result. Exactly87prospective
tests:44positive/empty,42input/budgetrefusals and1injectedlatearithmeticrefusal.
There are122exact-oracle invocations, below the2,048 ceiling, with each oracle
solving at most seven4x4 rational KKT systems. Fixed
positives include interior, edge, vertex, oblique/obtuse/thin represented geometry,
power-of-two scales, translated frames, poor feasible candidates, degenerate
hulls and independent mixed-scale batches. Negative controls include malformed
types/shapes, nonfinite/out-of-range/original-subnormal inputs, exact-infeasible
coefficients, budget boundaries and late full-result clearing.

For every admitted oracle case: lower<=exact optimum<=upper, checked using exact
rational squared inequalities (not a rounded square-root truth value); candidate
distance bounds enclose the exact distance to the exact feasible q; the separately
reported restoration radius contains materialized-q error. The24fixed optimizer
scenarios (two shapes ×three power-of-two scales ×four frames; three query rows
each) require exact-rational interval width<=2^-36*local_scale and nominal
candidate-distance error<=2^-40*local_scale. These constants were pinned by the
independent author before execution and replace the earlier unfrozen generic
512epsilon planning sketch; they distinguish enclosure width from midpoint error.
The separate world-rounding case requires nominal distance error<=2^-38 local
units. These are fixed-case engineering requirements, not universal bounds.
Seven additional positive branches cover exactzero vertex/interior/point-hull,
subnormal INTERMEDIATE differences from normal original coordinates, the allowed
1e100 input boundary, obtuse and moderately thin optimal candidates. Every one
has exact oracle containment and optimizer equality, without extending the24-case
tightness claim. Poor candidates need not have tight lower bounds. Translation/
unrelated pairs must not inflate local error limits. No failed required positive
becomes a passing abstention. All selected tests/failures/skips stay in the result.

One serial pure worker:60seconds parent wall,45seconds worker CPU,1GiB sampled
process-group RSS,1MiB output. Helper hard pair maximum65,536; smaller test-local
budgets remain fixed. No native job or full64k precision sweep is required by this
gate. Isolated imports, disabled plugin autoload and conftest, thread limits1,
private HOME/TMPDIR and retained STARTED/OUTPUT/COMPLETED with exact before/after
source hashes. Sampled RSS is not an OS memory sandbox; worker CPU excludes parent.

Command after independent pre-run GREEN only:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite triangle_distance_bounds_v3
```

Advance only if every required test passes, no unexplained skip, no budget stop
and no four-file source drift, followed by independent post-run review. Passing
advances this helper into the separately versioned closest-triangle/BVH and
connected-workflow implementation; it does not qualify a physical rerun alone.
Keep the all-eleven-case scorer/thresholds unchanged. The autonomous goal remains
active until integrated measured improvements and the full user criteria are met.
