# V3 finite-triangle search — prospective pure integration gate

Date2026-09-09. DRAFT until final exact-source independent pre-run audit. No
physical/native execution. Do not change the frozen V2 primitive or its terminal
Gold11 run; this is a separately versioned numerical mechanism.

## Decision and source scope

Nine physical V2 challenger cases failed soup-wide arithmetic admission before
their first finite query. The new distance helper has passed87exact-rational
checks under its separate frozen protocol. This gate asks whether the actual
whole finite-triangle search can preserve all primitives, correctly find finite
closest geometry and retain conservative ties while using local distance bounds.
It does not establish physical pose/measurement/count/F1 or CAD support recovery.

Named candidate: `backend/cad_guided_triangle_surface_v3.py`.
Frozen helper: `backend/cad_guided_triangle_distance_bounds_v3.py`.
Independent tests: `tests/test_cad_guided_triangle_surface_v3.py`, importing the
exact KKT oracle definitions from frozen
`tests/test_cad_guided_triangle_distance_bounds_v3.py`.
Bind all four files, this protocol and the exact new named-suite harness revision
before execution. No V2 algorithm replay, backend startup, native imports,
physical scans/CAD/Gold, email/held-outs or unrelated artifact reads.

## Mathematical and numerical contract

Keep immutable original triangle/face identities, all malformed-input checks,
minimumedge1e-100 and shapequality1e-8. No triangle removal, welding, hole filling
or input healing. The proposal minimizer uses triangle-local normalized edges;
the separate certificate uses its own power-of-two chart with outward intervals.
Do not claim the nominal max-edge proposal normalization is exact power-of-two
arithmetic. Its proposed interior point and three finite-edge candidates must
have exactly feasible represented barycentric coefficients. Failed candidate
feasibility cannot become a silent plane extension or deleted primitive.

Each feasible candidate supplies a sound conditional lower/upper distance bound.
For a single triangle, maximize available lower bounds and minimize feasible
upper bounds. First pass maintains minimum upperU* across triangles; conservative
outward AABB L-infinity bounds may exclude nodes/primitives only above that U*.
Second pass fixes U*, includes every primitive with ownlower<=up(U*+tieband),
and retains complete all-pair point/normal/face evidence within explicit budgets.
The full-surface lower bound is not the selected triangle's lower bound alone.
Rebuild its lower envelope in the complete fixed-upper second pass; obsolete
weak first-pass certificates from subsequently AABB-excluded primitives must not
widen that final envelope. Every true minimizing triangle survives sound AABB
pruning since its lower bound cannot exceed the true minimum, which is <=U*.

For potentially competitive triangles only, require outward intervalwidth<=
the declared1e-5 multiple of THAT triangle's minimum edge. This is a changed
per-primitive certificate-width admission metric, not a relaxation of a started
experiment or a universal condition-number theorem. Valid but wide intervals
whose lower bounds prove irrelevance need not fail the query. Unavailable
arithmetic without a valid excluding bound remains fatal; no uncertain competitor
may be silently skipped. No partial successful prefix escapes any fatal failure.

The selected candidate q need not equal the exact triangle projection x*. Convex
projection gives ||q-x*||²<=||p-q||²-dist(p,T)². Outward upper/lower squared
bounds plus separate world-point materialization error bound the reported
closest-point error. The displayed nominal distance, local interval, selected
triangle interval and returned-point error remain separate fields.

All numerical bounds inherit the helper's conditional IEEE binary64 model;
they do not certify analytic/native CAD, calibration or physical metrology.
Normals/region labels and their uncertainty flags remain conditioned engineering
evidence, not interval-certified geometry. No registration caller is updated by
this pure gate; frozen callers ignore numerical_distance_guard and recompute
distances from world points, so they cannot inherit certified-threshold claims.

## Fixed independent matrix and acceptance

The final test revision is pinned before execution. Current fixed matrix:
52tests,543exact KKT projection invocations, each at most seven4x4 exact solves.

- 12single-triangle cases:two independently specified shapes ×three dyadic scales
 ×two world origins, seven finite-region points each. Exact represented geometry
 is truth; both frames retain known FACE/EDGE/VERTEX sequence.
- 6mixed-scale soups:17triangles with edge scales2^-16..2^16 at remote2^22origins,
 three global unit scales and both ordering directions, four local queries each.
 These require local support to survive unrelated soup extrema.
- Hole, separate point/normal/identity ties, final-nearest tie band, two exact
 inclusive tie-distance boundaries, finite-plane extension versus real face,
 all-pair normal conflict in six orders, two unequal-scale equidistant searches,
 and a long-triangle distant-centroid sentinel.
- A scheduling control weakens only a remote triangle's valid helper lower bound
 to zero, leaving its feasible upper witness unchanged. With one-pair batches it
 is visited first, then AABB-excluded after a nearer witness; the final enclosure
 must remain locally accurate. This source-review regression predates execution.
- Three late invalid primitives; six exhausted search/work budgets, including
 the new certificate-pair channel; exact two-pass7-versus8certificate boundary
 at28workunits; late tie overflow, immutable array-base/source-identity snapshots,
 and empty-query work.

For every exact-oracle output: index integer type/shape/range is checked BEFORE
Python indexing; finite-face ID agrees; lower²<=exact optimum²<=upper²; nominal
distance error and width<=2^-32*declared local scale. Reported point-error radius
must contain error to the selected triangle's exact optimizer. A refusal is not
a passing required positive. Exact ties must not disappear under ordering.

The long-centroid sentinel has its own2^20-unit triangle scale, unchanged exact
containment/selected-ID and additional strict nominalerror<=2^-32 around1/4 and
width<=2^-20. This prospective correction followed independent SOURCE review:
the support-function enclosure can depend on the selected triangle's own long
edges, unlike a nominal-distance rounding error. It was made before any execution,
not to rescue a result. All other local-scale accuracy checks stay unchanged.
Independent pre-freeze review also closed negative-index aliasing and missing
region assertions and added exact tie-boundary controls, without observed results.

## Budgets, retention and advancement

One serial pure worker:60seconds parent wall,45seconds workerCPU,1GiB sampled
groupRSS,1MiB output. Disable plugin autoload/conftest/cache, threadlimits1 and
use private HOME/TMPDIR. Six exact source/test/protocol/harness hashes before and
after; retain fresh exclusive STARTED/OUTPUT/COMPLETED for every outcome.
Sampled RSS is not an OS memory sandbox; CPU is the worker only.

Primitive hard maxima retain500k triangles,200k queryrows,32m trianglepairs and
512m work units. The new certificate-pair cap is128m (four per trianglepair);
charge certificate work in addition to node/triangle/normal work. Exact final
work counters/weights and failure dispositions are independently source-audited
before release. Test-local reduced budgets remain fixed. No maximum-size native
or physical campaign is implied by these small geometry fixtures.

Only after exact-source independent pre-run GREEN:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite triangle_surface_v3
```

All52tests must pass with no unexplained skips, resource stop or source drift,
then independent post-run audit. This advances the primitive to separately
versioned registration/prefilter integration and the fixed connected workflow.
A new physical comparison still requires its own candidate/configuration/input/
reference/scorer/budget freeze and independent audit, retaining all11cases/71lines.
No product activation or goal completion follows from this pure matrix.
