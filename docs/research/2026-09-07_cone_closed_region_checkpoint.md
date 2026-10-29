# Noise-aware plane proposals under the locked roadmap

Date: 2026-09-07. Engine-only source/synthetic work. The user confirmed proceeding
and [locking the execution plan](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
This checkpoint addresses priority 1; it is not physical qualification, detector
integration, a protocol freeze or a resumed goal.

Follow-up: the [sparse-graph envelope checkpoint](2026-09-07_graph_envelope_checkpoint.md)
attributes the remaining flange cuts and compares explicit mutual/union graph
relaxations. This baseline and all receipts below are retained unchanged.

## Plan lock and bounded engineering deliverable

The existing roadmap now fixes the CAD-assisted/operator-reviewed destination,
engine-first order, M0–M6 acceptance checks, data/release gates and change control.
It permits implementation hypotheses to evolve against recorded evidence, while
material changes to direction or acceptance criteria require explicit approval.
M2 capture-reference preparation and M4 operational essentials may proceed in
parallel. Cosmetic UI work is deferred. No new control/issuance artifact was made.

The new engine hypothesis is **fixed uncertainty-cone proposal expansion followed
by a full-support plane fit at the original error bound**:

- [Cone-closed plane regions](../../research/geometry/cone_closed_plane_regions.py)
  and [direct math/behavior tests](../../tests/test_cone_closed_plane_regions.py).
- [Whole-component bend integration](../../research/geometry/cone_closed_bend_candidates.py)
  and [paired structural/noise tests](../../tests/test_cone_closed_bend_candidates.py).
- [Independent noise-location attribution](../../tests/test_noise_birth_attribution.py)
  on the retained adaptive baseline.

All previous position/adaptive/axis/radius sources remain unchanged. This is an
additive research policy, not replacement of the active detector or hidden tuning
of the recorded baseline. It still accepts at most 256 raw positions, not a full
scanner-scale cloud.

## What caused the earlier noise failures

The prior [adaptive checkpoint](2026-09-07_adaptive_mixed_radius_checkpoint.md)
exposed two different failure mechanisms. An independent, predeclared 48-case
ablation uses radius 2, angles `+/-30,90,150`, dense/sparse concentric mixtures,
and exact/flange-only/transition-only/all noise. Each geometry shares one seeded
noise array across its four ablations, with displacements 25–95% of the unchanged
`epsilon=1e-6`.

- Transition-only or all-point noise causes eight raw-seed cap failures, all at
  `+/-30` degrees. Slight perturbations change observed ties and distinct local
  supports. Overflow records only the lower bound of 65 accepted raw seeds; no
  uncapped rerun was used to obtain a favorable result.
- Flange-only noise causes no cap failures in this ablation, but fragments
  fixed-plane extensions. Dense retained associations increase from 30 to 408;
  sparse from 16 to 369. The fixed seed planes exclude 306/4,700 dense and
  331/4,550 sparse constructed seed–flange memberships. These are stage
  diagnostics with explicit denominators, not detection accuracy.

The second mechanism has an exact counterexample. Let true flange samples lie on
`z=0`, observed seed samples on `z=+epsilon`, and another observed sample on
`z=-epsilon`. Every observation meets the simultaneous error bound, but its
distance to the seed's fixed fitted plane is `2 epsilon`, so the old extension
rejects it. Normal error adds a lever-arm effect for distant samples. Averaging
the seed does not shrink a deterministic correlated error bound by `sqrt(N)`.

## Derived proposal gate and its limits

Let `m` be the observed seed centroid, `n0` the fitted normal, and `alpha` the
existing conditional normal-cone bound. For a candidate observation, set

```text
v = p - m
a = abs(dot(n0, v))
b = norm(v - dot(n0, v) * n0)
g = max(0, a*cos(alpha) - b*sin(alpha))
```

`g` is the minimum absolute dot product over the unoriented normal cone. A point
whose counterpart shares the seed's true plane must satisfy `g <= 2 epsilon`:
the observation and the observed seed centroid can each be epsilon from that
plane. The implementation adds a separately disclosed arithmetic guard using
scaled coordinates. This direct derivation relies on the existing seed cone's
conditional simultaneous-error premise; it is not a statistical confidence bound
or a floating-point interval certificate.

Each candidate point can satisfy the necessary condition using a DIFFERENT plane.
The mask therefore does not establish a joint plane or physical membership. Take
seed samples at `z=0` and broad corresponding patches at `z=+/-2epsilon`: all can
pass the individual gate, while the complete union fails an epsilon plane fit.
A persisted test requires the original-budget full-support check to reject it.

Neighborhood plane fitting, seed sensitivity and connectivity-dependent region
growing are established tools, described in [CGAL Shape Detection](https://doc.cgal.org/latest/Shape_detection/index.html).
Noise, density and neighborhood scale also affect normal estimation, as studied
by [Mitra, Nguyen and Guibas](https://graphics.stanford.edu/~niloy/research/normal_est/normal_estimation_socg_03_ijcga_04.html).
These sources motivate explicit uncertainty and failure tests; neither establishes
the present method's segmentation accuracy or calibration. The cone formula above
is derived here, not presented as an external theorem about this implementation.

## Explicit new search and resource policy

1. Keep every declared complete-tie observed kth-neighbor query at every unique
   sample; retain the previous nominal radius/tie guards. Do not enlarge the
   observed graph or choose smaller k after an over-cap query.
2. For each valid all-point seed plane, form ONE fixed centroid/cone mask. No
   iterative normal/cone updates or residual-based point removal occur.
3. In the unchanged observed mutual-neighbor graph, take complete induced
   components intersecting any seed member. Retain subminimum/split counters.
4. Fit each complete proposed support at the ORIGINAL epsilon. Keep successful
   witnesses; rejected closures get no seed fallback or trimmed rescue. Failure
   can hide valid subsets and is not proof of plane infeasibility.
5. Deduplicate only identical final support tuples, retaining overlapping and
   physically ambiguous alternatives. Fit results for identical supports are
   reused without changing their error budget.
6. Enumerate disjoint final-plane pairs and whole remainder components exactly
   as in the prior association policy. Do not run another fixed-plane extension.
   Existing single/two-radius checks, conditional interval scopes and explicit
   numerical-unavailability handling remain in force.

The budget unit changes explicitly: **64 FINAL refitted supports**, not the
baseline's 64 raw seed regions. Up to 1,024 already-bounded seed queries may be
examined; a new explicit ceiling of 2,048 unique seed/component plane fits bounds
the work. This is not a same-budget V1 comparison. The old raw-seed cap and source
remain intact. Pair/association/split caps remain 2,016/256/255. Every exhaustion
clears the new candidate prefix and marks its finite search incomplete. The
larger host does not waive the 16 GiB process-tree ceiling or any frozen limit.

Full-support TLS is still only a plane proposal, not complete minimax feasibility
search. Nearby coplanar objects can share one sample witness; sampled connectivity
does not identify a physical object. Graph rank sensitivity, partial visibility,
ambiguous surfaces and uncalibrated epsilon remain explicit limitations.

## Decision boundary

The predeclared continuation check required fewer known incomplete searches and
less proposal proliferation WITHOUT losing baseline complete-support or
conditional complete-support cases, with all applicable radius bounds containing
constructed truth. **This bounded development comparison passes**, as detailed
below. Continue the mechanism as research; do not promote it to runtime.

Neither passed synthetic checks nor a plan lock authorize physical access or
promotion. Priority 1 remains open where sparse graph connectivity is unstable;
scanner-scale behavior, adjacent bends, CAD-assisted integration and physical
calibration remain ahead. Failed closure can still hide valid subsets on other
inputs; passing these constructions does not remove that algorithmic limitation.

## Root verification and paired results

Root ran the final new source and tests: **140 checks passed in 32.04 seconds**.
This includes 52 cone-region checks, 55 integration checks, four independent
attribution checks and 29 existing independent axis/radius adversarial checks.
The prior 652-test full geometry receipt remains historical; the entire unchanged
suite was not rerun or claimed as rerun. Independent source review found no
blocking issue with the cone/refit/cap contract or the plan-lock boundaries.

The paired integration tests run the new and unchanged adaptive baseline on the
EXACT same observed array in each of 150 exact and 150 bounded-noise cases. The
noise seed is `2026090717`; norms are `0.2500009243–0.9499925473` times the original
epsilon on 14,700 positions. Units, geometric policies and constructed supports
are identical; the explicitly different region-budget unit is disclosed above.
Test-only exact row identities establish whole flange and transition membership.
No construction labels enter either analyzer.

| Noisy stratum, 30 cases each | Baseline / new retained associations | Baseline / new complete-support conditional cases | Baseline / new incomplete searches |
| --- | ---: | ---: | ---: |
| Clean | 1,407 / 30 | 30 / 30 | 0 / 0 |
| Missing middle arc | 1,644 / 54 | 30 / 30 | 0 / 0 |
| Concentric mixture | 1,374 / 102 | 23 / 30 | 6 / 0 |
| Sparse flanges | 1,233 / 24 | 24 / 24 | 0 / 0 |
| Sparse plus mixture | 624 / 52 | 15 / 16 | 6 / 0 |
| Total | 6,282 / 262 | 122 / 130 | 12 / 0 |

Every individual baseline-matched case is retained, both for complete joint
support and for a conditional model on that support. Gains on one case cannot
compensate for a lost case in this assertion. New noisy regions never exceed
seven per case, and total unique seed/component fits never exceed 118 per case.
These are observed workload counts, not universal resource guarantees or a
measured full-runtime speedup.

All 84 noisy matched single-radius and 92 matched band intervals contain truth.
The new 92 band intervals ALL correspond to complete constructed radial layers
with matching joint support; the previous noisy receipt had 76 matched out of
2,502 proposed bands. This is not physical precision. The new receipt counts all
single-analysis intervals, including rejected pooled mixtures; the previous
single denominator counted only single-surface strata, so those totals must not
be compared as if their scopes were identical.

Unmatched associations remain: 24 in the missing-arc stratum, 72 in dense mixtures
and 36 in sparse mixtures. They are not silently scored as correct or necessarily
incorrect; 12 extra single-radius positives in the missing-arc stratum remain
alternative sampled explanations, not extra physical bends.

The exact matrix retains the same 30/30 clean, 30/30 missing-arc, 30/30 mixed,
24/30 sparse and 18/30 sparse-mixture conditional cases, with no incomplete
searches and no per-case loss relative to baseline. Its 84 matched single and 96
matched band intervals contain truth. Noisy sparse mixtures still give only
16/30, versus 18/30 exact; this observed-graph limitation is retained. Zero
incomplete search does not mean zero abstentions or complete bend detection.

Reproduce the executed checks from `FullProject/`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_cone_closed_plane_regions.py tests/test_cone_closed_bend_candidates.py tests/test_noise_birth_attribution.py tests/test_two_radius_adversarial.py tests/test_axis_radius_independent_bounds.py -q -s
```

The new integration helper functions are `run_cone_closed_structural_matrix()`
and `run_cone_closed_noise_sensitivity_matrix()`; both return per-case paired
receipts and stratified counts. The independent 48-case helper is
`run_noise_birth_attribution()`. None accepts a real-input path.

Final engine SHA-256 snapshot:

```text
cone_closed_plane_regions.py     bec10015fd8cffdcaf92fa62248f5d1d8a3ae2fa49413f82f4b2cee14b56b7d4
cone_closed_bend_candidates.py  f05ee0959eba21147c2e3e6388f6800c4e5399fd1250adf9cf76a2599564146a
```

All six checked predecessor source hashes (fixed/adaptive region and association,
plane-conditioned single/two-radius) still match their earlier dated checkpoints.

No raw scans, held-outs, reserved email attachments, manifests, sealed truth,
scientific outputs or control/issuance records were accessed. No service, native
geometry pipeline, deployment, UI work, dependency change or commit occurred.
C15's retained freeze failures, terminal EXP35 and the paused goal remain unchanged.
