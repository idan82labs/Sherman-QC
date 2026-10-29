# Connected CAD-guided measurements: first output gain and retained failure

Date: 2026-09-08. Implements the user's CAD+scan offline-first direction.
Source and independently constructed positions only; no physical/held-out/email,
control, application/service, database, hardware or release access.

## Decision and actual output change

Retain an explicit **offline single-inner-face circular-bend measurement path**.
The new measured-flange-tangent variant passes the bounded constructed stress
rule after the initial independent-circle variant fails eight cases. This is
positive measurement-output progress, NOT Gold count/F1, physical metrology,
automatic CAD feature extraction, registration verification or floor readiness.

The connected source test sends raw scan positions and a declared CAD feature
through `inspect_cad_guided_offline` → measurement kernel → actual
`BendInspectionReport.to_dict()`. For nominal R=5 mm and constructed actual R=6 mm,
both new fitting variants report +1 mm deviation, with radius error <1e-8 mm.
The exact incumbent dense-refinement `DetectedBend` construction assignment
copies `cad_bend.target_radius` and reports zero radius deviation (1 mm error).
That control executes ONLY the actual construction assignment, not the incumbent
detector/alignment/full pipeline. The current default product path is unchanged;
the legacy copied-radius issue is bypassed here, not repaired in every caller.

Sources:
[measurement kernel](../../backend/cad_guided_measurement.py),
[dedicated pipeline entry](../../backend/bend_inspection_pipeline.py),
[report contract](../../backend/feature_detection/bend_detector.py),
[connected checks](../../tests/test_cad_guided_measurement_report.py).

The new report exports measured signed sweep, distinct opening angle, radius,
signed nominal deviations, fitted center-axis point/direction, sampled axial
interval and gaps, residuals and conditioning diagnostics. It does not fabricate
physical trim endpoints, a bend count, calibrated uncertainty or conformance.
Legacy summary/progress/table/rank/merge consumers explicitly reject this payload;
they cannot prefer a nominal-looking PASS or interpret unknown as completed.
Default report behavior remains unchanged. No UI or default worker activation.

## Mathematical model and independent review

Inputs declare millimetres, inner-face acquisition, a finite CAD feature, proper
orthonormal feature frame, nominal signed radial sweep, finite span and two
flange search windows. `DECLARED_ALIGNED` is a declaration, not verified pose.
Only scan positions enter fitting; simulator labels/normals/true radii stay in
scoring. CAD controls search windows and orientation, never the fitted radius.

1. Fit both disjoint finite flange windows by TLS using all selected points.
   Reject insufficient footprint, nonplanarity, orientation mismatch and weak
   footprint relative to the declared residual policy. Orient normals toward
   the declared same-side CAD normals, without using absolute dot products.
2. Derive axis `a` from the two fitted normals, orient only its sign toward CAD,
   and measure `atan2(a·(n0×n1), n0·n1)`. Opening is `180−abs(sweep)`, a different
   convention. Reject poorly supported/model-excluded directions and sweeps.
3. Select curved positions in the finite CAD region, excluding support near
   either measured flange plane. Project along the MEASURED axis. Keep center
   and radius free in the geometric objective `sum((||q_i−c||−R)^2)`;
   the algebraic solution is only an initializer, not the final fit.
4. Test competing flat support using the minimum enclosing strip, not a TLS
   line's maximum residual. For a convex polygon, minimum width occurs normal
   to a hull edge. Enumerate all hull-edge orientations, bounded at 2,048 hull
   vertices (~4.2 million vertex projections); overflow/Qhull failure abstains.
   No joggling, outlier trimming, successful-prefix rescue or error-limit increase.
5. Check the actual geometric Jacobian, not its squared normal matrix:
   `J_i=[−(q_i−c)_x/rho_i, −(q_i−c)_y/rho_i, −1]`. Report conditioning and normalized
   smallest singular value, not calibrated confidence. Every selected residual
   must pass its predeclared limit; mixtures and insufficient arcs abstain.

Geometric versus algebraic fitting is grounded in
[NIST's fitting analysis](https://www.nist.gov/publications/recovering-circles-and-spheres-point-data).
[Hopp's NIST sensitivity study](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir5501.pdf)
explains short-arc error amplification under stated point-error assumptions;
those assumptions are not silently transferred to a physical scanner.

Independent pretest review found and repaired a real flat-check error. For three
angular columns spanning 21° on R=1 mm, sagitta `h=1−cos(10.5°)` is about .016745 mm.
The best free-offset chord plane has maximum error `h/2≈.008373 mm`, while a TLS
line's maximum residual is `2h/3≈.011163 mm`. At a .01 mm allowance the original
TLS guard would miss the competing flat. The retained regression now abstains.
Other pretest repairs handle weak flange footprint, int64 absolute-value overflow,
non-string registration, raw-array admission and tiny projected hull edges.
Postreview added a distinct-axial-station guard for subnormal span-policy underflow.

### New hypothesis after the retained stress failure

Independent free-center circle fitting exceeded the radius-error rule in eight
sparse/noisy constructed cases below. The next hypothesis uses the already
declared physical model: the circle is tangent to BOTH scan-fitted inner planes.
It does not add a CAD-radius penalty or change the residual/error limits.

For planes `n_i·x=d_i`, 2-D projection basis `U` and origin `o`, define rows
`A_i=n_i^T U`, `b_i=d_i−n_i·o`. Inner-face tangency requires:

```text
A c = b + R 1
c(R) = h + R g,   h = A^-1 b,   g = A^-1 1
f_i(R) = ||q_i − h − R g|| − R
f_i'(R) = −unit(q_i − h − R g)·g − 1
```

The one-dimensional solve starts from the measured free-circle radius. Measured
contact-sector inequalities give linear radius bounds before optimization; they
exclude the wrong-sector second-circle branch. Full residual and post-fit sector
checks remain. The output retains free-circle radius/tangency/conditioning as
diagnostics and labels new tangency residuals **enforced constraints, not
independent validation**. Default `fit_model="free_circle"` remains the retained
baseline; use explicit `fit_model="measured_flange_tangent"` for the new candidate.

Independent Jacobian/closed-form checks cover both signs. At ideal contact angle
`phi`, half-turn `alpha=abs(sweep)/2`, derivative is
`cos(phi−alpha)/cos(alpha)−1`: zero at tangencies, still weak for small turns.
The RMS derivative cutoff is an engineering guard, not an equivalent condition
number or calibrated radius bound. Fitted-plane/axis error is coherent across
arc samples: `delta c=A^-1(delta b−delta A c)+g delta R`. More arc samples do not
eliminate it. Uncertainty remains UNCALIBRATED and conformance INDETERMINATE.

## Paired constructed evidence — no count/F1 claim

[Fixed geometry matrix](../../tests/test_cad_guided_measurement_geometry.py):
324 cases = radii 1/5/30 mm × six signed sweeps ±45/90/135° × radius factors
.9/1/1.1 × sweep-magnitude changes −3/0/+3° × two rigid poses. Truth is independently
parameterized; expected flat abstentions use an independent finite-polygon width
oracle, not the runtime hull implementation. All 324 dispositions match:
288 conditional measurements and 36 flat-compatible abstentions.

On the SAME 288 predeclared eligible pairs, nominal-radius-copy RMSE is
1.4344956528 mm; free-circle fitted RMSE is 7.56484e-14 mm. Maximum radius error
3.76588e-13 mm, signed-sweep error 9.23706e-14°, axis-line error 4.73586e-13 mm.
These near-machine-precision EXACT fixture results are not scanner precision.
Separate 24 deterministic bounded-noise cases meet their original .005R/0.1°
construction error checks. A 200,000-row repeated-support admission test passes;
200,001 rows reject the whole input. Replication is workload, not independent
geometry or real scanner-scale qualification.

[Stress matrix](../../tests/test_cad_guided_measurement_stress.py): 8,192 cases,
seed 20260908, 8,784,305 raw rows per execution; continuous radius .7–40 mm,
signed nominal turns 60–150°, actual radius ±10%, turn deviation ±5°, local tilt
±2°, translations, four sampling densities, shuffled/duplicated rows and four
bounded-noise levels. These are constructed perturbations, not a calibrated rig.
Frozen advance rule: >=75% conditional measurement yield across ALL cases,
every emitted radius error <=1% nominal R and sweep error <=.25°, plus >=90%
paired normalized-radius-RMSE reduction versus nominal copying. Low-density
finite-window loss is allowed only as an explicit abstention in the denominator.

| Same 8,192 cases | Retained free-circle baseline | Measured-flange-tangent candidate |
| --- | ---: | ---: |
| Conditional measurements | 8,191 | 8,191 |
| Insufficient-support abstentions | 1 | 1 |
| Cases exceeding frozen error bounds | **8 — FAIL** | **0 — PASS** |
| Maximum radius error / nominal R | 1.8601375% | .8489738% |
| Radius RMSE / nominal R | .0911839% | .0476955% |
| Maximum sweep error | .065499° | .065499° |
| Paired nominal-copy RMSE / nominal R | 5.745556% | 5.745556% |

The first free-circle failure is retained, not xfailed or rescored. Failed case
indices: 1948, 2492, 2540, 3084, 3212, 3452, 4140, 4476. Their signed relative
radius errors are respectively .0123854367, .0107473526, −.0147842633,
.0100355292, −.0136092687, .0105562751, −.0107427612, .0186013753.
All are in the sparsest/highest-noise stratum. **Sign and density are coupled in
this generator**, so this is not a causal isolation of density/noise or universal
sign robustness; the earlier signed matrix and independent tests are separate.
Each density/noise stratum has 512 cases; the sole abstention is density0/noise
.0005. The candidate is a DEVELOPMENT replay on the SAME now-consumed synthetic
cases, not independent confirmation. No additional independent cases are claimed
for the final baseline-preservation replay, which reproduces all eight failures
and the same numerical aggregates on the final kernel revision.

## Executed checks and resource receipt

[Independent tangent checks](../../tests/test_cad_guided_tangent_math.py) pass 85
tests: independent finite differences, signed contact derivatives, duplicate
sample sensitivity, 36 raw-position cases with wrong CAD nominal radius and
shifted axes, and incompatible/missing/flat support negatives.

Root executed one bounded worker at a time using
[fixed-path runner](../../scripts/check_cad_guided_measurement.py), installed
NumPy/SciPy, OMP/BLAS one thread, isolated HOME/cache, no site startup/plugin
autoload/project pytest configuration/conftest. Limits: 300 s wall, 240 s worker
CPU, 2 GiB sampled process-group RSS and 1 MiB forwarded output. This is not an
OS sandbox. No packages were installed and no native CAD job ran.

| Execution | Result | Parent wall | Worker CPU | Worker peak RSS |
| --- | --- | ---: | ---: | ---: |
| Initial connected report | 24 passed | .877202 s | .824349 s | 110,608,384 bytes |
| Initial geometry | 51 passed | .854621 s | .811947 s | 169,377,792 bytes |
| First free-circle stress | **1 failed; eight retained violations** | 8.553664 s | 8.513107 s | 88,178,688 bytes |
| Tangent math/geometry | 85 passed | .364932 s | .332865 s | 85,852,160 bytes |
| Tangent stress | 1 passed; same 8,192 cases | 11.436621 s | 11.398263 s | 88,981,504 bytes |
| Final geometry incl. new axial guard | 52 passed | .841988 s | .811818 s | 169,558,016 bytes |
| Final report incl. both variants | 25 passed | .907362 s | .872490 s | 110,739,456 bytes |
| Final free-baseline preservation | **1 failed; same eight violations** | 8.784629 s | 8.748012 s | 89,866,240 bytes |

No budget stop occurred. Final geometry sampled-group RSS was 165,363,712 bytes;
its larger worker high-water value above is retained. Other final rows' sampled
group maxima matched worker high-water values. The candidate's 11.44 s is the
entire constructed batch, not per-part full-pipeline latency. Final passing test
set has **163 tests** (25+52+85+1); the separately retained free-circle stress
test remains failing. This is NOT an all-suites-green claim. Reruns followed
source/coverage changes or baseline-preservation checks, not additional evidence.

Exact command pattern from Git root:

```sh
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite report
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite geometry
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite tangent_math
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite stress_tangent
# Intentionally retains the failed baseline; exit 1, not a production test pass:
.venv/bin/python -I -S scripts/check_cad_guided_measurement.py --suite stress_free
```

## Final source identity and next engine step

```text
backend/cad_guided_measurement.py
84520ddd98052f954d9004b8cdb032c16a0cba87a57ff80219b8df5845f675dc
backend/bend_inspection_pipeline.py
d2cee19311d0450258876fc20002884e1975417634ba3da4986b12eeb6072e35
backend/feature_detection/bend_detector.py
a94d7865517d020562443f15420d707162a3249461d901eeba5b01970110e68a
tests/test_cad_guided_measurement_report.py
6f0b1abad3b4f84db738158607ae354305dbb5a783a5ea044c6701535602ded1
tests/test_cad_guided_measurement_geometry.py
5997ecadadef5e673b4ddbd76ecde1617eda4ddf2d4e5462ec80b5154c71d171
tests/test_cad_guided_measurement_stress.py
efb68948bf67a12a8dbc2891e26167f02c5d639fb5cd49c1993bf4baa39dd96a
tests/test_cad_guided_tangent_math.py
f11e08f4171a7dfa3c34ab72ef9267dc70b7332f566b08fe8f776af4e6e159c6
scripts/check_cad_guided_measurement.py
89a8f1f6da2127c6033c01d3bf740c2aeb896495e6af86fa2ce060e7d149c3fc
```

Next: replace declared CAD-feature/frame inputs with a bounded native feature
adapter and validated registration, retaining this actual measurement/report
path. Quantify coupled flange/axis/registration uncertainty before tolerance
decisions; support characteristic-level partial results rather than discarding
an otherwise measurable angle when radius is unavailable. Single-side identity,
physical trims, legitimate shared-flange adjacent bends and noncircular features
remain outside this slice. No stage may fill these gaps with CAD nominal values.

M3 has a connected constructed measurement gain; native automatic identity,
registration and physical/generalization exits remain open. Count/F1 targets
have not been remeasured. Standalone zero-shot and floor preparation remain
deferred under the amended roadmap. No physical evaluation, default activation
or production promotion follows this result. Existing C15 failures, terminal
EXP10/EXP35, protected-data gates, budgets and the paused goal remain unchanged.
