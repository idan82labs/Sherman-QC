# Position-derived plane-pair direction hypotheses

Date: 2026-09-07. Source-only research engineering, following the
[axis-conditioned radius diagnostic](2026-09-07_axis_conditioned_radius_method.md)
and [analytical finite-bend fixture](2026-09-07_analytic_bend_testing_design.md).

The bounded artifact is
[`analyze_position_axis_hypothesis`](../../research/geometry/position_axis_hypothesis.py),
with [synthetic tests](../../tests/test_position_axis_hypothesis.py). It replaces
the *supplied exact direction* with an uncertain direction derived from positions,
conditional on an explicitly supplied pair of plane neighborhoods. It does **not**
discover those neighborhoods, connect them to a bend, or feed a production path.
This is not a new physical protocol, detector, calibration or promotion decision.

## Observation channel and conditional question

Inputs are `first_neighborhood`, `second_neighborhood`, `unit`,
`position_error_bound`, and `max_axis_angle_bound`. Both neighborhoods contain
positions only. No measured or oracle normal, face/component label, CAD value,
radius, expected count or exact axis enters the function. All dimensional inputs
share the declared `m` or `mm` unit; the angular policy is explicitly in radians.
The hard allocation cap is **256 total input rows across both neighborhoods**,
checked before array conversion and before exact-repeat removal. There is no
file access, native geometry dependency, trimming or batch/corpus reader.

The conditional model is that, for each neighborhood, every observed position
has a counterpart on one ideal plane within the stipulated simultaneous Euclidean
position-error bound. Each plane may have an independent location/orientation.
This is a model assumption, not established physical surface identity. The two
planes are not automatically flanges of the same part or tangent to one cylinder.
Normals inferred from their positions are not a second independent sensor.

`None` or an empty neighborhood is absent evidence and causes abstention. Fewer
than three distinct positions cannot constrain a plane. Exact repeated positions
within one neighborhood are removed without treating repeated rows as independent
evidence. They still count toward the input cap. No point is removed because of
its residual. Repeated measurements at slightly different positions remain.

The positive status is `BOUNDED_DIRECTION_HYPOTHESIS`; all other well-formed
geometric outcomes are `ABSTAINED`. In every result, `axis_is_verified` and
`regions_are_verified` remain false and production disposition is `NOT_AUTHORIZED`.
The angular policy requests conditional resolution, not a confidence level,
calibrated metrology tolerance or performance threshold learned from physical data.

## Plane witness from all positions

For one neighborhood, let the observed positions be rows of `P`, and let `X=H P`
be their centered matrix, with `H=I-11^T/N`. Centering/scaling precedes NumPy SVD:

```text
X = U diag(s1,s2,s3) V^T,       s1 >= s2 >= s3 >= 0.
```

The orthogonal-factor and ordered-singular-value conventions follow the primary
[LAPACK Users' Guide, Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html).

The candidate plane normal `n_hat` is the last right singular vector. A zero or
numerically rank-deficient `s2` leaves its normal unresolved; a good line fit is
not enough. This is a positional plane fit, not a normal supplied by the fixture.

The fitted orientation is held fixed while its offset is centered between the
extreme scalar projections. With an observed reference position `r`, write

```text
t_i = n_hat dot (p_i-r)
d   = (max(t_i)+min(t_i))/2
rho = max_i abs(t_i-d).
```

The returned plane uses `reference_point=r` and `offset_from_reference=d`, so its
equation is `n_hat dot (x-r)=d`. Every distance is evaluated; an RMS residual is
not used as the acceptance statistic. If `rho<=epsilon`, projecting each point
onto that plane explicitly constructs a sample-compatible plane under that
budget. The public field `compatible_with_error_bound` uses the disclosed
*effective* budget described below, not the stricter caller budget alone.

SVD orientation with a midpoint offset is not an unrestricted minimax-plane
solver. Failing this particular all-point witness therefore causes abstention;
it does not prove that no other orientation admits a plane. Conversely, a witness
does not establish that a real continuous surface between samples is planar.

## Deterministic normal cone, including centering

Perturbation of singular subspaces is an established numerical-analysis question:
[Wedin (1972), Perturbation bounds in connection with singular value decomposition](https://link.springer.com/article/10.1007/BF01932678)
studies the change of the invariant subspaces associated with `A^H A` and `A A^H`,
including rank-deficient applications. Here its publisher abstract/metadata were
checked; the full subscription article was not used to assert an exact theorem.
The argument below is a directly derived, special bounded-plane implication,
not an unverified invocation of Wedin's general perturbation theorem.

The following derivation is an exact-arithmetic consequence of the bounded-error
model, not a statistical approximation. Let the true position rows be `P+E`, with
`||e_i||<=epsilon`, and suppose they lie on a plane with unit normal `n`. Then

```text
H(P+E)n = 0
X n    = -H E n.
```

The centering matrix is an orthogonal projection, so its operator norm is one:

```text
||X n||_2 = ||H E n||_2 <= ||E n||_2 <= sqrt(N) epsilon.
```

This accounts for the movement of the sample centroid; it does not pretend the
centroid is a noiseless reference. No independence or zero-mean error assumption
is needed, and a correlated/common-mode perturbation remains covered.

Choose the sign of `n` so that its angle `theta` from `n_hat` lies in `[0,pi/2]`.
Its component perpendicular to `n_hat` has norm `sin(theta)` and lies in the span
of the first two right singular vectors. Consequently,

```text
||X n||_2^2 >= s2^2 sin(theta)^2
sin(theta) <= sqrt(N) epsilon / s2.
```

For `s2>0`, every compatible true plane normal is therefore in the *unoriented*
cone with half-angle

```text
alpha = asin(min(1, sqrt(N) epsilon / s2)).
```

The implementation applies this to the centered/scaled matrix with the equally
scaled effective position budget. It deliberately uses the simpler conservative
bound without subtracting `s3^2`. A small fitted residual alone cannot narrow
`alpha`: a thin strip, large error bound or short transverse footprint can leave
the normal wholly unresolved even when the candidate plane has zero residual.
If the assumed bounded-error plane does not exist, the implication is vacuous;
the code does not treat a cone alone as a plane-compatibility witness.

## Plane-pair direction and its uncertainty

For the two fitted normals, set

```text
c_hat = n_hat1 cross n_hat2
beta  = atan2(||c_hat||, abs(n_hat1 dot n_hat2)).
```

`beta` is the acute separation of unoriented normals, not the signed bend turn,
flange opening, or an inner/outer face convention. Parallel or antiparallel
normals have zero cross product and do not define the candidate direction.

If `beta<=alpha1+alpha2`, the normal cones do not exclude parallel true planes.
The result abstains. This check is only a necessary-cone comparison: overlapping
cones do not by themselves construct plane offsets/corrections that make a
parallel alternative feasible. A separate adversarial test supplies an actual
parallel alternative for its nearly parallel observed pair.

For an independently sign-aligned true normal inside cone `i`,

```text
||n_i-n_hati|| <= delta_i = 2 sin(alpha_i/2).
```

For `c=n1 cross n2`, telescoping the cross products gives

```text
c-c_hat = (n1-n_hat1) cross n2 + n_hat1 cross (n2-n_hat2)
||c-c_hat|| <= delta1+delta2 = B.
```

The unit lengths of `n2` and `n_hat1` are important: there is no omitted quadratic
term in this telescoping form. If `B>=||c_hat||`, the error ball reaches zero and
no finite informative direction bound is issued. If `B<||c_hat||`, that ball's
tangent cone gives

```text
angle(unoriented c, unoriented c_hat) <= asin(B/||c_hat||).
```

The result requires this finite bound to be at most the explicit angular policy,
requires both all-point plane witnesses, and requires all numerical/degeneracy
checks to succeed. Otherwise it abstains while retaining available diagnostics.
A displayed fitted direction may therefore exist in an abstained result; that
field is not evidence that its angular precision was established.

The method is intentionally conservative. It bounds every ideal plane pair
allowed by the per-point model, rather than finding the exact feasible-axis set.
It can abstain although a sharper method would resolve a direction. No caller
should tune the position budget downward or enlarge the angular policy merely
to obtain the positive status.

## Floating-point and representation limits

For each neighborhood, with `scale=max_abs(p_i-p_0)`, the disclosed allowance is

```text
numerical_allowance = 64 machine_epsilon max_abs(p_i)
                    + 64 machine_epsilon scale
effective_epsilon   = position_error_bound + numerical_allowance.
```

The rank guard compares `s2` with `64 machine_epsilon s1`. The code also discloses
`128 machine_epsilon` as a small dimensionless guard: it pads normal angles,
the cross-product error bound, final axis angle and parallel-cone comparison.
These are engineering guards, **not proven SVD forward-error bounds, interval
arithmetic, directed rounding or numerically certified enclosures**. The
exact-arithmetic derivation must not be advertised as numerical certification.
Unsupported numerical ranges and failed SVD fits abstain with finite JSON.

This distinction is supported by the primary
[LAPACK Users' Guide, SVD error bounds](https://www.netlib.org/lapack/lug/node97.html):
its discussion separates backward stability and near-orthogonality from singular
vector accuracy, which depends on singular-value separation. It also warns that
small singular values may not have high relative accuracy. Those numerical
results do not certify this implementation's chosen constants or physical budget.

A large coordinate translation can enlarge the numerical guard and force
abstention without changing physical geometry. Ordinary proper rigid transforms
preserve the unoriented direction and normal-cone geometry up to these disclosed
floating-point effects. Unit conversion scales positions, budgets, singular
values, reference points, offsets and residuals together; angles do not scale.

The largest-magnitude component is made positive only to choose a display sign.
That gauge can flip under rotation or ties. Correct comparisons use an unoriented
angle or projector `a a^T`, not a signed vector identity. Reversing the two
neighborhoods or their row order does not add a physical axis direction sign.

## What the direction cannot supply

The intersection line of two infinite tangent planes is not the center axis of
the corresponding finite-radius cylinder. It can be outside both finite flange
patches. Consequently this artifact returns **no cylinder-axis location, radius,
axis endpoints, continuous support or finite component identity**.

The analytic fixture's hidden-transition counterexample remains decisive:
identical visible planar patches can accompany radius 2 or radius 4 cylinders
with different center-axis locations. Their common direction does not resolve
those different radii, contact locations or axes. Another test translates one
plane neighborhood far away: it still yields a direction hypothesis, explicitly
not a claim that the neighborhoods belong to the same object or bend.

Subsequent same-day checkpoint: the separately scoped
[uncertainty-preserving radius bridge](2026-09-07_plane_conditioned_radius_bridge.md)
now implements a necessary outer relaxation and retains explicit unestablished
original joint compatibility. The following describes this axis-only snapshot.

There is **no radius bridge in this change**. Substituting the fitted direction
into the existing exact-axis-conditioned radius diagnostic would discard this
new axis uncertainty. A separate bridge would have to justify projection-error
propagation over its bounded sample footprint, reflect the enlarged budget in
radius semantics, and reconsider axial grouping/extent claims. It must never
set `axis_is_verified=True` from the same positions or fits. Region discovery,
curved-transition identity, physical error calibration, prospective evaluation
and production promotion remain separately unverified and gated.

## Executed source checkpoint

On 2026-09-07 the final isolated focused suite passed **71 tests in 0.16 s**,
using NumPy 2.4.6 and pytest 9.1.1 in the existing Python 3.11.15 environment. The
tests use independent Cartesian plane equations and retained construction truth,
not fitted normals as expected answers. They cover two hand-computed plane pairs,
parallel and nearly parallel pairs, exact/near-rank degeneracy, absent and repeated
positions, nonplanar/outlier cases with all-point residual checks, proper SE(3),
units, neighborhood reversal, row permutation, bounded correlated rotations,
scope preservation, input validation, the shared 256-row cap and numerical guards.

The fixed-seed vectorized synthetic construction has **360 cases**, not 360
independently calibrated experiments: five scales (`1e-6` through `1e6`), four
relative error budgets, six plane separations (`1e-5` radians through `pi/2`),
and three proper rotations. Each case has 50 positions, below the shared cap.
Its in-memory truth tensor has 54,000 float values; observation perturbations
are checked against their stipulated budgets before inference.

The executed sweep returned 270 bounded direction hypotheses and 90 abstentions.
Finite direction bounds were available in 285 cases; **every available bound
contained the known true axis**, and every normal cone contained its known plane
normal. Availability of a bound in an abstained case is intentional: another
required witness or requested angular-precision check can still fail. These
counts are synthetic falsification checks, not probabilities, calibration,
candidate recall, unseen-family accuracy or production-readiness scores.

A separate instrumented execution of that small sweep took **0.06245 s**
and reported process peak RSS **51,937,280 bytes** on Darwin, including
Python/NumPy/pytest. This bounded single-process observation is not a hard
performance guarantee, process-tree benchmark or native scan scalability result.

An additional **optional 10,000-case sweep** uses the same seed and bounded
observation channel, with independently varied proper rotations, scales from
`1e-9` to `1e9`, relative budgets from `1e-10` to `1e-1`, and separations from
`1e-8` radians to `pi/2`, plus exact-parallel and near/exact-orthogonal cases.
It stays outside default pytest collection; no per-call cap is raised. It has
500,000 total position rows distributed across 10,000 separate 50-row analyses.
The vectorized truth tensor has 1,500,000 floats (12,000,000 bytes); additional
arrays and interpreter workspace are included in the measured process RSS.

On the final source this expanded run completed in **1.61671 s** with peak process
RSS **124,141,568 bytes**. It returned 5,873 bounded direction hypotheses and
4,127 abstentions, with 6,472 finite axis bounds. **No normal-cone or available
axis-bound containment assertion failed.** Exact-parallel truth produced no
finite axis bound. The largest observed error/bound ratios were approximately
0.27459 for normals and 0.17867 for axes, showing conservatism in these particular
synthetic constructions, not empirical confidence or completeness. The broader
sweep is not a substitute for the hand-computed degeneracy and coherent-error
counterexamples, and available host memory is not a reason to relax data custody.

Reproduce the focused suite from `FullProject/`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_position_axis_hypothesis.py -q --durations=3
```

The optional sweep can be reproduced without project test configuration or files
other than this explicitly named synthetic test module:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python - <<'PY'
import importlib.util
import json
import resource
import time
spec = importlib.util.spec_from_file_location('synthetic_axis_tests', 'tests/test_position_axis_hypothesis.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
start = time.perf_counter()
result = module.run_synthetic_containment_sweep(expanded=True)
result.update(elapsed_seconds=time.perf_counter()-start,
              process_peak_rss_bytes_darwin=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print(json.dumps(result, allow_nan=False))
PY
```

Source imports are standard library and NumPy; tests add pytest. The repository
`research/__init__.py` is empty, and `research/geometry` has no initializer.
Project pytest configuration, conftest, plugin autoload, bytecode and cache writes
were disabled. No server, detector, native geometry, C15 execution, dependency
installation, physical input, held-out, email, protected truth/control artifact,
or existing scientific output was consumed by the implementation or tests.
Only this new source/test/narrative seam changed; no runtime integration occurred.

Exact final tested source snapshot (SHA-256; bytes):

```text
position_axis_hypothesis.py      bb3f2e4157165519f39337b6879c812299e2d6c61d84f986906ee6528448de0f  11737
test_position_axis_hypothesis.py 32f708d2eb08ea1a135a794f07997a0707bd27973c3eddbcfd9985e06139ef95  25190
```

Stop before physical access, ROI discovery, bend-state claims or promotion. The
next source-only decision is whether a separately scoped uncertainty-preserving
radius bridge or a position-only region hypothesis addresses the next missing
assumption. Neither this test pass nor the available host memory supplies the
physical evidence or authority needed for qualification.
