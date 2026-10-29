# Plane-derived axis uncertainty carried into radius constraints

Date: 2026-09-07. Bounded source-only engine research, not a physical protocol,
detector qualification or production integration.

Subsequent same-day work adds a
[position-only proposal front end](2026-09-07_position_region_discovery.md).
It generates candidate input regions for this unchanged bridge but does not
verify their physical identity or the original joint-model assumptions.

[`analyze_plane_conditioned_radius`](../../research/geometry/plane_conditioned_radius.py)
connects the existing
[position-only plane-pair direction hypothesis](2026-09-07_position_axis_hypothesis.md)
to the [axis-conditioned radius diagnostic](2026-09-07_axis_conditioned_radius_method.md)
without treating the fitted direction as exact. The bridge imports those two
pure analyzers and does not edit them. Its
[focused tests](../../tests/test_plane_conditioned_radius.py) retain analytical
truth outside the public observation channel.

The decisive regression is simple to state: exact samples from a radius-2
cylinder can project to a perfect radius-4 circle after a small axis tilt. The
old exact-axis plug-in reports a narrow radius interval excluding 2. The bridge
retains 2 and abstains after propagating the position-derived axis uncertainty.
This fixes a conditional research inference; it is not scan-only detection or
evidence that a physical bend is formed/conforming.

## Inputs and joint assumptions

The public inputs are three explicitly supplied position arrays:
`first_neighborhood`, `second_neighborhood`, and `cylinder_region`; an explicit
`m` or `mm` unit; one simultaneous `position_error_bound`; and the explicit
policies `max_axis_angle_bound` in radians, `max_axial_gap`, and `max_radius_width`.
All dimensional quantities use the declared unit. The **256-row cap is shared
across all three arrays before conversion and exact-repeat removal**. Empty or
absent regions cause abstention; repeated rows do not add independent evidence.

The conditional original model assumes all of the following:

1. Every position in each plane neighborhood has a counterpart on its stipulated
   ideal plane within the common Euclidean error budget `epsilon`.
2. Every cylinder-region position has a counterpart on one ideal circular
   cylinder within that same budget.
3. That cylinder's true axis direction is parallel to both true planes.

These assumptions are not established by the code. The bridge does not discover
regions, prove tangent contacts, choose a physical component, verify inner/outer
face identity, or establish that its three regions belong to one object. They
need not be disjoint, and errors may be correlated: none of the deterministic
implications requires statistical independence. Reusing positions or their
estimated normals is not extra evidence for the common-axis relation.

The plane-pair analyzer returns an unoriented direction `a_hat` and a cone of
half-angle `delta` containing every compatible true direction under its stated
assumptions. If its own witness/degeneracy/angular-resolution checks abstain, the
bridge does not invoke the radius analyzer with an unresolved direction.

## A single common rotation gives a radius-preserving relaxation

Fix the observed centroid `c` of the unique cylinder-region positions. It is an
explicit pivot computed from centered/scaled coordinates, not a measured axis
location or a noiseless estimate of a true centroid. In exact arithmetic any
fixed observed pivot would suffice; the centroid is a deterministic, local choice.

Consider **any** original joint model satisfying the three assumptions. Choose
the sign of its true unit axis `a` so its angle from `a_hat` is at most `delta`.
There is a proper minimal rotation `Q` taking `a` to `a_hat` through angle
`theta<=delta`. Apply this **same rotation about the same pivot to every true
cylinder counterpart**:

```text
q'_i = c + Q(q_i-c).
```

Orthogonal rotations preserve Euclidean lengths, and rotation plus translation
is a rigid displacement. These standard conventions are given in
[Lynch and Park, Modern Robotics, rotation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/)
and [rigid transformations](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/).
In particular, the transformed counterparts remain on **one cylinder with the
same radius `R`**, now parallel to `a_hat`; its location is allowed to move.
One cannot use different rotations for different points and retain this property.

The rotation displacement factor is derived directly. `Q` fixes its rotation
axis and acts as a planar angle-`theta` rotation on the perpendicular plane, so
the singular values of `Q-I` are `0,2 sin(theta/2),2 sin(theta/2)`. Therefore,

```text
||Q-I||_2 = 2 sin(theta/2) <= 2 sin(delta/2) = k.
```

For observed positions `p_i`, let

```text
D_observed = max_i ||p_i-c||
D          = D_observed + epsilon.
```

The extra `epsilon` is essential: the true counterpart can lie farther from the
pivot than the observation. Triangle inequalities then give

```text
||q_i-c|| <= ||p_i-c|| + ||q_i-p_i|| <= D
||q'_i-p_i|| <= ||(Q-I)(q_i-c)|| + ||q_i-p_i||
             <= k D + epsilon = epsilon_rotated.
```

Thus every original admissible radius `R` also belongs to a fixed-axis cylinder
sample explanation under the **inflated** budget

```text
epsilon_rotated = epsilon + 2 sin(delta/2)(D_observed+epsilon).
```

Feeding that budget to the existing fixed-axis radius outer-constraint method
therefore gives necessary outer constraints for the original radius under the
joint assumptions. This is an exact-arithmetic model implication, not a posterior,
confidence interval or calibrated physical uncertainty. All original admissible
models are covered by the same observed `delta` and `D`; the compatible rotation
may differ between models but remains common across samples within each model.

## Why a positive relaxed witness is not a joint-model witness

The implication goes only one way. A circle found within `epsilon_rotated` need
not have corrections decomposable into one common rotation and per-point errors
at most the original `epsilon`, nor need its axis relation be compatible with
both original plane neighborhoods. The relaxation can introduce explanations
that the original joint model does not admit.

Consequently, the bridge's positive status is
`CONDITIONAL_RADIUS_HYPOTHESIS`, requiring the plane-axis checks and all the
inflated fixed-axis diagnostic checks, including its flat alternative, radius
width and observed gap/extent checks. It **never** asserts original joint-model
compatibility. `original_joint_compatibility` is always `UNESTABLISHED`, including
for a positive result. The status retained inside `relaxed_radius_diagnostic`
describes only that rotated/error-relaxed problem, as its explicit `scope` states.

The radius interval is a necessary outer constraint, not the exact feasible
radius set. A finite narrow interval does not independently establish that the
original model exists. All axis/region/common-relation/joint-model/physical-support
verification flags stay false. `max_radius_width` remains the caller's conditional
resolution policy, **not physical qualification**. No tolerance is selected from
physical data or modified to obtain the positive status.

The relaxation can be deliberately overconservative. In a regression, the fitted
axis is exactly correct, but small plane footprints under an explicit error
budget leave a broad cone; its propagated positional allowance prevents a useful
radius result. That is expected abstention, not a license to ignore uncertainty.

## Axial coordinates are samples, not verified physical extent

For the same pivot and sign-aligned true axis, define

```text
u_i_true = a dot (q_i-c)
u_i_hat  = a_hat dot (p_i-c).
```

Since `Q a=a_hat`,

```text
u_i_true = a_hat dot (q'_i-c)
abs(u_i_true-u_i_hat) <= ||q'_i-p_i|| <= epsilon_rotated.
```

The bridge discloses this per-sample scalar-coordinate error budget with its
numerical allowances. It uses the same local centroid convention as the imported
radius analyzer. This is not a bound on unobserved physical endpoints. Under
axis/position uncertainty, samples can reorder and the groups formed from their
observed projections can split or merge. Therefore:

- Exported groups have `OBSERVED_PROJECTIONS_ONLY` scope and explicitly false
  physical-support and true-axis-membership flags.
- An observed gap is retained as a sampling-policy gap, not a material gap or
  proof of disconnected parts; it is never silently filled.
- Fitted axis locations and axis endpoints are **not exported** by the bridge.
  A line fitted in the rotated surrogate would not be a measured physical axis
  or an admissible surface marker in the original frame.

Neither the projected ranges nor the inflated residuals prove a finite curved
transition, continuous surface identity, forming state, thickness or conformance.

## Numerical ledger

The plane analyzer's position and angular guards are already included in its
reported `delta`. The bridge separately reports `D_observed`, `D`, `k`, `kD`,
and `epsilon_rotated`; then it adds a disclosed engineering allowance

```text
bridge_numerical_allowance = 128 machine_epsilon max_abs(p_i)
                           + 128 machine_epsilon scale
                           + 128 machine_epsilon D_observed,
scale = max_abs(p_i-p_0).
```

The radius analyzer receives `epsilon_rotated+bridge_numerical_allowance` and
adds its own existing numerical allowance. All three stages are separately
auditable in `uncertainty_propagation`. The pivot's footprint is evaluated about
the actually returned floating-point pivot. Unsupported ranges abstain with
finite JSON instead of exporting infinities as measured precision.

These constants are engineering guards, not directed-rounded interval arithmetic,
proven forward-error bounds or numerical certification. The limitations of the
two imported analyzers remain; the bridge cannot upgrade their numerical status.
Large global offsets can enlarge guards without supplying physical information.
Proper rigid transforms preserve the underlying radius/direction relation, and
unit conversion scales every dimensional input and propagated bound together.

## Independent zero-noise-cylinder counterexample

An independent mathematical reviewer supplied this 15-position construction.
Let true axis `a=(0,1,0)` and tilt angle `theta=0.01` radians. True plane samples
are the corners of `z=0` and `x=0`, with their two varying coordinates `+/-0.001`.
Rotate both observed plane patches about the x axis by `theta`. Each displacement
is bounded by the common budget

```text
epsilon = sqrt(2) 0.001 * 2 sin(theta/2)
        = 0.00001414207669823965.
```

For `phi=0.2,0.4,...,1.4`, construct seven **exact** cylinder-region observations:

```text
x = 2 cos(phi)
z = 2 sin(phi)
y = (cos(theta) z - sqrt(16-x^2))/sin(theta).
```

They satisfy `x^2+z^2=4`, so their true cylinder radius is 2; cylinder positional
noise is zero. The fitted plane-pair direction is
`a_hat=(0,cos(theta),sin(theta))`. Its perpendicular coordinate

```text
v = cos(theta) z - sin(theta) y = sqrt(16-x^2)
```

satisfies `x^2+v^2=16`. The wrong-axis projection therefore lies on a perfect
radius-4 circle. This is an equation-level oracle, not a generator/fitter inverse.
Plane fitting does not independently verify the erroneous direction as exact.

With explicit gap 1000 and radius-width policy 0.01, the executed comparison was:

| Analysis | Fitted radius / outer interval | Result |
| --- | --- | --- |
| Naive fitted-axis plug-in, original position budget | Radius 4; `[3.9983981041,4.0016029285]` | Sample witness falsely excludes true radius 2 if treated as exact-axis measurement |
| Uncertainty-preserving bridge | `[0,unbounded]` | Abstained; true radius 2 retained |

The plane-derived axis cone is about `0.02828863339` radians. The observed
footprint radius about the common pivot is about `66.91278190`; the propagated
position budget is about `1.89282258376`. Its loss of resolution is real for this
conservative relaxation. A perfect radius-4 residual cannot substitute for the
missing axis certainty. Separate tests directly rotate the true counterparts
with one common `Q` and verify their displacement and axial-coordinate bounds.

## Executed source-only verification

The final focused suite passed **58 tests in 0.18 s**, using Python 3.11.15,
NumPy 2.4.6 and pytest 9.1.1. Its additional near-tight normal-cone regression
constructs errors saturating `sin(theta)<=sqrt(N) epsilon/s2`, then checks the
true tilted-cylinder radius through the bridge.
The cases cover integer-coordinate planes/circle, explicit propagation arithmetic,
the independent radius-2/radius-4 construction, deliberate overconservative
abstention, proper SE(3), units, neighborhood reversal, row permutation, repeated
positions, short-arc flat ambiguity, observed support gaps, missing evidence,
invalid policies/inputs, shared allocation cap, and numerical-range abstention.

Default pytest includes an 80-case fixed-seed synthetic sweep. The separately
invoked expanded sweep has **1,000 cases**, each with 17 positions: four in each
plane neighborhood and nine in the cylinder region. It varies radius scale from
`1e-3` to `1e3`, relative error budgets from `1e-8` to `1e-2`, plane/cylinder turn
from `1e-3` radians to `pi/2`, proper orientations and translations, with bounded
point perturbations independently checked before inference.

The expanded execution returned **290 conditional radius hypotheses and 710
abstentions**. Propagation was available for 913 cases and finite radius upper
bounds for 416. Every reported radius outer constraint retained the known radius;
all available common-rotation and per-sample axial-coordinate bounds contained
their constructed truth. These are synthetic containment checks, not independent
physical trials, detection accuracy, probabilities or confidence calibration.

The bounded expanded run took **0.59817 s** and reported peak process RSS
**52,576,256 bytes** on Darwin, including Python, NumPy and pytest. It did not raise
the per-call cap. This is one synthetic single-process observation, not a hard
performance guarantee or native/production process-tree benchmark.

Reproduce focused tests from `FullProject/` using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_plane_conditioned_radius.py -q --durations=3
```

Reproduce the optional bounded sweep without reading project test configuration:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python - <<'PY'
import importlib.util
import json
import resource
import time
spec = importlib.util.spec_from_file_location('synthetic_bridge_tests', 'tests/test_plane_conditioned_radius.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
start = time.perf_counter()
result = module.run_synthetic_bridge_sweep(expanded=True)
result.update(elapsed_seconds=time.perf_counter()-start,
              process_peak_rss_bytes_darwin=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print(json.dumps(result, allow_nan=False))
PY
```

Imports remain standard library, NumPy and the two existing pure research
analyzers; tests add pytest. No runtime, detector, native geometry, dependency
installation, C15/frozen mechanism change, physical/held-out/email input, protected
truth/control record or prior scientific output was accessed or executed in this
bridge step. Its only edits are the new bridge, its tests, and this method note.

Exact final focused-test source snapshot (SHA-256; bytes):

```text
plane_conditioned_radius.py      c2b2a2972bfc1f832d591060fee4e06ac935010b0ff8df247dc3dc4a90239fee  11694
test_plane_conditioned_radius.py a892f1c5ad96374c8fd6407100ffeac2889348c06f7fe9091e36c63b545b0f4f  24407
```

The next missing engine assumption is position-only region/common-axis hypothesis
generation with the same honest uncertainty and abstention semantics. This bridge
does not supply that discovery. Physical calibration, prospective evaluation and
production promotion still require their separate evidence and authority.
