# Axis-conditioned radius: a bounded sample-compatibility method

Date: 2026-09-07. Source-only research engineering, following the
[analytic finite-bend fixture](2026-09-07_analytic_bend_testing_design.md).
This note supplies the mathematical contract for
[`analyze_axis_conditioned_radius`](../../research/geometry/axis_conditioned_radius.py).
It is not a detector, physical experiment, scanner calibration, runtime promotion,
or permission to inspect physical inputs. The executed checkpoint below records
the exact source-only verification separately from the mathematical claims.

## Question, inputs, and conditional meaning

Given at most 256 position samples, a supplied axis direction, one explicit unit,
a per-point Euclidean position-error budget, an axial sample-gap policy, and an
acceptable absolute radius-interval width, can the samples admit a sufficiently
bounded common circle after projection normal to that direction, while excluding
a competing parallel extrusion plane?

The inputs are `points`, `axis`, `unit`, `position_error_bound`, `max_axial_gap`,
and `max_radius_width`. Positions are the only observation channel. Exact fixture
normals, patch/face/component labels, CAD radius, expected bend count, and hidden
construction parameters are not inference inputs. The supplied direction is a
hypothesis, not an estimated or independently verified axis. The selected ROI is
also a hypothesis that its samples belong to one relevant face, not proof of that
assignment. No axis location is supplied as measured truth: the fitted circle
centre supplies only a candidate parallel axis location.

The conditional model assumes that the true counterparts of all selected points
lie on one ideal circular cylinder parallel to the supplied direction and differ
from their measured positions by no more than the stated bound. The analysis
does not establish those assumptions. In particular:

- An uncalibrated error budget does not become calibrated through fitting.
  The deterministic derivation needs a simultaneous bound for every point; it
  does not assume independent, zero-mean, Gaussian, or identically distributed
  errors and does not produce a confidence probability.
- Direction error is not silently covered by the point-error budget. A wrong
  direction changes the projection and can introduce radius/centre variation,
  yet special sparse samples may still fit. A future direction-uncertainty model
  would need its own justified propagation over a bounded spatial footprint.
- Mixed faces, flange samples, edges, other objects, noncircular form errors,
  and unbounded outliers invalidate the ideal-model interpretation. Some such
  mixtures can also be indistinguishable within the chosen budget; successful
  fitting does not validate ROI purity.
- Removing exact repeated positions avoids treating repeated rows as new
  geometric evidence. It is not independent resampling, denoising, robust
  segmentation, or permission to discard inconsistent points.

All dimensional parameters must use the declared unit. Converting units scales
positions, error budgets, sample gaps, radius estimates, and interval widths
together; the axis direction is dimensionless. Translation/rotation and axis-sign
changes must not introduce a new physical measurement, though finite-precision
guards can change at very large coordinate offsets.

## Projection and a constructive circle witness

Let `a` be the normalized supplied direction and let `e1,e2` be an orthonormal
basis of its perpendicular plane. With a local origin `o`, write

```text
y_i = (e1 dot (p_i-o), e2 dot (p_i-o))
u_i = a dot (p_i-o).
```

Orthogonal projection cannot increase Euclidean position error. If the true
cylinder is parallel to `a`, its true projected samples lie on one circle and
remain within the same error budget of `y_i`.

A centred/scaled algebraic least-squares circle seed followed by a bounded
geometric refinement supplies a candidate centre `c` and radius `r`. It is a
numerical search, not a proof of global optimality. For this particular candidate,

```text
rho_i = abs(norm(y_i-c) - r)
```

is the Euclidean distance to the corresponding infinite cylinder. If every
`rho_i <= epsilon`, moving each projected point to its nearest point on the
circle while keeping `u_i` fixed explicitly constructs admissible cylinder
samples. This is a positive sample-compatibility witness under that budget.

A small RMS residual is insufficient: it can conceal an individual violation.
Conversely, a candidate with maximum residual greater than the budget does not
prove that no admissible circle exists. The search may fail or a least-squares
solution may sacrifice one point even when another centre/radius satisfies the
maximum-residual constraint. Such a failure requires abstention, not a physical
negative conclusion. The parameter enclosure below is independent of this fit.

## Exact-arithmetic radius outer enclosure

The following bound is derived here; it is not a statistical formula or a NIST
certification. Consider any three observed projected points `p0,p1,p2`. Define

```text
u = p1-p0                       v = p2-p0
d01 = norm(u)                   d02 = norm(v)
d12 = norm(u-v)                 D = abs(cross(u,v))
B = epsilon (d01+d02+d12) + 3 epsilon^2.
```

Here `cross` is the scalar two-dimensional determinant, so `D` is twice the
observed triangle area. Let the true projected points be `q_j=p_j+delta_j`, with
`norm(delta_j)<=epsilon`. Each true edge length lies in

```text
d_j^- = max(0, d_j-2 epsilon),      d_j^+ = d_j+2 epsilon.
```

Expanding the signed determinant gives a linear perturbation

```text
cross(delta1,v) + cross(u,delta2) + cross(delta0,u-v)
```

and a quadratic perturbation

```text
cross(delta1,delta2) - cross(delta1,delta0) - cross(delta0,delta2).
```

The triangle inequality bounds these by `epsilon (d01+d02+d12)` and
`3 epsilon^2`, respectively. Therefore the true unsigned determinant lies in
`[max(0,D-B),D+B]`. For three distinct points on a finite circle the circumradius
identity is `R=d01 d02 d12/(2 D_true)`. It follows that every admissible common
circle obeys

```text
R >= product(d_j^-)/(2 (D+B)),              when D+B > 0
R <= product(d_j^+)/(2 (D-B)),              only when D > B.
```

When `D<=B`, this triangle supplies no finite upper bound; it must not be
regularized into a finite radius claim. If an edge can collapse, its zero lower
factor gives the harmless lower bound zero. In exact arithmetic, `D+B=0` with
all edge lower bounds positive is the special contradiction of three distinct
collinear points on a finite circle. With positive positional uncertainty,
arbitrarily large circles may instead remain compatible with nearly flat data.

A pair also gives `R >= max(0,d-2 epsilon)/2`. The implementation selects a
diameter pair of the observed projection and intersects the outer intervals of
its fan triangles with this pair bound. All selected triples must contain the
same radius under the model. No probabilistic independence between triangles is
needed; their errors can be correlated. Selection based only on observed
coordinates does not invalidate these deterministic implications.

A nonempty intersection is a **necessary**, not sufficient, condition for a
common-circle explanation. Bounds for different triangles can permit mutually
inconsistent point corrections or centres. It is not the exact feasible-radius
set. In exact arithmetic an empty intersection contradicts the bounded-error
common-circle hypothesis; the conservative public outcome still abstains and
does not turn that contradiction into an unformed, missing, or defective bend.

Three-point sensitivity is well established as an important metrology issue.
Hopp's analysis concerns first-order propagation of small random errors for
perfect circular form; it must not be substituted for a deterministic
all-points bound or used to cover unmodelled form deviations.
[Hopp, NISTIR 5501, The Sensitivity of Three-Point Circle Fitting (1994)](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir5501.pdf).

### Floating-point scope

The implementation centres/scales coordinates and uses an explicit allowance

```text
numerical_allowance = 64 machine_epsilon (max_abs_input_coordinate + scale)
effective_epsilon  = position_error_bound + numerical_allowance.
```

Computations use the effective budget consistently and disclose it separately
from the caller's positional budget. The allowance is an engineering guard;
it is not a proven forward-error bound, directed-rounded interval arithmetic,
or a certified numerical enclosure. The exact-arithmetic theorem above does not
magically certify its floating-point implementation. In particular, acceptance
within `effective_epsilon` must not be reported as a witness at the stricter
`position_error_bound`, and a large translation can enlarge the guard without
adding any physical information. Numerical boundary cases should abstain rather
than be interpreted as confidence or physical precision.

A compatible fitted radius outside the computed outer interval is a direct
numerical self-inconsistency and triggers `NUMERICAL_ENCLOSURE_INCONSISTENT`
abstention. It does not justify excluding a physical circle. The response echoes
both the gap policy and requested radius width so that these decision conditions
are not detached from their caller-supplied settings.

## A separately tested flat alternative

For a unit projected normal `n`, define the strip width

```text
w(n) = max_i(n dot y_i) - min_i(n dot y_i).
w = min over unit n of w(n).
```

The strip midline has maximum perpendicular residual `w/2`. Extruding that
midline along `a` constructs a plane explanation of the original positions.
Thus, in exact arithmetic, a plane parallel to the supplied direction is
compatible with budget `epsilon` exactly when `w<=2 epsilon`. This comparison
optimizes the plane location and orientation within that plane class; it does
not pretend a fitted or CAD plane is an exact, free reference.

The finite planar convex hull suffices in principle: a minimum-width strip has
a supporting hull edge, so checking all hull-edge normals obtains the minimum.
The implementation instead enumerates all distinct pair-edge orientations in
bounded chunks. These include the hull edges, and every tested width still uses
all samples. Additional orientations cannot introduce a false plane witness.
This avoids a hull tolerance silently deleting shallow-arc points. Degenerate
point/line sets have width zero; no optimized rotating-calipers sweep is needed.
[Houle and Toussaint, Computing the Width of a Set, section 2](https://cgm.cs.mcgill.ca/~godfried/publications/width.pdf).

This is not an unrestricted minimax fit of an arbitrary 3D plane, nor an
exhaustive comparison with every noncylindrical surface. Excluding this flat
alternative alone does not establish the physical surface class. The numerical
implementation compares against `effective_epsilon`, with the limitations above.

### Zero-residual but wrong-radius counterexample

An independent three-point construction makes the observability problem explicit:

```text
true samples:       (-1,0),   (0,h),    (1,0)
true radius:        (1+h^2)/(2h)
comparison plane:   y = h/2
midpoint samples:   (-1,h/4), (0,3h/4), (1,h/4).
```

The midpoint observations are at most `h/4` from both the true samples and the
plane. Yet the midpoint observations lie exactly on another circle, of radius
`1/h+h/4`. For `epsilon=0.001` and `h=0.004` in one consistent length unit,
the true radius is `125.002`, the zero-residual observed-circle radius is
`250.001`, and the observed strip width is `0.002=2 epsilon`. The radius
enclosure must retain the true radius and have no finite upper bound at this
boundary. Perfect fitting, more repeated rows, or normals re-estimated from
these same positions cannot remove the positional ambiguity.

## Finite sample support is not physical topology

Sort the observed axial coordinates `u_i` and retain separate sample groups under
the explicit `max_axial_gap` policy. Report their observed ranges separately.
Do not convert their total minimum/maximum into one supported cylinder segment.
Gap comparisons use adjacent original 3D point differences projected onto the
axis, with strict comparison against the supplied gap rather than a tolerance
that silently bridges it. Fitted-axis endpoints are marked as a circle-fit
hypothesis; without a usable fit those endpoints are unavailable, not replaced
by an arbitrary line through the sample centroid.
True axial sample positions themselves can differ by up to the positional
budget under the fixed-axis model; the grouping policy is about observations,
not a proof of true surface continuity.

Samples on ranges `[-8,-3]` and `[3,8]` must not silently fabricate observations
in the intervening 6-unit gap. Those same positions could come from disconnected
cylinders or from a continuous cylinder with missing capture. Neither component
identity nor the absence of physical material follows from the gap. A caller
must not choose a large gap merely to make an expected component pass.

A positive-span sample group is a conservative prerequisite for the proposed
witness status, but not proof of a two-dimensional cylindrical surface. One
circular cross-section can also be a sphere section. Several circular slices
can be interpolated by a surface whose radius varies between the sampled
locations. Three noncollinear flange-only projected positions also determine a
circle, even when no sampled point is on a curved transition. Finite position
samples without a validated acquisition/ROI model cannot uniquely identify the
unsampled physical surface.

## Result contract, falsification, and stop conditions

For a well-formed analysis, `SAMPLE_CIRCLE_WITNESS` requires all of the following:

- An all-point circle witness within the disclosed effective error budget.
- The fitted radius lies inside the computed enclosure; numerical disagreement
  causes abstention rather than silently repairing either result.
- No compatible extrusion plane parallel to the supplied direction.
- A finite radius outer-interval width no greater than the caller's explicit
  `max_radius_width`; fit quality cannot replace this requirement.
- One axial sample group under the explicit gap policy, with observed span
  greater than twice the effective error budget.

Other geometric outcomes are `ABSTAINED`, retaining explanatory diagnostics.
The `circle_witness` field can retain a candidate fit whose
`compatible_with_error_bound` is false; its presence alone is not a witness.
Without a usable candidate fit, group `axis_endpoints` are null and
`axis_geometry_source` is `UNAVAILABLE`, not an arbitrary line through the sample
centroid. Available endpoints are explicitly `CIRCLE_FIT_HYPOTHESIS` sample
extents, not on-face or admissible product markers. The result echoes both caller
gap/precision policies for reproducibility.
Even the positive status means only that these checks succeeded under the stated
hypotheses. It does not verify the axis or ROI, uniquely identify a physical
cylinder, recover an inner/outer face identity, determine thickness or a neutral
layer, recover full/signed bend angle, prove formation, establish conformance,
or authorize a count or product decision. An interval width is a bounded-model
resolution requirement, not a confidence level or calibrated production tolerance.

The first independent oracles should include hand-calculated circular triples,
rigid transforms and unit scaling, bounded radial/tangential/common-mode
perturbations, collinear and nearly collinear points, repeated positions,
inconsistent fourth points, both sides of the strip boundary, the midpoint
counterexample, wrong-axis alternatives, and disconnected sample ranges.
Test point generation must keep truth separate from inference inputs.

Stop or revise the mechanism before further claims if an admissibly perturbed
known circle falls outside its enclosure; a plane-compatible short arc receives
the positive status; residual summaries hide violating samples; disconnected
ranges become one claimed support segment; exact labels/normals/CAD values enter
the observation channel; or the status is consumed as physical bend verification.
Do not repair such failures by fixture-specific error budgets, hidden sample
removal, held-out threshold selection, or weakening the output meaning.

If representative use cannot provide justified axis, ROI, and error-budget
hypotheses, this conditioned mechanism is insufficient for that use. The next
scientific hypothesis must address those missing inputs prospectively rather
than rename the same fit an autonomous detector. Synthetic checks alone cannot
supply physical calibration, candidate recall, unseen-family generalization,
or production qualification. No protected data, detector, frozen C15 mechanism,
native geometry runtime, or physical inspection is needed for this source step.

## Executed source checkpoint

On 2026-09-07, independent test authoring and math review used `gpt-6-astra` at
`max` reasoning effort. The integrating agent reviewed the source, derivation and
tests, then ran **126 tests: all passed in 0.11 s**. No fixture-specific policy was
selected from physical data. No production/runtime/C15 source or historical
evaluator changed in this step; the new method is not wired into the application.

The test observation channel contains only positions and the stipulated axis.
Hand-computed Cartesian triples, direct 3D witness-distance checks and known
bounded perturbations supply independent oracles. Additional checks cover
transforms, units, strict gap boundaries, duplicated positions, mixed faces,
outliers, optimizer failure, inconsistent numerical bounds, JSON finiteness,
extreme finite scales and allocation rejection before array conversion.

Selected synthetic observations, not physical results:

| Case | Diagnostic result | Meaning |
| --- | --- | --- |
| Integer-coordinate quarter-circle, radius 5 mm; stipulated position budget 0.001 mm | Fitted radius approximately 5 mm; outer interval approximately [4.98585694, 5.01419062] mm; sample witness | Conditional positional compatibility and requested resolution, not calibrated accuracy |
| Midpoint ambiguity above, true radius 125.002 mm | Zero-residual fit approximately 250.001 mm; outer interval approximately [124.32981215, unbounded]; abstained | Perfect residual cannot replace observability |
| Sparse points from two flat flanges | Radius sqrt(82) sample-circle witness | A successful sample fit does not identify a physical curved transition |
| Same flat flange patches with intermediate positions | Abstained | Failure of the single-circle ROI hypothesis, not evidence of an absent/unformed bend |
| Two sampled axial ranges separated by a gap | Separate sample groups; abstained | No interpolation into unsampled support and no assertion of physical disconnection |

A separate fresh-process check used 256 distinct invented positions (16 angles
over a quarter-circle of radius 10 mm, repeated at 16 axial coordinates). One call
took **0.0207 s** and process peak RSS was **35,422,208 bytes** on Darwin, including
Python/NumPy. This is one small synthetic single-process observation, not a
process-tree production benchmark, a hard performance guarantee, or native
registration/scan scalability evidence.

Exact tested source snapshot (SHA-256; bytes):

```text
axis_conditioned_radius.py      9c6aa59594d813a0ad70b3f93aef99c4be80035c1d46d64cca5c9453dbf46bdc  16013
test_axis_conditioned_radius.py f12aa6fa46b8d1a905c50756d077953dc1bae0040a99664f86f6a6b3c624c8a1  28030
```

Reproduce from the `FullProject/` Git root with the installed application Python:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_axis_conditioned_radius.py -q
```

Imports are limited to standard library and NumPy in the implementation;
tests add pytest. No dependency installation, server/model/native detector,
physical input, held-out/email, protected artifact/control, or existing scientific
output was accessed. C15's previously reported three source-freeze failures and
the terminal EXP35 boundary remain unchanged; their suites were not rerun here.

The next bounded mechanism must supply position-derived axis and local-region
hypotheses without oracle labels, and test their error/ambiguity propagation.
Do not promote this conditioned verifier into a candidate-recall or bend-state
claim before those missing inputs are addressed. Physical calibration, reviewer
event agreement and prospective evaluation still require their existing gates.
