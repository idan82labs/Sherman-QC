# Adjacent bends: seed conditioning, observed coverage and competing models

Date: 2026-09-07. Engine-only continuation of the
[locked roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
This is additive source and invented-geometry development. Physical access,
runtime promotion, milestone gates and the paused goal are unchanged.

Follow-on: the [same-face tangency checkpoint](2026-09-07_same_face_tangency_checkpoint.md)
tests the retained competing flange explanations with a necessary center/centroid
bound. It preserves this checkpoint's source and raw candidate results while
adding a separately scoped relation assessment.

## Outcome and revised assumption

The new challenge exposes a plane-seed birth failure before radius association.
Adding every full observed radius-cap ball as a separately declared seed family
recovers long, sparsely sampled shared flanges. In each exact/noisy 72-cloud
matrix, complete-observed-support conditional bend events increase from 80 to
120 of 144. No previously matched event is lost. The prior 300-cloud matrix also
retains its complete-support conditional results and aggregate association counts.

This is not physical detection accuracy. Extra unmatched associations increase
from 136 to 144 per adjacent matrix, including 32 to 36 conditional alternatives.
The 24 remaining events lack an observed arc-to-flange seam link and abstain.
Matching all OBSERVED samples does not establish that the physical arc is fully
observed, connected, tangent, manufactured as one event, or conforming.

The initial concern was that an upper-graph bypass around the central flange
would merge two transitions into a compound remainder. That remains a real
model limitation, but it was not the primary failure in this fixed challenge.
Independent stage attribution instead found a connected central flange that
never reached plane fitting as a pure seed. A proposed halfspace/chamber splitter
was therefore NOT integrated as a fix. The separate side-bound primitive below
is only a necessary competing-model diagnostic.

## Independent construction and fixed challenge

The [fixture](../../tests/adjacent_bend_constructions.py) constructs a profile
`flange A -> arc AB -> flange B -> arc BC -> flange C`, extruded along one axis.
For start point p0, unit tangent T0, left normal N0, signed bend angle theta,
radius R, and fraction f, let `u = abs(theta)*f`. Frenet integration gives

```text
p(f) = p0 + R*sin(u)*T0 + sign(theta)*R*(1-cos(u))*N0
T(f) = cos(u)*T0 + sign(theta)*sin(u)*N0
kappa = sign(theta)/R; arc length = R*abs(theta)
```

The implementation uses a half-angle form for `1-cos(u)`. Its
[independent checks](../../tests/test_adjacent_bend_constructions.py) integrate
T(s) using 16-point Gauss-Legendre quadrature and also check hand-derived U/Z
endpoints, circle identities, seam tangency, signed curvature, rigid motion and
units. Analytic truth remains independent of the engine's fitting functions.
There is no thickness, self-collision or manufactured-part claim.

Before the new candidate ran, the
[challenge](../../tests/test_adjacent_bend_challenge.py) fixed:

- Four signed angle pairs: `(90,90), (-90,-90), (90,-90), (-90,90)` degrees.
- Two radius pairs: `(2,2), (2,3)` mm; outer flanges and axial extent 2 mm.
- Three shared-flange lengths: 0.25, 2 and 6 mm.
- Three coverage patterns: full interior arc sampling, one missing middle
  station on arc BC, or only the first half of arc AB's stations.
- Five profile stations by five axial samples per flange; seven interior arc
  stations by five axial samples before omissions. At most 145 raw rows.
- Exact positions and nonzero random errors of 0.25–0.95 times the original
  `epsilon=1e-6 mm`, fixed seed `2026090702`. These are stipulated error bounds,
  not a scanner noise model or calibrated confidence interval.

This is 72 geometries, 144 observed clouds, two constructed events per cloud.
The retained mutual and union baselines each see every identical observed array.
Construction row identities enter only test assertions. Serialized graph edges
are independently traversed to check complete remainder components. Every event
is attributed, and every unmatched association remains in the denominator.
Partial coverage never changes analytic truth or supplies hidden samples.

## Why the old minimum loses a connected flange

The [12-case attribution](../../tests/test_adjacent_seed_attribution.py) preserves
the challenge RNG stream and all old policies. It checks U/S profiles, unequal
radii, each central length, exact/noisy positions and the union graph.

The 25 central-flange samples form one induced graph component in every case.
At lengths 0.25 and 2, respectively 15 and 11 accepted pure-central seed queries
survive. At length 6, there are zero pure-central queried neighborhoods and zero
pure-central accepted seeds. Fifteen of 25 exact central queries exceed the
1.5 mm radius cap; noise raises this to 17 for U and 16 for S. The remaining
queried supports mix arc/flange points. Thus graph connectivity is not sufficient
to create the plane hypothesis.

The long flange has 1.5 mm profile pitch and 0.5 mm axial pitch. Its full observed
cap balls can contain only 5–7 exact points while still spanning two independent
directions. Diagnostic within-cap central subsets have second singular values
about 0.921–1.690 mm. Noise gives 3–7 points: some nearly collinear, others well
conditioned. These oracle subsets are attribution only, never inference seeds.

For centered observed coordinates X with second singular value s2, any ideal
coplanar counterpart under simultaneous Euclidean error epsilon satisfies

```text
||X*n_true|| <= sqrt(N)*epsilon
||X*n_true|| >= s2*sin(angle(n_true,n_fit))
sin(angle) <= sqrt(N)*epsilon/s2
```

Centering is an orthogonal projection, so it cannot enlarge the error-matrix
Frobenius norm. No independence or averaging-away of systematic errors is assumed.
The implementation retains its engineering numerical allowance in this bound.
Three noncollinear points can constrain a plane normal; many almost-collinear
points may not. Point count is therefore neither confidence nor a substitute for
the rank and cone tests. An exact three-point fit still says nothing by itself
about physical membership: three curved-surface samples also define a plane.

## Bounded cap-supported hypothesis

New additive sources:
[plane discovery](../../research/geometry/cap_supported_plane_regions.py) and
[whole-component bend adapter](../../research/geometry/cap_supported_bend_candidates.py).
The predecessor graph-envelope modules are not edited.

The candidate preserves all existing kth-neighbor queries, their minimum of
nine points and their guards. Independently, for EVERY observed point, it queries
the entire observed ball at the ALREADY declared radius cap. This family has an
explicit seed minimum of three, retains every point in that ball, and must pass
the existing all-point plane witness, rank check and normal-cone limit 0.1 rad.
It is not triggered by a failure, does not select a favorable triple, and searches
no new radii or k values. Observed ball membership can still change under noise.

Seed support size is distinct from final region size. The same fixed-cone
proposal and outer-graph components are refitted in full at ORIGINAL epsilon;
final regions still require at least nine points. There is no seed fallback,
residual trimming, greedy ownership assignment or approximate support merging.
The cone is a pointwise necessary relaxation, not a jointly feasible plane.
The inherited TLS-oriented witness is not a complete minimax-plane solver.

Exact support caching preserves query-origin eligibility: if one 3–8-point
support is proposed both by a cap ball and a kth ball, only the cap-ball origin
can contribute a cone or final-region seed reference. Sharing a fit cannot make
the old kth query pass its unchanged nine-point requirement.

No cap is increased: 256 raw rows, 1,024 seed queries, 2,048 unique seed/component
plane fits, 64 FINAL refitted supports, 2,016 plane pairs, 256 associations and
255 two-radius cuts per association. Query preflight now counts
`N*(number_of_k_values+1)`: the additional family consumes the same budget and
can abstain on a previously runnable large query schedule. The final-region
budget unit remains the already-disclosed change from adaptive V1 raw seeds.
Old kth balls can reach nominal cap plus two distance guards through their
two-stage checks; the new cap balls use one guard. Neither adds position epsilon
to the seed radius. This arithmetic guard is not a numerical certificate.

## Paired results and costs

The [paired integration tests](../../tests/test_cap_supported_bend_candidates.py)
run baseline union and candidate on each identical observed array. The earlier
seed `2026090717`, case order and generated errors are preserved for all 300
prior clouds. The adjacent comparison reuses all 144 original challenge clouds.
Together these paired matrices make 888 analyzer calls, not 888 independent parts.

Each of the following adjacent rows pools 24 clouds/48 constructed events per
noise state. Exact and bounded-noise results have the same counts:

| Coverage | Baseline / new matching conditional events | Baseline / new associations | New unmatched / conditional unmatched |
| --- | ---: | ---: | ---: |
| Full observed interior sampling | 32 / 48 | 64 / 80 | 32 / 4 |
| Middle station missing on arc BC | 32 / 48 | 132 / 156 | 108 / 28 |
| First half of arc AB only | 16 / 24 | 20 / 28 | 4 / 4 |
| Total | 80 / 120 | 216 / 264 | 144 / 36 |

All 40 gained events per matrix have the formerly missing length-6 central
flange. The shorter-flange events are unchanged. All matched single-radius
intervals contain their constructed radius; no search is incomplete. The 24
remaining first-half events have no observed second-flange seam link. No gap
filling, hidden contact inference or enlarged association radius rescues them.
The 48 extra associations comprise 40 new full matches plus eight unmatched
alternatives, including four extra unmatched conditional results. Ambiguity is
not solved, and unmatched intervals are not truth-scored.

The interval denominator is explicit: baseline 216 proposed single intervals,
80 matched/checked and 136 unscored; candidate 264 proposed single intervals,
120 matched/checked and 144 unscored, per noise state. Neither run proposes
two-radius band intervals on these adjacent clouds. A returned interval alone
does not imply conditional-positive status. Containment is asserted ONLY for the
matched single intervals, not for the other sampled-support explanations.

For each prior 150-cloud exact/noisy matrix, the candidate retains all 150
complete-support conditional cases, 294 associations, 144 unmatched associations,
90 checked single and 120 checked band intervals. All checked intervals contain
truth; proposed band supports are explicitly matched and unmatched counts kept.
No individual baseline match is lost, and searches remain complete. These are
development constructions reused for method selection, not held-out validation.

Observed maxima are 237 plane fits/seven final regions on the prior cases and
289 fits/six final regions on the adjacent cases. More seed work remains within
the old limits; this is not a scanner-scale performance result. One geometry
process with single-thread BLAS is used, below the 16 GiB process-tree ceiling.

## Signed-side mathematics: necessary model check only

The independent
[side envelope](../../research/geometry/tangent_plane_side_envelope.py) is not
connected to the discovery/association pipeline. It refits a supplied observed
plane region at the original error bound; it accepts no verified normal or face
label. Let m be its observed centroid, n0 a sign-aligned fitted normal, alpha the
normal cone, `v=p-m`, `a=n0.v`, `b=norm(v-a*n0)`, `r=norm(v)`, c=cos(alpha), s=sin(alpha).
The exact extrema over the unit-normal cone are

```text
U0 = r                 if a >= r*c;  otherwise a*c + b*s
L0 = -r                if -a >= r*c; otherwise a*c - b*s
true signed offset is in [L0 - 2*epsilon - guard, U0 + 2*epsilon + guard]
```

When +/-v lies inside the cone, the corresponding extremum is attained there;
otherwise it is on the cone boundary in the n0/v plane. The true plane centroid
and candidate position each contribute epsilon. Correlated errors forbid a
`1/sqrt(N)` centroid reduction. Unresolved intervals belong to BOTH possible
closed halfspaces; they are not assigned to whichever side fits best.

Robust positive and negative points exclude ONLY one SAME-FACE circular cylinder
tangent to the stipulated ideal plane: that cylinder lies entirely in one closed
halfspace. A valid inner/outer concentric mixture can straddle an inner flange
plane, so this rule MUST NOT reject that broader model. One-sided partial secant
arcs and detached circles can pass without contact or tangency. No chamber
restriction, new graph, multi-face rejection or physical diagnosis is implemented.
The [independent tests](../../tests/test_tangent_plane_side_envelope.py) retain
these counterexamples, cone attainers, correlated-error bounds, distant lever
arms, noise, rigid motion, units and fail-closed input/work limits.

## Research grounding and limits of transfer

[Mitra, Nguyen and Guibas' normal-estimation study](https://graphics.stanford.edu/~niloy/research/normal_est/normal_estimation_socg_03_ijcga_04.html)
examines neighborhood size, sampling density, curvature and noise in local
least-squares normals. It motivates examining spatial conditioning rather than
assuming a universal sample count. Our explicit bounded cap-ball policy and
error cone are not their claimed optimal estimator or a statistical guarantee.

[Schnabel, Wahl and Klein's Efficient RANSAC](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2007.01016.x)
is established primitive-detection work.
[CGAL's primary shape-detection documentation](https://doc.cgal.org/latest/Shape_detection/index.html)
distinguishes minimal primitive proposals from final support-size and connectivity
requirements. That distinction supports separating seed size from final support.
We do not adopt random successful-subset selection, largest-inlier ownership or
their performance claims. Their detection tolerance is not authority to loosen
our measurement epsilon.

[GlobFit's author description](https://graphics.stanford.edu/~niloy/research/globFit/globFit_sigg11.html)
uses global geometric relationships alongside local fits to recover engineering
models. Such constraints are useful future CAD-assisted hypotheses. Our QC
inference is that enforcing ideal tangency/parallelism before measuring defects
can erase the very departure under inspection. Model proposal, measurement and
manufacturing-event authority must therefore stay distinct. This is a design
inference, not an experimental validation of our engine by those papers.

## Verification and next boundary

Root's final isolated run passes **200 checks in 26.90 s**, peak RSS
**68,927,488 bytes (65.7 MiB)**. It includes 46 independent construction checks,
three baseline challenge checks, three attribution checks,
[32 independent cap-seed controls](../../tests/test_cap_supported_plane_regions.py),
21 paired-integration/budget/invariance checks, 66 side-envelope checks, and 29
retained axis/radius adversarial checks. No older full-suite run is represented
as rerun in this checkpoint.

```bash
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider \
  tests/test_adjacent_bend_constructions.py tests/test_adjacent_bend_challenge.py \
  tests/test_adjacent_seed_attribution.py tests/test_cap_supported_plane_regions.py \
  tests/test_cap_supported_bend_candidates.py tests/test_tangent_plane_side_envelope.py \
  tests/test_axis_radius_independent_bounds.py tests/test_two_radius_adversarial.py -q
```

The first combined run returned 199 passes and one failure: a side-envelope
analytic test required its floating-point centroid to equal zero exactly, while
the scaled/reference-anchored calculation returned `-2.220446049250313e-16` on
one coordinate. Its cone-interval and all other checks passed. This is retained
as a test-assertion correction, not hidden by a source tolerance or engine change.
The corrected assertion uses an independent machine-epsilon/input-scale allowance;
the exact cone-extrema tests are unchanged. The only source edit after the passing
run removes one extra EOF blank line. AST, scoped whitespace and local-link checks
cover the final files.

All eleven named predecessor discovery/graph/radius source hashes match their
pre-change values. New engine SHA-256 values:

```text
cap_supported_plane_regions.py
8030a3682163265a86e17a2c0139532c11d50ae0a808b7861a76292c256114cd
cap_supported_bend_candidates.py
24caff87fba6504d8dd22cf2e764396ba747c00fb791da6c55ae1d3d9e496335
tangent_plane_side_envelope.py
b64e22ce10a08e7ca612e6164297e703c47b86a0177427475ac28afa17e84cc9
```

Priority 1/M3 remain open. Next work should challenge competing surfaces and
original-joint-model ambiguity, then design a scanner-scale computation path
under the locked CAD-assisted sequence. Missing physical observations cannot be
replaced by additional computation. Calibrated capture errors, face/event mapping
and separately authorized prospective physical evidence remain necessary before
production-floor qualification. No physical/held-out/email inputs or control
records were accessed; no UI work, runtime promotion or service execution occurred.
C15's retained failures, terminal EXP35 and the paused goal remain unchanged.
