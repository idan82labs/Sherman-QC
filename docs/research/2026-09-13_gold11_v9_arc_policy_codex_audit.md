# V9 arc-policy feasibility: independent source and mathematics audit

Date: 2026-09-13. Reviewer: Codex/Astra `v8_flange_policy_math_audit`.
Scope: diagnosis Part E and the V1/V2 arc-selection and flat-witness source.
Only this new report was written. No engine/test source was edited, no module
was imported, no test/fixture/geometry job was executed, and no physical input,
Gold, attempt artifact or cache was accessed. The numerical table below was
checked with scalar trigonometric arithmetic only, not an engine invocation.
Fable retains implementation ownership; both prior reviewer reports remain
unchanged.

Inspected source identities:

- `docs/research/2026-09-13_fable51_v9_diagnosis.md`:
  `2d01be35ab89074d3b023d91a86aa81c7eb489bcae3877bbd9f74b1cb0962ece`.
- `backend/cad_guided_measurement.py`:
  `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb`.
- `backend/cad_guided_measurement_v2.py`:
  `59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d`.

The diagnosis was read fully; line references below refer to that snapshot.
This is not an independent verification of the reported 99 native-window
inventory, the author's scratch-script results, or physical measurement truth.

## Decision

**The central exact-nominal incompatibility is mathematically supported.**
For a perfect, correctly aligned, single-inner circular extrusion with radius
0.78–0.80 mm and 90-degree sweep, the frozen delta=0.05 mm / exclusion=0.075
mm policy removes so much near-contact arc that even the continuously sampled
remainder is compatible with a free-offset line at delta. No subset of that
ideal remainder can escape the kernel's flat-witness gate. At delta=0.05,
even reducing exclusion toward its allowed lower limit does not resolve this
particular ideal model.

**The stronger physical impossibility claim is not supported.** Nominal CAD
geometry is not the measured geometry; radius/sweep deviations, other selected
surfaces, noise and context-dependent selection change the actual point set.
Part E's “no measurement ... whatever the registration did” and “without this
change V9 cannot produce a measurement ... regardless of every other mechanism”
must be narrowed to exact-nominal clean-model feasibility. This distinction is
essential precisely because the engine is meant to measure deviations.

**Delta=0.03 / exclusion=0.04 is a defensible new development-policy candidate,
not a uniquely derived optimum or calibrated noise model.** It restores
continuous nominal feasibility with approximately 0.0112–0.0130 mm of
half-width margin before finite sampling and noise. It tightens maximum fit
residual while easing other coupled gates. Its actual yield/error behavior
requires the independently frozen discrimination tests and complete retained
outcomes; no F1 improvement is implied.

## Actionable findings

### P1 — Keep the impossibility theorem conditional on actual geometry equaling nominal

Part E at `v9_diagnosis.md:225` correctly analyzes a clean circular arc with
its true tangent planes. At `:246` and `:262`, it turns that into an assertion
about any possible physical V7/V8 measurement. The latter does not follow
from nominal radii and sweeps.

Source-independent counterexample to that inference: keep the declared
nominal radius at 0.8 mm, nominal sweep 90 degrees, and the frozen 3 mm search
margin, but let the actual clean inner radius be 1.5 mm. With the same center,
axis and true 90-degree sweep, the true tangent flange planes are displaced
only 0.7 mm from nominal and the radius remains below the nominal 3.8 mm arc
cap. After exclusion=0.075, the actual usable arc is approximately 53.61
degrees and its continuous minimum strip half-width is approximately 0.08059
mm, above delta=0.05. With adequately distributed finite samples and broad
flanges, the nominal-geometry flatness impossibility no longer applies.
The kernel fits radius freely; it does not prohibit this radius deviation
(`backend/cad_guided_measurement.py:190`, `:323`, `:333`).

This is not a claim that such a deviation exists in Gold11 or that any retained
physical case would pass every other gate. It demonstrates why a statement
about declared nominal geometry cannot rule out all measured outputs. A wrong
context might also select a different, larger curve and generate an incorrect
output; the nominal theorem does not certify the absence of such mistakes.

Recommended substance: “The frozen policy is infeasible for clean,
correctly selected exact-nominal 0.78–0.80 mm/90-degree arcs. This supplies a
generic engineering reason for a versioned policy study. It neither attributes
the recorded first-stage refusals nor proves every possible physical scan
subset is unmeasurable.”

### P2 — The 0.03 choice is a finite-fixture development choice, not the analytic maximum

At r=0.8, exclusion=0.04 gives continuous strip half-width 0.042982 mm;
at r=0.78 it gives 0.041193 mm. Both are also above delta=0.035. Therefore
the reported delta=0.035 failures at 0.3 mm sampling (`v9_diagnosis.md:239`)
are not forced by the continuous geometry. They concern the retained finite
fixture and its sampling phase/distribution or another gate.

The statement “largest 0.005-step value ... on perfect data” (`:253`) can be
retained only with its exact finite fixture qualification, not as a universal
mathematical upper limit. Selecting 0.03 after those results is allowed
consumed-development engineering, but it should be labeled that way. A new
physical freeze does not make the preceding synthetic grid unconsumed or
independent confirmation.

### P2 — Do not present the 3.5-times-RMS proxy as expected window refusal share

The diagnosis already calls the factor indicative, but the heading at `:265`
and conclusion at `:279` still extrapolate patch-proxy proportions into an
expected share of failed windows. The source applies an all-points maximum
to a fitted flange crop, not 3.5 times a small selected patch's RMS.

For a fixed residual vector of length N, the only immediate deterministic
relations are RMS <= max absolute residual <= sqrt(N)*RMS. A factor 3.5 is
neither a universal bound nor a fixed-confidence maximum. Even under an
additional independent Gaussian model, the probability that all N errors
lie within k sigma is `(2 Phi(k)-1)^N`, which depends strongly on N; fitted
residuals and overlapping scan-plane proposals do not automatically satisfy
that model. Waviness, selection bias, correlation, partial support and changing
population invalidate the direct patch-to-window conversion further.

Retain that table as “fraction of retained proposal-patch RMS proxies above
an illustrative threshold,” not calibrated expected NOT_PLANAR window share.
The actual maximum-residual ledger of each selected flange is the relevant
development observation. No physical noise calibration follows from tightening
delta or from the author's small synthetic-noise examples.

## What the kernel actually computes

The source requires `0 < delta < exclusion < search_margin`
(`backend/cad_guided_measurement.py:86`). The default nominal angular guard
is 8 degrees and the measured angular-coverage minimum is 20 degrees (`:47`,
`:48`). Its nominal arc mask uses that angular guard, the nominal radial cap
and the axial interval (`:319`). It then excludes points within the distance
band of either **fitted scan** plane (`:324`). The V2 change uses the positive
non-material signed halfspace instead of absolute distance
(`backend/cad_guided_measurement_v2.py:518`); on a perfect inner arc the signs
are positive, so the exclusion-angle calculation is identical in both versions.

Projection uses the measured plane-derived axis and basis (`V1:329`), then
`_circle` first rejects too few points or deficient projected rank (`:176`).
It computes the minimum enclosing 2-D strip of the actual projected point
hull (`:148`), not a sagitta from nominal radius. It refuses when half that
width is <= delta plus a numerical guard (`:187`). Only afterward does it
fit a free circle, check Jacobian conditioning and maximum radial residual
(`:190`). Later checks include tangency, measured sector membership, sampled
angular coverage and axial coverage (`:345`).

This ordering matters. Analytic nominal feasibility is only a preflight bound.
It is never a replacement for the actual strip computation, free fit, measured
support or the retained reason when a finite sample has insufficient rank.

## Derivation: exclusion, guard and usable arc

Assume a single clean circular extrusion with true radius r, true sweep
beta in (0,pi), true tangent planes and a context whose nominal center, axis,
start radial and sweep agree with that geometry. All angles here are radians
unless a degree unit is stated. Let theta run from 0 to beta along the arc.
The signed distances to the two inner outward tangent planes are:

    d0(theta) = r (1-cos(theta)),
    d1(theta) = r (1-cos(beta-theta)).

Both are nonnegative, for either signed sweep orientation. Requiring both to
exceed exclusion e removes the two contact neighborhoods. If
`e >= r(1-cos(beta/2))`, no inner-arc point can pass both strict inequalities.
Otherwise define alpha=acos(1-e/r). The allowed interval is
`alpha < theta < beta-alpha`.

Intersect it with the nominal angular guard `[g,beta-g]`. Let

    m = max(g, alpha),
    U = beta - 2m.

When U<=0 there is no usable arc. When U>0, U is the length of the enclosing
angular interval; its endpoints are not attained where exclusion is the active
strict bound. Thus the diagnosis's `beta-2acos(1-e/r)` formula needs the `max`
with g in general. For the examined radii/exclusions alpha exceeds 8 degrees,
so the simpler formula is correct for their ideal nominal calculation.

For measured angular coverage, actual finite selected samples must span at
least the declared minimum (20 degrees here). Merely having a nonempty U is
not enough; nor does U above 20 guarantee that the sampled coverage is above
20. A prospective robust design should leave positive angular margin beyond
that gate as well as beyond flatness.

If actual radius/sweep, fitted planes or nominal frame differ, use the actual
support geometry to assess eligibility. The interval above is then only an
idealized nominal calculation. In particular, the runtime intersects a nominal
angular window with scan-fitted-plane exclusions; there is no general formula
depending only on the CAD's r and beta for that intersection under arbitrary
pose and geometric deviations.

## Derivation: minimum strip width and finite sampling

For a continuous clean circular arc spanning U<=pi, let h=U/2 and rotate its
mid-angle radial direction to the x-axis. The arc lies between the chord
`x=r cos(h)` and the tangent `x=r`. Their separation is

    W_cont = r (1-cos(h)).

This is the minimum enclosing strip width, not merely one candidate width.
For a candidate strip normal at angle psi from the mid-angle direction,
symmetry allows 0<=psi<=pi/2. If psi<=h, the projection range is
`r(1-cos(h+psi))`, no smaller than W_cont. If psi>=h, the range is
`2r sin(psi) sin(h)`, likewise no smaller. The centered free-offset line
midway through the minimizing strip has maximum orthogonal residual
W_cont/2. Therefore the clean continuum can only pass the source's flat
alternative test when

    H_cont = r/2 (1-cos(U/2)) > delta + G.

G is the actual numerical guard at `V1:187` / `V2:348`, not physical noise.
It is 128 times binary64 epsilon times the larger of 1 and the maximum
absolute centered projected coordinate. This implementation uses ordinary
floating arithmetic, not an interval certificate; near equality, require
explicit arithmetic margin and preserve refusal in the grey zone.

For any finite point subset S of this clean arc, convex-hull containment gives
`W(S) <= W_cont`. Hence if H_cont<=delta+G, **all** such subsets are excluded
by flatness or an earlier gate. Conversely, H_cont>delta+G does not guarantee
any particular finite subset passes. Samples can lack endpoints/mid-angle,
be clustered, or occupy only two angular columns. Repeating the same columns
at many axial stations increases raw N but does not add 2-D arc curvature.

An independent finite oracle can be constructed from actual angular extrema
and a known midpoint sample: the triangle with arc endpoints at +/-h and its
midpoint has minimum width equal to the same sagitta. More generally a known
noncollinear triangle contained in the sample provides a lower bound on the
sample's minimum width via its least altitude. This can establish a genuine
positive finite witness without reusing the implementation's hull routine.
Using the continuous sagitta alone as a finite-data lower bound is invalid.

The kernel also requires >=12 projected rows, nonzero second projected
singular value, no more than 2048 hull vertices, an admissible free-circle
solution and acceptable Jacobian condition. The min-width gate alone does not
establish those properties or the accuracy of a radius estimate from a short
arc.

## Checked nominal arithmetic

These are continuous ideal-model values; rounding is for display. Rows marked
e=delta are limiting calculations only and violate the actual strict policy.

| r mm | delta mm | e mm | Usable angle degrees | Strip half-width mm | Half-width minus delta mm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.78 | 0.05 | 0.075 | 39.336650 | 0.022754 | -0.027246 |
| 0.80 | 0.05 | 0.075 | 39.984334 | 0.024104 | -0.025896 |
| 0.78 | 0.05 | 0.050, limit | 48.747238 | 0.034759 | -0.015241 |
| 0.80 | 0.05 | 0.050, limit | 49.271730 | 0.036410 | -0.013590 |
| 0.78 | 0.03 | 0.040 | 53.142661 | 0.041193 | +0.011193 |
| 0.80 | 0.03 | 0.040 | 53.610255 | 0.042982 | +0.012982 |
| 0.78 | 0.035 | 0.040 | 53.142661 | 0.041193 | +0.006193 |
| 0.80 | 0.035 | 0.040 | 53.610255 | 0.042982 | +0.007982 |

At fixed r, beta and delta, increasing e shrinks the selected arc and reduces
H_cont. Therefore the e down to delta limit is the supremum under the strict
delta<e policy, and the two negative limiting rows prove the stated clean
nominal incompatibility for **any allowed e** at delta=0.05. The angular
guard could only shrink this ideal remainder further.

For delta=0.03/e=0.04, the available continuous arc length is approximately
0.723–0.749 mm across r=0.78–0.8. A nominal 0.3 mm spacing is therefore only
about 2.41–2.50 spacing intervals across it. Sampling phase and angular/mesh
construction can dominate whether enough useful angular stations remain.
The author's sparse-fixture pass/fail grid must be preserved as such, not
reinterpreted as a continuous analytic threshold or expected physical yield.

## Noise, finite-sample margin and legitimate deviations

For fixed selected sample identities and a fixed orthonormal projection,
suppose every perturbed point lies within Euclidean distance eta of its clean
counterpart. Each directional projection range changes by at most 2 eta,
so taking the minimum over directions gives:

    |W_noisy - W_clean| <= 2 eta.

A sufficient worst-case nonflatness margin is therefore
`W_clean/2 - eta > delta + G`, using the **finite** clean support width.
At the proposed values, even the continuum only has 0.0112–0.0130 mm of
headroom. This is not a bound on allowable Gaussian sigma: eta is a uniform
pointwise bound in the stated comparison, and the selected finite support can
have much less margin. Changes to membership near the plane exclusion or
nominal angular boundary, and uncertainty in the fitted projection axis, need
their own bounds; the fixed-row inequality must not be applied after arbitrary
rows have been added/dropped.

Noise can also increase measured width. This is another reason clean nominal
infeasibility does not imply universal physical-output impossibility. Passing
the flat-witness test means the observed set is not compatible with one line
at the declared residual, not that the latent part's true radius is accurately
identified. Keep independently measured radius/sweep errors and calibrated
uncertainty absent unless separately established.

Radius increase, sweep increase, altered center and changed plane normals can
change both arc span and strip width, while the nominal angular mask can clip
those changes asymmetrically. The engine must continue to measure actual
deviations within its declared search scope, not shrink data toward nominal
until this feasibility calculation looks favorable. Outer skin/adjacent
surface identity remains a separate issue for inner-specific measurement and
material claims. Per the parent's independently checked scorer scope, the
frozen contact score evaluates generic conditional contact generators, not
inner arcs. Unknown material side is therefore not by itself a prohibition on
that explicitly generic diagnostic score, provided the unchanged scorer's
actual eligibility/binding contract is met and unknown identity remains
visible. A generic score must not be promoted into verified inner geometry,
observed manufacturing count, calibrated deviation or conformance.

## A generic admissibility rule, not a part-identity rule

For a separately declared ideal circular-extrusion model with radius r,
sweep beta, angular guard g, residual delta and exclusion e:

1. Require the existing policy ordering `0<delta<e<search_margin` and a
   supported sweep/model; do not clip invalid acos inputs into an apparent
   certificate.
2. Test `e < r(1-cos(beta/2))`. Otherwise the ideal two-plane exclusion leaves
   no inner-arc point.
3. Compute m=max(g,acos(1-e/r)), U=beta-2m. Require a positive margin beyond
   the intended sampled angular-coverage threshold.
4. Treat `r/2*(1-cos(U/2)) <= delta+G` as ideal-model infeasibility. The reverse
   inequality means only that finite measurement is possible, not established.
5. For a positive finite qualification, require an independently supported
   lower bound on the **sample** strip width, source row/angular coverage,
   rank, circle conditioning and declared measurement error limits. Then run
   the unchanged runtime checks on the actual selected raw rows.

Equivalently, in the valid monotone range a necessary continuous nonflatness
inequality is:

    beta/2 > max(g, acos(1-e/r)) + acos(1-2(delta+G)/r).

It displays the budget competition directly: contact exclusion consumes one
angular margin and distinguishing curvature from a free line consumes another.
In dimensionless terms the decisive ratios are delta/r and e/r, with g and
beta fixed. The older kernel test scale choices and the cohort's absolute
policy values were not geometrically equivalent.

Use nominal values only for a labeled preflight/engineering study. Do not
substitute this nominal test for actual inference or observed counts; do not
adjust delta per observed fit/Gold outcome to obtain a pass. A prospective
geometry-range policy may be defensible if its domain and evidence are
declared before evaluation, including all cases outside or failing it.

## Is tightening delta defensible?

Yes, as a **new, explicitly uncalibrated development sample-model policy**,
under the standing authority and a separately frozen independently audited
successor. It fixes a mathematically demonstrated exact-nominal incompatibility
without changing the external scorer. It does not qualify measurement truth
or make stricter fit residual synonymous with more accurate radius.

The coupled deltas must be recorded accurately:

- delta 0.05->0.03 tightens both plane and circle all-points residual limits;
- the same change relaxes the minor RMS flange-spread requirement
  0.5->0.3 mm and relaxes rejection against a free-offset flat alternative;
- e 0.075->0.04 admits more near-contact arc rows, with potential new material
  contamination/plane-transition ambiguity;
- the scalar tangency and other gates stay unchanged, not the whole acceptance
  behavior or row-selection population.

This is a combined measurement policy change, not simply “stricter quality.”
Report it separately from layer selection and triplet completion, retaining
component ablations on independently constructed cases where useful. The
author's current noisy examples (including radius 0.789 for a nominal 0.8
two-skin fixture) need declared error criteria, not just a MEASURED outcome.
Missing physical truth and uncalibrated uncertainty remain separate limitations.

## Proportionate independent test matrix

The following are proposed source/synthetic acceptance checks, not executed
by this reviewer. Freeze exact construction, budgets and pass/refusal/error
criteria first; do not rerun terminal physical attempts to obtain them.

- **Analytic boundaries:** radii spanning below/at/above the declared domain,
  several sweeps including both signs, and cases where angular guard or plane
  exclusion separately dominates. Check empty-sector, angular-coverage and
  flatness boundaries with strict/equality cases and numerical margins.
- **Finite sampling:** regular angle grids and arclength grids with multiple
  phases, sparse two/three-column cases, many axial repetitions, missing
  mid-angle/end samples, clustered coverage and partial scans. Keep all
  failures; use triangle/covariance oracles independent of the kernel hull.
- **Policy contrast:** frozen 0.05/0.075, proposed 0.03/0.04, and the retained
  0.035/0.04 diagnostic comparison on the same fixed synthetic matrix. Preserve
  the fact that previous grid feedback was consumed development, not a new
  confirmation. No post-run policy/spacing/seed change.
- **Declared noise/model classes:** bounded Euclidean perturbations, separate
  explicitly stochastic models, coherent warp, outliers and axial changes.
  Distinguish uniform eta from sigma. Check both false nonflatness and missed
  curvature, then radius/sweep error and uncertainty/abstention semantics.
- **Actual-versus-nominal geometry:** radius and sweep deviations, center and
  local axis changes, correct versus perturbed context, and the nominal0.8 /
  actual1.5 radius counterexample. Measurements must follow scan geometry;
  nominal infeasibility must not be misreported as impossible physical truth.
- **Material-side interaction:** clean inner-only, two skins, outer-only and
  adjacent curves. Preserve unresolved side identity and distinguish generic
  conditional-contact scoring from inner-specific measurement. A fitted outer
  radius must not become a claimed inner radius, count or conformance result.
- **Connected path:** native finite window, context crop, plane fits, actual
  strip diagnostic, free-circle fit, canonical refit, support cover and unchanged
  carrier/scorer interface. Retain all attempted windows and explain every
  failure; a local perfect-arc pass alone is not a Gold11 baseline.

Small direct analytical cases should precede broad noisy matrices. The
immediate promotion decision is whether the new fixed candidate is coherent,
tested against its failure modes and honestly versioned—not whether every
physical case is known to pass before another authorized run.

## Handoff

The nominal feasibility problem is a real and useful engineering finding.
Accept it with the restricted theorem above; correct the universal physical
claims and the patch-RMS yield forecast. The proposed 0.03/0.04 policy is
mathematically plausible for declared development, with limited finite/noise
margin and independent material-side risks still open. Only exact-source
testing and the separately audited complete execution can establish its actual
effect. This report does not authorize an execution, claim new F1, certify
historical comparability, or qualify factory-floor release.
