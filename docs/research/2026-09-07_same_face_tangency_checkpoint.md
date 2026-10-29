# Competing flange explanations: a bounded same-face tangency check

Date: 2026-09-07. Continuation of the
[locked engine-first roadmap](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07)
and [adjacent-bend seed checkpoint](2026-09-07_adjacent_bend_seed_checkpoint.md).
Source and invented geometry only; no runtime promotion or physical access.

Continuation: the [bounded stream/window checkpoint](2026-09-07_streamed_window_checkpoint.md)
adds large-cloud ingestion into declared LOCAL windows, with a nonempty synthetic
resource test. It does not qualify dense full-scan discovery or change this
checkpoint's radius/tangency results.

## Outcome and model boundary

The additional check identifies a concrete relation failure that independent
plane and radius fits cannot detect. A plane through two sampled cylinder
generators is a valid sample plane but a secant, not a tangent flange. Pairing
that plane with a real flange can still give the correct circle radius and a
conditional-positive radius result on a partial arc.

The fixed first attribution finds, per exact/noisy matrix:

| Development matrix | Conditional single-radius candidates | Complete observed-support single-face controls retained | Same-face tangent explanations excluded | Conditional two-radius candidates left unjudged |
| --- | ---: | ---: | ---: | ---: |
| Prior 150 clouds | 102 | 90 | 12 | 60 |
| Adjacent 72 clouds / 144 events | 156 | 120 | 36 | 0 |

All excluded explanations are unmatched construction-support alternatives; no
complete-support single-face control is excluded. There are no unavailable
diagnostics in these matrices. Exact and bounded-noise counts agree. Across both
noise states, 420 matched single-face controls remain not excluded and 96 extra
single-face explanations are excluded, out of 516 single-radius assessments.

This is NOT 100% precision, physical accuracy, or proof that the remaining
explanations are tangent. The original 294/264 associations per matrix, including
all 144 unmatched associations in each, remain intact. Their original radius
intervals/statuses are not overwritten. The new finding concerns an ADDITIONAL
single-same-face tangency relation, not whether any cylinder, multiface object,
formed bend, manufacturing event or defect exists.

## Mechanism and research grounding

The old bridge hypothesizes that two ideal planes and a cylinder share an axis
direction. It propagates axis uncertainty into a rotated/error-relaxed radius
problem. It does not require that either plane be tangent to the cylinder.
Its fitted center is deliberately not exported as a verified physical axis.
The new check neither substitutes that fitted center for truth nor snaps the
geometry to an ideal relationship.

[Al-Sharadqah and Chernov, sections 2 and 4](https://arxiv.org/html/0907.0421v1)
distinguish geometric and algebraic circle fitting, explain the squared-distance
linearization and discuss algebraic bias on incomplete arcs. Their analysis uses
an isotropic Gaussian statistical model. Here, a similar algebraic calculation
is only a CENTER PROPOSAL: a deterministic outer ball includes every admissible
center under the stipulated simultaneous bounded errors. It is not a new claim
that an algebraic estimate is accurate, optimal, unbiased or calibrated.

[GlobFit](https://graphics.stanford.edu/~niloy/research/globFit/globFit_sigg11.html)
couples local primitive fitting with global geometric relationships for engineered
objects. Our inference for QC is to test a relationship without enforcing it and
thereby erasing a possible deviation. Neither paper validates this implementation,
its invented-case result, or a physical production use. The inequalities below
are directly derived here and independently reviewed.

## All-point center enclosure

Assume a true cylinder with direction in the existing plane-derived axis cone.
Rotate that hypothetical cylinder AND its hypothetical flange planes by the SAME
rotation about the observed cylinder pivot o, aligning the true axis to the
working axis a0. The existing bridge bounds the resulting cylinder-sample
displacement by E. Project into a0's perpendicular plane. Projection is
nonexpansive, so every true circular sample is within E of its observed projected
sample qi. Let its true center be c and its radius lie in the existing necessary
interval `[Rlo,Rhi]`.

For every sample, the radial-distance error is at most E. Therefore

```text
delta_i = ||q_i-c||^2 - R^2
|delta_i| <= D = 2*Rhi*E + E^2
```

Let H center vectors by subtracting their mean, `A=2*Hq`, and
`b=H(||q||^2)`. Expanding squared distances and centering cancels the constant
`||c||^2-R^2`, so `A*c=b-H*delta`. For ANY computed center proposal c_hat with
residual `r=A*c_hat-b`,

```text
||A*(c-c_hat)|| <= sqrt(N)*D + ||r||
||c-c_hat|| <= eta = (sqrt(N)*D + ||r||)/sigma_min(A)
```

The first bound uses `||H||_2=1`, not independent errors or averaging away a
systematic error. The second requires two supported independent projected
directions. All cylinder points participate; there is no selected triangle,
favorable subset, circle refit at a looser measurement tolerance, or radius
replacement. Rhi, E, A and b share one normalized coordinate scale.

Unbounded/nonpositive radius, nonconditional original radius result, unresolved
rank, solver failure or unsupported numeric range gives unavailable/abstained,
not exclusion. The implementation includes residual, singular-value and
coordinate arithmetic allowances, but these are engineering guards, NOT
directed-rounded numerical certification. The center is reported only in a
rotated projected diagnostic frame, never as a physical marker or verified axis.

## Necessary centroid test under the SAME rotation

For each flange, let m be the centroid of ALL its distinct observed points.
The corresponding ideal centroid is in its stipulated ideal plane and is within
epsilon of m: correlated errors forbid dividing epsilon by sqrt(N).
If delta_axis is the direction-cone angle and `chord=2*sin(delta_axis/2)`, rotating
about o adds at most

```text
E_m = epsilon + chord*(||m-o|| + epsilon)
```

This uses the flange centroid's full THREE-DIMENSIONAL lever arm. Reusing only
the cylinder footprint or projected distance can understate uncertainty when
the flange is far away along the axis.

A plane tangent to a cylinder projects to a line tangent to its circular disk.
Every point on that line, including the projected ideal centroid, has distance
at least R from c. Consequently,

```text
||P(m-o)-c_hat|| + eta + E_m + arithmetic_guard < Rlo
```

excludes the same-face tangent model. The strict inequality preserves boundary
uncertainty. It requires no sampled contact endpoints and does not fill an
unobserved gap. Testing only a centroid is incomplete: a distant centroid can
hide a secant plane, and a detached circle can remain unexcluded. Passing is not
joint plane/cylinder compatibility, finite contact or physical evidence.

## Counterexamples that prevent overclaim

The independent controls include radius-5 circles with planes `x=3` and `z=3`:
these are valid planes and a valid cylinder, but centered flange samples project
inside the disk, excluding SAME-FACE tangency. A partial arc on one side of both
planes defeats a simple straddling-only test but not the center enclosure.

Actual tangent planes `x=5,z=5` remain unexcluded without contact samples.
Rigid transformations, unit changes, bounded noise and correlated-error/long-
lever-arm constructions test the uncertainty propagation. Direct all-point center
tests compare the outer ball with independently known circle centers, not the
implementation's own fitted-center truth.

An outer radius-6 face paired with a flange of an inner radius-5 face may fail
this SAME-FACE model while describing a valid multiface part. The result explicitly
keeps `two_radius_or_multiple_surface_model_is_excluded=False`. It never rejects
the whole candidate or labels that geometry defective. Likewise, no finite point
sample establishes the unsampled material between observations.

## Additive integration, attribution and budgets

New sources:

- [Standalone mathematical diagnostic](../../research/geometry/same_face_tangency.py).
- [Additive candidate audit](../../research/geometry/tangency_audited_bend_candidates.py).

The audit preserves the entire cap-supported result under `base_analysis` and
attaches candidate-ID-bound assessments only to conditional SINGLE-radius
candidates. It reuses their internal radius result: no additional radius fits,
new radius policies, graph edits, support trimming, reassignment or two-radius
rejection. Existing discovery/radius sources stay unchanged.

The extra center budget is explicit, at most 256 evaluations, with one all-point
two-column least-squares solve and two flange-centroid checks per single-radius
association. All old raw/query/fit/region/pair/association/split caps remain.
The evaluation counter records assessment attempts, an upper bound on solver
calls when an assessment stops before solving; this scope is explicit in output.
Budget preflight or unavailable analysis clears ALL audit assessments and marks
the audit incomplete; preserved base evidence remains clearly separate. This is
bounded research computation, not a scanner-scale path or a physical release.

The [fixed attribution](../../tests/test_same_face_tangency_attribution.py) reuses
the original 300 prior and 144 adjacent clouds, including exact RNG order and
nonzero bounded perturbations. Construction row identities are assertion-only.
All candidates, interval denominators, full/partial supports and abstentions are
retained. Earlier matrices are consumed development constructions, not held-outs.

Per noise state, all 12 prior excluded alternatives pair a complete flange with
two-generator arc-only secant samples and a partial observed transition. Among
the 36 adjacent alternatives, 24 have that composition and 12 pair a complete
flange with mixed construction support. Composition labels describe invented
sample ownership only; they do not establish real physical face identity.

The result does not recover the 24 adjacent events lacking an observed seam link.
It narrows one interpretation of existing candidates; it does not create missing
observations. No search or error threshold was tuned after reading these results.

## Verification and continuation

Root's final isolated combined run passes **86 checks in 23.61 s**, peak RSS
**76,988,416 bytes (73.4 MiB)**, in one process with single-thread BLAS:

- [70 independent mathematical/scope controls](../../tests/test_same_face_tangency.py).
- [Four fixed attribution and paired-wrapper checks](../../tests/test_same_face_tangency_attribution.py).
- [12 additive integration/budget/identity checks](../../tests/test_tangency_audited_bend_candidates.py).

The paired matrix performs 444 direct cap-supported passes, 444 wrapper-internal
cap-supported passes and 516 standalone radius recomputations: 1,404 top-level
analyzer/diagnostic calls on 444 observed clouds, not independent parts. The
wrapper performs 516 center-assessment attempts but zero additional radius fits.
For every cloud, `wrapper.base_analysis == baseline` exactly, and every wrapper
assessment equals the standalone diagnostic for the same candidate. No prior
200-check or older full geometry suite is represented as rerun here.

```bash
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider \
  tests/test_same_face_tangency.py tests/test_same_face_tangency_attribution.py \
  tests/test_tangency_audited_bend_candidates.py -q
```

The first 15-check run passed all three attribution checks and 11 wrapper
checks; one wrapper assertion incorrectly expected the default missing-arc fixture
to contain a conditional alias. That 90-degree/extent-2 fixture has none. The
fixture is retained as a zero-exclusion control, with no geometry/policy change.

Independent mathematical and source reviews found no blocking issue. Their two
provenance clarifications are included before the final passing run: carry the
original necessary-radius interval scope, and identify assessment counts as
upper-bound workload rather than invariably completed solver calls. AST checks
pass for the five new Python files; scoped whitespace and local-link checks pass.
Named predecessor source hashes remain unchanged. Final new-source SHA-256:

```text
same_face_tangency.py
338eb4b644a2890237b15c259224bd2e3592d02a722c5593ae2a5a541ae7ff17
tangency_audited_bend_candidates.py
46334c4fc304173c90c170780a4987f46a41a1de9852396ff031d8d10f0c13d2
```

Next is a bounded scanner-scale design and nonempty synthetic full-path resource
test, while maintaining the distinction between model-relation exclusion and physical
event mapping under the primary CAD-assisted roadmap. Genuine joint tangency,
multiface assignment, finite contact, scanner calibration and physical validation
remain open. M3 is not complete; no M5/M6, production/UI/service work, physical or
held-out/email access, or control-record access occurred. C15, terminal EXP35,
the locked acceptance gates and the paused goal remain unchanged.
