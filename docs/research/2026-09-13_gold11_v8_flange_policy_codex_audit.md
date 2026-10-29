# Gold11 V8 flange policy: independent source-only mathematics audit

Date: 2026-09-13. Reviewer: Codex/Astra subagent
`v8_flange_policy_math_audit`. Scope: source and the explicitly released V8
result narrative only. No source import, test execution, geometry calculation
job, raw input, Gold, attempt artifact or private-cache access occurred. The
reviewer owns only this new report; Fable owns implementation and its result
narrative. V8 and all predecessors remain terminal and unchanged.

## Verdict

The local flange gate is dimensionally consistent, but the current report
overstates the physical cause of its rejection. A global scan-to-CAD p95 of
0.70–0.90 mm does **not** establish local flange noise, nonplanarity, bad pose,
or the reason for a weak-normal failure. The kernel centers and fits the raw
scan subset; for fixed rows, its singular values and fitted-plane residuals
are rigid-transform invariant.

There is a real source-level admission mismatch: the native adapter can emit
very short, boundary-certified axial windows, while the measurement kernel
requires at least 0.5 mm minor RMS spread at the frozen 0.05 mm fit policy.
Some geometrically valid narrow windows are therefore impossible to measure
even on perfect planar scan data. The released aggregate does not disclose the
per-flange singular values or actual window spans, so it does not establish
how many V8 rejections this explains. Contamination, missing coverage, and
incorrect context selection remain competing hypotheses.

Do not raise the pose gate again or simply loosen the fit residual to force
output. First retain the missing per-flange decision statistics in a separately
versioned diagnostic path. If short-span support is confirmed, the smallest
plausible geometry improvement is to decouple reusable scan-derived flange
plane context from the narrow exclusive arc window, with explicit finite
support and ownership. This report does not greenlight that implementation or
a new physical execution without its independent gate.

## Findings and priorities

### P1 — Correct the causal claim before using it to choose the next repair

The released narrative at
`docs/research/2026-09-13_gold11_mesh_development_v8_result.md:164` reports 39
window failures with `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY` and four
with `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`. Its wording at lines 170–174
then treats these as an observed false-pose/uncertainty consequence of the
admitted global residuals. That causal inference is not supported by source or
those aggregates. The later recommendation at that document's measurement
blocker should preserve the same distinction.

What is established, subject to the independent physical POST: three admitted
contexts reached the measurement stage, and every native window was refused.
What is not established: that the poses are physically wrong, that global p95
is flange planarity error, or which half of the weak-normal disjunction fired.

Suggested replacement substance: “The three admitted contexts reached local
measurement but produced no eligible supports. All 43 windows failed a local
flange-fit gate. The current first-failure records do not distinguish insufficient
in-plane spread from excessive thickness/residual ratio, nor establish whether
window geometry, coverage, mixed surfaces, or a wrong context caused the
failure. Global proximity p95 is a separate quantity.” This is an evidence
correction, not a request to alter the terminal score or its artifacts.

### P2 — Topological window validity does not imply fit-policy feasibility

The V8 configuration fixes a flange ray interval [2.5, 12.5] mm, radial search
margin 3 mm, fit residual delta = 0.05 mm, plane exclusion 0.075 mm, and free
circle fitting (`scripts/run_gold11_mesh_development_v8.py:211`). Native V1
subtracts complete boundary enclosures from common axial spans, retaining
intervals longer than twice an 8e-6 mm numerical boundary guard
(`backend/cad_guided_native_features_v1.py:36`, `:97`, `:113`). It checks that
root-to-search strips lie inside the nominal flanges and outside boundary
crossings (`:647`, `:665`, `:674`) and passes the resulting span unchanged into
`CADGuidedFeature` (`:692`). There is no condition there tying axial width or
actual scan spread to delta.

This is not a false nominal-topology certificate: the emitted object is only
a search ROI, not a promise of measurability. But it is an important connected
policy mismatch that can explain why many valid ROIs still produce no output.
Increasing scan density alone cannot overcome the normalized spread threshold
for the same spatial distribution.

### P2 — First-failure diagnostics do not identify the failed flange or mechanism

`backend/cad_guided_measurement.py:123` uses one reason for two independent
conditions. `_measure` evaluates the flange list in order and stops on the first
exception (`:307`). The context consumer retains only that reason string
(`backend/cad_guided_context_measurement_v1.py:300`). It does not publish the
failed side, N, singular values, residuals, effective coordinate extents, or
the booleans for both branches. Thus 39 failing **windows** do not mean 39
measured noisy flanges, and the second flange need not have been evaluated.
The four planarity failures did reach the maximum-residual check, but still
do not distinguish surface noise from mixed support or model violation.

## Connected call path and units

The exact path in the V8 source is:

1. The runner declares `measurement` separately from registration and the
   stipulated initialization position bound (`run_gold11_mesh_development_v8.py:203`,
   `:211`, `:218`). `NativeGuidedPolicy` validates those values through the same
   feature constructor (`backend/cad_guided_native_features.py:31`).
2. V15 extracts native features with that policy
   (`backend/cad_guided_mesh_offline_v15.py:395`), then binds finite registration
   contexts and calls support-cover measurement on the original raw rows
   (`:488`). The V15 units are explicitly provisional common millimetres, not
   calibrated units (`:283`); the position bound is explicitly outside measured
   uncertainty propagation (`:311`).
3. Support-cover V2 first calls the existing context measurement consumer
   (`backend/cad_guided_support_cover_measurement_v2.py:144`). It cannot build
   a later cover from windows that never produced support records (`:154`).
4. Given p_CAD = R p_scan + t, the feature is pulled into the raw frame with
   origin R^T(o_CAD - t), axis R^T a, and radial R^T n
   (`backend/cad_guided_context_measurement_v1.py:62`). The rigid transform is
   checked for finite SE(3) structure, not accepted metrology
   (`backend/cad_guided_mesh_offline_v2.py:126`).
5. A conservative raw-row crop includes all possible kernel masks without
   downsampling (`backend/cad_guided_local_measurement_v1.py:36`). Cap-like
   exclusion happens before fitting (`cad_guided_context_measurement_v1.py:265`,
   `:285`). The actual kernel flange selection is axial-in-span, ray-in-window,
   and absolute nominal radial distance <= 3 mm
   (`backend/cad_guided_measurement.py:292`).
6. Two independent scan planes determine the measured axis and sweep
   (`cad_guided_measurement.py:307`); scan arc positions determine the free
   circle (`:329`). Canonical refitting likewise uses fixed raw rows and scan
   planes/circle, not nominal answers
   (`cad_guided_context_measurement_v1.py:69`). Preserve this separation.

All positional coordinates and deltas here have declared mm units. Plane
normals and ratios are dimensionless. Neither the 3 mm acquisition search
margin nor the 1 mm global admission threshold is a manufacturing tolerance.

## Critical mathematics

### Centered SVD and normalization

For N selected raw points, let A have rows p_i - mean(p), and let
s1 >= s2 >= s3 >= 0 be its singular values. The source computes this directly
at `backend/cad_guided_measurement.py:119`. Define alpha_i = s_i/sqrt(N), in mm.
Then alpha_i squared are the eigenvalues of the population covariance
A^T A/N, not the unbiased covariance with denominator N-1.

After N >= 12 and the rank/spread check, the conditions for continuing are:

- s2/s1 >= 0.01 (the feature default at `:52`, used at `:121`);
- alpha_2 > 10 delta = 0.5 mm, strictly;
- s3/s2 < 0.2, strictly;
- the fitted unoriented normal, oriented toward the supplied reference, is
  within 15 degrees of that reference (`:49`, `:126`);
- max_i |A_i dot fitted_normal| <= delta = 0.05 mm (`:131`).

The final residual vector has Euclidean norm s3, so its RMS is alpha_3. If the
maximum-residual condition holds, alpha_3 <= delta. Together with alpha_2 >
10 delta this implies s3/s2 < 0.1. Therefore the 0.2 ratio branch is redundant
for any eventual full pass, but it can trigger an earlier, less informative
reason before the actual maximum residual is evaluated. It is not legitimate
to infer delta-sized residuals from the shared early reason.

Replicating every row equally multiplies s_i by sqrt(replications) and leaves
alpha_i and all ratios unchanged. More point count is not a substitute for
two-dimensional spatial support. Nonuniform resampling can change weighting,
so this is not invariance to arbitrary sampling changes.

### A defensible interpretation of the factor 10, without calibration claims

Let n_hat be the TLS normal. Suppose some alternative unit plane normal n,
with a free plane offset, fits **every selected observed point** within delta.
Centering minimizes the sum of squared signed distances for this normal, so
||A n|| <= sqrt(N) delta. Expansion in the right-singular basis yields
||A n|| squared >= s2 squared sin(theta) squared, where theta is the acute
angle between n and n_hat. Consequently:

    sin(theta) <= delta / alpha_2.

The existing alpha_2 > 10 delta thus bounds every such sample-compatible
plane normal to an angle smaller than asin(0.1), approximately 5.74 degrees,
from TLS. This is a derived sufficient sample-model statement, not a proof
that the author intended this interpretation and not calibrated uncertainty.
One could apply the same inequality to a true latent plane only under an
independently justified error/model assumption. The observed fit residual is
not itself a scanner error bound, and the V15 position declaration explicitly
does not qualify measurement uncertainty.

This also explains a counterintuitive interaction: increasing delta to 0.1 mm
would require alpha_2 > 1 mm. It relaxes planarity but **tightens** the dominant
spread requirement. Lowering delta eases spread but tightens maximum residual.
Neither scalar adjustment is a justified cure for unknown failure causes.

### Finite-window feasibility bounds

Consider a clean nominally aligned planar flange patch in a rectangle with
axial length L and ray width W. In V8, W <= 10 mm after sampling/selection.
For any point distribution supported in a finite interval of width d, the
population variance is <= d squared / 4: after centering, this follows from
(x-low)(high-x) >= 0 and expansion of its mean. Applying this to the two
orthogonal in-plane coordinates gives:

    alpha_2 <= min(L, W)/2.

Thus a necessary condition to pass the current spread gate in this clean,
aligned model is min(L,W) > 20 delta = 1 mm. A perfect rectangle with L <=
1 mm is impossible to admit under that model, however many raw rows it has.
The bound is necessary, not sufficient: clustered/line-like scan coverage can
have much smaller alpha_2.

For the particular idealized distribution that is uniform independently in
the two rectangle coordinates, alpha_2 = min(L,W)/sqrt(12). Its threshold is
min(L,W) > 10 sqrt(12) delta, approximately 1.732 mm. This is a model example,
not an estimate of any Gold11 sample distribution. Finite regular grids have
their actual discrete variance, not exactly the continuous uniform value.

A more conservative necessary condition also accounts for the permitted normal
tilt and nonzero fit residual. Let a be the nominal axis; let n_hat make an
angle no greater than theta_max=15 degrees to the nominal radial normal.
Project a into the fitted plane: e=(a-(a dot n_hat)n_hat)/gamma, where
gamma >= cos(theta_max). For centered points, the total e-coordinate range
is at most (L + 2 delta sin(theta_max))/cos(theta_max), because fitted-normal
residuals range over at most 2 delta. As e is in the fitted two-dimensional
principal plane, alpha_2 squared <= variance along e. Combining the range
bound with alpha_2 > 10 delta yields the necessary condition:

    L > 20 delta cos(theta_max) - 2 delta sin(theta_max).

At the frozen values this is approximately 0.940 mm, apart from the explicitly
tiny arithmetic guards at axial boundaries. Hence sufficiently short windows
cannot pass even allowing the accepted tilt and residual. This does not prove
that the released V8 windows actually have these widths. Their individual
spans and actual scan distributions are absent from the allowed aggregate.

The native adapter admits spans far below these thresholds because its guards
are numerical topology guards, not fit-support gates. It does not assert that
a narrow nominal cylinder interval supplies enough independent plane context.

### Why global proximity p95 does not diagnose local plane fitting

V14 `_context_support` transforms the selected registration rows, queries
finite CAD triangles, chooses definite proximity rows within the 2 mm
correspondence threshold, then computes the p95 of those distance intervals
(`backend/cad_guided_plane_initialization_v14.py:744`). It is distance-to-CAD
proximity, not the local TLS flange residual and not a pose certificate
(`:768`). V4 computes a default-linear 95th quantile with outward arithmetic
(`backend/cad_guided_global_registration_v4.py:309`). Its population is not
the full raw flange subset.

For a rigid transform p' = R p + t, centered row matrices satisfy A'=A R^T.
Hence A'^T A' is an orthogonal similarity of A^T A: singular values are
unchanged, and the fitted plane/residuals transform equivariantly. This is
exact algebra; floating arithmetic near thresholds needs finite tolerance,
and CAD-oriented sign/orientation checks require the reference to be transformed
consistently. It does not claim invariance of selected row identities when
the context changes.

Counterexamples suffice to reject the causal shortcut. A perfectly planar
patch translated away from its nominal CAD plane has nonzero CAD distance but
zero local fit residual; a perfect, very narrow patch at the correct CAD pose
has zero global distance yet fails the spread gate. Conversely, a small noisy
or mixed local region can fail its all-points maximum residual while leaving
a whole-part 95th percentile low. A wrong context can matter **indirectly** by
selecting different rows, missing part of a flange, or changing reference
orientation; that mechanism must be demonstrated separately.

## Competing causes and discriminating evidence

1. **Insufficient spread from trim or partial coverage.** Retain the nominal
   span and each selected flange's actual axial/ray extents, N, alpha_1/2/3,
   rank ratio, normal reference angle, and maximum/RMS residual. A small
   alpha_2 with tiny residual distinguishes this from ordinary roughness.
2. **Mixed skins, neighboring geometry, or curved transition contamination.**
   The mask admits all positions in a +/-3 mm nominal radial slab; it does not
   assign inner skin from material identity (`cad_guided_measurement.py:300`).
   The upstream role filter only excludes unconflicted cap-like triangles;
   unknown/noncap rows remain (`cad_guided_context_surface_roles_v1.py:15`,
   `:351`). Two parallel skins closer than the slab width can therefore both
   enter in an invented example. That is a possible failure mechanism, not a
   claim about these physical inputs. Diagnose signed residual structure,
   topology-component membership and retained raw-row provenance; do not
   silently trim the inconvenient population.
3. **Wrong or fragile context.** Keep separate fixed-row fit diagnostics and
   context-dependent mask diagnostics. If only the context changes, a change
   of fit indicates changed row selection/reference checks, not that rotation
   itself created nonplanarity. Preserve every context disposition, including
   ambiguous alternatives; do not select whichever fits Gold.
4. **Actual scan noise/nonplanarity or unsupported local geometry.** A full
   supported patch can exceed delta despite correct registration. Without
   physical truth or calibration, retain this as a declared-model refusal,
   not as a measured defect or evidence that the true flange is out of spec.

The diagnostic schema should evaluate both flange sides and report independent
gate booleans even when the legacy primary reason would stop early. Retain
nonfinite/rank failures without attempting undefined divisions. Diagnostics
must be bounded, byte-bound to selected raw IDs and policy, and not turn
failed fits into carriers or measurements. Do not retrofit records into the
terminal V8 attempt; an authorized new diagnostic needs its own frozen scope.

## Adversarial synthetic matrix before any mechanism promotion

These are proposals, not tests executed by this reviewer. Freeze independent
construction parameters, budgets, outputs and error limits before execution.

- Analytic planar rectangles around the derived width thresholds, including
  line-like and clustered sampling at a fixed rectangle extent. Check the
  observed alpha_2 calculation against an independently calculated covariance.
  Keep both exact boundary cases and comfortable pass/fail controls.
- Uniform row replication versus nonuniform density, translations and proper
  rotations with consistently transformed reference frames. Verify expected
  normalized-spread/plane-residual invariance and sign conventions.
- A broad perfect plane offset from its CAD counterpart, and a narrow perfect
  plane exactly on CAD. Demonstrate explicitly that global residual and local
  weakness are distinct. Include a small contaminated local population within
  an otherwise exact whole part to test the quantile/max mismatch.
- Independent bounded noise, coherent tilt, smooth warp, an isolated extreme
  residual, two parallel skins, adjacent bends, and an intruding curved patch.
  Preserve all populations/abstentions; verify no CAD nominal substitution.
- A native flange split into short certified bend windows by holes/trim bands,
  with a truly broad common scan plane available elsewhere, plus the negative
  counterpart where the apparent neighboring context is another tilted face.
  Any reusable-plane proposal must improve the former without bridging curved
  evidence, holes, missing bends, or unrelated faces in the latter.
- Full native-feature -> context crop -> per-flange fit -> canonical fit ->
  support-cover -> unchanged carrier/scorer interface. An isolated `_plane`
  pass is insufficient; check source-row ownership and every refusal after it.

Existing test scale choices do not cover this distribution shift: the kernel
geometry matrix gives span 6r, flange width 2r and delta .01r
(`tests/test_cad_guided_measurement_geometry.py:65`); the 8192-case stress uses
the same ratios (`tests/test_cad_guided_measurement_stress.py:52`, `:75`). The
context fixture uses a 12 mm span, broad flange samples and delta .05 mm
(`tests/test_cad_guided_context_measurement_v1.py:65`). Those source fixtures
are useful but do not establish the short-native-span regime above. No current
or historical test result was re-executed or inferred from source existence.

## Smallest justified successor and stop conditions

The immediate justified successor is diagnostic, with no scoring-rule change
and no new accepted measurements: expose the already-computed plane spectrum,
both independent weak-gate conditions and support geometry before throwing.
This closes a specific causal-evidence hole rather than adding a broad
readiness phase. Use new versioned sources because the predecessor is frozen.

If diagnostics show that short axial ROI support dominates, investigate a new
measurement-context mechanism that estimates a flange plane from a larger,
explicitly justified **scan** support set, reusable across its finite bend
windows. Keep the curved-support population exclusive and bounded to its
original physical samples. Native face adjacency can propose which plane
context to search; it cannot supply the plane normal, radius, deviation or
observed count. Validate common plane compatibility, ambiguity and source IDs
before reusing context; do not concatenate incompatible patches just to increase
alpha_2. The current cover stage runs too late to repair failed initial plane
fits, so this would belong before `_measure` support admission, not as a
post-hoc scorer rescue.

If diagnostics instead show full-width mixed or genuinely nonplanar support,
the above expansion is not justified and may worsen contamination. A different
explicit role-assignment/model mechanism would need its own independent
synthetic discrimination. If measurements remain unavailable, preserve that
negative outcome rather than accepting CAD values or redefining F1.

This review makes no numeric F1 claim, no historical comparability claim, no
physical measurement accuracy claim, and no factory-floor release decision.
Acceptance check for this deliverable: source-cited algebra and connected-path
inspection only. The parent may use it to correct the narrative and scope the
next separately versioned investigation; engine implementation remains Fable's.
