# Connected C0/C1/C2 array fixture design — 2026-09-10

**Proposed, source-authored only; no execution or new gate authority.** This
packet implements the narrow curved-support proof requested after the
[consultation review](2026-09-10_chat_6_pro_consultation_result.md). It does not
retry V9/V11 terminal attempts or alter their geometry, policies or failures.

## Scope and arms

Six cases, each requested in C0/C1/C2: the original clean/noisy fifty-row planar
attribution controls, plus four genuinely curved raw-mesh cases below. C0 keeps
current V11 consumers; C1 changes proposal proximity only; C2 carries the coherent
stage contracts and quantity-specific context measurement. No result for C0/C1
is assumed. Retain every arm's failure stage and complete evidence. A planar
control cannot satisfy a radius/free-circle positive.

The first packet uses **analytic CAD triangles and explicitly declared nominal
CADGuidedFeature values**, not native CAD extraction or byte ingestion. The
independent CAD triangulation supplies neither scan rows nor scan connectivity.
An array digest is not a scan-file digest. A native/byte connected follow-up and
the existing independent execution/data gates remain necessary before Gold11.

| Curved case | Scan geometry | Axial evidence | Actual coordinate perturbation | Pose |
| --- | --- | --- | --- | --- |
| U-R | R5.30, 90° | Inner sheet cropped to z=[6,24], no cap | zero | A |
| U-A | R5.00, 90.5° | Same crop, no cap | deterministic ≤.002 mm | B |
| L-R | R5.30, 90° | Full z=[0,30], complete connected z=0 cap | deterministic ≤.002 mm | B |
| L-A | R5.00, 90.5° | Same complete cap | zero | A |

All arms declare the **same stipulated .05 mm coordinate bound** and use
`free_circle`; this declaration is not scanner calibration or propagated
measurement uncertainty. The crossed four-case design is not a factorial
estimate of noise, pose or deformation effects. It establishes only these
specific output obligations. Do not add rows/cases or change thresholds after a
failure to rescue a run.

## Independent geometry and exact work sizes

Nominal CAD is one R5/R6 annular-sector strip: 90° sweep, first/second tangent
flange lengths20/29 mm, axial length30 mm. Its mesh samples19 angular stations,
flange distances `[0,2.5,7.5,13,20]` and `[0,3,8,15,22,29]`, radial levels
`[5,5.5,6]`, and axial stations `[0,7,19,30]`. The complete solid boundary has
**564 triangles**, with stable CAD face/triangle ownership. Grid-volume nodes
are allowed as construction scratch only; inference receives finite triangles.
CAD is independently rotated by the proper cyclic coordinate permutation and
translated by `(11,-7,3)`.

Scan geometry uses25 angular stations, flange distances0..20 in2 mm increments
and `[0,2,...,28,29]`, yielding50 welded inner-profile stations. Its U grid has
9 axial stations6..24: **450 vertices /784 triangles**. Its L grid has11 axial
stations0..30 and the entire cap at z=0, sampled at3 radial levels. The cap's
inner boundary is welded to the sheet: **650 vertices /1176 triangles** total.
There are no triangles across the missing material inside the concave strip.

The entire connected bent cap is deliberate: prior disconnected .8 mm-wide
patches failed the .05 mm normal-cone gate. Both flange arms now supply genuine
two-dimensional cap extent without thickening the1 mm sheet. A source test will
require the constructed cap's smaller nonzero covariance eigenvalue>1 mm²;
the corresponding simple epsilon term `asin(sqrt(.01/1))≈5.74°` is below10°.
This is a fixture-design check, not a claim that every runtime core qualifies.

Scanner poses are independently constructed quaternion rotations (A: axis
`(1,-2,3)`, .91 rad; B: axis`(-2,3,1)`, -1.17 rad) and translations
`(37,-23,11)` /`(-17,31,47)`. Neither pose, CAD-to-scan correspondences, feature
support labels nor truth endpoints enter inference. Raw triangle winding and
connectivity come from scan construction; CAD face labels are CAD inputs only.

Noise uses a fixed row-index-only three-coordinate modular pattern, added in the
scanner frame with a predeclared .999 rounding reserve. Record the actual
represented displacement, not just the intended vector. Never reduce the noise
or epsilon after observing a result.

## Capture and measurement rationale

The curved packet uses the literal physical V3 policy values from the named
runner source: registration correspondence distance2 mm /scan coverage.75 /
inlier-plane p95 .5 mm; measurement flange window[2.5,12.5], search margin3 mm,
max-fit residual.05 mm, plane exclusion.075 mm and tangency compatibility.1 mm.
The earlier .5 mm/.95 registration pair belongs to the original fifty-row
planar control, NOT the physical configuration. Keep that control unchanged and
report the two policy contracts separately. The array packet still does not
exercise physical byte ingestion, native tessellation or32768-point CAD sampling.

R5→5.30 changes corresponding surface points by.30 mm. For a .5° angle change,
corresponding second-flange/cap points move at most
`2 sqrt(6²+29²) sin(.5°/2) < .259 mm`. CAD chord sagitta with5° angular spacing
is at most`6(1-cos(2.5°)) < .0058 mm`. Adding the full stipulated .05 mm still
leaves a constructive **known-pose capture witness below.5 mm**, hence also
inside the physical2 mm registration /3 mm measurement search. These are not
proofs that every generated pose lies in capture, or that arbitrary scan rows
belong to a particular CAD face. Runtime must establish its own associations.

The continuous guarded ROI spans74° on the90° case; the fixed25-angle scan
actually retains11.25°..78.75°, or67.5°. Its best strip half-width is
`R(1-cos(33.75°))/2 > .42 mm`, still well beyond .05 mm.
Thus this is a real curved-support/free-circle test, unlike the50-row planes or
the20° radius5 arc whose corresponding half-width is about.038 mm.

Proposed point-estimate acceptance ceilings: clean radius/sweep2e-5 mm/degrees;
the two ≤.002 mm perturbed cases radius.02 mm and sweep.1°. These are diagnostic
algorithm ceilings, not certified enclosures. Broad angular coverage and large
flange footprints motivate them; no theorem promises the particular free-circle
implementation will meet them. Actual measured radius and signed/absolute sweep
deviations must remain non-nominal, and `free_circle` must remain the fit model.

## Pose freedoms and localization oracles

For U, the constructed CAD axial range[0,30] and scan range[6,24] leave at least
the exact-containment shift interval[-6,6]. The ALL-row2 mm axial distance outer
range is[-8,8]; a .75 quota can admit a larger domain and must be computed from
every actual assigned support, not replaced by that all-row intersection.
Post-inference truth can exhibit ±4 mm aliases; those witnesses are
never supplied as seeds. Radius and inter-flange angle are invariant to an axial
shift. C2 must preserve useful quantities while explicitly retaining unresolved
CAD-frame axial placement; finite-domain completeness must be supported by the
new runtime's actual bounds, not inferred from three sampled shifts.

Withhold unique **CAD-frame** finite endpoints for U unless the runtime has real
additional evidence. A scan-frame finite carrier can still be useful: moving a
CAD gauge and applying its inverse can cancel. Do not impose a blanket line or
count veto merely because CAD axial position is unresolved; event count remains
unknown unless instance partition evidence actually supports it.

L has real cap evidence. Proposed finite support-line oracle is the measured
inner mid-sweep generator, not physical trim: `(R cos(phi/2), R sin(phi/2), z)`
for observed support z endpoints, transformed by the independent scanner pose.
Expected axis-support endpoints use `(0,0,z)`. Require clean endpoint error≤.02
mm and noisy≤.05 mm in the original scan frame; inspect the runtime's actual
sampled-row endpoints rather than silently substituting CAD trims. Cap-only
points may not become curved support. CAD-frame localization remains provisional
whole-part best fit, not an externally verified datum or positional conformance.

### Connected cap-contamination blocker found before execution

The unchanged full cap intentionally reveals a further connected support-role
gap. At the exact construction pose, its first-flange cap rows have
`d in {4,6,8,10,12}`, `z=0`, and `r in {R,R+.5,R+1}`. All pass the physical
position-only flange window and3 mm margin. Ten off-inner cap rows are added to
55 true longitudinal flange rows in each L case. A single inner-flange plane
cannot explain their .5/1 mm transverse spread at a .05 mm maximum residual;
the present kernel can fail before reaching the circle fit.

Even if correct scan flange planes were available, the cap points at theta45°,
z0 and radiiR+.5/R+1 pass the8..82° arc guard, radius≤8 search and both .075 mm
plane exclusions. They are a radial mixture, not observed inner-cylinder rows.
This is predicted directly from the current predicates, not a new failed run.

Do not remove the cap, delete these rows, shrink capture or label it a safety
negative to rescue localization. A coherent successor needs bounded raw-mesh
support-role handling, using observed triangle connectivity and admissible
normal evidence with original-row lineage. It must preserve a cap/sheet seam
vertex when there is usable incident sheet support; rejecting every mixed-normal
vertex would discard the actual z0 endpoint. A supplied simulator cap mask is
forbidden. The exact mechanism and any conditional direction premises need
separate review before this packet can be frozen as a required positive.

## Small adversarial companions and stop rules

Source/pure constructed companions should cover: unchanged-union planar
retessellation; deleted curved transition;20° observed arc with a .05 mm flat
alternative; middle axial fragments; near-bound .049 mm noise; and an explicit
exception/incomplete-publication path. Retessellation claims require the SAME
represented union: diagonal changes in noisy nonplanar quads do not qualify.
Sheets/folds/holes/source aliases need distinct correspondence/ownership status,
not global suppression of the existing ambiguity fields.

Do not call a radius departure “outside capture” from its true-pose distance
alone. Whole-part transverse translation can partly absorb it. A global
outside-capture refusal oracle needs a bound over the complete declared pose
family; otherwise it is only a known-pose diagnostic and must be labelled so.

One serial worker, fixed finite query/candidate/metadata budgets and complete
per-case/per-arm records must be frozen and independently reviewed before
execution. The sizes above are design bounds, not an allocation of new CPU/RSS
authority. No partial query/arm artifact supplies a successful prefix. Any false
confident output, unsupported C2 positive, custody violation or budget stop is a
retained negative and blocks advancement; no retuning/retry of that attempt.
