# Registered CAD-assisted finite-support output checkpoint

Date: 2026-09-09. Implements the user's two annotated requests: connect and test
registration/correspondence on independently posed, partial and ambiguous scans;
connect scan-supported finite instances to a count/localization evaluator.
Source and invented geometry only. No physical/corpus/control/email access,
application startup, production activation, release, or goal-state change.

## Outcome and exact closure scope

Both engineering interfaces now execute together through the actual offline
pipeline-entry and report source. Registration no longer requires a supplied
transform in this new entry. Successful measurements alone do not supply the
instance count. The frozen 66-case connected native matrix passes: 54 finite
support outputs and 12 required abstentions, all retained in evaluation.

This is **datum-based registration**: the caller still supplies stable datum-region
selection, CAD-frame reference positions/normals, their rigid frame binding and
common millimetre declaration. Datum selection is not automated or physically
verified. Pose acceptance is bounded search evidence, not global uniqueness.
The tested reference convention is **CYLINDER_CENTER_AXIS_SUPPORT**, with sampled
support endpoints, not physical trims or manufacturing-event identity.
`observed_support_instance_count` is new; physical `observed_bend_count` stays null.

Consequently these interfaces are closed within that declared engineering scope,
not as an unrestricted Gold-corpus or robotic-acquisition integration claim.
The existing corpus runner is not switched to this path. A compatible input and
reference adapter still needs review before a separately authorized comparison.
Default product routing, uncalibrated uncertainty, indeterminate conformance and
production-use prohibition remain unchanged.

## Connected source

- [Registration](../../backend/cad_guided_registration.py): at most 5,000 distinct
  datum positions per cloud, 24 proper PCA starts, bounded rigid refinement,
  bidirectional finite-support proximity, local surface observability and
  competing-pose rejection. No scale, truth correspondence or known pose input.
- [Pipeline](../../backend/bend_inspection_pipeline.py),
  `inspect_registered_native_cad_guided_offline`: strict explicit OCP/unit
  admission, integer datum-row bounds before casting, disjoint datum/inspection
  rows, proper SE(3) check, transform, native feature extraction, measurement,
  finite-support publication, actual `BendInspectionReport.to_dict()`.
- [Measurement](../../backend/cad_guided_measurement.py): retains private raw
  arc/flange ownership through fitting, then strips row IDs before serialization.
  In opt-in instance mode, plane-only context can be shared; any cross-feature
  reuse involving curved support invalidates all competitors. Default mode
  retains its prior conservative all-shared-support rejection.
- [Instances](../../backend/cad_guided_instances.py): recomputes finite axial
  support from raw curved positions and measured axis, rejects axial/angular gaps
  and inadequate local curved/flange support, never pads to CAD span or splits
  disconnected patches into invented extra bends.
- [Evaluation](../../backend/cad_guided_evaluation.py): explicit same frame, units
  and line convention; one-to-one finite-line scoring with the retained geometric
  rule. Exact count and exact localized-set metrics remain separate. Partial or
  unavailable inference cannot earn exact-case credit. Required
  `expected_case_ids` prevents shortened-case aggregation. This pure bridge
  consumes fresh connected output; declarations are not cryptographic provenance.

The native feature adapter remains limited to a single valid solid with supported
rectangular inner cylindrical faces and two natively adjacent rectangular tangent
flanges. Its inventory is nonexhaustive and import-local. `COMPLETE` means all
requested supported inspection features yielded instances, not that every physical
bend or manufacturing event was discovered. Unsupported jobs must not be removed
from a future predeclared benchmark.

## Critical mathematics and assumptions

For a posed scan datum point `p = R x + t`, matched CAD surface point `q` and unit
normal `n`, the point-to-plane residual is `r = n · (p - q)`. With reference center
`c`, RMS spatial scale `L` and rotation coordinate `alpha = L omega`, the local
Jacobian row is `[((p-c) × n)/L, n]`. Thus rotations and translations have consistent
length scaling before the singular-value ratio test. The proper finite update is
`R' = Q R`, `t' = Q(t-c) + c + delta_t`. Translation along an axis orthogonal to
all available datum normals remains unobservable even when finite point samples
make a point-to-point fit appear unique. The two-normal-family negative tests
check this explicitly. Surface rank does not resolve globally repeated/symmetric
poses; competing generated pose classes are tested separately.

Proper SVD rigid alignment uses determinant correction, consistent with
[Sorkine-Hornung and Rabinovich, Least-Squares Rigid Motion Using SVD](https://igl.ethz.ch/projects/ARAP/svd_rot.pdf).
The distinction between point-to-point rematching and point-to-plane observability
is grounded in [Bonnabel, Barczyk and Goulette, On the Covariance of ICP-based Scan-matching Techniques](https://arxiv.org/abs/1410.7632).
We do not infer calibrated covariance from the Hessian, singular values or fit
residual. PCA's finite 24-start search is not a global convergence theorem.

Only stable datum rows fit the pose; bend rows are held out from that fit. Datum
rows are then excluded from bend measurement. This prevents the measured radius
and angle deviations in the constructed tests from influencing pose estimation.
It assumes the declared datums really are rigidly bound to the CAD frame; the
synthetic fixture does not prove that acquisition contract on a real part/rig.

For a measured axis `(c,a)` with unit `a`, finite support uses
`t_i = (p_i-c) · a`, endpoints `c + min(t_i)a`, `c + max(t_i)a`. Shifting `c`
along its own axis changes the coordinate gauge, not endpoints. Local closed
axial slabs require distinct curved angular coverage and both flange contexts.
Exact duplicate positions provide no extra evidence. Numerical guards have length
units before angular conversion; geometrically unresolved radius/gap scales
abstain rather than allowing large roundoff allowances to hide gaps. Sample-gap
criteria are declared engineering policy, not proof of continuous material.

## Frozen connected matrix and results

[Executable fixture](../../scripts/cad_registered_guided_checks.py), frozen before
first execution: 3 CAD sweeps (60/90/135 degrees) × 3 scan deviations (nominal,
radius +1 mm, sweep +5 degrees) × 3 independently constructed scanner poses
(translated, oblique, axial flip) × 2 datum coverages (complete, one quarter of
each datum plane's rows removed) = 54 positives. Twelve negatives are four kinds
× the same three poses: missing curved scan support, central axial gap, missing
third datum-normal family, symmetric cube datums.

The CAD solid is built independently in native OCP; scan positions are analytic,
not tessellation or extracted nominal parameters. Input rows are permuted; known
creation poses and point identities are available only to the post-inference
oracle. CAD datums are separate explicitly constructed reference surfaces, not
automatically extracted/segmented physical datums. This new matrix uses in-memory
native solids, not file readers. Four real invented STEP/BREP reader round trips
remain separate prior evidence in the [September 8 receipt](2026-09-08_native_cad_guided_measurement_checkpoint.md).

Fixed native measurement policy: flange window 2.5–12.5 mm, search margin 3 mm,
maximum fit residual .005 mm, plane exclusion .075 mm, tangency error .1 mm,
`measured_flange_tangent`. Instance policy: axial gap ≤4 mm, angular gap ≤2 degrees,
local arc sweep ≥20 degrees, at least 12 local points per support role. Registration
uses the frozen `RegistrationPolicy` defaults. Independent fixture acceptance is
2e-6 mm/degrees. The separate evaluator retains 5-degree axis angle, 4-mm transverse
offset and .60 overlap of BOTH reference and predicted spans. No criteria were
tuned after observing this matrix.

| Connected outcome | Result |
| --- | ---: |
| Predeclared / evaluated cases | 66 / 66 |
| Generated scan rows across batch | 90,624 |
| Accepted bounded registrations | 60 |
| Unobservable / symmetric registration refusals | 3 / 3 |
| Finite positive instances | 54 (18 nominal, 18 radius-deviated, 18 angle-deviated) |
| Complete / partial datum positive outputs | 27 / 27 |
| Missing-arc / axial-gap finite-support refusals | 3 / 3 |
| Final expectation failures | 0 |
| Maximum recovered position error | 4.7374141420755887e-14 mm |
| Maximum radius error | 7.993605777301127e-14 mm |
| Maximum radius-deviation error | 7.815970093361102e-14 mm |
| Maximum sweep / sweep-deviation error | 5.684341886080802e-14 / 8.526512829121202e-14 degrees |
| Maximum sampled finite endpoint error | 1.3259068787973082e-13 mm |

The tiny errors are exact constructed binary64 geometry results, **not scanner
accuracy**. Central-gap controls retain successful radius measurements but emit
no finite instance: this directly disproves counting successful fits as bends.

Every case has one fixed synthetic reference line, including capture/pose failures.
The all-case support score is TP54 / FP0 / FN12, precision 1, recall 54/66=.81818,
F1 .90; exact count and exact localized-set success are both 54/66=.81818.
All 12 refusals pass the safety expectation but do NOT earn count/detection credit.
A recall penalty is not a defect verdict. The selected synthetic mixture and axis
convention do not establish Gold observed-bend accuracy, family recall, paired
incumbent uplift or the offline .95/.90 targets. No incumbent was run this turn.

## Validation, failures and resources

Root executed only the named invented-input suites below using
`scripts/check_cad_guided_measurement.py --suite NAME` under `.venv/bin/python -I -S`.
The scripts start one isolated child with fresh HOME/cache, disabled site startup,
pytest plugins/conftest disabled, BLAS/OMP threads 1, 300 s wall, 240 s worker CPU,
2 GiB sampled process-group RSS and 1 MiB output caps. RSS is sampled monitoring,
not an OS memory sandbox. All final rows exit 0 with no budget stop.

| Suite | Final tests | Parent wall s | Worker CPU s | Worker peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| registration | 36 | 2.459058 | 2.415442 | 74,694,656 |
| registered_source | 22 | .479641 | .462891 | 67,846,144 |
| report | 25 | .948682 | .890979 | 111,181,824 |
| geometry | 52 | .939704 | .908893 | 201,523,200 |
| instances | 35 | .247433 | .228085 | 61,603,840 |
| evaluation | 41 | .453293 | .424689 | 84,705,280 |
| native_worker_source | 22 | .116556 | .105134 | 37,093,376 |
| instance_integration | 12 | .305215 | .279439 | 85,131,264 |

Total: 245 passing checks, not 245 independent physical cases. Geometry includes
the unchanged 324 exact matrix (288 measurements /36 flat-compatible abstentions)
and bounded-noise controls. Evaluation includes 256 fixed invented line sets with
exact comparison against AST-selected historical matcher/aggregate source; no
legacy corpus module, default data paths or physical evaluator was executed.
The 12 ownership integration tests stipulate `_measure` only; they execute real
feature/admission/ownership/instance/report glue, including both feature orders.
They are not a native multibend geometry validation.

Failures are retained:

1. The first registered-source suite produced 13 failures /9 passes because its
   test import hook also blocked NumPy's lazy `numpy._core._methods` import. The
   hook alone was repaired to permit NumPy imports. Same 22 cases then passed;
   no production/geometry policy changed. First parent .831703 s, CPU .809917 s,
   peak 70,828,032 B, exit 1, no budget stop.
2. The initial evaluator's 39 cases and first native 66 matrix passed. Independent
   final review found contradictory per-line coordinate frames were ignored when
   case-level framing matched. Two added counterexamples reproduced this:
   **2 failed /39 passed**, .469328 s parent, .440532 s CPU, 84,934,656 B peak.
   `_lines` now rejects explicit per-line frame disagreement for both references
   and predictions; omission inherits the explicit case frame. Same 41 cases
   pass. No finite matching thresholds, fixture, geometry or denominator changed.
3. The earlier eight free-circle stress failures and native report-status failures
   remain in their prior receipts. That stress suite was not rerun this turn and
   is not claimed all-green.

Native command: `.venv/bin/python -I -S scripts/check_cad_native_provenance.py --registered-only`.
First 66-case run: zero expectation failures, parent 10.288046500179917 s,
worker CPU 10.183124 s, peak/group RSS 382,926,848 B. After the frame-boundary fix,
the unchanged full matrix replay again has zero expectation failures and identical
reported geometry maxima/counts. Final parent 10.172446667216718 s, worker wall
10.080673499964178 s, CPU 10.084297000000001 s, worker/group peak 382,009,344 B,
exit 0, no budget stop; OCP 7.9.3.1. One native job at a time. Replay is development
regression evidence, not another independent confirmation.

Independent source reviews rederived the pose Jacobian/update, audited disjoint
datum/inspection rows, finite-support math, curved ownership, scoring and complete
denominators. The per-line frame finding above was closed before final replay.

## Source binding

SHA-256 of final executable sources (full dirty-file contents, not a commit):

```text
a1d34872ec9466d8201e26c2ab4d43cc2677169a317d8675643090f0e5938475  backend/cad_guided_registration.py
d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda  backend/cad_guided_instances.py
3a6a5e1b5a17f5c939f44066ee0d621ef1ace7996422c46640506cacf788b1c3  backend/cad_guided_evaluation.py
d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb  backend/cad_guided_measurement.py
822fef073ce01266825b8591a0f80e5d063ba40c70a6c5035797b3d5e994c291  backend/bend_inspection_pipeline.py
4163b6b62682fa10ec8eb3c73162506b5b54ec6274ca386e45f221528800c674  scripts/cad_registered_guided_checks.py
4e1315e8efbb1f9d2f7222c8f8ebe323b0d5a4f0c8572b77c86ae74462f0b518  scripts/check_cad_native_provenance.py
419ff89996951171a769d254412102e1d5f74630c719cd205fcad1d4b91a4bee  scripts/check_cad_guided_measurement.py
ffb6b46dae55e5498e5edb147822f107fb1014001a6028e815c8a1729e6df2fe  tests/test_cad_guided_registration.py
0909651177db6d72d7b178ab087576427b76db5d5514b3c36d7b32a579d82a55  tests/test_cad_guided_instances.py
596781ee87c63c57f93a019f4b98b769a2651cfb8bd13471012a8bdca3b8e907  tests/test_cad_guided_evaluation.py
88c67ef300494b74a6f9eede088007c1a07b26190a7ae9980da0c9cb9df45a28  tests/test_cad_registered_guided_entry_source.py
19f7ecd24b5ab908792e79d7ecbf8e8759c111314bce3d74b9e65f72df8593e4  tests/test_cad_guided_instance_integration.py
56c447c786be9a441c4be3ffb2b34d2ac7df74b4689585cd306e9cdba4dab18a  tests/test_cad_native_worker_output_budget.py
```

The first native run used evaluator hash
`ce5a1981d046f49cbc5e482b2d4112e36c3db4bde3572f07187d3822c3fb79ee`.
Final frame repair above is the only runtime difference before its replay.
Final named-source AST validation passes for all 14 files above, scoped whitespace
checks pass, and all seven relative receipt links resolve to named source or
released research documents. No source imports were executed for those checks.

## Next finite decision — corpus comparability, not an unlimited synthetic campaign

M3 now has a connected posed CAD+scan → measured finite support → evaluator slice.
Do not require universal zero-shot detection, calibrated conformance, final score
targets or floor readiness before an informative development rerun. Do not run
the unchanged corpus runner and attribute its old path's score to this challenger.

Before a comparable existing-corpus run:

1. Review a bounded input adapter with the intended challenger explicitly wired,
   stable datum selections/reference geometry and frame/unit handling. If those
   inputs are unavailable, use only a separately labeled aligned-stage diagnostic
   or implement a different, explicitly tested registration contract; do not
   claim this datum path takes an arbitrary bare PLY automatically.
2. Establish the historical Gold reference-line convention and observable target
   scope. Cylinder center-axis support, sharp flange intersection and skin/contact
   lines are not interchangeable. This turn did not inspect physical truth or
   establish its convention. Do not convert references or pad predictions merely
   to improve overlap. Choose a reviewed comparable measurement convention first.
3. Freeze incumbent/challenger, same complete case list, policy/config, evaluator,
   bounds and failure handling; obtain the existing separate physical-development
   authorization before any protected access. Then run consumed development inputs
   promptly and retain unsupported/abstained jobs even if the score is poor.

After that development feedback, freeze a prospective candidate before the new
email parts are considered. They remain untouched and their inventory/eligibility
was not checked. Coupled measurement/registration uncertainty, per-characteristic
partial output and supplied tolerance decisions remain later offline correctness
work, not evidence already delivered by these two interfaces. Zero-shot, cosmetic
UI, floor work, C15/EXP custody and the paused goal remain unchanged.
