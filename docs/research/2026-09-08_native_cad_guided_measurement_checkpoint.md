# Native CAD to scan measurement checkpoint

Date: 2026-09-08. Invented geometry only, under the active CAD-assisted offline
roadmap. No physical/held-out/email, protected input/output or control artifacts
were accessed. No application/service, database, hardware or deployment ran.
C15/EXP boundaries and the paused goal remain unchanged.

## Delivered result

The opt-in offline path now consumes actual native CAD inspection features,
instead of requiring the caller to type their nominal radius, sweep, axis and
finite span. Four real STEP/BREP round trips take **CAD R5 → independently
constructed scan R6 → reported radius deviation +1 mm** through the actual
reader, native adapter, measurement entry and report serialization source.

Final fixed matrix: **70/70 pass: 58 conditional measurements and 12 expected
abstentions**. This closes the narrow native-parameter-to-measurement handoff,
not automatic registration, general CAD bend extraction, physical count/F1 or
qualified metrology. It extends the
[previous measurement checkpoint](2026-09-08_cad_guided_measurement_checkpoint.md);
its retained free-circle stress failure is not erased or rerun here.

The incumbent copied-radius assignment comparison remains the prior isolated
source control. No full incumbent detector/worker benchmark ran this turn and
there is no newly measured Gold count/F1 gain. Default product routing is unchanged.

## Actual connected source

- [CAD importer](../../backend/cad_import.py): explicit
  `import_file(..., retain_native_shape=True)` retains the **same successful
  reader-returned object**, not a second read or reconstructed mesh. Shape and
  binding are nonserialized/non-repr fields. Defaults retain nothing; failed
  imports cannot reuse a stale shape. The convenience wrapper is unchanged.
- [Native adapter](../../backend/cad_guided_native_features.py): explicit OCP
  admission, separated `NativeGuidedPolicy`, native finite geometry and native
  edge adjacency. No file access, detector estimates or CAD-copy measurements.
- [Pipeline entry](../../backend/bend_inspection_pipeline.py): new
  `inspect_native_cad_guided_offline`, feeding the existing position-only kernel
  and actual `BendInspectionReport.to_dict` branch. No default worker calls it.
- [Frozen constructed matrix](../../scripts/cad_native_guided_checks.py): actual
  source modules/entry/report with a guarded import graph, not an application
  startup. Native CAD shapes and scan positions are constructed independently.

The entry requires explicit `allow_ocp=True`, retained binding `ocp`,
`unit_status="DECLARED_MM"` and kernel `registration_status="DECLARED_ALIGNED"`.
These are declarations, not verified physical units or registration. In
particular the legacy import result's default `units="mm"` is not used as proof.
Feature IDs are import-local inspection features, not persistent manufacturing
events. Expected feature count never becomes observed bend count.

## Supported geometry and independently reviewed math

Scope: one top-level, valid, bounded, positive-material SOLID. Each admitted
inner cylindrical patch has one four-edge wire: two axial LINE generators and
two CIRCLE end arcs. Each generator is shared natively with one distinct planar
flange in the same solid. Flanges must be four-LINE rectangles containing the
entire declared search window and axial span. No geometric proximity supplies
native adjacency; no roots are merged by radius, axis or shared flange.

UV bounds alone are not accepted as trim proof. End arc radii, centers, plane
normals, span, endpoints, parameter lengths and midpoint branches must agree.
Thus a complementary major arc cannot pass merely by sharing endpoints.
Holes, extra wires, split/unsupported trim and nonrectangular flanges abstain.

For the cylinder before location,
`S(u,v)=C+R(cos(u)X+sin(u)Y)+vZ`. The implementation evaluates world points and
derivatives, then uses `a=unit(S_v)`, start radial `r0`, and
`sigma=sign(S_u dot (a cross r0))`. Signed sweep is `sigma*(u1-u0)`.
Physical span is the projection of evaluated end points onto `a`; raw V bounds
are not presumed millimetre lengths. Axis point is the cylinder centerline,
**not** the intersection of the two flange planes.

OCCT adaptors already apply location; the implementation does not apply it
twice. Face orientation is applied separately to `unit(S_u cross S_v)`. This
behavior was checked against the
[OCCT 7.9.3 adaptor implementation](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepAdaptor/BRepAdaptor_Surface.cxx).
Inner outward normals must equal negative cylinder radials; unsigned tangency
is insufficient. The away-from-arc flange directions are
`-sigma*(a cross r0)` and `sigma*(a cross r1)`.

A reversed whole solid is rejected via infinite-point classification OUT;
otherwise an outer surface could look inner. See the
[OCCT solid-classifier orientation check](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepClass3d/BRepClass3d_SolidClassifier.hxx).
Every native edge must have exactly two distinct face uses with opposite
orientations. SameRange/SameParameter and geometric exact-method validation are
required; finite nonnegative face/edge/vertex tolerances cannot exceed the fixed
2e-6 mm engineering guard. The
[OCCT analyzer source](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Analyzer.hxx)
documents the distinction from sampled validation. This is not interval arithmetic
or a calibrated measurement uncertainty certificate.

Independent review also caught iterator-reference alias risk, unchecked vertex
tolerances and an import-context mismatch; all were corrected before the first
native run. Source tests exercise both package and backend-directory imports,
strict admission, immutable policy and mutable-current-handle snapshots.

## Frozen matrix and all outcomes

The matrix and limits were fixed before native execution:

- 54 in-memory positives: R1/R5/R30 × 60°/90°/135° × identity/oblique/axial-flip
  proper poses × actual radius factors 1.0/1.1. CAD is an analytically checked
  positive-volume annular strip with tangent flanges, extruded by OCCT. Scan
  points are separate circle/flange equations, never CAD tessellation samples.
- Four real file round trips: STEP and BREP each for R5/90° identity and
  R5/135° oblique, all with scan R6. Files exist only in fresh worker-owned cache.
  The shape used for inference is the reader's retained shape, not the writer's.
- Twelve negatives: reversed solid, shell, plain box, convex full cylinder,
  insufficient finite flange, undeclared units, untrusted registration, missing
  OCP opt-in, wrong binding, unobserved arc, grossly translated scan despite a
  false alignment declaration, and disconnected compound.

Every positive must produce exactly one conditional measurement and satisfy
2e-6 mm nominal-radius/axis-endpoint/contact/measured-radius/deviation checks and
2e-6 degree nominal/measured sweep checks. These are numerical constructed-test
limits, not manufacturing tolerances. Every negative must produce unavailable
measurements and no observed count/conformance claim. No thresholds changed.

| Retained run | Measurements | Expected abstentions passing | Failures |
| --- | ---: | ---: | ---: |
| First native run | 58/58 | 10/12 | 2 aggregate status assertions |
| Same matrix after entry/report repair | 58/58 | 12/12 | 0 |

First failures were `native-guided:67` (arc unobserved) and `:68` (translated
scan), at connected-report line 422. The lower-level kernel's `COMPLETE` means
computation finished, even when features abstain. The new connected entry now
preserves that as `computation_status` and separately emits measurement
availability plus `UNAVAILABLE`/`PARTIAL`/`COMPLETE`. Native extraction failure
has computation `NOT_RUN`. Existing kernel semantics and all geometry/fitting
source stayed unchanged. The fixed negative assertions were not weakened.
The second run additionally verifies zero measured count and null measurements
for all negative rows. This replay is not independent confirmatory evidence.

Worst final errors across the fixed positives/nominal controls:

| Quantity | Maximum absolute error |
| --- | ---: |
| Native radius | 2.8422e-14 mm |
| Native sweep | 2.2667e-10 degrees |
| Native finite axis endpoints | 2.5172e-11 mm |
| Native radial contacts | 2.0436e-11 mm |
| Scan-measured radius | 5.2936e-12 mm |
| Radius deviation | 5.2625e-12 mm |
| Scan-measured sweep | 1.1369e-13 degrees |

All 31 known radius-deviation cases (27 in-memory +4 files) pass. These tiny
errors concern exact noiseless invented geometry, not scanner accuracy. Actual
angle perturbations/noise remain in the prior kernel tests, not this native matrix.

## Resources and regression checks

Command from the Git root:

```bash
.venv/bin/python -I -S scripts/check_cad_native_provenance.py --guided-only
```

OCP 7.9.3.1; one worker, OMP/BLAS1, fresh HOME/cache/cwd, no site startup or
application import. Fixed 300 s wall /240 s CPU /2 GiB sampled process-group RSS
/1 MiB output; generated file admission after write is 8 MiB/file. Existing
default native matrices are unchanged and were not rerun.

| Run | Parent wall | Worker CPU | Worker peak RSS | Sampled group peak RSS |
| --- | ---: | ---: | ---: | ---: |
| First, retained failure | 1.947557 s | 1.887232 s | 391,266,304 B | 391,233,536 B |
| Final, same matrix | 1.927268 s | 1.878412 s | 390,742,016 B | 390,709,248 B |

No budget stop. Parent monitoring is sampled; Python audit hooks are not an OS
sandbox, and topology caps alone do not bound arbitrary native spline complexity.
The external worker budget remains mandatory for unqualified native work.

Final passing source/regression set: 56 adapter admission/math-source checks,
35 importer/native-entry source checks, 21 worker guard checks, and 25 existing
measurement/report checks =137. Final entry semantics received four additional
cases after the first native failure; do not add earlier overlapping runs to this
tally. The 25-report bounded worker took .902239 s parent wall and 110,837,760 B
peak RSS. No unchanged 8,192-case stress replay was necessary; kernel hash is
unchanged and its earlier retained failed free-circle hypothesis remains failed.

## Source binding

Final source SHA-256 (no commit or clean-worktree claim):

```text
f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f  backend/cad_guided_native_features.py
51d77da3456a8a3ccc75e11e6679d3ffca202c05a1f69fa6477654288637ce1b  backend/cad_import.py
a458a03f734a37edcae6baa160459e2ea7da7987bc98d2264852e8cbb009013d  backend/bend_inspection_pipeline.py
84520ddd98052f954d9004b8cdb032c16a0cba87a57ff80219b8df5845f675dc  backend/cad_guided_measurement.py
a94d7865517d020562443f15420d707162a3249461d901eeba5b01970110e68a  backend/feature_detection/bend_detector.py
119e756cebb82347e4354742e22aba7f8acf0c5dc1f4c0e9bebe5b5aebd9a3fa  scripts/cad_native_guided_checks.py
9c8fc30844a733457e191add07a0dc73d802a6f0d3eeb6269cda33a4f0b11437  tests/test_cad_guided_native_features_source.py
57b85d7c5ff540828295336fdebe768fa66cf0534906449a211ae5061355835f  tests/test_cad_native_guided_integration_source.py
f230764c36836707babad1487ea5e13a53a7656043911a24c7c0d0543b4d0f40  tests/test_cad_native_worker_output_budget.py
```

First-run pipeline hash was
`92e7bbc7ad26540980e9d35540164f519a3601681798ded5495f673cde1a5c2d`;
only the new connected report availability mapping changed before native replay.
Adapter/kernel/fixture hashes were identical across both native runs. Executed
harness hash was `4827941c3f0c515ecac5e3e5131ada8c70a079b7daa166dece14f43f3826bd23`;
its subsequent docstring-only edit clarifies that the guided suite also generates
files in cache, with no executable change.

## Remaining exits and next step

M3 now includes a real native finite-feature → scan-derived measurement → report
slice. Next is an ambiguity-aware CAD/scan registration and correspondence
contract feeding this path, tested on independently posed/partial/competing
constructed inputs. Do not relabel `DECLARED_ALIGNED` as verified. Registration
observability and symmetry must be explicit; then couple registration/flange
uncertainty and allow per-characteristic partial results.

Native scope is intentionally incomplete: compound assemblies/disconnected solid
owners are rejected, not merged; no XCAF identity, verified source units, arbitrary
trim, holes/split cylindrical faces, tolerances or manufacturing event inference.
No native adversarial hole/trapezoid, location-scale or multi-bend-solid acceptance
matrix was run. The code's broader rejection checks are reviewed source, not a
claim that all native geometry classes were tested.

Observed bend count, physical trim/extent, calibrated uncertainty and conformance
remain unknown/indeterminate. Locked eligible offline targets remain CAD+scan
exact observed count ≥.95, P/R/F1 each ≥.90 and minimum-family recall ≥.80;
measurement correctness is an additional requirement. Zero-shot, physical-data
evaluation and floor rollout remain deferred under the existing gates.
