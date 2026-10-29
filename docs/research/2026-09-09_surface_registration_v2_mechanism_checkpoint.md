# Finite-surface registration V2 — mechanism checkpoint

The autonomous CAD-assisted goal remains active. This is a separately versioned
successor proposal after the retained negative Gold11 V1 run, not a retry or
promotion. Frozen R0 and the complete V1 physical source bundle stay unchanged.

## What the retained evidence establishes

All 11 physical registration payloads are identical between the two V1-run arms.
Per arm there are nine unsupported registrations, one ambiguous registration and
one accepted registration followed by unavailable native features. The V1 native
adapter offers windows in eight cases, versus zero for the R0 extractor, but no
physical measurement was reached and no output-accuracy gain was demonstrated.

Below is each case's highest-scan-coverage generated candidate, with plane RMS
as the tie-break. It is not a known true pose or an oracle-selected solution.
Gates were scan/CAD coverage at least .75/.25, inlier plane P95 at most .50 mm,
and observability ratios at least .005. Reasons can overlap.

| Case / part | Scan / CAD coverage | Plane P95 mm | Fit / weighted ratio | Primary registration result | V1 windows |
| --- | ---: | ---: | ---: | --- | ---: |
| CG08506AE97D / 10839000 | .7333 / .7042 | 1.333 | .433 / .446 | Coverage and residual refusal | 10 |
| CG191DEC7F81 / 49001000 | .8936 / .7924 | .395 | .299 / .291 | Accepted; native unavailable | unavailable |
| CG1E0BBA2933 / 38172000 | .9903 / .9949 | 1.127 | .391 / .378 | Residual refusal | 13 |
| CG4BDD1371B6 / 37361005 | .9976 / .9990 | .977 | .274 / .275 | Residual refusal | 17 |
| CG4FAF273093 / 49024000 | 1.0 / 1.0 | .386 | .291 / .290 | Ambiguous | unavailable |
| CG64C412CC73 / 40500000 | .7041 / .7145 | 1.013 | .169 / .154 | Coverage and residual refusal | unavailable |
| CG65CD575B1C / 38172000 | .6393 / .6045 | 1.360 | .136 / .112 | Coverage and residual refusal | 13 |
| CG84C8FA12D2 / 10839000 | .9733 / .9727 | 1.291 | .491 / .547 | Residual refusal | 10 |
| CGE4B0549E41 / 10839000 | .2774 / .2546 | 1.419 | .226 / .186 | Coverage and residual refusal | 10 |
| CGEBA9A869AC / 38172000 | .6596 / .6587 | 1.328 | .132 / .115 | Coverage and residual refusal | 13 |
| CGF41F49066A / 38172000 | .9878 / .9958 | 1.136 | .374 / .340 | Residual refusal | 13 |

Across 264 distinct V1 candidates, 258 residual failures, 229 coverage failures
and eight rank failures occur. The four high-coverage residual refusals are not
proof of defects or wrong pose: finite sampling, correspondence, side ownership,
deformation and residual pose error remain possible causes. The five cases below
75% coverage have best-candidate full-scan point P95 distances roughly 38–85 mm;
this bounds the failure of this search, not all possible registration methods.
The ambiguous case's eligible rotations differ by only .029–.292 degrees from
candidate 22; three hit the refinement limit and one was stationary.

## Constructed stage trace, executed once

Receipt: `cad_engine_v1_synthetic_checks/registration_stage_trace_v1-cytd_g9n/`.
Test SHA256: `37e7646498a2bc7994e4330cdc3e3f1f701034e44dad762e9f551ca0b2b0e6db`.
Output SHA256: `87894f496803eceeded062adfdde3f28b8d89f42775a74cd582a90abca5a13a0`.
Six tests pass; this is diagnostic completion, not successful registration.
Parent wall 1.040545 s, CPU .815081 s, worker peak 388890624 B; no budget stop.
Frozen budgets were 60 s wall, 45 s CPU, 1 GiB RSS and 1 MiB output.

Candidate 20 and the strict 32768-CAD-sample construction were fixed before
execution. Exactly one original coarse and one original fine refinement call
ran, each capped at 20 iterations. Oracle errors were added only afterward.
Snapshots describe the *next selection at the stage transform*, not the previous
iteration's actual final update. The CAD sample digests match the dense predecessor.

| Boundary snapshot | Pose action max / RMS mm | Admissible rows / cap rows / rank | Trimmed rows / cap rows / rank |
| --- | ---: | ---: | ---: |
| PCA start | 1.738923 / .865111 | 857 / 11 / 6 | 729 / 4 / 6 |
| After coarse, next selection | 1.517024 / .912184 | 857 / 14 / 6 | 729 / 0 / 5 |
| Before first fine selection | 1.517024 / .912184 | 824 / 2 / 6 | 701 / 0 / 5 |
| After fine, next selection | 1.502403 / .887232 | 818 / 0 / 5 | 696 / 0 / 5 |

This directly shows global trimming removing every cap correspondence at the
before-fine snapshot. The coarse pose was already wrong, so trimming alone is
not the complete causal explanation. Cap rows count correspondences, not distinct
independent constraints. Each subset recomputes its center and length scale;
SVD vectors must be converted to a common physical twist chart before comparing.
Float64 zero singular values are not exact nullspace certificates.

For the normalized point-to-plane Jacobian, the axial translation basis gives
`sigma_min(J) <= ||J e_tz|| = sqrt(mean(n_z^2))`; the weighted analogue is
`sqrt(mean(w*n_z^2))`. This is a local linearized bound. Wrong or repeated
correspondences can suggest apparent information and do not prove a correct pose.
See [Gelfand et al., 2003](https://pixl.cs.princeton.edu/pubs/Gelfand_2003_GSS/stabicp.pdf).

## Successor mechanism and audit gates

### First additive primitives verified

Independent source audit found and closed mutable-index backing, mixed-scalar
coercion, all-query tie-storage allocation and extended-budget-subclass refusal
issues before first execution. The final primitive uses bytes-backed immutable
arrays, query-batched tie storage and explicit whole-query clearing. These are
software contracts, not numerical interval certification.

Three isolated invented-input suites executed once, serially, with 60 s wall,
45 s CPU, 1 GiB sampled process-group RSS and 1 MiB output limits:

| Suite / retained receipt | Result | Parent wall s | Peak RSS B |
| --- | ---: | ---: | ---: |
| `triangle_surface_v2-qal62786` | 72 pass | .632675 | 69992448 |
| `triangle_surface_v2_independent-bqix3oxi` | 95 pass | .204832 | 51462144 |
| `cad_surface_v2-_pg929f0` | 31 pass | .166917 | 51003392 |

All are under `cad_engine_v1_synthetic_checks/`; no budget stop or source drift.
The first includes 64 queries against 32 triangles checked by exact rational
simplex minimization and an 8192-triangle grid with 180 independently generated
queries. The independent suite covers all seven triangle Voronoi regions under
all six vertex orders, rigid/scale cases, finite holes/boundaries and ambiguity.
These fixed matrices do not establish worst-case 500k-triangle/200k-query capacity.
The export suite is pure/mocked native plumbing; no OCP execution occurred.

Verified source SHA256:

- Finite query kernel: `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1`.
- Kernel tests: `acff2358b088589b04988c3b99d87580e404ea7ef5db50d72cbfa4717843e285`.
- Independent tests: `4358dc2713d41613ab9271030ee182c4371ec201fe4a37240875ad03a832231a`.
- CAD triangle/sample export: `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1`.
- Export tests: `decc24a59d3f898180b1d823430de7e27ad41d21e20b03a5606581e7670a9620`.
- Harness at these executions: `c0c13990d248cbda341c7828fc8e1a9d9e80e75bba067bc243e032ed15d74733`.

At that primitive checkpoint V2 registration was authored but unexecuted: weighted centroids/covariance,
finite Euclidean Huber residuals, center-based fine motion caps, original global
trim, and conservative FACE-interior normal rank on all supplied gated rows and
the actual trimmed/weighted subset. No refill or fixed-q Hessian rank is used.
The later execution receipts below supersede that stage. Native validity has a
separate one-CAD diagnostic draft; its physical supervisor is now under independent
source review and its frozen pre-run gate is not yet issued. No additional physical
input was accessed.

The native-validity source suite then passed 26 checks (one native test skipped),
receipt `native_validity_v2_source-uqu21lbp`, parent .146049 s, peak 53084160 B.
The first four-case invented native gate is retained at
`native_validity_v2_constructed-94b6qjy6`: **3/4**, with all 26 source controls
passing; parent .926150 s, peak 330874880 B, no stop/source drift. The open-shell
fixture failed before the diagnostic (`Standard_Failure: TopoDS_Builder::Remove`)
because the constructed box's builder-owned shell could not be edited. Valid box,
two-owner refusal and STEP reader-path box passed. This does not qualify the
missing invalid-solid case or authorize physical execution. A source-only fixture
repair assembles the same five retained faces in new shell/solid containers,
without unlocking/mutating the original box, healing, or changing the diagnostic
or any expected result; the four cases and all budgets remain unchanged.
After independent source audit of that exact fixture-only difference, the new
receipt `native_validity_v2_constructed-thbnotxp` passes **4/4** (27 pytest tests
including the 26 controls). Parent 1.071899 s, peak 329809920 B, no stop/drift.
Diagnostic source remains `e2767721c5b6c1bdf02b77cb42d4c6c2cbc68402b6e63407afecfb92d8cfca91`;
new test SHA `7b51023513629456c9bd16833eda2ff3ddee0e42a2cc302d73f6d508e2f6371b`.
This confirms the invented invalid-solid diagnostic path, not physical validity.
OCCT's exact-method option applies to its edge curve-on-surface checks; it is not
certified exact geometry, metrology or source-unit calibration.

### V2 registration checks completed

The separately audited registration source has passed 35 focused controls and
the fixed four-case public geometry matrix, without injected inference poses.

- Focused receipt `surface_registration_v2-jmbspwt0`: 35 pass, one matrix skip;
  parent .312121 s, peak 77725696 B.
- Matrix receipt `surface_registration_v2_matrix-auhunoti`: 4/4 cases, 36 total
  pytest tests; parent 2.599716 s, peak 79249408 B.
- Both used 60 s wall /45 s CPU /1 GiB /1 MiB, with no stop or source drift.
- Two scalene tetrahedron poses recovered maximum point-action errors
  `8.591146515148075e-15` and `1.1868486651150049e-14` mm under exact invented
  arithmetic geometry (limits .02 mm). Plane support refused for unobservability;
  a rectangular box refused generated-pose ambiguity. All 96 candidate records
  and post-inference oracle errors remain retained. No noise/native/generalization
  conclusion follows from these exact planar constructions.
- Source SHA `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3`;
  tests `323243bdba1c975e3e9d932d547d93699c5f85954f0c927fe610a4bc03dcf63e`;
  harness `61cd850fa552efe26b176a3df6d6e41a51f517f0c15a9ba028f4dcba262d6480`.

For fixed retained correspondences, Huber of distance is a concave function of
squared distance: `.5*s` below `delta^2`, otherwise `delta*sqrt(s)-.5*delta^2`.
Its tangent majorizer therefore has quadratic weight `min(1,delta/d)`. Weighted
Kabsch uses both weighted centroids and covariance and a proper-rotation SVD
correction. Fine motion is bounded about the weighted current-point center,
not by world-origin translation. Backtracking checks the same retained set's
Euclidean Huber loss; new nearest sets/gates across iterations preclude a blanket
global monotonicity or convergence claim. Normal-rank diagnostics exclude
boundaries and ambiguous rows without refilling the actual globally trimmed set.

The next test connects the exact original three curved-bend STEP/scan fixtures
to V2, with the dense predecessor's 32768 samples and every original geometric
error ceiling unchanged. Its separate prospective budget is 180 s wall /120 s
CPU /2 GiB /1 MiB. Fatal query/budget/incomplete search cannot count as a passing
missing-arc sentinel. Workflow mocked checks and independent source audit precede
this constructed native run. No new physical evaluation has started.

The final workflow audit also required explicit CAD-aligned geometry labels on
nested measurements, distinct from the inverse-transformed scan-frame predictions.
An independent affine oracle checks all four endpoint fields and guards against
in-place mutation of the retained CAD-frame instance. After final source review:

- `mesh_offline_v2-3ug71ne1`:88 pure/mocked tests pass; parent .531665 s,
  CPU .348035 s, peak87719936 B; output SHA
  `bc6057bc44329bebc74a3b3bcdd8b6846b5167dc01f5d76735bee57283263060`.
- `mesh_connected_surface_v2_source-ai9t21uw`:9 source controls pass, native
  matrix skipped; parent .304016 s, CPU .270588 s, peak85753856 B; output SHA
  `c18b5584454c557b17a7929af773f8dbc2897b698d8c49db932a8c812d871452`.
- Both60 s wall/45 s CPU/1 GiB/1 MiB; no budget stop/source drift.
- Workflow SHA `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf`;
  mocked test `10fdfeac837d32a4f61eff53e329fbfeb0b742fcf7d9834024819e6241bb9cfc`;
  connected test `e28ed83f8a6a61975e8ac183922fc5a08c00ff9cc5e23f5f99d202ea49f6881c`;
  harness `7ce457a9d3037d89cbd96d63ca4f41116a2114fd7d2e1c99e26923c8f8fbb7f5`.

The original one-CAD supervisor draft failed independent source audit before
any freeze or execution: its inherited supervisor could catch the110 s work
alarm and perform kill/reap/cache cleanup before the total120 s timer was
rearmed. A separate outer process watchdog is being designed. This does not
invalidate the four invented native diagnostic cases, but blocks physical access
through that draft. No original CAD was opened for this diagnostic.

### Connected V2 native result: retained negative

`mesh_connected_surface_v2-lqfsnygb` completed all three cases: **1/3**, with
nine source controls passing. Both nominal branches emitted zero predictions;
the missing-arc case refused and passed its sentinel. Parent91.725850 s,
CPU91.302646 s, peak396230656 B, no stop or source drift under180/120 s,
2 GiB/1 MiB. Output SHA
`dfa92f4a7480bbdb144af057715a86ca0c548619d1e9a5133ca9786e2c894b8b`.
All72 generated candidate records and post-inference oracle errors are retained.

The nominal best generated pose is candidate20 with maximum/RMS point-action
error1.937597/.999123 mm. Scan coverage .975654, CAD sample coverage .259094,
finite-surface P95 .470098 mm: it fails the unchanged .30 reverse-coverage and
.05 mm residual gates. Actual trimmed/weighted normal-rank ratios .079962/.040421
are above .005. Both coarse and fine stages hit their20-iteration limits. This
negative is not explained by local rank deficiency. Finite closest triangles
alone did not solve the initialization/refinement mechanism; no measurement or
physical improvement follows. Native triangle export itself completed on276
triangles, with the same32768 initialization samples as the dense predecessor.

Each nominal fit consumed1008 queries/909336 rows/38552251 triangle pairs and
366316335 logical work units. A source-only review identified redundant boundary
classification during the tie-only second pass as a possible equivalent speed
improvement; none has been implemented or qualified. The next scientific
hypothesis is a separately versioned fine Gauss–Newton update with actual finite
distance line-search acceptance, retaining coarse starts, global trim, thresholds
and strict error ceilings. It is not approved merely by this proposal.

The additive [fine-only V3 protocol](2026-09-09_finite_fine_v3_mechanism_protocol.md)
now has independent math/source review. The candidate-comparison source controls
passed2 (one native skip), receipt `finite_fine_comparison_v3_source-xxckic0s`,
parent .146181s, CPU .115085s, peak48054272B, no stop/drift. Output SHA
`2bcd8c7beb38650df79a0e84ccccb7543146fde3559f753a3186546d26524900`.
Helper qualification subsequently passed40 independent controls in
`finite_gauss_newton_v3-mxab6ibe`: parent .418498s, CPU .251694s, peak76742656B,
no stop/drift under60/45s/1GiB/1MiB. Output SHA
`277619e2e8adb7f5070aa5e177acc9f1a63bffed609a7f4badce737fbc51e471`.
This verifies the local sliding mechanism, finite-boundary safeguard, per-point
weights, chart/cap/frame arithmetic and fatal refusal contracts—not a native
or physical engine result. The independently audited one-start native diagnostic
is the next separate gate; no advancement criterion is presumed.

Source/research fallback, only if justified by later evidence: bounded scan/CAD
plane-triplet pose proposals may address partial-scan covariance bias without
relying solely on24 PCA starts. For matched unoriented planes, enumerate signs
and proper normal alignment; if R*n_scan=sigma*n_CAD, solve
`n_CAD dot t = d_CAD - sigma*d_scan`. Two nonparallel normals leave translation
along their cross product unresolved; never fill it from partial-cloud centroids.
Require a third independent observed plane or retain an underconstrained family.
Finite surface/refinement/coverage/rank/competing-pose gates must still accept
every proposal, and all caps/alternatives must be fixed before execution. This
is a derived plan for the planar-flange scope, not implemented global robustness.
Repeated inner/outer planes, absent third planes, nearly parallel normals,
disconnected coplanar tabs, partial coverage and nonplanar CAD need adversarial
proof. Local descriptors such as [FPFH](https://pointclouds.org/documentation/classpcl_1_1_f_p_f_h_estimation.html)
need neighborhoods/normals; repeated homogeneous flange neighborhoods can remain
indistinguishable. [TEASER](https://arxiv.org/abs/2001.07715) robustly addresses
putative correspondence registration, not verification of the material side.

### Fine-only V3 native result: retained negative, no advancement

The one-start diagnostic `finite_fine_comparison_v3-fxxcquiv` completed with
`candidate_level_positive_criterion=false`. Its three pytest checks passed
because both arms and their failure records were retained; this is **not** a
passing candidate geometry result. Parent2.881231s, worker CPU2.515437s,
peak385335296B, no budget stop/source drift under60/45s/1GiB/1MiB. Output SHA
`3f31c16475dd9b30abd79003ee47b547f55824fd0f1d3b7c119492b258889c4b`.

Both arms used the same invented nominal STEP/PLY, one finite surface/sample
export and one V2 coarse fit from generated PCA candidate20. Each fine arm had
the unchanged20-iteration budget; the oracle was attached only after both were
terminal. The control reproduces the earlier V2 candidate20 exactly:

| Quantity | V2 fine control | V3 fine challenger |
| --- | ---: | ---: |
| Maximum pose action error, mm |1.937597289991|2.242451001648|
| RMS pose action error, mm |.999123300256|1.096862233511|
| Scan coverage |.975654069767|.976380813953|
| CAD sample coverage |.259094238281|.261199951172|
| Finite distance P95, mm |.470097585358|.468162886177|

Both remain below the .30 CAD-coverage gate and above the .05mm residual gate;
both hit the fine iteration limit. V3's actual/weighted fit-rank ratios
.090877/.171220 are not the refusal. Slightly better residual and coverage
accompany worse oracle pose: local descent is not evidence of correct global
correspondence. The frozen <=.20mm/strict-control-improvement advance criterion
failed; do not advance this fine-only approach to the full24-start/three-case
gate, add iterations to this attempt, or treat its40 helper checks as engine gain.
The next justified source-only approach is the bounded plane-triplet proposal
above, with new protocol and independent audits before execution.

The single-CAD supervisor cleanup repair is independently source-green at driver
SHA `11049b18204501481d7ab6a378f8402ae5545382bb3fc4007e126b19dbde52e3`
and tests `bba23f44843921a9bb121287b695e4668a03140e6d8d4217f9dc4b4abd958950`.
An outer watchdog owns the complete coordinator/worker process group; cleanup
stays inside it, while a hard total deadline covers receipt work. Next gates are
pure/mock controls and two invented stdlib subprocess smokes. No freeze or
original CAD read has occurred, and source-green is not physical-run approval.

The pure/mock supervisor gate now passes73 controls, two subprocess tests skipped,
receipt `native_validity_v2_supervisor-bvagyz85`: parent .448226s, CPU .406765s,
peak51068928B, no stop/drift under60/45s/1GiB/1MiB; output SHA
`53f482c243498518887f09de400cbbea813a431878afb133a03099b52ddd3bdc`.
The freeze/execute messages in that output refer to pytest-owned invented bundles,
not the physical diagnostic directory. Independent harness review then blocked
the subprocess gate: its external monitor counts/kills only the pytest process
group, while the smoke creates another group. An acknowledged smoke-only process
group registry is being added before any smoke execution. The physical driver
and CAD remain unchanged/unopened. Passing pure tests does not close containment.

The new [V4 source protocol](2026-09-09_plane_initialization_v4_source_protocol.md)
separates triangle/raw-row lineage tests from independent plane-pose arithmetic.
Both are prerequisite gates, not an engine-positive experiment. Its normal-fit
translation uses intrinsic triplet intersections, preserving origin covariance
for imperfect normal congruence; direct offset mismatch remains diagnostic only.

V4 prerequisite sources are now stable and independently source-green, but not
executed yet: triangle parser `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734`
with50 controls `b40cbdd7892432e516f8b95afc7453cad0a0224baa0918320e1e7dd22ae3a62e`;
plane helper `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370`
with74 independent controls `bce06932fff6c575383ef3aa73576e22572fcbb4569ca1c25a20870769647be8`.
All four parse syntactically; AST success is not a passing test receipt.

The bounded extraction successor in development uses actual shared raw edges
for near-roundoff coplanar compression, then validates each entire component
against all original vertices. It must preserve finite triangle unions, holes,
disconnected patches and small caps; an accumulated curvature chain that fails
the whole-support test remains refused without trimming. This is deliberately
not noisy scanner qualification. Shared-edge neighborhoods, maximum vertex
distance and normal-angle criteria are established mesh-region-growing tools;
the new no-trim/full-component contract is our scoped design choice, not a
guarantee supplied by [CGAL](https://doc.cgal.org/latest/Shape_detection/index.html).
[Polylidar3D](https://arxiv.org/abs/2007.12065) provides broader mesh/normal-based
planar segmentation and finite-hole representations; no new dependency,
smoothing or dominant-normal filter is being adopted here. Its reported speed
and accuracy are not measurements of this implementation.

The V4 prerequisite suites have now executed serially and passed, with their
fixed60/45s/1GiB/1MiB limits and no source drift or resource stop:

- `mesh_topology_v4-1qt4n64f`:50 pass, parent .156505s, CPU .124292s,
  peak52805632B; output SHA
  `f30a051601b77f446e36b243d94e943767178481d643b153be3efb5e11bcfeee`.
- `plane_hypotheses_v4-xf3jfape`:74 pass, parent .241079s, CPU .206449s,
  peak52969472B; output SHA
  `c77f450e28132d7a63f1964636c0d5dde11fbe7d561bf1c2dd222df61e18ac65`.
- External smoke-registry controls `cad_smoke_group_custody-kbxitpnf`:45 pass,
  parent .256140s, CPU .229605s, peak47611904B; output SHA
  `597f4ac4facdc835cedf1a33a5d6fa8e6a9f96d93cc6064a9ad6b686816e5997`.

The harness at those receipts is
`5f6f157a99ecb69f3240959ea31a2a9f3e4505fb0f83cd9b5c0cd425d1948d25`.
Independent review first found and closed a false group-retirement bug: an
absent process-table row is not proof that the directly owned test child exited.
The new45-control suite includes no-false-retirement and direct-child/group kill
regressions. The V4 results qualify byte/index custody and supplied-plane math,
not connected extraction, native accuracy or a physical improvement.

The first actual subprocess gate then stopped negatively, retained at
`native_validity_v2_guard_smoke-yv3p32kc`: `SMOKE_GROUP_CUSTODY_FAILURE`, exit−9,
parent .461432s, peak51691520B, source hashes unchanged; output SHA
`48d47af2a0631b7b416199845b62bcf3edb680420afb2969cf5e64db25c7986f`.
Only one of the two required groups was acknowledged. All tracked groups were
confirmed empty; a cleanup PermissionError is retained, not silently ignored.
The harness kept the private invented cache/registry for diagnosis. The original
monitor exception was collapsed to a generic code, so its precise cause is not
yet established. No old process ID may be signalled after confirmed retirement.
Do not retry that attempt or freeze/open the physical CAD from this failed gate.
A diagnosed/instrumented source repair and new independently audited receipt
are required; the original physical driver remains unchanged.

Independent source diagnosis found two possible races, without retroactively
claiming either caused the old generic exception: requests were read after an
older process snapshot, and unlinking an atomic-publication staging hardlink
could mutate ctime during a concurrent stable-descriptor read. Cleanup could
also issue the old ACK. Exact failed-run bytes are preserved in
[the source snapshot](source_snapshots/smoke_registry_5f6f157a/scripts/check_cad_guided_measurement.py).
The [separate prospective successor](2026-09-09_v2_smoke_custody_successor_protocol.md)
captures fixed slots before a fresh process snapshot, retains bounded staging
hardlinks, forbids cleanup ACKs and records structured original exceptions.
Its independent review and new execution remain separate gates.

The actual finite-patch extractor now passes **99 independent constructed
controls**, receipt `mesh_plane_proposals_v4-rxn85jr_`: parent .2892992916s,
CPU .233529s, peak72744960B, no60/45s/1GiB/1MiB stop or source drift; output SHA
`57e5bf05902e678c39c98b0a88ad0c8a4b41b2e898691f8776263bcd136d3b67`.
Source `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48`,
test `c2551fbd8c81e3204a8970c7dba4aa3e8b3c0d1b0a06415c522b6f3b3424375a`,
[protocol](2026-09-09_mesh_plane_proposals_v4_protocol.md)
`2ca3469de02e83ca3fcb0a891b96df494edc95010f8a134029dece3d7c80e479`,
harness `cf31c93eee0c947ecbbe6a9c7b3d82d1b045544b2bca92f1f5276f3ef3082692`.
Checks include analytic triangle-area moments, nonuniform retriangulation, holes,
raw-ID-separated components, small caps, distant low-area contamination,
whole-chain rejection, frames/units, fatal clearing and deliberately noisy
fragmentation. A repeated in-plane covariance eigenspace cannot gain width
admission through arbitrary axis rotation: the additional invariant sufficient
bound is `2*sqrt(lambda_min_inplane)`. Its conservative false refusals are tested
and disclosed; it is not a calibrated enclosure or exact minimum-width estimate.
This qualifies the near-roundoff array seam only. Capacity and the connected
plane-initialized native workflow have not executed; no physical gain is claimed.

The [independent smoke successor pre-run gate](2026-09-09_v2_smoke_successor_pre_run_audit.md)
was satisfied:54 pure controls in `cad_smoke_group_custody-mokgunxf`, then75 tests
(73 pure and both original actual smokes) in
`native_validity_v2_guard_smoke-8gy31qav`. The latter took1.781326416s parent,
.478284s pytest-worker CPU, peak89931776B sampled across the declared groups;
output SHA `08c0345539586333f43662b4f086ec34e8c50f082fbbff48a625233867bb6f17`.
Both main-admission ACKs and complete group emptiness were confirmed; no primary
or cleanup errors, no budget stop or source drift. Successful private scratch
cleanup is recorded. This closes that invented containment gate, not physical
validity or engine accuracy.

Root then froze only the seven source files/runtime metadata for the single-CAD
diagnostic: bundle `4b91d6c7db4e8709a609eb6f9fc845271d83eebc316378875866f4a2b0f1fd5c`
at `v2_native_validity_diagnostic_attempt/FROZEN.json`. This operation did not
open/stat/hash the original CAD. Independent exact-bundle review is in progress;
execution remains blocked until that separate audit actually records GREEN.

That audit subsequently recorded exact-bundle GREEN, and root executed the sole
CAD diagnostic once. The [independent post-run report](2026-09-09_single_cad_validity_result.md)
confirms source/input/dispatch custody and a retained global native negative:
4.405096s parent,4.161675s worker CPU,394919936B sampled complete group plus outer,
no budget stop/drift/error, one import/analyzer, group/cache cleanup confirmed.
Result SHA `157cc870a6b8ca529875dfbb95b84e1b570c288a5133193b65ad7d99c7d06915`.
Forty-four face self errors and22 edge/face error-context records are observed,
but the context iterator omits468 incidence-derived witnesses. The same edge
coverage deficit occurs in previously retained invented boxes. Thus the exporter
flag `evidence_complete=true` is not complete context validity or a clean-local-
patch certificate. Source/invented context-coverage validation is required before
local admission research; no status bypass, tolerance increase or healing follows.

The independent [capacity protocol](2026-09-09_mesh_plane_capacity_v4_protocol.md)
and final harness `398a67585a02fb56e2244b9ed351d88119eb7a5d545178080ff736d2a46dd45c`
were source-reviewed by `/root/astra_repo_guidance_review` and root. The small
route `mesh_plane_capacity_v4_source-2_yg5v3w` passed2/skipped1, parent .2191951252s,
CPU .17647s, peak71155712B, no60/45s/1GiB/1MiB stop/drift, output SHA
`43575621f9f689d388747250ff674ab2ea2a517036a06c9f3df364eb0938bb62`.
The conditionally approved capacity route then ran once as
`mesh_plane_capacity_v4-dd_n4pq2`:3 pass including the exact800000-vertex,
1600000-triangle closed annular prism. All ten planes, all triangle rows and
807608 memberships over all800000 raw rows match the independent finite-support,
hole-moment, edge-incidence and multiplicity oracles. Extraction1.1134915s;
construction through verification1.1750495s; parent1.3756264s, worker CPU1.346889s,
worker/sampled-group peak761790464B. No120/90s/4GiB/1MiB stop or source drift;
output SHA `f3ab988e06b34998229a96619180a51cc36b5e3e3310ab3f4c7b06d615e7ba8b`.
Tests `4f65d7cf1aa70e79a93670c95122d4d506b1210599b9cf68c1b39d9aea5972b6`,
protocol `cb613de88a1be7d76dbdb454f645e528164bef2afa8cf8a9efe19e59a438b6fb`;
extractor and original mesh-plane protocol remain55356d/2ca346 as above.
This is measured capacity for one exact axis-aligned manifold, not worst-case
memory, noisy extraction, pose accuracy or physical count/F1 improvement.

The connected finite-plane initialization source and independent tests received
source-only GREEN before the first57-check execution. Review caught a fixture
issue before execution: scan rows coincident with CAD triangle vertices cannot
provide V2's required FACE-only observability. Those five cases remain required
negatives; five separately tessellated scan/CAD positives were added without
changing any engine gate. Four poses and one explicit126-of147 registration-row
selection recover the pose on all147 original rows within1e-8mm. Full prefilter
denominators remain49 rows per patch, even for the selected global cloud.

Receipt `plane_initialization_v4-ogojk5_z`:57 pass, parent .8262768327s,
CPU .797531s, peak82395136B; no60/45s/1GiB/1MiB stop or source drift.
Output SHA `7867e3323b4dd29cf8d99f1a5cb66e5762760f18d5ee9a7b1ff7b3785dffb748`.
Initializer `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724`,
tests `cc8a8ab7a878fcdb983458b7829779f62fbf42949464048e5f1c47be3be75417`,
protocol `f5463d6f94470153daca3035a1b84f7f35284ee2ce31ff0e5661cee2f9aaea82`,
harness `1f9f23e1f1cfc9853fb7538a47f349bc537e0c2cebecd6e5f4e7906f0e7adb91`.
This is source/constructed plane registration, not yet curved-bend measurement,
noisy-scan qualification or physical count/F1 improvement. The separately versioned
three-case curved-bend V4 workflow then underwent independent pre-run audit.

The [V4 pre-run audit](2026-09-09_mesh_connected_planes_v4_pre_run_audit.md)
closed two sentinel masking paths before execution: registration binding/transform
failure and local numerical/resource failure previously could resemble a geometric
abstention. Exact reason allowlists and full window denominators now separate
them; the nominal error gates and original fixture remain unchanged.
Source route `mesh_connected_planes_v4_source-5kif0r8_`:48 pass/1 native skip,
parent .7050137091s, CPU .368510s, peak90374144B, no60/45s/1GiB/1MiB stop/drift,
output `4ff81eb258dc8fadd691138bd63016e8d79af49a3b444b125e1d092a0e4c263e`.

Under that audit's conditional gate, the native route ran once as
`mesh_connected_planes_v4-epppmtsj`:49 pytest pass including all three connected
cases. V2's retained1/3 becomes V4's3/3 on the unchanged constructed geometry:
both nominal branches publish one actual measured finite instance; missing arc
has zero predictions with `COMPLETED_GEOMETRIC_REGISTRATION_ABSTENTION_NOT_OBSERVED_ABSENCE`.
Each case evaluates all1728 prefilter hypotheses, retains1720 geometric rejections
and refines all eight surviving starts, with zero numerical failures/unattempted
starts. No no-start sentinel exception was used in this run.

Both nominal cases: radius error7.9936057773e-15mm, absolute sweep error
1.4210854715e-14deg, generator/axis endpoint errors1.3800817981e-13/
1.3690630047e-13mm; whole-original-row pose action error7.8429591160e-6mm.
These ideal constructed errors are algorithm checks, not physical precision or
calibration. Parent42.8955746661s, CPU41.986215s, peak428785664B; no180/120s/2GiB/
1MiB stop or source drift. Output SHA
`d5215d53bdf010b092772a67158870423ac026c8d7ff3fa87f91df1ab377bb67`.
All three detailed records retained, respectively4921380/4866127/4919276 bytes:
`1656c22327e319fa024647b06693980f5a9b28e7aaaf8c7dbfb2eac3446d3ec5`,
`da5ceb30742a930ac558e95c63a02b3841182a5eeba47908e580b4dd2252c4f7`,
`42c3f892696da38a106e092b1f8a69e6b82a649da153706a33c0c27075df8e80`.
Workflow/test/protocol/harness hashes match the pre-run audit throughout. This
is concrete connected-output improvement within the exact synthetic contract.
The [independent post-run audit](2026-09-09_mesh_connected_planes_v4_result.md)
subsequently confirmed all artifacts/source bindings and exact equality of the
existing policies and downstream CAD arrays with V2. Serialized STEP hashes
differ; their bytes were not retained for exact header attribution, so this is
constructed geometric comparability, not a same-byte CAD claim. Report SHA
`0cbf67a467f2d02ca09481a14b6b806ed56e9c70a8b2474014b01de4dffb9b28`.
No physical Gold11 gain, native-invalid CAD bypass, noisy-scan qualification or
factory release is implied.

The independent context-denominator successor also completed its first bounded
source-only run: `native_context_coverage_v3-0ly2yyyq`,173 pass, parent .1681903750s,
CPU .136553s, worker peak45596672B/sampled group45580288B. No60/45s/1GiB/1MiB stop
or source drift; output
`b59e5ceb4a31cf9b31b0dcf5b3cf7523c15844537bad4c56f91e0059628cd30a`.
Independent author/reviewer `/root/noise_plane_math` found and regression-covered
a mutable public-default hard-cap bypass before any run; the implementation now
validates captured integer limits against independent literal maxima and uses an
immutable private snapshot throughout the call. Root read all173 prospective
controls and the repaired source; the named harness route received independent
GREEN before execution. Symbolic oracle:9 entities,12 occurrences,11 expected
context pairs,39 ancestor steps,20 clear status values. Seam multiplicity and
cumulative orientation do not erase or duplicate native context identities.

Validator `e9a366cf36106495dfd4d8563f639af8ba185f9113fea0ac40ad030d3c0b4eeb`,
tests `5665b9125ab329fe2419397ca2b00e3385ad5b2132499987b011c0474f3aa348`,
protocol `c5bb4777f4f80bcc6a3832418ec607a4bf0bf2a1a29cccf762e5471bbd07f02e`,
harness `2b45f3420949041a0fd4694a2b881db1ae2e2f1ee9b70675fca063b46e692e68`.
The harness changed only after the V4 native job terminated, adding the pure
coverage route. Its exact priorf80372 bytes are preserved in
`source_snapshots/mesh_connected_planes_v4_f80372ee/check_cad_guided_measurement.py`;
the snapshot is archival, not directly executable from its relocated ROOT.
No V4 engine/test/protocol or seven frozen physical-diagnostic sources changed.
Pure coverage does not authenticate native evidence or grant feature admission.
A new shape-only keyed cached-context collector is being authored source-only;
no new native or physical job is authorized by this test receipt alone.

Noisy-plane/scalable-correspondence work is separately outlined in
[working design notes](2026-09-09_noisy_plane_registration_design_notes.md), not
a frozen experiment. It cannot reuse exact-mesh success as noise qualification.

### Active successors after audited V4 output progress

The active goal remains the user's end-to-end CAD-assisted engine mandate.
The user asked how close the next Gold11 baseline is: the next physical scored
run still requires noisy extraction/bounded correspondence plus the needed native
context/admission mechanism and an independent exact-protocol pre-run audit.
It does not require completion of every final roadmap target. Historical F1
comparability remains unestablished; a new same-input paired baseline is the
honest fallback. No new physical score is available. The following source-only
state preceded the three executed gates recorded below; it is retained as history.

New, not-yet-qualified source:

- `backend/cad_guided_noisy_mesh_planes_v5.py` and its
  [protocol](2026-09-09_noisy_mesh_planes_v5_protocol.md): author
  `/root/cad_measurement_integration_review`; original-edge two-scale cores and
  independently retained fixed-cone closures. Correspondence is not implemented.
  Root reviewed the draft and required core-cap enforcement before allocating
  all disconnected blocks. Numerical computed-normal cone review is pending.
- `scripts/v3_native_context_collection.py` and its
  [protocol](2026-09-09_native_context_collection_v3_protocol.md): both cached
  channels, no cache fill or native-invalid bypass. Independent Astra review
  found arbitrary scalar-to-status coercion and swallowed native-property
  budget exceptions. Root repaired both before execution and qualified the
  inherited tolerance field as unverified native units. Independent tests are
  owned by `/root/noise_plane_math`; eight fresh native cases remain unopened.
- `scripts/v3_native_context_test_supervision.py`: new-route-only sampled
  115-second work deadline/120-second absolute cleanup deadline, bounded waits,
  retained private cache/output and explicit group-empty confirmation. Independent
  pure controls are owned by `/root/cad_support_lineage`. No test has run yet.
  The current harness has separately named collector-source, supervisor-source
  and native routes. Native requires both source gates plus exact independent
  audit; its mere presence in the harness is not an execution result.

The exact successful173-test harness remains byte-preserved at
`source_snapshots/native_context_coverage_v3_2b45f342/check_cad_guided_measurement.py`
with SHA `2b45f3420949041a0fd4694a2b881db1ae2e2f1ee9b70675fca063b46e692e68`.
This is archival source, not a relocated executable entrypoint. V4 successful
source/test/protocol and every frozen physical attempt remain unchanged.

Independent mathematical review of the working notes confirmed the conservative
conditional covariance cone and fixed-normal offset interval. It corrected the
intersection bound to retain nominal solve residual and supplied explicit TLS-
witness failure and pointwise-versus-joint feasibility counterexamples. The
GraffMatch adaptation warning now uses a nonorthogonal example that actually
changes the shifted metric under independent anchor re-canonicalization; this is
not a counterexample to the paper's transported-anchor premise. No calibration
or execution approval follows from these mathematical checks.

#### First context-collector gates: observation succeeded, coverage gap remains

After [independent pre-run audit](2026-09-09_native_context_collection_v3_pre_run_audit.md),
the two pure gates ran once, serially: collector93 pass/one native skipped in
`native_context_collection_v3_source-rat4ghag` (parent0.429339625s, CPU0.367249s,
peak56328192B, output967edf96a65a6925171e73b9f5f9f7d174c14d690ed27e32792989c9be20e987),
and supervisor50 pass in `native_context_collection_supervision_v3-9vk8vjw4`
(parent0.196238500s, CPU0.17025s, peak49790976B,
outputf5d83e7ddc4b9153227f6aac890280b762c4e74ce5b8f677dee3a64e6ba2f504).
Both honored60s/45 worker CPU/1GiB/1MiB with no stop or source drift.

The subsequently released native gate `native_context_collection_v3-1i3cpnja`
completed once:94 pytest passes (93 pure plus one eight-case matrix), all8/8
observation contracts pass. Parent3.470438916s, worker CPU2.756593s, sampled
group peak323862528B/worker peak323829760B. No120s wall/115s work/90 worker
CPU/1GiB/1MiB stop, source drift, primary or cleanup error. The created worker
group was confirmed empty; its old PID must not be signaled. The private test
cache is intentionally retained at `/var/folders/yp/yrd9ss4s73zdt3d4_2221h580000gn/T/native-context-v3-tests-9wkj9ii9`.
Output3797B SHA `fcf71fe0f48dd0bc52b7a06eeea7ee8170aa7608ed8064d1e052dc8b5269336f`.
All eight exclusive, fsynced, readback-bound artifacts survive, totaling163068B.

The scientific result is **negative for complete cached-context evidence**.
Both read channels agree, as do all four fresh read-order pairs, but both omit
required contexts in every case: box/reversed-box63 observed of79 (12 EDGE/FACE
and4 VERTEX/FACE missing); open-shell56/70 (8 EDGE/FACE,2 VERTEX/FACE,4 VERTEX/EDGE
missing); capped-cylinder13/17 (2 EDGE/FACE,2 VERTEX/FACE missing). The open shell
retains one native context error and global-invalid; the other constructs remain
native-valid. Keyed access therefore does not repair this by avoiding iterator
export. No absent context becomes NoError or grants local feature admission.
The [independent post-run audit](2026-09-09_native_context_collection_v3_result.md)
is complete: all eight artifacts, source closure, counters and the 100 missing
pairs per channel across eight cases agree. Upstream cached-read interfaces do
not justify absent-to-NoError conversion. The installed cause is not established;
a passive cache-lifecycle source diagnostic is a possible separate successor,
not a prerequisite for the next all-case score.

Exact executed harness39e9fee1a92781e37a8aa14304c86d1a012741a40c77067f577c285238e28342
is archived at `source_snapshots/native_context_collection_v3_39e9fee1/check_cad_guided_measurement.py`.
Only after native termination did root add the new V5 pure route. Collector,
supervisor, tests, frozen protocol and dependencies remain unchanged.

For the next physical score, do not turn universal local admission into a new
prerequisite. The retained V1 diagnostic already produced native ROIs for eight
of eleven selected cases. Qualify the noisy registration/correspondence successor
and carry unsupported remaining cases as explicit abstentions in the same-input
all-case comparison. Continue native coverage research separately; do not bypass
native-invalid evidence merely to raise the score.

#### First noisy-extraction V5 gate: 75 checks pass

After the [independent pre-run audit](2026-09-09_noisy_mesh_planes_v5_pre_run_audit.md),
the named pure gate ran once in
`cad_engine_v1_synthetic_checks/noisy_mesh_planes_v5-poyezgz9`:75 passed, no skips.
Parent0.703134417s, worker CPU0.511843s, sampled/worker peak76726272B; no
60s/45-worker-CPU/1GiB/1MiB stop and all five source hashes unchanged.
Output SHA `148dc354917f68543f4576815dcc8a45fda73ec0ca325e1fbec97e86da0fbef8`.

Source `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75`,
test `56c12020f3ae77dfafb7525b94d9ab823a9e2f630b98537ef1bac1434f715580`,
protocol `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b`,
math notes `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52`,
harness `61a04fddcd84f79856a24871f8ccb05c9593c3a0f819eb6c03c85e3cc46996c2`.
The four required noisy frames/scales and the actual 132-triangle failed-closure
core-preservation fixture ran unchanged. Critical eigengap, computed-normal,
heterogeneous error, strip-witness and pointwise-versus-joint counterexamples,
immutable original lineage, holes/disconnected supports and late-failure controls
also passed. Independent terminal-receipt review is complete:33 test functions
expand to75 cases, the whole file ran without filtering/skips, and both required
fixture families executed. The log is aggregate, not stored per-case geometry;
do not invent more precise values than the tested limits. No rerun is needed.

This advances only the invented-array proposal contract under stipulated,
uncalibrated position-error bounds. No 108-case study, large-scan qualification,
connected noisy registration or physical F1 result is implied. The separately
authored bounded plane-search source remains unexecuted. Immediate next gates
are independent search review/tests, bounded scan-scale/connected measurement
qualification, then freezing and auditing the next all11 paired Gold11 run.
Do not postpone that run until every roadmap target or every unsupported CAD
path is solved. Historical-score comparability remains unestablished.

#### V5 search and connected noisy poses, plus two locked negatives

The [plane-search pre-run audit](2026-09-09_plane_search_v5_pre_run_audit.md)
released one pure gate after independent source/math review and repairs to late
CAD-participation ledgers. `plane_search_v5-atzvsc3k` passed77 controls, parent
0.271061333s, CPU0.221372s, peak55672832B, no60/45/1GiB/1MiB stop or four-file
drift. Source13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98,
testadd36429382e4a5ce50775ef9fd61b068838a083e466493452dcd3f881944c22,
protocol2b0983d9dc6612147d5b1ecc45303f44f819bd2f7f14ed5e30c3b366af4257b0;
outputb5a28add50dadbb4596c6a4bf59902a6310a39ec7c4326492994e78ec60a9e03.
Its b11c0185 harness is archived under `source_snapshots/plane_search_v5_b11c0185/`.
This proves declared-plane candidate controls, not full physical correspondence.

The separately pre-audited noisy capacity source gate
`noisy_mesh_capacity_v5_source-_09yfli0` passed two controls/one explicit large
skip, parent0.603706583s, CPU0.44826s, peak100220928B. Full gate
`noisy_mesh_capacity_v5-ksogonyt` then failed once:1 failed/2 passed, internal
`MAX_METADATA_BYTES_EXCEEDED` before closure scheduling. Exact900000 raw rows,
1800000 triangles,5400000 edge records and2700000 shared edges were supplied;
22452 cores were appended and13847 qualified before all proposals were cleared.
Fit count22453 includes the next fitted, unappended core. No full-size
partition/witness/normal/ten-representative oracle was reached. Parent6.583640833s,
CPU6.545417s, worker peak982007808B, no120/90-worker-CPU/4GiB/1MiB outer stop
or six-file drift. Output957cfa8b576e8d54a05f5566d6c768fda6996e5c08f06ddecd998e56c896f5a0.
[Independent post-run audit](2026-09-09_noisy_mesh_capacity_v5_result.md) confirms
the locked representation-capacity negative. Exact da3828c1 harness is archived
under `source_snapshots/noisy_mesh_capacity_v5_da3828c1/` after termination.

Root's new finite initializer uses the V5 noisy proposals,32 stratified unsigned
assignments/all their proposed signs, full original supports versus complete
finite CAD faces, exact passing-pose groups and48 stratified V2 refinements.
Independent source audit closed a separate negative-ID snapshot race and two
negative-ledger issues before the first test. The [frozen pre-run audit](2026-09-09_plane_initialization_v5_pre_run_audit.md)
binds all19 files. Attempt `plane_initialization_v5-ie1hcgl5` reports40 pass/1 fail,
parent0.816736500s, CPU0.723176s, worker peak87982080B, no60/45/1GiB/1MiB stop
or source drift. Both required independently posed noisy scan/CAD cases passed
actual extraction/search/finite-registration with start<=1e-7mm, clean-pose<=.01mm
and noisy-row<=.011mm action errors under their fixed declared model. No exact
per-case errors were logged, so do not claim tighter measured values.

The overall gate is NOT GREEN: its final100-byte report-overflow control lost
`discarded_hypothesis_records` through double refusal-envelope wrapping.
Root diagnosis is the second `_bounded_report` seeing already-cleared arrays;
independent diagnosis confirms the same evidence-loss mechanism. Sourcec37d2bfa916914c64af74e9edd5638756131914600222c63fe8bf77cfd71d2ee,
protocol593833164918b8dbc555a2b1620c5cc53681f1be5b56cc5d17eec4fba653fb60,
testb1031fda243055cd38b7f35146912a186702d77e62fe590989e8b4981cb3de60,
harness501cab2a2d6c6cb1be565e970523d1257a0fa46ac6634b53831f064fbaff0f52,
output9382cff58d0bdc90d3ab1bf75aa7e956c7256e4a03633ba5ae197036e742a57d.
Preserve all those terminal bytes; do not edit/retry V5 to hide either negative.

Next is a separately versioned lossless schema-table evidence representation,
authored in `backend/cad_guided_noisy_mesh_planes_v6.py` with its own protocol.
It reuses frozen V5 geometry/graph/cones/schedules and unchanged original
capacity fixtures/oracles. The same numeric64MiB cap measures a CHANGED encoded
transport, explicitly not unchanged V5 expanded-JSON admission. Independent
codec caps include256 schemas/128 keys/depth64/64M traversal nodes and256MiB
expanded logical-JSON equivalent; all late fields/counters and exact scalar/order
roundtrips need independent audit. No V6 result exists. Root must then carry
idempotent refusal handling plus the representation into a NEW initializer and
connect the byte/native/measurement path. The unexecuted V5 byte wrapper source
0c81530de9306b689798d7f7c50081bf301d1529f6d29ad00e35a7f7d2931837 has source review
but no protocol/test/native qualification yet. All11 paired Gold11 remains the
next physical milestone; unsupported CAD cases stay explicit abstentions.

#### V6 refusal positive, compact negative, and V7 transport correction

The subsequent [boundary receipt](2026-09-09_v6_evidence_boundaries_result.md)
supersedes the preceding source-only next action. V6 refusal helper
c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e
passes307 independent tests in `report_envelope_v6-b30hskri`,parent.178658916s,
CPU.152494s,sampled47464448B/worker47448064B,no stop/four-file drift.
Exact schema,idempotent/nested first evidence,known/unknown counts and hostile
ordinary exceptions are covered; physical geometry is not inferred from this.

V6 compact source1f5fe2fd447c445efd8c0e12f9947bfe484126fd15b2fe1be1149f2d5baf2f72
then failed78-pass/13-fail in `noisy_mesh_evidence_v6-7ltfj07_`,parent.574812666s,
CPU.400884s,peak77381632B,no stop/nine-file drift. The actual V5 analysis produces
exact NumPy float64 guards; V5 JSON serializes them,but V6 exact builtin-only
encoder rejected them. Independent Astra confirms the source/type mismatch.
V6 source,test and both protocols stay frozen; no capacity run followed.

New V7sourceb0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f
only normalizes exact NumPy float64 to builtin float locally in the encoder.
Decoder remains strict; internal scalar/array bytes,math,epsilon,schedules and
budgets stay unchanged. Exported-tree tests now model actual JSON transport,
with stronger separate internal float64-byte comparison. Independent exact
[pre-run audit](2026-09-09_noisy_mesh_evidence_v7_pre_run_audit.md) released100
mandatory controls; `noisy_mesh_evidence_v7-tvf90s3c` passed100,parent.386880958s,
CPU.345476s,peak74973184B,no60/45/1GiB/1MiB stop or13-file drift.
Output1559b2c0a6532c4f3d738a60ce0dccbda7b4573d960e59ea865b98080a215741.

Harness501cab2a and76a73cdb are archived exact under `source_snapshots/` for the
earlier terminal attempts. Current harness33b85e3434d81b66c9a1c3cb6f41aa40e382159d7aaf8be79faecd5c4b092132
adds V7 pure routing and retargets only unexecuted capacity/initializer drafts.
Next: independently audit/execute unchanged900k/1.8m capacity and43 initializer
cases,then the finite byte/native/measurement gate. No physical score yet.
The four-case native draft retains original three cases/gates plus one bounded
noisy nominal; all use a pre-execution stipulated.005mm proposal error bound.
The initially proposed.05mm bound was rejected analytically before execution:
thin cap-patch covariance.1066667mm² implies about17.83deg cone versus10deg
limit; .005mm yields about1.75deg and exceeds the invented .001mm noise. This
is not physical calibration or a chosen Gold11 noise parameter. No cone or
original geometry/measurement gate was widened; native draft remains unexecuted.

#### V7 transport closes the unchanged full-size capacity and initializer gates

After exact independent audits, `noisy_mesh_capacity_v6_source-xzu8djk7`
completed2pass/1declared skip, then `noisy_mesh_capacity_v6-3ym9gf8g` completed3pass.
The full900000raw/1800000triangle case retained all partitions and passed every
qualified support/cone/lineage check. All24237cores evaluated;14854qualified;
all ten constructed faces have qualified CORE support. All16scheduled closures
were rejected and14838qualified cores were deliberately unscheduled for closure.
Encoded43153466B/expanded72541767B metadata,15572663visits/699725920hashedarrayB.
Parent24.930094833s,CPU24.877765s,sampled1022672896B/worker1025605632B;
no120/90/4GiB/1MiB stop or12-file drift. Full output
82fb69b4098618ade5a3644e5f66382fce366c7163e41bb56c12b2fd12b42dfa.
[Result and scope](2026-09-09_noisy_mesh_capacity_v6_result.md) preserve the old
expanded-metadata negative and distinguish the new compact-byte metric.

`plane_initialization_v6-v65a3fq7` also completed43pass under the independently
frozen28-file closure. Parent1.049947084s,CPU1.011105s,sampled86310912B/worker88260608B,
no60/45/1GiB/1MiB stop/drift. Both actual noisy-pose positives retain unchanged
1e-7/.01/.011mm limits; new compact and double-refusal evidence checks pass.
Output8dcb18b8e5971f1b556dea7ea2b9b441108b7e69195a410b05f5fdbdba80827c.
Executed harness33b85e34 is archived byte-identically under
`source_snapshots/compact_capacity_initializer_v7_33b85e34/` before native routing.

The remaining finite engineering gate is now the four-case byte/native/full-raw
measurement path. Pre-run review is correcting test evidence only: represented
binary64 noise displacement (with fixed.999 amplitude reserve) and complete
passing-group origin partition. No engine threshold or original case gate changes.
Then freeze/audit the next all-eleven-case physical paired development baseline;
do not add universal CAD admission or broad research as a prerequisite.

#### Connected V6 regression: candidate starvation before measurement

After independent final source audit, the35-source-check/1native-skip prerequisite
passed. Full `mesh_connected_planes_v6-58gybuu9` then completed35pass/1failed
matrix test: actual1/4, only missing-arc completed-prefilter abstention passes.
Parent6.385630667s,CPU5.567089s,sampled/worker436977664B; no180/120/4GiB/1MiB
stop or48-file drift. Four artifacts total18711722B are independently hash/size/
denominator verified. OUTPUT bbc09f61000b956204a09ed9395012a3066f6bf8ccfae5e254a5fa058a232571.
Full evidence and critical attribution are in
[the terminal negative](2026-09-09_mesh_connected_planes_v6_result.md).

All four have128 scheduled signs/384 fully queried assigned supports, zero
numerical failures/passing groups/refinement/predictions. Native metadata has one
available feature in each. Best generated clean pose error is1.4142135623730825mm,
noisy1.4151783228480563mm; none meets.20mm. Clean best scheduled cap support5/15
correctly refuses at unchanged.95 coverage. This is initialization search failure,
not native feature availability or measurement corruption, and not just outer quota.

Retained masks show94 clean/88 noisy compatible joins repeatedly choose the same
four tuple ranks [0,19,39,59] from60. The earlier12-pair selection also misses
relevant combinations. Independent review supports a source-only fair coverage
successor: integer rank bins rotated by occurrence of an EXACT repeated pair
window/compatible layout, no geometry merge or case labels; all existing geometry,
12pairs/4unsigned per triplet and8192signed caps remain. Prospective fairness is
across repeated layouts only, not exhaustive matching or guaranteed pose recovery.
Root/independent author are reviewing bounded key memory and integer coverage.

`scripts/run_gold11_mesh_development_v2.py` is an unexecuted source-only draft
SHA5ea870e8bc7d4139f74b740a35a5a4778e8311212b22ea0c0c66be9b63013146.
It prepares controlV1workflow/nativeV1 versus challengerV6workflow/nativeV1,
unchanged scoring/registration/measurement gates, explicit defaults, uniform
exploratory.05mm proposal bound and equal128MiB result transport. That bound is
conceptually permissible only as stipulated/uncalibrated, not scanner evidence;
larger bounds can remove small patches via wider cones. Draft has no tests,
protocol freeze, source audit or physical execution; pending candidate may change
before first freeze. It must not be run from this checkpoint or treated as ready.
The goal remains active; no new physical score or engine promotion is claimed.

#### V7 measured noisy positive and remaining outer-screen omission

The independently reviewed fair-search V6 pure gate passes115, V7 initializer43
and V7 byte-workflow source35/1declared native skip. All three terminal receipts
are independently verified, no resource stops or source drift. Full connected
V7`mesh_connected_planes_v7-n1y2vlbi` is a retained2/4: missing-arc safe abstention
and one genuine noisy measured bend pass; both clean nominal cases still fail.
Parent8.090838333s,CPU7.754568s,peak449937408B,58-file consistency, no stop.
Four complete artifacts total19055129B. See
[the V7 result](2026-09-09_mesh_connected_planes_v7_result.md).

Noisy radius error0.0000709769mm, sweep0.000299622degrees, generator endpoint
0.000761224mm, axis endpoint0.000755740mm, pose0.002069763mm are actual
constructed output progress, not physical accuracy. Clean exact proposals are
now present (h31/h255, errors~4.45e-14/5.49e-14mm) but omitted by outer32 unsigned
strata. The sole finite-passing wrong pose is correctly globally refused; no
clean measurement runs. Root and two independent reviewers agree on attribution.

A new UNEXECUTED V8 initializer removes the outer rank selection: all proposed
signs from the already bounded search must be finite-screened, or the whole
obligation is refused. Max eligible unsigned IDs1024 is now a HARD bound, not a
quota; smaller limits refuse after retaining the full known hypothesis ledger,
before support/BVH/query work. Existing64M rows/32768 calls and all finite-work/
report/RSS/CPU limits remain, as does48 exact-group refinement. Native search
and final refinement are still incomplete. Retained clean/noisy counts give
loose row obligations12.15M/11.62M below64M, not a CPU or physical capacity proof.
No new rank is chosen from truth, no cap of a started experiment changes.

Sources: initializerV8`cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b`,
wrapperV8`afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec`.
Independent tests/protocols and final source review are in preparation; no V8
execution is authorized by this checkpoint. Harness874 used by V7 is archived
exactly at`source_snapshots/fair_search_v7_87478a2c/`; prospective new routes
currently have harness6d58919f8b5582ec7611751b6ca262c7060a0a099b015dcd644965852f17632d.

The unfrozen physical V2 runner was source-only retargeted to V7 before any
freeze: SHA96bd4bef691997171b4bca482b61019ac7c1460957d17e8af29ccb382f4ab95b.
Its source/mock tests are being authored in parallel, but candidate may change
before first freeze. The failed connected gate prevents its current dispatch.
No physical run, new baseline, integration promotion or goal completion claimed.

#### V8 connected gate closed; proceed to the paired baseline

V8 source is now frozen and independently pre/post-audited:53 initializer
passes,37 byte-workflow source passes/1native skip, then actual4/4 (38 tests).
`mesh_connected_planes_v8-nu8mjr21` retains all four complete artifacts28,681,798B.
Parent30.99921162519604s,CPU30.850386s,peak450363392B,no stop/64-file drift.
Both clean positives now measure correctly; noisy pose0.0017310043mm,
radius0.0000764727mm and endpoints~0.000761mm meet unchanged criteria.
Missing arc is nonempty completed geometric registration refusal, not absence.
All1472/1440/1472/1408 proposed signs and4/10/4/8 passing groups are evaluated,
with zero numerical failures and no outer/refinement omissions. Initializer/
wrapper/harness hashes above stay unchanged. Detailed independent audit:
[V8 result](2026-09-09_mesh_connected_planes_v8_result.md).

The physical V2 draft was narrowly retargeted BEFORE any V2 test/freeze to
V8/nativeV1 versus V1/nativeV1, hard1024 all-generated screening; same scorer,
common configs, uniform hypothetical0.05mm and both-arm128MiB full-result cap.
Current runner SHA`94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422`;
pure test`d13df5c2eec485fdc4640cd1ba0f8f20d40e4aca3869e23e85eeb63edccfce63`,
test protocol`ef2e7c4b72cbefde98c1e83ae887c27a2c6636082db5e446e4cafa557475c4f6`.
The89-control/1unconditional-skip source gate passes and is independently
post-GREEN: `gold11_mesh_runner_v2-0s3nl07n`,parent0.608464875259s,
CPU0.574698s,sampled53,870,592B,no stop/six-file drift. Its OUTPUT is
`512b84bcda96dc2ddec4b4e06e8678628f3351ae61360480a30d93c8b87b80ab`.
A separate tiny guarded-runtime import/constructor smoke is under final review.
The first unexecuted draft has two pre-run corrections: retain the bounded child
log/receipt even on failure, and defer runtime-version metadata to the physical
freeze rather than falsely claiming this smoke captures it. Root added a fresh
parent-owned retained ARTIFACTS destination; current UNEXECUTED harness is
`4f2a778150fa106d7e9c8155333a12d010eace0020b2e9c39f9836b7be8cce19`.
Actual-only opt-in and same-group60/45/1GiB outer,20/10/512MiB child remain.
Exact used6d589 harness is archived at `source_snapshots/all_generated_v8_6d58919f/`.
No smoke attempt or physical freeze/run has started; new test/protocol final
review and27-file closure freeze precede first source then actual smoke.
The final prospective physical protocol must describe actual phase-specific wall
limits accurately: the180s worker allowance subtracts preparation, but post-worker
retention uses parent/global guards rather than a hard180s whole-case deadline.
No started budget or retained result changes. Then freeze/audit the exact complete
bundle and run all11 cases promptly. No universal-CAD/new-simulation prerequisite.
No physical score yet; goal remains active and product defaults unchanged.

#### Physical V2 completed: dominant finite-index admission failure

The tiny guarded-runtime gate completed32passes after31sourcepasses/1skip;
27 source bindings stable and independently post-GREEN. Physical V2 then froze
bundle `077d782626fe3308da6221b5b8a0180af62898703da700ddcb72224b2ad42005`
and executed once after exact-bundle independent pre-run GREEN. It is now terminal,
not rerun-ready or in progress. Both arms retain11refusals,0lines,TP0/FP0/FN71,
recall0 and undefined precision/F1. No score gain or physical count evidence.
Parent367.95208854135126s,sampledcomplete-run1,919,434,752B,22durableterminal
outcomes, no outer resource stop or source drift. See
[full V2 negative result](2026-09-09_gold11_mesh_development_v2_result.md).

Independent diagnosis of all11 challenger RESULTs finds9assigned-face index
`UNSUPPORTED_TRIANGLE_ARITHMETIC_CONDITIONING` failures before firstquery,
one20Mtriangle-membership closure cap, and one all158triplets ill-conditioned
search. No challenger finite query, registration candidate evaluation or
measurement executes. The9indexfatal ledgers retain allsigned hypotheses and
schedule everyPROPOSEDsign; zeroactualqueries is not rejected geometric support.
Full custody post-audit is now GREEN as validnegative/no gain/no promotion,
verifying29sourcebindings,22resultbindings and the complete scoring barrier.
FrozenV8/V2source/protocols/results remain
unchanged. No physical/source replay or new input needed for this diagnosis.

Next bounded source-only mechanism is locally anchored/scaled triangle arithmetic
with per-primitive conditioning rather than soup-wide extrema coupling. Critical
math and independent exact-oracle tests are under review; no relaxedguard,
triangle removal or successful physical recovery is presumed. A separately
versioned connected candidate and new frozen/audited physical protocol follow
only after the mechanism proves its declared synthetic contract. Keep all11
cases/71references and historical-comparability/measurement-truth limitations.
The autonomous goal remains active; default product routing stays unchanged.

#### V3 arithmetic successor: current source-only design review

No V3 triangle source/test has executed or been frozen. Independent reviewers
recommend per-primitive anchored power-of-two charts, separate local-distance
and world-coordinate output-rounding error, and pair-specific distance ranges.
The nearest/tie BVH must use node lower bounds and candidate upper bounds;
a winner-only guard can incorrectly prune a numerically uncertain competitor.
Keep all primitive identities and reject the whole query on incomplete work.
No exact/interval certificate is implied by an empirical engineering guard.

A connected caller issue is now explicit: V2 `_chosen`/`_diagnose` and V4/V8
finite screening gate raw `distances` without consuming `numerical_distance_guard`;
refinement recomputes distance from returned world-coordinate closest points.
Local distance accuracy alone cannot establish interval-safe downstream gates.
The successor must bound/disclose restoration disagreement and test this path,
not simply swap the index constructor or suppress an admission refusal.

The independent proposed oracle converts represented float64 coordinates exactly
to rationals, solves all seven nonempty simplex active sets by generic exact KKT
elimination, and checks the global convex-projection inequalities. This avoids
copying the runtime chart formulas. Proposed fixed small matrix:24single-triangle
scenarios ×7queries;6mixed17-triangle soups ×12queries with reversed-order repeats;
8boundary/tie/hole scenarios and8refusal scenarios, ≤2048distinct oracle pairs.
Proposed pure gate60s wall/45s CPU/1GiB/1MiB, no native/physical access. Final exact
source/tests/limits and independent pre-run review must precede execution.
Distance checks use represented local geometry; closest-point checks separately
allow unavoidable world-coordinate rounding. Do not inflate local error by an
unrelated soup's largest triangle or an intended-but-unrepresented exact pose.

Primary mathematical references reviewed for this design:
[Eberly](https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)
formulates finite-simplex minimization; [Shewchuk](https://www.cs.cmu.edu/~quake/robust.html)
addresses adaptive exact predicates/error-free arithmetic. Neither certifies
this unimplemented distance kernel or its downstream registration. Detailed
supported numerical-regime review is still in progress; no recovery claim.

Preferred first prototype after that review: a bounded local primal/dual check
of candidate finite-triangle points, rather than declaring a4096epsilon envelope
certified. For a genuinely feasible barycentric point q, r=p-q, U=norm(r) and
h=max_i r dot(v_i-q), the Cauchy/support inequality gives
d(p,T)>=max(0,(U^2-h)/U), with U an upper bound and the r=0 case separate.
This is a source-review derivation, not an implemented floating-point certificate.
Outward evaluation, exact feasibility, subtraction/scaling and exceptional-range
handling must be independently established. A node's outward AABB L-infinity gap
is a conservative Euclidean lower bound without a soup-wide hypot guard. Use
minimum candidate upper bound U* for first-pass pruning and every competitor's
own lower<=U*+tie band in the second pass. Neither distance bounds nor an AABB
bound establishes normal, native-CAD, returned-point or registration correctness.

#### Local distance helper implemented and passed, full search next

The first source-only prototype is now frozen and independently pre/post-GREEN:
`backend/cad_guided_triangle_distance_bounds_v3.py`
SHA2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7.
It accepts exactly feasible represented barycentric candidates and computes
outward local primal/dual distance enclosures plus separate world-restoration
error; it is not itself a minimizer or a normal/region certificate. Original
nonzero subnormal inputs refuse; intermediate gradual underflow is enclosed
under the explicitly conditional IEEE model. Model sentinels are not attestation.

Independent test09deb22ee510fe62e1c69dd1469005ffb949fbea25179755bf713998d2ab254a
passes87tests/122exact-rational oracle calls, no skip. Actual attempt
`triangle_distance_bounds_v3-c9yh4uq6`:parent0.3152538752183318s,
CPU0.275642s,worker/sample54,591,488B,no60/45/1GiB/1MiBstop/four-file drift.
OUTPUTd364bc00f83073a6a725b947d4f3a48d7e1e22a8885e4688c29fa76b91314a69.
Protocol5341d7cb03ae6adb6a43d5319fb7560bf3c09e17fa96bf531e06e2ff59bc85e8,
usedharness66eb2f5877a4bc9c49d1bbaf07e09d8d3af4b8db24d2ddcdbb9f8c59c2a68644.
Prior4f2usedharness is byte-identically archived under
`source_snapshots/guard_smoke_v2_4f2a7781/`; archive66eb before next harnessedit.
Detailed [pre/post receipt](2026-09-09_triangle_distance_bounds_v3_pre_run_audit.md).

Next source draft owner `/root/cad_support_lineage` implements
`backend/cad_guided_triangle_surface_v3.py`; root owns independent
`tests/test_cad_guided_triangle_surface_v3.py` (46prospectivecases; unexecuted).
Reviewer `/root/cad_measurement_integration_review` reviews both independently.
The new finite search uses minimum feasible upperbound U*, conservative node/
primitive AABB lowerbounds and a complete fixed-U* secondpass. Per-triangle width
admission uses outward width<=1e-5*thattriangle's minimumedge on POTENTIALLY
COMPETITIVE triangles only: valid certificates with lower>U*+tieband prove
irrelevance regardless of width. Unsupported certificates without an excluding
valid bound remain fatal. This is proof-based pruning, not dropping primitives.
Preserve identities, normals/region engineering caveats, everytie and whole-query
clearing. Certified-distance bounds still need explicit downstream policy handling;
the frozen V2caller ignores numericalguards and cannot inherit certification.
No new full-surface test, connected test or physicalrun has executed. Goalactive.

#### Full V3 finite search now passed; caller integration is next

This receipt supersedes the source-draft/46-case next-action wording above.
Final source8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10
passed the independently pre/post-audited52-test public-search matrix, with543
exact KKT calls. Teste0d0be929915829c95c43e794127818d8a9c1b7fede18981926844c71c09e1bc;
protocola47c9b3f591e91e52f3e6e801209a7df895792ac0e68c3098db2c369319269fd.
Attempt`triangle_surface_v3-cdxuhpo6`:parent0.7578618340194225s,
CPU0.727166s,worker/sample62,881,792B,exit0,no60/45/1GiB/1MiB stop or
six-source drift. OUTPUTb552c8f5a3f79f7324e67c509dc7864d020dcfce717052fbf5b5489090c6e201.
Detailed [pre/post audit and immutable closure](2026-09-09_triangle_surface_v3_pre_run_audit.md).

Pre-run review added exact tie boundaries, index/region assertions and separate
certificate/work-budget controls. Root repaired a valid-but-weak first-pass lower
bound surviving into the final envelope after its triangle became AABB-excluded.
The complete fixed-U* second pass now rebuilds that envelope; minimizing triangles
cannot be pruned, so soundness is preserved without obsolete width contamination.
A valid-lower weakening control and exact-oracle checks cover the repair.

The helper, surface and their tests/protocols are now frozen/terminal. Used66eb
harness is preserved in`source_snapshots/triangle_bounds_v3_66eb2f58/`; used025ae
surface harness is now byte-identically preserved in
`source_snapshots/triangle_surface_v3_025ae1d4/`. No need to rerun unchanged suites.

Source-only integration mapping identifies the minimal additive seam:
new global-registration consumer, V9 initializer and V9 byte-workflow wrapper.
Frozen V4`_FaceQueries`, V8 prefilter and V2`_Queries/_chosen/_diagnose` hardbind
old surfaces and nominal-distance gates; an import-only swap is insufficient.
New support/final gates must consume distance bounds, uncertain normal/point
flags and region-rank exclusions. Finite boundary support need not require a
FACE rank role. Preserve V2's materialized-candidate Kabsch/Huber surrogate as
explicit engineering optimization if reused; do not mix its fixed-q objective
with local candidate-distance certificates and claim a common certified loss.
Aggregate certificate and triangle-box work must propagate through all callers.
No caller implementation or connected run has yet occurred. PhysicalV2 remains
terminal/no gain; next physical successor requires its separate frozen gate.

### Mathematical and integration boundary

Implement bounded closest-point queries on finite oriented CAD triangles, not
distance to sparse sampled points or unlimited supporting planes. The constrained
triangle quadratic reaches its minimum in its interior or on finite boundary
segments; see [Eberly's point-to-triangle derivation](https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf).
Use deterministic AABB pruning, scale-aware arithmetic, explicit node/triangle/
tie/normal-pair operation budgets and whole-query refusal on overflow or numerical
unsupportedness. No partial successful prefix may escape.

Nearest-triangle ties must separately expose incompatible normals, distinct
closest points and face-tag ambiguity. A nearest finite edge or vertex cannot
silently use the facet's plane residual: a point outside a triangle but inside
its plane has zero plane residual and positive finite distance. Registration must
handle boundary constraints explicitly. Native tessellation remains approximate,
not analytic CAD distance or calibrated metrology.

Before physical use: independent pure-geometry and adversarial tests, native
triangle-export checks, a connected fixed synthetic comparison, critical math
review and a new frozen all-case protocol with independent pre-run audit. Keep
the V1 physical acceptance thresholds; do not select carrier, pose or thresholds
using Gold. Diagnose the accepted case's native invalidity separately without
silently healing input. No favorable outcome is presumed.
