# Connected plane-initialized V4 — terminal constructed result

Date: 2026-09-09. Independent post-run reviewer: /root/astra_repo_guidance_review.

**Audit decision: valid positive constructed mechanism result, 3/3 fixed cases.**
Both nominal branches produce the required one scan-measured finite instance;
the missing-arc case produces a completed geometric registration abstention.
The retained V2 predecessor remains 1/3. This is not physical Gold11 improvement,
unseen-family accuracy, observed-absence detection, or product/floor promotion.

## Evidence and custody

The separately reviewed [prospective protocol](2026-09-09_mesh_connected_planes_v4_protocol.md)
was followed by source controls and one serial native matrix:

| Retained attempt | Terminal result | Parent wall | Sampled group peak RSS |
| --- | --- | ---: | ---: |
| [V4 source controls](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4_source-5kif0r8_/COMPLETED.json) | 48 passed, 1 native skip | 0.705014 s | 90,374,144 B |
| [V4 connected native](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/COMPLETED.json) | 49 pytest checks passed; actual matrix 3/3 | 42.895575 s | 428,785,664 B |
| [Retained V2 negative](cad_engine_v1_synthetic_checks/mesh_connected_surface_v2-lqfsnygb/COMPLETED.json) | matrix 1/3; process exit 1 | 91.725850 s | 396,230,656 B |

V4 worker CPU was 41.986215 s. Source and native receipts report no budget stop
or before/after source drift. Native limits remained 180 s wall / 120 s worker CPU
/ 2 GiB sampled process-group RSS / 1 MiB stdout; source controls used 60/45 s and
1 GiB. These are the reviewed sampled harness limits, not hard real-time or
tamper-proof operating-system attestation. The wall figures are observations,
not a controlled general speedup claim.

The native [STARTED receipt](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/STARTED.json)
and terminal receipt agree on all 28 source hashes and all resource/artifact limits.
The source-control closure is the same. The compact [native output](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/OUTPUT.txt)
is 7,172 bytes; its independently recomputed SHA-256 is
d5215d53bdf010b092772a67158870423ac026c8d7ff3fa87f91df1ab377bb67,
matching COMPLETED. Source-control and V2 output hashes also match their terminal
receipts. No retry, dropped case, changed gate, successful-prefix substitution,
or budget shift is present in this reviewed attempt record.

Exactly the following three files exist in the new ARTIFACTS directory. Each
parses as a complete case record; its case, result, exact byte length and SHA-256
match the compact manifest. Total retained case bytes are 14,706,783, below the
prospectively separate 48 MiB allowance; every case is below 16 MiB.

| Artifact | Bytes | Independently verified SHA-256 |
| --- | ---: | --- |
| [case_00.json](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/ARTIFACTS/case_00.json) | 4,921,380 | 1656c22327e319fa024647b06693980f5a9b28e7aaaf8c7dbfb2eac3446d3ec5 |
| [case_01.json](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/ARTIFACTS/case_01.json) | 4,866,127 | da5ceb30742a930ac558e95c63a02b3841182a5eeba47908e580b4dd2252c4f7 |
| [case_02.json](cad_engine_v1_synthetic_checks/mesh_connected_planes_v4-epppmtsj/ARTIFACTS/case_02.json) | 4,919,276 | 42c3f892696da38a106e092b1f8a69e6b82a649da153706a33c0c27075df8e80 |

## What changed, and what stayed comparable

The new mechanism derives exact-mesh plane patches and plane-triplet starts,
screens every assigned finite-face support, and feeds surviving starts into
unchanged V2 coarse/fine refinement, global diagnostics and final pose selection.
It has no PCA fallback, oracle pose injection, nearby-start clustering or
result-dependent threshold change. The original nominal assertion function is
called directly; the common legacy assertion changes only its PCA-specific
candidate_count==24 requirement. The separate completion/sentinel safeguards
were tightened prospectively, before execution, to reject fatal, numerical,
resource, binding and publication failures as apparent negative-control passes.

All 17 V2 closure paths occur in the V4 closure: 16 have identical hashes; the
reviewed harness version is the only changed shared path. The new workflow,
plane mechanism, topology adapter, tests and protocols add 11 paths. Direct
case-by-case comparison of retained payloads confirms equality of all existing
workflow/registration/measurement/support policies and surface/registration
budgets, including 32,768 CAD samples, 3,000 registration-row limit and all numeric
acceptance gates. Native feature metadata, unit receipts, native mesher facts,
CAD triangle/sample digest dictionaries and sampling coverage are also equal.

The original analytic fixture and helper hashes are unchanged. Both nominal
cases contain 2,752 raw scan rows/5,080 faces; the missing-arc case contains 1,389
rows/2,504 faces. Scan byte hashes and input row bindings exactly match V2:

- Nominal scan SHA-256: 578f76a6feca92eb19328dc3d57434a64020640e4753cf29e3290e56a7e02cdc.
- Missing-arc scan SHA-256: 87f40bd9c9f264c139f8edad12e835e01a7cba7fa8f5ade5592f8217d6583f80.

The independently serialized invented STEP bytes are **not identical**:
V2 CAD SHA-256 is 11d31e14e1bee64a9112d5afde48578771265b8f7a4adff9731b23f8ad2487d2;
V4 is c3732fa20adb3e91923e8a8342c9c5407cac99667aa2bede7bfbcb6059509d0b.
Serialization headers can vary, but the original STEP bytes were not retained
here and their precise byte difference was not inspected. The stronger available
geometry evidence is equal native metadata and exact downstream array digests:
10 faces, 24 edges, 276 triangles, 296 mesh-node occurrences, 32,768 samples,
identical oriented-triangle digest 4c8f198b4cb401c12f364d6e51face780e003d6e25daa1919360f3558d4f457f
and sample-point digest 3c34ca5d6f39250da2491503d520b0bc3090ca298c69517a396b54aae01c7a89.
This supports constructed geometric comparability, not a same-byte paired CAD or
physical experiment claim.

## Fixed case outcomes and unchanged gates

| Fixed case | Retained V2 | V4 |
| --- | --- | --- |
| v1_nominal_oblique | No prediction; required nominal failed | One measured finite instance; passed |
| v1_missing_arc_oblique | Zero predictions; abstention passed | Zero predictions; completed geometric registration abstention passed |
| r0_common_wrapper_nominal_oblique | No prediction; required nominal failed | One measured finite instance; passed |

Every V4 artifact retains all 1,728 hypotheses, exact IDs 0–1727 and all three
assigned-support records per hypothesis. All successful/attempted query counts
equal their complete positive support denominators. Each case has 1,720 completed
prefilter rejections and 8 passing hypotheses/start groups; all 8 groups are bound
to their initialized poses and hypothesis IDs, attempted and fully diagnosed.
Candidate numerical failures are zero. All 1,728 post-inference hypothesis action
errors and all 8 refined-candidate action errors are retained, not only the winner.

For each nominal, four refined candidates are eligible in one final pose class;
candidate 5 is selected without a comparable distinct competitor. Its scan
coverage is 0.99818314 against 0.95; CAD-sample coverage 0.3642578125 against 0.30;
finite-surface p95 is 0.001479893 mm against 0.05 mm. Full/fit/weighted observability
ratios are 0.19603655/0.21006843/0.21006843 against 0.005. Four other candidates retain
their surface-residual refusals. No failed candidate was relabeled eligible.

The best post-hoc nominal V2 candidate 20 had maximum pose-action error 1.93759729 mm,
CAD coverage 0.25909424 and surface p95 0.47009759 mm, failing the unchanged coverage
and residual gates despite adequate local rank. V4's selected nominal maximum
pose-action error is 7.842959116e-6 mm, below the original 0.20 mm ceiling. Its best
initial hypothesis already has post-hoc action error 2.647084621e-14 mm. This
supports the declared initialization mechanism on this fixture; oracle errors
are evaluated only after inference and are not the runtime selection rule.

The missing-arc case does **not** use the newly allowed no-start prefilter branch:
eight starts survive and all eight fail the unchanged CAD coverage requirement.
Their CAD coverage ranges 0.28955078125–0.295166015625, below 0.30. The exact final
reason is NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE, with no pose, measurement or
prediction published. Its near-exact best post-hoc pose does not override that
refusal. The disposition is
COMPLETED_GEOMETRIC_REGISTRATION_ABSTENTION_NOT_OBSERVED_ABSENCE, not missing-bend
detection, a local-measurement pass, or a fatal-error loophole.

## Actual scan measurement, not CAD answers

The source-audited workflow imports one owned native shape for both CAD sampling
and feature proposals, cross-binds scan bytes/raw triangle rows, and sends all
original raw positions through registration alignment into the unchanged local
fitter. CAD supplies search proposals, not measured radius/sweep/carrier answers.
The independently analytic scan is not generated from the CAD tessellation.
Known fixture pose and nominal truth enter only construction and post-inference
evaluation; neither is an inference argument.

Both nominal records contain one completed measured window with 1,073 unique arc
rows and 261 unique rows for each flange. This audit independently recomputed
their little-endian int64 support SHA-256 values, checked index bounds/uniqueness
and arc/flange disjointness, and recomputed radius/sweep errors and analytic
scan-frame generator/axis endpoint errors using only retained values and scalar
arithmetic. No inference, original fixture generation, native import or fit ran.

| Nominal error, identical in both branches | Retained value | Original ceiling |
| --- | ---: | ---: |
| Radius | 7.993606e-15 mm | 2e-5 mm |
| Absolute sweep | 1.421085e-14 degrees | 2e-5 degrees |
| Generator endpoints | 1.380082e-13 mm | 0.02 mm |
| Axis endpoints | 1.369063e-13 mm | 0.02 mm |
| All-row rigid-pose action | 7.842959e-6 mm | 0.20 mm |

Measured radius is 4.999999999999992 mm and sweep 90.00000000000001 degrees.
Independent scalar generator recomputation is 1.354698e-13 mm; the negligible
difference from the retained NumPy oracle reflects floating-point evaluation
order and does not affect the gate. This exact fixture has nominal CAD and scan
radius both 5 mm; these two nominal cases alone are not a new empirical
off-nominal/no-copying experiment. The no-copying conclusion here is limited to
the reviewed unchanged measurement data path and retained scan-support evidence.

## Source identity and subsequent work

The terminal run used:

- Workflow SHA-256: b737678bf61400d60d6d5e01fc32b6d29add7e6587e4a1832bcd1c39ecf030f4.
- Test SHA-256: fd69d6a9361ec32741ea840c2681120671bd5a42d74907d0c24c10a3be03a3df.
- Protocol SHA-256: ae1ae584cd2e92bfe6a734cfccfd612d03c8ed50406353058120b1c731371c98.
- Harness SHA-256: f80372ee3df0ef9ade3c77caade3f2d0cc34840d61036cc7c959d58db70a126b.

At post-run audit, 27 live closure files still match. The harness changed only
after terminal completion: root added the independent native_context_coverage_v3
pure suite's map/bindings, 60/45-second 1-GiB tier, namespace and retained-receipt
membership. Its then-current SHA-256 is
2b45f3420949041a0fd4694a2b881db1ae2e2f1ee9b70675fca063b46e692e68.
The reviewer inspected that exact narrow delta and verified the preserved
[run harness source](source_snapshots/mesh_connected_planes_v4_f80372ee/check_cad_guided_measurement.py)
still hashes to f80372ee above. This is documented post-run evolution, not in-run
drift. The snapshot is archival: its relative ROOT assumes the original source
location, so it is not a direct execution entrypoint from the snapshot directory.

## Qualification boundary

The evidence advances this consumed, exact constructed initialization/measurement
mechanism. It preserves the original V1/dense/V2/V3 negatives and does not promote
the product or assert a physical metric gain. The two nominal branches reuse the
same constructed geometry; they are not independent families. The r0 branch is
an r0 feature extractor in the new common wrapper, not the historical incumbent
or untouched frozen R0 pipeline.

Observed bend count remains unknown; conformance is indeterminate, uncertainty
uncalibrated and production use prohibited. Units/stable datum/acquisition side,
physical trim endpoints, finite-carrier compatibility and noisy/general mesh
coverage remain unqualified. Exact-support prefiltering is nonexhaustive: a
rejected start could have refined successfully. A future physical-development
run still needs its own applicable authorization, frozen closure/policies and
independent audit; this note supplies neither a retry command nor a release gate.

This reviewer performed retained-JSON/source/hash analysis and authored only this
new note. No native/inference rerun, original CAD/scan/Gold/held-out/email access,
frozen-source edit or terminal-receipt mutation occurred.
