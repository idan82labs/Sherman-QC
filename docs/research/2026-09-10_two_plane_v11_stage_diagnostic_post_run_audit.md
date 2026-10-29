# V11 stage recording: independent retained POST

Date: 2026-09-10. Independent AI reviewer: `production_architecture_review`,
not the runtime, diagnostic-test, protocol or harness author.

Decision: **GREEN_VALID_DIAGNOSTIC_RECORDING_ORIGINAL_NOISY_POSITIVE_FAILED**.
Blocking findings for diagnostic recording: none. The original positive outcome
is **1/2**, not two successes. This does not qualify the engine, release another
execution or constitute human manufacturing approval.

## Exact retained attempt and custody

First-once attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_v11_stage_diagnostic-6yfh9z2j`.
Root observed parent exit0; retained COMPLETED independently records child exit0.
This reviewer did not execute or repeat the diagnostic.

| File | Bytes | Independently verified SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 6,195 | `38da319fce65cd59c9aad6cca4bcc2d62607ab703f1947021c4772f9c5965532` |
| `OUTPUT.txt` | 861 | `fe36e7fcc7395141367d96e27c51e960d7cf988f70fe6ba97793109fea0737b4` |
| `COMPLETED.json` | 6,377 | `96d7d0b259f8e7026744e2d7f7a36e8e8a61ac53eeefcd0be55b855fbda45e9c` |
| `ARTIFACTS/case_00.json` | 938,002 | `7909527ca840a320470688339205743bc22601cf47da6b06647226a2ec2c38fc` |
| `ARTIFACTS/case_01.json` | 1,034,275 | `e89a5443dd4cbf9a10e1aa9bf725dc74b75074322511bc03e7cbb7c9a19d5dd2` |

Inventory is exactly these five regular nonsymlink files and the ARTIFACTS
directory: 1,985,710B total, including 1,972,277B of case JSON. No extra, partial
or missing file was found. The completion output digest and the two summary
artifact digests/lengths match actual retained bytes. Both case names/order,
the suite/test/harness, owned artifact destination and limits agree with release.

The exact [PRE](2026-09-10_two_plane_v11_stage_diagnostic_pre_run_audit.md), SHA256
`44b12b25bbea74e9267a93068406730d3becaebd0a68dda94579ac6a40e0405d`,
binds all48 files. STARTED and COMPLETED contain the same complete48-file table;
each entry independently matches the PRE and the current source bytes.
`source_before_after_consistent` is true. Test8dbe3e7c, protocolf57ff260 and
harnessb96502ea retain their exact PRE hashes. The prior
[source POST](2026-09-10_two_plane_v11_stage_diagnostic_source_post_run_audit.md),
SHA256 `5efc07bcf9ec0555e0ffb25a50149d9df53cd5d3932ef46187411903c3e0b50f`,
released this single recording only after32PASS/one declared source-route skip.

## Full evidence and observer reconciliation

For each case, all1,082 serialized arrays were independently reconstructed from
their retained values, dtype and shape, without importing NumPy or the runtime.
All raw-byte SHA256 values match: 620 float64,110 int64,320 boolean and32 Unicode
arrays; 269,408 reconstructed array bytes per case. Shape/value cardinality,
finite floating-point values and the actual maximum JSON depth17 were checked.
The recorder's tree-node counts reconstruct exactly: clean65,816/noisy65,780,
both below1,000,000; no truncation or unsupported transport was observed.

Full inputs retain50 scan rows,64 scan triangles,4 CAD triangles and their face/
source IDs. Their array identities agree across input, preparation context and
returned input-digest fields. All50 positional declarations remain .05mm. The
complete registration policy matches the original `_policy(50)` and its source
defaults; the complete search budgets match the original source defaults with
only the predeclared `max_bank_planes=2`. The .5mm correspondence gate and .95
minimum support remain unchanged and distinct from the .05mm positional bound.

Each artifact has exactly one preparation call/context and complete public
answer, with numerical and box-screen completion true. Both retain all64
generated and returned hypotheses; the selected scan bank is [4,1]. Its two
supports retain original raw IDs0–24 and25–49, with no favorable row reduction.

Per case, the actual128 box calls/3,200 rows and32 finite queries/800 rows are
present in complete order. Every attempted index, before/after counter delta,
returned payload and request row count was independently checked. There are no
raised calls, missing primitive payloads, observer errors or reconciliation
errors. All eight runtime attempted/completed counters match the reconstructed
counts. Every two-support hypothesis maps to its actual two box observations;
the32 non-excluded supports map in sequence to the finite-query observations,
with identical face IDs and point arrays. This is retained-call accounting, not
a rerun of a box or finite-surface query.

## Separate original outcome and localized rejection

OUTPUT reports `33 passed in 0.63s`, zero skips:32 source/mock checks plus one
two-case recording test. Its summary correctly sets `qualifies_engine=false`.

| Original case | Original oracle | Passing hypotheses | Start groups |
| --- | --- | ---: | ---: |
| `original_fifty_row_clean` | PASS | 8 | 1 |
| `original_fifty_row_balanced_normal_noise` | FAIL | 0 | 0 |

The noisy first failure remains the original `AssertionError`:
`['NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS']`. The14 original post-call statements
and1e-7mm action-error ceiling were source-bound before recording, not replaced
with a successful-abstention assertion.

Both cases retain action error0 for hypothesis slots24–31. All eight are the
same pose, not eight distinct poses; their rotation/translation equal the
retained expected pose exactly. Pose-action error itself was computed by the
frozen post-inference evaluator and was not recomputed in this audit. All eight
aliases were checked against query records16–31, in face22/face11 order.

| True-pose support evidence, identical across all eight aliases | Clean face22 / face11 | Noisy face22 / face11 |
| --- | --- | --- |
| Definite distance inliers | 25 /25 | 25 /25 |
| Outside / threshold-unresolved rows | 0 /0 on each face | 0 /0 on each face |
| Closest-point uncertainty exclusions | 0 /0 | 5 /4 |
| Remaining support | 25/25 /25/25 | 20/25 /21/25 |
| Retained lower coverage | 1 /1 | .7999999999999999 /.84 |
| Region-uncertain and boundary rows | 19 on each face | 19 on each face |

Noisy face22's uncertain local rows are [0,6,12,18,24], raw IDs[25,31,37,43,49].
Face11's are [6,12,18,24], also their raw IDs. Only
`closest_point_ambiguity_uncertain` contributes to ambiguity exclusion: definite
closest-point, normal, normal-uncertainty and face-identity ambiguity flags are
false. Every listed row has two near-tied triangles. Exact rational inspection
of the retained nominal row coordinates places each on its two CAD triangles'
shared edge, including endpoints; no projection query was rerun. Noisy maximum
distance upper bounds are .0019999999999988946mm and .0019999999999997823mm,
comfortably below .5mm. Both lower coverage values are below unchanged .95.

The source chain confirms the local cause: global-registration V3
`_unambiguous` (lines250–255) excludes closest-point uncertainty; V11
lines542–568 intersects that mask with definite-distance support and applies
the per-face coverage gate. Region uncertainty alone is not in that exclusion
mask. Thus a correct proposal and complete finite-distance support exist in this
toy, but correspondence uncertainty prevents a passing start. This does not
prove unsound numerical bounds or establish a safe policy change.

## Resources, limits and advancement boundary

| Observation | Value |
| --- | ---: |
| Parent monitored wall seconds | .7562372917309403 |
| Worker CPU seconds | .7104599999999999 |
| Worker high-water RSS bytes | 96,174,080 |
| Sampled group peak RSS bytes | 96,174,080 |
| Captured output bytes | 861 |
| Child exit code / budget stop | 0 / null |

Both receipts retain60wall/45workerCPU/1,073,741,824B sampled RSS/1,048,576B
output and two16,777,216B artifact slots/33,554,432B total. Every reported measure
is below its cap. Per-case channel maxima128 calls/3,200 rows are respected.
Sampled RSS is not a hard allocation bound; the protocol's disclosed source
hashing, receipt IO, cleanup timing and inherited reaping limitations remain.

This audit authenticates a complete diagnostic with a reproduced negative. It
does not supersede the original40PASS/1FAIL qualification or release workflow,
native, Gold11 or another diagnostic run. It establishes neither the physical
V3 cause, retessellation invariance, global pose observability, safe ambiguity
bypass, measured bends, free-circle success, improved F1 nor floor readiness.
The retained broad50-row fixture is not the original thin native part.

Review used only the exact owned attempt and released source/protocol evidence.
No original physical/raw/CAD/Gold/catalog/prediction input was opened, and no
target module, test, geometry query, inference or scorer was executed. Array
byte reconstruction, retained-statistic tallies and exact nominal-edge identity
checks are post-audit verification only. Only this additive POST was written.
