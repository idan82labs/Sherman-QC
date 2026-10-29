# V11 stage diagnostic: correct pose rejected by ambiguity-excluded support

Date:2026-09-10. Root observed both first-once commands exit0. Source gate is
independently POST-GREEN; recording POST is also GREEN for diagnostic evidence.
This diagnoses a constructed initializer failure, not a repaired engine or new
physical score. Original V11 qualification remains40PASS/1FAIL. Its workflow,
native and Gold11 successors remain blocked/unexecuted under that release.

## Main finding

The unchanged noisy50-row case **does generate the exact correct pose**.
Hypothesis slots24–31 all contain that same pose (eight aliases, not eight
distinct poses), with recorded action error0mm. Both original complete25-row
supports are present and all25rows per face pass the finite-distance upper-bound
test. The correct pose is then rejected because the support consumer excludes
rows carrying `closest_point_ambiguity_uncertain`.

| Correct-pose evidence | Clean | Noisy |
| --- | --- | --- |
| Selected scan bank |[4,1]|[4,1]|
| Generated hypotheses / correct-pose aliases |64 /8|64 /8|
| Correct-pose action error |0mm|0mm|
| Face22 definite distance inliers |25/25|25/25|
| Face22 uncertainty-excluded rows |0|5|
| Face22 accepted support |25/25=1|20/25; lower enclosure .7999999999999999|
| Face11 definite distance inliers |25/25|25/25|
| Face11 uncertainty-excluded rows |0|4|
| Face11 accepted support |25/25=1|21/25=.84|
| Passing hypotheses / distinct start groups |8 /1|0 /0|
| Original positive oracle |PASS|FAIL|

The frozen minimum support is .95, requiring at least24/25rows on each of these
supports. Neither noisy face meets it after uncertainty exclusion. No runtime
threshold, point, candidate, flag or selection changed in the diagnostic.

## Observed row-level mechanism

Following the source's scheduled-hypothesis/support order, finite-query records
16and17 correspond to hypothesis24, faces22and11. Other exact-pose aliases carry
the same rejected support pattern. In those noisy queries:

- Face22: uncertain closest-point rows0,6,12,18,24.
- Face11: uncertain closest-point rows6,12,18,24.
- Every listed row has two near-tied CAD triangles. Their nominal projection
  lies on the two triangles' shared diagonal, including shared corner endpoints.
- Definite closest-point ambiguity, normal ambiguity and normal-ambiguity
  uncertainty are false. Distance-threshold unresolved count is zero.
- Maximum distance upper bounds are .0019999999999988946mm on face22 and
  .0019999999999997823mm on face11, far inside the unchanged .5mm correspondence
  gate. The .05mm stipulated positional bound is a separate parameter; neither
  is scanner calibration or a manufactured tolerance.
- Both clean/noisy queries have19region-uncertain/boundary rows per face.
  That flag alone does not exclude finite support, so it is not the decisive
  change here. The clean queries have zero closest-point uncertainty flags.

The source chain is explicit:

1. [Triangle surface query](../../backend/cad_guided_triangle_surface_v3.py)
   lines558–565 compares materialized optimizer separation plus optimizer-error
   bounds with its geometric guard. A nonzero uncertainty flag does not assert
   that two physically distinct closest points exist.
2. [Registration support consumer](../../backend/cad_guided_global_registration_v3.py)
   lines250–255 excludes closest-point ambiguity uncertainty along with other
   correspondence ambiguities. The distance-only `_support_masks` stays separate.
3. [V11 prefilter](../../backend/cad_guided_plane_initialization_v11.py)
   lines542–568 intersects that mask with definite distance support, then applies
   the .95per-face threshold. It rejects both true-pose supports.

This establishes the rejection mechanism in this construction. It is not proof
that a floating-point bound is unsound: the bound can correctly report inability
to certify coincident optimizers while the downstream policy is too restrictive
for proposal-level proximity. The source's convex-triangle optimizer bound uses
sqrt(U²-L²) plus restoration error; small nonzero residuals can yield optimizer
error bounds much larger than the geometric coincidence guard even when surface
distance is comfortably inside its gate.

## What this rules out, and what it does not

For these two cases only, failure is not missing the correct proposal, a different
selected plane bank, reduced raw support, an AABB rejection of the true pose,
distance-gate violation, noise exceeding .05mm, a resource stop or source drift.
Normal/boundary ambiguity alone is not the observed rejection trigger.

It does NOT establish that the same cause explains physical V3, where most
assignments are spatially excluded and no correct physical pose truth is known.
It does not establish retessellation invariance, a safe general flag bypass,
global rank/pose eligibility, native finite-bend coverage or free-circle
measurement success. The original50-row control is broad and planar, not the
original thin native part. No Gold/reference or physical input was used.

The next justified question is a separately reviewed proposal-support contract
that preserves finite-union proximity without claiming unique correspondence.
It must retain genuine ambiguity for optimization, pose and measurement, and
survive same-union retessellation plus real-hole/fold/separate-sheet falsifiers.
The requested strategic model consultation remains pending. No implementation
or follow-on experiment is authorized by this result alone.

## Reproducible retained execution

Exact PRE [48-file review](2026-09-10_two_plane_v11_stage_diagnostic_pre_run_audit.md)
SHA256`44b12b25bbea74e9267a93068406730d3becaebd0a68dda94579ac6a40e0405d`.
Protocol SHA`f57ff260fbee688d42293f626986ffbfdbd4945fa9989fa64f8361e0a03f633e`;
test SHA`8dbe3e7c2286fdb6807047efb9d56fdbd99cced338f7a954d12fb6ffe6cd7fca`;
harness SHA`b96502ea02e234e9a801e62add142efb361f7cd21d89687af6bafea7a9344fa2`.
The previous5675harness is preserved in its source snapshot.

Executed once in order, without edits between commands:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic_source
# Independent retained source POST-GREEN before the following command:
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic
```

Both attempts are terminal. These are execution records, NOT commands to rerun
in place. Reproduction requires a separately frozen successor for a documented
purpose; do not overwrite, truncate, change caps or hide a retry.

Source attempt `two_plane_v11_stage_diagnostic_source-mvlk7orx`:32PASS and the
one declared recording skip; parent .2002563327550888s, worker CPU .171461s,
worker/sample peak55,312,384B. Three regular files12,199B. Independent
[source POST](2026-09-10_two_plane_v11_stage_diagnostic_source_post_run_audit.md)
SHA`5efc07bcf9ec0555e0ffb25a50149d9df53cd5d3932ef46187411903c3e0b50f`
released one recording execution. No geometry ran in that source gate.

Recording attempt `two_plane_v11_stage_diagnostic-6yfh9z2j`:33PASS/no skips,
meaning32source/mock checks and one complete two-case recording loop. Original
positive outcomes remain1/2, not33positive engine tests. Parent .7562372917309403s,
worker CPU .7104599999999999s, worker/sample peak96,174,080B. No stop or48-file
drift. Both cases have128box calls/3,200rows and32finite queries/800rows, zero
raised calls or capture errors; all eight channel counters reconcile.

| Recording artifact | Bytes | SHA256 |
| --- | ---: | --- |
|ARTIFACTS/case_00.json|938,002|`7909527ca840a320470688339205743bc22601cf47da6b06647226a2ec2c38fc`|
|ARTIFACTS/case_01.json|1,034,275|`e89a5443dd4cbf9a10e1aa9bf725dc74b75074322511bc03e7cbb7c9a19d5dd2`|
|OUTPUT.txt|861|`fe36e7fcc7395141367d96e27c51e960d7cf988f70fe6ba97793109fea0737b4`|

Limits per run stayed60wall/45workerCPU/1GiBsampledRSS/1MiBstdout; recording
artifacts stayed within two16MiBslots. Sampled RSS is not a hard allocation bound;
the protocol's inherited setup/receipt/cleanup timing limitations remain.
The independent [recording POST](2026-09-10_two_plane_v11_stage_diagnostic_post_run_audit.md),
SHA256`ccffba86f4d0097430c455ce8d55fc539cc99bc4498b0b140102ecc120498535`,
verified the exact five-file inventory, all48 source bindings, all2,164 array
digests, complete query accounting and the rejection mechanism across all eight
correct-pose aliases. Its GREEN decision authenticates this diagnostic and its
reproduced negative; it does not qualify the engine or release another run.

Cached Gold11 F1 remains .4580152672. No new physical baseline, improved engine
output, calibrated uncertainty, production integration or floor release.
