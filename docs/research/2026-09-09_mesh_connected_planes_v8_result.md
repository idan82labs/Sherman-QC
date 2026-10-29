# Connected V8 result — four fixed constructed cases pass

Date: 2026-09-09. Constructed engine-output progress, not physical Gold F1.

## Before and after

The retained V6 connected result was1/4, V7 was2/4, and the new independently
pre-audited V8 result is4/4 on the unchanged four-case matrix. V6 fair inner
search had restored essentially exact clean proposals in V7, but the outer32
quota omitted them. V8 removes that omission: all generated PROPOSED signs
receive finite screening under the same total query/work/48-refinement limits,
with a hard1024 eligible-ID bound rather than a selection quota. It does not
change geometry, noise, measurement/global gates, evaluator rules or prior results.

Both clean cases and the bounded-noise case now emit one correctly measured
finite instance. Missing arc remains an honest geometric registration abstention,
not proof of observed absence. The r0-named case is the original extractor control
inside the new wrapper, NOT frozen R0 or a historical pipeline replay.

| Fixed case | V7 | V8 | Radius error mm | Sweep error deg | Generator endpoint error mm | Axis endpoint error mm | Pose action error mm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| V1 clean nominal | Fail | Pass,1 line | 2.4868995751603507e-14 | 1.5631940186722204e-13 | 2.384291766412974e-13 | 2.3057084404723514e-13 | 9.803490207218836e-6 |
| V1 missing arc | Pass,abstention | Pass,abstention | unavailable | unavailable | unavailable | unavailable | not a positive oracle |
| r0 extractor/common-wrapper nominal | Fail | Pass,1 line | 2.4868995751603507e-14 | 1.5631940186722204e-13 | 2.384291766412974e-13 | 2.3057084404723514e-13 | 9.803490207218836e-6 |
| V1 bounded3D-noise nominal | Pass,1 line | Pass,1 line | 0.00007647266923171969 | 0.0003265560458203254 | 0.0007610068492877022 | 0.0007563836828831091 | 0.0017310042767456032 |

Exact and noisy criteria remain those frozen in the
[protocol](2026-09-09_mesh_connected_planes_v8_protocol.md). Numerical clean
errors near roundoff are not physical precision or calibrated error estimates.
Noisy radius/sweep errors slightly increase versus V7 while remaining within the
unchanged limits; pose improves. The claim is fixed-matrix4/4, not monotonic
improvement of every quantity or generalization beyond this constructed model.

## Execution and complete ledgers

Terminal attempt:
`cad_engine_v1_synthetic_checks/mesh_connected_planes_v8-nu8mjr21`.
38 tests passed, no skips; one test includes all four actual cases.
Parent30.99921162519604s, workerCPU30.850386s,
sampled group and worker peak450,363,392B. No180s/120s/4GiB/1MiB stop.
All64 frozen source/test/protocol/harness hashes remain unchanged.
OUTPUT SHA-256
`f372b90ed3cf3e3433677f0fcdff118d96ee1425d93c6379dd3d8cb495f3a9f0`.

| Case slot | Total hypotheses | Proposed/screened | Nominal refused | Complete finite calls | Complete finite rows | Passing hypotheses / exact groups / refined |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | 3008 | 1472 | 1536 | 4416 | 673664 | 4 /4 /4 |
| 01 | 2880 | 1440 | 1440 | 4320 | 669216 | 12 /10 /10 |
| 02 | 3008 | 1472 | 1536 | 4416 | 673664 | 4 /4 /4 |
| 03 | 2816 | 1408 | 1408 | 4224 | 350416 | 8 /8 /8 |

Every proposed sign is screened, every three-support obligation completes,
planned rows/calls equal actual, and outer/refinement omissions and numerical
failures are zero in these four cases. Exact groups retain all hypothesis origins.
The upstream bank/triplet search remains incomplete; this is not global pose
uniqueness. Missing arc's ten refined candidates are nonempty real work; even
correct starts lack required CAD coverage, about0.295166 versus0.30. This is not
an empty-search, numerical, resource or report failure relabeled as a sentinel.

## Retained artifacts

All four complete JSON artifacts are retained under the attempt's `ARTIFACTS`.
Total28,681,798B, largest7,366,484B; each below16MiB, total below64MiB.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `case_00.json` | 7366484 | `5cc441f3cddda203652311b22df5b34144dee384629a2769791bc87452bef421` |
| `case_01.json` | 6971645 | `1011fd27ff24faad88fdb7926f9b4508c4d2542bf4db6ff39d36502fd472dd8a` |
| `case_02.json` | 7364380 | `9d622f47dc8cf675bfae33f88bf0c27efc9697731b00fc4083535d8b64122b65` |
| `case_03.json` | 6979289 | `6d63293d8c3c0cffc1afdb43282e5465be18728d32a292f3ffb6a8d85f5cabaa` |

The [pre-run audit](2026-09-09_mesh_connected_planes_v8_pre_run_audit.md)
retains the complete64-file closure, independent release and prerequisite
receipts:53 initializer passes and37 source passes/1declared native skip.
Sources are V8 initializer`cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b`,
wrapper`afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec`,
harness`6d58919f8b5582ec7611751b6ca262c7060a0a099b015dcd644965852f17632d`.

## Independent post-audit and next gate

Independent retained-only reviewer `/root/astra_repo_guidance_review` has
verified all64 source hashes, all artifacts, complete sign/support/query ledgers,
independently summed full-row/query obligations, exact-group origins and final
measurement bindings. Integral/in-range/disjoint supports use2752 original rows.
Decision: POST-AUDIT GREEN FOR THIS CONSTRUCTED4/4 RESULT. All declared uncertainty,
datum, physical-trim, manufacturing-count and production caveats remain intact.
No new geometry, native replay or physical data was used for that post-audit.

Advance this post-GREEN candidate to the separately audited all-case
Gold11 runner freeze: exact same11 scans/6CAD/71references, unsupported cases
retained. A source-only V2 runner draft was retargeted before any test/freeze to
V8/nativeV1 versus V1/nativeV1. It still needs source/mock/runtime qualification
and exact-bundle audit. Do not require universal CAD support before that run.

Observed physical bend count stays unknown, uncertainty uncalibrated, units and
whole-part best-fit provisional, and production use false. The physical draft's
uniform0.05mm proposal assumption is NOT validated by this0.005mm construction.
There is no new physical score, historical comparability, product promotion or
floor release. The end-to-end roadmap goal remains active.
