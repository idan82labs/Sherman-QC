# Gold11 mesh-workflow V2 — terminal negative, no score improvement

Executed once on 2026-09-09 under the standing consumed-development mandate and
exact independent pre-run audit. All eleven selected scans, six CADs and the
unchanged 71 reference lines are retained. `COMPLETE_DEVELOPMENT_ONLY` describes
execution completion, not engine qualification or roadmap completion.

Frozen bundle: `077d782626fe3308da6221b5b8a0180af62898703da700ddcb72224b2ad42005`.
Attempt: `docs/research/gold11_mesh_development_v2_attempt/`.
[Frozen protocol](2026-09-09_gold11_mesh_development_v2_protocol.md).
Control is retained V1 workflow/native V1; challenger is V8 all-generated noisy
plane initialization, finite-triangle registration/native V1. Neither is an
attested historical-incumbent replay. All frozen source, protocol and attempt
bytes stay unchanged; do not execute this terminal attempt again.

## Actual paired result

| Metric | V1 control | V8 challenger |
| --- | ---: | ---: |
| Selected cases / terminal outcomes | 11 / 11 | 11 / 11 |
| Emitted support lines | 0 | 0 |
| TP / FP / FN | 0 / 0 / 71 | 0 / 0 / 71 |
| Precision / localized F1, as frozen scorer defines them | undefined / undefined | undefined / undefined |
| Recall / minimum-family recall | 0 / 0 | 0 / 0 |
| Exact emitted-support count equality | 0/11 | 0/11 |
| Known physical observed-bend counts | 0/11 | 0/11 |
| Explicit refusals / malformed cases | 11 / 0 | 11 / 0 |

This is a reproducible negative development baseline, **not a new defined F1
baseline or evidence of improvement**. Undefined metrics are not silently
converted to zero or compared as a numeric delta. The physical observed count
remains unknown, not zero; emitted-line count is a separate diagnostic.
Historical F1 `.4580152672` comparability remains UNESTABLISHED and no claim that
it was beaten is allowed. All locked physical-engine targets remain unmet.

The pre-run synthetic gate improved to [4/4](2026-09-09_mesh_connected_planes_v8_result.md),
but that did not transfer to physical output. Synthetic and physical metrics
remain separate. Per-bend physical measurement truth is absent; uncertainty is
uncalibrated. Uniform 0.05 mm proposal error, common millimetres and whole-part
best-fit are exploratory assumptions, not calibration or stable-datum metrology.
Gold frame/carrier compatibility remains unverified. No product promotion,
manufacturing conformance, unseen-family generalization or floor release follows.

## Independent retained-result failure diagnosis

The independent integration reviewer inspected all eleven challenger RESULTs
serially, their terminal receipts and relevant frozen source, without replay,
raw-data/Gold reads or edits. All result hashes and emitted scan/CAD/topology
bindings match their frozen declarations. Every challenger stops before its first
finite query, registration candidate evaluation or local measurement.

| Case | Immediate stop | Failed CAD plane / triangle count | Signed hypotheses / proposed |
| --- | --- | ---: | ---: |
| CG08506AE97D | Triangle arithmetic admission | 1 / 197 | 6,720 / 3,356 |
| CG191DEC7F81 | Triangle arithmetic admission | 11 / 345 | 4,864 / 1,192 |
| CG1E0BBA2933 | Triangle arithmetic admission | 10 / 199 | 5,600 / 2,752 |
| CG4BDD1371B6 | Triangle arithmetic admission | 8 / 264 | 5,312 / 2,656 |
| CG4FAF273093 | All 158 generated scan triplets ill-conditioned | — | 0 / 0 |
| CG64C412CC73 | Triangle arithmetic admission | 100 / 1,122 | 6,912 / 3,450 |
| CG65CD575B1C | Triangle arithmetic admission | 10 / 199 | 7,904 / 1,845 |
| CG84C8FA12D2 | Triangle arithmetic admission | 1 / 197 | 6,624 / 3,298 |
| CGE4B0549E41 | Triangle-membership work budget | — | 0 / 0 |
| CGEBA9A869AC | Triangle arithmetic admission | 10 / 199 | 5,760 / 2,848 |
| CGF41F49066A | Triangle arithmetic admission | 10 / 199 | 7,520 / 3,744 |

The nine index failures are `ASSIGNED_FACE_SURFACE_UNAVAILABLE` wrapping
`UNSUPPORTED_TRIANGLE_ARITHMETIC_CONDITIONING`. Their complete signed-hypothesis
ledgers are retained; every PROPOSED sign is scheduled, zero outer omissions,
but actual finite-query/evaluated-row counts are zero. Usable groups are cleared;
the number of omitted passing groups stays unknown, not a fabricated zero.
This is not evidence that geometric matching rejected the scan support.

CGE4B0549E41 exhausts the 20-million triangle-membership cap in closure 16;
21,392 cores, 15 completed closures and 9,450 proposals are preserved as discarded
counts, with all output invalidated. The earlier 900k synthetic capacity success
therefore does not establish this physical workload's capacity. CG4FAF273093's
triplet condition ratios are 0.00333–0.00530 against the unchanged 0.02 gate;
no nominal CAD join or pose is generated. Its empty search is not observed absence.
CG191DEC7F81, CG4FAF273093 and CG64C412CC73 also retain `INVALID_NATIVE_SOLID`.

The control retains nine unsupported-registration refusals, one pose ambiguity,
and one accepted registration followed by native-feature refusal. Its legacy
phase is stale in that last case; V2's descriptive nested-status projection flags
that fact without modifying the retained V1 result or scoring.

## Execution and custody evidence

All 22 case receipts are TERMINAL, exit 0, source-binding valid and durable; no
outer resource stops. Parent wall 367.95208854135126 s. Coordinator wall
367.9496540832333 s. Sampled complete-run peak RSS 1,919,434,752 B includes outer
supervisor 33,095,680 B and coordinator tree 1,886,339,072 B. Log 3,765 B.
The internal triangle-membership refusal above is a mechanism work-bound stop,
not an outer CPU/RSS/wall supervisor stop. Source before/after consistency is true.

Exact retained hashes:

- SCORED.json: `0a869f1dc98ca60557bf2ff02eeb0cfe1a2ce665a9ed3a3a41d466a550efd7af`.
- COMPLETED.json: `bbb57c7ac860ab33b9793553002d419956f49c098bbca798c8d6addea277da13`.
- INFERENCE_TERMINAL.json: `f00821a6a4a69eaef103ce028edb7d4c33191676ff9309dfdcab389b21f5c01c`.
- COORDINATOR_COMPLETED.json: `a846f6748eef7ed63c02c895ba8a6d4de4f8d2ca9c485c72bfdda955392414ae`.
- PRE_RUN_AUDIT.json: `f253983601c4004b84f3194e653e36be34adf925412688321fa6ae14b46674c6`.
- POST_RUN_AUDIT.json: `95ecc808c77880e66b38f39d484bbe4c16d641d855065921393cd4336c02cdc5`.

The separate complete [post-run custody audit](gold11_mesh_development_v2_attempt/POST_RUN_AUDIT.json)
is GREEN only as `VALID_TERMINAL_DEVELOPMENT_DIAGNOSTIC_NO_GAIN_NO_PROMOTION`.
It independently verified all 29 live/frozen source hashes, 22 distinct dispatches,
all RESULT-to-terminal bindings, all case/family/reference denominators and the
terminal-before-scoring barrier. Reconstructed scoring-input map SHA-256 is
`9747f98750b7b6d3e65cb77dac441331f4f49a86caefe8e42eeb42ad079e0387`.
Largest RESULT is 60,022,395 B, below both-arm 128 MiB transport cap. Scorer
wall 0.23703912505879998 s, tree RSS 77,430,784 B, exit 0/no stop. Largest case
parent wall 39.0794412917 s and worker wall 38.4328647498 s; largest sampled
worker tree RSS 1,701,412,864 B. No limits changed during execution.
These are AI technical reviews, not manufacturing signatures, continuous
filesystem surveillance, binary attestation or geometric correctness proof.

## Next justified mechanism, not a retry

The dominant source seam is `backend/cad_guided_triangle_surface_v2.py`:
its admission guard combines soup-wide absolute-coordinate/maximum-edge extrema
with minimum triangle quality, then tests against the soup's minimum edge.
Query admission repeats that coupling. The retained reports do not identify the
limiting primitive/extrema, so successful recovery is not yet established.

Develop a separately versioned locally anchored/scaled finite-triangle mechanism
with per-primitive conditioning/error accounting. Preserve every triangle, face
identity, hole, boundary and tie; do not drop slivers or merely widen a guard.
Independently review arithmetic, conservative BVH pruning and downstream guard
semantics; use a separately constructed high-precision oracle for mixed scales,
translations, slivers, boundary distances and ties. Then connect and test the
fixed native workflow before freezing/auditing a new all-eleven-case physical
protocol. Keep the old numerical refusals, limits and all failed bytes unchanged.
Do not require universal CAD support before the next justified comparison.

The autonomous roadmap goal remains ACTIVE. No new user approval is needed for
ordinary scoped successors; a new physical run still needs its own frozen
protocol and independent pre-run audit.
