# Gold11 A0 provenance trace V2 — 2026-09-10

## Outcome

**The exact eleven approved scan-SHA-to-pool-token associations are now traced.**
The A0 prediction root, its eleven prediction payloads and the complete historical
60-line union remain **untraced, not established missing**. No A0 prediction was
opened or hashed in this task, because the root-to-payload binding was not found
within the fixed twenty-record metadata/source budget.

This is a partial provenance recovery, not a historical replay, score, fresh
inference result, or production qualification. The existing 74-test retained
prediction scorer file was left unchanged. Only this additive note was authored.

## Binding chain recovered

The [EXP_CAD_0008 published result](2026-08-27_exp_cad_0008_visibility_event_fusion_result.md)
names `output/research/autoresearch-loop/EXP_CAD_0008_PREREGISTRATION.json`.
That retained preregistration's `frozen_inputs` explicitly declares:

- `scan_candidate_manifest` =
  `output/research/f1-gold27-20260803/candidate_pool_v1/manifest.json`;
- `scan_candidate_manifest_sha256` =
  `51f70c11ed3a726277c94594bf1da53d4d54723bc8e4445af5bb39b9d60b7fd4`.

The current manifest byte hash **equals that recorded digest**. Its `cases`
array was filtered by the eleven literal approved SHA constants from the
[earlier recovery note](2026-09-10_gold11_retained_control_recovery.md) **before**
row metadata was displayed or analyzed. Exactly eleven rows matched, with eleven
distinct approved hashes and eleven distinct tokens. No other case rows, raw
geometry, candidate-pool payloads or source-path targets were inspected.

The following table records those exact metadata associations. The two count
columns are copied historical expectations from the earlier recovery note,
**not measurements of recovered payloads** and not criteria for repairing data.

| Approved case | Approved scan SHA-256 | Pool token | Reported A0 lines | Reported union lines |
|---|---|---|---:|---:|
| CG08506AE97D | `cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b` | C003 | 1 | 4 |
| CG191DEC7F81 | `7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020` | C021 | 3 | 3 |
| CG1E0BBA2933 | `4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9` | C011 | 5 | 11 |
| CG4BDD1371B6 | `d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038` | C008 | 2 | 3 |
| CG4FAF273093 | `70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade` | C022 | 1 | 3 |
| CG64C412CC73 | `152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614` | C015 | 5 | 6 |
| CG65CD575B1C | `07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880` | C009 | 1 | 2 |
| CG84C8FA12D2 | `71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215` | C001 | 5 | 11 |
| CGE4B0549E41 | `08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc` | C002 | 1 | 2 |
| CGEBA9A869AC | `f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63` | C010 | 5 | 6 |
| CGF41F49066A | `2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba` | C012 | 4 | 9 |
| **Historical reported total** | | | **33** | **60** |

The selected manifest `path` fields identify
`candidate_pool_v1/Cxxx.json`, **not** `prediction_owned_regions.json`.
They therefore do not establish an A0 prediction root. There is no prediction
content hash in those selected mapping rows.

## What the source proves — and does not

[Original fusion source](../../scripts/diagnose_f1_cad_scan_fusion_gold11.py:79)
joins `source_scan_sha256` to `case_token`, then reads
`<caller-supplied a0-root>/<token>/prediction_owned_regions.json` at lines96–98.
It normalizes `owned_bend_regions`, keeps all A0 lines and appends nonduplicate
CAD lines. The output report contains counts/scores, not the full union endpoint
payload or the root/manifest command-line values.

The later [visibility-fusion source](../../scripts/run_cad_visibility_event_fusion_v1.py:39)
uses the same SHA/token join and caller-supplied root convention, at lines47–70.
Crucially, its output payload records `scan_sha256`, `a0_sha256`,
`a0_availability` and the number of scan-first candidates at lines83–93;
its batch summary binds each resulting `semantic_result` path and digest at
lines105–110. Those are useful next provenance witnesses, but **their actual
payload values were not inspected here**. The later algorithm's cropped/fused
predictions must not be substituted for the retained August22 union.

The August25 stage graph and its builder bind source files by hashes computed
when the builder runs; they do not preserve the August22 A0 invocation. The
graph names `run_blind_cad_ply_benchmark.py` for both trusted registration and
A0. This label is not evidence that the benchmark produced the required
`owned_bend_regions` payloads. No runtime mechanism identity was inferred from it.

## Current bytes versus historical attestation

| Record | Current SHA-256 observed in this task |
|---|---|
| Earlier recovery note | `114e60dc1505c09871ba6d0a747234f8a2e92bdd8c10d80c99318a80764f085e` |
| EXP_CAD_0008 preregistration | `012be266a2dce0aecb84500080ec02d75efe18397be80b0ed7380c8be1d4c596` |
| Named candidate manifest | `51f70c11ed3a726277c94594bf1da53d4d54723bc8e4445af5bb39b9d60b7fd4` |
| Original fusion source | `cc34184166118fb2d00bd4b15edeabf6aaab3e69ea6f7c8e33b9394fffba5b52` |
| Visibility-fusion source | `f0af10663f7913984ee1a75fabed894c41a07b0fe7fb950d75b90556ff52f674` |
| EXP_CAD_0008 published result | `29bf1da49923e5cb769e3fb54e5663fd5df122906c338583d9f01c44c242f2b1` |

The manifest has a retained preregistration digest witness, and the current
fusion-source hash equals the retained August25 stage-graph value. These checks
establish current-byte consistency with those **retained declarations**. They
do not independently timestamp their creation, prove which CLI executed on
August22, establish A0 payload byte identity, bind the full historical import
closure/environment, or establish equality of historical/current reference
coordinates. A0 token recovery alone does not close historical comparability.

## Smallest next record lead

Read the specifically named
`output/research/autoresearch-loop/EXP_CAD_0008_RESULT.json` as metadata only.
It is cited directly by the published EXP_CAD_0008 result but was **not opened**
because this task reached its twenty-record limit. Look for the actual command,
input root, execution receipt or exact batch-summary path. If it supplies only
another reference, follow only that explicitly named record under renewed scope.

The useful next evidence is either:

1. an actual invocation/input record binding the above manifest and A0 root,
   followed by the eleven exact `prediction_owned_regions.json` payloads; or
2. an exact retained full-union payload with case/source-byte provenance.

The visibility-fusion batch/payload hash fields offer a later byte-consistency
cross-check once their exact paths are established. They do not themselves prove
the August22 invocation. Do not guess a root from the pool directory, count-match
unrelated files, replace A0 with later scan-only versions, rerun inference, or
repair a prediction set to force the reported33/60 totals. Preserve any mismatch.

## Bounded read ledger

Twenty substantive supporting records were read, fully or by targeted metadata/
source projection. Canonical workspace instructions/status were also refreshed.
Filename-only source/doc discovery was used to identify these records; no blind
historical pool-content search was performed.

1. `docs/research/2026-09-10_gold11_retained_control_recovery.md`
2. `docs/research/2026-08-22_f1_cad_guided_zero_shot_e2e_result.md`
3. `docs/research/2026-08-27_cad_assisted_f1_deep_dive.md`
4. `scripts/diagnose_f1_cad_scan_fusion_gold11.py`
5. `docs/research/artifacts/2026-08-25_f1_retained_stage_graph_v1.json`
6. `scripts/build_f1_retained_stage_graph_v1.py`
7. `scripts/run_f1_local_plane_fragment_probe_batch.py`
8. `scripts/run_blind_cad_ply_benchmark.py`
9. `docs/research/2026-08-25_f1_post_cibec_dual_lane_preregistration_v1.md`
10. `docs/research/artifacts/2026-08-27_f1_program_status_v2.json`
11. `docs/research/2026-08-22_f1_step_topology_targeted_e2e_result.md`
12. `scripts/score_f1_cad_guided_gold13_e2e.py`
13. `scripts/diagnose_f1_cad_assisted_multisource_gold11.py`
14. `scripts/run_cad_visibility_event_fusion_v1.py`
15. `scripts/run_cad_contextual_assignment_v1.py`
16. `scripts/run_cad_ontology_grouped_capacity_fusion_v1.py`
17. `docs/research/2026-08-27_exp_cad_0008_visibility_event_fusion_result.md`
18. `docs/research/2026-08-27_exp_cad_0009_contextual_assignment_result.md`
19. `output/research/autoresearch-loop/EXP_CAD_0008_PREREGISTRATION.json`
20. `output/research/f1-gold27-20260803/candidate_pool_v1/manifest.json`

No new raw scan/CAD, Gold coordinates, email/heldout, A0 prediction, candidate
geometry, runtime import, inference, scoring or test execution occurred. The
larger manifest read was metadata-only and its row analysis was restricted to
the exact eleven approved scan hashes. The historical33/44/60 line totals and
F1 values remain separately labeled reported evidence, not recomputed outcomes.
