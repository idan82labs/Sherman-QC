# Gold11 retained-control recovery — 2026-09-10

Status: **bounded recovery note and manifest proposal; no inference, fusion reconstruction, or scoring executed**.

## Outcome

The historical nonzero path is partially recoverable, not demonstrated lost.
All eleven retained scan-frame-gated CAD prediction files are present and parseable:
**44 predictions**, with two finite three-coordinate endpoints and unique bend IDs
within each case. Current-byte SHA-256 values are recorded below.

The full historical scan/CAD union is **not yet reconstructed**. Its exact A0
scan-SHA-to-token/root mapping and corresponding **33 A0 predictions** remain
untraced in the inspected records; alternatively a retained full **60-prediction
union** would close this payload gap. No whole-pool search or guessed token/path
was used. "Untraced" does not mean the files are absent.

Keep the historical reported fusion F1 **0.4580152671755725** separate from any
new cached-prediction score and from the recent end-to-end zero-output diagnostic.
A new control must not silently relabel the CAD-only component as that fusion.

## Scope and stopping rule

This pass was capped at 16 named substantive source/doc/provenance records beyond
mandatory repository context, and at most 22 exact approved-case prediction files
(11 A0 and 11 CAD), plus targeted source/doc searches for declared CLI roots.
It stopped after 14 substantive records and the eleven CAD files: the known paths
and hashes were established, but no declared A0 root/token binding was located.
Source search hits were only root-discovery leads, not permission to open their
data or follow other cohorts.

The immutable eleven scan SHA constants in the proposal below were the selection
allowlist. The larger Gold13 slice and visual manifest were filtered to those
constants and their exact approved case IDs before case analysis. The selected
visual `geometry` string equals the selected slice `scan_path` in all eleven cases.
This is a metadata association, not an intrinsic prediction-to-raw-byte signature.
No raw scan/CAD geometry was opened, statted, or rehashed; no unrelated case
geometry, held-out, new email, truth mutation, benchmark, or scientific run occurred.
Historical 13-case aggregate execution/resource values are not asserted for Gold11.

## Available evidence and historical counts

The [August 22 narrative](2026-08-22_f1_cad_guided_zero_shot_e2e_result.md)
identifies the primary diagnostic, support audit, summary, and native visual
directory. The [August 27 correction](2026-08-27_cad_assisted_f1_deep_dive.md)
confirms that the valid modality comparison is this same consumed eleven-case,
six-family, 71-reference slice.

These are **retained reported metrics, not rescored in this pass**:

| Arm | Predictions | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|
| Scan-only A0 | 33 | 19 | 14 | 52 | 0.36538461538461536 |
| Scan-frame-gated CAD | 44 | 20 | 24 | 51 | 0.34782608695652173 |
| Retained scan/CAD union | 60 | 30 | 30 | 41 | 0.4580152671755725 |

Source: [exact Gold11 diagnostic](../../output/research/f1-cad-guided-gold13-e2e-20260822/FUSION_GOLD11_DIAGNOSTIC.json).
It retains per-case counts and scores but no union endpoints, A0 input roots,
reference-coordinate hashes, prediction-content hashes, or complete source/runtime
closure.

The [fusion implementation](../../scripts/diagnose_f1_cad_scan_fusion_gold11.py)
requires explicit `--slice`, `--gold-catalog`, `--a0-pool-manifest`, `--a0-root`,
`--cad-prediction-root`, and `--output`. It joins pool
`source_scan_sha256` to `case_token`, loads
`{a0_root}/{token}/prediction_owned_regions.json`, and reads
`owned_bend_regions`. It retains all normalized A0 lines, then adds each CAD line
not duplicate of an existing union member, in input order. Its duplicate rule is
5 degrees, 4 mm, and 60% **shorter-span overlap**. Do not confuse that fusion
overlap rule with the scorer's reference/prediction coverage conditions.

The current [CAD score exporter](../../scripts/score_f1_cad_guided_gold13_e2e.py)
requires a 13-case result and caller-provided A0 inputs. It is not an exact-eleven
reconstruction command to run unchanged.

The primary `summary.json` top-level metadata and
`SCAN_FRAME_GATE_AUDIT.json` contain no A0 root or token mapping. The support
audit declares 201 samples, 3 mm support distance, minimum supported fraction
0.1, and minimum supported span 8 mm. No support gate was rerun.

## Current-byte CAD manifest proposal

These paths are relative to the Git root. Each observed hash is SHA-256 of the
complete retained prediction file bytes on 2026-09-10, not a newly discovered
August 22 hash. The retained [visual manifest](../../output/research/f1-cad-guided-gold13-e2e-20260822/native-visual-audits-scan-frame-gated/manifest.json)
declares these prediction paths and image hashes; it does **not** hash the
prediction files. Prediction payloads contain no embedded scan/reference/source
hash binding.

The `historical_*` counts below come from the retained diagnostic. Only
`prediction_count` was checked against the corresponding CAD payload here.
The scan SHA values are the fixed approved identities, also declared in the
selected slice records; they were not computed from raw inputs in this pass.

```json
{
  "schema": "GOLD11_RETAINED_CAD_CONTROL_MANIFEST_PROPOSAL_V1",
  "status": "PROPOSAL_NOT_EXECUTED_OR_IMMUTABLE_BUNDLE",
  "case_count": 11,
  "cad_prediction_count": 44,
  "records": [
    {
      "case_id": "CG08506AE97D",
      "scan_sha256": "cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG08506AE97D/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "51d9238df430c2090eb5d5787e24dcdc90aa299dbcb1bf279b59d18d332a014e",
      "prediction_count": 4,
      "historical_a0_prediction_count": 1,
      "historical_union_prediction_count": 4
    },
    {
      "case_id": "CG191DEC7F81",
      "scan_sha256": "7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG191DEC7F81/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "ecd34f506c1510d42e0d4bc4f6e540613c9477dfe307b221a36d7fdae2f6631d",
      "prediction_count": 1,
      "historical_a0_prediction_count": 3,
      "historical_union_prediction_count": 3
    },
    {
      "case_id": "CG1E0BBA2933",
      "scan_sha256": "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG1E0BBA2933/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "3d42db809281eeca6b8d1c5ae0f8c97ec2aea8f7f2d55c578ad058778521e6e8",
      "prediction_count": 9,
      "historical_a0_prediction_count": 5,
      "historical_union_prediction_count": 11
    },
    {
      "case_id": "CG4BDD1371B6",
      "scan_sha256": "d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG4BDD1371B6/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "a2c5f4b000f65932c44b238414fd428ca0b2f9f9f31cf5e98e6fa238c48fcc3e",
      "prediction_count": 3,
      "historical_a0_prediction_count": 2,
      "historical_union_prediction_count": 3
    },
    {
      "case_id": "CG4FAF273093",
      "scan_sha256": "70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG4FAF273093/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "2fb45cc5366c9527b8f84fd09ab90beae70978b653c6cea6f8354046ef96be3b",
      "prediction_count": 3,
      "historical_a0_prediction_count": 1,
      "historical_union_prediction_count": 3
    },
    {
      "case_id": "CG64C412CC73",
      "scan_sha256": "152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG64C412CC73/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "7fd5fd125403af4f965f22a61cbe7781290a6e1326763eea47d74ff88d4172cd",
      "prediction_count": 1,
      "historical_a0_prediction_count": 5,
      "historical_union_prediction_count": 6
    },
    {
      "case_id": "CG65CD575B1C",
      "scan_sha256": "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG65CD575B1C/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "dfd6e66b5716a9c2ecb1b12798a8a1b2ae6b93e1fdeba3b7bfbad6502146464d",
      "prediction_count": 2,
      "historical_a0_prediction_count": 1,
      "historical_union_prediction_count": 2
    },
    {
      "case_id": "CG84C8FA12D2",
      "scan_sha256": "71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CG84C8FA12D2/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "685a2b0ddc86af01e4179f450e7bf45a5ed471267db7e04cae01364c2785d966",
      "prediction_count": 11,
      "historical_a0_prediction_count": 5,
      "historical_union_prediction_count": 11
    },
    {
      "case_id": "CGE4B0549E41",
      "scan_sha256": "08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CGE4B0549E41/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "27713f575b55867de7d73cdfbba960a65bc40d25c2e51179831b02daab9066ad",
      "prediction_count": 2,
      "historical_a0_prediction_count": 1,
      "historical_union_prediction_count": 2
    },
    {
      "case_id": "CGEBA9A869AC",
      "scan_sha256": "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CGEBA9A869AC/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "22209579f6ee4a965d0b6b96b6140af9e6e68aade335e9c43bce95de54618231",
      "prediction_count": 1,
      "historical_a0_prediction_count": 5,
      "historical_union_prediction_count": 6
    },
    {
      "case_id": "CGF41F49066A",
      "scan_sha256": "2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba",
      "prediction_path": "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated/CGF41F49066A/semantic_result.json",
      "prediction_sha256_observed_2026_09_10": "8f6286463ca37c7ebc76b124ab7d48c63faa53c411295a66374c205c612b5d34",
      "prediction_count": 7,
      "historical_a0_prediction_count": 4,
      "historical_union_prediction_count": 9
    }
  ]
}
```

## Source attestation: useful retained binding, incomplete historical closure

A newly traced [August 25 retained stage graph](artifacts/2026-08-25_f1_retained_stage_graph_v1.json)
declares hashes for CAD-stage source files. Its
[builder](../../scripts/build_f1_retained_stage_graph_v1.py) hashes current files
when invoked; the graph is not by itself an August 22 execution receipt.

Two current files were hashed in this pass and exactly match that retained graph:

| Source | Current and graph-declared SHA-256 |
|---|---|
| `scripts/diagnose_f1_cad_scan_fusion_gold11.py` | `cc34184166118fb2d00bd4b15edeabf6aaab3e69ea6f7c8e33b9394fffba5b52` |
| `scripts/apply_f1_cad_guided_scan_frame_gate.py` | `f5d91f841e15570d26557ba04793872889234a8b25de09a358d36ec416a643e1` |

The graph additionally declares native extractor and benchmark entrypoint hashes;
those current files were not hashed here. It labels the same benchmark script as
both trusted registration and scan-A0 stages, but supplies no A0 prediction root,
case-token join, CLI, imported dependency closure, environment, or reference bytes.
Therefore it improves source provenance without proving historical end-to-end replay.

The prior [September 9 input review](2026-09-09_gold11_input_review_checkpoint.md)
records current reference-container hash
`4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`
and selected-record hash
`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
These were not recomputed here and do not establish equality with historical Gold
coordinate bytes. The earlier [readiness audit](2026-09-09_rerun_readiness_audit_checkpoint.md)
said a machine-readable binding was **not opened under its source-only scope**.
That is a permission-era observation, not evidence the binding never existed.

## Proposed next control, without changing historical claims

1. Freeze an additive **cached gated-CAD control** from exactly the eleven file
   hashes above. Capture the selected 71-reference bytes and reference hash,
   scorer/config/import closure, runtime metadata, case order, frame declarations,
   and failure/undefined-metric conventions before scoring. Recheck every current
   hash against this proposal; refuse substitutions, missing/duplicate cases,
   endpoint transformations, or mutable-manifest cohort expansion.
2. Score those unchanged cached CAD predictions against the same frozen references
   used for the new challenger comparison. This can establish a newly measured
   component control without raw geometry or inference. Its historical F1 was
   0.3478; do not promise a nonzero or identical new score before measurement.
3. Recover the retained fusion only if a specific provenance record identifies
   the eleven exact A0 SHA/token/path bindings, or a full-union payload. Inspect
   and hash only those approved prediction records. A0 per-case counts must match
   the proposal's historical counts (total 33); a reconstructed union should
   match the retained per-case counts (total 60). Counts alone do not prove
   endpoint identity, so preserve mismatch diagnostics instead of repairing to
   expected counts.
4. Freeze the fusion reconstruction independently of scoring, preserving input
   ordering, IDs, duplicate semantics, and endpoint bytes. A successful newly
   frozen reconstruction/rescore is **cached-prediction control evidence**, not
   proof of the old raw-to-prediction execution or unseen-family generalization.
   If A0 remains untraced, keep the .458 historical report and .3478 historical
   CAD component distinct; do not fabricate a fusion or replace missing A0 with
   the recent empty baseline.
5. Compare any new end-to-end candidate with these controls on identical frozen
   references, while separately reporting registration/measurement abstentions,
   runtime success, visual admissibility, and actual output. The current terminal
   zero-output runs remain unchanged and do not erase the older nonzero evidence.

## Verification and remaining blocker

Completed: exact-eleven metadata joins; eleven JSON parses; payload count,
finite endpoint-shape, and per-case unique-ID checks; eleven prediction-file
hashes; two source-hash comparisons. No model, scorer, application import,
database, native geometry, fusion reconstruction, or test suite was executed.

Substantive records read/hashed in this pass: the two dated narratives; fusion
source; stage-graph builder; CAD score exporter; August 25 dual-lane
preregistration; source-only root-discovery inspection of
`run_f1_local_plane_fragment_probe_batch.py` (not accepted as an A0 mapping);
the selected original slice; primary-summary metadata; support-audit metadata;
retained CAD-stage graph; selected visual-manifest records; exact Gold11
diagnostic; and support-gate source hash. Targeted source/doc CLI searches did not
supply a concrete A0 root, and no data paths from other campaign leads were opened.

**Smallest remaining fusion blocker:** an attested exact-eleven A0 token/root
mapping and its 33 line payloads, or a retained full-union payload. Historical
reference/source/runtime completeness remains a separate attestation question.
This pass stops here; this document is the only additive artifact.
