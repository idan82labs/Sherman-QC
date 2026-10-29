# Gold11 V9 — independent scoring-ledger POST seam

Date: 2026-09-13 13:43 UTC. Reviewer: Codex subagent `v9_post_score_ledger`,
non-author of the V9 candidate. This is a bounded arithmetic, disposition and
retained-byte binding review, **not the complete physical custody POST**.

## Decision

`ACCEPTED_NEGATIVE_SCORING_LEDGER_WITH_LIMITED_SCOPE`.

No arithmetic, case-omission, projection-hash or original/enriched-report
mismatch was found in the checks below. This accepts the negative ledger as
reported; it does not accept engine performance, integration promotion, a new
numeric F1, historical comparability, or factory-floor use.

Both arms have **TP 0 / FP 0 / FN 71, recall 0, precision null, F1 null**.
All 22 selected case/arm slots remain in the denominator. There is no new
numeric F1 and no positive measured localization progress demonstrated by V9.

## Scope and execution boundary

The root agent confirmed V9 waiter/runner/coordinator/worker quiescence at
13:36 UTC and the actual waiting-parent exit record with return code zero,
finished at 13:34:04 UTC, before assigning this review. Those process and outer
exit observations are supplied by the root, not independently repeated here.

Read `AGENTS.md`, the newest checkpoint in `docs/CURRENT_STATUS.md`, repository
map, Astra workflow and the complete V9 protocol (including its frozen copy).
The status document still described the launch; the root's newer terminal
observation governed this POST-only access.

Inspected only V9 derived score reports, authorized root `COMPLETED.json`
inventory/terminal metadata, `FINAL_GATE.json` bindings, the 22 exact
inventory-named case receipts and projections, eight exact retained result
files for streaming byte hashes, and narrow frozen scoring source. No scorer,
projection constructor, inference or geometry was executed. No raw input,
reference content, earlier attempt, opaque cache, email part or held-out was
opened, hashed or traversed. All attempt and source files remain unchanged.

The independent standard-library check used JSON parsing only for small
reports/receipts/projections and 1 MiB streaming chunks for the eight large
result files. Limits declared before that helper: 50 CPU seconds / 60 wall
seconds, within the assigned 60 CPU seconds / 1 GiB review budget. Observed:
0.157106 CPU seconds, 0.135723 wall seconds, 24,920,064 bytes Darwin peak RSS,
exit zero. No repository modules were imported. The eight full result JSON
structures were **not** decoded or reprojected in this seam.

## Exact report and manifest bindings

Paths in this section are relative to
`docs/research/gold11_mesh_development_v9_attempt/`.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `SCORER_RESULT.json` | 29,758 | `f561f65c620e1d526129a7e3ab549ebcfba391f90a4f33e83c4b38218bbab63d` |
| `SCORED.json` | 106,125 | `f89a9d7b29b0a18545ea5da10c9de23873dd37ada4af275131afd2841c43695c` |
| `COMPLETED.json` | 199,674 | `792d35ad231db751c42f86a2326c8ff77d8ee8be74d4b8e1575aafd73fdbfe29` |
| `FINAL_GATE.json` | 1,057 | `1226ec0dda6d63e322c6146d21affaf3d64f7185fb9d975ca57bb025ccc0e1eb` |

All **37 original top-level scorer fields**, recursively including every
case, family, metric and claim boundary, were compared by canonical JSON bytes
with the corresponding enriched-report fields: zero differences. Both report
file hashes and sizes match the root manifest; the final gate binds those same
report hashes, and the enriched report binds the original scorer file hash.

The gate's canonical completed-document hash was recomputed as
`5e544c5dcf0e092c4cce6164ca385add0ac99a8876616593c9b4a8cd5ab6ae97`.
Its `owned_artifact_custody_sha256` was recomputed as
`06ceca84b2c03800ff2b77e2997f124c91473089e3b690edbc3ba7acc0a20d15`.

The selected-reference snapshot declaration is consistently
`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`
in the frozen protocol, enriched report, final gate and completed manifest.
The manifest declares 24,751 bytes for that file. Its **content was not read or
hashed here**, so this is declaration reconciliation, not an independent
reference-content or cohort-custody attestation.

The review's observed 56-file hash/size map (four root files, 44 case
receipt/projection files, eight result files), canonicalized with sorted keys
and compact separators, has SHA256
`29eba96e176c707aad45107fc7e0c18f9eee4c926a067ca03e162e3fb1a30233`.
Every case path came from the exact completed inventory, not a directory scan.
That subset digest is not the runner's full-tree inventory digest.

## Complete 22-slot ledger

All rows below have zero predictions, TP zero, FP zero, FN equal to the listed
reference count, precision/F1 null, recall zero, no matches, no unmatched
predictions and a unique unmatched-reference ID list of exactly FN entries.
Both exact emitted-support-count and exact localized-target-set flags are
false in every case. Each arm has 11 distinct cases and six families; the case
sets equal the 22 unique terminal `(case_id, arm)` slots. All 22 terminal
records are marked durable, with hashes checked against their actual bytes.

`CPU` below means exit `-24` (SIGXCPU on this host), exact retained supervisor
reason `INFERENCE_WORKER_FAILED`, no returned result. It does not identify the
last internal geometry stage; that stage stays unknown in these receipts.

| Case | Family | References per arm | Control outcome | Challenger outcome |
| --- | --- | ---: | --- | --- |
| CG08506AE97D | 10839000 | 6 | CPU | CPU |
| CG191DEC7F81 | 49001000 | 5 | CPU | CPU |
| CG1E0BBA2933 | 38172000 | 10 | CPU | CPU |
| CG4BDD1371B6 | 37361005 | 4 | Measurement contract refusal | No supported measurement context |
| CG4FAF273093 | 49024000 | 3 | CPU | CPU |
| CG64C412CC73 | 40500000 | 7 | CPU | CPU |
| CG65CD575B1C | 38172000 | 6 | No finite prefiltered start | No finite prefiltered start |
| CG84C8FA12D2 | 10839000 | 8 | No finite prefiltered start | No finite prefiltered start |
| CGE4B0549E41 | 10839000 | 4 | Noisy-plane initialization unavailable | Noisy-plane initialization unavailable |
| CGEBA9A869AC | 38172000 | 8 | CPU | CPU |
| CGF41F49066A | 38172000 | 10 | CPU | CPU |
| **Total per arm** | **6 families** | **71** | **11 refusals** | **11 refusals** |

The 14 supervisor refusals have cleanly retained terminal receipts, exit
`-24`, `budget_stop: null`, `failure_type: ValueError`, zero retained result
artifacts and a scoring envelope equal to the receipt's supervisor refusal.
`budget_stop: null` must not be interpreted as successful inference: the
nonzero exit is preserved and controls the unavailable projection.

The eight returned workflows have exit zero, no supervisor failure type and
no budget stop, but remain **workflow refusals**, not successful inspections.
Their outer schedule wrappers are `ADMITTED` only as structurally accepted
transport; their inner contact projections are all `REFUSED`, with zero
predictions. The scorer correctly uses the inner refusal.

Exact returned workflow reasons:

- CG4BDD1371B6/control: `MEASUREMENT_REFUSAL`,
  `SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES`.
- CG4BDD1371B6/challenger: `WHOLE_PART_REGISTRATION`,
  `PROVISIONAL_REGISTRATION_UNAVAILABLE` and
  `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`.
- CG65CD575B1C and CG84C8FA12D2, both arms:
  `WHOLE_PART_REGISTRATION`, `PROVISIONAL_REGISTRATION_UNAVAILABLE` and
  `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS`.
- CGE4B0549E41, both arms: `WHOLE_PART_REGISTRATION`,
  `PROVISIONAL_REGISTRATION_UNAVAILABLE` and
  `NOISY_PLANE_INITIALIZATION_UNAVAILABLE`.

## Arithmetic and family reconciliation

Independent arithmetic checked strict nonnegative integer counts and
`TP + FP = prediction_count`, `TP + FN = gold_count` for each slot, then
recomputed all per-case, family and arm metrics without importing the scorer.

| Family | Cases per arm | Reference lines / FN per arm | TP | FP | Recall | Precision / F1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10839000 | 3 | 18 | 0 | 0 | 0 | null / null |
| 37361005 | 1 | 4 | 0 | 0 | 0 | null / null |
| 38172000 | 4 | 34 | 0 | 0 | 0 | null / null |
| 40500000 | 1 | 7 | 0 | 0 | 0 | null / null |
| 49001000 | 1 | 5 | 0 | 0 | 0 | null / null |
| 49024000 | 1 | 3 | 0 | 0 | 0 | null / null |
| **Total** | **11** | **71** | **0** | **0** | **0** | **null / null** |

Per arm: emitted-support lines 0; exact emitted-support-count accuracy 0/11;
exact localized-target-set accuracy 0/11; minimum-family recall 0; refused
cases 11; malformed cases 0; completed-no-localized-carrier cases 0.

The frozen contract computes precision as `TP / (TP + FP)` when the
denominator is nonzero, otherwise null. It computes F1 only if precision and
recall are both available and their sum is nonzero. Thus F1 is deliberately
null for this run, not a computation crash. Replacing it with zero using a
different convention would change the frozen reporting rule; this review
does not do so.

All aggregate and six family paired deltas were independently checked as
challenger minus control, propagating null if either input is null. Count,
recall and exactness deltas are zero; precision and F1 deltas are null, not
zero. The all-22 accounting contains 142 arm-weighted reference opportunities,
but the cohort is still 71 distinct reference lines, never 142 distinct Gold
lines and never a reduced eight-returned-workflow denominator.

## Receipt/projection/result byte checks

For all 22 slots, verified the actual receipt and scoring-input file hashes
and lengths against both root `owned_artifact_custody.files` and the terminal
record's two-file map. Checked receipt arm-request/projection-kind/hash fields
against `SCORED.json.projection_bindings`; checked every receipt's workflow
stage diagnostics against the enriched report's corresponding attribution.

For all eight returned workflows, checked the contact projection's disposition,
phase, diagnostics and result binding against its score disposition; checked
actual RESULT file hash/length against the receipt, inventory, outer wrapper,
inner contact binding and enriched binding. The canonical contact input hash
matches the declared `scorer_input_canonical_sha256` for all 22 slots.

All 14 supervisor projection files are the same 159-byte unavailable envelope,
SHA256 `3a4dd85d5df2c5ccbeac7047f13d17d607889b5e718b725ccd17d19ee1ddeea1`;
its canonical scorer-input hash is
`406dde05977da1b16deef7f923df3917c7c8be79e3cbf7c06d92f48c58d9960a`.
The shared payload does not collapse cases: case/arm identity is supplied by
the separately checked 22 terminal records and score maps.

The eight other `SCORING_INPUT.json` hashes, at `cases/<case>/<arm>/`:

| Case / arm | Bytes | SHA256 |
| --- | ---: | --- |
| CG4BDD1371B6 / control | 6,290 | `7705798f0452bd7d6f60fc2c923a92b7b23d977a753b3baca4e698f9594c3eaa` |
| CG4BDD1371B6 / challenger | 10,932 | `621085f86896f3da2a7a45d4950bb207cd9033a4c01298befe2b0e1572d98da4` |
| CG65CD575B1C / control | 6,303 | `299b2ab4253d5e781f75f92c5544ea20b1037c01aa14e841c97a78c4b8ab3622` |
| CG65CD575B1C / challenger | 10,925 | `d2fe60cdfdb3e364aac104302fca57f5c81100fa043b75cb14cdf9f656e263cb` |
| CG84C8FA12D2 / control | 6,303 | `4b79685746fee85c1468025d1cafcb56a34d1995a1d32525d393b4f73440b9d8` |
| CG84C8FA12D2 / challenger | 10,925 | `753d05cd4d725289c3035dbb91015e1c09127816a5177391c41039377365cc4a` |
| CGE4B0549E41 / control | 6,069 | `48bd275a5737fbd9cea91fd242ffb4dbf923c9d6d4506fa925e43d3744527796` |
| CGE4B0549E41 / challenger | 10,663 | `d56be110d32c2b0331c355ea760fde901b37a4d736e9db937cc4fe0337bee101` |

Actual retained `RESULT.json` byte hashes at the same eight paths:

| Case / arm | Bytes | SHA256 |
| --- | ---: | --- |
| CG4BDD1371B6 / control | 60,624,780 | `583eb18fd87a9212d20af433b3008e9b565c161ef904de2c91606d9e114a9cb8` |
| CG4BDD1371B6 / challenger | 61,733,466 | `32dd73a4565f2887436caf5a68ca3243b582ce165f4b7754201191431832003a` |
| CG65CD575B1C / control | 57,300,936 | `c192260e3fe6ee33cc28e521801e38fc0d935893cf48dffcf38317473e3add6e` |
| CG65CD575B1C / challenger | 58,946,054 | `a8c0c9e226d842e805532e4fbc0bc85348f92fd517d162f0b73284980bd8d0ae` |
| CG84C8FA12D2 / control | 74,637,411 | `8b026a8086933f9b02c4598fb03dad40b277ebf3c8d10f2d8b861fabdb76da1f` |
| CG84C8FA12D2 / challenger | 75,811,102 | `12e8eca3b912d12bf9cedf16a090578e87699098d75716cb42269743495ba462` |
| CGE4B0549E41 / control | 85,765 | `2aa1d46339521a778bf098801fa66bb3df76325bc21ba15512125718c4e4b980` |
| CGE4B0549E41 / challenger | 90,951 | `0c0d0dd8789f8fae506114f2c00ebe54c8dd07de0045eab570b170127b525a86` |

## Claim boundaries and actionable engineering evidence

1. **No score success.** Zero carriers mean no numeric F1 under the frozen
   contract. Outer exit zero establishes completed bookkeeping, not successful
   scan measurement. The 14 killed leaves and eight workflow refusals must be
   retained as failures of this candidate's complete offline workflow.
2. **Contract failure reaches the physical workflow.** The sole returned
   `MEASUREMENT_REFUSAL` is the explicit `INVALID_SUPPORT_COVER_FEATURES`
   validation failure above. It must not be described as a measured part
   defect, insufficient curvature, or ordinary physical flange rejection.
   The precise producer/consumer mismatch is outside this ledger seam and
   needs source/connected-contract diagnosis before any successor.
3. **Resource failure is dominant in the retained ledger.** Fourteen of 22
   leaves ended with exit -24; their terminal projections correctly retain
   all false negatives. No last internal stage is inferred from the absence
   of returned results. This review neither authorizes higher caps nor
   establishes which algorithm consumed the CPU.
4. **Observed count remains unknown.** Both arms retain 0 known / 11 unknown
   physical observed-count cases, unknown rate 1, strict count target false;
   every case's `observed_bend_count` is null. Zero emitted support lines is
   not zero physical bends. All noncomplete contact diagnostics are null,
   including material-side carrier counts; they are not rewritten as zero
   measured groups or validated inner surfaces.
5. **No metrology or historical claim.** Per-bend measurement accuracy is
   `NOT_EVALUATED_NO_MEASUREMENT_TRUTH`, uncertainty `UNCALIBRATED`, conformance
   `INDETERMINATE`, raw-frame/unit interpretation provisional. Historical
   comparability is `UNESTABLISHED`; historical-beaten, production-use,
   floor-release and strict-physical-target flags are false. Both fresh arms
   share V16, so their paired comparison is not old engine versus new engine.
6. **Nonblocking metadata caution.** The inherited scorer's `comparison_scope`
   contains `NEW_V13...` while the enriched experiment/arm identities correctly
   identify V9/V16. This legacy compatibility label is not a source identity
   proof; report the actual frozen runner/arm bindings. It did not alter
   arithmetic or disposition handling, and no frozen file is changed here.

## Frozen source references checked

Paths relative to the attempt's `source/` directory:

| Source | SHA256 |
| --- | --- |
| `scripts/gold11_contact_scoring_v3.py` | `4f0cbea8cc97056295ab49092465f27230d0707ff53eaa8915f78feea031927b` |
| `scripts/f1_physical_evaluation_contract_v2.py` | `28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7` |
| `scripts/gold11_mesh_scoring_v1.py` | `aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234` |
| `scripts/run_gold11_mesh_development_v9.py` | `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md` | `e6fcd96aeb85bf3d91543f63f2001511ca22ff13a46c2bd1f1a51ca810fef54f` |

Reviewed scorer refusal/count rules and aggregation around lines 680–835;
metric-contract ratios/F1 around lines 21–110; legacy summary and delta
functions; runner report enrichment and parent report-reconciliation source.
This is not a new complete source or mathematics audit.

## Remaining independent POST seams

The complete POST must separately establish raw/cohort/reference-content
custody, runtime/source/control before-and-after consistency, one-shot and
process/resource receipts, actual outer exit/stdout, complete final inventory
phases, and reconstruction of projections from original returned result JSON.
This review verifies those result bytes are the bound retained bytes but does
not execute the projector, decode their full geometry evidence, or verify the
raw inputs behind them. A source declaration or mutually consistent hash
ledger alone is not physical truth.

Preserve V9 as terminal. The authorized next engineering action is diagnosis
of the explicit contract refusal and dominant CPU failures, followed only by
a justified separately versioned candidate, frozen protocol and independent
pre-run audit. No retry, rescore or favorable reinterpretation is permitted.
