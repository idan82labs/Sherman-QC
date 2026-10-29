# V28 frozen prospective replication 1 — terminal precommit stop

Date: 2026-08-14

## Outcome

`V28_FROZEN_PROSPECTIVE_PAIRED_RETURN_REPLICATION_1` stopped before any
fixture, detector, or real-scan execution.

Terminal token:

`V28_PRECOMMIT_PIPELINE_CONTRACT_FAILURE_RETIRE`

The frozen V27 and V28 module hashes match the GPT-5.6 Sol Pro ruling exactly:

- V27: `d67db9e952d8e127cebe4c8117fe3d604830a83a084d73c6aeeb8ec812015527`
- V28: `0f0408bb5040d11b3cd6180a4ab46224d720f2a82ff431b6f6b6eff1b06cc9af`

The terminal failures are:

1. `V28_INPUT_ORDER_DEPENDENT_FITTED_AXIS_BYTES`
2. `POST_V28_ORDINARY_NMS_ABSENT`
3. `POST_V28_EXISTING_LANE_PRECEDENCE_ABSENT`

## Why the stop is mandatory

The frozen selector performs the existing bounded-complex NMS inside
`select_unified`, before V27 and V28. After V28 appends its proposal, the
selector directly calls `build_result`. That function sorts and serializes the
rows but performs no NMS. Consequently, the frozen pipeline cannot enforce the
professor-required rule that an existing selected lane survives a duplicate
collision and the V28 proposal is suppressed.

The V28 rank itself is explicit and includes a stable candidate-ID tiebreak:

1. absolute separation error, ascending;
2. stitched span, descending through negation;
3. candidate ID, ascending.

However, qualifying rows are accumulated in `all_rows` input order. The frozen
V27 fitter then stacks component endpoints in component-index order and anchors
axis sign to `indices[0]`. There is no floating-point or endpoint
canonicalization in the V28 comparison or output path. Therefore the required
canonical fitted-axis bytes are not proven independent of candidate ordering
and endpoint reversal.

The Pro ruling says that either condition is a precommit failure and that no
tie patch or detector modification is permitted. The source-free fixture pack
and the single prospective real scan therefore remain unopened.

## Evidence

- Verbatim ruling:
  `docs/research/2026-08-14_gpt56_sol_pro_post_v28_ruling_verbatim.md`
  (`bd7ba581c7c637f04a13a2b7ea6276cb133a351dbda72e2353faabc16e471795`)
- Static audit wrapper:
  `scripts/audit_f1_v28_prospective_replication_precommit.py`
  (`17509d4678c398feb3e5d6444e21cbde46a3833f4aec3538b5d28378a28a9393`)
- Machine manifest:
  `output/research/f1-new-parts-20260814/v28_frozen_prospective_replication_1/precommit_manifest.json`
  (`1d6d8fa5da8d7da3fe3e873f35943ebce29e87c6c75284ee246940aee99549bb`)
- Focused tests: `4 passed in 0.10s`.
- Manifest size: 4,042 bytes, below the 1 MiB non-image audit limit.

## Visual-accounting statement

No native-resolution fixture render exists for this action because the
professor-mandated precommit failed before fixture construction or execution.
Producing or reviewing a model-output visual after that failure would violate
the terminal order. This is not a missing visual audit: it is evidence that the
visual gate was correctly never entered.

## Scientific disposition

`PAIRED_RETURN_COMPLETION_V28` is retired under its current architecture. No
V28b, threshold relaxation, alternate fixture, replacement scan, second
positive, route expansion, rank modification, or U04-derived exception is
authorized.

V27 remains frozen and research-only. The retained production/runtime path is
unchanged.
