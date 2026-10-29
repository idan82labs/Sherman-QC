# EXP_CAD_0031 terminal primary-source timeout

## Ruling

EXP_CAD_0031 / CF-PDG V4 is consumed and terminal. Its single authorized primary source child exceeded the frozen 300-second wall limit and was killed by the bound orchestrator. No timeout extension, patch, rescue, replay, or rerun is permitted under this experiment identity.

This is a runtime failure, not an accuracy result. No pairwise deformation component, opaque-member attribution, bend-line prediction, F1 score, generalization claim, or production decision was produced. Its Generalization Progress Indicator is `GPI-C_NEGATIVE_PHYSICAL_SOURCE_RUNTIME_INFEASIBLE_NOT_F1_PROGRESS`.

## Immutable evidence

- Orchestration result: `output/research/autoresearch-loop/EXP_CAD_0031_ORCHESTRATION_RESULT.json`
- Orchestration SHA-256: `bf115d29440f6317619ec57fb0fadf3f304fd2f9425fb0725580785df4b25c58`
- Durable journal: `output/research/autoresearch-loop/exp-cad-0031-cfpdg-v4-physical-primary/SOURCE_SCAN_ACCOUNTING_JOURNAL.json`
- Durable journal SHA-256: `032b513b111193c6b1f485355cc3cd7c1ec43d14edfbbc5a287d55135a822d02`
- Preregistration SHA-256: `20f4a370eac176dc4a91df19e68c2e2618dc7f383d0d45eb526267292b718eb6`
- CAD-reuse approval SHA-256: `ed05b399ccb1cea6759eed17ec0cff0c79480c03085cfab32f51c97a20449219`
- Source-runner approval SHA-256: `6beac90de70681d066d3ecf7d21f155b4f36b209d391fae56046ab2e4ea3f7d5`

The terminal orchestrator atomically persisted `FAIL_TERMINAL_NO_RETRY` at `PRIMARY_SOURCE`. The replay root is absent and neither renderer was launched.

## Exact execution accounting

Only the primary source child ran:

- source children: primary 1, replay 0;
- renderer children: primary 0, replay 0;
- physical-scan SHA-256 calls: 2;
- physical point-cloud loads: 2;
- physical-scan byte-open operations: 4.

The journal's final durable stage is `SOURCE_INTERFACE_AND_PARENT_SCIENCE`, with the full `2/2/4` accounting already checkpointed. Because the child was killed before publishing `SOURCE_RESULT.json`, this evidence does **not** separate time spent in intake from time spent in the unchanged parent science and does not prove the exact filter counts or interface-actionability decision. It only proves that the combined stage was entered and did not finish within 300 seconds.

## Visual status

`SOURCE_VISUAL_PAYLOAD.json` was not produced, the parent native renderer was not invoked, and native-resolution visual adjudication is `NOT_REACHED`. No visual-pass or visual-fail claim is made.

## Sealed authority state

Evaluator-manifest contents, evaluator roles, labels, gold, expected counts, and quality identity remained sealed. This failure provides no F1, quality, or directional evidence.

## Successor boundary

Any successor requires a fresh identity and approvals. It may either prove a semantics-equivalent, bounded runtime implementation of CF-PDG without changing its scientific thresholds or decision graph, or retire CF-PDG and preregister a materially different source-only architecture. EXP31 may not be rescued by increasing its timeout, reducing gates after observing this run, or replaying the same implementation.
