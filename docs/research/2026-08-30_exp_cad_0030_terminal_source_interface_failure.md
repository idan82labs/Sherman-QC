# EXP_CAD_0030 terminal source-interface failure

## Ruling

EXP_CAD_0030 / CF-PDG V3 is consumed and terminal. The authorized primary source child failed with `ValueError: zero scan normal` while canonicalizing the second opaque member. No patch, rescue, replacement input, replay, or rerun is permitted under this experiment identity.

This is a source-interface failure, not an accuracy result. No pairwise deformation component, member attribution, bend-line prediction, F1 score, generalization claim, or production decision was produced. Its Generalization Progress Indicator is `GPI-C_NEGATIVE_SOURCE_INTERFACE_FAILURE_NOT_F1_PROGRESS`.

## Immutable evidence

- Orchestration result: `output/research/autoresearch-loop/EXP_CAD_0030_ORCHESTRATION_RESULT.json`
- Orchestration SHA-256: `c77fa40df26929675be34d3cdbc958fad18e7dfa7f8ea785639770942eedd39c`
- Terminal accounting result: `output/research/autoresearch-loop/EXP_CAD_0030_RESULT.json`
- Terminal accounting SHA-256: `e0f477e22b8562b178d0185ec25941dbe4c7532271ad792eca9a7174f26ad03f`
- Preregistration SHA-256: `c9129ccc103d45c89ccdc67eef938b242ccdbb081d3d0f261145cc5a5b9b8368`
- Independent CAD-only approval SHA-256: `5721370b6bcb585c5dc9309aa561bd09f020b7f16eac8affefa4452c5e113d31`
- Independent source-runner approval SHA-256: `a412fa498585ca14352db3656b49d73f687197d672ec1a8e53e796d05512aae8`

The terminal orchestrator atomically persisted `FAIL_TERMINAL_NO_RETRY` at stage `PRIMARY_SOURCE`. The primary and replay output roots are absent.

## Exact execution accounting

Only the primary source child ran:

- source children: primary 1, replay 0;
- renderer children: primary 0, replay 0;
- physical-scan SHA-256 calls: 2;
- physical point-cloud loads: 2;
- physical-scan byte-open operations: 4.

Both scan hashes and both point-cloud loads completed before `execute_source` rejected a zero-length normal in the second opaque member. Therefore this accounting cannot be reported as zero merely because `SOURCE_RESULT.json` was never reached.

## Visual status

`SOURCE_VISUAL_PAYLOAD.json` was not produced, the native renderer was not invoked, and native-resolution visual adjudication is `NOT_REACHED`. No visual-pass or visual-fail claim is made.

## Sealed authority state

Evaluator-manifest contents, evaluator roles, labels, gold, expected counts, and quality identity remained sealed. This failure provides no F1 or direction evidence.

## Successor boundary

A successor requires a fresh experiment identity and fresh approvals. Its only permitted change is a preregistered, deterministic source-intake filter applied before transform and partitioning. It must remove rows with nonfinite points, nonfinite normals, or normal norm at most `1e-12`, report original-row identities and per-reason counts, enforce pre-frozen invalid-fraction and minimum-valid-count gates, and perform no interpolation or replacement. The CF-PDG scientific semantics remain unchanged.
