# E4 G1 scorecard: tier regression (DEV)

**SYNTHETIC ENGINEERING EVIDENCE (E2 suite); never physical or unseen-family accuracy -- DEV / NOT QUALIFYING**

Verdict: **INCOMPLETE**; regression verdict (A1's per-change signal): **PASS**; qualifying: False (not: TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL, FREEZE_LEDGER_MISMATCH)
Thresholds E4-G1-2 (2026-10-17) sha256 d475df3a97e2a458
Binding gates failed: none; not evaluable: G1.OCCLUSION_RAISES_UNKNOWN (of which not applicable to this tier: G1.OCCLUSION_RAISES_UNKNOWN)

| Gate | Verdict | Value | Threshold | n | 95% CI (family bootstrap) |
|---|---|---|---|---|---|
| G1.ACC.FLANGE_LENGTH.BIAS | PASS | 0.010366 | 0.05 | 119 | [-0.004729, 0.02493] |
| G1.ACC.FLANGE_LENGTH.P95 | PASS | 0.103367 | 0.2 | 119 | [0.08467, 0.1436] |
| G1.ACC.INCLUDED_ANGLE.BIAS | PASS | 0.001521 | 0.1 | 58 | [-0.002646, -0.0004707] |
| G1.ACC.INCLUDED_ANGLE.P95 | PASS | 0.019033 | 0.3 | 58 | [0.0143, 0.02056] |
| G1.ACC.BEND_LINE_POSITION.P95 | PASS | 0.072959 | 0.3 | 58 | [0.07065, 0.0845] |
| G1.COVERAGE | PASS | 0.996933 | 0.9 | 326 | [0.9896, 1] |
| G1.FALSE_PASS | PASS | 0 | 0 | 553 |  |
| G1.FALSE_FAIL | PASS | 0.0 | 0.005 | 273 |  |
| G1.U_CALIBRATION | PASS | 0.984615 | 0.95 | 325 | [0.9716, 0.9942] |
| G1.MEDIAN_U.mm | PASS | 0.106348 | 0.15 | 244 |  |
| G1.MEDIAN_U.deg | PASS | 0.115944 | 0.25 | 81 |  |
| G1.DECISIVENESS | PASS | 0.983553 | 0.9 | 304 | [0.9745, 0.9938] |
| G1.SLOPE.FLANGE_LENGTH | PASS | 1.007928 | [0.95, 1.05] | 150 | [0.9943, 1.025] |
| G1.SLOPE.INCLUDED_ANGLE | PASS | 1.000035 | [0.95, 1.05] | 81 | [0.9991, 1.001] |
| G1.SLOPE.BEND_LINE_POSITION | PASS | 0.993448 | [0.95, 1.05] | 81 | [0.9818, 1.023] |
| G1.SLOPE.BACKGAUGE_LENGTH (info) | NOT_EVALUABLE | 0.987304 | [0.95, 1.05] | 13 | [0.6105, 1.022] |
| G1.STAGE_STATE | PASS | 1.0 | 0.98 | 67 | [1, 1] |
| G1.SYMMETRIC_NO_WRONG_ID | PASS | 0 | 0 | 2 |  |
| G1.LINE_ON_OWN_BEND (info) | PASS | 1.0 | 1.0 | 81 |  |
| G1.POSE_CHECK (info) | PASS | 1.0 | 1.0 | 48 |  |
| G1.AMBIGUOUS_ON_UNAMBIGUOUS (info) | PASS | 0 | 0 | 47 |  |
| G1.BACKGAUGE_DETECTION | PASS | 1.0 | 1.0 | 2 |  |
| G1.NOT_YET_FORMED_NEVER_FAIL | PASS | 0 | 0 | 176 |  |
| G1.WRONG_OR_MIRRORED_NEVER_PASS | PASS | 0 | 0 | 32 |  |
| G1.SUMMARY_NEVER | PASS | 0 | 0 | 50 |  |
| G1.MUST_BE_UNKNOWN | PASS | 0 | 0 | 13 |  |
| G1.ENGINE_FAILURES | PASS | 0 | 0 | 50 |  |
| G1.OCCLUSION_RAISES_UNKNOWN (n/a in tier) | NOT_EVALUABLE | {"engine_failures": 0, "false_pass": 0, "u_calibration": null} | {"engine_failures": 0, "false_pass": 0, "u_calibration_min": 0.95} | 0 |  |
| G1.RUNTIME.P95 (info) | NOT_EVALUABLE | 13.9641 | 45.0 | 50 |  |
| G1.RUNTIME.RSS (info) | NOT_EVALUABLE | 899907584 | 4294967296 | 50 |  |
| G1.TIER_BUDGET (info) | PASS | 376.8 | 1200.0 | 50 |  |
| G1.SUITE (info) | FAIL | {"cases": 50, "families": 7, "feature_gaps": {}} | {"min_cases": 200, "min_families": 7} | 50 |  |

Counts: {"OK": 50}

