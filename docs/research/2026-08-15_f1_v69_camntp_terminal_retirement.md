# F1 V69 CAMNTP V1 — Terminal Retirement

Date: 2026-08-15

## Decision

`V69_CANDIDATE_ALIGNED_MULTISCALE_NORMAL_TURN_PERSISTENCE_V1` is **terminally retired**. It may not be tuned, rescued, rerun on replacement inputs, or used in production/runtime/main.

The corrected professor-authorized U01–U04 actionability preflight used all 48 required pre-NMS thresholded real-corpus candidates. V69 vetoed 47 and retained 1. The 97.9% deletion rate exceeds the maximum allowed 50% deletion rate and fails the mechanical actionability gate before visual adjudication or fresh-scan opening.

## Corrected result

| Scan | Candidates | Vetoed | Retained | Replay byte-identical |
|---|---:|---:|---:|---|
| U01 | 13 | 13 | 0 | yes |
| U02 | 20 | 19 | 1 | yes |
| U03 | 3 | 3 | 0 | yes |
| U04 | 12 | 12 | 0 | yes |
| **Total** | **48** | **47** | **1** | **yes** |

- mechanical gate: failed
- next gate: `TERMINAL_RETIRE_V69`
- fresh scan: remained sealed
- output: `output/research/f1-v69-camntp-u01-u04-actionability-v1-20260815-attempt2`
- summary SHA256: `59215cd9fa02a095ed6b6c6645aa0961ac31b62140867a7f3ccd520ae40c1883`
- lock SHA256: `6c03416072a23ea04faac2b83636d110236bbaf6f0be581b654f0cf7706daf3b`

## Interpretation

The implementation was deterministic and operationally bounded, but the representation was not compatible with the real candidate distribution. The fixed multiscale persistence criteria treated nearly every real candidate as invalid. That is not a marginal threshold miss; it is an architecture-level actionability failure.

No native-resolution V69 reason visualization was opened because the frozen protocol explicitly terminates the architecture on mass deletion above 50%. Visual work should now fund the next representation and the corpus-similarity gate, not a post-hoc rescue of V69.
