# F1 V69 Actionability Preflight — Attempt 1 Protocol Invalidation

Date: 2026-08-15

## Decision

The first V69 preflight attempt is **invalid evidence** and may not support retention, retirement, tuning, or a generalization claim.

The run consumed 21 candidates from the frozen V52 post-composition stream. The professor-authorized protocol required the 48-row pre-NMS thresholded V16/V9 stream: 13 candidates for U01, 20 for U02, 3 for U03, and 12 for U04. The first attempt therefore tested the wrong candidate population.

No fresh scan, gold labels, CAD, expected counts, identity information, runtime/main, or production path was opened. The invalid output is preserved rather than overwritten.

## Invalid attempt result

- candidates: 21
- vetoed: 20
- retained: 1
- mechanical gate: failed
- output: `output/research/f1-v69-camntp-u01-u04-actionability-v1-20260815`
- summary SHA256: `4cc7c558ac24f2bdedd23b5b88c82b7bfdf902e3cd92fd599968504839dc431b`
- lock SHA256: `223dc1c55cf1bedb420c7f4d05aea83dc7538d48cb2da32f2b0fbcefbeca6e28`

## Correction

The corrected single primary-plus-replay attempt used the exact 48-row pre-NMS thresholded stream and was written to a separate output root. Only that corrected result is admissible.
