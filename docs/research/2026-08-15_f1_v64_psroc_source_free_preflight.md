# F1 V64 PSROC source-free preflight

Date: 2026-08-15

Status: **LOCKED SOURCE-FREE IMPLEMENTATION GATE PASSED**

Architecture: `V64_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE_V1`
(`PSROC_V1`)

The implementation follows the preserved GPT-5.6 Sol Pro `EXECUTE_ONE`
ruling plus its formula-only correction. The formerly truncated row is locked
as:

\[
\max_{10 \le j \le 49}|\delta_j| \le \max(6^\circ,0.18T).
\]

No other ruling constant or rule changed.

## Eight preregistered scientific fixtures

Command:

```text
.venv/bin/python -m pytest -q tests/test_f1_paired_smooth_ribbon_ownership_v64.py
```

Result: `8 passed in 2.03s`.

| Fixture | Frozen expected result | Observed |
|---|---|---|
| circular 60-degree smooth ribbon, radius 24h, length 80h, two interior candidates | delete both | PASS |
| variable-curvature 75-degree ribbon with density `1 + 0.5 cos(2 pi q)` and bounded 0.15h displacement | delete both | PASS |
| sharp 60-degree fold plus parallel distractor | delete neither | PASS |
| U-channel with two structural 90-degree bends | delete neither | PASS |
| return/hem with 150-degree total structural turn | delete neither | PASS |
| smooth-roll singleton | no mutation | PASS |
| three missing station slabs, at most four usable sections | no mutation | PASS |
| identical replay and globally reversed face winding | identical decision | PASS |

The fixture construction was corrected before the final run in two ways that
do not alter PSROC: the variable-curvature lane uses 100h axial overlap so its
75-degree chord enters the already frozen 0.35-overlap eligibility bound, and
its physical flanks extend 0.40g so the prescribed 0.15h displacement cannot
artificially cross the frozen 0.25g support boundary.

## Resource observation

The eight-fixture process completed in 2.28 seconds wall time with maximum RSS
132,284,416 bytes (approximately 126.2 MiB), far below the 16 GiB ceiling.
This is a source-free implementation observation, not the prospective-lane
P95 or process-wide resource claim.

## Frozen hashes

- implementation: `f5cfc0393ce45721840f2149ea002737b469754d58ae499abcd78319d30a7528`
- scientific fixtures: `be9bb2c476d74ccc7817fb88fd1d49b6a8e6b97833e02ba353bf053ba53f4791`
- frozen configuration: `89aa7480bd3169bb11f8107d7fe6cf62ccd5c66cf1c7a39a68357c77f0d42ee8`
- original Pro ruling: `1e298fcc20c72e3cbdef5111cbedf2b1e2517f49e86bc20d478c6f85c60dbd8b`
- Pro formula correction: `b31125952ce3d1525569427d932c8577f3b28d06fe4647aa3230ee0e2154cad7`
- consolidated correction record: `1c970094a67ef2835a4ead9bfed7210617a18227b29aa5b206d692f3da5e4286`

## Contract notes

- CPU-only, deterministic binary64 geometry.
- Deletion-only simultaneous pair mutation; no refit, merge, split, append,
  second pass, or NMS.
- Missing or ambiguous evidence produces no deletion.
- Resource failure returns the complete scan's original V52 prediction list.
- Compact audit stores hexadecimal floats, ordinals, station summaries,
  statuses, decisions, and hashes only.
- Source vertices, triangles, section paths, normal samples, labels, CAD,
  counts, identity, and V63 state are not serialized or consumed.

The prospective lane remains unopened. Passing this source-free gate does not
authorize use of V5R3B, scoring, truth opening, retention, runtime/main, or
production promotion.
