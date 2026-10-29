# Luna-max V11 multi-scan checkpoint — 2026-09-15

The terminal full V11 attempt was not reopened.  The scheduled Gold11 heartbeat
was stopped at the user's request.  This checkpoint records a new, separately
versioned healing candidate and its one frozen development run.

## Candidate

`V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING`
(`backend/cad_guided_v11_topology_recovery_v2.py`) preserves a retained V11
accepted context when available.  If V11 had no accepted context, it uses a
retained finite registration candidate only when the candidate's own published
finite-distance p95 upper bound is `<= 2.0 mm` and represented scan coverage is
`>= 0.55`.  Selection is deterministic and label-independent.  Native STEP
axes remain topology proposals; finite endpoints and support statistics come
from the raw scan.  The source, scorer, input hashes, 50,000-row preparation
limit, 3 mm support radius, and 201 samples are frozen in
`2026-09-15_gold11_v11_multi_scan_luna_max_v1_protocol.md`.

## Run and score

Run command:

```text
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_gold11_v11_multi_scan_luna_max_v1.py \
  --output docs/research/gold11_v11_multi_scan_luna_max_v1_run1
```

The reproducible result is in
`docs/research/gold11_v11_multi_scan_luna_max_v1_run1/RESULT.json` with result
checksum `c7a34191f6697a5ce1c2df0a285ca93a0863793a4852a9429d651f457b9f22af`.
All eleven cases and all 71 references remain in the denominator:

| metric | result |
| --- | ---: |
| TP / FP / FN | 24 / 17 / 47 |
| precision | 0.5853658537 |
| recall (bend-count capture) | 0.3380281690 |
| F1 | 0.4285714286 |
| cases with finite output | 7 / 11 |
| explicit abstentions | 4 / 11 |

The three cases that already had accepted V11 contexts produced 14 TP, 5 FP,
and 5 FN over 19 references; four additional rejected-context poses passed the
new gate.  Cases without a finite gated pose were retained as abstentions, not
converted into failures or fabricated outputs.

This is a numeric consumed-development baseline, below historical F1
`0.4580152672`; historical comparability remains unestablished, so no claim of
beating the historical benchmark is made.  The candidate is not production or
floor eligible: datum/calibration and uncertainty are unverified, CAD nominal
radii are not measurements, and observed manufacturing-event count is unknown.
Promotion requires an independent audit, adversarial multi-part replay, and
integration behind a controlled maintained-engine path.

## Paired arm run

The paired protocol and runner were then frozen to evaluate both retained V11
arms once.  The corrected run is in
`docs/research/gold11_v11_multi_scan_luna_max_v2_paired_run2/RESULT.json` with
result checksum `2bc5731cd5df6c7e3dd9f4e9219d462fbd7320a704b8cbadba3d6a703efc2b87`.
The first paired probe is preserved as diagnostic because its per-case delta
ledger read a missing F1 field; it was not overwritten.

| arm | TP / FP / FN | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| control | 24 / 17 / 47 | 0.5854 | 0.3380 | 0.4286 |
| challenger | 21 / 18 / 50 | 0.5385 | 0.2958 | 0.3818 |

The only material arm delta was `CG4FAF273093`: control recovered 3/3 lines,
while challenger produced 1 unsupported candidate.  Both arms remain
research-only and below the historical number; no comparability or promotion
claim is made.
