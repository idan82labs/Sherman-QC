# Gold11 V11 simple-family diagnostic protocol — 2026-09-16

This protocol freezes a focused diagnostic of the V11 healing candidate on the
lower-complexity families in the already consumed Gold11 cohort.  It is a new
development run, not a retry or reopening of the terminal full V11 attempt.

## Fixed scope

The subset is fixed before execution by these three explicit part keys and
their hash-bound reviewed cases:

| part key | case token | reference id |
| --- | --- | --- |
| `37361005` | `CG4BDD1371B6` | `corpus/05_37361005` |
| `49001000` | `CG191DEC7F81` | `corpus/10_49001000` |
| `49024000` | `CG4FAF273093` | `corpus/11_49024000` |

The runner authenticates the complete September Gold11 review list (eleven
scans, six CADs) and the unchanged selected reference snapshot before scoring.
No new parts, email attachments, held-outs, or unrelated datasets are read.

## Candidate and limits

The candidate is
`V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING` in
`backend/cad_guided_v11_topology_recovery_v2.py`.  It retains accepted V11
contexts and applies the fixed rejected-context gate (`p95 <= 2.0 mm`, coverage
`>= 0.55`) when an accepted context is absent.  Raw scan support trims each
line; native STEP supplies topology proposals only.

The runner is
`scripts/run_gold11_v11_simple_family_diagnostic_v1.py`.  Its fixed limits are
50,000 prepared scan rows, 3.0 mm support radius, 201 line samples, one heavy
job, and `OMP_NUM_THREADS=1`.  Unsupported or ungated cases remain explicit
abstentions.  Every selected case and prediction is retained in the result.

The frozen source hashes at execution are:

| source | SHA-256 |
| --- | --- |
| `scripts/run_gold11_v11_simple_family_diagnostic_v1.py` | `d7f86359041d47d35090e9a7c199c43b075d1f875e9d2a419521100c869017ef` |
| `backend/cad_guided_v11_topology_recovery_v2.py` | `f745c3f8471971e2ccae95d319aff174857f41ad1db88ea8c9d0f302c3f0c413` |
| `scripts/run_gold11_v11_multi_scan_luna_max_v1.py` | `07ff813521c224a321c89692f4c43b7d64703311f8050ca99739fe1152d3fa54` |

## Scoring and inspection

All three inference attempts complete before the selected Gold records are
opened.  The unchanged `UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST` matcher
and its existing thresholds score the 12 reference lines in this diagnostic
subset.  The subset score is reported separately and must not replace the
full 71-line Gold11 headline score.  Per-family counts, failure modes, pose
source, support fractions, distances, endpoints, and matcher assignments are
retained.  Derived 3-D overlays may be generated only after inference for
visual diagnosis; they are not inputs to inference.

## Decision rule

This run is diagnostic.  A higher subset score cannot establish historical
comparability or promotion.  A candidate change is eligible for a later
multi-scan paired evaluation only if the inspection identifies a mechanism,
the fix is independently tested on adversarial synthetic geometry, and the
full source/protocol/evaluator/input bindings are frozen before that run.
The terminal full V11 attempt remains untouched.

## Reproducible command

```text
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_gold11_v11_simple_family_diagnostic_v1.py \
  --output docs/research/gold11_v11_simple_family_diagnostic_v1
```
