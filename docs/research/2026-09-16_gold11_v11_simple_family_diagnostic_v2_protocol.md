# Gold11 V11 simple-family diagnostic protocol — successor V2 — 2026-09-16

The predecessor launch using
`run_gold11_v11_simple_family_diagnostic_v1.py` is terminal.  It stopped
before any raw scan or CAD bytes were read because its deterministic fixed
scope assertion expected the selected cases in a different order than the
review source.  It produced no inference, score, or output directory.  This
successor corrects only that order mapping and is bound to a new output.

## Fixed scope

The subset is fixed by explicit part keys and hash-bound reviewed cases:

| part key | case token | reference id |
| --- | --- | --- |
| `37361005` | `CG4BDD1371B6` | `corpus/05_37361005` |
| `49001000` | `CG191DEC7F81` | `corpus/10_49001000` |
| `49024000` | `CG4FAF273093` | `corpus/11_49024000` |

The successor authenticates the complete September Gold11 review list (eleven
scans, six CADs) and the unchanged selected reference snapshot.  No new parts,
email attachments, held-outs, or unrelated datasets are read.

## Frozen source and candidate

The successor entrypoint is
`scripts/run_gold11_v11_simple_family_diagnostic_v2.py` with source SHA-256
`448becf37d8745a22d70828f3c8952b0493a8dcbabc9647e19040ef0fb3a175a`.  It reuses the predecessor implementation's
inference, limits, and scorer without changing their behavior.  The candidate
is `V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING`
in `backend/cad_guided_v11_topology_recovery_v2.py` (SHA-256
`f745c3f8471971e2ccae95d319aff174857f41ad1db88ea8c9d0f302c3f0c413`).
The predecessor source is retained at its original hash and is not edited.

Fixed limits are 50,000 prepared scan rows, 3.0 mm support radius, 201 line
samples, one heavy job, and `OMP_NUM_THREADS=1`.  Unsupported or ungated cases
remain explicit abstentions.  Every selected case and prediction is retained.

## Scoring and inspection

All three inference attempts complete before selected Gold records are opened.
The unchanged `UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST` matcher scores the
12 reference lines in this diagnostic subset.  The subset score is reported
separately and cannot replace the full 71-line Gold11 score.  Per-family counts,
failure modes, pose source, support fractions, distances, endpoints, and
matcher assignments are retained.  Derived overlays may be generated only
after inference.

## Reproducible command

```text
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_gold11_v11_simple_family_diagnostic_v2.py \
  --output docs/research/gold11_v11_simple_family_diagnostic_v2
```
