# Gold11 V11 simple-family support-radius sensitivity protocol — 2026-09-16

This is a bounded mechanism diagnostic after the focused simple-family run.
It is a new development evaluation and does not reopen the terminal full V11
attempt or change any scorer threshold.

## Fixed scope and hypothesis

The exact fixed cases are `CG4BDD1371B6` / `37361005`, `CG191DEC7F81` /
`49001000`, and `CG4FAF273093` / `49024000`.  The hypothesis is that the
3.0 mm raw-scan support gate rejects otherwise present bend axes because the
transformed native axis is offset from the scan's manual line convention by a
few millimetres.  The sweep is diagnostic; it does not select a production
radius.

## Frozen candidate and limits

The runner is
`scripts/run_gold11_v11_simple_family_support_radius_sweep_v1.py`.  It uses
the V2 healing candidate, the retained pose evidence, 50,000 prepared scan
rows, and `OMP_NUM_THREADS=1`.  The only varied parameter is
`support_radius_mm ∈ {3.0, 3.5, 4.0, 4.5, 5.0}`.  All records and predictions
are retained for each radius.  Gold is opened only after all 15 inference
attempts have returned.

The source hashes at freeze are:

| source | SHA-256 |
| --- | --- |
| `scripts/run_gold11_v11_simple_family_support_radius_sweep_v1.py` | `ec7bae6b4ba1abb50c493bcf619d0e9859b41adb4b23d7cc3122172f3e382a18` |
| `backend/cad_guided_v11_topology_recovery_v2.py` | `f745c3f8471971e2ccae95d319aff174857f41ad1db88ea8c9d0f302c3f0c413` |

## Scoring and decision boundary

The unchanged `UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST` matcher and
existing thresholds score the selected reference rows after inference.  Scores
are reported by radius and remain a 12-line subset diagnostic; they cannot
replace the full 71-line Gold11 score or establish historical comparability.
No radius is promoted based on this sweep.  A useful result must be followed by
independent thickness/centerline and genuine-near-surface falsifier tests
before a candidate change is considered.

## Reproducible command

```text
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_gold11_v11_simple_family_support_radius_sweep_v1.py \
  --output docs/research/gold11_v11_simple_family_support_radius_sweep_v1
```
