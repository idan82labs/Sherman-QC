# Gold11 V11 multi-scan Luna-max paired protocol — 2026-09-15

This protocol freezes a paired consumed-development evaluation of the V11
healing candidate.  The terminal full V11 attempt is not reopened, edited, or
rescored.  Both retained V11 arms are run once through the new candidate; all
22 case/arm outcomes are retained, including abstentions.

## Fixed scope

- Eleven hash-bound scans, six CADs, and the unchanged 71-line selected
  reference snapshot from the September 9 review.
- Case authority: `scripts/review_gold11_inputs_v1.py`, SHA-256
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
- Catalog SHA-256:
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`.
- Selected reference mapping SHA-256:
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
- No new email parts, held-outs, unrelated datasets, hardware, deployment,
  shared infrastructure, or external service is accessed.

## Frozen candidate and execution

The candidate is
`V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V2_PROVISIONAL_POSE_HEALING`.
The paired runner is frozen at:

| source | SHA-256 |
| --- | --- |
| `backend/cad_guided_v11_topology_recovery_v2.py` | `f745c3f8471971e2ccae95d319aff174857f41ad1db88ea8c9d0f302c3f0c413` |
| `scripts/run_gold11_v11_multi_scan_luna_max_v2_paired.py` | `1970dc7b65e762390fd5516dbf5ccc6fad48124879392927e09025a913b7a6bf` |
| `scripts/run_gold11_v11_multi_scan_luna_max_v1.py` | `07ff813521c224a321c89692f4c43b7d64703311f8050ca99739fe1152d3fa54` |

Both arms use the same global pose-healing gate: retained V11 accepted
contexts are preferred; otherwise a retained finite registration candidate is
eligible only with p95 proximity upper bound `<=2.0 mm`, represented scan
coverage lower bound `>=0.55`, and a proper rigid transform.  Candidate
selection is deterministic by p95, RMS, descending coverage, and index.  Raw
scan support trims every emitted line; CAD supplies nominal topology only.

Fixed numerical limits are 50,000 prepared scan rows, 3.0 mm support radius,
201 samples per line, one heavy job at a time, and `OMP_NUM_THREADS=1`.
Malformed or unsupported cases remain explicit abstentions.  No candidate is
reopened or retried inside the evaluation.

Gold is loaded only after all 22 inference attempts complete.  The unchanged
one-to-one minimum-cost matcher then scores control and challenger separately,
each over all 71 references.  The paired ledger reports per-case deltas but
does not alter the scorer or thresholds.

The first paired probe was retained as diagnostic because its per-case delta
field read a non-existent matcher F1 field.  The frozen run uses the corrected
count-derived delta function; its predecessor is not overwritten or reused as
the official result.

## Release boundary

The run is research-only consumed-development evidence.  It does not establish
historical comparability, calibrated uncertainty, manufacturing conformance,
unseen-family generalization, or floor release.  Promotion requires independent
audit, adversarial multi-part replay, and integration into a maintained engine
behind a controlled path.
