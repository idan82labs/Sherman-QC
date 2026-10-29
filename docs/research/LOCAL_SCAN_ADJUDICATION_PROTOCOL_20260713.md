# Local Scan Candidate Adjudication Protocol (2026-07-13)

## Scope

The content-hashed local capacity audit identifies 25 unique geometries that
are not release truth. The prediction-hidden queue and reviewer template are:

- `output/migration/local-scan-adjudication-v1/review_queue.json`
- `output/migration/local-scan-adjudication-v1/reviewer_template.json`

The queue is an administrative provenance record. Reviewers receive separate
copies of the reviewer template, not the queue, because the queue records CAD
references that must remain hidden during the first pass.

## Current readiness

- 20/25 geometries have exactly one resolved part identity.
- 5/25 require part-identity resolution.
- 2/25 geometries, both associated with `49125000`, have a drawing available.
- 23/25 lack a locally accessible manufacturing drawing and cannot complete a
  drawing-backed first pass.

No legacy bend total, F1 output, candidate count, route, or holdout membership
is present in reviewer entries. Geometry, drawing, queue, registry, and CAD
identities are SHA-256 locked. CAD paths are retained only in the administrative
queue under `cad_evidence_withheld_until_reconciliation`.

## Independent first pass

Each reviewer must use a separate template copy and preserve these assertions:

```json
{
  "independent_of_other_reviewer": true,
  "f1_outputs_seen": false,
  "legacy_counts_seen": false,
  "holdout_membership_seen": false,
  "cad_evidence_seen": false,
  "source": "raw_geometry_and_manufacturing_drawing_only"
}
```

For every candidate, the reviewer must independently establish:

1. Whether the source is a measured scan, CAD tessellation, or still unknown.
2. The stable part identity.
3. Full/partial scan state and formed-stage state.
4. Typed feature family.
5. Visible conventional-bend count, without counting rolled transitions,
   forms, hems, holes, or free edges.
6. Bend-instance annotations in `reference_mesh_raw` coordinates.

Two matching count reviews are insufficient for release unless measured-scan
provenance, drawing evidence, and reconciled instance geometry are also
available. CAD feature history may be consulted only after both first-pass
reviews are frozen, during reconciliation.

## Rebuild

```bash
.venv/bin/python scripts/build_local_scan_adjudication_queue.py \
  --capacity-audit output/migration/local-scan-expansion-capacity-v1.json \
  --registry data/bend_corpus/annotations/ground_truth_registry.json \
  --drawing-root data/bend_corpus/incoming/2026-07-06_eyal_buingu_50_parts/extracted \
  --queue-output output/migration/local-scan-adjudication-v1/review_queue.json \
  --template-output output/migration/local-scan-adjudication-v1/reviewer_template.json
```
