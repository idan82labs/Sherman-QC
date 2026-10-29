# Scan Benchmark Expansion to 50 — Verified Data Gap (2026-07-13)

## Result

The local archive cannot currently support a 50-scan release benchmark.

- 13 unique scan geometries are release-eligible.
- 20 additional unique geometries are registered but still require provenance
  confirmation and/or independent adjudication.
- 5 unique PLY geometries are not registered and require both provenance and
  adjudication.
- Four duplicate file paths collapse under SHA-256 identity.
- The optimistic local ceiling is therefore 38 unique geometries, assuming
  every pending item proves to be a measured scan and becomes valid truth.
- Even under that optimistic assumption, at least 12 new external measured
  scans are required to reach 50.
- From the current verified state, 37 additional verified scans are required.

The exact inventory is generated at
`output/migration/local-scan-expansion-capacity-v1.json`. It is a capacity
audit, not truth and not a benchmark slice.

The 25 pending unique geometries now have a hash-locked, prediction-hidden
review queue at `output/migration/local-scan-adjudication-v1/`. Only two
geometries have an accessible drawing; 23 remain drawing-blocked and five also
need part-identity resolution. The workflow is documented in
`docs/research/LOCAL_SCAN_ADJUDICATION_PROTOCOL_20260713.md`.

## Why the optimistic ceiling is not a commitment

The 25 pending local geometries include legacy-unadjudicated records,
pre-bending geometry, excluded cases, STL/OBJ records with uncertain scan role,
and unregistered PLY files. Some may be CAD tessellations or may not represent
the countable formed state. None is promoted merely because its filename says
`scan` or its extension is PLY.

The 50 Eyal packages do not change this scan count. They contain drawings,
SLDPRT files, and STL meshes, but no proven measured scan. Their native feature
history remains reviewer-assist evidence only.

## Minimum request for each new benchmark item

1. Raw measured scan (`.ply` preferred), before any CAD substitution or mesh
   cleanup that destroys scanner provenance.
2. Stable part ID and scan ID.
3. Scanner/export provenance and confirmation that the mesh is a physical
   measurement rather than CAD tessellation.
4. Full-versus-partial scan state and formed-stage state.
5. Manufacturing drawing PDF and CAD reference (`STEP` or native SLDPRT).
6. Independently reviewed visible conventional-bend count.
7. Bend-instance locations in the raw scan coordinate frame, plus typed
   exclusions for rolled/rounded transitions, hems, forms, holes, and free
   edges.

The practical request should be for at least 37 new complete scan+truth records.
Requesting only 12 assumes all 25 pending local candidates survive provenance,
semantic, and instance-geometry review, which is unlikely.

## Reproduce

```bash
.venv/bin/python scripts/audit_local_scan_expansion_capacity.py \
  --registry data/bend_corpus/annotations/ground_truth_registry.json \
  --scan-root /path/to/source-assets \
  --target-scan-count 50 \
  --output output/migration/local-scan-expansion-capacity-v1.json
```
