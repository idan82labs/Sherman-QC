# V6 noisy capacity successor on the unchanged V5 construction

Date: 2026-09-09. Source-only prospective gate; not executed. The terminal V5
capacity failure remains unchanged. The engineering reason for this successor is
the separately reviewed lossless metadata representation, not a geometry change.

## Fixed comparison and custody

The entire construction, noise, oracle and default geometry/work policy in
[the frozen V5 protocol](2026-09-09_noisy_mesh_capacity_v5_protocol.md) apply
unchanged. V5 test SHA is
`48f4f99537c2456cb40a53ab6165f5c608c75e707274153f42e5a7d079d68795`;
its protocol SHA is
`cd446f96a086ac820630df1aa796ea40fad827c0cc88883af843e7756c09b1c5`.
Neither is imported, edited or rerun. New test
`tests/test_cad_guided_noisy_mesh_capacity_v6.py` is a mechanical source copy
changing only extractor import/call, versioned flag/receipt identifiers, and
adding V6 metadata-schema/work-limit assertions. All original geometry and
lineage assertions remain. Independent review must verify that delta.

The candidate is `backend/cad_guided_noisy_mesh_planes_v7.py`, governed by
[its representation protocol](2026-09-09_noisy_mesh_evidence_v7_protocol.md).
It reuses the frozen V5 geometry helpers. The numeric 64 MiB metadata ceiling
now measures compact schema-table bytes, NOT V5 expanded JSON. Additional
256 MiB expanded-equivalent, 64-million traversal and 4 GiB descriptor-hash
work limits remain as prospectively specified there. The full-size case is
not required to pass merely because compact metadata is smaller.

Pre-execution amendment: this capacity campaign has not run. After the separate
V6 codec78-pass/13-fail gate, its unexecuted test was retargeted to V7 exact
NumPy float64 JSON transport. Only candidate import/call/expected schema changed;
all capacity geometry,noise,oracle and limits remain. Require the new100-case
codec pass and fresh exact-source audit; prior conditional V6 source approval
does not authorize execution of the revised source closure.

## Exact finite gates

First require independent codec/roundtrip/geometry-equivalence controls and
source review to pass. Then the V6 capacity prerequisite route executes the
unchanged two small cases (160/320 constructor and18,304/36,608 actual noisy
extraction), with the sole large case explicitly skipped. Budget:60s wall,
45s worker CPU,1 GiB sampled process-group RSS,1 MiB output. Expected2 pass/1 skip.

Only after both prerequisites are green and the exact full source closure has
an independent pre-run audit may the separately named full route set
`SHERMAN_RUN_NOISY_MESH_CAPACITY_V6=1`. It runs all three cases once, serially:
**120s wall,90s worker CPU,4 GiB sampled process-group RSS,1 MiB output**. These
limits include constructor, extractor, complete oracle and pytest. No changed
started limit, hidden retry, selected prefix or permitted small-case abstention.
The harness records before/after hashes, parent wall/sampled group RSS and
worker CPU/lifetime RSS separately. No other heavy process runs alongside it.

The sole large case remains the shared-index annular prism `(500,250,175)`:
900,000 used raw rows,1,800,000 triangles,2,700,000 shared raw edges. Dyadic
nonzero noise on every axis uses the original modulo sequences/8192;
epsilon stays0.05 mm. Ten evaluator-only analytic faces each require at least
one qualified CORE with50 original rows. Every qualified support witness and
every qualified pure-face normal cone is checked, not just ten representatives.
All original raw/triangle partitions, hashes, immutable supports, aliases,
qualified/rejected dispositions and work counters remain in the oracle.
Neither analytic truth labels nor desired counts enter inference.

The compact receipt prefix is `NOISY_MESH_CAPACITY_V6_RECEIPT=` and its original
32 KiB cap remains. Failed or incomplete outcomes are retained as terminal
negatives. No full raw arrays are printed or written. No physical CAD/scan/Gold,
mail, held-out, native loader, app startup or paid service is involved.

A pass qualifies one declared noisy capacity construction only. It does not
prove physical noise calibration, registration correctness, observed bend
counts, physical F1 or release. The next connected workflow/Gold11 freeze has
its own finite integration and independent audit gate.
