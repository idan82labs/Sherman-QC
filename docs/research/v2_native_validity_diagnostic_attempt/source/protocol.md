# V2 native-validity diagnostic — proposed single-CAD protocol

Status: **source draft; not executed or authorized by this document alone**.
Root owns independent review, source freeze, one-shot supervision and any future
physical access. Do not change or retry frozen V1/R0 or earlier terminal attempts.

## Engineering question and evidence

The frozen V1 result for consumed scan `CG191DEC7F81` / CAD family `49001000`
reports conditional accepted registration but `NATIVE_FEATURES_UNAVAILABLE`.
Its native metadata has `INVALID_NATIVE_SOLID`, no candidates and incomplete
candidate inventory. In the frozen source that reason is emitted by the exact
`BRepCheck_Analyzer(solid, True, False, True).IsValid()` gate, after bounded
topology, numeric tolerance, orientation and edge-use checks. It is not a
failure of the old global 2e-6 tolerance gate. Maximum retained edge/vertex
tolerance is 0.0013532160452609844 mm; face maximum is 1e-7 mm. Those observations
do not identify the invalid entity or justify weakening the validity check.

Evidence read during preparation was restricted to released checkpoints, source,
the frozen V1 configuration and this retained result:

- `gold11_mesh_development_v1_attempt/FROZEN.json`
- `gold11_mesh_development_v1_attempt/cases/CG191DEC7F81/v1/RESULT.json`
- September 9 input-review summary and bound review-script declarations.

The diagnostic asks: which **native self/context validity statuses** explain the
global refusal, and where do they occur in the exact imported topology? It does
not decide that a locally invalid shape is safe for feature extraction. A later
local-admission proposal would need its own mathematical and source audit.

## Exact consumed scope

One CAD only, from the September 9 consumed-input review:

- Family: `49001000`, associated scan token `CG191DEC7F81` for provenance only.
- SHA-256: `d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5`.
- Exact bytes: `260402`; public diagnostic rejects any size/hash mismatch.
- Prior review: one top-level solid, 114 faces, 2026 topology occurrences;
  source length `MILLIMETRE`, transfer target set/query 1.0, one transferred root.

No scan bytes, normals, Gold/reference records, pose, label, desired count or
case-selected policy enters this diagnostic. No other CAD, corpus, held-out,
reserved email input, control mutation or production/floor action is in scope.
The script does not accept a raw filename or an expected-hash override.

## Maintained artifacts and mechanism

- `scripts/v2_native_validity_diagnostic.py`
- `tests/test_v2_native_validity_diagnostic.py`
- This protocol, frozen together with source hashes by root after review.

The public function `diagnose_v2_native_validity(cad_bytes, allow_ocp=True)`
accepts exact immutable bytes. The explicit CLI accepts those bytes on stdin
only. Root's supervisor must read the approved source once, bind the descriptor
snapshot/hash/length, and supply those bytes after durable one-shot STARTED.
No caller-supplied file path is followed by the diagnostic.

One STEP reader performs the same unit-bound import as frozen V1. A source test
compares the function AST to frozen V1, permitting only the descriptive docstring
and private temporary-directory prefix to differ. The transfer's existing native
reader defaults remain unchanged; no post-import healing or tolerance mutation
is called. One native solid handle passes directly to one exact, nonparallel
`BRepCheck_Analyzer`. There is no second/permissive analyzer, additional transfer,
meshing, sampling, inference, feature extraction or scoring.

Before validation, a bounded occurrence tree and location-aware native identity
map are built. Single-child owner containers are retained and mapped; multiple
material owners refuse without running the analyzer. The report contains types,
occurrence/parent/orientation tuples, entity counts and import-local IDs. It then
reads the analyzer's existing self and contextual statuses, plus per-entity
subtree validity, native tolerances and analytic geometry-type names. Contextual
status reads do not call `InContext`, repair an entity or replace its status.
Coordinates, meshes and serialized repaired shapes are not emitted.

OCCT distinguishes a result's own status from its status in an owning shape;
the analyzer also propagates failures through subshapes. Consequently, both
are needed to locate this refusal; a face-level false result alone is not a
diagnosis of that face's underlying surface.
[OCCT analyzer](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_0/src/BRepCheck/BRepCheck_Analyzer.cxx),
[result/context interface](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_0/src/BRepCheck/BRepCheck_Result.hxx).

## Fixed limits and stop behavior

Future root-supervised physical diagnostic, once only:

- 120 s wall, including stdin/private snapshot/import/diagnostics/output.
- 90 s worker CPU; 1 GiB sampled process-tree RSS; 1 MiB combined stdout/stderr.
- Input cap 1 MiB, further restricted to the exact 260402-byte hash above.
- 4096 topology occurrences, depth 32, 2048 unique entities before validation.
- Unique type caps: 32 compounds, 16 compsolids, 8 solids, 64 shells, 256 faces,
  512 wires, 1024 edges, 1024 vertices.
- 1024 entity-detail records, 32 statuses per list, 64 contexts per entity,
  4096 total context records; structured JSON cap 768 KiB.

These are fixed new diagnostic limits, not increases to a prior terminal run.
An independently audited supervisor must enforce wall/CPU/RSS/output limits
externally; the Python script is not a sandbox and cannot interrupt all native
operations itself. One job at a time. Root must isolate the private scratch
directory and restrict actual worker I/O to bound source/runtime, exact input
bytes and its private STEP snapshot/output. No raw CAD path is opened by worker
code; native translation may read that one private snapshot.

Every error, truncation, missing status, API incompatibility and budget stop is
terminal and retained. No retry, healing, cap increase, unfavorable-row removal
or alternate validator follows automatically. Root records STARTED, exact source
and runtime bindings, original output, limits/exit/resource receipt and completion
in a NEW exclusive `v2_native_validity_diagnostic_attempt` directory. The output
directory name is a proposal, not permission to create or execute it now.

## Pre-run checks and interpretation

Default tests are source/double-only: byte/hash/opt-in gates, one-read identity,
failure/no-retry behavior, frozen import-body equivalence, status preservation,
truncation, output refusal and absence of inference/repair imports.

A separately reviewed four-case **invented-only** native matrix is proposed:
valid box; a constructed open-shell invalid solid; multiple owners; and a valid
box written to STEP then imported through the same unit-bound reader. All four
remain scored, with explicit native opt-in and root supervision. Neither that
matrix nor the source tests have been executed by the author. Root should use
60 s wall /45 s CPU /1 GiB RSS /1 MiB output for this constructed gate; it does
not open the bound physical CAD or loosen the physical diagnostic limits.

`DIAGNOSTIC_COMPLETE` means evidence collection completed, **not a valid CAD**.
`native_valid` is the observed frozen-level native result. Missing/truncated
detail is `PARTIAL_DIAGNOSTIC` or `UNAVAILABLE`, never a favorable prefix. Even
fully valid native geometry is not bend completeness, physical conformity,
calibrated uncertainty, scan registration or production authorization. A false
native result and every associated code remain negative evidence; no count/F1
or engine-improvement claim is produced. Frozen V1's zero-line result stays.
