# Noisy-mesh capacity V5: independent terminal-result audit

Date: 2026-09-09. Reviewer: `/root/astra_repo_guidance_review`.

**Verified terminal negative; no full-size capacity qualification.** The one
900,000-row / 1,800,000-triangle invented case returned `UNAVAILABLE` with
`MAX_METADATA_BYTES_EXCEEDED`. Its positive oracle correctly failed. This was an
internal metadata-admission refusal, not an outer wall/CPU/RSS/output stop. It
neither establishes a geometric failure of the construction nor establishes that
the geometry would pass after a representation change.

This audit read the named retained controls/output, the complete 407-line test,
the prospective protocol and pre-run audit, and relevant extractor/harness
source. It performed read-only hash and receipt comparisons. It did not import
or execute the extractor/tests, reconstruct a mesh, access original/native/
physical/Gold inputs, signal a process, or modify frozen evidence. This new note
is the only authored file.

## Custody and fixed gates

The [pre-run audit](2026-09-09_noisy_mesh_capacity_v5_pre_run_audit.md) records the
source prerequisite and its conditional release of one full invocation under
the [prospective protocol](2026-09-09_noisy_mesh_capacity_v5_protocol.md).
Both gates used exactly the following six-file closure. The reviewer rehashed
all six current source files; each matches both gates' `STARTED.json` and
`COMPLETED.json` maps and the pre-run audit. The maps have exactly six entries.

| Source | SHA-256 |
| --- | --- |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `tests/test_cad_guided_noisy_mesh_capacity_v5.py` | `48f4f99537c2456cb40a53ab6165f5c608c75e707274153f42e5a7d079d68795` |
| `docs/research/2026-09-09_noisy_mesh_capacity_v5_protocol.md` | `cd446f96a086ac820630df1aa796ea40fad827c0cc88883af843e7756c09b1c5` |
| `scripts/check_cad_guided_measurement.py` | `da3828c14c38f6eeabe1abadd00b7f455ebf421f0526abe76e25c597860237b7` |

| Retained gate | Prerequisite | Full capacity |
| --- | --- | --- |
| Attempt | [`noisy_mesh_capacity_v5_source-_09yfli0`](cad_engine_v1_synthetic_checks/noisy_mesh_capacity_v5_source-_09yfli0/COMPLETED.json) | [`noisy_mesh_capacity_v5-ksogonyt`](cad_engine_v1_synthetic_checks/noisy_mesh_capacity_v5-ksogonyt/COMPLETED.json) |
| Test result | 2 passed, 1 declared skip | 1 failed, 2 passed |
| Exit code | 0 | 1 |
| Fixed wall / worker CPU / sampled group RSS / output cap | 60 s / 45 s / 1 GiB / 1 MiB | 120 s / 90 s / 4 GiB / 1 MiB |
| Parent wall seconds | 0.6037065829150379 | 6.583640833385289 |
| Worker lifetime CPU seconds | 0.44826 | 6.545417 |
| Parent sampled group peak RSS bytes | 98,107,392 | 979,861,504 |
| Worker lifetime peak RSS bytes | 100,220,928 | 982,007,808 |
| Retained output bytes | 101 | 5,783 |
| Outer budget stop | null | null |
| Source before/after consistent | true | true |

The exact output hashes, independently verified against retained bytes, are
`1a15f37335a6e0e7dfc387d00073bee23a4ae342e2b0e5c08c086434d8ea5e59`
for the prerequisite and
`957cfa8b576e8d54a05f5566d6c768fda6996e5c08f06ddecd998e56c896f5a0`
for the full gate. Both gates' limits are identical before and after execution.
The parent sampled peak and worker lifetime peak are different measurements;
neither is a certified instantaneous process-tree peak.

The source route has no capacity opt-in flag. Only the separate full route sets
`SHERMAN_RUN_NOISY_MESH_CAPACITY_V5=1`. The full test source contains one large
extractor call and no retry loop. Its two small controls are repeated as expressly
required components of that separate gate, not retries of the large failure.
The audited receipts show no source change, budget shift or retry of the large
case. This is a scoped custody check of these named attempts, not an assertion
about uninspected external execution history.

## What stopped, and what the counters mean

The [full retained output](cad_engine_v1_synthetic_checks/noisy_mesh_capacity_v5-ksogonyt/OUTPUT.txt)
contains one compact `NOISY_MESH_CAPACITY_V5_RECEIPT` with extractor status
`UNAVAILABLE`, oracle status `FAILED`, and `AssertionError` at the oracle's first
assertion (`result["status"] == "COMPLETE"`, test line 175). The extraction call
reported 6.067123834043741 seconds; this excludes construction, the small cases,
and outer harness work and is not the end-to-end qualification time.

| Attempted extractor evidence | Retained value |
| --- | ---: |
| Input raw rows / original triangles | 900,000 / 1,800,000 |
| Edge records / shared raw-edge pairs | 5,400,000 / 2,700,000 |
| Appended/evaluated core records before fatal clear | 22,452 |
| Qualified cores / discarded proposal records | 13,847 / 13,847 |
| Fit evaluations | 22,453 |
| Charged raw memberships / full-support row tests | 1,999,107 / 1,999,107 |
| Charged triangle memberships | 5,551,393 |
| Charged retained-array bytes before fatal clear | 89,204,000 |
| Charged metadata bytes / unchanged metadata cap | 67,106,100 / 67,108,864 |
| Scheduled/evaluated closures, closure tests, cone-row tests | all zero |

V5 `_append` serializes `_metadata(record)` as expanded dictionary-key JSON,
charges its bytes, and only then appends it (extractor lines 103–116). The
metadata ledger had 2,764 bytes of remaining capacity; the next charge exceeded
that allowance. The fit count is one greater than the appended/evaluated core
count. Together with the core-loop order (fit, record admission, then core and
qualified counters), this identifies a fitted next core whose metadata was not
admitted. It is not a failed geometric fit or an accepted extra core. Its exact
record size and its qualification are not retained and are not inferred.

The failure handler clears every core, closure, proposal and partition payload,
marks all completion flags false, and preserves attempted counters and discarded
record counts (extractor lines 139–149). Thus `core_count=0` is an uncompleted
final-accounting field, not evidence of zero constructed or physical planes.
The 13,847 partial qualifications are no longer usable proposals. The charged
array-byte counter is not a measurement of live memory after clearing.

The full-size oracle did **not** reach its input-digest, once-only partition,
all-lineage, all-qualified-witness, true-normal-cone, or ten required
representative checks. The extractor itself did not reach closure scheduling or
closure geometry. Those full-size claims remain unproved. The two small controls did
pass, including the real 18,304-row / 36,608-triangle extractor case and its
ten-representative oracle. That evidence cannot substitute for the larger case.

## Prospective representation successor, not a rescued V5 result

The proposed separately versioned V6 mechanism is reasonable **at design level**:
reuse frozen V5 graph/fit/cone/scheduling math and unchanged internal dictionaries
and immutable arrays, but encode repeated ordered dictionary schemas once.
Tagged dictionary and sequence nodes can preserve the logical metadata tree
without repeating each record's field names. Array metadata remains the exact
shape/dtype/hash descriptor; decoding metadata does not reconstruct the arrays.

Keeping the number 64 MiB does not keep the V5 budget metric unchanged. V5 caps
expanded dictionary-key JSON; V6 would cap a different, losslessly encoded
transport. This is an explicit representation-contract change, not evidence that
V5 passed, a cap-free retry, calibration, or a geometry correction. The original
fixture, nonzero dyadic perturbations, stipulated 0.05 mm error radius, all
records/refusals, selected schedules, and positive oracles must remain frozen
for a comparable successor experiment. This terminal V5 negative remains locked.

Before any successor execution, independent source/tests must establish:

- Exact tagged round trips of every logical key, scalar/type, sequence order,
  refusal, null/false value and array descriptor; no dropped records or rounding.
- Actual encoded-byte accounting includes schemas, IDs, values, delimiters,
  envelope and every late-mutated field. A fresh final whole-tree encoding must
  be checked, not merely summed early record estimates.
- Final metadata and encoder-work counters agree with the published payload;
  charging encoder visits while serializing mutable statistics must not leave
  stale self-referential accounting.
- Strict schema-ID/type/arity/key-uniqueness/finite-value checks and separately
  bounded encode/decode depth, schema/key counts and cumulative node work.
  Overflow or malformed transport yields whole refusal, never an accepted prefix.
- The same outer resource envelope and exact new source/protocol/test/harness
  closure are prospectively reviewed; decoding/in-memory work is not treated as
  free merely because encoded bytes are smaller.

The outlined 256-schema / 128-key / 128-character-key / depth-64 limits and
64-million cumulative encoder-node budget, with a separate decoder-node budget,
are useful bounded design controls, not a completed implementation audit.
No compression ratio, remaining-closure success, full-size geometry success,
108-case robustness, registration gain, physical accuracy or promotion is
established here. A new named, audited successor experiment is required to learn
its outcome; no execution is authorized by this result note.
