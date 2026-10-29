# V10 B3 exact-query reuse — bounded independent source audit

Date: 2026-09-13. Reviewer: Codex `/root/v9_material_identity_chain`.

## Decision and scope

**Not yet accepted at this draft snapshot.** The full policy-binding and attempted/completed query-accounting corrections are substantively present. A retained-result integrity failure still falls back to a new query and can publish successful context evidence, contrary to the explicit fatal-corruption rule in the V10 coordination checkpoint. The owned result also remains mutable despite its immutable-result description.

Fable is the sole engine/test editor and these are unfrozen drafts. Read `AGENTS.md`, current status and the root V10 coordination checkpoint; read the complete 451-line V5 registration module and 272-line focused test, plus narrow V4 query/diagnosis, surface-index ownership and stage-classification source. No imports, tests, native/geometry execution, physical inputs, references, attempts, controls, filesystem caches or other run artifacts were accessed. This is not a PRE, execution attestation, speedup measurement or physical progress claim. This review does not carry the earlier B1 test result onto subsequently edited integration files.

Before/after source/test hashes were stable (tools `0aec20`, `3102be`, `0ca889`, `34ac96`):

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_global_registration_v5.py` | `4bba69e6ace23fed07c815a6c6291d0e93822f7eae154f8b1e90224bb54f9134` |
| `tests/test_cad_guided_global_registration_v5.py` | `8d61304067583190d854539279a8bfdfb40378a8ef306d85173856fd567368ea` |
| `backend/cad_guided_global_registration_v4.py` | `5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273` |
| `backend/cad_guided_triangle_surface_v4.py` | `f0a01f666f143184960f78f4cec2aab5b7a39510485b136b85b63fdb4225f0d4` |
| `backend/cad_guided_stage_policies_v2.py` | `699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d` |

## Exact reuse path and accepted closures

`_diagnose_with_reusable_query` (:380–405) delegates diagnosis to unchanged V4 through a recorder. It captures the first full-scan query, recomputes the expected `scan @ R.T + t` positions, requires matching shape/values and COMPLETE status, copies the result and records transformed-row bytes, row count, pose bytes, query-object/index identity, distance gate and policy digest. It returns no reusable result if diagnosis itself raises, including a later failed refinement-subset query.

`_context_support_v5` (:408–451) recomputes the bindings. For a valid matching result it records a reuse without another primitive query; a legitimate changed binding calls the same query object afresh. Proximity masks, bounded metrics, thresholds and conditional/non-registration semantics remain the predecessor's. The stage helper revalidates the full query grammar before computing masks. The actual V5 query path also uses V4's index-bound result validation before accepting each completed primitive call.

**Policy binding is materially closed at this scope.** `_query_policy_digest` (:316–331) now covers current surface budgets, execution budgets, chunk size, every field of the validated registration policy, consumer/chunk/reuse implementation declarations and surface result-contract declarations. Reassigning budgets on the same query object no longer evades the key. Existing surface-index geometry arrays use immutable-byte-backed views (`triangle_surface_v4:_read_only`) and index replacement changes the recorded identity. This is an in-process object binding, not a persistent or serialized index-content certificate.

**Attempted versus completed query accounting is materially closed.** `_QueriesV5._one` (:279–292) charges lane rows/calls before invoking the superclass and records elapsed CPU/wall in `finally`; completed counters advance only after the superclass returns a validated COMPLETE answer. A rejected/malformed/failed primitive therefore cannot disappear from the attempted lane ledger. An admission failure before the primitive still counts as an attempted V5 request, not as completed primitive work. `_CpuMeter.summary` publishes both sets separately (:219–230).

Reuse updates only `reused_query_calls/rows` (:311–313), not performed query rows or completion counters. Chunking checks allowance before each primitive block; completed earlier chunks remain charged if a later boundary expires, but no partial merged result is returned. `_merge_chunks` preserves row ordering, sums additive work statistics, takes maxima for batch peaks, and requires complete passes; the merged result is structurally revalidated. These are source observations, not an execution/performance proof.

## P1 — Corrupted retained evidence is silently recomputed

**Location:** V5:421–431; focused test:224–239.

One conditional combines key matching with result integrity. A retained record whose `query.status` changes from COMPLETE to UNAVAILABLE, or whose result bytes change without a matching digest, simply fails that conditional and falls into `answer = query_surface(points)`. A successful fresh query then produces an ordinary context result. The focused tests explicitly require this behavior for an unavailable retained query and a changed distance array.

This is not the same as a valid cached record for a different pose/row order/index/policy. It is an internal retained-evidence integrity failure. The root V10 coordination rule explicitly requires corrupted query results and invariant failures to remain fatal, clearing usable output rather than recovering a successful prefix. The recomputed query may itself be numerically valid, but that does not close the evidence-integrity policy or justify the claim that context support consumed the same immutable answer as diagnosis.

**Minimum closure:** validate the supplied retained record's schema, COMPLETE status and bound result integrity separately from lookup-key matching. Corrupt/malformed retained evidence raises the existing fatal `_SurfaceFailure` (not typed allowance expiry); a valid retained record with a changed pose/rows/index/policy may perform a new query. Change the stale-status/digest-tamper tests to require fatal refusal and no fresh-query/reuse increment. Add one no-retained-record case and retain the existing legitimate-key-mismatch cases. Root's connected caller review owns verifying that this fatal exception clears prior usable candidate output.

## P2 — Owned copy is not yet immutable

**Location:** V5:334–377; focused test:228–239.

The new copy removes aliasing with the primitive's original arrays, an important improvement. However, `np.array(..., copy=True)` owns writable storage; setting its `write` flag false can be reversed with `setflags(write=True)`. The outer `query` and nested diagnostics are ordinary mutable dictionaries. The frozen dataclass only prevents normal replacement of its own attributes. The existing test rejects a direct element assignment but does not test re-enabling writes or mutating those dictionaries.

The digest detects many later mutations, but currently routes them through the nonfatal fallback described above. Its array component includes key, dtype and element bytes, but not shape. Downstream structural validation can reject malformed shapes, so this is not by itself proof of false numerical acceptance; it does mean the purported exact immutable payload is not fully represented by the hash alone.

**Minimum closure:** use the existing immutable-byte-backed array idiom, keep private immutable metadata or an explicitly tamper-evident owned snapshot, and make every retained-result integrity failure fatal before reuse. Bind array shapes and validate the complete result against the expected row count/index. Do not weaken the frozen downstream grammar just to accept a new storage wrapper: reconstruct an owned validated consumer dictionary if necessary. Test original-answer mutation isolation, attempted write re-enablement, diagnostics/status/array/shape tampering, and unchanged reuse equality. No new persistent cache or general caching framework is needed.

## Focused test coverage and limits

- Four small invented surface families and two chunk sizes compare chunked versus whole row values/dtypes and metadata, check additive work identities, and count performed calls/rows. The assertion uses `np.array_equal`, not a byte comparison; the title's “byte-identical” wording is stronger than that check for signed zero. Use explicit bytes if exact bit identity is the acceptance claim.
- The reuse test runs real V4 diagnosis and real V5 support, compares performed/reused metrics, and checks changed translation, row count/order, query object, gate, surface/execution budgets, chunk size and a nongate policy field. Its “query object / index” comment is only partly tested: the new query object uses the same index. Add same-query index replacement if claiming that independent key dimension is exercised.
- The primitive-refusal test verifies 1000 attempted versus 500 completed rows and 2 attempted versus 1 completed calls. Its fake clock has zero increments, so it does not establish nonzero failure CPU timing; the `finally` source nevertheless contains the required accounting.
- Quota tests preserve two completed chunks before a typed third-chunk stop. They do not independently qualify upstream lane allocation, completion enumeration charging, final context publication or a process-stop budget; those belong to the other bounded reviews.

The requested narrow successor direction is sound: reuse exact completed work, preserve per-row results and distinguish performed/reused/failed work. At this draft, close fatal integrity handling and the claimed snapshot-immutability contract, then re-pin and record focused test results actually executed. Do not chase broader geometry redesign or a physical run to validate these local contracts.

## Append-only B3 delta review — source checkpoint stable since 15:07

**Original corruption/immutability findings are closed at source, but the delta is not yet accepted unchanged: one regression in the requested legitimate-index-miss behavior remains.** The original rejected snapshot above is preserved. This review only reconciles its requested fixes and a narrow connected fatal-exception path; it does not broaden into runner, allowance or physical-run qualification.

Before/after hashes match (`dca614`, `f0be84`, `7a4cec`):

| File | Delta SHA256 |
| --- | --- |
| `backend/cad_guided_global_registration_v5.py` | `19137bd8008c43de03fef1f6501c9c4e58eab8dd17868e103d9a67a937370406` |
| `tests/test_cad_guided_global_registration_v5.py` | `5fdb03ea9af47c27754cb61e65dd0211586e8504a89cfc42f8a3a5fc5f5118fc` |

Narrow caller snapshot: `backend/cad_guided_plane_initialization_v16.py` SHA256 `7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b`. Its B3 call/catch/final-publication sections were read, not its entire evolving scheduler. Frozen V4 validator remains `5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273`.

### Closed requested findings

- **Fatal retained corruption:** V5:388–418 now validates retained schema, row ownership, immutable storage, result digest and COMPLETE metadata before considering a reuse hit. Its structural failures raise `_SurfaceFailure`, not `_AllowanceExhausted`; malformed-record exceptions are converted to the same fatal class. Tests:232–258 require status, array, shape, diagnostics, row-count and foreign-record failures without performed-query or reuse increments. The no-record case and legitimate pose/row/policy/query-object changes remain fresh-query cases.
- **Owned immutable storage / tamper detection:** `_FrozenQueryPayload` contains an ordered tuple of row arrays/face IDs and immutable JSON strings, not mutable metadata dictionaries. `_immutable_array` (:347–351) makes private bytes-backed views; `_bytes_backed_read_only` verifies their base chain before consumption. Tests explicitly require element writes and `setflags(write=True)` to fail (:245–250). Consumers receive a reconstructed exact dictionary, so frozen downstream type checks are unchanged.
- **Shape binding:** the digest includes row count, every array key, dtype, shape and element bytes plus both metadata texts (:374–385). Thawing also validates row-leading dimensions and the complete V4 array grammar. The changed-shape test is present. The code owns the original bytes; test-source presence is not an execution attestation.
- **Literal chunk-result comparison:** test:90–91 now compares dtype, shape and `.tobytes()`, including signed-zero distinctions, for each non-face-ID array. Existing geometry and resource-accounting assertions remain.
- **Attempted/completed accounting and policy bindings:** the previously accepted source paths remain in place. This delta does not change their meaning or establish a measured speedup.
- **Connected fatal path:** initializer V16 imports the same V4 `_SurfaceFailure` aliased by V5. At :1786–1806 the B3 support call is inside the candidate block; this exception marks the in-flight candidate FAILED and rethrows, rather than entering typed expiry. The outer handler (:1832–1839) sets UNAVAILABLE, clears selected/measurement contexts and marks the search incomplete. Final subset/numerical completeness remains false and nonconditional return clears owned measurement support (:1854–1880). This verifies the narrow corruption-exception route in source, not all scheduler failure semantics.

### Remaining blocker — intrinsic integrity is checked against a different lookup index

`_thaw_query` validates the retained OLD answer with `index=getattr(query_surface, "index", None)` at V5:414. `_context_support_v5` calls this at :480 **before** comparing the retained and current `index_id` at :481–482. Frozen V4:214–219 requires the query's triangle indices, face IDs and normals to match that supplied index.

Consequently a fully intact record made against index A is falsely treated as corrupt when the same query object legitimately switches to a valid index B with different face IDs, normals or triangle count. It raises `V3_QUERY_SOURCE_IDENTITY_MISMATCH` or `V3_QUERY_NORMAL_BINDING_MISMATCH` before reaching the branch that should perform a fresh query. This is a false fatal refusal, not stale numerical acceptance, but it violates the explicitly requested distinction between corruption and valid changed-index lookup.

The new test (:225–230) only builds the same `box_with_hole` a second time. That changes object identity while keeping the face IDs/normals/triangle layout identical, so the erroneous pre-key validation still succeeds. It does not cover a genuinely different valid index.

**Minimum closure:** validate retained intrinsic integrity, schema/digest and original row count first without substituting the current index for the creation index. Once the complete key matches, bind the retained answer to the current index before reuse. On a valid different-index key, query that index afresh. Alternatively retain the original owned index and validate the old record against that identity before lookup. Preserve fatal handling for actual retained corruption even when other key fields differ. Add one same-query-object replacement with a genuinely different valid index (different geometry/face identities or normals), require `PERFORMED_NEW_QUERY`, and compare its result with a no-reusable fresh call. No full benchmark is required for this closure.

Execution status: the parent reports an eleven-suite process ended but has not received its final outcome/exit checkpoint. This reviewer ran no tests or imports and claims no pass count, runtime speedup, PRE or physical progress. Stop this review at the pinned delta; do not chase the author's in-progress runner edits.

## Append-only final B3 ordering repair — narrow source acceptance

**ACCEPT the remaining different-valid-index source fix.** The earlier rejected snapshots above remain preserved; no B3 source blocker remains within this bounded review scope.

Before/after hashes match (source reads `2d42e3` and the final hash check):

| File | Accepted SHA256 |
| --- | --- |
| `backend/cad_guided_global_registration_v5.py` | `00c1eab81fe016faa44f0a569e40e06b6b37f4275badadb2dc16656e4bcc6f19` |
| `tests/test_cad_guided_global_registration_v5.py` | `14dcb8bc29765bc9687e12930dd120a288d475fc1b88d0c0b1daeeb876095bb0` |

V5:388–417 now checks intrinsic retained integrity against its original row count with no index by default. The context consumer calls this at :485 before key comparison, preserving fatal schema/storage/digest/status handling. Only after all lookup keys match does :488 validate the retained answer against the current index. A valid different index instead reaches the real fresh-query call at :493. This correctly separates a legitimate lookup miss from retained corruption without relaxing index validation on actual reuse.

The focused test now replaces the same query object's index with the genuinely different `stepped_plate` mesh (:225–231), requires `PERFORMED_NEW_QUERY`, and then separately checks an identical-geometry/new-object replacement (:232–234). These real index/query calls exercise the formerly failing order; existing fatal-tamper checks and unchanged-binding reuse remain. The different-index assertion checks routing and row binding, not an additional independent geometric-accuracy oracle, which is sufficient for this ordering repair.

Fable's focused **14-pass** receipt was reported to this reviewer by the parent at these hashes. This reviewer did not execute tests or independently inspect execution artifacts. Acceptance is source-level closure of the requested B3 contracts, not a speedup result, full successor integration, PRE, physical-run authorization or new F1. No further source edits or review expansion were performed.
