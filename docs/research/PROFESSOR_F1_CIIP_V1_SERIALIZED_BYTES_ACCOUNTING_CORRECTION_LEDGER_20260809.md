# CIIP-V1 serialized-byte accounting correction authority ledger

Date: 2026-08-09  
Primitive: `CIIP_DIGITAL_OCCUPIED_SHELL_H0_V1`  
Preserved ruling: `PROFESSOR_F1_CIIP_V1_ONE_SHOT_STOP_ACCOUNTING_RULING_GPT56_PRO_RESPONSE_20260809.txt`

## 1. Exact authority

The sole controlling authority token is:

`CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_AUTHORIZED_NO_EXECUTION`

This authority permits one paper/code-only accounting correction and nothing else. The exact authorized production change is:

> Delete only the default `"serialized_bytes": 0` entry from the `resource_values` mapping constructed by `execute_semantic_role`.

No other production behavior, constant, schema, identifier, parser, sampler, partition, field, graph, persistence, counterfactual, validator, resource cap, watchdog, role, source-selection rule, or terminal token is authorized to change.

Execution authority remains `NONE`. This ledger does not authorize importing or compiling the corrected implementation, running tests, opening a raw PLY, preparing a replacement execution lock, launching any role, performing a benchmark, accessing gold/CAD/labels/expected counts, or integrating with runtime/main/production.

## 2. Consumed claim and terminal STOP preservation

The prior one-shot execution claim is consumed and remains terminal evidence. It must not be deleted, rewritten, reset, superseded, or interpreted as reusable capacity.

The prior terminal outcome remains:

`INVALID_CII_PERSISTENCE_PREFLIGHT_STOP`

The correction does not convert that STOP into a pass and does not reopen the smoke. The following values remain permanently false for the consumed attempt:

- `rerun_authorized`
- `replacement_input_authorized`
- `gold_or_labels_accessed`

No repaired output may be substituted for the stopped attempt. No replacement scan, parameter change, replay, synthetic rescue, or post-hoc completion is authorized.

## 3. Stale V7 code and execution-lock status

Applying the authorized source deletion changes the code bytes and therefore changes the aggregate code SHA-256. Consequently, the V7 code binding and the consumed smoke's preexecution lock become stale for any corrected build.

The old execution lock remains immutable historical evidence. It must not be edited, regenerated in place, relabeled, or reused with corrected code or binaries. A corrected implementation must fail closed against that stale code binding.

This ruling does not authorize construction of a new execution lock. Any later lock preparation or execution requires separate explicit authority after the correction evidence is reviewed.

## 4. Construction-versus-emission accounting contract

The correction preserves the following ownership boundary:

1. `execute_semantic_role` constructs resource observations known before serialization. It must not predeclare or guess `serialized_bytes`.
2. `encode_container` is the sole construction-time owner of `resource.serialized_bytes`. After all six canonical tables and the fixed metadata region are sized, it inserts the exact complete container byte length.
3. A caller-supplied `resource.serialized_bytes`, including zero, remains a conflicting computed-field assertion and must be rejected rather than overwritten or tolerated.
4. Post-emission resource patching may update measured CPU, wall, RSS, and scratch values only. It must preserve the encoder-produced serialized-byte value exactly.
5. Container decoding and the independent validator must require `resource.serialized_bytes == len(exact_container_bytes)` and fail closed on any mismatch.
6. The 64 MiB serialized cap, one-MiB metadata region, canonical JSON length prefix, zero padding, exact six-table directory, contiguous offsets, and no-trailing-byte rules remain unchanged.

This is an accounting ownership correction, not a semantic-algorithm change.

## 5. Required source-free test mapping

Tests may be implemented and run only under later authority that permits them. The minimum required mapping for correction review is:

| ID | Required assertion | Required failure/success behavior |
|---|---|---|
| ACCT-01 | `execute_semantic_role` constructs `resource_values` without a `serialized_bytes` key before emission. | Construction reaches the encoder without a computed-field collision. |
| ACCT-02 | `encode_container` inserts `resource.serialized_bytes` equal to the exact final container length. | Exact equality succeeds. |
| ACCT-03 | Supplying `resource.serialized_bytes` before encoding, including `0`, is rejected. | Fail closed with the computed-field conflict reason. |
| ACCT-04 | Post-emission CPU/wall/RSS/scratch metadata patching preserves serialized-byte accounting. | Patched container validates and retains `serialized_bytes == len(container)`. |
| ACCT-05 | Mutating serialized-byte metadata above, below, or independently of the exact container length is rejected by both neutral decoding and independent validation. | Fail closed with a serialized-byte accounting mismatch. |
| ACCT-06 | All six canonical table byte strings are identical before and after the correction for the same source-free fixture. | Byte-exact table identity is required. |
| ACCT-07 | Case, partition, directory, source, schema, configuration, dependency, and identifier commitments remain unchanged for the same source-free fixture. | Byte-exact/hash-exact identity is required. |
| ACCT-08 | The complete frozen 18-class tamper contract remains passing without weakened expectations. | Any regression remains terminal. |
| ACCT-09 | The execution-result validator continues to accept only the two frozen terminal tokens and preserves false rerun/replacement/gold flags. | Changed token or reopened authority fails closed. |
| ACCT-10 | Corrected code is rejected by the stale consumed execution lock's code binding. | Fail closed before source access. |

No test may read corpus PLY data, gold, CAD, labels, expected counts, prior candidates, renders, benchmark outputs, or the stopped smoke's raw semantic payload.

## 6. Byte identity and noninterference requirements

For identical source-free inputs, the authorized deletion must not change:

- any of the six canonical table byte strings or table SHA-256 values;
- table row counts, row sizes, ordering, offsets, or directory commitment;
- case-token or partition commitments;
- any point, node, cell, field, support, edge, death, or interval identifier;
- parser diagnostics, sample selection, Morton assignment, support membership, tensors, gates, filtration values, H0 dispositions, rejection rows, isotropic relationships, or attribution decisions;
- resource caps, watchdog behavior, role ordering, one-shot enforcement, import isolation, or terminal-result validation;
- closed metadata keys or canonical metadata encoding, except that construction no longer supplies the encoder-owned field prematurely.

The intended emitted `resource.serialized_bytes` value and its exact encoding remain unchanged: they are computed by the encoder from the final container length. Any other byte difference requires STOP and renewed review.

## 7. Changed-file inventory

This ledger records planned scope; it does not assert that the production correction or tests have been applied.

| Classification | Path | Allowed change | Status / hash placeholder |
|---|---|---|---|
| Authority ledger | `docs/research/PROFESSOR_F1_CIIP_V1_SERIALIZED_BYTES_ACCOUNTING_CORRECTION_LEDGER_20260809.md` | New accounting-correction ledger only | `CREATED; SHA256=<TO_BE_RECORDED>` |
| Authorized production | `research/f1_ciip_v1/semantic_core.py` | Delete only default `"serialized_bytes": 0` from `execute_semantic_role.resource_values` | `NOT_APPLIED_IN_THIS_LEDGER_STEP; BEFORE_SHA256=<TBD>; AFTER_SHA256=<TBD>` |
| Required tests | `<SOURCE_FREE_ACCOUNTING_TEST_FILE_TBD>` | ACCT-01 through ACCT-10 only | `NOT_APPLIED_IN_THIS_LEDGER_STEP; SHA256=<TBD>` |
| Runner/container/validator | `<NONE_AUTHORIZED_WITHOUT_SEPARATE_JUSTIFICATION>` | No production change under this ruling | `UNCHANGED_REQUIRED` |
| Prior ruling, claim, lock, STOP evidence, V7 packet | `<PRESERVED_EXISTING_ARTIFACTS>` | No edits | `IMMUTABLE` |

Any changed file not listed by a later completed inventory is out of scope. The completed correction record must replace every `<TBD>` placeholder with exact paths and SHA-256 values before review; it may not modify this preserved planning ledger in place if immutability has already been asserted.

## 8. Sole next state

The only authorized next state is a paper/code-only correction record demonstrating the exact one-line deletion and source-free evidence mapped above. It must continue to state:

- execution authority: `NONE`;
- prior result: `INVALID_CII_PERSISTENCE_PREFLIGHT_STOP`;
- consumed claim: terminal and non-reusable;
- rerun, replacement, and gold access: false.

Nothing in this ledger authorizes another smoke or weakens the frozen V7 terminal contract.
