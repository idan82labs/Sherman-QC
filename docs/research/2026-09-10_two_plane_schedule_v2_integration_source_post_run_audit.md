# Scheduler V2 controlled integration — independent exact POST

Date: 2026-09-10. Disposition: **RED, terminal source gate**.
The single attempt returned 25 PASS and 44 setup ERROR outcomes in 7.36 seconds,
not the required 69 PASS. All 44 errors share the same incomplete mock-evidence
construction. Custody and resource checks pass; the integration qualification
does not. Preserve this attempt, sources, archive and negative result unchanged.

This audit used only the three explicitly released terminal attempt files, the
52 frozen sources, their exact PRE and the named 53-payload source archive.
No test, collection, project import, geometry/native execution or rerun occurred.
No physical input, other attempt or private cache was inspected.

## Exact attempt and retained files

Attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v2_integration-hzr4caio`.
Parent reports unified session 43951 terminal with exit 1; the independently read
completion receipt also records exit 1. The exact immediate inventory is the
following three regular, non-symlink files, with no additional entry or scientific
artifact. All were read completely under the frozen bounds, with stable-file
identity checks. Both receipt JSON objects decode without duplicate keys or
non-finite constants.

| File | Bytes | SHA256 |
|---|---:|---|
| `STARTED.json` | 6198 | `cdf9ba1e2939790e981d691abe31bbce44280e873a88014044a2b2382d02c10c` |
| `OUTPUT.txt` | 311003 | `3cef53bb7b33c5e540a55badb82dc55050e4021f8a376329f5b8d7418ded736f` |
| `COMPLETED.json` | 7215 | `8a7d0dcecdb3d9883c0b2e1527a33c0cca29290c319ddf5a6371102250d0ed0c` |

The two receipts are each below 64 KiB; the full output is below 1 MiB and matches
the completion receipt's observed size and SHA256. STARTED fixes 69 outcomes.
OUTPUT contains exactly 44 setup traceback sections, 44 corresponding short
summary error entries and the terminal summary `25 passed, 44 errors in 7.36s`.
Thus all 69 outcomes are accounted for: zero test-body failures, skips, xfails,
xpasses or other reported dispositions; setup errors remain failures of this gate.

## Independently verified cause and reached boundary

Every setup traceback ends with `KeyError: 'scan'` at
`backend/cad_guided_mesh_offline_v12.py:81`. Every one follows the same chain:
the adapter fixture calls `install("v2")` at test line 301; its first actual
registration-binding call at line 298 reaches `_scan_plane_binding` through
`_registration_binding` at backend line 251. After replacing only hexadecimal
object-address strings, all 44 complete traceback bodies are identical
(one 6,846-character normalized body repeated 44 times). The cause is therefore
verified across the entire output, not inferred from one representative error.

The new fixture's `initialization.update(...)` at lines 281–287 supplies
`patch_evidence=dict(cad=...)`, with no `scan` entry. The existing default V7
scan-plane binder permits omitted patch evidence in legacy mock reports, but
once `patch_evidence` is present it requires its `scan` envelope and checks that
envelope's schema. Indexing the absent key fails before the fixture can establish
its intended baseline binding. Even the tests intended to install V1 later fail
in this unconditional initial V2 setup. This immediate cause is a mock-construction
error, not evidence that the intended workflow assertions succeeded or failed.

| Unreached byte-workflow test function | Setup errors |
|---|---:|
| `test_actual_byte_workflows_bind_requested_v2_configuration_before_all_measurements` | 2 |
| `test_omitted_and_explicit_v1_byte_paths_have_identical_outputs_and_no_extra_kwargs` | 2 |
| `test_stale_requested_returned_schedule_contract_never_measures` | 32 |
| `test_second_context_schedule_binding_failure_discards_successful_first_binding` | 2 |
| `test_byte_workflow_request_snapshot_survives_caller_budget_mutation` | 2 |
| `test_byte_request_contradictions_refuse_before_any_input_access` | 4 |
| Total | 44 |

None of those six test bodies ran. In particular, their requested/returned
configuration, whole-prefix clearing, mutation and contradictory-request oracles
are not qualified by this attempt. The prior independent PRE missed this mock
envelope mismatch; its source-review disposition is not a substitute for the
negative execution result.

The other six functions account for the 25 passes (`2+1+9+2+7+4`). They cover
configuration ownership/admission and the bounded actual 50-row prefilter paths,
including two positives and eleven pre-context schedule negatives. Preserve
these passes at that narrow scope. They do not establish complete public-byte
workflow integration, native registration, measurement recovery or physical use.

## Resources and writer custody

| Observation | Actual | Frozen bound |
|---|---:|---:|
| Pytest-reported duration | 7.36 s | Informational |
| Worker CPU from terminal output | 7.357245 s | 45 s |
| Parent wall time | 7.511111417319626 s | 60 s |
| Work stopped | 7.494120083283633 s | 55 s |
| Worker high-water RSS | 126894080 B | Informational alongside sampled bound |
| Sampled process-group peak RSS | 126894080 B | 1073741824 B |
| Complete output | 311003 B | 1048576 B |

The receipt records group 38138 as the only observed group PID, reaped at
7.494121958035976 seconds and checked empty at 7.510955167002976 seconds.
`process_group_confirmed_empty` is true; `requires_group_attention` is false.
Budget-stop, primary-error and cleanup-error fields are null, with no source
after-error. The receipt-identified private cache remains retained and uninspected.
Sampled RSS/cleanup are engineering observations, not hard real-time guarantees;
source hashes and receipt I/O remain outside the supervised clock.

## Complete source and archive consistency

The exact PRE is
`docs/research/2026-09-10_two_plane_schedule_v2_integration_source_pre_run_audit.md`,
15,441 bytes, SHA256
`cf2325d81829c56f1c66acd13d59da671fb8d703d6ef139d8fdb16072e0c1594`.
Its 52 unique source rows agree exactly, in order, with the harness's literal
source tuple. Every current source was read with a 2 MiB per-file bound and
stable regular-file/no-follow checks. All 52 bytesizes and SHA256 values match
the PRE and both complete before/after receipt maps; total source bytes are
1,218,132. The completion consistency flag is independently corroborated.

Archive:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v2_integration_frozen.k_3yvpk6/SOURCE.tar`,
1,328,640 bytes, SHA256
`66a21e47e8dc9e7161a15f8aefc97c4028f4062273e1b34b74d789e87ad71ca3`.
In-memory archive inspection verified exactly 53 unique, named regular payloads:
the same 52 sources plus this exact PRE, totaling 1,233,573 payload bytes. Every
payload equals the corresponding retained/current bytes, with no extra member,
duplicate, symlink, hard link or AppleDouble entry. Nothing was extracted.

The frozen 24-byte `tests/__init__.py` matches its PRE/archive hash. A repeated
`lstat` check confirms the enclosing repository `__init__.py` remains absent,
including no dangling symlink. This closes the reviewed repository package
boundary for this attempt; it does not attest interpreter/library binaries or
repair the separately disclosed older connected-gate closure omission.

## Terminal conclusion

The exact gate is RED because 44 required integration outcomes never reached
their assertions. No retry, changed engine tolerance, cap increase or alteration
of this packet follows. A separately versioned fixture-only successor may correct
the mock's declared V7 envelope while retaining the original engine and oracles,
but requires its own frozen source/protocol, independent PRE and archive before
execution. This audit does not qualify or pre-approve that successor.

The connected 2/4 negative and all native/Gold11 promotion gates remain unchanged.
There is no new physical score, count authority, calibrated uncertainty or factory
release. The only file written by this audit is this new POST document.
