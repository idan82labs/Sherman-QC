# Context V3 integration V2 — independent terminal failure POST

Date: 2026-09-12. Reviewer: process_v2_qualification, independent of test/design and harness/protocol authors.

Disposition: TERMINAL FAIL, 93 passed / 1 failed out of the frozen 94. Evidence publication/custody reconciles; full integration qualification does not. The failed outcome remains in the denominator. No rerun, source repair, test import/collection/execution, native call, process query/signal, old-attempt access or private-cache traversal was performed by this reviewer.

## Exact accepted release and actual attempt

Root reports it fully read and accepted the 23202-byte PRE before archive check e1ff71. PRE path:
`docs/research/2026-09-12_context_v3_engine_integration_v2_source_pre_run_audit.md`;
SHA256 `d21838fc43776c40808b476ca17cd46607639738a5a3d2474d79b16d6b9ae1b9`.

Root's sole fixed invocation terminated with exit 1 (terminal tool 74926f), outcome gate false, final publication gate true, and reported parent-through-inventory wall about 4.371895 seconds. Root's separate external process observation 9c65ff found no Sherman worker. Those command/process observations are root-provided execution evidence, not commands independently repeated here.

Released attempt:
`docs/research/cad_engine_v1_synthetic_checks/context_v3_engine_integration_v2-13wnk1ap`.

Independently read its exact three regular files in full, parsed both administrative objects and the sole worker marker, verified path metadata stability through reads, and checked the three-name directory metadata inventory. No other attempt or cache was traversed. These fresh bindings match root's exact released final-file bindings:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `COMPLETED.json` | 19797 | `3138880c1da4166cc9874a3eeaae08c2a3b4a070440a37a8d7c998935a67dea6` |
| `OUTPUT.txt` | 12104 | `bf0e2c4a3fec586d74c2fae794d50ec1220dc67aae2abf8b4f391f116c4107a7` |
| `STARTED.json` | 9539 | `7a20648f0f5cf9e2b56320f9be2ee99beda9e743044f89696b049c49dfa7af04` |

Total 41440 bytes. Canonical compact sorted JSON filename-to-{bytes,sha256} inventory SHA256:
`427f9f224dee7124908a5d13cbd3ca3592ee48edce0e80fc1f4427f77d872e17`.

OUTPUT contains exactly one worker receipt; its parsed object equals COMPLETED.worker. Supervisor OUTPUT SHA256 equals the measured output hash. COMPLETED remains status FAIL and outcome_gate=false, with no postflight errors. Final publication success is root's actual stdout observation, supported by the independently matching exclusive three-file inventory; it is not inferred from a PASS label inside a file.

## Observed outcomes and semantic scope

| Disposition | Actual count |
| --- | ---: |
| Collected unique outcomes | 94 |
| Ordinary passing calls | 93 |
| Fully passing setup/call/teardown triplets | 93 |
| Failed calls | 1 |
| Errors, skips, xfail, xpass, deselections, collection errors | 0 each |
| Invalid/duplicate phase report flag | false |

Pytest reports `1 failed, 93 passed in 4.05s`. The 94-character source-order progress stream contains 93 dots and one F, at one-based node 89. Independently reconstructed all 94 source-order node IDs from literal parametrization and matched their digest to the worker:
`449a51bca56379a613533e2e418260beab5d37d08e0d90e51fb484ca98a25fb7`.

Sole failed node:
`tests/test_cad_guided_context_v3_engine_integration_v2.py::test_actual_workflow_v2_schedule_request_retains_mandatory_context_v3`.

At test line 716, actual answer status is UNAVAILABLE, expected PARTIAL; exposed phase is WHOLE_PART_REGISTRATION. The fixture's preceding assertion of a COMPLETE empty V2 schedule succeeded. Pytest elides the answer's full dictionary, including its reason list. The retained output therefore does not directly expose the internal exception or a traceback from the caught workflow branch.

All other 93 outcomes passed, including the new valid-policy/configuration guard, all four exact reached-once budget-selector rejections, both real small initializer/scheduler/context/finite-prefilter chains, actual helper tampering/whole-refusal controls, V1-schedule byte workflow, all selected-context/tampered-publication/index/late-failure controls, and context/schedule marker distinction. The 73 earlier policy-blocked outcomes are not retroactively rescored. This is a separate terminal V2 attempt.

The two real small-chain passes establish their frozen invented expectations: 50 distinct scan rows, 64 scan triangles, independently triangulated 80 CAD triangles, two ordered matches, eight signed families, 64 slots, three extrema indexes with 50/25/25 supports, complete finite-support accounting and the independently stipulated recovered start tolerance. They do not establish registration refinement, native CAD or local fitting. Byte-workflow boundary mocks remain mocks. The failing V2-schedule byte workflow is not qualified by these other passes. The old dense outcome was not selected or rerun.

## Source-only failure localization

A concrete fixture defect is present in the failing test, independently of any hidden runtime traceback:

1. The normal byte fixture's initialization omits `patch_evidence` entirely. The inherited legacy scan binder expressly allows this omission for a limited test/report record.
2. The V2-schedule test adds `patch_evidence=dict(cad=dict(patches=[]))` at lines 712–714 to supply the empty CAD equation table needed by the real V2 schedule binder. It does not add a `scan` entry.
3. V14 `_registration_binding` first validates the requested context publication and V2 schedule, then unconditionally calls the inherited `_scan_plane_binding` when no prepared scan binding was supplied (V14 lines 69–73).
4. Default V7 reaches V12 `_scan_plane_binding` with expected_configuration=None. Since `patch_evidence` now exists, V12 lines 80–81 immediately evaluate `initialization["patch_evidence"]["scan"]`. The new partial container has no such key. This source path necessarily raises KeyError if reached; omission of the whole optional container and presence of an incomplete container are not equivalent contracts.
5. V14 catches a selected binding exception, clears usable bindings/context indices, and returns UNAVAILABLE through its outer workflow exception path. That behavior is consistent with the observed phase and status.

The preceding schedule-specific premises were checked against the actual source, not executed: both empty canonical equation lists hash identically on the scheduler and binder sides; both expected configurations contain the exact default V2 caps; retained triplet counts are 3/3 with max joint 8192/1024; ordered capacity is min((8192-3)//32,1024-3)=255, both-order bundle capacity 127, odd remainder 1, zero pair reservations and retained joint counts 3/3. Empty bank/patch lists, no matches, usable_schedule=false and COMPLETE breadth/final ledgers satisfy that declared empty-scheduler path. Its zero-match schedule is already observed COMPLETE before the failed workflow assertion.

This is a definite source-level incomplete-fixture blocker and the strongest localized explanation consistent with the retained runtime evidence. The exact runtime KeyError/reason was not retained, so this POST does not mislabel that exception as directly observed. It does not prove that supplying the missing scan envelope would make every subsequent assertion pass.

The actual production preparation path is different: V13 delegates to frozen preparation; V8 preparation writes the scan metadata envelope at line 118 and CAD patches at line 142. No evidence here demonstrates a production default omitting the scan envelope. Do not weaken the production binder to accommodate this partial stub. A future separately authorized test revision should supply the coherent declared V7 scan envelope when adding patch_evidence, retain the real scan/schedule/context binders and all 94 cases, and assert the intended path/failure reason explicitly. That is a proposal, not a repair or rerun authorization.

This reviewer checked constructor/call prerequisites in PRE but missed this conditional container-presence requirement. The frozen PRE is preserved; this POST records that additional fixture premise failure rather than claiming the earlier source review guaranteed runtime success.

## Resources, lifecycle and limitation correction

| Quantity | Actual | Frozen limit/scope |
| --- | ---: | --- |
| Worker CPU seconds | 4.304897 | 45 |
| Worker high-water RSS bytes | 105775104 | Below 1073741824; HWM is not sampled group RSS |
| Sampled group peak RSS bytes | 104726528 | 1073741824 |
| Observed output bytes | 12104 | 1048576 |
| Supervisor wall seconds | 4.36203383281827 | 60, work limit 55 |
| Parent prepublication seconds | 4.371502125170082 | Outside supervisor clock |
| Parent through final inventory | About 4.371895, root stdout | Outside supervisor clock |

Receipt identifies one observed worker PID 20599, child exit 1, no budget stop, primary error or cleanup error; process_group_confirmed_empty=true and requires_group_attention=false. Phase times: work stopped 4.34946074988693; reaped 4.349464708007872; group checked 4.361898375209421; terminal decision 4.36203383281827 seconds. Together with root's external no-worker observation, these support quiescent terminal publication for this actual attempt. They do not make numeric PID/group custody atomic or generation-aware. The private cache path present in the receipt was treated as opaque narrative only, never opened, statted or traversed.

Important correction to the broad protocol/PRE wording: the NEW harness parent does guard its output parsing/postflight on process_group_confirmed_empty. However, the inherited short supervisor itself closes its capture handle and unconditionally computes `hashlib.sha256(output_path.read_bytes())` at V12 supervisor line 140, including when empty_confirmed is false. Therefore “no output reads unless all writers are quiescent” is not a whole-supervisor guarantee. The parent guard is narrower. This actual attempt has empty=true before that hash, so no unquiescent read was observed here; the latent broader guarantee is unsupported. Frozen source/PRE remain unchanged. No hard synchronous-I/O bound, hostile-filesystem sandbox, hard OS allocation limit or generation-aware supervision claim is made.

## Archive, live source, PRE and runtime custody

Independently read the accepted archive without extraction to disk or execution:
`docs/research/cad_engine_v1_synthetic_checks/context_v3_engine_integration_v2_frozen.9lqru16d/SOURCE.tar`.

Archive: 1330176 bytes, SHA256
`f42793c5320165253c5adb60db4ea7989fbf52259ec54b52c7b94f357237e699`.
Exactly 56 unique regular payloads, 1285856 total payload bytes: all 55 declared source files plus accepted PRE. No extra member, duplicate, link or unexpected path admitted. Every payload size/hash matches its accepted/live binding.

Fresh live source map equals STARTED, parent postflight and worker postflight: 55 files, 1262654 bytes; canonical path-to-SHA256 map
`e5ab90d6477d254251c9c32c83e90669e5e1aaabfb45c46a75c16da87a655098`.
Full path/size/hash table remains in the unchanged accepted PRE; it is not replaced with a new scope here. Accepted PRE bytes/hash/declaration match both postflights and archive.

Fresh application-Python -I -S -B metadata collection, without target/package import, reproduces runtime digest
`3065f28fced0b459c575c783852e8fb3a96d7935a0322ade43e9d16ff0d90473`
and exact STARTED/worker/parent objects. Python 3.11.15 canonical executable/stat, NumPy 2.4.6, SciPy 1.17.1 and pytest 9.1.1 initializer bytes/hashes are unchanged. This is ordinary loader entrypoint metadata, not complete transitive binary attestation. Final forbidden-loaded ledger is empty. No environmental/source drift or resource stop explains this failing assertion.

## Terminal handoff

Preserve V2 as FAIL 93/94, with verified publication and the incomplete scan-envelope fixture blocker above. Preserve the original FAIL21/94 and separately passed dense evidence independently. All source, test, harness, protocol, PRE, archive and attempt bytes remain frozen; only this NEW POST was written. A reasoned new fixture version plus independent source/semantic audit must precede any new qualification. No physical-input access, scoring, production promotion or rerun follows from this POST.
