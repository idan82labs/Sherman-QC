# Exact POST — native depth diagnostic

Date: 2026-09-11. Independent reviewer: `cad_blocker_closure_review`.

## Decision

**RED_NATIVE_QUALIFICATION_AND_DIAGNOSTIC_INTEGRITY.** The sole opportunity is terminal: **99 passed / 1 failed**, with no other reported pytest disposition. The original 39-outcome module contributes 38 pure/mock passes and the failed actual native case; the supplemental module contributes 61 pure/mock passes. Those 61 passes do not establish the actual native ledger's integrity.

The actual workflow returned a fully retained, completed `UNAVAILABLE` result with zero predictions and reason `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`. Its original oracle fails before the later byte/context/support assertions, and its supplemental ledger fails record preflight before seven scientific criterion groups are reached. Therefore neither the original positive-geometry gate nor the prospective complete-integrity diagnostic gate is satisfied. **This POST does not release Gold11, a native retry, promotion, or an F1 claim.**

Artifact/source custody and observed resource accounting pass their narrower checks below. A separately versioned, bounded retained-evidence validator is a defensible next diagnostic proposal; it must not retroactively turn this attempt or its unexecuted assertions GREEN.

## Scope and immutable evidence

The only terminal attempt inspected is:

`docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_v1_native-d4dw1v2p`

Root reported session 1807 closed at exit 1 and independently confirmed process group 89392 empty before authorizing these reads. This reviewer read the complete four administrative files and four scientific files, the full exact PRE, all 111 named frozen source payloads, and the exact source archive. Reads used bounded regular-file admission with no-follow/nonblocking opens and stable descriptor/path checks; source archive members were checked in memory without extraction. JSON was parsed with duplicate-key/nonfinite rejection. No project import, test/collection, native library, inference, refit, physical input, private cache, or other attempt was accessed or executed. Only this new POST is written.

The attempt root has exactly the four admin names below plus `ARTIFACTS/`; that directory has exactly the four science names below, with no partials or extra scientific files. The retained cache path in the administrative receipt was not inspected.

| Exact file | Bytes | SHA-256 |
| --- | ---: | --- |
| STARTED.json | 14125 | f908c624b324dcddcf0c46b181f1a55ae43faca5441f15fe7edf22e4d9d9f10e |
| COMPLETED.json | 14948 | f5a8629505ff4e4735eabb95b25321c208e06812c7bec8c071c1edad05ab7796 |
| ARTIFACT_INVENTORY.json | 574 | e566c9d984f93a888a45f37011bf7e5686121722db9022e1ad0136a78cc02646 |
| OUTPUT.txt | 7140 | 0868bfc465282e5549370682916aeb696264cbccdf5c2167f4016abaf810fbd1 |
| ARTIFACTS/CASE_START.json | 1000 | ba7d2a4e3c05ba8e47b4b65e9bf44e55bd13487fdce80e2e1ba55acc34524a55 |
| ARTIFACTS/CAD.step | 30521 | 63e7ec51b132ed4bf514e3b753f204e84abbeed84ef5025abf9799d3d2b70ad6 |
| ARTIFACTS/SCAN.ply | 31131 | 05ae6538c3b0c7262241193dc75fe4ccf8c70cf937911e6809c63bfce92efac1 |
| ARTIFACTS/CASE_RESULT.json | 27980423 | 4b6705fdfea4b8c3dd0b307c1b5e14677be6419b60a695828a55fc385b7d59ca |

All scientific size/hash pairs agree with the independently read inventory, sole OUTPUT case receipt, and result/start metadata where applicable. Science totals **28,043,075 B**, within the frozen **144,707,584 B** aggregate; each individual cap is also satisfied. The inventory reports complete, no errors, and that same total read count.

### Source and archive closure

The [exact PRE](2026-09-11_two_plane_depth_native_diagnostic_pre_run_audit.md) is **33,948 B**, SHA-256 `2735475c77c0a45da7546174693a07994ab44870da6d0200c2612d4da5e1690a`. Its full 111-file declaration is inherited here by exact identity, not retyped or inferred from a nominal count. The declaration was mechanically resolved and its complete current mapping independently compared with STARTED before, COMPLETED after, and every corresponding archived payload: **111/111 identical; no mismatch**.

There are **84 Python / 27 Markdown / 2,396,049 source payload bytes**. The sorted compact path-to-SHA map is **13,395 B**, SHA-256 `5f07c4166b4382fdf4bfe855ee1b6ea76c39803b063b2680cd366cbbd697d022`. The bound `tests/__init__.py` remains included; repository-root `__init__.py` remains absent, including the dangling-symlink boundary.

The exact archive is:

`docs/research/cad_engine_v1_synthetic_checks/two_plane_depth_native_diagnostic_frozen.z7jpsdy8/SOURCE.tar`

It is **2,519,040 B**, SHA-256 `1a54bff59bc9fd19f86896cc40246e293d55276181a118e0602dc5ed105fce26`. It contains exactly **112 regular payloads / 2,429,997 B**: all 111 sources plus the PRE. There are no duplicate names, extras, links, AppleDouble entries, or path-traversal members. All source and PRE bytes match live exact identities. Root's prelaunch full PRE read and archive verification are separately reported launch custody, not actions retrospectively supplied by this POST.

## Terminal denominator and resources

The complete OUTPUT contains exactly one native case receipt, one final worker CPU/RSS footer, the original module's 38 success markers before the sole native failure, the supplemental module's 61 success markers afterward, and the summary `1 failed, 99 passed in 519.03s (0:08:39)`. The only reported failure is `test_two_plane_depth_v1_single_native_byte_case`; there are no skipped, xfailed, xpassed, error, or other reported outcomes. This is a full-output/source-order reconciliation; the harness did not retain a separate per-test outcome observer.

| Quantity | Observed | Frozen cap |
| --- | ---: | ---: |
| Worker CPU, sole final footer | 518.23214 s | 2520 s |
| Work stopped | 519.2708464590833 s | 2815 s |
| Parent terminal decision | 519.2861869591288 s | 2820 s |
| Sampled process-group peak RSS | 858046464 B | 4294967296 B |
| Worker peak RSS footer | 858046464 B | 4294967296 B |
| Complete OUTPUT | 7140 B | 1048576 B |
| Scientific artifacts | 28043075 B | 144707584 B |

COMPLETED has exit 1, `budget_stop=null`, `primary_error=null`, `cleanup_error=null`, `process_group_confirmed_empty=true`, `requires_group_attention=false`, complete artifact inventory, equal source maps, and absent root initializer after execution. Its only observed group PID is 89392. Reaping is recorded at 519.2708485843614 s and group checking at 519.2860857923515 s, before terminal publication. These are sampled/boundary-enforced receipts, not a hard-real-time OS guarantee.

Within the native case, fixture construction reports 0.452248 CPU / 0.934305792 wall s; the sole workflow reports **515.781441 CPU / 516.177906042 wall s**. These measured timings are useful bounded planning evidence for this one invented case, not a physical-case completion guarantee. The ledger itself stopped in approximately 0.002948 CPU / 0.002954 wall s, so this attempt does not measure the full seven-group diagnostic validation cost.

## Independent retained byte checks and report disposition

The exact PLY header declares binary little-endian, **650 vertices with double x/y/z** and **1176 faces with uchar-count/int32 vertex indices**. A standard-library `struct` decoder verified the exact header and total payload length, finite XYZ within the declared bounds, exactly three distinct in-range indices per face, and no trailing bytes. It did not run a geometry algorithm.

The original XYZ payload is **15,600 B**, with canonical little-endian float64 SHA-256:

`994b3d8a4de940c6470393fdb60dc20d9fb368434c135bd33cbc496872cfce3c`

The face rows converted to canonical little-endian int64 triples are **28,224 B**, SHA-256:

`8c4f5c76ca41d6fd9d63a7523a4cace886392e307fe07b56140dd8b510c12de6`

Both agree with the retained recorder input binding, workflow preparation/topology/measurement bindings, and the supplemental ledger's before/after supplied-input digests. STEP bytes agree with every published CAD hash and have the expected ISO-10303-21 container markers. **No native STEP reparse, BRep equivalence check, recovered CAD tessellation, or sampler-array reconstruction was performed.**

CASE_RESULT retains the full workflow before oracle evaluation: `RETURNED_PENDING_ORACLE`, `full_returned_workflow_retained=true`, `source_consistency=true`, no execution error, and `oracle_disposition=NOT_RUN_RESULT_MUST_BE_DURABLE_FIRST`. This latter field is correct pre-oracle publication order, not the final native outcome. The sole OUTPUT receipt supplies the later failure.

The retained workflow has V13 schema, `UNAVAILABLE/COMPLETE`, `connected_measurement_contexts=true`, `conformance=INDETERMINATE`, unknown observed bend count, and production disabled. The 0.05 mm bound remains stipulated/un-calibrated and scoped to initialization/context surface policy, not measurement uncertainty. Its report contains:

- 48 registration candidates and 11 context indices: 33, 34, 35, 36, 40, 41, 42, 44, 45, 46, 47.
- Completed initialization/context/schedule reports, 8192 hypotheses and 252 scheduled pair matches.
- One native feature (`solid:0:face:5:window:0`) and reported CAD sampling count 32768. Native units report millimetres, one transferred root and explicit post-read/pre-transfer target-unit action; physical calibration remains not established.
- 11 context windows, 5 original support records, one unavailable support group, zero predictions, and complete source/cover computation with `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`.

These are independently counted/inspected retained report fields, **not a newly passed registration, pose, sampler, cover, or accuracy oracle**. The report also retains runtime cover-validation output: plan revalidated, 3910 cover-row checks, 754 graph links, 1452 graph-row visits, zero member comparisons, no repeated cover refit and zero publication geometry-row visits. Its existence must not be confused with an independently rerun validator or the original oracle having reached that call.

## Failure boundaries — no inferred successful prefix

### Original oracle

The receipt retains `oracle=null`, `positive_geometry_passed=false`, and an `AssertionError` whose message is the reasons list. OUTPUT's stack identifies only the outer native test assertion at line 858; the caught inner traceback was not retained.

The earliest source-consistent false predicate can nevertheless be identified from frozen source and exact result values:

1. [Native _evaluate](../../tests/test_cad_guided_two_plane_depth_native_v1.py#L291) first calls the byte/context binder.
2. That binder at lines 128–139 checks V13 schema, connected mode and position-error scope, then creates a historical view changing **only** the schema to V12.
3. The [legacy binder](../../tests/test_cad_guided_connected_v12_native_bytes.py#L152) accepts the projected V12 schema but immediately requires `status == "PARTIAL" and phase == "COMPLETE"` at line 154. The actual status is **UNAVAILABLE**, so this assertion is false and emits precisely the retained reasons list.

This is a source/result reconstruction, not a claim that an inner traceback was captured. The later nonempty-group assertion at native line 320 emits the same message and **must not be cited as reached**. The schema/connected/scope entry checks are consistent with the retained values; none constitutes a complete planned integrity group.

Consequently the subsequent legacy native/scan/sampler/context checks, composite/schedule binders, repeated cover validator, projection/window/nominal assertions, truth-based measurements and carrier endpoint checks are **not established as executed and are source-unreachable after the identified false predicate**. No planned eight-group integrity criterion is wholly proved by this original-oracle prefix. The external byte/source checks in this POST stand on their own.

### Supplemental actual-native ledger

The whole ledger is present inside the 7140-byte OUTPUT, comfortably below its separate 32 KiB envelope. It reports `NATIVE_DEPTH_DIAGNOSTIC_LEDGER_V1`, status FAIL, no accuracy/truth access, zero inference/refit calls, no fixed-request digest, no populated report denominators and no complete-no-measurements qualification.

| Planned criterion | Actual result | Predicate visits |
| --- | --- | ---: |
| publication_source_input | FAIL | 5 planned / 5 visited / 3 failed |
| raw_native_sampler | UNREACHED | 0 |
| context_origin_pose | UNREACHED | 0 |
| composite_schedule_profile | UNREACHED | 0 |
| raw_support_cover | UNREACHED | 0 |
| windows_nominals | UNREACHED | 0 |
| quantity_carrier_projection | UNREACHED | 0 |
| branch_denominator | UNREACHED | 0 |

The three failures are `bounded_record_before`, `bounded_record_after`, and `immutable_supplied_evidence`, each retained only as error type ValueError. The two supplied-input digest checks succeed and agree; both record digests remain null. The helper visited 20,384 cumulative preflight nodes before its two early failures. This is **failure to establish record immutability, not evidence that mutation occurred**. Durable record-object binding and the seven later groups were never reached; `file_identity_reverified=false` is also explicitly retained.

A concrete source contract mismatch supports the preflight diagnosis: [helper line 138](../../tests/native_depth_diagnostic_ledger_v1.py#L138) accepts only exact string dictionary keys, while [the CAD sampler](../../backend/cad_guided_cad_sampling_v1.py#L245) constructs `sample_count_by_source_face` and `area_by_source_face` with exact `int(f)` keys. [V13 publication](../../backend/cad_guided_mesh_offline_v13.py#L184) retains that non-array coverage dictionary. The durable JSON serializer accepts such integer keys and writes string object keys. Thus legitimate current engine output is incompatible with the stricter helper preflight. The receipt did not preserve the ValueError code or original key types, so it does not by itself identify the exact first failing path or recover original Python types. This source-supported mismatch warrants a successor design; it does not invalidate the preserved failure or qualify the remaining criteria.

## Smallest prospective recovery path

A **new retained-evidence contract and validator**, not a retry of this native opportunity, can potentially establish the diagnostic admission needed before the later Gold11 score without another inference invocation:

1. Pin this exact eight-file terminal inventory, original RESULT byte length/hash, PRE/archive and complete source map. Retain all old failed outcomes unchanged. Read the original result once with strict duplicate/nonfinite/schema/container/depth/node/byte admission and no status rewriting.
2. Bind original RESULT **bytes** separately from a canonical, decoded string-key working object. Hash that decoded object before and after validation under an explicitly named wire-format contract. Different order/digest is expected and must not be presented as the original in-memory object digest. Do not coerce keys back to integers, replace UNAVAILABLE with PARTIAL, discard a failing prefix, or invent fixture-cache histories.
3. Recover the exact 650 raw double XYZ / 1176 integer triangle rows from the immutable PLY as above. Reuse the unchanged seven scientific criterion predicates/all-record loops, composite/schedule checks and complete support-cover publication validator where their contracts accept the retained report. Supply fixed policies from fresh, independently source-qualified constructors; separate this new request binding from the old same-worker `_requested` cache-identity branch.
4. Freeze the source/dependency closure, all record denominators, read/work/resource limits, failure handling and explicit new assurance label before a new retained-only check. The review should preserve full traversal/no-successful-prefix semantics, post-validation immutability checks, no refit/native/loader calls, and separately count all actual criteria. Pure tamper/serialization tests and an exact PRE/POST are still required; this document does not supply their results.
5. Admit only a complete retained-applicable ledger with the expected whole completed refusal or conditional publication, while disclosing the original accuracy and same-worker ledger FAIL. Keep early limited refusal, missing records, unavailable predicates, or invalid publication blocking. A new prospective admission amendment is necessary because the frozen old criterion required same-worker evidence this attempt did not establish.

The defensible assurance is **retained bytes plus source/report binding**, not recovered historical Python types, module/cache identity, exact omitted native arrays, independently verified material topology, calibration, accepted/global-unique pose, manufacturing bend count, or measured accuracy. A successful retained successor could support an honest consumed-development diagnostic score with all known refusals retained; it could not repair the failed original native qualification, the connected 2/4 positive-measurement result, or the roadmap's promotion/physical-generalization gates.

No need is established to rerun expensive native inference solely for predicates recoverable from immutable report/source/PLY. Equally, those predicates are **not yet passed** merely because their source exists. Root must accept this exact RED POST and separately freeze/review any retained successor before its execution; V5's source and physical-release gates remain unstarted here.

