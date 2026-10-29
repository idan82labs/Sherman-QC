# Diagnostic V1 — independent exact frozen physical PRE

Date: 2026-09-12. Reviewer: `process_v2_qualification`, independent AI technical
reviewer, not the candidate/test/protocol author or a human manufacturing
approver. Decision: GREEN FOR EXACT UNSCORED DIAGNOSTIC V1. No blocking finding
remains for the exact frozen bundle below. Root must still read/accept both PRE
artifacts, verify the external 66-payload archive and observe no overlapping
heavy writer before the sole launch. This audit is not a launch receipt.

## Exact controls and inventory

Attempt: `docs/research/gold11_context_v3_diagnostic_v1_attempt`.
Root reports the sole source-only freeze e4baf6 exited0. This reviewer read only
that released attempt's exact source/control tree, its 64 named live sources,
named accepted review evidence and ordinary runtime metadata. No physical input
path was opened, statted, resolved or hashed; no Gold/catalog, other attempt,
private cache, native/process query or signal was accessed. No target was
compiled, imported, collected or executed. Only PRE_RUN_AUDIT.json and this
external audit were written.

| Artifact / identity | Bytes or count | SHA256 |
| --- | ---: | --- |
| FROZEN.json exact file | 55246 | 75544f52112071bbe2d60854e7f96629dfb83ef96355ae383e72964d59e18a4b |
| FROZEN decoded canonical bundle | — | 78da586200e6ab4be809f22229c59531d922589fd0889d0074a0bc3f3b31b7aa |
| PRE_RUN_AUDIT.json exact file | 15687 | fef3951bebe846727203517f9e0c29900cee48ac79b0ee4c72e58a69be28ed71 |
| PRE decoded canonical object | — | 1307046fea9c844d8c3abb7cf7d917930d6f5b309166c0b313879554281e3cfc |
| Released source map | 64 files / 1697501 bytes | f540758ad5b6d3d3434c508ee80db05e8bb7593e8eb8eae1e8a3e8d008fb0c9e |
| Before PRE: name→{bytes,sha256} inventory | 65 files / 1752747 bytes | 946fa004a649c8a1315773824dbfab17530de76513c26d5bf16fb6b956577f29 |
| After PRE: name→{bytes,sha256} inventory | 66 files / 1768434 bytes | 2959565f1447fe8f0cdac90207de9a4547fdc3984a937ec0ed88219a8dd7c07e |

Canonical objects use UTF-8 JSON, sorted keys, compact separators,
`allow_nan=False`, no trailing newline. Canonical control hashes are not
substitutes for the separately listed exact file-byte hashes. The two inventory
maps use paths relative to the attempt, including the `source/` prefix.

All64 live/frozen source pairs are byte-identical. Bounded no-follow reads found
only single-hard-link regular files, with stable descriptor/path fingerprints
and directory names/identities. Six directories include the root. After PRE,
root names are exactly `FROZEN.json`, `PRE_RUN_AUDIT.json`, `source`; no STARTED
or extra artifact exists. FROZEN and PRE retain all64 individual source hashes.

## Independent expected-bundle comparison

Application Python3.11.15 `-I -S -B` parsed ASTs and independently interpreted
only inert constants/containers/dict constructors/named references/arithmetic.
No target function or compiled AST ran. Independently assembled arm requests,
profile, source-derived scope and fresh runtime give canonical equality with
the complete frozen bundle, including its exact field set.

| Exact field binding | Canonical SHA256 |
| --- | --- |
| source_profile | bb86039eb34078aed2cbd2890738bc82b8498a7c045cde5b4a4b63838a96c1a5 |
| CONFIG | 9411366c98a7664a9efdea3f1cd4368ea0459350dab553b9e9792c88e0275575 |
| certificate_caps | 1af7fa4a1b522fcb00c83a1bf754422e7752ebe98f4e90085e21413f106cd7e9 |
| both arm_requests | 54cbece489ca6631357d212c2cd4a0b2eae3d762970c1340f2dbde7584aa490b |
| control arm_request | 668084c33aca57fa6b637c4d7897a96aea5ae6e2d7853ea376d5b5d78aa15777 |
| challenger arm_request | 9fbfd4ac5e2fa65e5a48f84c7e0f0328593b910d086a3a7fe03e3898feb2e73d |
| limits | 91e84707e41e2ae9c27933f40f4c0246bbc26cebe34454da4827cf76e19a1cd3 |
| no_score_policy | 0d0706a1bb7963cb961b3ae08c01f43987e26d42f96c307d8f02798905b9d6d8 |
| runtime_metadata | 79a084f7e8a3681abeafaad5d4369e04c5734dc674df11ce205475e60af25907 |
| original raw17 path→SHA map | 2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157 |
| ordered selected three-case array | 19048aa4ac59338406ca69d688da9a16dd98352a17ac05442612531ca02b8842 |

Both V14/contextV3/nativeV1 arms, CONFIG/caps/default maps, 55 functions/two
classes/21 constants/seven callbacks and six-slot no-score profile match source.

Runtime was inspected through stdlib executable/platform and distribution
metadata only, without importing the packages:

- Python: `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]`.
- Canonical executable: `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`.
- Platform/machine: darwin/arm64.
- numpy 2.4.6; scipy 1.17.1; cadquery-ocp 7.9.3.1.1.

Exactly one metadata distribution per named package was found; this is version
metadata, not binary attestation or native initialization.

## Exact source-declared physical scope

Authority is solely `scripts/review_gold11_inputs_v1.py`, SHA256
`41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
Its literal CASES gives11 scans/six CADs/17 distinct path/hash declarations.
`reference_id` is stripped, with no reference bytes supplied. The selected
array and raw17 map exactly match the independent input-scope review and
FROZEN. No raw file was consulted to construct or corroborate this table.

| Case order / token / part | Kind | Exact source-declared absolute path | Original declared SHA256 |
| --- | --- | --- | --- |
| 1 / CG08506AE97D / 10839000 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply` | cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b |
| 1 / CG08506AE97D / 10839000 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp` | ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37 |
| 2 / CG4BDD1371B6 / 37361005 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply` | d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038 |
| 2 / CG4BDD1371B6 / 37361005 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.stp` | 81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e |
| 3 / CG4FAF273093 / 49024000 | SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply` | 70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade |
| 3 / CG4FAF273093 / 49024000 | CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp` | 3cdfcba8fbea3735e3c9b3610b69ccab2bc8bf681919ef2c86839693ef54479f |

These are six distinct path declarations and six distinct hashes, not verified
physical bytes in this PRE. Slot order is CG08506AE97D/control,
CG08506AE97D/challenger, CG4BDD1371B6/control, CG4BDD1371B6/challenger,
CG4FAF273093/control, CG4FAF273093/challenger. Each arm uses its case's same
complete pair. No other raw17 file may be read merely to corroborate authority.
No substitution, favorable prefix, reference selection, score or subset F1 is
permitted. The actual released invocation must verify each original input hash
under its existing caps; unestablished input binding aborts later dispatch.

## Frozen limits and outcome rule

| Limit | Exact value |
| --- | ---: |
| case_cpu_seconds | 1860 |
| case_log_bytes | 1048576 |
| case_result_bytes | 134217728 |
| case_wall_seconds | 3540 |
| compact_case_bytes | 8388608 |
| coordinator_log_bytes | 2097152 |
| coordinator_rss_bytes | 4294967296 |
| global_tree_rss_bytes | 12884901888 |
| max_cad_bytes | 8388608 |
| max_scan_bytes | 67108864 |
| parent_read_seconds | 180 |
| sampled_worker_tree_rss_bytes | 8589934592 |
| terminalization_reserve_seconds | 2400 |
| total_wall_seconds | 25200 |

6×3540 +2400 reserve +1560 overhead =25200 seconds. The separate at-most-one
one-second emergency veto permits no engine/input/result/inventory extension.
Sampled RSS, HWM and synchronous I/O retain their declared limitations.
Unknown-custody caches remain opaque.

All six slots remain: handled engine refusals/timeouts are negative evidence,
not an all-positive prerequisite. Runner/custody/publication failures veto
acceptance. No score/promotion follows; all11_release_allowed remains false.

## Accepted qualification, PRE fields and next action

PRE.evidence retains the exact paths/bytes/hashes of the accepted source review,
input-scope review, release-only audit and46 POST; all were hash-verified unchanged.
The invented gate's limitations remain in PRE; no suite was rerun. Released
source is57936 bytes/SHA256
`47cdd2c716ca82f65c8994a095826a212a9f0c485b62306dc24d9a4355bc50a2`;
the exact two-literal reversal is established in the accepted release audit.

The published PRE passes the source-defined exact field checks: decision
`GREEN_FOR_EXACT_UNSCORED_DIAGNOSTIC_V1`; exact canonical bundle/profile hashes;
selected_case_order as above; terminal_slots is integer6, not Boolean; exact
limits/no_score_policy; nonblank independent AI reviewer; blocking_findings=[]
and complete expected-bundle equality. Added custody/evidence fields do not
broaden those conditions. The post-publication check again found all64 live
sources and named evidence unchanged, and exactly66 admitted regular payloads.

Root must now fully read/accept both new PRE artifacts and retain/verify an
external archive of exactly those66 payloads (1768434 bytes before archive
headers), then observe no overlapping heavy writer. No archive hash or current
process observation is invented in this audit; both are pending root actions.
Provided nothing bound changes, the sole command from FullProject is:

```sh
.venv/bin/python -I -S -B -u scripts/run_gold11_context_v3_diagnostic_v1.py --execute
```

No source edit, cap increase, retry, replacement slot or alternative invocation
is authorized by this PRE. Actual terminal stdout/exit, external quiescence and
independent exact six-slot POST must establish completion after the run.
