# Full11 V7 — independent exact frozen physical PRE

Date: 2026-09-12. Reviewer: `process_v2_qualification`, independent AI technical
reviewer, not a human manufacturing approver or the runner/protocol/release
author. Decision: **GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN** for the exact
bundle below. Blocking findings: none. Root must still accept both PRE artifacts,
verify an external archive and observe fresh no-overlap before the sole command.
This document is not a launch receipt or score.

## Exact freeze and PRE

Root reports the sole source-only freeze36104c exited0. This reviewer inspected
only its released source/FROZEN tree, the named70 live sources, accepted narrative
evidence and ordinary runtime metadata. No raw/Gold/reference path, private
cache, other physical attempt or source-gate attempt was inspected. No target
was compiled/imported/executed, test run, archive created or launch attempted.
Only the new PRE_RUN_AUDIT.json and this external audit were written.

Attempt: `docs/research/gold11_mesh_development_v7_attempt`.

| Exact identity | Bytes/count | SHA-256 |
|---|---:|---|
| FROZEN.json file | 36954 | `230593dc3a5d6dd10216fcd728e9881f732d07065885a9fcc68b177216882182` |
| FROZEN decoded canonical bundle | — | `65c69cf82063358476dd7a6a3ef406b190ab8492baeafe1583d10c7f5f7efc49` |
| PRE_RUN_AUDIT.json file | 31934 | `7712fe7bc1ddcabfe0b7b16f6ff582658283e5dff9a7687fcb0c21eab1eb96bf` |
| PRE decoded canonical object | — | `d47c20dd48ef21b713671a7768f20f4c2d26b78ae70ea9e67bcddbf11d46a5a7` |
| Live/frozen source map | 70 /2284566 | `e50c182e5db3139e9df54e67b10ce0ee70ba804b803d84eca91befeac24fac31` |
| Before PRE, complete payload map | 71 /2321520 | `26e8c0eb2a3b2d88fd9f4721be2eed082d771a6e2006e605f9a8429d02e7a9d7` |
| After PRE, complete payload map | 72 /2353454 | `e6f2c300cb2ca99b124bc2b7826b3ff2cf4d0ac535e3eed11f9eacb936f13c42` |

Canonical objects/maps use sorted compact JSON, default ensure_ascii,
allow_nan=False and no newline. Exact file hashes include publication framing.
Payload maps are attempt-relative path→{bytes,sha256}, including `source/`.

All70 live/frozen pairs are byte-identical. The exact source-derived hierarchy
has seven directories including root. Bounded no-follow reads verified regular
single-link payloads and stable device/inode/mode/link-count/size/mtime/ctime;
atime is deliberately excluded. All declared directory names/types/identities
match, with no unknown or symlink branch. Before PRE, root held exactly
FROZEN.json/source; after PRE it holds exactly FROZEN.json/PRE_RUN_AUDIT.json/source.
There is no STARTED or other terminal name. All named Python source parses as
AST data; none was compiled or imported by this review.

Released runner is169180 bytes,
`e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52`;
protocol is13957 bytes,
`32c84d3dcc12a3f25024fafebeb65eb242c098eeb4d4a50548570ccfbc67319a`.
Only these two members changed from the previously reviewed70-source map;
replacing their hashes reproduces prior `beb7b07ccd93dff8b84e08529f8499f92c02fb02e15f87423f5cb15b11ac8b23`.
The separate accepted release audit connects the actual21 tested unreleased
identity to this released-declaration identity; the old21 ledger is not relabelled.

## Independent bundle/control reconciliation

Source ASTs were interpreted only as inert constants/containers/arithmetic and
the explicit source-declared arm/bundle shapes. That independently constructed
complete bundle has canonical hash65c69cf8 above, exactly the actual FROZEN
object, including its field set. The source70 map, CASES literal, reference hash
constants and fresh runtime metadata were separately compared with FROZEN.

| Frozen component | Canonical SHA-256 |
|---|---|
| SOURCE_PROFILE | `c9a8aa6b78181a0c560a29a89e59bf674c3f8ee555a5b35b74b9248ae31b1811` |
| CONFIG | `9411366c98a7664a9efdea3f1cd4368ea0459350dab553b9e9792c88e0275575` |
| Certificate caps | `1af7fa4a1b522fcb00c83a1bf754422e7752ebe98f4e90085e21413f106cd7e9` |
| Both arm requests | `54cbece489ca6631357d212c2cd4a0b2eae3d762970c1340f2dbde7584aa490b` |
| Limits | `8fd437267181e3df7a4a2b516649889ecb71d16c98fbf70adcb703a17f436865` |
| Scoring transport | `0e2f41383d1d6d03af3b9663a7ff3261fa4f3ac3f7f96f650856b510d790775f` |
| Complete ordered11-case array | `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e` |
| Runtime metadata | `79a084f7e8a3681abeafaad5d4369e04c5734dc674df11ce205475e60af25907` |

The preserved V6 basis is166049 bytes/SHA
`7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80`.
Both fresh nativeV1/V14/contextV3 arms are fixed: control scan-planeV7/scheduleV1;
challenger composite_v1/scheduleV2. CONFIG, certificate caps and full explicit
default maps are unchanged. PRE binds all70 individual source hashes and exact
projection/contact/matcher dependencies. No case identity/reference input enters
the inference request; the coordinator retains source-declared reference_id only
for the later post-terminal reference lookup.

Fresh runtime metadata equals FROZEN: CPython3.11.15, darwin/arm64; canonical
executable `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`;
numpy2.4.6/scipy1.17.1/cadquery-ocp7.9.3.1.1. Exactly one ordinary distribution
metadata record per package was found without importing those packages.
Version metadata is not binary attestation or native initialization.

| Control | Actual or prospective bytes including newline | Existing cap |
|---|---:|---:|
| FROZEN | actual36954 | 262144 |
| PRE | actual31934 | 262144 |
| STARTED | source-exact prospective944 | 4096 |
| COORDINATOR_DISPATCH | dominating prospective277 | 4096 |
| COORDINATOR_VERIFIED | dominating prospective8663 | 32768 |

The latter three are not yet artifacts. They use the actual frozen source/runtime
maps and fixed64-character hash representations, with20 decimal characters
allocated to generated PID/stat scalars. Every observed frozen/source stat
scalar is at most19 characters. Fixed70 source paths are ASCII, maximum67 bytes.
Source explicitly uses `tempfile.mkdtemp(..., dir="/private/tmp")`; the inspected
application Python3.11 stdlib supplies eight ASCII suffix characters. Under the
declared trusted canonical Darwin namespace, the coordinator path is43 ASCII
bytes. No cache was created or probed by PRE; namespace integrity and future
scalar widths remain the stated assumptions, not universal foreign-path claims.

Accepted conditional finalization accounting remains unchanged: owned-chain
request192964188 bytes (<192MiB), successful parent terminal259386 including
newline (<256KiB, only2758 bytes headroom), actual-schema enriched report12285070
including newline (<32MiB). Exact names/runtime/control representations match
those constructions. Prefix event caps16 and nested ENTERED/RESOLVED accounting
remain unchanged. Clean successful terminal shape can contain negative engine
outcomes; it excludes unbounded foreign fields, unresolved/error histories and
arbitrary exception shapes. This is not an all-failure terminal-fit, decoded
memory, continuous RSS or hard synchronous-I/O guarantee. Actual encoded-length
vetoes remain mandatory, with no budget increase.

## Exact source-declared input and reference scope

Authority is only the literal CASES in `scripts/review_gold11_inputs_v1.py`,
SHA `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
Its complete array equals FROZEN:11 distinct scan paths, six CAD paths and17
distinct path/hash bindings. The raw17 map digest is
`2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157`.
These are source declarations, NOT raw bytes verified in PRE. No referenced
path was opened, statted, resolved or hashed. The sole future invocation must
verify complete original buffers under its fixed caps; no mutable manifest,
new email part, held-out, substitute or corroborating raw file is authorized.

| Kind | Exact original source-declared path | Original declared SHA-256 |
|---|---|---|
| SCAN | `/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply` | `2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba` |
| SCAN | `/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_X.ply` | `4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply` | `71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp` | `ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_1.ply` | `08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000/10839000_01_scan_2.ply` | `cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.ply` | `d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.stp` | `81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply` | `07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_2 1.ply` | `f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp` | `4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0 1.stp` | `f9b8965c0a172756d6cf69110f3336b9ecaa0dcb14fe3848b9d3187a885d7f8f` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0_SCAN2 1.ply` | `152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply` | `7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.stp` | `d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5` |
| SCAN | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply` | `70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade` |
| CAD | `/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp` | `3cdfcba8fbea3735e3c9b3610b69ccab2bc8bf681919ef2c86839693ef54479f` |

The exact11 case order below has control then challenger for each, hence22 slots.
PRE retains the complete case→scan/CAD/reference_id mapping, not just these names.

1. `CG08506AE97D` (10839000); control, then challenger.
2. `CG191DEC7F81` (49001000); control, then challenger.
3. `CG1E0BBA2933` (38172000); control, then challenger.
4. `CG4BDD1371B6` (37361005); control, then challenger.
5. `CG4FAF273093` (49024000); control, then challenger.
6. `CG64C412CC73` (40500000); control, then challenger.
7. `CG65CD575B1C` (38172000); control, then challenger.
8. `CG84C8FA12D2` (10839000); control, then challenger.
9. `CGE4B0549E41` (10839000); control, then challenger.
10. `CGEBA9A869AC` (38172000); control, then challenger.
11. `CGF41F49066A` (38172000); control, then challenger.

Reference-container source declaration:
`4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`;
selected71-record snapshot:
`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
No Gold bytes were read by this PRE. Only the frozen post-terminal path may
verify/read them after all22 durable results/refusals have been reconstructed.
Scoring retains11 cases/six families/71 references, with unchanged one-to-one
minimum-cost matching (5 degrees,4mm,.60/.60 overlap), no transform/unit estimate,
and no favorable subset. Refused/malformed/empty cases retain their reference
FNs; unexpected matcher/custody errors veto. Null/undefined metrics stay explicit.

## Budgets, retained negatives and accepted evidence

Inference limits are3540 wall/1860 CPU seconds,8GiB sampled tree,1MiB log,
128MiB result and8MiB compact projection/receipt. Input buffers are64MiB scan and
8MiB CAD. Coordinator HWM limit4GiB; outer12GiB; coordinator log2MiB.
Scorer180 wall/120 CPU seconds,2GiB sampled tree,192MiB request and32MiB result/
enriched report. Parent bounded read180s; campaign104400s with2400s terminal
reserve inside it. 22×3540 +180 +2400 =80460, leaving23940s for other work.
Source preflight precedes the campaign clock. The disclosed extra at-most-one
one-second emergency veto grants no engine extension or late success.

One heavy job, serial leaves. An unestablished input binding aborts later
dispatch/scoring while preserving its actual reason. Unknown writers or failed
retention abort further work and leave private caches opaque. Retained partial
bytes are never successful inference. Terminal fill, source/control postflight,
reads and inventories require their exact quiescence predicates. Sampler timing
is observational; missing terminal telemetry remains unknown. No all-positive
geometry prerequisite or retry is introduced.

All five accepted narrative identities below were read/previously reviewed as
applicable and freshly hash-verified. Their underlying attempts were not reopened
during this PRE. They are bound separately, not inserted into the frozen70 map.

| Accepted evidence | Bytes | SHA-256 |
|---|---:|---|
| `2026-09-12_gold11_context_v3_diagnostic_v1_physical_post_run_audit.md` | 15827 | `079dc79286a9fe310569d0bb3c25495a540f3a367a2e2e7440a1fd698b49e8c7` |
| `2026-09-12_gold11_mesh_development_v7_delta_source_post_run_audit.md` | 9866 | `822fe3e1c4f6231e81d17324416f11f885e452a089f39844a6616d282fef95d3` |
| `2026-09-12_gold11_mesh_development_v7_finalization_accounting.md` | 13134 | `2455811a3e5a13f597239e8b72b7db52b77ce055973e9c4b482d6b24a7e10dd4` |
| `2026-09-12_gold11_mesh_development_v7_release_delta_audit.md` | 7941 | `f49c6b4cec15c5a1dbb2b53b032fd685babc748b64fe015a380cab55a88d6842` |
| `2026-09-12_gold11_mesh_development_v7_transport_accounting.md` | 8955 | `befda2c11ed146f68beba085d02ad96b1a17e05e7cb005fc1b8fdef8d3d0181b` |

The diagnostic POST accepts six trustworthy negative/limited outcomes (five
unavailable workflows, one CPU termination), not geometry success. The21 POST
qualifies only invented new runner boundaries. Source/release/accounting evidence
does not assert physical observed counts, metrological measurement accuracy,
historical F1 comparability or unseen-family generalization. The reference71
snapshot is unchanged, not reconstructed historical truth. Production release
and historical-score victory claims remain prohibited.

## Exact next action and stop conditions

The published PRE has the exact source-required decision, canonical bundle hash,
nonblank independent reviewer and empty blocking_findings; full frozen equality
and the stronger bindings above were independently checked. Its31934 bytes fit
the existing256KiB limit. Root must now fully read/accept both PREs, create and
verify an external archive of exactly72 regular payloads/2353454 bytes
(before archive headers), and freshly confirm no overlapping heavy writer.
Archive hash and no-overlap evidence are pending root actions, not invented here.

Only after those conditions, with all bound source/runtime/controls unchanged,
the sole command from FullProject is:

```sh
.venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v7.py --execute
```

Do not retry/resume, relax caps or replace a slot. Real outer exit/final stdout,
external quiescence and an independent exact all22/scorer/report/inventory POST
must precede any accepted score. This PRE permits only the exact conditional
consumed-development run under that boundary, not production/floor release or
completion of the locked physical count/measurement roadmap.
