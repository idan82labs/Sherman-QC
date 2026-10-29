# Two-plane scheduler V2: exact source POST

Date: 2026-09-10. Independent reviewer: `radius_math_review`.

Decision: **GREEN for the single isolated invented-equation source gate**.
The retained attempt completed **80 PASS, zero other dispositions**, with clean
terminal custody and the complete eight-file repository source closure unchanged.
This is not connected, native, physical, maximum-size performance or production
qualification. The attempt is terminal and must not be rerun.

## Exact attempt and denominator

Authorized attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v2_source-8uab1hd1`.

The reviewer read only the authorized three-file attempt, exact eight sources,
PRE, named archive and the named root-package-boundary absence check. Receipt
and source reads used bounded no-follow regular-file reads with stable identity
checks; JSON decoding rejected duplicate keys and nonfinite constants. Archive
payloads were compared in memory, without filesystem extraction. No tests,
project/third-party imports, collection, native execution, raw data, other
scientific artifacts or private cache were accessed. Only this new POST was
written; all execution evidence below comes from the retained attempt.

| Exact attempt file | Bytes | SHA-256 |
| --- | ---: | --- |
| `STARTED.json` | 1169 | `5587da014f9ae61aec5e7d1147d4675c435fa08ca6532e337a0eae57b59dd366` |
| `OUTPUT.txt` | 168 | `509ce4684365fd10c9de10f908fa7fdb3ea7ffe8c9973b47c9f1795a98c27a65` |
| `COMPLETED.json` | 2175 | `da7d834bb65b1a63d95fb036ae512e97f2974e9da09b079f1e66d4ba2181513d` |

These are exactly the three directory entries, all regular files; there are no
scientific artifacts or unexpected attempt entries. STARTED/COMPLETED are below
the 64-KiB audit limits and OUTPUT is below 1 MiB. The retained OUTPUT digest and
168-byte length equal the completion receipt's values.

OUTPUT contains exactly three lines: 80 passing progress dots, the summary
`80 passed in 0.96s`, and the worker CPU/HWM JSON footer. There are no failures,
errors, skips, xfails, xpasses, deselections, interruptions or additional output
dispositions. STARTED declares 80 expected outcomes. Independent static AST
recount of the unchanged test source confirms 29 functions expanding to 80
outcomes. Thus the actual denominator matches the frozen prospective denominator;
the conclusion does not rely only on exit zero.

## Resource and terminal custody evidence

| Evidence | Retained value | Frozen limit or scope |
| --- | ---: | --- |
| Worker CPU footer | 1.020349 s | 45 s worker CPU limit |
| Work stopped | 1.0751597499474883 s | 55 s monitored work |
| Parent terminal wall | 1.092457375023514 s | 60 s monitored parent |
| Sampled process-group peak RSS | 56737792 bytes | 1073741824 bytes |
| Worker HWM footer | 56737792 bytes | Independent worker diagnostic, not group-RSS attestation |
| Retained/observed output | 168 bytes | 1048576 bytes |

The receipt reports exit0, `budget_stop=null`, `primary_error=null` and
`cleanup_error=null`. The only observed group PID is37292;
`process_group_confirmed_empty=true` and `requires_group_attention=false`.
Reaping at1.075162542052567s and group confirmation at1.0923528340645134s follow
work stop and precede the terminal decision. All phase values are ordered and
within their frozen limits. STARTED and COMPLETED agree on CPU/RSS/output/wall
limits, with the work-clock field's documented naming difference.

The receipt's separately identified private cache remains untouched: its path
was not opened, inventoried, hashed or cleaned. Sampled RSS/deadline custody is
not a continuous instantaneous RSS bound or hard real-time OS/filesystem
guarantee. Source hashing and receipt I/O remain outside the monitored work
clock. The worker footer is present; no missing resource evidence was treated
as zero.

## Exact PRE, source and archive binding

PRE: `docs/research/2026-09-10_two_plane_schedule_v2_source_pre_run_audit.md`,
10801 bytes, SHA-256
`f60a7aaeba5f3c0f99ce346dc5f6929ef96d87cbf784db8d0e54a5addf575827`.

Archive:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v2_source_frozen.FrVPGs/SOURCE.tar`,
167936 bytes, SHA-256
`eb66f9157ac0a13dd14109951d2d5655a0c5c7d43f0ecc528c1d5b12f93ae2da`.

The archive has exactly nine unique named regular payloads: all eight files in
the PRE source ledger plus that exact PRE. There are no extra payloads, symbolic
links, hard links or alternate names. Their payload sizes total151404 bytes:
140603 source bytes plus10801 PRE bytes. Every payload byte equals its current
named source/PRE counterpart; verification did not extract files to disk.

For every one of the eight sources, the PRE hash/byte ledger, STARTED before
hashes, COMPLETED after hashes, current bounded-read hashes and archived payloads
agree exactly. `source_before_after_consistent=true` is independently corroborated.
The source set is exactly:

- `scripts/check_cad_guided_two_plane_schedule_v2.py`
- `scripts/check_gold11_mesh_v4_custody.py`
- `backend/cad_guided_plane_search_v6.py`
- `backend/cad_guided_two_plane_schedule_v1.py`
- `backend/cad_guided_two_plane_schedule_v2.py`
- `tests/test_cad_guided_two_plane_schedule_v2.py`
- `tests/__init__.py`
- `docs/research/2026-09-10_two_plane_schedule_v2_source_protocol.md`

The pytest package initializer is explicitly frozen, archived and unchanged; its
24 bytes remain the comment audited in PRE. The named
`FullProject/__init__.py` path is absent at this independent POST check, including
no dangling symlink. Root also reported checking that same absence immediately
before launch. This closes the identified repository package-boundary dependency
for this attempt; it is not an attestation of arbitrary transient host mutation,
Python/pytest/NumPy binaries, compiled libraries or the operating system.

## Qualified claim and retained limits

The 80 passing checks support the exact candidate's tested small-bank contracts:
complete V1 breadth equality; bounded omitted-target residual allocation; both
target orders and reservation accounting; declared-margin priority and known
starvation limits; malformed-input/cap handling; cache identity; whole-result
clearing; and explicit downstream nonmonotonicity. The test fixture remains
bounded to six scan and eight CAD equations, with at most256 ordered matches.

No test supplies real scan support, finite face extents, curvature or a physical
correspondence oracle. This result does not establish maximum128/2048-bank or
128-MiB-layout runtime/RSS, calibrated normal cones, recovered registration,
translation observability, all-four connected recovery, native/Gold11
improvement, historical superiority, observed bend count or production use.
Any integration or broader experiment requires its own prospective source,
fixture, custody and evaluation gate; this successful source attempt remains
immutable and terminal.
