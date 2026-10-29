# Connected V12 source gate — independent POST, 2026-09-10

Decision: **GREEN_VALID_FROZEN_CONNECTED_V12_SOURCE_GATE**.
Reviewer: independent AI agent `/root/production_architecture_review`; not the
candidate, test or harness author. No human/manufacturing approval is implied.
Blocking findings: none within this source-gate scope.

## Exact attempt and receipts

First-once attempt:
`docs/research/cad_engine_v1_synthetic_checks/connected_v12_source-v3w2abt4`.
Root reports directly observing parent exit0 for the exact released command.
This reviewer independently inspected the retained bytes, not the process's
live execution, and did not rerun/import any target code.

The attempt has exactly four regular files and one empty real `ARTIFACTS`
directory, no other names. The four files total **13,512 bytes**:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 6,036 | `82e367c3a64ece00f7e77b9ae69eff37a18fc15e1805687b0e1a0a01dcbe7cf9` |
| `OUTPUT.txt` | 358 | `5e961fb4d026aae1a986dd1889125e2e23d464acdacd0a3554e8e9d866f122e0` |
| `ARTIFACT_INVENTORY.json` | 93 | `30f576d2f0179e0e33e9e748f6874822fae389cbfe365f98dbdfbee51739d3a7` |
| `COMPLETED.json` | 7,025 | `4c9135250bfcc87d50d4eb42040e7dfff90ee85bca10fc8b972f787f822b54b4` |

STARTED's exact49 hashes, COMPLETED's exact49 hashes and every current source
hash equal the complete ledger in
[source PRE](2026-09-10_connected_v12_source_pre_run_audit.md), raw SHA256
`72174dfb50f47ea432fc1df34aba1f58f6f7f93473372b88256809e3e7fe5272`.
The PRE bytes also still match that digest. File lengths/hashes were checked
one bounded receipt at a time; before/after file identity was stable.

## Outcomes and bounds

OUTPUT contains exactly256 pass markers and one skip marker, the summary
`256 passed, 1 skipped in 5.89s`, then the worker resource JSON. This agrees with
the source-reviewed 168 context +69 role +19 fixture outcomes; the sole frozen
skip is the full connected packet. The concise pytest log does not print
individual node IDs; its count is reconciled against the exact unchanged tests,
fixed three-file selection and sole opt-in skip, not an invented verbose log.

| Evidence | Observed | Frozen bound |
| --- | ---: | ---: |
| Worker CPU | 5.766323 s | 45 s |
| Worker high-water RSS | 132,661,248 B | 1,073,741,824 B |
| Sampled owned-group peak RSS | 132,661,248 B | 1,073,741,824 B |
| Parent supervised interval | 6.071503791026771 s | 60 s |
| Work stopped | 6.054326958023012 s | 55 s |
| Reaped | 6.0543307908810675 s | within total bound |
| Group-empty checked | 6.071423874702305 s | within total bound |
| Observed/stored stdout | 358 B | 1,048,576 B |

COMPLETED reports child exit0; budget_stop, primary_error and cleanup_error are
null; group-empty is true and attention is false. The only observed group PID
is83280. Limits/suite/artifact paths agree across receipts. The recorded output
SHA and observed byte count match the raw OUTPUT. Inventory is exactly complete,
empty entries/errors/dispositions, zero read bytes; direct bounded enumeration
confirms the empty artifact directory. No arm ran under the source gate.

The retained private-cache path is recorded, but this reviewer did not inspect
or stat that cache. CPU/high-water and sampled RSS are complementary evidence,
not a hard allocation guarantee. Source setup and final artifact/receipt I/O
remain outside the supervisor work timer as disclosed in the frozen protocol.

## Advancement and limits

This validates the one source gate, including small invented-array actual fits,
codec/role logic and mocked aggregation/publication controls. It does not prove
real extraction finds cap roles, global pose observability, native/byte support,
physical accuracy, known bend count or a Gold11 improvement. Original terminal
V11/V3 negatives remain unchanged.

The source prerequisite is satisfied for a **separate exact PRE** of the one
18-arm constructed-array packet. This POST alone does not release that command,
a native test, physical inputs or another run. Review activity was confined to
this attempt's four files/empty directory, the49 current source files and PRE;
no original inputs, other attempts, target imports or scientific reruns.
