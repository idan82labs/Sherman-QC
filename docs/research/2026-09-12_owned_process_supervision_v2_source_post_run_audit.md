# Owned-process supervision V2 — independent exact mock POST

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/process_v2_qualification`, independent of the harness author and operator.
This is an independent technical reconciliation of the single released mock
attempt, not human approval or installed-host/private-API qualification.

## Decision

GREEN / PASS for the frozen invented-only source gate: 127 collected outcomes,
127 passing calls and 127 complete passing setup/call/teardown triplets; zero
failure, error, skip, xfail, xpass, deselection or collection error. The sole
worker exited zero, resource/cleanup fields are clear, and retained evidence,
accepted archive, live source, exact PRE and runtime metadata reconcile.

The accepted result is mock-only. Actual V2 libproc/API behavior, generation-
coupled signaling on this host and owned-child lifecycle remain unqualified.
This does not authorize a native-host run or physical benchmark, integrate V2,
reopen terminal V5, or establish a new F1.

## Scope and evidence provenance

The released attempt is
`docs/research/cad_engine_v1_synthetic_checks/owned_process_supervision_v2-j36u6own`.
I fully read its three named files, independently checked its exact namespace
and raw hashes, read the accepted source archive without filesystem extraction,
compared all eight archive payloads with the seven live files plus PRE, and
recollected only the authorized ordinary runtime metadata with isolated stdlib
Python. No target or pytest module was imported; no tests, subprocess fixtures,
native API operations, process observations or signals were rerun.

Root reports that the sole invocation was tool `0f9b5f`, terminal exit zero,
with final stdout outcome/publication gates true. Root's separate process-only
check, tool `481fb3`, confirmed group 16618 absent and no Sherman Python writer.
Those are operator-supplied execution observations, not new process observations
made by this reviewer. The retained supervisor independently records successful
reap, empty group and no group attention. Final namespace/hash verification was
performed independently here after that quiescence release.

No private cache, old attempt, rejected-archive content, raw data, Gold, held-out,
email or scientific artifact was read. The private-cache path appears only as a
receipt field and was not followed. This POST is the only file written in this
review. No source, PRE, protocol, attempt or archive was modified.

## Exact retained inventory

The attempt is a directory containing exactly these three regular files, with
no extra member. Each raw byte count and hash independently matches root's
release and the appropriate retained bindings.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 2601 | `15eb481de09ca75b4ab277909f8d9eb0bba6a1359664d64ff35caed89c1aa0af` |
| `OUTPUT.txt` | 2767 | `4f9ec0f7e22f413ec51f28d780f7e19059b53b38b05885031f0e0e818a4a4cf7` |
| `COMPLETED.json` | 6046 | `16784624e9fa6a4efdbcd35f943c9d57414fceed38a128bb4ba38ea8132e8c14` |

`OUTPUT.txt` has one unique worker JSON marker. Its complete parsed ledger equals
the worker object in `COMPLETED.json`. The output hash equals the supervisor's
recorded output SHA256, and observed output bytes equal its actual 2767 bytes.
`COMPLETED.json` retains the truthful intermediate status
`PASS_PENDING_FINAL_PUBLICATION_AND_INDEPENDENT_POST`; it is not rewritten by
this later audit. The independent exact namespace and hashes satisfy the
publication check rather than interpreting that intermediate status alone as
final success.

## Denominator and resource reconciliation

The frozen test source was independently AST-counted in PRE as 51 functions /
127 prospective outcomes. The actual retained accounting now reports:

| Field | Actual |
| --- | ---: |
| Collected / passed / complete passing phase triplets | 127 / 127 / 127 |
| Failed / errors / skipped | 0 / 0 / 0 |
| Xfailed / xpassed / deselected / collection errors | 0 / 0 / 0 / 0 |
| Invalid report | false |
| Worker pytest exit code / enclosing worker exit code | 0 / 0 |
| Worker CPU seconds | 0.559117 |
| Worker high-water RSS bytes | 56164352 |
| Sampled group peak RSS bytes | 56164352 |
| Supervisor total wall seconds | 0.6167160416953266 |
| Observed output bytes | 2767 |

Pytest's retained text says `127 passed in 0.38s`. The node-ID list digest is
`35d86b17bda95b314f2eefb38c84eab6e88582c8df789a644805f447071fedd8`.
The retained ledger is aggregate phase accounting plus that digest, not a raw
per-phase event transcript. Its interpretation relies on the exact reviewed
hook, which rejects duplicate/non-unique collection, undeclared test files,
missing or duplicate phases, xfail markers, and nonpassing phase triplets. No
missing disposition has been inferred from pytest's summary line alone.

The actual limit fields match the frozen 20 CPU / 25 work / 30 total supervision
seconds, 536870912-byte sampled group RSS and 1048576-byte output caps. The
observed values are below their corresponding limits. Supervisor budget_stop,
primary_error and cleanup_error are null. Work stopped at 0.6022197920829058s,
reap completed at 0.6022224170155823s and the group check at
0.6166162500157952s. Observed group PIDs are exactly [16618],
process_group_confirmed_empty is true and requires_group_attention is false.

Parent prepublication wall is 0.6216648747213185s. Root reports parent time
through final inventory as 0.6219972908s; that final-stdout measurement is not a
field in the three retained files. Parent pre/postflight, archive and publication
I/O are outside the supervisor clock by the frozen protocol. These observations
are sampled memory and elapsed-time evidence, not hard OS allocation limits or
synchronous-I/O cancellation guarantees.

## Source, PRE, archive and runtime custody

The seven source bindings in STARTED, worker postflight, parent postflight and
current live files are byte-for-byte identical. Their total is 139618 bytes;
canonical path-to-SHA256 map digest remains
`cbd13011199d0b3469665d9af810e24811156dd6da75c2b9c7bcc7bcf1fc9950`.
The exact paths and individual hashes are bound by the accepted PRE:
`docs/research/2026-09-12_owned_process_supervision_v2_source_pre_run_audit.md`,
11563 bytes, SHA256
`0c9344f8181671353c791795512003f271d493d9370a80a51ec8e83958e852e7`.
Its live bytes, STARTED declaration, worker postflight and parent postflight all
match. Its unique declaration is GREEN, INVENTED_ONLY, expected_outcomes 127
with the same exact source and runtime digests.

The accepted archive is
`docs/research/cad_engine_v1_synthetic_checks/owned_process_supervision_v2_frozen.ZcR0fg/SOURCE.tar`,
157696 bytes, SHA256
`80a421556c91265d49da99cfe40ab4d871fa29413561c271d21364871ca2a4b8`.
Independent archive inspection found exactly eight unique, expected regular
payloads totaling 151181 bytes: the seven source files and exact PRE. Every
payload size/hash matches the live counterpart and PRE/STARTED bindings. No
archive member was extracted to the filesystem, and no extra payload was read.

Root also reports an earlier rejected packaging archive with seven unexpected
AppleDouble members. It is not the accepted archive, was not executed, and was
not inspected by this POST. A packaging rejection is not a second mock-suite
invocation; this audit does not convert it into accepted frozen custody.

The independently recollected runtime object equals STARTED and both retained
postflight objects. Its canonical digest remains
`2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689`.
It binds CPython 3.11.15 on darwin/arm64, the same canonical application
executable and five stat fields, pytest 9.1.1, and the 5644-byte pytest initializer
SHA256 `ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d`.
The root initializer remains absent. The worker reports no forbidden loaded
modules; its reviewed checks bind actual pytest path, version and initializer.
Parent postflight_errors is empty. This ordinary-entrypoint metadata is not a
complete transitive runtime binary or bytecode closure.

## Meaning and next gate

The single executed gate supports the 127 invented generation, ownership,
receipt, public-loop and adapter contracts at these exact bytes. The public
controls execute real V2 orchestration/custody against fakes; adapter tests use
fake calls and do not load libproc. The stale-token test still demonstrates
fresh-token refresh across direct calls, not a complete public exec-retry cycle.
The enclosing V12 supervisor's successful short mock-worker cleanup does not
qualify V2 native signaling or remove its own bounded lifecycle assumptions.

No retry is needed or released. Root may accept this mock POST and continue
separately gated installed-host qualification design and integration review.
Any actual native qualification requires its own exact source/protocol/PRE and
owned-child limits. Sampling remains distinct from continuous containment,
generation coupling remains distinct from atomic group membership at signal
time, and native/I/O calls are not hard-cancellable Python deadlines.

Terminal V5 remains failed and unscored. No physical input was consumed here,
no real geometry was evaluated, and no accuracy or F1 claim follows.
