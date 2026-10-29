# Pure contact-scorer V2 — independent exact POST

## Decision

**GREEN for the frozen pure source gate: 123 passed, zero other outcomes.**
This qualifies only the invented JSON/contact-carrier projection and scoring
contract in the [exact PRE](2026-09-10_gold11_contact_scoring_v2_source_pre_run_audit.md).
It does not qualify connected engine output, native result bytes, physical
accuracy, manufacturing truth, a physical campaign or floor release. The prior
connected negative remains negative.

This review examined the single identified terminal attempt, the eight exact
frozen live sources, PRE and qualified source archive. Main reports this was the
first and only gate execution, after its complete PRE read and archive checks.
The reviewer performed only bounded, no-follow stable file reads, hashing,
strict receipt parsing and in-memory archive-member comparison: no project
imports, collection, tests, rerun, extraction, private-cache access, raw/physical
data or other attempt access. Only this new POST was written.

## Terminal evidence and custody

Attempt, relative to `/Users/idantal/ShermanQC/FullProject`:

`docs/research/cad_engine_v1_synthetic_checks/gold11_contact_v2_source-pzmra0af`

The completion receipt was read first and confirmed its writer group empty
before output or directory inventory inspection. The exact nonrecursive
inventory contains only the following three regular, non-symlink files; all
were read in full within the authorized 64 KiB receipt / 1 MiB output caps.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `STARTED.json` | 1,176 | `78dbc4c4e48aa58717ca6de0cf7b34b3c4e50572b3573c293b65183e7954b95c` |
| `OUTPUT.txt` | 222 | `689c6e040f13d2b00a65bcd98841f8c047c6ca0449aa51c3624b29f0e9a844ec` |
| `COMPLETED.json` | 2,168 | `e7cbd8139e911b7d9879cd95f4c660e604707e0f109367a749e5268013bf8c64` |

The retained output contains exactly 123 passing progress marks, the summary
`123 passed in 1.39s`, and one CPU/RSS footer. There are zero failed, errored,
skipped, xfailed, xpassed, deselected or other outcomes. This independently
matches the PRE's 29-function / 123-outcome denominator; qualification does not
rely on exit zero alone. Observed output length and SHA exactly match the
completion receipt; no output truncation or overflow is indicated.

The worker exited zero. The recorded observed process-group PID list is exactly
`[36865]`; `process_group_confirmed_empty` is true and
`requires_group_attention` false. Budget stop, primary error and cleanup error
are all null. The ordered stop, reap, group-check and terminal-decision times
are consistent and within the declared deadlines. No scientific artifacts are
expected or present in this three-file namespace. The separately recorded
private cache was neither inspected nor changed.

| Quantity | Observed | Frozen bound |
| --- | ---: | ---: |
| Pytest elapsed | 1.39 s | Diagnostic only |
| Worker CPU | 1.4096140000000001 s | 45 s |
| Work stopped | 1.5220774170011282 s | 55 s |
| Monitored parent wall | 1.5389901669695973 s | 60 s |
| Worker RSS high-water footer | 95,354,880 bytes | Reported observation |
| Sampled process-group peak RSS | 95,354,880 bytes | 1,073,741,824 bytes |
| Observed / retained stdout | 222 bytes | 1,048,576 bytes |

Both receipts agree on the fixed limits. RSS supervision remains sampled, not
a proof of all transient memory peaks; the worker high-water value is a separate
reported observation. Source hashing and final receipt I/O remain outside the
monitored clock. No resource or real-time claim is broadened by this pass.

## Exact source and archive checks

The eight live sources match every byte count and SHA in the PRE, totaling
144,208 bytes. The exact eight-key STARTED before-map and COMPLETED after-map
both equal that ledger and the current live hashes; the reported
`source_before_after_consistent` true is independently confirmed. No source
or denominator amendment followed the run.

The unchanged PRE is 8,557 bytes, SHA-256
`df8b7d78f13de2f22601317423c02581c699154313d077787c05538899595828`.

The qualified archive is:

`docs/research/cad_engine_v1_synthetic_checks/gold11_contact_v2_source_frozen.JhKHhW/SOURCE.tar`

It is 169,984 bytes, SHA-256
`ad49dd373e2e364ae4e01c400fe29ddd00624d8648d2bc6868996e8dfdbbb7b4`.
An in-memory tar read confirmed exactly nine uniquely named regular members:
the eight PRE-listed sources plus that unchanged PRE. Member content totals
152,765 bytes and is byte-for-byte identical to the corresponding live files;
there are no extra, missing, duplicate, link or sparse members. Nothing was
extracted or executed. The full unchanged source ledger remains in the PRE
and is not duplicated here.

Main separately reports a preserved **archive-construction failure**, the
earlier `FOfHwj/SOURCE.tar` with SHA-256
`5e4446b28e5cbf63f8f64a34bb353e285be4945242b2fa73888f69a6ab513ac4`,
which contained nine extra AppleDouble metadata entries and was rejected before
execution. It was not used for this gate. The replacement used
`COPYFILE_DISABLE`; this POST independently verifies the qualified replacement
above, not the rejected archive. That reported construction failure is not a
second test execution or a failed scientific outcome. Neither archive was
deleted or overwritten in this review.

## Supported result and limits

The unchanged tests now provide positive source-level evidence for strict
whole-case malformed handling, typed quantity and axis/generator serialization,
literal V2 contact projection, explicit refusal/no-carrier dispositions,
preserved reference denominators and differential agreement with the unchanged
matcher and metric aggregation. All favorable examples are invented fixtures;
their 11 cases, 6 families and 71 references are not a physical Gold11 result.

The supplied-byte SHA boundary does not attest original execution, physical
identity or filesystem custody, and compact revalidation does not authenticate
the original workflow. No actual geometry fit, raw support, registration,
measurement accuracy, manufacturing-event count or conformance was tested.
The small-fixture pass also does not qualify maximum-size JSON allocation.

This closes the pure source gate only. Any byte-connected runner, native or
physical evaluation needs its own reviewed source, custody, fixed resources,
PRE and POST. No successor execution or promotion is authorized by this POST.
