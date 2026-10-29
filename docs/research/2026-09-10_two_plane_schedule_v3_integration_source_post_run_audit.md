# Scheduler integration V3 — independent exact POST

Date: 2026-09-10. Disposition: **GREEN for the frozen invented integration gate
only**. The single V3 attempt completed 69 PASS and zero other dispositions in
6.56 seconds. Exact output, source, archive, resource and writer-custody checks
pass. The prior V2 result remains terminal RED, unchanged.

This independent audit read only the three explicitly released terminal attempt
files, the 53 candidate sources, the two referenced PRE documents and the exact
V3 source archive. No project import, collection, test, geometry/native operation
or rerun occurred. No private cache, physical input or other attempt was inspected.
The only file written is this new POST.

## Exact attempt and denominator

Attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v3_integration-4j0mupcm`.
Parent reports session 18931 terminal with exit 0; the independently decoded
COMPLETED receipt also records exit 0. The immediate inventory contains exactly
these three regular, non-symlink files, and no scientific artifact or extra entry:

| File | Bytes | SHA256 |
|---|---:|---|
| `STARTED.json` | 6353 | `0280031deaffefa2a707f80d88bf94a94d951b87461d21e2bc22944c83cc5867` |
| `OUTPUT.txt` | 158 | `217c9101d90d136c1f950aa08035dfcb0fca388dbb61686421f59caf211bd71e` |
| `COMPLETED.json` | 7365 | `c413d977f0cd741c162c05a9cc7d34a142680bf9bd1202e3eb3f934bc8889822` |

All three files were read completely with bounded no-follow/regular-file and
stable-identity checks: 64 KiB per receipt and 1 MiB for output. Receipt JSON
decoding rejected duplicate keys and non-finite constants. The full 158-byte
output comprises 69 progress dots, `69 passed in 6.56s`, and the worker resource
JSON line. There is no traceback, error, failure, skip, xfail, xpass or other
reported disposition. STARTED's expected denominator is 69; the actual complete
summary independently satisfies it. The output hash and size agree exactly with
the COMPLETED receipt, not just the parent's report or exit status.

## Resources and writer custody

| Observation | Actual | Frozen bound |
|---|---:|---:|
| Pytest-reported duration | 6.56 s | Informational |
| Worker CPU | 6.639034 s | 45 s |
| Parent wall time | 6.73109016707167 s | 60 s |
| Work stopped | 6.713502542115748 s | 55 s |
| Worker high-water RSS | 135266304 B | Informational alongside sampled bound |
| Sampled process-group peak RSS | 135266304 B | 1073741824 B |
| Complete output | 158 B | 1048576 B |

The terminal worker line reports the CPU and high-water RSS shown above.
The receipt records 38674 as the only observed group PID, reaped at
6.713505499996245 seconds and checked empty at 6.731016791891307 seconds.
`process_group_confirmed_empty` is true and `requires_group_attention` is false.
Budget-stop, primary-error and cleanup-error fields are null; no source-after
error is present. Limits match the frozen 45 CPU/55 work/60 parent seconds,
1 GiB sampled group RSS and 1 MiB output. The receipt-named private cache remains
retained and uninspected. Sampling and cleanup are engineering observations,
not hard real-time guarantees; source/receipt I/O remains outside the clock.

## Exact source and archive verification

The V3 PRE is
`docs/research/2026-09-10_two_plane_schedule_v3_integration_source_pre_run_audit.md`,
9,839 bytes, SHA256
`f780ec01720bdec8bd7bda4ddc905b12f44be9174f9ef721bd4826c1f465f267`.
Its inherited ledger reference was resolved against the exact V2 PRE,
15,441 bytes, SHA256
`cf2325d81829c56f1c66acd13d59da671fb8d703d6ef139d8fdb16072e0c1594`:
remove the three explicitly named old harness/test/protocol rows and add the
four exact V3 delta rows. This yields 49 inherited plus four added sources,
not a directory scan or floating revision.

All 53 current regular source payloads were independently read with 2 MiB
per-file bounds and stable no-follow checks. Every byte count and SHA256 matches
the defined PRE ledger and both complete STARTED/COMPLETED source maps.
The total is 1,226,880 bytes; the exact unique path set equals the new harness's
literal 53-source tuple. The completion consistency flag is therefore independently
corroborated. The predecessor RED POST remains one of those frozen payloads:
8,005 bytes, SHA256
`9ef59b01a46a66a940a52aa2dfcdea390248051c0bb7b416c87c6371aa34c0ad`.

Archive:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v3_integration_frozen.js9c0q05/SOURCE.tar`,
1,333,248 bytes, SHA256
`d27ededa434cad6bd1857e91107d16370d54ef5e847fff58aa66f7d205415142`.
Bounded, in-memory inspection verified exactly 54 unique named regular payloads:
all 53 candidate sources plus this exact V3 PRE, totaling 1,236,719 payload bytes.
Every payload equals the corresponding current/retained bytes; there is no extra,
duplicate, symlink, hard link or AppleDouble member. Nothing was extracted.

The frozen 24-byte `tests/__init__.py` matches its inherited ledger and archive.
A repeated `lstat` check confirms no object, including dangling symlink, at the
enclosing repository `FullProject/__init__.py`. This verifies the reviewed package
boundary, not interpreter/library binary identity or any historically omitted
initializer in an older gate.

## Qualification and next boundary

The exact fixture-corrected V3 candidate satisfies all 69 frozen integration
oracles, including the 44 byte-workflow outcomes that V2 never reached. This is
evidence for the tested default/opt-in configuration, equation/joint binding and
whole-prefix refusal behavior in the declared tiny-array and mocked-byte paths.
The actual 50-row finite-prefilter checks and the 72-raw/36-selected-row mocked
byte checks keep their distinct scopes. The added declared V7 envelope header
does not establish complete real V7 metadata or plane preparation.

V2's 25 PASS/44 setup ERROR record remains RED; V3 does not rewrite it or establish
broader historical equivalence. No engine physics, measurement threshold or
scoring rule was changed by this fixture correction. There is no new physical
score, accepted global pose, observed bend count, calibrated uncertainty or
factory release. The connected 2/4 negative remains unchanged.

After root reads this POST fully, it may serve as the prerequisite for a new,
separately frozen connected protocol/PRE. That later gate still requires all four
original measurement successes and its own custody/resource audit. Native-byte
and Gold11 qualification remain later gates. This successful terminal source
attempt must not be rerun.
