# V4 pure custody gate: independent retained POST

Date: 2026-09-10. Independent AI reviewer: `production_architecture_review`,
not the runner, tests, harness or protocol author.

Decision: **GREEN_VALID_FROZEN_PURE_CUSTODY_GATE**. Blocking findings: none for
this narrow22-outcome gate. This is not complete V4 runner qualification,
physical execution permission, engine improvement or human manufacturing approval.

## Exact terminal evidence

First-once attempt:
`docs/research/cad_engine_v1_synthetic_checks/gold11_v4_custody-q_9d0_vp`.
Root observed the parent command exit0. The retained completion separately records
child exit0. This reviewer did not execute or repeat the command.

| File | Bytes | Independently verified SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 950 | `2b7a33c18de783d11d9a2f482e5667cbfb9cd946fbcaa4420b219eacfa835706` |
| `OUTPUT.txt` | 110 | `d93526eba6a0a710b792e96c932dcb2f23a32937a022a0e3cadcca3dff6b7a1b` |
| `COMPLETED.json` | 1,924 | `8f0008819476c229bd521af720b805fcee3a7f3d8c6fa9429b7e258276d492d7` |

The attempt contains exactly these three regular nonsymlink files,2,984B total.
OUTPUT's actual length and digest match COMPLETED's observed-byte count and
digest. JSON receipts contain no duplicate keys. The separately retained private
cache is identified by COMPLETED; it was neither inspected nor cleaned by this
reviewer. No claim of independently inventoried cache contents is made.

## Exact source closure

The prior independent source PRE released one pure gate on these exact bytes.
Both receipt maps contain exactly this six-file closure, and independent current
source hashes match every entry. `source_before_after_consistent` is true.

| Source | SHA256 |
| --- | --- |
| `scripts/check_gold11_mesh_v4_custody.py` | `579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b` |
| `docs/research/2026-09-10_gold11_v4_custody_gate_protocol.md` | `ddff925c4b0b3e411463f08ad14e72cd3869bed9ffdd787ca47a61aa3b98852c` |
| `scripts/run_gold11_mesh_development_v4.py` | `559095a685d6ae5badecdb9704952d5e80304dcd728f8afa0f131a25707ebd85` |
| `tests/test_gold11_mesh_v4_terminal_custody.py` | `68defd348ceced0246f5aa8f0daa637ced09a1263365a8b54f6db6a53f26ad8d` |
| `scripts/gold11_terminal_artifacts_v4.py` | `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838` |
| `scripts/run_gold11_mesh_development_v3.py` | `77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7` |

## Outcomes, resources and group custody

OUTPUT contains exactly22 pass markers, `22 passed in 0.48s`, and one worker
resource JSON line. There are no skips, failures, xfails or extra outcomes.
This matches the source-reviewed14 functions/22 parameterized outcomes and
STARTED's explicit expectation. No test-count inference from exit0 alone is used.

| Observation | Value |
| --- | ---: |
| Parent supervised wall seconds | .6464532916434109 |
| Work stopped seconds | .6289271246641874 |
| Direct child reaped seconds | .628933500032872 |
| Group checked seconds | .6463854997418821 |
| Worker CPU seconds | .547206 |
| Worker high-water RSS bytes | 58,081,280 |
| Sampled group peak RSS bytes | 56,197,120 |
| Observed/stored output bytes | 110 /110 |

The phase times are ordered and below the fixed55s work/60s supervised wall
limits. CPU is below45s; both RSS observations are below1,073,741,824B; output
is below1,048,576B. STARTED's `work_seconds` and COMPLETED's `work_wall_seconds`
both declare55. All other numeric limits agree exactly.

COMPLETED reports the single observed group PID82435, confirmed-empty group,
`requires_group_attention=false`, and null budget stop, primary error and cleanup
error. The copied supervisor's reviewed reaping/group-check path supports those
receipts. This is retained termination evidence, not a new live process scan or
proof against arbitrary process-identity interference. Sampled RSS and worker
high-water RSS remain different observations, not continuous hard enforcement.

## Qualified scope and limits

The pass supports the tested AST-extracted custody adapter: global distinct-name
accounting and one overflow witness, mixed-directory inventory, deadline escape
through the qualified helper, no later terminal-fill IO after its timeout,
all22 failure dispositions, and changed-byte/inode rejection across publication
phases. Real helper/filesystem operations use tiny invented temporary files.

Signals, schema generation and neutral leaf validation remain explicit mocks.
The tests do not qualify actual timer delivery, every asynchronous interruption,
monotonic-only expiry, final-audit timeout paths, the real complete campaign
schema/leaf validators, whole-runner supervision or candidate dispatch. The
protocol's excluded setup/source-hash/cache-creation/receipt-publication timing
and non-hard-real-time IO limitations remain unchanged.

No engine, native, physical, Gold11 or full-runner execution is released by this
POST. V3's spent evidence remains untouched; no numerical score or output claim
changes. This gate is terminal and must not be repeated unchanged. Further
integration requires its own exact source/protocol review and release.

Review accessed only the named new attempt and six released current sources.
No cache, other attempt, original data, geometry, Gold or prediction was read;
no runtime import, test, inference or scorer was executed. Only this additive
POST audit was written.
