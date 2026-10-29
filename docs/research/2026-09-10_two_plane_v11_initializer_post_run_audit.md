# V11 focused initializer: retained terminal-negative POST audit

Date: 2026-09-10. Independent AI reviewer: `production_architecture_review`,
not the author of this initializer, focused test, protocol or harness.

Decision: **VALID_RETAINED_TERMINAL_NEGATIVE_QUALIFICATION_FAILED**.
Evidence custody is complete in the scoped three-file attempt. The required
positive gate failed; advancement to the workflow/native gates is **BLOCKED**.
This is not a passing safety sentinel, physical output result, or human approval.

## Exact attempt and release binding

First-once attempt:
`docs/research/cad_engine_v1_synthetic_checks/plane_initialization_v11-eyeny0mn`.
Root reports one execution and observed parent exit1 from the closed session9100.
The independent retained evidence separately records child exit1.

The source-only PRE is
`docs/research/2026-09-10_two_plane_v11_integration_pre_run_audit.md`, SHA256
`cb232da25c94f2d4490a387811cece0b59da50e43ab495775bb0ae181f88fbe4`.
The exact protocol remains
`2748fed1bc8b7de8fc17bc93e3bb8b4c7628c0ec136fe12a93038fcf4d3f674b`.

Independently verified all three files are regular, nonsymlink entries and that
the owned attempt contains exactly these three entries, with no extra artifacts:

| Retained file | Bytes | SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 5,660 | `1029e287a6af09766836b201ca57c78752e373ebc2a2f1b000e808b0ae932e4e` |
| `OUTPUT.txt` | 1,381 | `d6aac0d77271d9bd5711ad874a2b5489ccc062d00e7cb43d0a522e3faace9e68` |
| `COMPLETED.json` | 5,835 | `5aac937186d98b8daac0ccdcd39542176c92707458caea5fdefba7aae2e24fe6` |

Total retained bytes:12,876. The COMPLETED output hash matches the actual OUTPUT
bytes. Suite identity, harness hash, test hash and attempt path agree. Every one
of the 46 before/after source hashes equals the exact PRE table and the current
source bytes; no drift is present. In particular:

- Harness: `5675caa55fac23507c4d818cffb311af44020d0f71a0940acc30927e26545552`.
- Initializer: `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e`.
- Initializer test: `16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e`.

## Outcome and earliest demonstrated failure

OUTPUT has 40 pass markers and one failure marker, with the terminal pytest
summary **1 failed, 40 passed in 1.44s**. All 41 expected outcomes ran; no skips
or xfails are reported. The sole failing outcome is
`test_actual_fifty_row_two_plane_point_zero_five_lane_required_positive[True]`.

The traceback reaches test line139 after the frozen complete-prefilter assertion
helper returns. The next required assertion, nonempty `answer["start_groups"]`,
fails because the list is empty; the reported reason is
`NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS`. Thus the noisy .05mm, 50-row,
two-rectangle positive did not produce a required usable start. The clean
parameterization and remaining 39 outcomes passed. Noisy-positive action-error
and subsequent detailed support assertions were not reached for the failed case.

This authenticates a completed-prefilter/no-start failure, not a timeout or
numerical-incompleteness stop. The captured traceback does **not** retain the
full returned context, per-origin poses or support/query ledger. Consequently
this POST cannot determine whether the decisive cause was candidate generation,
roundoff/offset behavior, finite support, boundary/ambiguity classification or
another downstream prefilter condition. It does not infer those values from
the earlier physical V3 failure or rerun the failed case to fill the gap.

The test remains an invented broad-rectangle initializer control, not the
original thin-part native/free-circle diagnostic. Its failure is nevertheless
a failure of a predeclared positive requirement and cannot be counted as a
successful abstention, waived because other tests passed, or promoted into a
physical result.

## Resource and terminal checks

Both receipts retain exactly the predeclared limits:60s monitored wall,
45s worker CPU,1,073,741,824B sampled group RSS,1,048,576B output.

| Observation | Retained value |
| --- | ---: |
| Parent monitored wall seconds | 1.557764417026192 |
| Worker CPU seconds | 1.496111 |
| Worker high-water RSS bytes | 97,697,792 |
| Sampled group peak RSS bytes | 93,388,800 |
| Output bytes | 1,381 |
| Child exit code | 1 |

`budget_stop` is null and `source_before_after_consistent` is true. All reported
resource observations are below their frozen caps. The high-water and sampled
RSS measures are distinct; sampling is not a hard allocation limit. Existing
monitoring exclusions for setup, hashing, terminal I/O and cleanup remain as
disclosed in the PRE/protocol. No resource pass overrides the positive failure.

## Custody and next boundary

This review read only the exact retained attempt, its released PRE and the 46
named current source files. It parsed/hashed evidence without importing runtime,
collecting/running tests, executing inference, accessing original raw/CAD/Gold/
cached data, or changing the frozen sources. Only this additive POST was written.

Preserve the failed attempt and all 46 source bytes. Do not rerun this gate,
weaken its geometry/error/coverage assertions, enlarge its started budget, or
execute the workflow/native successors under this PRE. Root reports those routes
remain unexecuted. Further diagnosis requires a separately authored and frozen
bounded diagnostic with independently reviewed retention and explicit scope;
this POST releases no new execution. The previously qualified pure helpers and
the recovered cached UNION benchmark remain unchanged.
