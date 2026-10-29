# Independent POST: composite scan-plane source gate terminal RED

Date: 2026-09-10. Reviewer: `/root/cad_blocker_closure_review`.

## Decision

**RED / terminal NEGATIVE.** The first and only attempt
`composite_plane_v1_source-tiicptej` completed with **167 passed, 11 failed,
zero other outcomes**, worker exit 1. The frozen decision required 178 passed
and zero other outcomes. There is no retry or promotion from this attempt.

The failure is a test invocation construction defect, not evidence that the
intended public workflow contract passed or failed. All 11 affected checks
stop at the same missing-required-arguments error before calling that workflow.
The successful subsets remain evidence for their narrower invented-source
contracts; they cannot replace the missing 11 checks.

Source preservation, archive equality, resource limits and retained inventory
are clean. They do not turn this failed acceptance gate GREEN. No connected
four-curved successor, native bytes, physical/Gold11 run, default routing or
floor promotion is greenlit by this result.

## Exact attempt and independent checks

The [exact PRE](2026-09-10_composite_plane_v1_source_pre_run_audit.md) is
20,584 bytes, SHA256
`0c283a89b50c8f5bf2bf481809b2e9b613d54d9e6b26ba5d9e89a3377b35f22c`.

Retained attempt:
`docs/research/cad_engine_v1_synthetic_checks/composite_plane_v1_source-tiicptej`.

I read all three JSON receipts and the complete 521-line `OUTPUT.txt`, then
independently checked bounded regular-file bytes and hashes, the exact immediate
attempt inventory, empty scientific directory, all current candidate files and
all approved archive members. `COMPLETED.json` confirms an empty worker group
before the inventory/archive audit. No cache directory or unrelated attempts
were inspected.

All 62 current files equal the PRE's exact size/SHA ledger, totaling 1,589,350
bytes. Both `STARTED.source_sha256_before` and
`COMPLETED.source_sha256_after` equal that complete ledger, not merely each
other. The PRE itself is unchanged.

Preservation archive:
`docs/research/cad_engine_v1_synthetic_checks/composite_plane_v1_frozen_source.GTlDjf/SOURCE.tar`.

- Archive file: 1,723,904 bytes, SHA256
  `d6c4d347233ff57a879502a25495e509c40a03511d19d2bc3463a88fc6b1b000`.
- Exactly 63 unique regular members: the 62 candidate sources plus the PRE.
  No additional sidecars, links or directories.
- Every member was read in memory and equals its approved current/frozen source
  bytes. Total regular payload is 1,609,934 bytes.
- Nothing was extracted, executed or modified in either archive or attempt.

Immediate attempt inventory is exactly the four regular administrative files
below and an empty `ARTIFACTS` directory. The scientific inventory reports
`complete=true`, empty entries/dispositions/errors and zero bytes read.

| Administrative file | Bytes | SHA256 |
| --- | ---: | --- |
| `ARTIFACT_INVENTORY.json` | 93 | `30f576d2f0179e0e33e9e748f6874822fae389cbfe365f98dbdfbee51739d3a7` |
| `COMPLETED.json` | 8437 | `052764a2e04c7f3659f25e7b3cbf72cd16715723af9e51df1149534a8436ec6e` |
| `OUTPUT.txt` | 37912 | `281c4cf47611d509ae569873eb4c028cb24acbf6ac83dfd80329361ee4d108fa` |
| `STARTED.json` | 7483 | `f799ad6f94bd1420b17c2edd8bd945bfe55a32391773a27e9c1e6067c34d397d` |

## Denominator and actual failure boundary

The progress sequence is 124 pass markers, 11 fail markers, then 43 pass markers.
The complete summary says `11 failed, 167 passed in 9.36s`; its 11 named
failures and all tracebacks agree. There are no errors, skips, xfails, xpasses
or deselections. Mapping the frozen full-module order and literal parameter
counts to the failure names gives:

| Frozen module | Passed | Failed | Scope retained |
| --- | ---: | ---: | --- |
| Scan-plane preparation V1 | 41 | 0 | Composite production, work caps, snapshots and clearing |
| Composite surface roles V1 | 59 | 0 | All-lane binding, original unions and conditional selection |
| Composite integration V1 | 24 | 11 | Earlier initialization/search/role controls only; public adapter cases below unreached |
| Existing V13 adapter regressions | 43 | 0 | Their original explicit mock-boundary contracts |
| Total | 167 | 11 | Gate remains RED |

Each failure is a call-phase `TypeError` from the fixture-supplied `run()`
helper at `tests/test_cad_guided_composite_plane_integration_v1.py:442`:

```
NativeGuidedPolicy(fit_model="free_circle")
```

The unchanged dataclass in `backend/cad_guided_native_features.py:31` requires
five fields without defaults: `flange_window_mm`, `search_margin_mm`,
`max_fit_residual_mm`, `plane_exclusion_mm` and `max_tangency_error_mm`.
Only `fit_model` has a default. Python raises while constructing keyword
arguments; the public `inspect_cad_guided_mesh_v12/v13` call at the end of
the helper is not reached. These are not pytest fixture-setup errors and are
not native CAD failures.

The ten unreached workflow-refusal checks are the following five fault
alternatives, each for V12 and V13:

| Requested/returned binding fault | Unreached checks |
| --- | ---: |
| Missing returned configuration | 2 |
| Top-level budget mismatch | 2 |
| Initialization budget mismatch | 2 |
| Unrequested composite configuration | 2 |
| Hidden composite schema without configuration | 2 |

The eleventh failure is
`test_v13_prepares_composite_metadata_once_and_reuses_same_token_for_every_context`.
Its public workflow metadata-once, shared-token and deliberate measurement-refusal
assertions are also unreached.

The fixture had already completed actual composite preparation/role admission
and two direct `_registration_binding` positive controls before returning
its invocation helper. This is a narrower direct-helper observation, not a
substitute pass for any of the 11 named tests. The source PRE did not catch this
constructor-argument mismatch; the frozen execution exposes that review gap.

The 24 earlier integration passes do not establish a finite registration
positive: their real tiny search has zero starts, and the role-dispatch controls
insert an explicitly invented start followed by intentional refinement refusal.
The existing 43 adapter regressions use their own complete explicit policy and
mocks. They do not exercise the missing new public composite cases by proxy.
No physical gain or newly successful curved measurement is established.

## Resource and custody receipt

| Observation | Recorded value | Frozen limit |
| --- | ---: | ---: |
| Pytest reported elapsed | 9.36 s | Diagnostic, not the parent clock |
| Worker CPU | 9.432102 s | 45 s |
| Worker maximum RSS | 133,955,584 B | 1,073,741,824 B |
| Sampled process-group peak RSS | 133,939,200 B | 1,073,741,824 B |
| Work stopped | 9.534823749680072 s | 55 s |
| Worker reaped | 9.53482895810157 s | Within parent deadline |
| Empty group checked | 9.551354041788727 s | Within parent deadline |
| Parent terminal decision | 9.551512457896024 s | 60 s |
| Captured output | 37,912 B | 1,048,576 B |

Worker HWM and sampled group peak are distinct observations; their 16,384-byte
difference is not a source/inventory discrepancy. The complete worker footer
and monitored parent receipt are present. One worker PID was observed; no
unexpected descendant was recorded. `budget_stop`, `primary_error` and
`cleanup_error` are null; group empty is true, group attention false,
inventory complete true and before/after consistency true. Both recorded
limit dictionaries equal the PRE's frozen limits. No resource or custody
failure explains the test failures.

These remain sampled engineering bounds, not hard-real-time or hard-allocation
certification. No claim is made about unobserved process history beyond the
unchanged supervisor's stated custody mechanism.

## Narrow next step

Recommend a **separately named and frozen test-only successor**, preserving this
terminal V1 source, PRE, archive and receipts. Supply the existing complete
explicit measurement-policy values in the invocation fixture; do not add
backend defaults or relax backend validation to make the test constructor work.

A consistent source is the already frozen V13 test fixture at
`tests/test_cad_guided_mesh_offline_v13.py:239`: flange window (2.5, 12.5) mm,
search margin 3 mm, fit residual 0.05 mm, plane exclusion 0.075 mm and tangency
error 0.1 mm, with the same free-circle fit model. These are declared test
policies, not values inferred from this failure or physical calibration.

Before any successor execution, independently check the complete fixture call
signature, preserve all intended assertions and the original denominator,
freeze unchanged backend bytes and budgets, and obtain a new exact PRE/archive.
Do not silently overwrite/rerun V1, omit the 11 failed cases, loosen thresholds
or promote the 167 passing subset. This POST authorizes no new execution by
itself.

This review changed only this POST document. No project import, collection,
test rerun, native geometry, physical input, held-out, email, control or
authorization record access occurred.

