# Terminal-artifact V4: independent retained POST audit

Date: 2026-09-10. Decision: **POST-GREEN for the exact first invented-filesystem
gate below.** The complete retained result satisfies the unchanged PRE release:
114 passed, zero skips, exit 0, no recorded budget stop, and four stable source
bindings. This is not a runner, physical-input or engine-output qualification.

Reviewer: `cad_measurement_integration_review`, who authored neither the helper
nor its independently authored tests. Root executed the gate. This POST used
only the named retained attempt, the PRE and four source-file hashes. No test,
helper, native operation, inference or scoring was imported or rerun; no raw
scan, CAD, Gold or cached prediction was accessed.

## Exact release and attempt

PRE: `docs/research/2026-09-10_gold11_terminal_artifacts_v4_pre_run_audit.md`,
SHA256 `3a08e18db60b88536b8fb69599bbf71efb3aecd71bfd5098b2e16524da146138`.

Attempt:
`docs/research/cad_engine_v1_synthetic_checks/gold11_terminal_artifacts_v4-sb2up7o7`.

The complete recursive inventory contains the attempt directory and exactly
these three files, with no extra entries, nested directories or symlinks:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 881 | `6e566ff876e1a795448d43c06d2e394b116302dbdcb7e79837f13dfd3b16d4b6` |
| `OUTPUT.txt` | 202 | `56556ec578ff4c53ec97dfb4e3cf3834044d4ee889590e8c55722d40ef0e7ce3` |
| `COMPLETED.json` | 1062 | `e2dadacdc3019d9ac1ef99e193c1978f9c2f8f35bfe31a1d6269df5fb8df9706` |

All three are regular files with link count one and distinct current inodes.
The total is 2,145 bytes. This inventory describes the retained gate envelope;
the tests' owned temporary fixture files are not retained historical artifacts.

Both receipts name `gold11_terminal_artifacts_v4`. STARTED records
`STARTED_SYNTHETIC_CHECK_NO_OVERWRITE`. COMPLETED binds the correct absolute
attempt path and its output hash matches the actual complete OUTPUT bytes.

## Outcome and limits

The complete OUTPUT reports `114 passed in 0.36s`, followed by worker resource
evidence. The selected single test file's 39 functions / 114 prospective
parametrized outcomes were independently counted in PRE. There is no selection
filter or stop-after-first-failure in the reviewed route. The aggregate pytest
result therefore covers all 114 frozen outcomes, not a favorable prefix.

| Evidence | Observed | Frozen bound / requirement |
| --- | ---: | --- |
| Passed / skipped | 114 / 0 | 114 / 0 |
| Exit code | 0 | 0 |
| Recorded budget stop | `null` | None |
| Parent monitored wall seconds | 0.45205862494185567 | 60 |
| Worker CPU seconds | 0.41811 | 45 |
| Sampled process-group peak RSS bytes | 53,968,896 | 1,073,741,824 |
| Worker peak RSS bytes | 53,968,896 | Separately reported worker high-water evidence |
| Retained OUTPUT bytes | 202 | 1,048,576 |

STARTED and COMPLETED contain the same limits. COMPLETED reports
`source_before_after_consistent: true`. Root additionally reported the launcher
command exited 0 with no live execution session; the retained exit code agrees.

## Independent source-binding verification

Each actual current hash equals PRE, STARTED's before hash and COMPLETED's after
hash. The separate STARTED harness/test fields agree as well:

| File | SHA256 |
| --- | --- |
| `scripts/gold11_terminal_artifacts_v4.py` | `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838` |
| `tests/test_gold11_terminal_artifacts_v4.py` | `d750fae94a56096aa92d86d611a48a646c09ea9128cb8bc1d5378343ba81a0d8` |
| `scripts/check_cad_guided_measurement.py` | `24480e4b18e00a08a9e6efecd654041965c6a2c5a0f3ac10cc8f9f684b741171` |
| `docs/research/2026-09-10_gold11_terminal_artifacts_v4_protocol.md` | `7caaafb4e49a9f292016c290be50de6eea2084b5c179e9ee45f78cd5fcd904de` |

No source drift, missing receipt, extra retained name, output mismatch,
unreported skip or failed outcome was found in this exact attempt.

## Narrow meaning and next boundary

The passing suite includes the concrete repeated-fill regression: 22 invented
directories, two existing terminals each, two fill passes and exactly 44
unchanged files. It also exercises same-byte distinct-inode extra-file veto,
bounded complete-directory inventory, no favorable alias deduplication and
controlled publication/read/close/fsync failures. These are useful evidence for
the new helper's stated contract, not a repaired receipt for the earlier V3 run.

The helper still requires a caller absolute deadline, trusted read-only
validator and quiescent writers. Single-directory inventory is not full-campaign
custody; two scans are not an atomic adversarial snapshot. The tests do not
exercise the 128MiB-file, 8GiB-total or 1024-entry ceilings at capacity. Harness
setup, hash/receipt I/O and cleanup remain outside parts of its monitored wall
interval, and sampled RSS is not an OS-enforced memory guarantee.

Preserve this terminal attempt unchanged. A separately reviewed runner may now
use this gate as one prerequisite for integration, but this POST does not release
any existing runner or physical experiment, restore old custody, produce bend
measurements or demonstrate historical-F1 or production progress.
