# Terminal-artifact V4: independent first-gate PRE audit

Date: 2026-09-10. Decision: **PRE-GREEN for the exact named invented-filesystem
gate below**, conditional on unchanged four-file closure. No imports, tests,
native operations, physical inputs or retained scientific artifacts were
executed/accessed by this review. This document is source review, not a result.

Reviewer: `cad_measurement_integration_review`. Helper author:
`cad_support_lineage`; independent test author: `production_architecture_review`.
The reviewer authored neither this helper nor its tests. Root separately read
both full files and owns the harness/protocol and any execution.

## Exact release closure

All four live files were read within the declared source scope and hashed:

| File | SHA256 |
| --- | --- |
| `scripts/gold11_terminal_artifacts_v4.py` | `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838` |
| `tests/test_gold11_terminal_artifacts_v4.py` | `d750fae94a56096aa92d86d611a48a646c09ea9128cb8bc1d5378343ba81a0d8` |
| `scripts/check_cad_guided_measurement.py` | `24480e4b18e00a08a9e6efecd654041965c6a2c5a0f3ac10cc8f9f684b741171` |
| `docs/research/2026-09-10_gold11_terminal_artifacts_v4_protocol.md` | `7caaafb4e49a9f292016c290be50de6eea2084b5c179e9ee45f78cd5fcd904de` |

The archived predecessor harness at
`docs/research/source_snapshots/two_plane_schedule_pre_628f3a84/check_cad_guided_measurement.py`
was separately hash-checked as
`628f3a84c98806f1d24c7b47f8d7683911a2f8bfb7f67dc8698887f1d8eda0f4`.
Its complete diff against the new harness adds only the terminal-artifact and
independent two-plane-scheduler route mappings, source bindings, 60/45/1GiB
budget memberships and retained-receipt memberships. No worker algorithm,
environment, native opt-in, subprocess supervisor or cleanup mechanism changed.
This PRE does not authorize execution of the separate scheduler route or `all`.

## Source and oracle findings

The full 519-line helper and 676-line test file were independently read.
No blocking finding remains for their explicit synchronous, caller-owned,
quiescent-writer contract.

Publication checks existence before creating staging files. A valid existing
terminal is bounded/read/stability checked, validated by singleton `True`, and
returns without calling its factory, generating a stage name, writing, linking,
unlinking or fsyncing. New bytes use an exclusive stage and exclusive target
hardlink, with short-write, file/target identity and fsync/close checks. A race
that creates the target after staging remains a failure; no favorable
preexisting-record reclassification is made. Ordinary failures preserve partial
evidence. Removing a stage is restricted to the normal successful-link path;
late failure may retain two names or only the target, and receipts distinguish
those states. Empty successful publication is correctly separate from writing
nonzero content bytes.

Inventory enumerates all names in one owned directory, including dotfiles,
under bounded initial/final passes. Declared reads are streamed under exact
per-file and cumulative admission bounds. Unexpected files receive no content
read and always veto completion. Identical contents never deduplicate names or
charges; actual device/inode/link metadata distinguishes hardlink aliases.
Missing names, wrong types/hash/size, cap excess, read/close failures and observed
name/identity changes remain unavailable. Path traversal opens every component
without following symlinks and rechecks the original pathname after the pinned
descriptor operation.

The concrete regression creates 22 invented directories with two existing
terminal records each. It performs two complete fill passes and verifies exact
unchanged names/inodes/content/timestamps and a full owned-tree inventory of
exactly 44 files. The collision control creates a distinct-inode same-byte
partial, then verifies metadata-only extra detection and whole inventory veto.
This directly tests the missing custody seam without touching the failed V3
attempt or any retained control bytes.

Static count confirms **39 test functions, 114 outcomes, zero skips**:

| Fixed coverage group | Outcomes |
| --- | ---: |
| New/existing publication, 22-directory repetition, mixed records, collision | 7 |
| Validator/factory failures and target appearing during factory | 13 |
| Fsync/unlink/write/close failures and BaseException descriptor closure | 13 |
| Existing-record instability, types, no-follow components, pathname rebinding | 10 |
| Directory/name/cap validation and publication byte boundaries | 22 |
| Inventory entry/aggregate/hash/size boundaries and hardlink charges | 12 |
| Unexpected files/dotfiles/types/overflow/empty directory | 8 |
| Inventory budget and expectation-schema negatives | 16 |
| Late reads, changes and both enumerator failure passes | 8 |
| Empty publication, inventory closes, admission/read race, import closure | 5 |
| Total | 114 |

Faults use explicit controlled syscall/callback seams plus real tiny files.
The custom failed enumerator implements its iterator and context-manager
interfaces; it is not the earlier non-iterable mock. All filesystem fixtures
resolve `tmp_path` canonically and stay below that owned directory. Symlink and
replacement targets also stay there. No runner, backend, native, scientific
input or subprocess experiment is imported or invoked. The helper's imports
are exactly `hashlib`, `os`, `secrets`, `stat`, `sys`; source tests also inspect
the absence of import-time I/O.

## Harness, limits and one permitted gate

The reviewed route `gold11_terminal_artifacts_v4` selects exactly
`tests/test_gold11_terminal_artifacts_v4.py`, without ROOT namespace insertion,
native/child-smoke flags or invocation of another test module. Source bindings
are exactly the helper, protocol, selected test and current harness.

The worker uses isolated Python, the configured pytest installation,
`--noconftest`, `/dev/null` pytest configuration and disabled plugin autoload/cache.
It has an RLIMIT_CPU of 45 seconds; the parent applies 60 seconds monitored wall,
1GiB sampled process-group RSS and 1MiB output, with numerical threads one and
isolated HOME/TMPDIR/XDG cache. The parent creates one fresh retained attempt
and exclusively writes STARTED/OUTPUT/COMPLETED with before/after source hashes.

Only root may execute once, after this exact PRE:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_terminal_artifacts_v4
```

Required result is 114 pass, zero skips, exit0, no stop, all four hashes unchanged
and independent retained-only POST-GREEN. A failed or incomplete run stays
terminal. No retries, favorable-prefix claims or changes to started limits.

## Limits of this approval

The helper is not a deadline supervisor. The caller must enclose its synchronous
filesystem/callback work in the campaign's absolute deadline and stop writers.
The inherited harness also has setup/hash/receipt I/O and cleanup outside its
monitored interval; sampled RSS is not an OS memory isolation guarantee.
Tests do not exercise 128MiB-file/8GiB-total/1024-entry maxima or certify arbitrary
concurrency, filesystem faults or callback behavior.

Two name/fingerprint passes are not an atomic adversarial snapshot. Single-
directory success is not whole-tree custody; a future runner must enumerate and
bind every campaign directory. Observed hashes without expected values are not
historical authentication. Successful qualification cannot restore old V3
custody, qualify an existing runner, permit a metadata-only physical rerun,
produce measured output or establish F1/production progress. Integration into a
separately justified engine successor remains a separate reviewed step.
