# V9 runner and waiting launcher: bounded closure review

Date: 2026-09-13, 04:50 UTC follow-up. Reviewer: root Codex, independent
of Fable's implementation. Source/test/declaration reads and hashes only;
no imports, tests, raw/Gold, attempt or opaque-cache access. No physical PRE
or execution authorization is issued by this report.

## Exact reviewed checkpoint

- Runner: `scripts/run_gold11_mesh_development_v9.py`,
  `ad61e12b0a06c0cafb7e515687019e42c6e137e26a419753aaeef2360f5b4241`.
- Waiting launcher: `scripts/gold11_v9_execute_waiter.py`,
  `a0b8c0b3e3ddb2e3041f25f7bd5ea0a56b4430871996f3f0c957a2823ad6f680`.
- Author closure response:
  `31def50d1d4795c22f8c2c79b8899eafe445c388ecfd7041b1c4697da0985de4`.
- Protocol/design snapshots are the response's
  `16f20a99766838c6ca210be86582c40089aafeae70540dc6d874848ab5b4c1f2`
  and `03fa29937f2614971a515028bb9c500cb1736f91f962b275fb9cd81082ae0be9`.

## Source closures and retained scope

Runner routes both arms to mesh V16 / initializer V15 / kernel V2 and binds
completion/layer configuration in the full arm request. The scoring source
load order includes contact V3 before wrapper V4; the actual scoring call is
V3, and report validation recognizes V3. This closes the old consumer-schema
route at source. The reviewed V8-to-V9 diff retains the supervision and
terminal-fill mechanism, all22/full11/reference71 denominator, legacy matcher
and fixed outer resource limits. The new additional completion allowance is
declared rather than represented as unchanged total work.

The source test includes an in-memory freeze/verify round trip, both actual
policy constructors, source-import closure and exact comparison of unchanged
V8 constants after removing the listed deltas. Root read it but did not
execute it. Fable reports 826 tests passed at the checkpoint; that is
author-observed evidence, not a root rerun or physical score.

The protocol now lists the angular-support delta and scopes its spacing and
phase claims to the constructed fixture. Material identity and full
completion-origin transport are covered by the separate final reviewers;
their remaining findings must be reconciled before whole-candidate closure.

## P2 — Waiting launcher is outside the declared frozen source inventory

`SOURCES` at runner lines64–120 omits `scripts/gold11_v9_execute_waiter.py`.
`freeze()` lines1020–1050 retains only that inventory; `verify_bundle()` checks
only its exact hashes. The protocol now requires this launcher to supply the
missing actual outer-exit evidence, so its exact bytes must be included in the
frozen bundle and external archive/PRE, or equivalently separately bound by an
explicit checked launch-source declaration. Merely naming its path in the
protocol does not bind the executed implementation. The smallest fix is add
it to `SOURCES` and the source-closure assertion; no scoring change is needed.

## P2 — Waiting-parent entry can overwrite the exit evidence it is meant to preserve

`waiter()` opens stdout with append at line35 and PID/exit records with `w`
at lines38/44. The public `--waiter` dispatch at line74 bypasses `launch()`'s
record-existence and overlap checks. A second waiter invocation can therefore
append to an earlier stdout file and replace its PID/return-code record, even
though the scientific runner separately refuses a spent attempt. Two
concurrent fresh launchers can also pass the non-atomic existence check before
either waiter opens its files. This is an exit-evidence preservation defect,
not a claim that the runner would successfully re-execute physical inference.

Reserve the outer evidence once, exclusively, before starting the runner;
apply admission to every entry path and never append/truncate an existing
receipt. Keep the runner's own one-shot gate intact. A few isolated fake-child
tests can check a successful and nonzero return code, duplicate/direct-waiter
refusal with unchanged existing records, and launch failure, without touching
the real attempt or invoking any benchmark. Record unobserved return codes as
unobserved if the waiting parent itself fails. Correct the line8 scope comment:
the current launcher enumerates attempt-root names, although it does not read
their scientific contents.

These are bounded execution-evidence fixes. They do not justify another broad
readiness campaign or requiring a positive result on every physical case.

## Disposition

Runner scientific transport is source-closed within the reviewed delta.
Resolve the two launcher P2s and the separate material/completion findings,
record revised hashes and focused results, then reconcile those exact edits.
The next steps remain exclusive freeze, independent physical PRE/external
archive, one execution, observed outer exit, independent POST and honest score.

## Exact launcher delta accepted — 05:08 UTC follow-up

The 05:06 author checkpoint retains the initial review history and changes
the runner to `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1`,
waiter to `85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688`,
and new fake-child test to
`20feb44003cb82e5e3583a6b0a4360afd23491e521ae99af300ea8e6b625c04c`.
Root fully read the revised waiter and its tests and checked these hashes.
Removing only the added waiter line from the runner produces exactly the
previous `ad61e12b...4241` hash (9b1fb3); no other runner source changed.

Both P2s above are closed within the declared local one-shot evidence scope.
The waiter is in `SOURCES` and the source-closure assertion. Launch reserves
PID/exit/stdout records with O_EXCL; the reserved waiting parent checks token,
pending status and an exclusive advisory lock before claiming that reservation.
Duplicate, spent, direct-without-token and missing-reservation paths refuse;
the final exit record atomically replaces only this invocation's pending
record. The runner's scientific one-shot gate remains unchanged. Claimed or
partially reserved evidence is never silently retried after failure.

Root independently ran the entire named fake-child suite ONCE after reading
its fixtures: 9 passed in 0.16 s, outer exit0 (tool ce2dcf). Prospective local
test budget was 45-second alarm and 20-second CPU limit; no numerical/native
geometry work. Every tree was synthetic and every launched child was the
explicit Python fake command; no real attempt, raw/Gold or cache was touched.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -c 'import resource,signal,pytest; resource.setrlimit(resource.RLIMIT_CPU,(20,20)); signal.alarm(45); raise SystemExit(pytest.main(["-c","/dev/null","--noconftest","-p","no:cacheprovider","tests/test_gold11_v9_execute_waiter.py","-q"]))'
```

These tests establish fake-child success/nonzero-exit capture, refusal without
overwriting prior records, lock serialization and explicit unobserved failure
handling. The real CLI-entry test invokes a waiting-parent subprocess, not
the launcher's detached-spawn branch; detachment remains a source-reviewed
`start_new_session=True` behavior. This is not a power-loss, SIGKILL or
adversarial-filesystem durability guarantee: if no actual exit is observed,
the physical POST must keep it unobserved. No score is inferred from an exit
record, and no physical execution occurs in this check.

No remaining blocker in this runner/waiter delta. Whole-candidate acceptance
still depends on the separately scoped material/completion delta reviews and
the independent physical PRE after freeze.
