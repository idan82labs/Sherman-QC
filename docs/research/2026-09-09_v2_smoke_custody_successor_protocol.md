# V2 smoke custody successor — 2026-09-09

Status: **PROSPECTIVE, SOURCE-ONLY; NOT EXECUTED.** This is an engineering
successor to a retained negative, not permission to retry that attempt, open
physical inputs, or freeze the single-CAD physical diagnostic. Root and an
independent reviewer must approve the exact successor source and its pure/mock
evidence before one new invented-subprocess attempt. No permission, native
algorithm, physical diagnostic driver, input scope, or historical limit changes.

## Retained negative and immutable source

The first actual registered smoke, [native_validity_v2_guard_smoke-yv3p32kc](cad_engine_v1_synthetic_checks/native_validity_v2_guard_smoke-yv3p32kc/COMPLETED.json),
terminated with `SMOKE_GROUP_CUSTODY_FAILURE`, exit -9, parent wall
0.46143204206600785s and sampled group peak51,691,520B. One ACK was recorded;
all owned groups were subsequently confirmed empty. A cleanup signal reported
`PermissionError`. The original main-loop exception was not retained, so that
receipt cannot establish which source race fired. Nor does its ACK prove main
admission: the old cleanup path could also publish an ACK.

Root retained exact source bytes under
[source_snapshots/smoke_registry_5f6f157a](source_snapshots/smoke_registry_5f6f157a/scripts/check_cad_guided_measurement.py).
Their independently re-read hashes match the negative receipt:

| Frozen old source | SHA256 |
| --- | --- |
| `scripts/check_cad_guided_measurement.py` | `5f6f157a99ecb69f3240959ea31a2a9f3e4505fb0f83cd9b5c0cd425d1948d25` |
| `tests/test_cad_smoke_group_custody.py` | `1ee518753d6d89197ae66639217af8e9073138e0fb687290805dd75b21415e00` |
| `tests/test_v2_native_validity_supervisor.py` | `e77553d16d6d76af778ca73ed874f2ee22732f69a4a8c65c557793bf7fb7966d` |

These snapshots, prior receipts and prior process identities are terminal and
must remain unchanged. Do not signal any historical PID/PGID or resume its
private cache. The cleanup permission error is preserved, not interpreted as
authority to widen permissions or to signal those identities again.

## Engineering hypotheses and narrowly scoped repair

Independent source review found two concrete races, separately from the
unattributable original exception:

1. The harness sampled `ps` and then read registration requests. A request
   published between those operations could be tested against an older process
   table, falsely rejecting a newly created coordinator as absent.
2. Atomic publication linked a completed staging inode to its final name, then
   unlinked the stage. That unlink changed the shared inode's ctime while a
   concurrent bounded reader could be checking descriptor stability.

The successor captures only the two fixed request slots **before** obtaining a
fresh process table, and validates only that captured mapping. Later publication
is deferred to the next poll. The single live sampling method enforces this
order for main admission and cleanup. Main admission alone can publish an ACK.
Cleanup may retain a freshly verified pending group solely for termination;
it never publishes an ACK or converts cleanup discovery into successful admission.

Completed staging hardlinks now remain until terminal registry/cache cleanup.
Two request slots and two ACK slots produce at most eight fixed target/stage
names. Files remain exclusive, private, regular, no-follow, at most2KiB each;
descriptor identity, length, modification/change timestamps and link count
must be stable. The reader is not weakened to tolerate a mutation.

ACKs contain `MAIN_ADMISSION` and a monotonic admission-decision timestamp.
The terminal registry record separately identifies first observation phase/time,
ACK phase/time and completed-publication time. A cleanup-only record has no ACK
phase/time. The client rejects cleanup-phase or future-dated ACKs.

The first exception retains its original type, fixed reason code and exact
stage; arbitrary exception path text is excluded. Cleanup failures are separate
structured records and cannot overwrite that first failure. The test-side
registration proxy likewise preserves its original failure if kill/reap fails.
The missing-live-pytest-row refusal and direct-owned-child kill remain intact.

## Unchanged two invented smoke cases and budgets

The same two smoke cases remain selected from
`tests/test_v2_native_validity_supervisor.py`:

- Guarded stdlib success: an invented Python namespace and text, bounded
  TemporaryDirectory cleanup, and required denials of unrelated reads, sockets
  and subprocess creation. No OCP/native diagnostic import or physical input.
- Stuck-cleanup control: an invented coordinator catches its own timeout and
  sleeps while a same-group stdlib child remains alive. The watchdog must stop
  and confirm the complete group. No geometry fixture, healing or inference.

The outer harness remains60s wall,45s pytest-worker CPU,1GiB sampled RSS for the
pytest tree/group plus all registered groups, and1MiB forwarded/retained output.
Only this smoke keeps its existing nominal55s work stop and5s cleanup reserve.
Process-table reads remain1s bounded; sampling and OS scheduling are not a
hard-real-time or adversarial PID-reuse attestation. The inner success case keeps
its10s work/15s fatal boundary and stuck-cleanup case its1s work/5s fatal boundary.
The registration ACK wait remains2s and consumes those existing case intervals.
No cap is raised and no case is removed or simplified.

The outer harness creates a0700 private registry and random nonce only for this
smoke. A forwarding new-session Popen proxy blocks before returning until a
live pid=pgid/ppid=pytest-child sample is ACKed. Thus the watchdog cannot release
stdin first. Both fixed coordinators reject EOF/malformed stdin before any work.
Acknowledged groups remain owned across reparenting, retire only after an empty
sample, and are never re-admitted after retirement. Cleanup independently keeps
the live direct pytest child owned, scans before group kills, and preserves the
private registry/cache when terminal custody cannot be confirmed.

## Prospective gates and decision

1. Root and an independent reviewer inspect the exact successor harness, both
   test files and this protocol. Old frozen bytes and the physical driver remain
   unchanged. The physical driver remains bound to
   `11049b18204501481d7ab6a378f8402ae5545382bb3fc4007e126b19dbde52e3`.
2. Run only the named `cad_smoke_group_custody` pure/mock suite under60s/45CPU/
   1GiB/1MiB after source approval. The prior45 pure/mock gates remain, with
   explicit API/stage-retention adaptations, plus deterministic race controls:
   publication after capture; request→fresh sample→ACK order; cleanup-only
   custody without ACK; a descriptor opened at publication remaining stable
   when the publisher returns; reproduction of the old unlink interleaving
   still refusing; and original-error preservation. No subprocess/native/raw
   execution occurs in these controls. A test's source is not a passing result.
3. After that retained evidence and independent approval, root may create one
   NEW `native_validity_v2_guard_smoke` attempt with the same two cases. Its
   automatic STARTED/source-before closure binds this protocol and all named
   source/test dependencies. Record the exact hashes in the independent audit;
   this protocol does not embed a circular self-hash or pre-claim success.
4. Success requires all selected tests passing, two distinct main-admission
   ACKs, no budget stop, no primary/cleanup errors, unchanged source hashes and
   all owned groups confirmed empty. Missing, unavailable, failed or ambiguous
   evidence is terminal negative, never permission to auto-retry or raise caps.
5. No physical freeze is allowed until this successor passes. Even then, the
   separate single-CAD freeze and independent exact-bundle pre-run audit remain
   mandatory. This smoke is engineering containment evidence only—not native
   validity, physical accuracy, Gold performance, promotion or production proof.

Author validation at handoff is source inspection and AST/hash checks only.
Execution and the independent decision remain pending.
