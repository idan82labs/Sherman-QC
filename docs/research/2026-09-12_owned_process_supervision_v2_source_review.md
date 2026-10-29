# Owned-process supervision V2 — independent source review

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/cad_blocker_closure_review`; independent of the supervisor author and
test author. This is a source/design review, **not an exact PRE, test result,
host qualification, integration approval or execution authority**.

## Disposition and exact reviewed source

Conditional source GREEN for proceeding to the bounded qualification work
below. Both identified receipt defects are closed in the reviewed source;
no additional unsafe-signaling path was identified within its declared sampled
scope. No target code, test, native API, process signal or experiment executed.
No protected inputs, attempt artifacts or private caches were accessed for this
review. V1 and the terminal V5 failure remain unchanged.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/gold11_owned_process_supervision_v2.py` | 37329 | `649815cad6dc10d70b34cf9ae079fe77146e371ee12d1bcc4b7ea64e340492e5` |
| `tests/test_gold11_owned_process_supervision_v2.py` | 15423 | `d8d3811477d2ad44ec38a258827888a110abd46e779536603cf4994bc999cade` |

The earlier supervisor was fully read at 36204 bytes / 688 lines, SHA256
`23f75856bd39a00a6e92ce55b71cb16612672cae6c7235122c61bb30b90b0b1b`.
The complete receipt amendment was read. Reversing only its enumerated text
changes in memory reproduced that exact earlier SHA256, establishing that the
new 706-line source differs only by the two receipt fixes and their publication
fields/counters. This was a stdlib source-text/hash/AST check, not execution of
the supervisor. The complete 358-line test module was read; AST counting gives
**21 test functions / 40 prospective outcomes / zero executed outcomes**.

## Closed findings

1. **Historical unknown-PID peak.** `_mark_unknown` at lines 284–287 immediately
   updates the peak on insertion. Both unavailable owned-query and signal
   branches use it. Later successful inspection or verified retirement may
   clear current uncertainty, but cannot erase its historical peak. The
   prospective transient-query regression distinguishes peak one from final
   still-present false.
2. **Causal identity evidence after routine churn.** `_event` at lines 265–282
   now retains a separate first-16 critical-event history, with a separate
   discarded counter. Routine absence/replacement retirements cannot exhaust
   those slots before a later escape or unavailable query. `publish` emits
   `identity_error_events`; the receipt declares both 16-event caps. The
   prospective late-escape regression fills routine history first and then
   requires the causal PID/group evidence. This remains bounded first-event
   evidence, not a complete unbounded process history.

## Identity, signal and cleanup assessment

- Lifetime identity is `(pid, uniqueid)`. Explicit kernel absence or a verified
  replacement retires the old lifetime. A replacement is admitted again only
  through current owned-group or positive parent-generation evidence; a new
  unrelated process is not signaled using the old entry. Same-lifetime exec
  refreshes `idversion` without treating it as PID reuse. Previously admitted
  descendants remain tracked after ordinary reparenting.
- New descendant admission requires current PPID and original parent unique-ID
  agreement. The inner supervisor does not admit unrelated members of its
  shared group. The outer supervisor uses its unreaped direct child as the
  numeric group anchor. Every accepted row/RSS sample has bounded repeated
  identity reads; unstable identity, short responses and permissions failures
  fail closed rather than becoming absence or zero RSS.
- Signaling rechecks the lifetime and group, skips zombies/replacements/foreign
  groups, and uses only the fresh audit-token SIGKILL operation. The kernel
  resolves PID/version and validates an exact process identity before signaling;
  the wrapper returns errno as its function result. Token ESRCH is not treated
  as proof of process retirement. These interpretations agree with Apple's
  [libproc wrapper](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c)
  and [kernel implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/proc_info.c).
- No explicit numeric kill, killpg, terminate or poll call exists. Reaping is
  deferred until the observed root is zombie and its owned-group set is empty.
  After reaping, only a read-only final outer-group census is allowed; a failed
  final check does not restore signaling authority. Unconfirmed cleanup returns
  explicit attention state, not permission to signal receipt PIDs later.

The flavor-18 BSD/unique-ID layout and zombie inclusion were checked against
Apple's [private header](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info_private.h)
and the kernel's argument handling. Upstream source agreement does **not**
establish that these private interfaces exist or behave identically on this
installed host. That remains a separate bounded host qualification.

## Bounded work and import scope

The declared limits are 4096 PIDs per enumeration with an overflow slot,
8192 tracked lifetimes, 1000000 total admissions, 3000000 work samples and
100000000 attempted work-native calls. Separate failed-work cleanup has
1024 samples / 262144 attempted native calls and a five-second reserve. A sample
has at most 65536 attempted native calls and a one-second deadline, also bounded
by its phase deadline. Identity coherence has two attempts. Both event lists
have 16 slots; cleanup error detail retention is separately bounded. Cap checks
are precharged; these values are limits, not measured costs or proof that every
combination within the nominal envelope completes.

The supervisor imports exactly ten stdlib modules: `ctypes`, `errno`, `json`,
`math`, `os`, `selectors`, `signal`, `subprocess`, `sys`, and `time`. It has no
project imports and loads no native library at import time. The test fixture
loads only the named supervisor definitions; native-adapter tests construct
objects via `__new__` and replace native operations. The mock census never
spawns or signals a process. Tests also import pytest and ordinary stdlib
helpers. A later exact harness must bind the supervisor/test bytes, its own
source dependencies, applicable test-package initializer and ordinary runtime
entrypoints. This review does not attest a complete installed binary closure.

## Remaining qualification and assurance boundaries

The 40 proposed controls cover replacement churn (4096 invented generations),
exec, reparenting, foreign descendants, uncertain queries, explicit absence,
root-anchor loss, surviving descendants, tracking caps, receipt retention,
ABI layouts, RSS coherence and fresh-token races. They are useful source tests,
not a completed qualification matrix. Before execution/integration release:

- Add and independently review public orchestration controls: normal EOF and
  reap; producer exit with surviving descendants; failed initial/fresh
  authentication; post-reap nonempty census with no subsequent signal;
  dispatch, log write/flush/close, selector and cleanup failures; truthful
  uncertain-storage and process-attention receipts.
- Add bounded native-adapter list/call/sample controls for malformed, duplicate,
  short, denied and overflowing responses; exact cap edges; deadline expiry;
  independent cleanup reserve; and failure paths that must not retire or signal
  an unverified lifetime. Include zombie/RSS transitions and stale-token exec
  races without treating accepted signaling as confirmed exit.
- Freeze the resulting exact source/runtime/protocol and independent PRE before
  a bounded mock gate, then separately qualify the installed host's private
  API/owned-child behavior. Neither gate is physical-engine or scoring evidence.

Native lists and locally cached rows are point-in-time observations, not one
atomic whole-tree snapshot or continuous containment. A descendant born,
escaped and reparented between observations may be missed. Audit-token identity
coupling is not an atomic group-membership condition at signal time. RSS is
sampled rather than an OS allocation cap; an unseen peak is not zero. Kernel
calls, launch/stdin/log I/O cannot be forcibly cancelled by these Python
deadlines, so the caller still needs its own enclosing budget and failure
handling. The caller must remain the sole waiter with SIGCHLD at its required
default. No current benchmark gate, inference rule, scoring rule or terminal
attempt is changed by this source review.
