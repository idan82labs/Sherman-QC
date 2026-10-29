# Connected V12 array packet — independent terminal POST, 2026-09-10

Decision: **NEGATIVE_FROZEN_ARRAY_GATE_ADVANCEMENT_BLOCKED**.
Reviewer: independent AI agent `/root/production_architecture_review`, not the
candidate, fixture/test or harness author. Retained negative evidence is
authenticated; the connected gate is not successful or complete. No retry,
native/byte promotion, full-runner or physical execution is released.

## Exact attempt and custody

First-once attempt:
`docs/research/cad_engine_v1_synthetic_checks/connected_v12_connected-cxx6ws5m`.
Root directly observed parent exit1. The retained worker exit is-24; OUTPUT is
empty, with no pytest summary or worker CPU/high-water footer. This reviewer
inspected retained evidence only and did not rerun/import target code.

The frozen [array PRE](2026-09-10_connected_v12_array_pre_run_audit.md), SHA256
`e411cdc66dc2170b325699e51f0f855cc511d6b12cd49f8f237a0561cff89372`,
required one complete18-arm packet with eight positive obligations. All49
STARTED/COMPLETED/current source hashes match the complete
[source PRE](2026-09-10_connected_v12_source_pre_run_audit.md) ledger, SHA256
`72174dfb50f47ea432fc1df34aba1f58f6f7f93473372b88256809e3e7fe5272`.
No source drift is implicated in this negative.

Exactly four regular top-level files and the real `ARTIFACTS` directory were
found. That directory contains exactly19 expected regular files:10 START and
9 RESULT, no extra/partial/symlink names. Every retained leaf's length and SHA
matches inventory, checked one bounded file at a time with stable identity.
All JSONs parse without nonfinite constants. Artifact bytes total3,377,247:
7,336 START bytes and3,369,911 RESULT bytes. No retained file exceeds its cap.
Top-level files total17,326 bytes; the complete owned regular-file count is23.

| Receipt | Bytes | SHA256 |
| --- | ---: | --- |
| `STARTED.json` | 6,045 | `d1817b0d560712b1de6a0e79ff8b19988ed35922f48e49003aeeaf729133500e` |
| `OUTPUT.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `COMPLETED.json` | 7,040 | `0a7158b200e59cad62b34e06da712b8719b9cd1f3143f74268e16697a2ee93a8` |
| `ARTIFACT_INVENTORY.json` | 4,241 | `21eb37bb41e23ca8695fd15c538cba8d27f4d858f236f8a2adfc0708b8718971` |

The manifest's `complete:false` is correct:17 of the required36 artifacts are
missing. Its sole inventory error is `INCOMPLETE_18_ARM_ARTIFACT_DENOMINATOR`.
The exact18-row parent disposition ledger is present and was independently
reconciled to actual names. Missing results are not dropped or called successful
abstentions. Source setup/final I/O and cache limitations remain as frozen;
the recorded private cache and optional separate source archive were not read.

## All18 arm dispositions

Every returned RESULT has `COMPLETED_ARM`, execution_error=null and
safety_error=null, but positive_oracle_passed=false. This is nine completed
negative arm records, not nine passed tests. No aggregate pytest outcome exists.

| Case | Arm | Required positive | Independently observed disposition |
| --- | --- | --- | --- |
| P-clean | C0 | no | returned; CAD-point admission refusal before initialization |
| P-clean | C1 | yes | returned; same admission refusal; positive failed |
| P-clean | C2 | yes | returned; same admission refusal; positive failed |
| P-noisy | C0 | no | returned; CAD-point admission refusal before initialization |
| P-noisy | C1 | yes | returned; same admission refusal; positive failed |
| P-noisy | C2 | yes | returned; same admission refusal; positive failed |
| U-R | C0 | no | returned; possible generated whole-part poses ambiguous |
| U-R | C1 | no | returned; same ambiguity; no measurement |
| U-R | C2 | yes | returned; five contexts, then all support records conflict; positive failed |
| U-A | C0 | no | START only; worker termination/publication interruption; internal stage UNKNOWN |
| U-A | C1 | no | no START/RESULT; not attempted or start publication failed |
| U-A | C2 | yes | no START/RESULT; required outcome UNKNOWN |
| L-R | C0 | no | no START/RESULT; not attempted or start publication failed |
| L-R | C1 | no | no START/RESULT; not attempted or start publication failed |
| L-R | C2 | yes | no START/RESULT; required outcome UNKNOWN |
| L-A | C0 | no | no START/RESULT; not attempted or start publication failed |
| L-A | C1 | no | no START/RESULT; not attempted or start publication failed |
| L-A | C2 | yes | no START/RESULT; required outcome UNKNOWN |

Thus zero positive-oracle passes are observed among nine returned arms; five
required positives have returned failures and three required positives remain
unobserved. Do not report zero measured successes out of18 completed trials.
The same-case input digests match across retained STARTs/RESULTs and arms.

## Proven stops versus unobserved causes

**Planar integration-fixture admission error.** All six planar policy snapshots
declare exactly8 CAD initialization points, while preserving50 raw scan rows,
64 scan triangles and4 CAD triangles. Their registration reason is
`WHOLE_PART_ARRAY_OUTSIDE_SCOPE`, candidate counts are zero, numerical/candidate
completion is false, and no initialization object exists. The oracle's
`KeyError('initialization')` is secondary to that refusal.

The cause is source-established: `run_array_arm` in the frozen constructed test
uses unique CAD triangle vertices as `cad_initialization_points`; V11:704 and
V12:713 call `_canonical_scan` before `_initialize`. In
`backend/cad_guided_global_registration_v1.py:71`, `_positions` requires at
least36 input points and raises that exact reason; `_canonical_scan` also
requires36 unique points. Eight corners cannot pass this wrapper. The earlier
planar geometry was preserved, but its new full-registration call was not an
admissible integration. The source gate and PRE missed that call-contract
mismatch; they did not execute the planar registration path. These six records
do not test whether V12 fixes the original noisy planar proposal failure.

**U-R reaches geometry but does not produce an eligible quantity.** All three
arms use284 CAD initialization points and retain complete initialization with
five start groups, five attempted candidates, complete candidate evaluation and
complete numerical consumption. C0/C1 refuse
`AMBIGUOUS_POSSIBLE_GENERATED_WHOLE_PART_POSES`; their measurement stages do not
run. These are completed ambiguity refusals, not missing proposals.

C2 instead retains five supported conditional context indices0..4. Registration
is `CONDITIONAL_MEASUREMENT_CONTEXTS`, accepted_registration=false; no pose is
accepted. All five context windows are MEASURED_CONTEXT, but they induce five
different exact raw-support records. Every record is ultimately UNAVAILABLE
with `AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT`, measurements=null and
finite_support=null. The retained ownership ledger reports1,245 row visits,
2,372 competing claim pairs and five invalidated records. Independent integer
set checks find a qualifying shared raw-ID conflict between every pair of the
five records; duplicated coordinates are not needed to explain this veto.
No fitting, scorer or production validator was rerun to make that check.

C2's measurement stage completes with
`NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS`; predictions are empty,
observed_bend_count=null and production_use_allowed=false. Actual publication
validation passed, which authenticates an honest refusal, not a positive.
Five full role bitmaps retain all450 rows, exclude zero, and report zero cap-like
triangles/784 unknown triangles. Their packed zero bits and kept/excluded raw-ID
hashes were independently reconstructed. Cap exclusion therefore did not cause
this U-R refusal. The full-raw crop ledger reports2,250 attempted/completed row
visits,1,589 local rows,1,245 canonical-refit row visits,1,520 role-membership
visits and11,760 incidence visits. These do not prove a valid canonical answer
survived; the final records explicitly withhold it. No best alias is promoted.

**Global CPU termination limits the remainder.** Worker return-24 is SIGXCPU
on this host, consistent with the frozen installed240-second worker CPU limit.
There is no worker completion footer, so final measured worker CPU and its
high-water RSS are unavailable, not zero. Parent `budget_stop:null` only says
the wall/RSS/output supervisor did not issue its own stop; it does not erase
the CPU-signal failure. U-A C0 has only START, so its last internal stage and
any computed-but-unpublished answer remain UNKNOWN. The eight later arms have
no durable start; no claim is made about their geometry or cap handling.

| Retained resource evidence | Value |
| --- | ---: |
| Parent supervised interval | 240.6564266658388 s |
| Work stopped / reaped | 240.63894904125482 /240.63895345805213 s |
| Group-empty checked | 240.65640412503853 s |
| Sampled owned-group peak RSS | 116,293,632 B |
| Observed/stored stdout | 0 /0 B |
| Frozen wall/work/CPU | 300 /295 /240 s |
| Frozen sampled RSS/stdout | 4,294,967,296 /1,048,576 B |

Only PID83582 was observed in the owned group; group-empty=true,
requires_group_attention=false, primary_error=null and cleanup_error=null.
There is no observed wall, sampled-memory or output overflow. This is valid
terminal cleanup, not successful scientific completion. Per-arm CPU attribution
is not retained; do not assign the entire240-second campaign cost to U-A.

## Authenticated result files and conclusion

All ten START receipts are bound by the exact inventory digest above; the nine
returned RESULT files were read and independently hashed in full:

| RESULT | Bytes | SHA256 |
| --- | ---: | --- |
| `ARM_RESULT_00_C0.json` | 7,202 | `8d89522b97c8900af00596296c036a3f6a495394a951f395e85a754ecb78281e` |
| `ARM_RESULT_00_C1.json` | 7,720 | `82279e0d8c7bd1407c7876a57df42f998e962b3b86bc22aef8076b3edc95b70e` |
| `ARM_RESULT_00_C2.json` | 7,750 | `d2ab2b8ed5be0580e184be1ce2ceb9f9ab7d7e539660d0a2ad1e483d0b2083cd` |
| `ARM_RESULT_01_C0.json` | 7,202 | `9b4fc84f3f3a9f7bfecb5c538fed57b85c7dd101947464a48490c19428916841` |
| `ARM_RESULT_01_C1.json` | 7,720 | `03db7d8c02c641b813fcb5374de8ee5835cc7b5a54fc5de8664d89ab03c4f711` |
| `ARM_RESULT_01_C2.json` | 7,750 | `1fc534261c3441a14b1debde9de3dabf46148c33172e7b98195de9494511726e` |
| `ARM_RESULT_02_C0.json` | 1,093,048 | `f2bccaf77b919af84a2fbd4d300d8559c02988d801061a15a5faee79795c90ca` |
| `ARM_RESULT_02_C1.json` | 1,093,552 | `99dda0aa06aaf8fbbc425842b54a7f0a73ef66c6023d3d71b779d47929d4e250` |
| `ARM_RESULT_02_C2.json` | 1,137,967 | `3e28ac21c569c815884058ddd9c39f7eefddbd044ca94071331553ee1541dafe` |

Preserve all failures and missing stages. The retained evidence separately shows
an integration-fixture error, a completed context/ownership output refusal, and
an incomplete global CPU-bounded packet. It does not establish native support,
physical regression/improvement, unsafe confidence, cap-filter success onL,
known count or any change in historical/cached F1. Source256PASS remains valid
for its narrower scope; it cannot substitute for this failed connected gate.

Only this exact attempt, its current49 source bindings and already released
PRE were inspected. No original geometry, Gold/catalog/cached predictions,
other attempts, native or scientific execution occurred; no retained byte was
modified. A justified separately frozen successor requires its own review and
authority; this POST supplies no automatic execution release.
