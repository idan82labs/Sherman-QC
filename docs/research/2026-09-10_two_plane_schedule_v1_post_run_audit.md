# Two-plane schedule V1: independent retained post-run audit

Date: 2026-09-10. Decision:
**GREEN_VALID_FROZEN_PURE_TWO_PLANE_SCHEDULE_V1_GATE**.
Blocking findings: **none within the declared pure-scheduler qualification**.

Reviewer: `/root/production_architecture_review`, an AI reviewer independent of
the scheduler author and root-authored test/protocol/harness route. This review
uses the exact retained evidence, not a second test run. It is not a human
scientific or manufacturing approval.

## Exact first execution and retained inventory

Root reports FIRST ONCE execution after exact PRE-GREEN and observed parent
exit zero. The retained child receipt independently records exit zero. Command:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_schedule_v1
```

Terminal attempt:
`docs/research/cad_engine_v1_synthetic_checks/two_plane_schedule_v1-sf_xu7sd`.
Its immediate inventory contains exactly STARTED/OUTPUT/COMPLETED: three regular
files, 2,307 total bytes, with no additional, partial or symlink entry. Each was
read in full and hashed; metadata remained unchanged around each bounded read.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `STARTED.json` | 984 | `62340f0950127634c517dcbf66e9453e492d56311bdef925a025154d15f991ac` |
| `OUTPUT.txt` | 166 | `ade05a0727c3cefc74df7e652eb0c53eff989a61e2c79b1971ee12ea34a41110` |
| `COMPLETED.json` | 1157 | `5f5d5363f4463525000bf624e46116adc32207036c55527e2bdd08982ac5c382` |

The output has exactly 68 passing progress markers, `68 passed in 0.54s`, and one
worker resource JSON line. No failure, skip or xfail is reported. This equals the
17 functions /68 parameterized outcomes independently enumerated before release.
The raw output digest matches COMPLETED's binding, and both suite fields and
the retained-directory field match the exact release and owned attempt.

## Unchanged five-file closure

The [exact pre-audit](2026-09-10_two_plane_schedule_v1_pre_run_audit.md) remains
unchanged at SHA-256
`08b6a68f97f5c9018c9471193496ebb8ba99ee9d31fe9ac429fba98b7881e4a6`.
Its disclosed Boolean-oracle defect was repaired before execution; this attempt
uses repaired test05b7d92b, not the unreleased93ab157b draft. No failing runtime
attempt or retry is inferred from that pre-execution source correction.

STARTED's before map, COMPLETED's after map and independent current-source
rehashing all contain exactly these five entries and identical values:

| Source | SHA-256 |
| --- | --- |
| `backend/cad_guided_two_plane_schedule_v1.py` | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `backend/cad_guided_plane_search_v6.py` | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `tests/test_cad_guided_two_plane_schedule_v1.py` | `05b7d92bda320cd4fea79bde7753afb6415a375a6d669c1b828c7b01e785fcd9` |
| `scripts/check_cad_guided_measurement.py` | `24480e4b18e00a08a9e6efecd654041965c6a2c5a0f3ac10cc8f9f684b741171` |
| `docs/research/2026-09-10_two_plane_schedule_v1_protocol.md` | `355851393b3811e77457df3a913ecb0b480e7c43cbe2ec36c6b1ccb6430dda3e` |

Separate test/harness identity fields match. STARTED has the expected exclusive
synthetic-check status. COMPLETED reports source consistency true; direct map
comparison and current hashes independently confirm this rather than trusting
that Boolean alone. No source or prior terminal evidence was changed.

## Resource and terminal evidence

| Quantity | Retained observation | Frozen limit |
| --- | ---: | ---: |
| Parent monitored wall | 0.6351049593649805 s | 60 s |
| Worker CPU | 0.5986779999999999 s | 45 s |
| Worker reported high-water RSS | 59,359,232 B | 1,073,741,824 B |
| Sampled process-group peak RSS | 59,359,232 B | 1,073,741,824 B |
| Retained output | 166 B | 1,048,576 B |

STARTED and COMPLETED limits agree exactly. Child exit is integer zero,
`budget_stop=null`, and complete positive worker CPU/RSS observations are present.
No CPU termination, declared budget stop, output truncation or source drift is
observed. Root's externally observed parent exit is explicitly separate from
the child exit recorded in COMPLETED.

The existing supervisor limits remain: sampled RSS is not an allocation cap or
continuous-peak proof; some setup/hash/receipt IO and cleanup lie outside the
monitored timer; forwarding/reap are not independently hard-deadlined. This pure
test creates no subprocess of its own and does not qualify escaped native-child
containment. No per-fixture geometry artifact was required by the protocol;
the three retained receipts are its complete declared artifact set.

## Qualified scope and next decision

The frozen invented suite qualifies its bounded schedule contract: independent
rank/cone checks, complete tested angular windows, both target orders, triplet-
first joint slot reservation, explicit starvation/omissions, repeated-layout
fair-bin clocks, and whole-schedule clearing on the tested late failures.
Fairness is only repeated exact-layout exposure at a fixed actual bin count,
not exhaustive correspondence search, per-plane completeness or a true-pose
coverage guarantee. Small fixtures do not prove maximum2048-CAD runtime or memory
performance. Joint triplet counts and plane/support provenance remain caller
declarations requiring authentication by a connecting adapter.

This is not a pose, finite-support, registration, measured-bend or F1 result.
It permits proceeding with separately reviewed connected qualification only.
Actual byte-bound supports, two-support finite screening, joint consumer caps,
unchanged global refinement/rank/ambiguity checks and original .05-mm/free-circle
output oracles remain separate requirements. This audit does not release V11,
the companion terminal-artifact route, native/physical execution, calibration,
unseen-family generalization or product promotion. Existing physical negatives
and their custody findings are unchanged.

Review operations were limited to the exact three owned receipts and their
immediate inventory, five current sources, unchanged PRE, and stdlib hash/JSON
binding comparisons. No test collection/import, scheduler/native/inference/scorer
execution, original raw/CAD/cache/Gold access, retry or cleanup occurred. Only
this new post-run audit was authored; the successful unchanged gate is terminal.
