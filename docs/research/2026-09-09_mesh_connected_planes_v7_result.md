# V7 fair-search connected result — 2/4, not promotion-ready

Date: 2026-09-09. First source and native executions under the exact
[58-file pre-run audit](2026-09-09_mesh_connected_planes_v7_pre_run_audit.md).
The full connected gate remains **FAILED**: two cases pass and two required
clean nominal cases fail. No physical Gold11 run or product promotion follows.

## Prerequisites and terminal resource evidence

Search115 passes were independently post-audited in
[the search result](2026-09-09_plane_search_v6_result.md).
V7 initializer `plane_initialization_v7-wyjtifck` passes43/43, no skips,
35 source hashes unchanged, no budget stop. Parent1.0590830002911389 s,
CPU1.0280159999999998 s, sampled86,786,048 B, worker88,686,592 B.
Its OUTPUT SHA-256 is
`0fc0013b408b3815c450138dfe7c92495d9dc571ddeade599db7277882aa4b4a`.
Both mandatory actual noisy-array poses and unchanged1e-7/.01/.011 mm criteria
pass, including V6-search seam assertions and V6 idempotent-refusal preservation.

Source `mesh_connected_planes_v7_source-_kjmmq4h` passes35 controls with one
declared native skip. Parent0.3858119579963386 s, CPU0.35646999999999995 s,
sampled94,011,392 B, worker94,486,528 B, no stop/58-file drift.
OUTPUT `d5cb16c090444bd2e88b8cb39689080dfe558316f2b86566d6ebe0d68f92f3cb`.
Both prerequisite receipts were independently verified by
`/root/astra_repo_guidance_review` without re-execution.

Full terminal attempt: `cad_engine_v1_synthetic_checks/mesh_connected_planes_v7-n1y2vlbi`.
Pytest35pass/1fail corresponds to actual matrix2/4, not35 successful native cases.
Parent8.090838333126158 s, CPU7.754568 s, sampled/worker peak449,937,408 B.
No180 s wall/120 s CPU/4 GiB RSS/1 MiB stdout stop; all58 hashes unchanged.
OUTPUT `4a7c1f5372e56b8eb3bfd73a3ec891c466fd7b70f205d3417ebfd63bd1bbf700`.

All four complete JSON artifacts are retained, total19,055,129 bytes:

| Case artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `ARTIFACTS/case_00.json` | 4911765 | `61e39ea298f761591e7ac90b2d30948d9c72542cde08f564dfb706d0e7dae322` |
| `ARTIFACTS/case_01.json` | 4573669 | `cad710c468c7997a3478d6239d639829a3723f746790359a97c0a5c87c28bfdd` |
| `ARTIFACTS/case_02.json` | 4909727 | `f03bcaa31b8a903713ecf78f7eb45d5ca36b661f2f6fa42fae611eeabbb67601` |
| `ARTIFACTS/case_03.json` | 4659968 | `a08eb4b91aae7f9d60b8190b1a6423cdd30d1e47df4025560b36d6a573ab38d1` |

## Actual output and narrow before/after

| Fixed case | V6 | V7 | V7 disposition |
| --- | --- | --- | --- |
| V1 clean nominal | FAIL | FAIL | One wrong finite start; global coverage refusal, no measurement |
| V1 missing arc | PASS | PASS | Completed nonempty geometric finite-prefilter abstention, not observed absence |
| r0 extractor/common-wrapper clean nominal | FAIL | FAIL | Same wrong start/global refusal; not frozen R0 pipeline |
| V1 bounded 3D noise nominal | FAIL | PASS | One actual scan-derived finite bend, accurate registered pose |

The noisy positive now reaches full raw measurement under unchanged gates:
radius error0.00007097692126478705 mm; sweep error0.0002996216673238905 degrees;
generator endpoint error0.000761224448477113 mm; axis endpoint error
0.0007557403158828327 mm; all-row pose-action error0.002069763395236448 mm.
This is genuine constructed engine-output progress, not physical F1 or calibrated
uncertainty. All three nominal cases require output; two still fail.

Every nominal retained one exact passing group/refinement and completed it with
zero numerical failures. Each case screened128 proposed signed hypotheses,
checking384 complete assigned supports. Missing arc retained no passing group.
Detailed search omissions and all wrong hypotheses remain in the artifacts.

## Retained-only failure diagnosis

Unlike V6, V7 now generates essentially exact clean poses. Case00 hypothesis31
has maximum all-row action error4.451537480371452e-14 mm; hypothesis255 also
has error approximately5.49e-14 mm. Both are omitted by the outer32-unsigned
rank schedule (unsigned IDs3 and31). These post-inference identifiers diagnose
omission only; they must not become inference selectors or case-specific rules.

The best scheduled clean pose is hypothesis2343 at1.0000000000000178 mm and
is finite-rejected. The sole finite-passing hypothesis2619/unsigned327 is a
wrong locally support-consistent permutation. Its fully refined all-row action
error is56.55458219859193 mm. Global scan coverage0.8859011627906976 falls below
the unchanged0.95 gate, so registration correctly refuses it for
`INSUFFICIENT_FINITE_SURFACE_SUPPORT`; measurement never runs. This is not a
measurement-kernel failure and does not justify relaxing coverage.

The noisy case's selected2359 initial error0.0024916876751239227 mm improves to
0.002069763395236448 mm through unchanged refinement and yields the positive.

Root and independent `/root/cad_measurement_integration_review` agree on the
outer scheduling failure from retained-only evidence. No hypothesis was rerun,
new pose generated, data changed or oracle fed back into inference.

Next justified mechanism under independent review: remove blind outer rank
omission by finite-screening every proposed sign from the already bounded
generated search, retaining existing finite-query/work/CPU/RSS/report hard stops
and48 exact-group refinement cap. This requires a new source/protocol/test
successor, not a V7 retry or a favorable rank shift. Its resource feasibility and
whole-obligation refusal must be reviewed before implementation/execution.

Independent artifact post-audit by `/root/astra_repo_guidance_review` is complete:
all four file hashes/sizes, output and58 source hashes match; all3008/2880/3008/
2816 hypothesis/action-error ledgers align, each selected sign has three complete
supports, query plans equal actual work, and exact passing-group origins partition
correctly. The noisy positive's1073 arc and260/263 flange support rows belong to
the full2752-row measurement contract. Its actual fitted radius4.999929023 mm and
sweep89.999700378 degrees agree with the retained error report. Missing arc also
contains eight near-exact generated but omitted poses: its current abstention is
not proof of observed absence. Scoped audit: valid terminal2/4 negative with a
real constructed noisy positive, not a completed accuracy gate.

The physical V2 runner remains unfrozen/source-only (currently targets V7); it
cannot proceed under this failed connected gate. Preserve all V7 source/test/
protocol/harness bytes. The exact used harness874 is archived under
`source_snapshots/fair_search_v7_87478a2c/`; subsequent additive routes do not
change the terminal receipt or archived execution bytes.
