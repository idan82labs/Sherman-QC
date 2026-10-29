# V6 connected workflow: retained nominal regression, no physical rerun yet

Date:2026-09-09. Independent post-run audit confirms a valid terminal negative.
The earlier V4 constructed3/3 remains unchanged. No physical scan/CAD/Gold used.

## Frozen execution

[Pre-run audit](2026-09-09_mesh_connected_planes_v6_pre_run_audit.md) binds48
source/protocol/test/harness files. Source prerequisite
`mesh_connected_planes_v6_source-1uzfu9rl` passed35/1declared native skip.
The single native attempt `mesh_connected_planes_v6-58gybuu9` then completed
35pass/1failed pytest check: the actual four-case matrix passed only1/4.

Parent6.385630667209625s,workerCPU5.567089s,sampled and workerpeak436,977,664B.
No180wall/120workerCPU/4GiB/1MiB stop and all48 hashes matched before/after.
OUTPUT SHA256 `bbc09f61000b956204a09ed9395012a3066f6bf8ccfae5e254a5fa058a232571`.
All four complete exclusive artifacts are retained:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| case_00.json | 4826196 | `cf582f479ea7a5b14d06f1d0da784006e6c3ef82d326a0bb13960f0c292200f3` |
| case_01.json | 4506058 | `7a3e27781fc963d4d06808ee49a0ccad782d545ebd3b8a38c455f13d19e56547` |
| case_02.json | 4824158 | `c2569bd696347437e95e9c04ad4b19d947c9ec754799ce7148ee30e2c4d51670` |
| case_03.json | 4555310 | `e7283148910bbea88493ca47cb6b728457f16d5b6746553301d470d3a63c3e8a` |

Total18,711,722B. Independent Astra verified every file/hash/size, exact four-file
artifact set, OUTPUT digest, source brackets and budgets. Exit1 is the matrix
assertion, not a native crash/resource stop or concealed partial execution.

## Results and attribution

Both original nominal cases and the new noisy nominal produce zero predictions:
no scheduled plane pose passes the assigned finite-support gate. The missing-arc
case passes only its predeclared complete-prefilter abstention contract, not an
observed-absence claim. Native feature metadata is AVAILABLE with one feature in
all four cases. No refinement or local measurement ran in any case.

| Nominal case | Best generated all-row pose error mm | Best scheduled mm |
| --- | ---: | ---: |
| native V1 clean | 1.4142135623730825 | 1.4142135623731202 |
| native R0 common-wrapper clean | 1.4142135623730825 | 1.4142135623731202 |
| native V1 noisy | 1.4151783228480563 | 34.9265725328645 |

No generated pose, including omitted candidates, meets the unchanged.20mm
action target. Therefore simply increasing the outer32-unsigned prefilter quota
cannot recover an already demonstrated correct pose. Every case has128 scheduled
signs,384 fully evaluated assigned supports, no numerical failures, zero passing
groups and complete search/query/error ledgers. The clean best scheduled pose's
two main supports pass, but its cap support has only5/15 inliers versus.95 required;
the finite gate correctly refuses the displaced pose. Do not loosen that gate.

Root and independent Astra trace repeated inner search starvation:94 clean joins
and88 noisy joins with60 compatible CAD tuples always choose positions
[0,19,39,59], repeating the same four assignments. Compatible inner-plane tuples
are visible in retained masks but unscheduled. The earlier12-pair sampling also
omits CAD pairs; neither all-pair coverage nor recovery is established.
All56 eligible clean scan planes entered the bank; bank exhaustion is not the
identified issue in this construction.

## Narrow justified successor, not a rescued run

Preserve the entire V6 execution closure/test/protocol. A generic deterministic
fair-bin schedule across repeated exact pair windows and compatible CAD layouts
is the next source-only hypothesis. It must keep12pairs/4unsigned per scan triplet,
1024unsigned/8192signed hard caps, all geometry/refusal/measurement gates and all
support provenance. Exact repeated-layout fairness is not pose recovery,
exhaustive matching, label-independent calibration or ID-renumbering invariance.
Independent integer coverage proof, adversarial tests and a new frozen connected
candidate precede execution. No geometry or desired answer may select a rank.

The separately prepared physical V2 runner is only an unexecuted source draft.
The all-eleven-case Gold11 baseline remains next after this connected gate and
its complete physical freeze/audit. No new F1, promotion or floor readiness.
