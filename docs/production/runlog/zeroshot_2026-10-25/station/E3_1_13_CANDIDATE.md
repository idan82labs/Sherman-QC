# E3 1.13.0-rc candidate: short-panel skin fix and resample safety net (2026-10-26)

> **Research candidate. Nothing was applied.** The live engine is still `e3-1.12.0`. Its inspect fingerprint is `5fbed707…`, checked before and after every run. No contract, eval, ledger or pin file was edited, and nothing was installed or committed.
>
> The candidate is a patched copy in a scratch shadow tree, run through the same CLI entry. The shadow tree symlinks every other part of the repo.
> - Candidate fingerprint: `46daefac…`.
> - Patch: `e3_1_13/e3_1_13_0_rc.patch`. It changes 3 inspect files and adds 1 test, and it dry-run-applies cleanly to the live tree.
> - Machine-readable results: `e3_1_13/results.json`. Tools and configs: `e3_1_13/tools/`.
>
> Heavy runs went through `sherman_slot.sh`, with at most 2 slots of mine and `OMP_NUM_THREADS=1`. Peak RSS stayed under 3 GiB.
>
> Inputs were the 88 survey scans only. No FRESH, DEV-CHECK, TEST-B, N1, HOLDOUT10, EHO4 or SEALED input was used. G1 truth was read only by the scorer.

## Answer first

| Arm (each vs e3-1.12.0, 88 scans × 7 realizations) | chars FAIL↔PASS | chars FAIL→PASS | scans FAIL↔PASS | coverage loss, native (of 959 decided) | G1 false PASS | E3 wall P50 / P95 / max |
|---|---|---|---|---|---|---|
| e3-1.12.0 (survey) | **11** | **8** | 2 | – | 0 | 8 / 29 / 35 s |
| **root fix only** (b) | **0** | **0** | 0 | **0.63 %** (6; physical 0.93 %, G1 0) | 0 | 8 / 29 / 35 s |
| **root + safety net** (a, K = 2) | **0** | **0** | 0 | 1.46 % (14; physical 2.17 %, G1 0) | 0 | 24 / 87 / 102 s |
| root + honest U (c, K = 3), comparison arm | **0** | **0** | 0 | 0.94 % (9; physical 1.40 %, G1 0) | 0 | 31 / 116 / 132 s |

![verdict stability](e3_1_13/e3c_verdict_stability.png)

- **The root fix alone removes all 11 FAIL↔PASS characteristics.** None of the arms creates a PASS on any of the 616 runs.
- **23553000 B now lands in one mode: a consistent FAIL.**
  - On the full scan, B5 flange B, B6 A/B and B7 A are FAIL in all 7 realizations; e3-1.12.0 had 2 FAIL and 5 PASS. The values are 19.39–19.53, 72.68 and 72.73, against native 19.46 / 72.68 / 72.71.
  - On scan 2, the two-branch characteristics (B5 A/B, B6 B) are REVIEW in all 7 realizations.
  - The summary is OUT_OF_TOLERANCE_FOUND in every realization of both scans.
- **Native verdicts on the other 86 scans.** Under the root fix alone, 4 native characteristics change on 3 other scans, all decided → REVIEW and explained below; no value moves. The native results of the 2 B scans change too: 1 characteristic on the full scan and 2 on scan 2.
- **G1 is unchanged for every arm.**
  - 314 of 314 native decided verdicts are correct, with 0 false PASS and 0 false FAIL.
  - Across the 1,980 perturbed G1 verdicts, the arms decide 1,884 (root), 1,882 (net) and 1,883 (c), all correct.
  - The frozen G1 regression tier gives regression PASS with every gate value identical to e3-1.12.0 except runtime.
  - Frozen A1 recompile: 18/18 hash-identical. A1 physical: every G2/G3 gate, GOLD_SET_V1, BROAD and per-bend V3 result is identical.
  - **One exception, the runtime cap.** With the safety net on (the promoted configuration), the same checks pass except **A1 G2(e): P95 89.5 s against a +25 % cap of 35.7 s, FAIL by design**. G1 RUNTIME.P95 rises to 42.3 s. That cap needs an owner decision before the bump.
- **New finding: a second mechanism that the root fix does not reach.** On B full, resamples of the s95_9504 realization (about 90 % density) land on a different **registration pose**:
  - total 1.275 against 1.431, pose margin 0.044;
  - 0.63° / 0.95 mm off native;
  - four ADJACENT_BEND lengths read PASS (73.5–73.7) instead of FAIL (72.6–72.7).

  This happens identically on e3-1.12.0 and on the candidate. The 7 survey realizations never hit it, so the root-only arm scores 0. Both safety-net arms withhold it as REVIEW. **That is why the safety net is still needed.**

**Recommendation: promote root fix + safety net (a), K = 2, serial**, as `e3-1.13.0`. The exact apply steps are at the end. Option (c) loses less coverage (0.94 % against 1.46 %), but it costs 4× instead of 3× and comes within 48 s of the hard stop on the largest surveyed scan. Keep it as the fallback if coverage matters more than budget.

## What was built (`e3_1_13_0_rc.patch`)

All three parts are general geometry. E3 still sees no filename or metadata, and no part, SKU or family logic was added. Each part has its own config flag.

### 1. One-skin swap in the station re-seat (`one_skin_swap`, root fix)

**Where:** `metrology._reseat_plane`.

**When it applies.** The side's panel region holds **one skin** (`PanelFit.skins_seen` has length 1), and the bend's station strip shows **two parallel skins one measured thickness apart** (`skin_peaks`, separation t ± 0.25 t). Then one of two things is true:
- the plane sits on the inner skin, so the OML level is near +t;
- the plane sits one thickness outside the outer skin, so the level is near −t.

**What it does.** The plane is re-seated on the outer skin, a shift of ±t. The share guard then counts that two-skin band instead of the whole strip.

**Why e3-1.12.0 refused the re-seat.** The strip also holds the arc and other sheets, so a skin was always below 25 % of the strip. That refusal is exactly why e3-1.12.0 kept the wrong skin.

**What evidence decides which skin it is:**
- the measured thickness: the datum's two-skin t, and the second skin found one t away;
- the neighbouring bends' strips. A panel between two bends is re-seated independently on each side: on B full, the B5 parent side moves +t and the B6 child side moves −t.

The tilt guard and the 3-pass refit are unchanged.

![B5 section](e3_1_13/e3c_B5_section_before_after.png)

The figure shows B full, seed 9501. The short panel's region holds only its inner skin.
- e3-1.12.0 puts the OML plane on that skin: red, with material one thickness *outside* it.
- The candidate moves it onto the outer skin: blue, matching the two-skin native plane (dotted green).
- Lower row: the same panel seen from B6, where its OML is the other contract skin, so the plane sat one t *above* all material and moves in.

### 2. Near-tie withholding (`branch_withhold`, `branch_tie_ratio` = 0.75; part of the root fix)

Three discrete skin and window choices record a **near-tie** when the two candidates' supports are within 0.75 of each other:
- **`_reseat_plane`:** the densest level against a second, non-partner level peak at least 0.3 mm away. This is the B scan 2 branch: two layers 1.4 mm apart, not a sheet pair.
- **`_skin_level`:** the target level against a distinct densest level.
- **`_flat_start`:** the two skins' counts, counted only when the other skin's flat start differs by at least 2 steps.

**How a near-tie is resolved.** A bend that met a near-tie is measured a second time with every such choice taken the other way. Its characteristics are measured again with that alternative geometry (`engine._branch_withhold`).

**What gets withheld.** A **decided** characteristic is withheld only when the alternative reading decides differently. It becomes REVIEW: U gets |Δ| added, and is at least the guard-band width. Characteristics that both readings decide the same keep their reading, so the withholding never creates a PASS.

**Cost:** about 0.25 s per tied bend.

### 3. Resample safety net (`resample_check_k`, `resample_check_policy`, and related flags)

**Sequence:**
1. The primary inspection is built but not written.
2. K deterministic 95 % subsamples of the ingested points (seeds 7101, 7102, 7103) are inspected the same way, within what is left of the budget.
3. The primary is rebuilt and written.

**Policy 1, option (a).** A PASS or FAIL that any resample decides otherwise becomes REVIEW: U = U + spread, at least the guard band. The same applies when a resample is UNKNOWN, did not run or is not VALID.

**Policy 2, option (c).** U' = √(U² + (2σ)²). A resample without a value withholds.

**Fail-safe.** A resample that the budget cannot hold is not started. That leaves the check incomplete, which withholds every decided characteristic (warning `RESAMPLE_CHECK_INCOMPLETE`). Neither policy can create a PASS, because U only grows.

**Optional settings:**
- `resample_check_parallel=1` runs the resamples in forked worker processes, not threads, so no state is shared. Its readings are byte-identical to the serial ones.
- `resample_check_record=1` writes `resample_check.json`, the per-resample readings used by this evaluation.

The rc ships with the net **off** (`resample_check_k=0`). The promoted config turns it on (see the apply steps).

## Method

**Protocol.** Exactly the survey protocol:
- the same 88 unique scans: G1 50, A1 26, DEV cohort 33, with 21 shared;
- `pc_b4719db3dd69` refused (compile BLOCKED, rc 2), as before;
- realizations: native, `subsample.py` 95 % seeds 9501–9505, and `thin_exp.py` 0.40 mm;
- the production limits of 180 s / 170 s, at 1 thread.

Altogether 616 candidate CLI runs (`tools/sweep.py`, 2 workers). The e3-1.12.0 arm is the survey's records.

**One run serves all arms.** Each run used `resample_check_k=3`, `policy=2` and `record=1`, so one CLI invocation per realization yields every arm:
- **root:** the primary readings before the check;
- **net (a, K = 2):** policy 1 over the first 2 resamples, applied offline. The offline policy reproduces the in-engine K = 2 result exactly on the corpus-max scan, serial and parallel.
- **c (K = 3):** the written result. An offline recompute agrees on all 595 VALID realizations; the 21 non-VALID primaries run no resamples.

**Coverage loss** is the native characteristics that e3-1.12.0 decides (PASS/FAIL, 959) and the arm does not. Over all 7 realizations the figures are: root 46/6,692 (0.69 %), net 109/6,692 (1.63 %), c 78/6,692 (1.17 %).

## Results by arm

### Stability

| | e3-1.12.0 | root | root + net (a) | root + U (c) |
|---|---|---|---|---|
| chars FAIL↔PASS / FAIL→PASS (all) | 11 / 8 | 0 / 0 | 0 / 0 | 0 / 0 |
| scans FAIL↔PASS | 2 (both 23553000 B) | 0 | 0 | 0 |
| chars with any decision change (of 1,619) | 55 | 45 | 61 | 62 |
| G1 / physical chars with any change | 2 / 53 | 2 / 43 | 2 / 59 | 2 / 60 |
| native decided → withheld (G1 / physical) | – | 0 / 6 | 0 / 14 | 0 / 9 |
| decided → decided flips on native | – | 0 | 0 | 0 |

**Where the remaining changes are.** The arms reach 0 FAIL↔PASS, but other decision changes remain:
- **Root only:** they are PASS↔REVIEW or REVIEW↔FAIL.
- **Root + safety net:** a perturbed realization can still read UNKNOWN or REVIEW where the native decides.
- **G1's 2** are the survey's own: ZS-006 UNKNOWN↔FAIL and TD-001 FAIL↔REVIEW. Neither is ever a wrong decided verdict.

**Root against e3-1.12.0, per realization** (all 7, `results.json: realization_changes_root_vs_base`): 72 changes, all on A1 and cohort scans.
- **PASS→FAIL 24, FAIL→PASS 1, REVIEW→PASS 1, all on B full.** They are the perturbed realizations returning to the native mode:
  - B5 B, B6 A/B and B7 A in s1, s2, s3, s5 and thin.
  - In thin, also the other short panel: B2 B, B3 A/B and B4 B. B2 A reads 9.94 (PASS) and so matches its native PASS.
  - s5 B5 angle 60.10 (PASS), as the native reads 59.91.
- **The remaining 46 are decided → REVIEW by near-tie withholding.** Values are unchanged; only U widened.

### G1 (50 cases; truth read by the scorer only)

| | e3-1.12.0 | root | net | c |
|---|---|---|---|---|
| native decided / correct / false PASS (of 330 scored) | 314 / 314 / 0 | 314 / 314 / 0 | 314 / 314 / 0 | 314 / 314 / 0 |
| perturbed decided / correct / false PASS (of 1,980) | 1,884 / 1,884 / 0 | 1,884 / 1,884 / 0 | 1,882 / 1,882 / 0 | 1,883 / 1,883 / 0 |

### Which mode 23553000 B lands in

**Full scan (`sc_6eba02723d4b`), root fix:**
- B5 flange B, B6 flange A, B6 flange B and B7 flange A are **FAIL in 7 of 7**. In e3-1.12.0 they were FAIL in 2 and PASS in 5.
- B5 flange A is REVIEW in 7 of 7; its native value was REVIEW too.
- The native result is unchanged except B11 angle, which goes FAIL → REVIEW (a near-tie).
- **Safety net:** it also withholds the s95_9504 realization's B6 A and B7 A to REVIEW, from its pose-mode resamples (next section).

**Scan 2 (`pc_186463746daa`), root fix:**
- The station-branch characteristics (B5 flange A, B5 flange B, B6 flange B) are **REVIEW in 7 of 7**. In e3-1.12.0 they were PASS/REVIEW/FAIL depending on the realization.
- B5 angle is REVIEW in 5 of 7 and FAIL in 2.

  The withholding is honest here, because no truth says which branch is right. The two layers in the B5 strip are 1.4 mm apart: neither is the other's partner skin.

**Summary:** OUT_OF_TOLERANCE_FOUND on all 7 realizations of both scans, in every arm.

### The second mode on B full (registration, not skins)

**What was seen.** The net's resamples 7101 and 7103 of the s95_9504 realization chose a datum pose:
- 0.63° and 0.95 mm from native;
- best total 1.275 against 1.431, datum score 0.586 against 0.724, pose margin 0.044.

Under it, B3 B, B4 B, B6 A and B7 A read 73.5–73.7 mm (PASS) against 72.6–72.7 (FAIL), and several bend-line positions move about 1.5 mm.

**Confirmation.** That resample was re-created (`subsample.py` 9504, then the engine's own seed-7101 subsample) and run on both engines. e3-1.12.0 and the candidate give identical results on it.

**What it means.** A ~90 % density realization of this scan can register into a worse pose, and lengths then flip to PASS. The root fix cannot see this. The 7-realization protocol itself never produced it, so the root-only arm's 0 is a lower bound.

**Status:** unverified beyond this one scan. It was not searched for on the other scans.

### Native verdict changes vs e3-1.12.0, every scan, each explained

All changes are decided → REVIEW. No value changed, and no PASS was created.

| Scan | Characteristic | 1.12 | root | net | c | Why |
|---|---|---|---|---|---|---|
| pc_186463746daa (B scan 2) | B5 flange B | PASS 18.523 U0.114 | REVIEW (U0.373) | REVIEW | REVIEW | reseat-level near-tie: the alternative layer reads 18.78. This is the survey's PASS↔REVIEW branch. |
| pc_186463746daa | B5 angle | FAIL 61.692 U0.553 | REVIEW (U0.693) | REVIEW | REVIEW | the same near-tie: the alternative reading's own U puts it in the guard band |
| pc_186463746daa | B9 flange B | PASS 19.336 | PASS | REVIEW | PASS | 2 resamples read 19.337 with U 0.164–0.181, just past the 19.5 limit (REVIEW); the survey's PASS→REVIEW |
| pc_6eba02723d4b (B full) | B11 angle | FAIL 61.602 U0.563 | REVIEW (U0.603) | REVIEW | REVIEW | flat-start near-tie: 0.04° from the guard band; the survey's FAIL→REVIEW |
| pc_6eba02723d4b | B5 angle | PASS 59.912 | PASS | REVIEW | PASS | all 3 resamples read 59.99–60.13 with U 0.96–1.01 (REVIEW). The survey had it PASS↔REVIEW. |
| pc_6eba02723d4b | B16 flange A | FAIL 460.300 | FAIL | REVIEW | REVIEW | all 3 resamples read 460.43–460.45 (REVIEW); the survey's FAIL→REVIEW |
| pc_cec08dfca9ab | B7 angle | PASS 89.567 U0.521 | REVIEW (U0.723) | REVIEW | REVIEW | near-tie: 0.09° inside the guard band. The survey listed the angles on this scan as changing. |
| pc_cec08dfca9ab | B12 flange A | PASS 41.010 | PASS | REVIEW | PASS | 1 resample reads 40.973 with U 0.287 (REVIEW); nominal 41.20 ± 0.5 |
| pc_71cf0b216f43 | B9 line position | PASS −0.206 U0.215 | PASS | REVIEW | PASS | 1 resample reads −0.222 with U 0.279 (REVIEW). The survey listed it PASS↔REVIEW. |
| pc_38091ac91d9d | B1 line position | PASS 0.204 U0.235 | PASS | REVIEW | PASS | 1 resample reads 0.270 with U 0.250 (REVIEW); the survey's PASS→REVIEW |
| pc_d1732de18023 | B1 line position | PASS 0.029 U0.469 | PASS | REVIEW | PASS | 1 resample reads 0.042 with U 0.469 (REVIEW); the survey's PASS→REVIEW |
| pc_70b39f530f00 | B2 line position | PASS 0.368 U0.109 | PASS | REVIEW | REVIEW | the resamples read 0.13 / 0.41 / 0.18: a 2.6·U spread (the survey's 3.3·U case) |
| sc_8026eedce786 | B7 angle | PASS 150.492 U0.497 | REVIEW (U0.509) | REVIEW | REVIEW | near-tie: 0.01° inside the guard band; the survey's PASS→REVIEW |
| sc_c27168a2de83 | B2 angle | PASS 127.264 U0.431 | REVIEW (U0.470) | REVIEW | REVIEW | near-tie: 0.04° inside the guard band (new; this scan was stable in the survey) |
| sc_822f083e3dfc | B2 flange A | PASS 14.649 | PASS | PASS | REVIEW | the 3rd resample (seen only by c) jumps to 10.84 (U 3.3), the survey's 6.4 mm short-flange family; net K = 2 does not see it |

The swap itself fired on **no native run** of any scan. It only fires where a one-skin region coexists with a two-skin strip.

### Wall time and memory

**Sweep (2 workers, host shared with other agents; arm wall times):**

| Arm | P50 | P95 | max | Peak RSS |
|---|---|---|---|---|
| e3-1.12.0 (survey) | 8.1 s | 28.7 s | 35.3 s | 0.93 GiB |
| root | 7.7 s | 29.3 s | 35.1 s | 1.19 GiB* |
| root + net (a, K = 2), serial | 23.7 s | 86.7 s | 102.2 s | 1.19 GiB* |
| root + U (c, K = 3), serial | 31.2 s | 115.5 s | 131.7 s | 1.19 GiB |

\* Read from the K = 3 runs, which hold the resamples' peak.

**Corpus-max scan `pc_2714fb7e0c0b`, one run each:**

| | wall | peak RSS |
|---|---|---|
| root | 34.0 s | 0.72 GiB |
| net K = 2, serial | **98.6 s** | 0.89 GiB |
| c K = 3, serial | 126.8 s | 0.91 GiB |
| net K = 2, parallel (3 processes) | 68.7 s | 1.80 GiB |
| c K = 3, parallel (4 processes) | 70.0 s | 2.67 GiB |

**Against the 180 s hard stop (170 s budget):**
- Every surveyed scan fits K = 2 serial, with 70 s margin at the maximum. K = 3 serial leaves 48 s.
- Parallel costs about 2× the primary: the resamples start after it.
- A dense 40 M-face scan (140–160 s native, DENSE_READINESS) cannot hold any resample. The fail-safe then withholds every decided characteristic. **The safety net needs the dense thinning step (or parallel workers) before dense scans go through it.**

## Frozen regression checks (the version-bump checklist)

Each check was run twice: on the **root-fix** candidate (safety net off), and on the **promoted** configuration (root + net, `resample_check_k=2`, policy 1, serial, `tools/cfg_promote.json`).

**Frozen A1 recompile** (`prefreeze compile-frozen`): **18/18 match the freeze**, 0 mismatches; status counts COMPILED 17, BLOCKED 1.

**A1 physical** (`run-physical`, `score-physical` with the anchor, then the iter_7 and fixes3 analysis tools):
- Every G2 and G3 gate is **identical** to e3-1.12.0 (`fixes3/followup/a1/keynums_fu2.json`). G2(a) FAIL and G2(f) FAIL are as before.
- GOLD_SET_V1: 0.990 / 0.974. BROAD: 0.702 / 0.817. Gold11 relabel: 0.987.
- Per-bend V3: 192/203, strict 176, family-macro 0.938, false-PASS proxies 3 (the same 3 bends).
- 4 characteristic changes, the near-tie REVIEWs listed above.
- G2(e) runtime P95 rose from 28.35 to 33.29 s (gate PASS). That is host contention: native root walls equal the survey's elsewhere.

**G1 regression tier** (`zsprod/e2_regression_card`):
- Regression verdict **PASS**, no failed gate.
- **FALSE_PASS 0/553**, WRONG_OR_MIRRORED_NEVER_PASS 0/32.
- 29 of 32 gate values are identical. Only RUNTIME.P95 (13.8 → 15.2 s), RSS and TIER_BUDGET moved.
- The scorecard is not qualifying, with the same 4 reasons as the e3-1.12.0 scorecard, including FREEZE_LEDGER_MISMATCH.

**The promoted configuration (root + net, K = 2), the same checks:**
- **A1 physical.** Every G2 and G3 gate is identical **except G2(e) runtime: FAIL.** P95 is 89.5 s and the maximum 102.1 s, against the +25 % cap of 35.7 s. GOLD_SET_V1, BROAD, Gold11 and per-bend V3 (192/203, the same 3 proxies) are identical.
- **A1 characteristic changes against e3-1.12.0: 12, all decided → REVIEW.** They are the A1 rows of the native table above:
  - 4 from near-ties;
  - 8 from the resample check (`RESAMPLE_CHECK_WITHHELD`).

  This is exactly what the offline net arm predicted, which confirms the in-engine K = 2 path.
- **G1 tier.** Regression **PASS**, no failed gate, FALSE_PASS 0/553, WRONG_OR_MIRRORED 0/32. Every accuracy and decision gate is identical. RUNTIME.P95 is 42.3 s (NOT_EVALUABLE in this tier), RSS 1.05 GiB, and TIER_BUDGET 1,115 s (PASS).

**Named tests on the candidate tree:**
- 15 files: inspect a1 … iter7, cli, units, fixes, fixes3, ingest, contract_parity, eval_scorer and freeze_physical. **289 passed, 1 skipped, 1 failed.** The failure is `test_prod_engine_fixes3_2026_10_19::test_d4_a_…`, which asserts `engine_version == "e3-1.12.0"`; it is expected and gets updated with the bump.
- New `tests/test_prod_engine_e3_1_13_candidate.py`: **13 passed**. It covers:
  - the swap both ways;
  - that e3-1.12.0's refusal is reproduced;
  - no swap on a two-skin panel or with the flag off;
  - that withheld U is always REVIEW;
  - that U-widening never makes a PASS.

## Recommendation and apply steps (for a later owner-approved re-pin)

**Promote root fix + safety net (a), K = 2, serial, as `e3-1.13.0`.**

**Why this arm:**
- It is the only arm that is both root-correct (B is a consistent FAIL) and covered against the registration-pose mode found here.
- It costs 3×, with 70 s margin at the corpus max.
- Its rule is the simplest to audit: any disagreement → REVIEW, which never makes a PASS.
- Its price is 1.46 % of decided characteristics (2.2 % physical, 0 % G1), all of them within about U of a limit.

**Why not the others.** Option (c) has 0.5 points less loss but costs 4× and leaves 48 s margin; keep it as the documented alternative (`resample_check_policy=2`, `k=3`). Root-only is the fallback if the budget cannot take 3×, but it leaves the pose-mode false PASS uncovered.

**Blocking item:** the A1 G2(e) runtime gate (+25 % cap) fails at 3× by construction and needs an owner waiver or amendment. If the owner will not accept 3×, promote root-only now and add the net later.

Bundle with the ENGINE_IO-A5 re-pin (32 GiB) and zs_lane_v3 rev3 (`v3/REPIN_PLAN.md`, "The re-pin event") as one change set, after the owner approves it.

1. **Preconditions.**
   - Owner approval.
   - The fresh-scan gate in REPIN_PLAN step 1, if the bundle includes the zs flip.
   - Live fingerprints unchanged: inspect `5fbed707…`, compile, eval and contracts as in ledger seq 15/16.
2. **Apply.**
   - From the repo root, run `patch -p1 < docs/production/runlog/zeroshot_2026-10-25/station/e3_1_13/e3_1_13_0_rc.patch`.
   - In `domains/production/engine/inspect/config.py`, set:
     - `ENGINE_VERSION = "e3-1.13.0"`;
     - the defaults `resample_check_k: int = 2` and `resample_check_policy: int = 1`; keep `resample_check_parallel = 0` and `record = 0`.
   - Update the fixes3 test's version assertion, and the candidate test's version and `resample_check_k` default assertions.
   - Expected inspect fingerprint: `46daefac…` plus that config edit. Record the new value.
3. **Tests (named only):** the 15 files above plus `tests/test_prod_engine_e3_1_13_candidate.py`. Expect all to pass.
4. **Frozen recompile:** `prefreeze compile-frozen`. Expect 18/18.
5. **A1 physical:** `run-physical`, then `score-physical --ledger-anchor cf01e709…`, then keynums and per_bend_all_v3. Expect:
   - every gate identical;
   - the per-characteristic changes listed in this report: 4 from the root fix plus the net's REVIEWs;
   - G2(e) P95 about 3×: measured 89.5 s, maximum 102.1 s, within the 180 s engine limit. **G2(e)'s +25 % runtime cap FAILs by design (measured).** It needs an owner decision, a waiver or threshold amendment, before the bump.
   - The 12 characteristic changes listed in this report.
6. **G1 tier:** `python -m domains.production.engine.eval run --tier-dir …/zsprod/e2_regression_card --ledger docs/production/runlog/freeze/LEDGER.json --ledger-anchor cf01e709…`. Pass criteria:
   - regression PASS;
   - FALSE_PASS 0/553;
   - WRONG_OR_MIRRORED 0/32.

   RUNTIME.P95 rises to about 3×: measured 42.3 s, against 15.2 s for root only. That is close to the 45 s figure; it is NOT_EVALUABLE in this tier, but it binds elsewhere. If it needs to come down, run the net only on scans with a flange under 25 mm: the survey's gated variant, about 30 % fewer runs. Alternatively enable `resample_check_parallel=1` on hosts with free cores (about 2× instead of 3×).
7. **Dense scans.** Do not enable the net for 40 M-face inputs until the dense thinning step lands. Otherwise their decided characteristics all become REVIEW (`RESAMPLE_CHECK_INCOMPLETE`).
8. **E2E.** Run `run_e2e_real.mjs` (13/13), with the zs re-pin in the same build.
9. **Ledger.** Add an AMENDMENT/FREEZE entry. It binds:
   - engine e3-1.13.0 and its fingerprint;
   - the config sha;
   - the A1 and G1 scorecards;
   - this report's sha.

   Then re-pin the zs_lane_v3 rev3 pin and the A5 bundle as in REPIN_PLAN.

**Rollback:** reverse the patch (`patch -R -p1`). `ENGINE_VERSION` returns to e3-1.12.0 and the fingerprint to `5fbed707…`.

## Unverified and limits

- **The fix is tuned on one part.** All 11 FAIL↔PASS characteristics are on one part number, and the fix was developed on it. 0 is measured on 7 realizations per scan, so it is a lower bound for unseen parts.
- **Thresholds.** The near-tie ratio 0.75 and the swap tolerances (±0.25 t separation, ±0.3 t offset) were set by reasoning on B and checked on this corpus only.
- **The registration-pose mode** was found on one scan by chance (the net's own resamples). Its prevalence elsewhere is unknown.
- **Wall times** are from a host shared with other agents. Serial and parallel timing is one run each, on one scan.
- **Not run:**
  - real-part E2E;
  - any dense scan with the net;
  - option (c) through the frozen A1 and G1 runners. Its numbers come from the 7-realization sweep only.
