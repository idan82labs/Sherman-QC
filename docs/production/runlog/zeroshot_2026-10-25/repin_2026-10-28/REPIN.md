# Production re-pin, 2026-10-28: E3 e3-1.14.0 (pose wells + dense thinning): APPLIED

**Owner decision (2026-10-28), quoted:** "extend the waiver, thin dense scans, go ahead". It is recorded in freeze-ledger NOTE seq 30.

The candidate report is `station/E3_POSE_AMBIGUITY.md`, applied as patch `station/e3_pose/e3_1_14_0_rc.patch` on top of live e3-1.13.0.

Machine-readable record: `results.json`. Evidence: `evidence/`. Rollback procedure: `ROLLBACK.md`.

## What is live now

### E3 `e3-1.14.0`
- Inspect fingerprint `8805148f…` (was `f319ed5c…`). Engine config sha `4f107f5a…`.
- Compile, eval and contracts fingerprints are unchanged.
- Contents: e3-1.13.0, plus:
  - **Pose wells.** The chosen pose is refined again from 8 tilted starts. A well whose total beats the chosen pose by more than 0.03 replaces it.
  - **Alternative-pose withholding.** A well within 0.06 of the chosen total is measured too. Any decided characteristic it reads differently, or cannot read, becomes REVIEW. This never creates a PASS.
  - **Dense ingest thinning.** A cloud above 15 points per occupied 0.63 mm cell keeps one actual point per cell, the one nearest the cell mean. Thinning replaces the e3-1.13 refusal on that criterion.
  - **The 3 M-point refusal is kept after thinning** (UNSUPPORTED_INPUT, `DENSITY_ABOVE_VALIDATED_RANGE`).
  - **The K=2 resample net is unchanged:** policy 1, serial.
- `ENGINE_VERSION` is `e3-1.14.0`. The version assertions in `test_prod_engine_e3_1_13_candidate.py` and `test_prod_engine_fixes3_2026_10_19.py` were updated, and the new `test_prod_engine_e3_1_14_pose_wells.py` was added.

### zs pins
The zs pins hash E3 `config.py` and `engine.py`, so both lanes were re-pinned. Lane code, models and parameters are unchanged.

| Lane | Code fingerprint | Pin sha256 | config sha256 |
|---|---|---|---|
| zs_lane_v1 (default, live) | `c712c15c…` (was `52fd21d6…`) | `0087fb23…` (was `94e71f30…`) | `33d9270a…` |
| zs_lane_v3 rev 3 (only via `SHERMAN_ZS_LANE_PIN`) | `871f23fa…` (was `49a16399…`) | `711bd079…` (was `0d910b6f…`) | `b65eec59…` |

`DEFAULT_LANE` stays `zs_lane_v1`. Pre-change copies are in `archive/`.

### PREREG template
`v3/PREREG_FRESH_TEMPLATE.json` is now `60b08372…` (was `90394a42…`).
- The arm fingerprints and config shas were moved to the values above.
- A new `arms_repin_2026_10_28` block records the before values.
- The decision-rule shas are unchanged: `6ae84658…` and `18c0aeec…`.

### ENGINE_IO.md
The A5 row notes that e3-1.14.0 thins instead of refusing. It is a doc change only; the contracts code fingerprint is unchanged.

## Gates

Runs were reniced to 10 and used at most 2 slots. No E2E browser ran while the other agent's E2E was running.

| Gate | Result |
|---|---|
| G1 regression tier | **PASS.** FALSE_PASS 0/553, WRONG_OR_MIRRORED 0/32. Every accuracy and decision gate is identical to e3-1.13.0. Runtime only: P95 56.2 s, and TIER_BUDGET FAIL (1426 s), as before. |
| A1 physical | **PASS, G2(e) waived.** Characteristics are identical to e3-1.13.0 on **26/26** native cases, and per_bend_all_v3 is identical. G2(e) P95 is 113.0 s and the maximum 132.4 s. |
| Frozen recompile | **18/18** |
| Named tests | E3: 18 files, **317 passed, 1 skipped**. zs / overlay / contracts / production_line: 17 files, **447 passed**. Overlay tests 50/50 with each pin. |
| Dense set (16 meshes, production limits 180 s / 170 s / 32 GiB, K=2) | **0 FAIL→PASS, 0 decided flips.** Max wall 149.1 s, max RSS 5.47 GiB. |
| zs production rehearsal (50 consumed-DEV scans) | **PASS.** v1 and v3 are both 50/50 OK, with 0 false PASS and 288 lines. Max 13.1 s / 16.6 s, ≤ 3.25 GiB. Every scan's line counts and kinds equal the rev-2 rehearsal, and no effective state changed between v1 and v3. |
| Real-part E2E `20261028T-real-repin-e3-1-14` | **PASS except the known flake: 13/14.** The only failure is W3b with the ticketed signature (B7/B10 markers missing, labels `[]`). |

**A1 timing.** The 113.0 s P95 is +26 % over e3-1.13.0's quiet reference of 89.5 s; this run had load about 3–12. The candidate's quiet-host single-scan cost is +21 % (corpus max 99.0 → 119.7 s).

**Dense set, case by case.**
- **Thinned and decided (10 meshes):** A_n5/6/9, B_n5/7, C_n3/4/5, D_n2/3. Each has 0–1 decided → REVIEW changes against native.
- **B_n5:** a pose well replaced the chosen pose (total 1.264 → 1.410). The 7 FAIL→PASS of attempt 1 (2026-10-26) are gone, and its only change is B14 angle FAIL → REVIEW.
- **B_n3** (4.25 M points, 13.2 per cell): refused by the point count.
- **A_n3 / A_n3c:** not thinned. They keep e3-1.12.0's REVIEW → PASS on B7 line position. That is not FAIL→PASS.
- **Timing:** natives took 29–108 s. The in-envelope meshes B_n2 / B_n2c took 149 s and 148 s at load 3–12. The thinned meshes took 32–101 s.

**E2E.** All other 13 steps passed. That includes direct-E3 equality on all three real scans, the other agent's new P1 part-picker step, and Hebrew RTL. Every job carries engine e3-1.14.0 and an overlay on lane zs_lane_v1 / `c712c15c…`, with the verdict not lowered.

## What changed for live behaviour
- **Every E3 inspection runs the pose-well search.** It takes about 2–7 s per pass and is paid 3 times with the K=2 net, for about +21–26 % wall time. **G2(e) runs under the extended owner waiver.**
- **A pose that a tilted restart beats by more than 0.03 is replaced.** A close alternative pose (within 0.06) withholds the characteristics it reads differently, as REVIEW.
- **Dense scans (more than 15 points per 0.63 mm cell) are now measured on a thinned cloud.** In e3-1.13.0 they were refused. Scans with more than 3 M points after thinning are still refused.
- **Native survey verdicts are unchanged.** The candidate report measured this over 88 scans × 7 realizations, and A1 confirms 26/26.

## Rust adapter (`research/zeroshot/zs_lane_v3_rs`, not edited)
- `config_v3r3_rs.json` fingerprints `domains/production/engine/inspect/config.py` and `engine.py`, on base `49a16399…`.
- Both now mismatch, so **its `--strict-pins` refuses until the Rust agent re-runs `freeze_config`** against the new base `871f23fa…`.

## Coordination
The other agent's E2E `20261028T-picker-final` started after the patch was applied, so it ran against e3-1.14.0. Its earlier hotfolder run had finished before any change.

## Unverified
- **Windows station:** the runtime with +21–26 % and the K=2 serial net, and spawn workers with the supervisor's RSS accounting.
- **Quiet-host timing for this exact tree.** Every timing here ran with other agents active.
- **Alternative-pose withholding has never met a real near-tie in the corpus.** It is covered by synthetic tests only.
- **The pose-well mechanism was observed on one part** (23553000 B).

## Ledger
| Seq | Kind | Content |
|---|---|---|
| 30 | NOTE | Owner decision, quoted |
| 31 | NOTE | RE-PIN |
| 32 | NOTE | PREREG fingerprint update (lane code unchanged, decision rule unchanged) |
| 33 | NOTE | RESULTS, binding this file, `results.json` and `ROLLBACK.md` |
