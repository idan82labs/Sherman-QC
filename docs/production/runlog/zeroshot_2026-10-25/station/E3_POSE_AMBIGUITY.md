# E3 registration-pose mode: pose wells and alternative-pose withholding (2026-10-26 to 10-28)

> **Research candidate. Nothing was applied.** Live E3 is `e3-1.13.0` (re-pin, ledger seq 27). Its inspect files were sha-checked before and after this work and are unchanged. No contract, eval, pin, ledger or test file in the repo was edited. Nothing was installed or committed.
>
> - Candidate: `e3-1.14.0-rc`, built in a scratch shadow tree (`scratchpad/e3pose/repo2`) that symlinks the rest of the repo.
> - Patch: `e3_pose/e3_1_14_0_rc.patch`, against the **live e3-1.13.0 tree**. It dry-run-applies cleanly and touches `config.py`, `engine.py` and 3 tests (1 of them new).
> - Machine-readable results: `e3_pose/results.json`. Tools and configs: `e3_pose/tools/`.
> - Heavy runs used 1 geometry slot (`sherman_slot.sh`, `OMP_NUM_THREADS=1`). Peak RSS was 6.7 GiB, on dense input.
> - Inputs: the 88 survey scans and the 16 dense meshes only. No FRESH, DEV-CHECK, TEST-B, N1, HOLDOUT10, EHO4 or sealed input was used. G1 truth was read only by the scorer (`arms2.py`).

## Answer first

| | e3-1.12.0 | e3-1.13.0 (live, K=2 net) | **candidate e3-1.14.0-rc** |
|---|---|---|---|
| Survey: chars FAIL↔PASS / FAIL→PASS (88 scans × 7 realizations) | 11 / 8 | 0 / 0 | **0 / 0** |
| Survey: chars with any decision change (of 1,619) | 55 | 61 | **52** |
| Coverage loss, native (of 959 decided) | – | 1.46 % | **1.46 %** (the same 14 characteristics) |
| Coverage loss, all 7 realizations (of 6,692) | – | 1.63 % | **1.49 %** |
| G1 false PASS (native / perturbed) | 0 / 0 | 0 / 0 | **0 / 0** |
| G1 decided-and-correct (native / perturbed) | 314/314, 1,884/1,884 | 314/314, 1,882/1,882 | **314/314, 1,882/1,882** |
| Native verdicts changed vs e3-1.13.0 | – | – | **0** of 1,619 |
| Dense, 16 meshes, FAIL→PASS vs native | 7 (B_n5 thinned) | refused by the density guard (7 when thinned) | **0** (decided flips 0) |
| Corpus max pc_2714, K=2 serial, quiet host | – | 99.0 s | **119.7 s** (+21 %, 60 s margin) |

- **Alternative-pose check on native scans: fired 0 of 88 times** (and 0 of 528 perturbed realizations).
- **Pose switch** (a better well replaces the chosen pose): 0 of 616 survey runs. On dense input it switched once, on B_n5, the failing case.
- On one other scan (`sc_c27168a2de83`, 7 of 7 realizations) a second well was articulated. Its total was 0.098–0.107 below the chosen pose, outside the 0.06 alternative margin, so nothing changed.

![gate summary](e3_pose/e3pose_gate_summary.png)

**Recommendation for e3-1.14.0:** promote **pose wells + alternative-pose withholding + dense thinning in place of the density guard's ppc refusal**. Keep the guard's max-points refusal, applied after thinning.
- It turns the dense FAIL into a correct FAIL: the right pose is found, not withheld.
- It changes no native survey verdict.
- It costs about +21 % wall time.
- The K=2 net stays. It is still the backstop for mechanisms that pose wells do not cover.

The decision rule is set in the gating table at the end; it needs an owner decision on the open items.

## What was wrong (diagnosis)

`scratchpad/e3pose/tools/diag.py` runs the CLI's engine in-process. Pass-through wrappers record every hypothesis, refine call and fine search. It was run on B_n5 (23553000 B, thinned to 333,827 points) and on the native B full scan.

1. **The objective is not tied: it prefers the native pose.** The e3-1.13 choice scores datum 0.576 and total 1.264. The native-density pose, refined on the *same* thinned cloud, scores **0.707 and 1.412**. The wrong pose wins only because the search never offers the right one.
2. **The two poses differ by a pure tilt, not an in-plane shift.** The difference is 0.63° about the datum's short axis, which lifts the ends of the 460 mm datum by ±2.58 mm. In-plane the two poses agree to <0.05 mm.
3. **The mechanism is the datum plane fit's capture band.** `refine_datum` refits the plane from points within 0.6 mm of the current plane, after a first pass of ±(t/2 + 1.5) mm.
   - The coarse peak started 1.08° off, so the datum's far ends lay outside the band.
   - The plane converged on the middle of the sheet: a self-consistent tilted plane, which is a local optimum.
   - In-plane perturbations of the start (±1°, ±1.5 mm, 50 starts) all return to the same well.
   - Tilt perturbations of 0.25–1.0° reach the native well. The landscape has several discrete wells (0.07°, 0.37°, 0.63°, 1.04° from native). See the left panel below.
4. **Why the K=2 net missed it.** A 95 % resample of a 35-points-per-cell cloud thins back to an almost identical cloud. The search then fails the same way, and both resamples agree.

![tilt wells](e3_pose/e3pose_B_n5_tilt_wells.png)

## What was built (`e3_1_14_0_rc.patch`)

All of it is general geometry, with no part, SKU or filename logic. Each piece has its own config flag.

### 1. Pose wells (`pose_wells = 1`): multi-start refinement of the chosen pose (find)

**When it runs:** after the articulated ranking and before the mirror check, on the chosen proper pose.

**The starts.** `refine_datum` runs again from **8 tilted starts**, tilting about each in-plane datum axis through the outline's centroid.
- The angles move the datum's far edge by ±`pose_well_edge_mm` (1.5 mm) and ±3 mm, at most 2°.
- So the angle scales with the datum's own size: 0.37° and 0.75° on the 460 mm axis of B.

**What counts as a well.** Results more than 0.5 mm apart anywhere on the datum outline are distinct wells.

**What gets articulated.** Wells with a datum score no worse than the chosen one's by more than 0.1 get the articulated fit, at most 3 of them.

**The switch rule.** A well whose total beats the chosen pose by more than **0.03** (the existing `ambiguity_margin`) replaces it. The replaced pose becomes an alternative.

### 2. Alternative-pose withholding (`pose_well_alt_margin = 0.06`): detect

A well whose total is within 0.06 of the final chosen total is decoded and measured as the chosen pose is, without the crease, wrong-way or free-direction steps. Then:
- a PASS or FAIL that the alternative pose reads differently (REVIEW included) becomes REVIEW, with |Δ| added to U;
- a PASS or FAIL that the alternative pose cannot measure, or that sits on a bend it decodes as another state, also becomes REVIEW;
- a budget that runs out during the check withholds every decided characteristic (`POSE_ALTERNATIVE_CHECK_INCOMPLETE`).

U only grows, so this check can never create a PASS. It runs in the primary and in each K=2 resample.

### 3. Dense ingest (`dense_thin = 1`): the live guard's ppc criterion thins instead of refusing

- **Where:** after the unit check.
- **When:** the cloud has more than `density_guard_ppc` (15) points per occupied `density_guard_cell_mm` (0.63 mm) cell.
- **What it keeps:** one actual point per cell, the one nearest the cell mean. Ties go by coordinates, then file order.
- **Resamples** inherit the decision.
- **Then the live guard runs on the result.** Its max-points criterion (3 M) refuses what is still too large. Example: B_n3, 4.25 M points at 13.2 points per cell, is not thinned and stays refused (UNSUPPORTED_INPUT, so it can never PASS).
- `dense_thin = 0` restores e3-1.13.0 exactly.
- The selection is implemented without a full lexsort, and its output is identical to the reference formulation, checked with ties.

## Survey evaluation (88 scans × 7 realizations, 616 runs)

**Protocol.** This is exactly the survey protocol:
- 50 G1, 26 A1 and 33 DEV-cohort scans (21 shared); `pc_b4719db3dd69` is refused, as before;
- realizations: native, `subsample.py` 95 % seeds 9501–9505, and `thin_exp.py` 0.40 mm;
- 1 thread, K=2, policy 1, serial, with `resample_check_record`.

**Budget.** The host load was 2–20 (other agents). So the decision sweep ran with an **890 s budget**, which keeps host-load timeouts from posing as withholding. Runtime was measured separately at production limits.

**Arms.** Arms are compared per characteristic id against the e3-1.12.0 survey (`e3stab/runs`) and against e3-1.13.0. The e3-1.13.0 arm is the e3c sweep's `primary_raw` plus the first 2 resamples under policy 1, offline; this reproduces the E3_1_13_CANDIDATE numbers exactly (1.46 % / 1.63 %, G1 1,882/1,882).

| Arm | FAIL↔PASS chars (scans) | FAIL→PASS | any change | coverage loss native / all | G1 perturbed decided (all correct) |
|---|---|---|---|---|---|
| e3-1.13 primary (net off) | 0 (0) | 0 | 45 | 0.63 % / 0.69 % | 1,884 |
| e3-1.13.0 (net) | 0 (0) | 0 | 61 | 1.46 % / 1.63 % | 1,882 |
| candidate primary (net off) | 0 (0) | 0 | 45 | 0.63 % / 0.69 % | 1,884 |
| **candidate (net)** | **0 (0)** | **0** | **52** | **1.46 % / 1.49 %** | **1,882** |

**Native results** are identical to e3-1.13.0 for every characteristic of every scan, in both arms. That is expected: no survey primary switched pose or fired the alternative check.

**Lower all-realization loss.** The candidate loses less over all realizations (1.49 % against 1.63 %; 9 fewer changing characteristics). This is because perturbed realizations and their resamples no longer fall into the tilt well, B full `s95_9504` included. That is the case E3_1_13_CANDIDATE found by chance. On that realization e3-1.13's net withheld 9 characteristics as REVIEW: B1/B10/B11/B13 line positions, B3 B, B4 B, B6 A, B7 A and B15 A. The candidate decides all 9 again, with the native decisions: 7 FAIL and 2 PASS.

**Thinning** fired on no survey realization: 0 logs contain "dense input thinned".

**Incomplete net.** One realization's check was incomplete: `sc_43a32fe3b4a3 s95_9504`, where resample 7101 came out MIRRORED_PART_SUSPECTED. It is identical in e3-1.13.

**Cost of the wells step.** Median 2.0 s, P95 7.0 s, max 11.1 s per inspection pass. That is 22 % of a primary at the median and 30 % at P95. In the K=2 net it is paid 3 times.

## Dense evaluation (16 meshes, `dense_ready/meshes`, N1 excluded)

**Reference:** the candidate on the native-density scan of the same part (A_n1, B_n1, C_n1, D_n1), with the same K=2 net. Budget 890 s.

**Result: FAIL→PASS 0 on all 16; decided flips 0.**
- **11 meshes thinned:** A_n5/6/9, B_n5/7, C_n3/4/5, D_n2/3, plus B_n7 at 68 points per cell. Their changes are 0–1 each, all decided → REVIEW.
- **B_n5:** pose well switch 1.264 → 1.410. The only change is B14 angle FAIL → REVIEW (U 0.65). The 7 e3-1.13 FAIL→PASS are gone, and B3/B4/B6/B7 lengths read FAIL again.
- **Unthinned A_n3 / A_n3c:** they keep e3-1.12's REVIEW → PASS on B7 line position. That is not FAIL→PASS. It is pre-existing and decided at native density.
- **In the live tree** (dense ppc thin, then the 3 M max-points guard), B_n3 is refused. Every other mesh stays under 3 M points.

**Production limits, quiet host (180 s / 170 s, K=2):**

| Mesh | Wall | Peak RSS |
|---|---|---|
| B_n5 | 112.1 s | 3.5 GiB |
| B_n7 (23 M points) | 140.2 s | 6.3 GiB |
| D_n3 | 72.5 s | 5.5 GiB |
| A_n9 | 65.4 s | 5.8 GiB |

Verdicts are as above. The rebased tree (`repo2`, on live e3-1.13.0) gives decision- and value-identical results on B_n1, B_n5 and B_n7. It refuses B_n3 by points, as designed.

## Runtime

**Corpus max `pc_2714fb7e0c0b`**, native, production limits, quiet host (load 2), one run each:
- e3-1.13.0 (wells off): **99.0 s**, primary 33.9 s;
- candidate: **119.7 s**, primary 40.4 s, of which wells 6.7 s.

That is **60 s inside the 180 s hard stop**. B full scan: 79.9 → 92.6 s.

**Under load the margin goes.** At load ~8–20 the same scans took 150–170 s for both versions. Resamples were then skipped and the fail-safe withheld everything (`RESAMPLE_CHECK_INCOMPLETE`). This is a host issue that both versions share, but the candidate's +21 % makes it more likely. Parallel resamples (spawn) remain the documented remedy.

## Tests

On the rebased tree (live tests symlinked; the 3 version-sensitive files copied):

- 18 named E3 files: inspect a1 … iter7, cli, units, fixes, fixes3, ingest, contract_parity, eval_scorer, freeze_physical, the e3_1_13 candidate, the e3_1_13 density guard and the new pose_wells file. **317 passed, 1 skipped.**
- The new `tests/test_prod_engine_e3_1_14_pose_wells.py` (9 tests) covers:
  - thinning: one actual point per cell nearest the mean, order-independent;
  - cell density;
  - tilt and gap geometry;
  - config validation;
  - that alternative withholding never creates a PASS;
  - that a thinned cloud passes the guard;
  - that `dense_thin = 0` restores the guard.

## Limits and unverified

- **One failing case.** The mechanism and the switch were observed on one part (23553000 B, a 460 mm strip datum). Nowhere in the survey did the alternative-pose check fire. Its withholding path is exercised by synthetic tests and code review only, not by a real near-tie.
- **Margins are set by reasoning, not tuned.** The 0.03 switch margin equals the existing `ambiguity_margin`. The 0.06 alternative margin, 0.1 score slack, 1.5 / 3 mm edge starts and 0.5 mm well gap were chosen by reasoning on B and checked on this corpus only.
- **Tilt only.** Starts are perturbed out of plane only. In-plane local optima were not seen: 50 in-plane starts returned to one well on B. They are still covered by the fine search and the K=2 net, not by wells.
- **Timings.** Each production-limit timing is a single run on a quiet host. The survey sweep's wall times (P50 28.8 s, P95 101 s, max 128 s under load at an 890 s budget) are not production timings.
- **RSS.** Dense RSS reaches 6.7 GiB. That is within the ENGINE_IO-A5 32 GiB cap and above the old 4 GiB eval-CLI cap; G1-tier runs are unaffected because they never thin.
- **Not run:**
  - the frozen A1 recompile;
  - A1 run-physical / score-physical;
  - the G1 regression tier through `e2_regression_card`;
  - E2E.

  Native results are characteristic-identical to e3-1.13.0 on all 88 scans, so these runs are expected to match except runtime. Not measured.

## Gating for an e3-1.14.0 re-pin (owner decision; not taken here)

1. **Apply** `patch -p1 < docs/production/runlog/zeroshot_2026-10-25/station/e3_pose/e3_1_14_0_rc.patch`, then set `ENGINE_VERSION = "e3-1.14.0"` and the test assertions.
2. **Gates:**
   - named tests;
   - frozen recompile 18/18;
   - A1 physical: expect characteristic-identical to e3-1.13.0, with G2(e) runtime about +21 %, which needs the waiver extended;
   - G1 tier: FALSE_PASS 0/553;
   - **dense through the production path: 0 FAIL→PASS, measured here as 0 of 16.**
3. **Decide the density policy:** thin, as recommended, or keep e3-1.13.0's refusal (`dense_thin = 0`). The pose wells are independent of it.
4. **Re-run the dense gate on the station host.** Timing at load is the open risk.
