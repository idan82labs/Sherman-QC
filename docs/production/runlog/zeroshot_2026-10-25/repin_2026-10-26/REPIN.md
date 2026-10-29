# Bundled production re-pin, 2026-10-26: APPLIED (attempt 2, with a density guard instead of thinning)

**Authority.**
- Owner, quoted: "1a, approve the bundle, station is Windows". Recorded in freeze-ledger NOTE seq 25.
- The coordinator then narrowed the bundle: no dense thinning, and a fast, fail-closed density guard in its place.

**Attempt 1 was rolled back.** It included the 0.63 mm thinning, and the thinned B_n5 gave 7 FAIL→PASS. Its record is in `evidence/attempt1_thinning/` and ledger NOTE seq 26.

Machine-readable record: `results.json`. Evidence: `evidence/attempt2/`. Rollback procedure: `ROLLBACK.md` (attempt 2 section).

## What is live now

### E3 `e3-1.13.0`
- Inspect fingerprint `f319ed5c…`, engine config sha `038eb11f…`.
- Contents:
  - the rc patch: one-skin swap and near-tie withholding;
  - the resample safety net, policy 1, **K = 2, serial**;
  - parallel resamples now use `spawn`, default off.
- Native scans with K = 2 take about 3× longer: G2(e) is waived by owner 1a.

### Density guard (`engine._density_guard`)
**Trigger.** It runs after cleaning and the unit check, and fires when either holds:
- more than **15.0 points per occupied 0.63 mm cell** (the densest of 132 native survey scans reads 12.94);
- more than **3,000,000 points** (the largest native scan holds 2.40 M).

The point-count criterion goes beyond the coordinator's spec. It is strictly more conservative, and it catches B_n3 (4.25 M points, 13.2 points per cell).

**What it returns.** The scan is not measured. E3 writes a normal result in 1–5 s:
- exit 0, not a worker error;
- validity `UNSUPPORTED_INPUT`, with the reason `DENSITY_ABOVE_VALIDATED_RANGE:<metric>`;
- every characteristic UNKNOWN(INVALID_INSPECTION);
- summary `INVALID_INSPECTION`.

It can never PASS.

**Vocabulary.**
- `ValidityState` is a closed contract Literal. Adding a state would be a contract amendment, which this re-pin does not cover.
- Validity reasons are free strings, so the explicit code is carried there. No contract vocabulary was added.
- **The outcome is INVALID_INSPECTION, not REVIEW_REQUIRED.** A VALID result needs a registration (`InspectionValidityV1` validator), and a fast return cannot register. "Withheld" here therefore means UNKNOWN under an invalid inspection, which is the closest existing withhold outcome.
- The station shows "Inspection invalid". The validity label is "Scan file not supported"; the reason string carries the real cause.

### ENGINE_IO-A5, IN FORCE
- `ENGINE_RSS_LIMIT_BYTES` and the vocabulary mirror are 32 GiB.
- The supervisor ceiling is 32 GiB.
- `SHERMAN_LINE_RSS_LIMIT_MB` defaults to 32768, max 32768.
- ENGINE_IO §7.1 marks A5 IN FORCE.
- The eval CLI keeps 4096 MB, so the G1 run conditions are unchanged.

### zs pins
| Lane | Code fingerprint | Pin sha256 | config sha256 |
|---|---|---|---|
| zs_lane_v1 (default, live) | `52fd21d6…` (was `af87df2f…`) | `94e71f30…` (was `e77f1f73…`) | `5f41595b…` (was `dec044e8…`) |
| zs_lane_v3 rev 3 (only via `SHERMAN_ZS_LANE_PIN`) | `49a16399…` (research rev 3 was `dc23a905…`) | `0d910b6f…` (was rev 2 `115b7077…`) | `9b173ea8…` (was `8e002f3e…`) |

- Only `contracts/production/line_job.py` and E3 `config.py` / `engine.py` / `metrology.py` moved.
- Lane code, models, thresholds and parameters are unchanged.
- Z2 calls E3 `_flat_start` / `_reseat_plane` with defaults and no branch record, so its behaviour is unchanged. The rehearsal confirms it: every scan's line counts and kinds equal the rev-2 rehearsal.
- `DEFAULT_LANE` stays `zs_lane_v1`, and the overlay still never creates a PASS.
- The pre-re-pin configs and pins are archived in `archive/`.

### PREREG template
`v3/PREREG_FRESH_TEMPLATE.json` (now `90394a42…`):
- the primary arm moves to `49a16399…` / `9b173ea8…`, and the comparison arm to `52fd21d6…` / `5f41595b…`;
- a new `arms_repin_2026_10_26` block records the before values;
- **the decision-rule shas are unchanged**: rule `6ae84658…` and flip_allowed_if_all `18c0aeec…`.

## Gates

| Gate | Result |
|---|---|
| G1 regression tier | **PASS.** FALSE_PASS 0/553, WRONG_OR_MIRRORED 0/32. Every accuracy and decision gate is identical to the candidate's net arm. Runtime only: P95 87.7 s, and TIER_BUDGET FAIL (1923 s) because of host load (8–25). |
| A1 physical | **PASS, G2(e) waived.** The guard fired on no native case. Characteristics are identical to the candidate's net arm on **26/26** (see the note below). G2(e) P95 is 152.6 s and the maximum 158.9 s under load: FAIL, waived. |
| Frozen recompile | **18/18** |
| Named tests | E3: 17 files, 308 passed, 1 skipped. zs / overlay / contracts / production_line: 17 files, 447 passed. Overlay tests: 50/50 with the v3 pin and 50/50 with the v1 pin. |
| Dense set (16 meshes, production limits 180 s / 170 s / 32 GiB) | **0 FAIL→PASS.** 11 guarded meshes return withheld in 1.0–4.6 s at ≤ 2.75 GiB (see the note below). |
| zs production rehearsal (50 consumed-DEV scans) | **PASS.** v1 and v3 are both 50/50 OK, with 0 false PASS, 288 lines, max 30 s / 36 s, ≤ 1.82 GiB, and no effective state changed between v1 and v3. |
| Real-part E2E | **PASS on the third run.** Run `20261026T-real-repin-e3-1-13c` is 13/13 (see the note below). |

**A1 note.** In the main run, pc_07062eee17aa and pc_2714fb7e0c0b ran under load. Their primary took about 70 s against 33 s, so the budget skipped one resample and the fail-safe withheld; the values were identical. Both cases were re-run on the same code with the same runner and came out identical to the candidate. per_bend_all_v3 from the main run shows proxies 2 against 3 for those withheld cases; it was not recomputed after the re-run.

**Dense set note.**
- The 11 guarded meshes are A_n5 / n6 / n9, B_n3 / n5 / n7, C_n3 / n4 / n5 and D_n2 / n3.
- The 5 meshes inside the native envelope (A_n3, A_n3c, B_n2, B_n2c, C_n2) are measured like native scans. They took 135–171 s on this loaded host; native B itself took 139 s in the same window.
- A_n3 / A_n3c keep e3-1.12.0's REVIEW→PASS on B7 line position. That is not FAIL→PASS.

**E2E note.**
- Runs `…-13` and `…-13b` were 12/13: each failed only W3b, a Vendor Setup studio UI step before any E3 job. Its signature (bend markers B7/B10 not found) matches the known flake of `20261019T-real-fixes3c` / `3d`.
- Every job in the passing run carries:
  - engine e3-1.13.0;
  - an overlay with lane `zs_lane_v1` and fingerprint `52fd21d6…`;
  - verdict not lowered, effective state = E3 state.

## What changed for live behaviour
- **Every E3 inspection runs two extra 95 % resamples**, at about 3× the wall time.
  - A decided characteristic that a resample decides differently becomes REVIEW.
  - When the budget cannot fit the resamples, every decided characteristic becomes REVIEW (`RESAMPLE_CHECK_INCOMPLETE`). This was seen under host load in A1 and on in-envelope dense meshes.
- **23553000 B's short-panel skin is fixed** (a consistent FAIL). Some near-ties become REVIEW.
- **Scans above the validated density or size** come back fast as INVALID_INSPECTION with `DENSITY_ABOVE_VALIDATED_RANGE`.
- **The per-step RSS cap is 32 GiB** for E3, zs, compile and render.
- **The zs overlay** runs zs_lane_v1 under its new pin, with outputs unchanged. zs_lane_v3 rev 3 is selectable only with `SHERMAN_ZS_LANE_PIN`.

## Unverified
- Windows: station runtime, the 3× net cost, and `spawn` with supervisor RSS accounting.
- Unloaded-host timing: every runtime figure here was measured at load 8–25.
- In-envelope dense scans near the budget.
- A deployed station `.env` may still pin `SHERMAN_LINE_RSS_LIMIT_MB=4096`; it was not read.

## Ledger
| Seq | Kind | Content |
|---|---|---|
| 25 | NOTE | Owner decisions |
| 26 | NOTE | Attempt 1 rolled back. Its bound REPIN.md / results.json now live, byte-identical, at `evidence/attempt1_thinning/REPIN_attempt1.md` and `results_attempt1.json`. |
| 27 | NOTE | RE-PIN |
| 28 | NOTE | PREREG fingerprint update (lane code unchanged, decision rule unchanged) |
| 29 | NOTE | RESULTS |
