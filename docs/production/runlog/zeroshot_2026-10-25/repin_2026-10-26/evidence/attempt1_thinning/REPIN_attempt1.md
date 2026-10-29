# Bundled production re-pin, 2026-10-26: applied, gate failed, ROLLED BACK

**Live state is unchanged.** The bundle was applied in full and measured, but one gate failed. That gate is *dense meshes through the production path with no FAIL→PASS against native*. Every file was restored byte-identically, and the live tree is back to:

- `e3-1.12.0`, inspect fingerprint `5fbed707…`;
- zs_lane_v1 pin `e77f1f73…` (fingerprint `af87df2f…`, verify `[]`);
- zs_lane_v3 pin rev 2 `115b7077…`;
- research `config_v3.json` rev 3 (`dc23a905…` / `8e002f3e…`);
- ENGINE_IO-A5 still PROPOSED, with the cap at 4 GiB.

Machine-readable record: `results.json`. Evidence: `evidence/`. Rollback procedure: `ROLLBACK.md`.

**Owner decision** (main session, 2026-10-26), quoted: "1a, approve the bundle, station is Windows". It is recorded in freeze-ledger NOTE seq 25.

## Gate table (the bundle as applied)

| Gate | Result | Detail |
|---|---|---|
| G1 regression tier | **PASS** | FALSE_PASS 0/553, WRONG_OR_MIRRORED 0/32. Every accuracy and decision gate is identical to the candidate's net arm. Runtime only: P95 55.6 s against the candidate's 42.3 s, and TIER_BUDGET FAIL (1404 s against 1115 s). The cause is host load (load average 9–24). |
| A1 physical | **PASS** except the waived G2(e) | 26/26 cases are characteristic-identical to the candidate's promoted (net) arm, and per_bend_all_v3 is identical. G2(e) P95 is 121.5 s and the maximum 126.6 s, against 89.5 / 102.1 for the candidate under the same host load; it FAILs, as waived by owner 1a. |
| Frozen recompile | **18/18** | COMPILED 17, BLOCKED 1, as frozen. |
| Named tests | **PASS** | E3: 17 files, 308 passed, 1 skipped. zs / overlay / contracts / production_line: 17 files, 447 passed after one test was updated for the rev-3 pin. Overlay tests: 50/50 with the v3 pin and 50/50 with the v1 pin. |
| **Dense through the production path** | **FAIL** | **B_n5 (thinned): 7 FAIL→PASS against native.** All 16 meshes stayed within 180 s (max 173.8 s) and 32 GiB (max 3.3 GiB). |
| E2E | not run | Stopped after the dense failure. |
| zs production rehearsal | not run | Stopped after the dense failure. |

## Why the dense gate failed

`B_n5` is 23553000 B (sc_6eba) at 11.8 M points, which is 35.4 points per 0.63 mm cell. Thinning fired on it and kept 333,827 points.

The thinned cloud registers into the **registration-pose mode** that the candidate report found and left unverified:
- datum score 0.576, total 1.264, pose margin 0.0495;
- B3 B / B4 B / B6 A / B7 A read 73.5–73.7 (PASS) against native 72.6–72.7 (FAIL);
- B1 / B10 / B11 line positions read PASS against native FAIL.

Both K = 2 resamples land in the same pose, so policy 1 sees agreement and withholds nothing. The parallel (spawn) mode gives the same 7.

**The 0.63 mm thinning was not "proven" on B.** The `dense_rel` record `E3_THINNING_EXPERIMENTS.txt` already shows this same 7-FAIL→PASS pose signature at 0.63 mm on e3-1.12.0 (`B_n2_c063`, `B_n5_c063`). The root fix does not reach it; the candidate report says it fixes skins, not registration.

**The other dense meshes:**

| Mesh(es) | Thinned | Result against native |
|---|---|---|
| A_n5 / n6 / n9, C_n3 / n4 / n5, D_n2 / n3 | yes | 0–1 changes, no FAIL→PASS |
| B_n7 | yes | Budget ran out, so the net was incomplete and every decided characteristic was withheld (fail-safe) |
| A_n3 / A_n3c, B_n2 / B_n2c / B_n3, C_n2 | no (density within the native range) | No FAIL→PASS. B_n2c and B_n3 hit the budget and were all withheld. A_n3 keeps e3-1.12.0's REVIEW→PASS on B7 line position. |

**Diagnostic.** B_n5 with thinning off has 0 FAIL→PASS, but only because the primary used the whole budget (175.8 s): every decided characteristic was withheld.

## What was built (reverted, reproducible)

1. **E3 e3-1.13.0.** `e3_1_13_0_rc.patch`, with:
   - `ENGINE_VERSION = "e3-1.13.0"`;
   - `resample_check_k = 2`, policy 1, serial;
   - the fixes3 test's and the candidate test's assertions updated.
2. **Dense thinning in E3 ingest** (`engine._dense_thin`):
   - Where it runs: after gross-outlier removal and the unit check.
   - What it keeps: one actual point per 0.63 mm cell, the one nearest the cell mean (ties by coordinates, then file order).
   - When it fires: only when points per occupied 0.63 mm cell exceed **15.0**. The native maximum, measured on all 132 survey scans, is 12.94 (`evidence/density_native.jsonl`).
   - Resamples inherit the primary's decision (`InspectOptions.dense_thin`). ingest algorithm id `ply_stl_explicit_unit_dense_cell_thin_v2`.
   - Native scans were untouched: A1 is 26/26 identical to the candidate.
3. **Parallel resamples** switched from `fork` to `spawn`. Default off.
4. **ENGINE_IO-A5 as written:**
   - `ENGINE_RSS_LIMIT_BYTES` and the vocabulary mirror at 32 GiB;
   - supervisor `MAX_RSS_LIMIT_BYTES` at 32 GiB;
   - `SHERMAN_LINE_RSS_LIMIT_MB` default and max 32768;
   - runner messages, env.example, the runbook, and the ENGINE_IO.md §7.1 A5 row marked IN FORCE;
   - tests updated in settings, contract_parity, supervisor, runner, overlay_runner and e2e_fast.
   - The eval CLI keeps `--rss-limit-mb 4096`, so G1 run conditions are unchanged.
5. **zs re-pin.** Both research configs were re-fingerprinted. Only `line_job.py` and E3 `config.py` / `engine.py` / `metrology.py` moved. Z2 calls `_flat_start` and `_reseat_plane` with defaults and no branch record, so its behaviour is unchanged by the patch.

   | Pin | Fingerprint | Pin sha | Notes |
   |---|---|---|---|
   | zs_lane_v1 | `7bc37b14…` | `80e8fe9b…` | |
   | zs_lane_v3 rev 3 | `6a119bb0…` | `9d19a7c7…` | config_v3 `e84d625a…`, via `make_pin.py` |

   `DEFAULT_LANE` stayed zs_lane_v1. These values are valid only for that exact bundle, so they must be regenerated on re-apply.

   Tests updated: `test_v1_and_v2_fingerprints_unchanged`, `test_v3_pin_matches_config_and_verifies`, `test_v3_pin_fails_closed_on_the_second_model` and `test_rev2_run_lane_v3_passes_the_pinned_budget`.

## Parallel resample mode (Windows)

The corpus-max scan pc_2714 was run serial and with `spawn` workers:
- resample readings, characteristics and every engine file are identical, except `resample_check.json` timings;
- wall time is 96.5 s against 131.5 s;
- RSS is 1.0 GiB against 0.63 GiB.

**Choice: keep serial as the default.** Spawn is byte-identical here and avoids fork, but it was not run on Windows. Supervisor RSS accounting of spawned children on Windows is also unverified. It does not fix the dense failure.

## Decisions needed before a re-attempt (not taken here)

1. **Dense input.** Options:
   - (a) Ship e3-1.13.0 without thinning. Dense scans then rely on the budget fail-safe: all decided characteristics withheld, and B_n5 ran to 175.8 s, close to the 180 s stop.
   - (b) Find a thinning or registration that does not enter the pose mode on B, and validate it on all dense meshes.
   - (c) Make the net resample the raw scan differently, so that it perturbs the pose (for example a grid-offset thinning per resample).

   Each of these is a new engineering change.
2. **Runtime.** On this loaded host, unthinned meshes within the native density range (B_n2c, B_n3, 1.9–4.3 M points) used the full budget and were withheld. Station timing is unmeasured.
3. **PREREG.** On re-apply, `v3/PREREG_FRESH_TEMPLATE.json` still binds `dc23a905…` / `8e002f3e…`. The re-pin changes the v3 fingerprint, so the PREREG primary-arm fingerprint needs an owner update.

## Ledger

| Seq | Kind | Content |
|---|---|---|
| 25 | NOTE | Owner decisions: the waiver and the bundle approval, quoting the owner. |
| 26 | NOTE | RESULTS: re-pin attempted, gate failed, rolled back. It binds this file and `results.json`. |

No RE-PIN entry was written, because nothing was re-pinned. Ledger verify is OK.
