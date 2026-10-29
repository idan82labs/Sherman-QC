# zs coverage run (2026-10-28 to 10-29)

Both frozen zero-shot lanes were run once, blind (no CAD), on every permitted real scan. FE1 was excluded.

- **Freeze-ledger NOTE seq 41** (`frz_b0b27e85055c`) was recorded before the run. It binds:
  - RUN_MANIFEST sha `f80c3b1d…`
  - COV_INVENTORY sha `8e357263…`
  - the tool hashes and the config
- **Config:** the same as the 30820000_A01 demo (seq 34):
  - n_bends 999 (uncapped)
  - t_mm from the Z2 self-recipe estimate (v3 reuses v1's value on the same bytes)
  - max_families 20
  - `lane_cli.run_lane` / `run_lane_v3`, run only after `PIN.verify == []`, with an exact pin-sha and fingerprint check

## Files
- `COV_INVENTORY.json`: one row per unique sha256 (metadata only).
- `RUN_MANIFEST.json`: the 192 permitted scans.
- `runs/<scan_id>.json`: both lanes for each scan. Resumable.
- `work/`: raw worker outputs.
- `SCORES.json`, plus `scores_T2_<lane>_per_case.json`: the tiers. They are never blended.
- `tools/`:
  - `census.py`
  - `build_cov_inventory.py`
  - `cov_worker.py`
  - `run_coverage.py`
  - `score_coverage.py`, run with `--allow-sealed` only after coordinator authorization
  - `gold_setup_cov1.py`

## Labels
- **T2** is post-hoc on consumed data. DEV is in-sample for both lanes, because the production GBMs were fit on all 125 DEV scans.
- **Not scored, because their gold is under `*_SEALED*`:**
  - 61 TEST-B scans (`ai_bend_lines_zs_v1_SEALED`)
  - 18 FRESH-in-DEV scans (`ai_bend_lines_fresh_v1_SEALED`)
- **Reused:** the 30820000_A01 (V) v1 output comes from the owner demo (same pin and config), so it has no wall or RSS figures.
