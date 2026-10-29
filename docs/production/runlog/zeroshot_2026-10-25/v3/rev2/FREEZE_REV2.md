# zs_lane_v3 revision 2 freeze (2026-10-25): large-scan runtime and memory

> **What changed: runtime and memory only.** The models, threshold, post-processing, EXT / Z3R parameters, scorer and gold are unchanged.
> - Revision 2 **supersedes revision 1** (freeze-ledger NOTE seq 20, fingerprint `e34e1ba7…`, config `9860f46b…`) before any fresh-scan run.
> - Every accuracy number below is consumed-DEV out-of-fold (125 scans).
> - N1 was used for timing only: no lines were kept and nothing was scored. TEST-B, DEV-CHECK and FRESH were not run.
> - E3 engine code, gold and labels are unchanged. zs_lane_v1 and zs_lane_v2 are unchanged (`af87df2f…`, `39c0efb9…`).
> - `pin.DEFAULT_LANE` stays `zs_lane_v1`.

## Pins

| Item | Revision 2 |
|---|---|
| Code fingerprint | **`ab31d061c6231f5f7581f4cf03d451e77dbc5667e0193e66f3acac955e30a03d`** (68 files = revision 1's 66 + `z3_merge.py` + `budget.py`) |
| `config_v3.json` sha256 | `1c3f8cb7787185f33a61922dda8a2fced900c2e3011ea0105612f4fd5cc0fc53` (adds `revision: 2`, `budget.max_faces: 3000000`) |
| Production pin `zs_lane_v3.pin.json` | `115b70773fa20626ee8ead279e9c648013bd827871decb19374ab77092f17521` (`lane_params.budget`, `revision: 2`; 71 pinned files; `verify() == []`) |
| Models | unchanged: zs `ade6cde4…ffeb`, z3 `62e99d13…e28c` |
| Revision-1 copies | `rev1_archive/config_v3_rev1.json` (byte-identical, sha `9860f46b…`) and `rev1_archive/zs_lane_v3_rev1.pin.json` (`39c4d3b7…`) |

## 1. Profile: where the time went

**N1 wall times were mostly host contention, not the algorithm.**
- 41018000 (1.69 M vertices, 3.37 M faces, one axis family) ran **130 s uncontended under cProfile**. On N1 it took 514–1535 s.
- The same stage on the same scan varied by up to 130× between N1 arms: `snap_all` took 817 s on N1 and 6 s uncontended; Z2 on 47085000 took 18 s in one arm and 233 s in another.

**The real hotspot was Z3 `merge_segments`.** It used 99 s of the 130 s (Z3 107 s; Z2 12 s; snap 6 s; evidence 4 s; I/O 3 s).
- The frozen loop restarts its full O(n²) pair scan after every merge, and each pair runs a Python per-point offset median.
- On a rolled shell all 820 segments are parallel, with the same angle and bisector, so the cheap tests prune nothing: 2.8 M medians and 16.6 M `_unit3` calls.
- Without the profiler the merge alone took 50 s on 41018000 and 6.6 s on 51101001. On the large DEV scans it is negligible (≤ 0.1 s).

**Memory grows with mesh size.** On DEV it is about 1.1 GiB per million vertices at peak: 3.27 GiB at 2.4 M vertices, because Z3's mesh was still resident while Z2 loaded.
- FRESH (headers only) holds customer scans up to **5.6 M vertices** (2 of them over 3 M).
- Without a budget those scans would exceed the runner's 4 GiB zs limit.

## 2. The fix (no part-specific logic)

1. **`z3_merge.py`: an exact bounded merge.** Same output as the frozen function, with bounded work.
   - A pair is skipped only when it is provably rejected:
     - it was already tested, and both items are unchanged;
     - or a vectorised prefilter puts it on the reject side of an original test by a margin (1e-9 on cosines, 1e-6 mm on lengths). The offset median uses a count bound: at least ⌊n/2⌋+1 offsets above the threshold implies median > threshold.
   - Every other pair runs the original scalar code, in the original loop order.
   - `detect` is the frozen `z3s.detect`, verbatim except for the merge (checked by a test through the AST).
   - Result: 41018000 50 s → 3.5 s; 51101001 6.6 s → 1.0 s. Outputs are equal on 41018000, 51101001, 2 large DEV scans and 3 synthetic seeds.
2. **Z2 runs before Z3.** Only Z2's P / N / t are kept, for the snap fallback. The two stages are independent and deterministic, so the results are unchanged; the peaks no longer add (at 2.4 M vertices: 3.26 → 2.74 GiB).
3. **`budget.py`: a 3 000 000-face budget, about 1.5 M vertices**, checked from the file header.
   - Above the budget the lane runs on a decimated copy of the scan.
   - Decimation is **normal-aware quadric vertex clustering**:
     - the cluster key is the grid cell plus the signed dominant normal axis, so opposite skins are never merged;
     - each cluster's vertex is placed by the quadric error metric (planes of the incident faces), regularised to the cluster mean and clamped to the cell;
     - the cell size is the first step of a deterministic geometric ladder that meets the budget.
   - The step is chunked, int32 and memory-lean. A lean rewrite was checked **bit-identical** to the first version on the 6 budgeted DEV scans, 41018000 and a 13.7 M-face stress mesh.
   - Within the budget the scan is used unchanged.

## 3. DEV accuracy (OOF, `oof/REV2_OOF_SUMMARY.json`)

**Dump.**
- **119 / 119 DEV scans within the budget are bit-identical** to the revision-1 dump: candidates, features, Z2 lines and evidence.
- 6 scans are over the budget: sc_147d3123a21d, sc_203b95d7a753, sc_3b3e8c9f04e3, sc_741c10b768dc, sc_cfe91a0336d4 and sc_ead69b8181a8. Their decimation runs from 3.2–4.8 M faces down to 2.5–2.9 M.

| Arm | micro F1 [95 % CI] | family-macro [CI] | TP / FP / FN | Control FP lines | Rolled FP |
|---|---|---|---|---|---|
| v3 revision 1 (frozen) | 0.9660 [0.951, 0.979] | 0.9534 [0.923, 0.977] | 597 / 20 / 22 | 1 (28870000) | 0 |
| **rev 2, v3 fold models (production-faithful: models fixed, only inputs change)** | **0.9660 [0.951, 0.979]** | **0.9534 [0.923, 0.977]** | **597 / 20 / 22** | **1** | **0** |
| rev 2, full procedure refit on the rev-2 dump | 0.9651 [0.949, 0.979] | 0.9524 [0.922, 0.977] | 595 / 19 / 24 | 1 | 0 |

**Paired comparison with revision 1** (10 000 resamples, seed 20261025):
- **Fixed-model arm:** Δ 0.0000 [0, 0], with 0 families better and 0 worse. 119 line sets are identical. The 6 budgeted scans' lines move geometrically, but every scan keeps the same TP / FP / FN.
- **Refit arm:** Δ micro −0.0009 [−0.0055, +0.0032] and Δ macro −0.0010 [−0.0058, +0.0038], with 3 families better and 4 worse.
  - All 7 count changes are on **unbudgeted** scans: the fold models refit on 6 changed scans' rows, which is refit noise.
  - No budgeted scan changed its counts.
- **Against v1:** +0.0216 [+0.0034, +0.0458] for the fixed-model arm and +0.0207 [+0.0019, +0.0449] for the refit arm.
- **No new control or rolled false line in either arm.**

**Verdict:** accuracy is unchanged. The final models are not refit; revision 2 ships revision 1's models.

## 4. Runtime and memory

All runs were one process per scan through guard and slot, with `--strict-pins` and `nice`.

| Run | n | P50 | P95 | Max | Peak RSS |
|---|---|---|---|---|---|
| DEV CLI, revision 1 (frozen) | 125 | 6.59 s | 21.7 s | 31.7 s | 3.27 GiB |
| DEV CLI, rev 2 at `ea202aa5` (pre-lean budget step; identical lines) | 125 | 6.63 s | 19.4 s | 28.0 s | 2.70 GiB |
| **DEV CLI, rev 2 final `ab31d061` (`dev_cli/`)** | 125 | 6.65 s | 24.1 s | 2017 s* | 2.63 GiB |
| N1 timing, rev 2 (41018000 / 45351000 / 47085000 / 51101001) | 4 | | | 26.1 / 18.3 / 8.9 / 21.7 s | 2.52 / 2.08 / 1.56 / 2.11 GiB |
| N1, as run on 2026-10-25 (v2 = the revision-1 pipeline) | 4 | | | 1535 / 39 / 24 / 46 s | ≤ 2.6 GiB |
| Stress: DEV scan subdivided ×4 (6.9 M vertices / 13.7 M faces) | 1 | | | 13.0 s | 3.29 GiB |
| Stress: DEV scan subdivided ×4 (9.6 M vertices / 19.2 M faces) | 1 | | | 20.3 s | 3.25 GiB |
| Production rehearsal, real `lane_cli` zs step, v3 pin final (`rehearsal/`) | 50 | 8.25 s | | 16.6 s | 2.58 GiB (v1 pin: 7.92 / 15.6 s, 3.24 GiB) |

\* **Host stall during the final DEV CLI.**
- Four scans finished between 18:14 and 18:28: 2017, 501, 183 and 160 s.
- Their slow stage was unrelated to size: Z2 took 1602 s on a scan that ran in 14 s; Z3 took 156 s on a 4 s scan.
- The same four scans took 26.8, 11.6, 5.7 and 4.1 s at `ea202aa5`, whose code path for unbudgeted scans is identical.
- Even with the stall counted, P95 is 24.1 s.
- Excluding it, the final run matches the `ea202aa5` run.

**Operational note for the fresh run.** The rule's runtime clause is measured from `run_record.json`. N1 and this run show that host contention or suspension can inflate wall time 10–100×. Run the fresh cohort on an otherwise idle host.

The N1 stress / timing runs, the dump and the pre-lean CLI outputs are in the session scratch.

**Rehearsal result.**
- 50 / 50 OK, 50 valid overlays, **0 false PASS**, 0 lowered to REVIEW.
- 288 lines, identical kinds to the v1 pin (WITHHELD 15 / UNEXPECTED 4 / REFUSED 3).
- EXT fired once.
- No scan changed its effective state.
- 0 runs over 90 s or 4 GiB.

## 5. Tests (named)

| Tests | Result |
|---|---|
| `tests/test_prod_zs_lane_v3.py` | 33 / 33 passed, default pin and `SHERMAN_ZS_LANE_PIN=zs_lane_v3` |
| `tests/test_prod_zeroshot_overlay_{rules,lane,runner}.py` | 50 / 50 passed, default (v1) and v3 pin |
| `tests/test_prod_zeroshot_lane_v{1,2}_contract.py` | 14 / 14 passed |

The v3 test file now has 7 new revision-2 tests:
- the bounded merge equals the frozen merge on 3 synthetic seeds;
- `detect` is verbatim (AST);
- the budget leaves in-budget scans unchanged;
- the budget decimates and never merges the two skins;
- decimation is deterministic;
- `lane_cli` passes the pinned budget;
- the config and pin carry revision 2 and the budget.

## 6. Files changed

**New:**
- `research/zeroshot/zs_lane_v3/z3_merge.py`
- `research/zeroshot/zs_lane_v3/budget.py`
- `research/zeroshot/zs_lane_v3/rev2_eval.py`

**Edited, in `research/zeroshot/zs_lane_v3/`:**
- `core.py`: budget step, Z2 first, the bounded merge;
- `fingerprint.py`: the two new files;
- `dev_dump.py`: the budget and `--no-budget`;
- `freeze_config.py`: `revision` and `budget`;
- `make_pin.py`: pins the budget and revision; skips copying a model whose sha already matches;
- `config_v3.json`: re-frozen.

**Edited elsewhere:**
- `domains/production/zeroshot/lane_cli.py`: passes `lane_params.budget`;
- `domains/production/zeroshot/assets/zs_lane_v3.pin.json`;
- `tests/test_prod_zs_lane_v3.py`;
- `v3/PREREG_FRESH_TEMPLATE.json`: primary arm → revision 2 fingerprint and config; the decision rule is untouched (OWNER_CONFIRMED, ledger seq 21).

Not touched: `train.py`, `fixes.py`, `lane.py`, `run_cohort.sh` and every frozen v1 / v2 / Z2 / D1 / E3 module.
