# zs_lane_v3 revision 3 freeze (2026-10-26): a density-invariant budget step

> **What changed.** Only the budget step: `budget.py`, plus `core.py` passing `budget.cell_mm`. These are unchanged: the models, threshold, post-processing, EXT and Z3R, the trigger (3 000 000 faces in the file header), Z2/Z3, snap, evidence, scorer and gold.
> - Revision 3 **supersedes revision 2** (freeze-ledger NOTE seq 22, fingerprint `ab31d061…`, config `1c3f8cb7…`) before any fresh-scan run. Ledger NOTE: **seq 24**.
> - **This is a research freeze only.** The production pin `zs_lane_v3.pin.json` stays revision 2 until a separate re-pin event (see `../REPIN_PLAN.md`, "Revision 3"). `pin.DEFAULT_LANE` stays `zs_lane_v1`. ENGINE_IO-A5 was not applied.
> - Every accuracy number below is consumed-DEV out-of-fold (125 scans), or densified copies of 4 DEV scans checked against the same lane at native density.
> - N1 was used for timing only (lines discarded). TEST-B, DEV-CHECK and FRESH were not run or scored.
> - Not touched: E3, gold, labels, zs_lane_v1 / v2 (`af87df2f…` / `39c0efb9…`), every production file, the zs_rust crate.

## Pins

| Item | Revision 3 |
|---|---|
| Code fingerprint | **`dc23a905bc2cf83a2fff2a4e1270ca654e95b8d3d6f2077db97c50559c85476f`**. Same 68 files as revision 2; only `budget.py` and `core.py` changed. |
| `config_v3.json` sha256 | **`8e002f3e9dac696fc6072d2d6a3abca6d964bd9704a25b54dd8f3b084b08b8b4`**. Changes: `revision: 3`; `budget: {max_faces: 3000000, cell_mm: 0.3, rule}`; `revision_note`; `note`. |
| Models | Unchanged: zs `ade6cde4…ffeb`, z3 `62e99d13…e28c`. |
| Production pin | **Unchanged, revision 2** (`115b7077…`). |
| Revision-2 copies (`rev2_archive/`) | `config_v3_rev2.json` (`1c3f8cb7…`, byte-identical); `zs_lane_v3_rev2.pin.json` (`115b7077…`); `budget_rev2.py` (`dc1d55ca…`); `core_rev2.py` (`4415ee74…`). The last two equal the pin's file hashes. |

## 1. Diagnosis: why revision 2 lost lines on dense scans

All measurements are on the station's densified meshes, through the production path (`lane_cli.run_lane_v3` with CAD priors).

1. **Revision 2's cell grew with density.** The ladder started at `mean edge × sqrt(F / 3M)`.
2. **A fixed cell alone did not fix it.** This was the suggested starting point (fixed cell 0.25 mm, 6-bin normal key).
   - A: 6/7 + 1 FP at 4 M faces, down to 1/7 + 4 FP at 40 M.
   - B: 16 → 8 of 16 lines.
   - C: 12 → 8 of 12.
   - Output face counts still rose with density (B 2.7 → 4.4 M). The vertex clustering's normal-bin cells split more often as the vertices per cell grew.
   - Without the bins (0.25 mm, `none`), and with an opposite-normal split, results were better but still not stable:
     - A at 40 M: 3/7 + 4 FP;
     - B at 46 M: 14/16;
     - C at 35 M: 9–10/12.
   - At 0.45 mm with the opposite-normal split: A at 40 M 3/7.
3. **Root cause.** Vertex clustering of a mesh much denser than the cell builds its output faces from whichever fine triangles straddle 3 cells. Around every 4-cell junction this leaves overlapping and missing faces: on A at 40 M faces, 1.8 % of the faces fold over and there are 31 k crack edges.
   - Z3's slices then break up at the bends.
   - On the lost lines, the candidate's `support_frac` fell from 0.93 to 0.43 and `n_parts` rose from 2 to 6. The pooled GBM dropped them (p 0.99 → 0.10–0.18).
   - Z3 produced 31–42 candidates where it produced 9 at native density.

## 2. The rule (no part-specific logic)

Above the unchanged trigger (more than 3 000 000 faces in the header), the scan is **resampled on a fixed absolute grid** (`budget.py`; full detail in `RUST_NOTE.md`):

1. **Grid.** Cells of h mm, with corners at integer multiples of h in the scan frame. No data-derived origin and no density term.
   - h is the first value of the ladder `0.30 × 1.12^k` whose output has at most 3 M faces (2 per crossing).
   - The crossing count is a property of the surface, so h does not depend on the density either: every densified copy of a part got the same h (table in §4).
2. **Crossings.** Every point where a scan triangle crosses a grid line, with that triangle's normal and the interpolated file normal.
3. **Skin separation** (kept from revision 2's intent). Within a cell, crossings more than 120° from the cell's reference normal form a second vertex. The two skins of a sheet (about 180°) never merge. A bend apex, or even a sharp 90° corner, stays one skin.
   - Revision 2's 6-bin dominant-axis key was dropped: its 45° boundaries split vertices at every bend apex once the input was dense.
4. **Bend-edge preservation** (kept from revision 2). One vertex per (cell, skin), at the QEF minimum of its crossing planes. This is the same regularised quadric as revision 2 (0.05 × trace / 3 towards the mean, mean fallback beyond h). Flats stay on their plane and a bend's two planes meet at their intersection.
5. **Faces.** One quad per crossing: the 4 (cell, skin) vertices around its grid edge, oriented by the scan's winding and split on the shorter diagonal. Exact repeats are kept once.
   - Everything runs in a canonical crossing order, so the output does not depend on vertex or face order (tested on shuffled input).

**Why 0.30 mm.** Sweep with the same resampler on all 14 densified meshes (match to native lines, frozen matcher):

| Cell | Result |
|---|---|
| 0.30 | 14 / 14 exact |
| 0.35 | 2 A meshes add a line |
| 0.40 | 1 A mesh adds a line |
| 0.45 | 1 A mesh adds a line |

The 3 M ceiling then moves B and D to 0.336 and C to 0.376; their lines matched at 0.35 and 0.40 too. The sweep used DEV-derived meshes only. The DEV OOF gate (§3) was checked after the cell was fixed and was not used to choose it.

## 3. DEV accuracy (OOF, `oof/REV3_OOF_SUMMARY.json`, tool `rev3_eval.py`)

**Dump** (`dev_dump`, gold guard, geometry slot):
- **119 / 119 DEV scans within the budget are bit-identical** to the revision-2 dump: candidates, features, Z2 lines, evidence and status.
- The 6 budgeted scans are resampled; per-scan budget info is in `oof/BUDGETED_DEV.json`.

| Scan | Faces in | rev 2 cell | rev 3 cell | rev 3 faces out | Budget step |
|---|---|---|---|---|---|
| sc_147d3123a21d | 3.97 M | 0.546 | 0.663 | 2.51 M | 5.5 s |
| sc_203b95d7a753 (D) | 3.96 M | 0.258 | 0.336 | 2.60 M | 4.6 s |
| sc_3b3e8c9f04e3 | 3.44 M | 0.243 | 0.300 | 2.79 M | 4.0 s |
| sc_741c10b768dc | 3.18 M | 0.528 | 0.743 | 2.45 M | 5.0 s |
| sc_cfe91a0336d4 | 4.15 M | 0.469 | 0.592 | 2.61 M | 5.4 s |
| sc_ead69b8181a8 | 4.80 M | 0.632 | 0.743 | 2.42 M | 6.1 s |

| Arm | micro F1 [95 % CI] | family-macro | TP / FP / FN | Control FP lines | Rolled FP |
|---|---|---|---|---|---|
| v3 revision 1 = revision 2, v3 fold models (frozen) | 0.9660 [0.951, 0.979] | 0.9534 | 597 / 20 / 22 | 1 | 0 |
| **rev 3, v3 fold models (production-faithful: models fixed, only inputs change)** | **0.9660 [0.951, 0.979]** | **0.9534** | **597 / 20 / 22** | **1 (28870000)** | **0** |
| rev 3, full procedure refit on the rev-3 dump | 0.9612 [0.945, 0.975] | 0.9478 | 595 / 24 / 24 | 1 | 0 |
| (rev 2, full procedure refit, for reference) | 0.9651 [0.949, 0.979] | 0.9524 | 595 / 19 / 24 | 1 | 0 |

**Fixed-model arm.**
- Δ vs v3 is 0.0000 [0, 0], with 0 families better and 0 worse.
- **Every one of the 6 budgeted scans keeps its TP / FP / FN** (2/0/0, 2/0/0, 1/0/0, 2/0/0, 2/0/0, 2/0/0). Their lines move geometrically; the 119 others are identical.
- No new control or rolled FP.
- Against v1: +0.0216 [+0.0034, +0.0458].

**Refit arm** (disclosed; not shipped, because the final models are not refit).
- Δ micro −0.0048 [−0.0109, 0.0000] and Δ macro −0.0057 [−0.0120, −0.0004], with 2 families better and 7 worse.
- **All 9 count changes are on unbudgeted scans.** Their inputs are bit-identical; the fold models refit on the 6 changed scans' rows, which is refit noise, as in revision 2.
- No budgeted scan changed its counts, and no new control or rolled FP appeared.
- This arm is weaker than revision 2's refit (0.9651), which is worth knowing if the models are ever refit.

**DEV CLI consistency** (`dev_cli/CONSISTENCY.json`; `run_cohort.sh`, guard and slot, `--strict-pins`): **125 / 125 identical line sets** against the dump pipeline with the final models, and 125 / 125 pins_ok at `dc23a905…`.

**Verdict.** DEV accuracy is unchanged. Revision 3 ships revision 1's models.

## 4. Dense meshes vs native density (`dense/DENSE_MATCH.json`)

**Method.**
- Production path (`lane_cli.run_lane_v3`, CAD priors, v3 models and lane params), with the revision-3 budget.
- Each mesh was run in its own process through the slot, at 1 thread.
- "Native" is the same lane on the original scan. For D that original (3.96 M faces) is itself budgeted.
- Match: frozen matcher (5° / 4 mm / 0.6 / 0.6, Hungarian) with the native lines as reference. The fraction is matched / max(n_dense, n_native).
- DEV gold: frozen matcher against the DEV gold of record.
- Controls `n3c` and `n2c` are the same N without jitter or noise.

| Set | Mesh | Faces in | rev 3 cell | Lines | Native | **Matched** | **Fraction** | DEV gold (rev 3) | DEV gold (rev 2, station) |
|---|---|---|---|---|---|---|---|---|---|
| A | native | 0.49 M | – | 5 | 5 | – | – | 5/7 FP 0 | 5/7 FP 0 |
| A | N=3 | 4.41 M | 0.300 | 5 | 5 | 5 | **1.000** | 5/7 FP 0 | 7/7 FP 0 |
| A | N=3 control | 4.41 M | 0.300 | 5 | 5 | 5 | **1.000** | 5/7 FP 0 | – |
| A | N=5 | 12.25 M | 0.300 | 6 | 5 | 5 | **0.833** | 6/7 FP 0 | 5/7 FP 0 |
| A | N=6 | 17.63 M | 0.300 | 5 | 5 | 5 | **1.000** | 5/7 FP 0 | 4/7 FP 0 |
| A | N=9 | 39.68 M | 0.300 | 5 | 5 | 5 | **1.000** | 5/7 FP 0 | **2/7 FP 3** |
| B | native | 0.94 M | – | 16 | 16 | – | – | 16/16 FP 0 | 16/16 FP 0 |
| B | N=2 | 3.77 M | 0.336 | 16 | 16 | 16 | **1.000** | 16/16 FP 0 | 16/16 FP 0 |
| B | N=2 control | 3.77 M | 0.336 | 16 | 16 | 16 | **1.000** | 16/16 FP 0 | – |
| B | N=3 | 8.49 M | 0.336 | 16 | 16 | 16 | **1.000** | 16/16 FP 0 | 16/16 FP 0 |
| B | N=5 | 23.59 M | 0.336 | 16 | 16 | 16 | **1.000** | 16/16 FP 0 | 10/16 FP 0 |
| B | N=7 | 46.23 M | 0.336 | 16 | 16 | 16 | **1.000** | 16/16 FP 0 | **6/16 FP 0** |
| C | native | 1.41 M | – | 12 | 12 | – | – | 12/12 FP 0 | 12/12 FP 0 |
| C | N=2 | 5.65 M | 0.376 | 12 | 12 | 12 | **1.000** | 12/12 FP 0 | 12/12 FP 0 |
| C | N=3 | 12.71 M | 0.376 | 12 | 12 | 12 | **1.000** | 12/12 FP 0 | 11/12 FP 0 |
| C | N=4 | 22.60 M | 0.376 | 12 | 12 | 12 | **1.000** | 12/12 FP 0 | 8/12 FP 0 |
| C | N=5 | 35.32 M | 0.376 | 12 | 12 | 12 | **1.000** | 12/12 FP 0 | 10/12 FP 0 |
| D | native (budgeted) | 3.96 M | 0.336 | 2 | 2 | – | – | 2/2 FP 0 | 2/2 FP 0 |
| D | N=2 | 15.85 M | 0.336 | 2 | 2 | 2 | **1.000** | 2/2 FP 0 | 2/2 FP 0 |
| D | N=3 | 35.66 M | 0.336 | 2 | 2 | 2 | **1.000** | 2/2 FP 0 | 2/2 FP 0 |

**Summary.** Every native line is reproduced on all 16 dense meshes (157 of 157 native lines matched, counted per mesh). The set-level match is exact on 15 / 16.

**The one exception is A N=5** (12.25 M faces), which adds one line.
- It is a real bend: gold scores it TP, giving 6/7 against native 5/7.
- Its candidate scores 0.911 on the resampled mesh. At native density the same bend's candidates score 0.369, below the 0.5 threshold.
- So it is a borderline bend that native density misses, not a false line.
- It does not appear on A at N=3, 6 or 9, nor in an earlier rev-3 candidate run at N=5: that run used a skin threshold of 90° instead of 120°, and all 14 meshes were exact there.
- The skin threshold was not tuned to remove it. The 120° value was set to stop floating-point splits at sharp right-angle corners, a determinism fix covered by the shuffled and densified synthetic test.

**Against DEV gold:**

| Revision | TP / gold over the 14 dense meshes | FP |
|---|---|---|
| Revision 3 | 137 / 144 | 0 |
| Revision 2 | 111 / 144 | 3 |

Revision 3 equals native density on every mesh except A N=5, which has one TP more.

The output size is also flat in density. For example, B gives 2 502 465 faces at 3.8 M input and 2 505 295 at 46 M; A gives 1 520 588 at 4.4 M and 1 522 616 at 40 M.

## 5. Runtime and memory (`RUNTIME.json`; 1 thread, one process per scan, one geometry slot)

| Run | n | P50 | P95 | Max | Peak RSS |
|---|---|---|---|---|---|
| DEV CLI rev 2 (frozen, `../rev2/dev_cli`; host stall excluded, see FREEZE_REV2) | 125 | 6.65 s | 24.1 s | – | 2.63 GiB |
| **DEV CLI rev 3** (`dev_cli/`, `--strict-pins`) | 125 | **6.78 s** | **20.0 s** | **30.8 s** | **2.72 GiB** |
| **Dense, rev 3** (16 meshes: 14 densified + 2 controls; process wall) | 16 | 16.1 s | 28.7 s | **28.7 s** (B N=7, 46 M faces) | **5.75 GiB** (B N=7) |
| **DEV + dense** | 141 | 7.7 s | **23.6 s** | **30.8 s** | 5.75 GiB |
| Dense, rev 2 (station, for reference) | 14 | | | 31 s | 6.2 GiB |

- **Budget step on dense meshes:** 2.8 s at 4.4 M faces, 18.7 s at 46 M. It scales with the input because the whole input is read.
- **Budget step on the budgeted DEV scans:** 4–6 s. The ladder is predicted from the crossing count at 0.30 mm, then checked exactly.
- **Memory.** On dense input it is set by loading the input mesh: 4.4–5.7 GiB at 35–46 M faces. That is over today's 4 GiB runner cap and under the seq-23 rule clause (< 32 GiB). The A5 status is unchanged; see `station/DENSE_READINESS.md`.
- **Lane after the budget:** at most 3 M faces, as in revision 2.

**N1, timing only** (research pipeline, final models; lines discarded, never scored):

| Scan | Faces | Budget | Process wall | Peak RSS |
|---|---|---|---|---|
| 41018000 | 3.37 M | cell 0.592 → 2.78 M faces (6.0 s) | 34.9 s | 2.78 GiB |
| 45351000 | 2.13 M | not budgeted | 19.4 s | 2.07 GiB |
| 47085000 | 0.96 M | not budgeted | 9.6 s | 1.57 GiB |
| 51101001 | 2.96 M | not budgeted | 24.3 s | 2.12 GiB |
| 41018000 N=2 | 13.50 M | cell 0.592 → 2.78 M faces (11.0 s) | 38.2 s | 3.27 GiB |
| 41018000 N=3 | 30.37 M | cell 0.592 → 2.78 M faces (18.6 s) | 46.1 s | 4.16 GiB |

41018000 is slower in its Z3 stage (rolled shell) than in revision 2 (26.1 s). Its line count is not recorded as a score. Every N1 run is under 60 s.

## 6. Tests (named)

| Tests | Result |
|---|---|
| `tests/test_prod_zs_lane_v3.py` | **35 / 35 passed**, with the default pin and with `SHERMAN_ZS_LANE_PIN=zs_lane_v3` |
| `tests/test_prod_zeroshot_overlay_{rules,lane,runner}.py` + `lane_v{1,2}_contract.py` | 64 / 64 passed with the default (v1) pin |

With `SHERMAN_ZS_LANE_PIN=zs_lane_v3`, `test_prod_zeroshot_overlay_lane.py` has 2 expected failures: the pin is still revision 2 until the re-pin (see `REPIN_PLAN.md`).

Revision-3 tests, new or rewritten:
- `test_rev3_budget_keeps_scans_within_it_unchanged`;
- `test_rev3_budget_resamples_above_it_and_never_merges_the_two_skins` (t = 0.6 mm < 2 cells);
- `test_rev3_cell_is_fixed_and_laddered_only_to_meet_the_ceiling`;
- `test_rev3_resampling_is_density_invariant_and_order_free`: the same surface at 1× and 2× sampling, and with shuffled vertex / face order, gives the same mesh;
- `test_v3_pin_matches_config_and_verifies` and `test_v3_pin_fails_closed_on_the_second_model`, adapted: the pin is the archived revision 2 and fails closed on exactly `budget.py` and `core.py`;
- `test_rev2_run_lane_v3_passes_the_pinned_budget`, extended: a revision-3 budget's `cell_mm` reaches `core`;
- `test_config_v3_is_v2_plus_the_frozen_fixes`: revision 3, `cell_mm` 0.30.

## 7. Files

**New:**
- `research/zeroshot/zs_lane_v3/rev3_eval.py`
- this folder: `FREEZE_REV3.md`, `RUST_NOTE.md`, `RUNTIME.json`, `oof/`, `dev_cli/`, `dense/`, `rev2_archive/`

**Edited:**
- `research/zeroshot/zs_lane_v3/budget.py`: rewritten;
- `research/zeroshot/zs_lane_v3/core.py`: `BUDGET_CELL_MM`;
- `research/zeroshot/zs_lane_v3/freeze_config.py`: revision 3 and `cell_mm`;
- `research/zeroshot/zs_lane_v3/dev_dump.py`: docstring;
- `research/zeroshot/zs_lane_v3/config_v3.json`: re-frozen;
- `research/zeroshot/zs_lane_v3/README.md`;
- `tests/test_prod_zs_lane_v3.py`;
- `v3/PREREG_FRESH_TEMPLATE.json`: the primary arm now points to revision 3 and supersedes revision 2 (seq 22); a context note was added. The decision rule is byte-for-byte unchanged (canonical sha `6ae84658…`, as in seq 23);
- `v3/REPIN_PLAN.md`: a revision-3 note.

**Not touched:**
- `domains/production/**`, including the pin and `lane_cli.py`;
- `contracts/**`;
- E3, gold, labels;
- zs_lane_v1, zs_lane_v2, `z3_merge.py`, `fixes.py`, `lane.py`, `train.py`, `make_pin.py`, `run_cohort.sh`;
- `research/zeroshot/zs_rust`.

The dumps and probe runs are in the session scratch (`scratchpad/r3/`).
