# zs_lane_v4 working notes

Running log. Newest entries at the bottom. Read this first after a restart.

## 2026-10-29 07:30 start

- **Scope.** Owner goal (2026-10-29, freeze-ledger NOTE seq 48): v4 should beat v1. Every gold-labelled real scan is the v4 DEVELOPMENT pool; nothing in-house is held-out for v4. v1 stays the production default. Promotion is the owner's decision after a fresh-scan test.
- **Hard rules kept.** No edits to the pinned v1/v3 lanes, their pins, the `domains/production/zeroshot` defaults, gold or past ledger entries. Heavy geometry only through `docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh`. No installs. No Rosetta. No `.env`.
- **Pool.** `POOL.json` (`tools/build_pool.py`): 217 scans = DEV 125 + TEST-B 61 + N1 4 + FE1 27. 207 line scans, 10 zero-bend controls, 152 families (part_id[:5]), 1148 gold lines.
- **Pinned lanes on the pool** (`tools/score_pinned.py` → `results/PINNED_POOL.json`; stored outputs only, post-hoc, DEV in-sample for both):

  | Lane | TP/FP/FN | micro [CI] | macro [CI] |
  |---|---|---|---|
  | v1 | 1028/73/120 | 0.914 [0.874, 0.948] | 0.925 [0.897, 0.950] |
  | v3 | 1023/66/125 | 0.915 [0.868, 0.952] | 0.926 [0.895, 0.953] |

  FN causes (v1): MISSED 51, SPAN<60 % 54, OFFSET 13, TAKEN 2. FP causes (v1): SPAN 33, TAKEN_BY_OTHER 13, OFFSET 2, SPURIOUS 25.
- **Lanced-tab merge (gb_68cd06), first look from stored outputs.** v3's 4th line is op `keep`, fix `EXT`, 413 mm long. It started as v1's 52.5 mm tab line (z2+z3, p 0.965). EXT adopted the extent of the absorbed Z2 line, which runs along the whole row of tabs. EXT's dedupe then dropped the separate tab lines, because each now "duplicates" the 413 mm line. To confirm with the cache.
- **Cache.** `tools/cache_build.py` (one scan per process through the slot) writes `cache/<case_id>.json.gz`:
  - `v1`: the pinned v1 blind pipeline with every candidate snapped;
  - `v4`: `research/zeroshot/zs_lane_v4/core.process`, i.e. the v3 rev3 detection stages plus support profiles and metadata;
  - `v3x`: budgeted scans only.

  `tools/validate_cache.py` replays the pinned selections and must reproduce the stored outputs exactly. The first scan reproduces both lanes exactly. The full build was launched at 07:52 (`tools/run_cache.sh`, log `cache/run_cache.log`).

## 2026-10-29 08:05 protocol hashed

- `PROTOCOL.md` sha256 `2f8206d502c1007219bf3b1d364daa4515c8d075af8013c5079db5e48515d5b1`.
- Freeze-ledger NOTE **seq 49** (`frz_9fe194a3e30c`, entry sha `1f8ff8ff…`), appended through `tools/ledger_note.py`. The ledger verifies (49 entries).
- At the time of hashing:
  - only the pinned stored outputs had been scored;
  - the cache reproduction had been checked on one scan;
  - no v4 variant had been scored.
- From here on, every LOFO-scored v4 variant goes to `results/DESIGN_LOG.jsonl`.

## 2026-10-29 08:40 diagnosis, first pass (206 of 217 scans cached)

- **Lanced-tab merge: root cause confirmed** by replaying v3 step by step on the cache (`tools/diagnose.py`, `v3_steps`) on `gb_68cd06`.
  - Z2 line 3 is 413 mm long and runs through the whole row of tabs (cover 0.886). It pooled onto Z3 candidate 3, a 52.5 mm piece that is itself an FP in both lanes (two tabs merged by Z3).
  - **EXT** extended candidate 3 to 413 mm.
  - **EXT's dedupe** then dropped the 9 selected tab lines (candidates 5, 6, 8, 9, 10, 11, 13, 14 and 15), because each was "duplicated" by the 413 mm line.
  - **Z3R** could not rescue tabs 7 and 12 (p_z3 0.94 and 0.955), because they too duplicate the EXT line.
  - The v2 join was **not** the cause: every tab pair has cb = False.
  - Without EXT, v3 would output 3 long lines + 9 tabs + 2 Z3R tabs = 14 matches.
- **Taxonomy** (`results/DIAGNOSIS.json`, v1 lane, 206 scans):
  - FN 118: NO_CAND:SPAN 50 (a candidate on the fold has the wrong extent), CLS_REJECT 45 (a matching candidate exists but the GBM gives p < 0.5), NO_CAND:FAR 19 (no candidate near: hems, ears and others), TAKEN 2, SNAP_LOST 1, NO_CAND:OFFSET 1.
  - FP 65, all from the classifier: SPAN 31, SPURIOUS 19, TAKEN_BY_OTHER (duplicates) 13.
  - v3 adds EXT_DEDUPE 9 and EXT_ALTERED 3 FN, plus 6 EXT FP and 1 JOIN FP.
- **SPAN misses** (`tools/span_analysis.py`, 50 rows):
  - 21 are "long": in 12 a Z2 line has the right extent; 9 need clipping.
  - 25 are "short": 17 are fixed by the union with collinear neighbours, 13 by the Z2 extent or the union with it.
  - 4 are both long and short.
  - In 25 cases the attached Z2 line, as is, matches the gold. Pooling discards that geometry.
- **Short ears** (`gb_138090` / `gb_1f3fe0`): the gold lines are 5–6 mm long, and one marker did not mark them. On `gb_138090` candidates exist with p ≈ 0.001; on `gb_1f3fe0` there is no candidate with the right span. They are not targeted specifically.
- **Hem generator probe** (`research/zeroshot/zs_lane_v4/hem.py`, crown sweep + two-layer stack). The first version fired on the tilted flanges of the strip's V-folds (50 per strip). After requiring flat parallel skins (normal within 12° of ±n, IQR ≤ 0.35 t, slab ≥ 60 % skin), 3 of the 6 strip and open hems on 4 scans matched, with 1 FP and short partial runs on `gb_959f4a`. The next version grows runs beyond the edge cluster and dedupes crowns found twice.

## 2026-10-29 09:15 cache complete; baselines and the first LOFO variants

- **Cache.** All 217 scans cached (`cache/run_cache.log`, 0 failures).
  - Replaying the pinned selection on the cache reproduces the stored outputs: v1 217/217 exactly; v3 215/217 exactly, and the other 2 within 0.14 µm with identical TP/FP/FN.
  - FastScorer equals the frozen matcher on 1414 line sets.
- **Diagnosis written:** `DIAGNOSIS.md`.
- **LOFO baselines** (`results/LOFO_BASELINES.json`). A0/A2 are the stored pinned outputs, in-sample on DEV.

  | Arm | TP/FP/FN | micro [CI] | macro [CI] | ctrl FP |
  |---|---|---|---|---|
  | A0 v1 pinned | 1028/73/120 | 0.9142 [0.874, 0.948] | 0.9251 [0.897, 0.950] | 0 |
  | A1 v1 retrained | 1007/95/141 | 0.8951 [0.852, 0.931] | 0.9064 [0.874, 0.936] | 2 |
  | A2 v3 pinned | 1023/66/125 | 0.9146 | 0.926 | 0 |
  | A3 v3 retrained | 1001/72/147 | 0.9014 | 0.9094 | 2 |

  - A1 − A0 = −0.019 [−0.034, −0.005]. That is A0's in-sample advantage on DEV.
  - On non-DEV scans (TEST-B + N1 + FE1, held-out for the pinned lanes):
    - A0: 0.8422 / 0.8644
    - A1: 0.8346 / 0.8626
    - v4_01: **0.8574 / 0.8748**
- **LOFO variants** (DESIGN_LOG, micro / macro / control FP):

  | Variant | micro | macro | ctrl FP |
  |---|---|---|---|
  | v4_00 base26 + v3 selection (EXT v3, no Z3R) | 0.9060 | 0.9136 | 2 |
  | v4_01 base26 + safe EXT | 0.9137 | 0.9165 | 2 |
  | v4_02 all v4 features + safe | 0.9030 | 0.9088 | 5 |
  | v4_06 + sweep | 0.9115 | 0.9150 | 2 |
  | v4_07 + all but scan-level | 0.9051 | 0.9181 | 4 |
  | v4_08 + scan-level | 0.9124 | 0.9152 | 3 |
  | v4_09 base26 + nested 96-grid | 0.9020 | 0.9118 | 3 |
  | v4_10 extent hypotheses | 0.8934 | 0.8967 | 5 |
  | v4_12 base26 + safe_sup EXT | 0.9128 | 0.9213 | 2 |
  | v4_13 base26 + no EXT | 0.9119 | 0.9196 | 2 |

  v4_09 used the nested 96-option grid. The inner CV mostly picked thr 0.4, which does not transfer to the outer models.
- **Lessons:**
  1. The new candidate features overfit across families. base26 is best; the extra feature groups lose 0.2–1.1 points of micro.
  2. Nested selection of the threshold hurts: inner OOF models are less confident than the outer refit.
  3. Extent hypotheses compete badly (the GBM cannot rank extents).
  4. The gain over A1 comes from the v3 candidates plus v2 post-processing (sweep gate, plate-continuity join) with EXT made safe. v3's EXT costs about 0.006 micro under LOFO; safe or safe_sup EXT recovers it.
- **Pass 2** (`tools/cache_pass2.py`: hem generator and Z3 re-slice stability) is running. So far, hems are found only on the 6 FE1 hem scans.
- **Hem probe history** (`results/hem_probe/`):
  - flat-parallel-skin test;
  - run growth;
  - crown dedupe;
  - crown roundness (≤ 35 % of the normals in one bin);
  - PCA crown line;
  - orientation-free outer layers;
  - **flange-end step test.** The double layer must step down to one layer of local thickness t_local, with S ≥ 1.6 t_local. This killed 5 FPs on the rolled control `sc_94da60c3c13d`, whose global t estimate (0.54 mm) is wrong for a 2 mm sheet.

  Final probe: 9/11 gold hems matched, 0 FP on 29 probe scans (10 controls, 12 random line scans and the hem scans' other edges). The design was inspected on these pool scans; this is disclosed in the design log.
