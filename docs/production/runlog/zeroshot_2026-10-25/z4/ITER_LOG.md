# Z4 / Z5 iteration log (DEV only, out-of-fold)

Ledger: consumed development. DEV = SPLIT_FROZEN dev (121 scans: 116 complete-label line scans + 5 zero-bend
controls). Folds: the integrator's `../dev_folds.json` (5 folds, family_group5). Matcher: frozen
`score.py score_case` (5 deg / 4 mm / 0.6 both spans, Hungarian, apex, raw frame). No TEST-B / sealed path
opened; no CAD; no filename or part features. Code: `research/zeroshot/z4_selector/`.

## Iter 0 (2026-10-25 ~03:50): why A0 is empty on 36 scans; pool rebuild

- `diagnose_empty.py` -> `a0_empty_diagnosis.json`, read from the stored A0 run logs only.
  Of the 41 DEV scans with zero A0 lines (36 line scans + 5 controls):
  - 36 `POOL_NONEMPTY_SOLVER_SELECTED_ZERO`: the CP-SAT typed union ran to OPTIMAL and counted 0 bend
    components. Of these, 4 line scans have a near-empty pool (1 to 16 candidates: sc_cca7, sc_d82b,
    sc_822f, sc_24c5), so for them the pool stage effectively failed; the other 28 line scans have
    76 to 1130 candidates, so the typed union filtered everything out.
  - 3 `TYPED_UNION_TIMEOUT` (600 s lane budget; pools of 535 to 1118 candidates).
  - 2 `POOL_EMPTY` (sc_9ef2, 4 gold lines; and the control sc_f99d).
  - So the empty outputs are a selection failure in about 31 of 36 line scans, not a birth failure.
- Pool rebuild: decompose + pool only (no typed union). It takes 13 to 60 s per scan, against about 4 h for
  the full lane at 2 slots earlier: the typed union was most of the cost. 103 DEV pools were rebuilt, 3
  workers, one slot each, held for the whole batch (`rebuild_pools.py`). The other 18 come from the
  phase-1 dev work dir.
- **Iter-0 bug, fixed.** Each worker built its share from a list filtered by "pool already exists", and
  that filter ran at start-up. The workers got their slots at different times, so they saw different
  lists and their shares overlapped: 30 scans were built twice and 33 were skipped. The fix: stride
  over a fixed list and skip existing pools inside the loop. The 33 missing pools were rebuilt, and all
  121 DEV pools are now present and parse (103 rebuilt, 18 from phase 1).
- Candidate table (`candidates.py`, tag `base`):
  - Pool rows (candidates and line_objects) are merged at 2 deg / 1.5 mm / 0.8 mutual overlap. That
    gives a median of 642 candidates per scan.
  - 81 features per candidate:
    - the 20 pool descriptors;
    - consensus and generator flags;
    - the jev features_v2 cross-section: dihedral, flange fit RMS, flange widths, apex side, corner
      distance, near-line support, and slice classes;
    - neighbour context: parallel lines within 6 mm, at 6 to 40 mm, and collinear;
    - bbox distance;
    - D1 thickness, the two-skin duplicate distance t/sin(a/2)+1, apex side / t;
    - per-scan ranks.
  - It runs at 1 to 90 s per scan.
- **Pool recall** (`pool_recall.py` -> `pool_recall_base.json`, frozen score_case with every candidate
  kept):
  - **476 / 586 = 0.812** on the 116 complete-label DEV scans, against A0's 0.220.
  - On the 36 A0-empty scans: 88 / 121 = 0.727.
  - 12 scans have pool recall < 0.5, including the near-empty pools (sc_cca7, sc_d82b, sc_822f, sc_9ef2).

## Iter 1: GBM / LR selector, NMS at 4 mm (matcher thresholds)
- OOF over the integrator folds; the threshold is chosen by inner grouped CV on the training folds.
- GBM: AUC 0.980. **micro 0.584 [0.534, 0.627], family-macro 0.509** (P 0.606, R 0.563). 21 scans
  empty, 2 control FP lines.
- LR: AUC 0.965, micro 0.519, macro 0.462. Its threshold ran into the top of the grid (0.95, a
  class_weight effect), so the grid was extended.
- Error budget (GBM):
  - FN: 110 not in pool, 81 in pool scored < 0.2, 39 scored 0.2 to 0.5, 26 suppressed.
  - FP: **175 of 215 are duplicates of an already-matched gold line**. They are parallel, have full
    overlap, and sit 4 to 6 mm from the matched line (median 4.5 mm). They are other-skin or
    bend-zone twins just outside the 4 mm NMS radius.

## Iter 2: wider dedupe (fixed 6 / 8 mm), diagnostic only
- d6: 0.653. d8: 0.677. These were picked by looking at OOF, so they are not a claim.

## Iter 3: dedupe radius chosen jointly with the threshold by inner CV (grid 4, 6, 8, 10, 12, 15 mm)
- Inner CV picks 8 mm in 4 folds and 10 mm in 1.
- **micro 0.675 [0.617, 0.729], family-macro 0.615 [0.543, 0.688]** (P 0.750, R 0.614). 14 scans empty,
  6 control FP lines.
- Error budget:
  - FN: 110 not in pool, 80 in pool scored < 0.2, 22 scored 0.2 to 0.5, 14 suppressed.
  - FP: 44 duplicates, 38 spurious, 26 near-span, 9 near-angle, 3 near-offset.
- Postprocessing A/B (nothing fitted):
  - span repair (trim/extend to the two-arm fold run): 0.664, worse;
  - extend-only: 0.675, neutral;
  - D1 gated outer-apex snap: 0.675, neutral (424 of 486 lines moved). It is kept as the placement step,
    as D1 recommends.

## Iter 4: Z5 CP-SAT global consistency (`z5.py`, `z5_solve.py`, .venv-f1-structured ortools 9.15)
- A first version optimised a weighted independent set over all candidates above the threshold. It
  scored 0.647, below Z4: the global optimum prefers two mid-score lines to one high-score line.
- Rebuilt on the greedy-NMS survivors, so that with no extra terms Z5 equals Z4 exactly. The terms:
  two-skin rule t/sin(a/2)+1 mm (D1 t), flange pairs from the pool plane_table (skins merged),
  strip-evidence penalty, at most one line per flange pair and carrier, near-tree bound.
- Parameters are chosen per fold on inner-CV scores, from a grid of 36.
- Inner CV picked **no flange term in any fold**, and the two-skin rule in 3 of 5.
- OOF **micro 0.677 [0.619, 0.731], family-macro 0.617** (P 0.757, R 0.613): +0.002, which is noise.
- Why the flange graph does not help: the pool plane_table has a median of 7 planes per scan. Only 27 %
  of the true candidates find a flange pair, against 12 % of the negatives, so a strip-evidence
  constraint removes more TPs than FPs. Z5 needs a denser flange segmentation (Z2 / Z6) to work.

## Iter 5: stage-2 stacking (`stage2.py`): neighbour stage-1 scores as features, nested OOF
- AUC 0.980 -> 0.981. **micro 0.686 [0.629, 0.739], family-macro 0.642 [0.571, 0.714]** (P 0.791, R 0.606).
  14 scans empty, 5 control FP lines.

## Iter 6: merged sources, A0 pool + Z2 it05 eval_lines (`candidates.py --source z2=...`)
- Z2 lines are merged with the pool rows. Where a cluster has a Z2 line, Z2's geometry is used, and the
  Z2 metadata (score, cover, stations, ro_mm, sweep) become features, plus `scan_has_z2`.
- Z2 it05 covers 103 / 121 DEV scans. Scored alone on all 116 line scans it gives 0.832, or 0.910 on
  the scans it covers.
- Pool recall 0.937. Selector OOF **micro 0.837 [0.787, 0.878], family-macro 0.782**, P 0.865, R 0.811.

## Iter 7: A0 pool + Z3 it7 candidate stream (with its OOF score) + Z2 it05; Z3 geometry has priority
- Pool recall **583 / 586 = 0.995**.
- Selector OOF **micro 0.895 [0.852, 0.930], family-macro 0.876 [0.825, 0.921]** (P 0.900, R 0.891).
  4 scans empty, 1 control FP line. Inner CV picks 6 mm dedupe in all folds.
- This is below Z3's own pooled Z3+Z2 selector (0.937). The ~650 A0 candidates per scan dilute the
  table, and my per-candidate features are A0-pool descriptors that Z3/Z2-only candidates lack.
- Error budget:
  - FN: 3 not in pool, 28 suppressed by a duplicate, 33 scored low.
  - FP: 22 spurious, 21 duplicates of a matched gold line, 15 near-span.
- Caveat: the Z3 score feature is Z3's own OOF probability on the same folds. For the training rows of
  fold k, that score came from Z3 models that saw fold k: mild stacking leakage, and the same caveat
  applies to any integrator that reuses it.

## Iter 8: Z5 on the iter-7 union
- Inner CV picked no term in any fold (no two-skin, flange, tree or strip term). The result equals Z4:
  0.895 / 0.876.

## Stop (orchestrator wrap-up, 06:40)
- Slots are freed.
- Pools stay in /private/tmp/claude-501/sherman_zeroshot_work/z4_pools (103) and dev/ (18), and the
  feature tables in z4_feats/{base,a0_z2it05,a0_z3_z2}.
