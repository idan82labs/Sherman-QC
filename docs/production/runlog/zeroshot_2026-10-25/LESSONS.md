# Lessons for the zero-shot programme (2026-10-25)

Everything below is consumed development evidence. None of it is an unseen-family claim. Sources: `final_eval_2026-10-19/{REPORT,FIXES,FIXES2,FIXES3}.md`, `a1/OWNER_DIRECTION.md`, `a1/iter_1..7/CHANGE.md`, `jev_ab_v2/JEV_INPUT_OPT.md`, `jev_ab_e3/JEV_AB_E3.md`, `BROAD_BASELINE_2026-10-17.md`, `broad_baseline/gold11_audit/GOLD11_AUDIT.md`, `GOLD_SET_V1.md`, `GOLD_SET_V2.md`, `BENCHMARK_V3.md`, `docs/research/2026-09-07_production_generalization_roadmap.md`, `docs/research/2026-09-10_strategic_engine_audit_recommendation.md`, and `RESEARCH.md` (this folder).

## 1. No-CAD detection: what was tried and what it scored

| Detector / attempt | Where | Result | Why it fell short |
|---|---|---|---|
| **A0 strict-blind typed union**: surfel/atom graph → flange and bend hypotheses → CP-SAT typed labelling, with many junction/corridor/birth recovery passes (`backend/patch_graph_latent_decomposition.py`, 10.4k lines; `scripts/run_f1_structured_union_e2e.py --disable-birth`) | `research/broad_baseline/run_lane.py` (`best_f1` without `--cad`) | GOLD_SET_V2: **0.340 micro / 0.273 family-macro** on 99 scans (P 0.63, R 0.23). On the NO-CAD list it scores 0.286, and **0.220 on the 42 new no-CAD scans**. Gold11 scan-only: 0.365. | Recall. Of 305 BROAD gold lines, 116 are never proposed as a final line, 40 fail the span rule and 32 the 4 mm offset. Angles are fine (MAE 0.56° when matched). |
| **best_f1** = A0 ∪ gated CAD-guided runtime detections | same | 0.458 Gold11 (historical); 0.445 broad-57; 0.413 GOLD_SET_V2; **0.389 pool** in the one-time final eval | The CAD half adds about +0.09 at the price of more FPs. It is not zero-shot. |
| **F1 V29–V51 / C1–C15 campaigns** (Aug–Sep): line accounting, parent bridges, observed-phase vetoes, explosion guard, axis/span recovery, source recovery, V52 precision fit | `research/f1_*`, `docs/research/2026-08-*` | Legacy37 0.634 vs daily28 **0.197**: transfer collapsed. Source recovery: in-sample 0.341 → grouped LOCO 0.192. Axis/span: 0.632 in-sample → 0.483. V52 precision 1.0 came from 23 cutoffs with full calibration/eval overlap. Many later V-runs ended with **zero output** (F1 undefined). | Rules and cutoffs fitted to small consumed cohorts. Heavy process with gates coupled together. Candidate birth missing: on daily28, 35 of 51 misses were absent from the whole candidate stream. |
| **Jev (TypeSafe System One) filter / ranker** | `research/jev`, `research/bench_labeling/jev_experiment.py` | Filter: F1 flat (0.443 → 0.444). Jev AUC 0.69–0.88, against **0.93 for plain-feature LR**. On E3 lines Jev AUC was 0.51, and every filter removed only true lines. **Track closed.** | It ranks less well than the features it is given, and it wrongly called joggles non-bends. |
| **Liberal pool + selector** (A0's own `f1_uniform_scan_only_candidate_pool_v3`, about 1000 candidates per scan) | `research/jev/pool_v2.py`, `apply_v2.py` | The pool holds **0.83–0.86 of gold lines**, so a perfect selector would reach F1 0.92. LR + NMS: **0.478** vs lane 0.429 on 37 test scans (ΔCI [−0.05, +0.14], not significant). A single feature (two-sided transition ≥ 0.258) gives +0.02, the only significant gain. | On the broad set, **selection** is the bottleneck; on daily28, **birth** is. Both exist, and which one dominates depends on the cohort. |

**The structural diagnosis.** No zero-shot module models **sheet thickness or the two skins** (`RESEARCH.md` §0). Every lane treats the outer and inner skin as unrelated planes. That produces parallel near-duplicate planes, lines on the wrong skin (at 90° the two apex loci are 1.41·t apart, so ≥ 2.9 mm stock fails the 4 mm gate on its own), and doubled lines. It is the leading hypothesis for most OFFSET FNs/FPs and part of the SPURIOUS FPs. The D1 agent is testing it now (`d1/`).

**The phase-1 baseline shows that recall is the gap** (`BASELINE.md`). A0 scores 0.322 on the 116 DEV scans, at P 0.60 and R 0.22.
- **81 % of its FNs are MISSED.** No prediction lies within 15° / 15 mm of the gold line.
- OFFSET accounts for only 26 FNs.
- **A0 emits no line at all on 36 of 116 scans** (33 NO_PREDICTIONS, 3 timeouts). That alone loses 21 % of the gold lines.

The next detector must be built for recall and must not abstain by default. A two-skin fix alone moves little; its value is as the basis for a better segmentation.

## 2. What E3 (the CAD engine) taught us

These lessons transfer directly to a scan-only detector.

- **When E3 has the flange graph it is near-perfect.**
  - GOLD_SET_V1: 0.990 micro / 0.974 family-macro.
  - Gold11 relabel: 0.987.
  - FRESH engine-ran: 0.94 with **0 FP**.
  - Per-bend riser-aware: 192/203.

  Its measurement core is per-flange **outer-mould-line skin plane fits**, with the skin chosen explicitly (the outer of two skin peaks t apart), plus a hinge search and station profiles. **The hard part is knowing which flanges exist and how they connect, not measuring them.** For zero-shot, this argues for building a flange graph from the scan and reusing the E3-style metrology on it.
- **E3's losses are coverage, not precision.** Final eval: 0.527 headline vs 0.857 engine-ran. The causes:
  - STEP compile failures: rolled parts, a joggle on a narrow strip (37126000), weldment read failures;
  - wrong-part refusals: non-round holes taken as circles (fixed in iteration 1), hole-fill tests;
  - mirror refusals;
  - the amendment-7 truth-table gate.
- **Geometric pitfalls already paid for**:
  - A pose off by about t silently swaps skins, which gives wrong lengths and lines.
  - Real inner radii are often much larger than in CAD (2.6 mm vs 0.8 mm), and springback tails exist. A window starting 0.5 mm past the tangent tilts the planes, so start each side on its own flat material (`metrology._flat_start`).
  - Large bases bow by 0.7–1.8°, so measure **local** planes near each bend, not global ones.
  - A datum with a free in-plane direction makes positions along it undetermined. Withhold them; do not guess.
  - Symmetric near-twins, wrong-way bends (a mirrored hinge search is needed) and 3.6 mm joggle risers are real.
- **Jev on E3 output:** every filter or rescue arm only lowered F1. Do not add a learned filter on top of a precise detector without a family-grouped check.

## 3. Known gold-label errors and conventions

- **Gold11 human labels:**
  - Only 13 of 71 were OK. 20 had a convention offset, 23 were real errors, 15 were not bends, and 31 bends were missing.
  - **45 of 71 "human" lines were machine drafts** taken from F1 detector regions, plane lattices or shifted STEP axes. So the historical 0.458 was partly graded against detector output.
  - Replaced by `ai_bend_lines_gold11_relabel_v1` (owner-adopted). `human_bend_lines_v1` stays historical and untouched.
- **Legacy Codex catalog labels:** 34 of 40 needed fixes (85 %). These are real wrong lines, not a frame offset. Fixed in GOLD_SET_V2.
- **AI vision labels (bench_labeling):** 75 % unchanged at audit; errors were position or extent only, on about 3.5 % of lines.
- **Specific open items**:
  - 11979003 PLY/STL twins disagree on the window-merge convention (7 vs 9 lines).
  - 50757000 `sc_752e2bd54aad` B2 covers 72 % of its length (<80 % rule), so it counts as partial-length.
  - 23553000 SCAN_2 (`sc_186463746daa`) B3 is on the inner skin, so the scan was excluded (needs a re-mark).
  - **49207000 gold B10 is a welded-plate edge, not a bend** (FRESH label error, not edited).
  - 38898000 lines are seams on a rolled part, out of press-brake scope.
  - 10839000 scan_2 has 4 state rows contradicted by hole evidence.
  - 38172000 SCAN_1 has a tab-merge convention question (2 FPs).
  - 42288005 FULL was missing two 15° joggles (fixed 6 → 8).
- **Bend-angle floor.** The owner rule changed from 15° to **10°** on 2026-10-18. The sweep for 10–15° bends covered GOLD_SET_V1 only, and GOLD_SET_V2-new already used 10°. The **FRESH labels (2026-10-17) were made under the 15° floor**, so they may miss 10–15° folds.
- **Incomplete or no-truth labels:**
  - FRESH `sc_4d5dc2b40b67` is partial (recall-only).
  - `sc_0178c0155934` needs rework.
  - `sc_766be04dc456` is STL-only, so it has no studio label.
- **Process lesson:** verifiers that saw the labeler's notes were not fully blind (`BENCHMARK_V3.md`). Strip notes and engine output from anything a marker or verifier sees.

## 4. Why previous no-CAD attempts failed (synthesis)

1. **Wrong representation.**
   - There is no thickness or two-skin model.
   - Planes are extracted greedily and sequentially, so small flanges and short bends are never born. This is the MISSED class.
   - Spans come from patch overlap, not from support along the fold. This is the SPAN class.
2. **Selection by policy, not evidence.** Count rescues and cutoffs fitted on Gold11/legacy37 did not transfer (0.63 → 0.20). The pool already contains about 85 % of the gold, but the lane keeps 36 %.
3. **The evaluation was circular and small.**
   - Gold11 was partly machine drafts.
   - Development and evaluation cohorts overlapped (V52).
   - Families were few, and headline numbers were in-sample.
4. **Complexity over output.** About 10k lines of decomposition plus several-stage gates produced runs with no lines at all. The CAD path became good only when the problem was reduced to per-flange plane fits with explicit skins.

## 5. Open ideas (ranked in PLAN.md §3)

- D1 two-skin/thickness pre-pass (`RESEARCH.md` D1, running in `d1/`).
- D2 axis-family slicing: born from the axis, with span from slice support.
- **Scan-derived recipe → E3-style metrology**: build the flange graph from the scan, then reuse E3's skin-aware local fits.
- D3/D7 global multi-model plane fitting.
- D4 bend-strip model plus D10 global consistency and unfoldability.
- Pool + family-grouped selector, which uses the 0.86 pool ceiling directly.
- D5 CAD-as-prior graph matching for pruning without registration.
- D8 synthetic-trained light classifier (only after the geometric lanes).
