# Plan: line-F1 ≥ 0.85 on 40+ scans, zero-shot first (2026-10-25)

Inputs to this plan:
- `INVENTORY.json` and the summary in `README.md`;
- `SPLIT_FROZEN.json`, sha256 `d25ffd4f4d50…`, frozen before any new engine output;
- `BASELINE.md`;
- `LESSONS.md`;
- `RESEARCH.md`, the D1–D10 literature directions;
- the running D1 diagnostic in `d1/`.

The matcher is frozen and is never changed: 5° / 4 mm / 0.6 of both spans, one-to-one Hungarian, outer-apex locus, raw scan frame (`research/broad_baseline/score.py score_case`). The rules hold throughout: no part-specific logic, no tuning on the sealed portion, one writer per engine file, and heavy work through `/private/tmp/claude-501/sherman_slot.sh`.

## 0. Owner decisions needed first

**OD-1. What counts as the honest test.**
- Every usable scan on disk is consumed: CONSUMED, DEV-CHECK or FRESH. By the 2026-10-19 rule, FRESH may never be a held-out set.
- So the literal answer is that **no honest local test set exists**. An honest 0.85 claim needs **≥ 40 new scans** from new deliveries, from ≥ 30 families, with gold made by §2 before any engine sees them.
- Proposed interim, which needs owner sign-off: **TEST-B**.
  - It is 61 scans / 50 families / 329 gold lines.
  - 60 of the scans are FRESH no-STEP scans and 1 is 17139000, which has a STEP.
  - No engine or lane has ever run on any of them. Their labels were never read in any tuning loop. Their 5-digit family groups are disjoint from every dev scan.
  - TEST-B is sealed now for this programme. It is scored once at the end and reported as "programme-sealed, consumed-by-rule", never as unseen-family accuracy.
  - The A0 baseline was run on it once, frozen code, aggregate only. Per-case rows are in `baseline/scores_TESTB_SEALED.json` and `baseline/a0_runs_TESTB_SEALED/`. The `_SEALED` name keeps every custody guard away from them.
- **Recommendation:** adopt TEST-B as the programme test, **and** request ≥ 40 new scans for the final claim (§1.3).

**OD-2. Gold of record for TEST-B.**
- The FRESH labels were made on 2026-10-17 under the old 15° floor and were not 3-auditor audited.
- Recommendation: run the §2 blind re-mark and adjudication on all 61 TEST-B scans, and freeze the result before any phase-2 engine output exists.
- Cost: about 61 × 2 marker passes.

**OD-3. The 5 DEV-CHECK cases with no E3 output.**
- The cases: `pc_eaa89c6d1f1f`, `pc_152f1b1aacb2`, `pc_c623d2e45bff`, `pc_9ef2cf4e52cb`, `pc_7a1773eca9fa`.
- Keep them unrun as a tiny CAD-path one-time check (the default here), or release them to dev.

**OD-4. Eyal50 quarantine scans.**
- The scans: 49773004_B, 2927000_E2, and 44773000_C V/X.
- Their 20/30 dev/holdout split file was never read. The owner decides whether they can be used. If they can, they need §2 gold.

## 1. Data and split (frozen)

| Portion | Scans | Families (5-digit groups) | Gold lines | Gold quality | CAD |
|---|---|---|---|---|---|
| DEV | 121 (5 zero-bend controls, 16 partial) | 94 (80) | 586 | 103 audited vision (GOLD_SET_V2), 18 accepted AI (FRESH-with-STEP + group overlaps) | 59 with STEP, 55 compile |
| TEST-B (sealed) | 61, all full | 50 (49) | 329 | 61 accepted AI (FRESH), re-audit per OD-2 | 1 |

**1.1 DEV use.**
- All tuning, ablation and diagnosis happen on DEV only.
- Anything with fitted parameters (selectors, thresholds) is chosen by 5-fold grouped cross-validation over the 5-digit family groups inside DEV. Report the out-of-fold numbers.
- The fold assignment is frozen once, in `dev_folds.json`, made by the integrator before any lane fits anything.

**1.2 TEST-B use.**
- Exactly one scoring run at the end. Each lane is frozen by a code fingerprint plus a config hash recorded beforehand.
- Before that run:
  1. TEST-B gold is frozen (OD-2).
  2. A ledger NOTE is appended (`freeze/LEDGER.json` NOTE via its tool, owner OK) binding the split sha, gold shas and lane fingerprints.
- No TEST-B scan, label or output is opened before then. Paths contain `SEALED`.

**1.3 New-data test (the honest claim).**
- Ask Shaked for ≥ 40 new scans from new parts, and STEP exports where they exist: the 56 SLDPRT parts are already flagged.
- Run §2 on them before any engine output.
- Score once, with the frozen final lane.

## 2. Gold-creation protocol

This protocol applies to every scan without gold, to the OD-2 re-audit and to new deliveries. It must stay independent of any engine output.

**2.1 Pre-registration.**
- Before anything else, append the scan to `gold/GOLD_LEDGER.jsonl` with its scan sha256, path, batch id, status `REGISTERED` and timestamp.
- Every engine and lane driver in this programme must refuse a scan whose ledger status is not `FROZEN`. The integrator adds this guard to the shared harness in week 1. It never touches engine code.

**2.2 What markers may see.**

Allowed:
- the raw scan;
- `python -m research.bench_labeling prep` output: planar-patch segmentation with numbered renders, a measurement aid that proposes no bends;
- shaded raw-mesh renders, close-ups and perpendicular cross-sections (`lines` verification renders, the overrides `plane_from_points` / `extent_t` / `extent_xyz`);
- the Bend Label Studio, for viewing.

Never shown:
- A0, pool, E3, best_f1 or any lane output;
- CAD bend lines or counts;
- prior labels, including the existing FRESH label during the OD-2 re-audit;
- the other marker's files or notes.

**2.3 Two independent markings.**
- Markers A and B are separate agents with separate output dirs (`gold/<batch>/<scan>/marker_{A,B}/`). Each:
  1. picks patch pairs, or overrides for flanges without a patch;
  2. runs `lines`;
  3. checks every line on close-ups and sections;
  4. does a **missed-bend sweep** on raw renders and sections that does not use the patch list. A fold the segmentation missed must be added by `plane_from_points`.
- The gold rule is the GOLD_SET_V2 rule:
  - outer-skin apex;
  - parallel to the fold;
  - covers ≥ 80 % of the visible fold;
  - formed folds ≥ 10° count;
  - a joggle counts as 2;
  - a fold split by windows or reliefs between the same two plates is 1 line;
  - not bends: hole walls, edges, embosses, rolled kinks, weld-plate edges;
  - partial scans: formed bends only;
  - zero-bend scans are valid.

**2.4 Adjudication.**
1. A script matches A against B with the frozen matcher.
2. Matched pairs with offset ≤ 1.5 mm and angle ≤ 2° are accepted. The line is taken from A, or from the better-supported side.
3. Every disagreement goes to an adjudicator agent. It sees both markings, the sections and the rules, never engine output. Its verdicts are ACCEPT_A, ACCEPT_B, MERGE, NOT_A_BEND or MISSING_ADD, each with a written reason.
4. In the OD-2 re-audit, the existing FRESH label is brought in only at adjudication, as a third marking.

**2.5 Write and freeze.**
- `write-label` writes to a new root `data/bend_corpus/annotations/ai_bend_lines_zs_v1/`, never `human_bend_lines_v1`. TEST-B labels go to `ai_bend_lines_zs_v1_SEALED/`.
- The provenance sidecar holds both markings, the adjudication record and the render hashes.
- Then the ledger is updated: status `FROZEN` with the label sha256. Only then may any engine run on that scan.

**2.6 Quality control.**
- Each batch of 6 scans carries 1 blind **canary**: a DEV scan with GOLD_SET_V2 gold, unknown to the markers.
- Marker-vs-canary line-F1 must be ≥ 0.95, or the batch is redone.
- The batch's A/B agreement F1 is reported.

**2.7 Parallel layout.**
- Batch = 6 scans + 1 canary.
- Per batch: 2 marker agents + 1 adjudicator. Batches are independent.
- Heavy `prep`/`lines` calls go through the slot semaphore, which is 3 slots.

**2.8 Scans that need gold.**

| Need | Scans |
|---|---|
| New gold, usable local scans | **4**: `sc_186463746daa` (23553000 SCAN_2, re-mark), `sc_4d5dc2b40b67` (45141000, complete the partial), `sc_0178c0155934` (48381001, rework), `sc_766be04dc456` (48389000; STL, so it needs STL support in `write-label` or a same-frame PLY export) |
| OD-2 re-audit | 61 TEST-B scans |
| If OD-4 is approved | 4 Eyal50 scans |
| Honest new-data test | ≥ 40 new scans |

Gold is not the bottleneck for DEV: 121 scans are already gold.

## 3. Zero-shot directions, ranked

Each direction is an independent work package (WP) prototyped in an **isolated copy** of the code. The steps are the same for every WP:
1. `git worktree` or a copy under `research/zeroshot/<wp>/`, with no edits to `backend/`, `domains/` or `scripts/`.
2. A common CLI: `scan.ply → eval_lines.json` (`sherman.broad_baseline.eval_lines.v1`, raw frame).
3. A common scorer: `tools/score_baseline.py` adapted to DEV only, with out-of-fold numbers.
4. The common error budget: FN MISSED / SPAN / OFFSET / ANGLE, FP SPURIOUS / span / offset.

All WPs report against the same DEV baseline (`BASELINE.md`): A0 is **0.322 micro / 0.267 family-macro** on the 116 complete-label DEV line scans, at P 0.60 / R 0.22. **81 % of its FNs are MISSED.** A0 emits nothing on 36 of the 116 scans, which loses 21 % of the gold lines. OFFSET is only 26 FN.

| Rank | WP | Idea | Why (LESSONS) | Attacks | First milestone (DEV, out-of-fold) |
|---|---|---|---|---|---|
| 1 (enabler) | **Z1 Two-skin / thickness pre-pass** (`RESEARCH.md` D1) | Estimate t and pair the skins. Segment on the mid-surface. Place each line on the outer skin (the outer of two skin peaks t apart, as E3 does). | No lane models t. At 90° the inner/outer apex gap is 1.41·t. Prior lanes produce doubled and offset lines. On DEV, A0 OFFSET is small (26 FN). Z1 earns rank 1 as the **base for Z2/Z3 segmentation and line placement**, not as an A0 patch. | OFFSET, duplicate FPs; enables Z2 | Already running in `d1/`. Then: as an A0 post-pass (snap/dedupe A0 lines to outer skin), ΔF1 on DEV |
| 2 | **Z2 Scan-derived flange graph → E3-style local metrology ("self-recipe")** | Local planar flanges, then an adjacency graph with bend-strip evidence (D4), then per edge an E3-style skin-aware local plane fit (flat-start windows, station profiles) to place the line and span. No CAD and no registration: the graph comes from the scan. | E3 scores 0.99 once it knows the flanges. Its pitfalls are already solved (skin swap, radius > CAD, bowed bases). Zero-shot then reduces to flange discovery. | MISSED, SPAN, OFFSET | Flange recall ≥ 0.95 on DEV GOLD_SET_V2 full scans; line-F1 ≥ 0.70 |
| 3 | **Z3 Axis-family slicing** (D2) | Find axis families on the Gauss map. Slice perpendicular to each axis and fit a 2D profile polyline. Track vertices across slices to get line and span. | It is a birth channel for short bends and small flanges. Span comes from slice support. 35/51 daily28 misses were never born. | MISSED, SPAN | Candidate recall ≥ 0.90 at ≤ 3× gold count |
| 4 | **Z4 Pool + grouped selector** | Keep A0's liberal pool (recall 0.86) and add Z1–Z3 candidates. Rank with plain features (LR or GBM) chosen by grouped CV, then NMS. | Selection was the broad-set bottleneck: pool 0.86 vs kept 0.36. LR AUC 0.93. Cheap, and gives a quick lower bound. | MISSED (selection), SPURIOUS | ≥ 0.50 out-of-fold (v2 had 0.478 on 37 scans) |
| 5 | **Z5 Global consistency / unfoldability selection** (D10) | CP-SAT selection over the union of candidates: every line separates two flanges with strip evidence; the two-skin duplicate rule; the flange graph is (near-)tree and unfoldable. | It removes FPs without learned filters, which only hurt E3. | SPURIOUS FP | Precision ≥ 0.90 at no recall loss |
| 6 | **Z6 Global multi-model plane fit** (D3 / D7) | Replace the sequential greedy RANSAC. Open3D `detect_planar_patches` is the quick win. | Greedy extraction loses small flanges. | MISSED | Plane-recall comparison inside Z2 |
| 7 | Later | D6 crease scoring (recall booster), D8 synthetic-trained light classifier, D9 learned primitives (research only) | | | Only once Z1–Z5 plateau |

**Integration.**
- One integrator agent owns the combined lane `zs_lane_v1`: Z1 + (Z2 ∪ Z3 ∪ A0 pool), then Z4, then Z5.
- It owns the DEV scorer and `dev_folds.json`. It is the only one that ever triggers the TEST-B run.
- Milestones on DEV, out-of-fold:
  - M1 ≥ 0.55 (≈ +0.2);
  - M2 ≥ 0.70;
  - M3 ≥ 0.85 micro **and** family-macro ≥ 0.80, worst family reported.
- Stop rule: two consecutive integrator rounds with < +0.01 means plateau. Report and stop.

## 4. CAD-assisted path: CAD cuts the search

Baseline: E3 e3-1.12.0 scores **0.951 micro / 0.820 family-macro** (P 0.993, R 0.912) on the 49 E3-eligible complete-label DEV scans. A0 scores 0.356 on the same scans. E3-else-A0 is no better than E3: A0 emits nothing where E3 fails. All 25 E3 FNs are coverage: withheld positions on 49207000, a refused mirror, a rolled seam part, and DEV-CHECK/FRESH misses (`BASELINE.md`). CAD feeds the zero-shot search at three levels, and none of them needs E1 to compile:

1. **Rotation-invariant priors, no registration** (Z7 = RESEARCH D5). From the STEP (`cad_bend_extractor.py` / E1 dev candidates) take:
   - the bend count;
   - the multiset of included angles;
   - t and the inner radius;
   - bend lengths;
   - flange areas.

   These prune Z1–Z5 directly:
   - t is fixed for Z1, which matters on one-sided scans;
   - candidates must have an angle within 5° of a CAD angle;
   - a count cap: at most n_CAD, or fewer for partial scans;
   - flange-size gates;
   - Z3 slicing only along CAD axis families, once matched;
   - spectral or CP-SAT matching of the scan flange graph (Z2) to the CAD flange graph, which allows unmatched nodes for partial scans.
2. **E3 as primary when it is VALID.** On refusal (NO_RECIPE, COMPILE_BLOCKED, WRONG_PART, MIRRORED) or on E3-UNKNOWN bends, fall back to the CAD-pruned zero-shot lane.
   - The measure is `e3_else_zs` against `e3_else_a0` (the baseline: 0.951 on E3-eligible DEV).
3. **Search time.**
   - Known axis families and angles shrink the Z3 slicing and Z2 hinge windows.
   - Report the P95 runtime per scan, with and without CAD. E3 runs about 10–90 s; A0 needs 160–600 s.
   - Target: the CAD-pruned lane ≤ 60 s P95.

**CAD WP (Z7)** is independent of Z1–Z6. It is prototyped on the same copy and harness, and on the DEV with-STEP cohort only (59 scans).

## 5. Parallel work breakdown (for the coordinator)

| Agent | Package | Writes | Depends on |
|---|---|---|---|
| G-batch × k | §2 gold batches: TEST-B re-audit first, then the 4 local re-marks, then new deliveries | `gold/<batch>/`, `ai_bend_lines_zs_v1[_SEALED]/`, `gold/GOLD_LEDGER.jsonl` (append-only; the adjudicator writes) | OD-1/OD-2 |
| Z1 (running) | two-skin pre-pass | `d1/`, `research/zeroshot/d1_two_skin/` | none |
| Z2 | self-recipe flange graph (top recall lever, together with Z3) | `research/zeroshot/z2_flange_graph/` | Z1 API (t, skin labels) is optional at first |
| Z3 | axis slicing | `research/zeroshot/z3_slicing/` | none |
| Z4 | pool + grouped selector | `research/zeroshot/z4_selector/` | A0 pools: kept under `/private/tmp/claude-501/sherman_zeroshot_work/{dev}` for the scans re-run here. The GOLD_SET_V2 pools were lost when tmp was cleaned, so a re-run is needed (≈ 4 h at 2 slots) |
| Z5 | consistency selector | `research/zeroshot/z5_consistency/` | candidates from Z1–Z4 |
| Z7 | CAD-prior pruning + `e3_else_zs` | `research/zeroshot/z7_cad_prior/` | Z2/Z4 outputs |
| Integrator (single) | `zs_lane_v1`, `dev_folds.json`, DEV scoreboard, final TEST-B run | `zeroshot_2026-10-25/integration/` | all of the above |

Each lane must deliver:
- eval-lines files for all DEV scans;
- a run record (code fingerprint, wall time, RSS);
- an error-budget histogram;
- overlays for its 5 worst scans, opened and checked by eye.

Visual checks are mandatory before any claim. No lane edits `backend/`, `domains/` or the frozen matcher.
