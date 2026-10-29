# zs_lane_v4 development protocol (v4 dev protocol, version 1)

Written 2026-10-29. This file is hashed into the freeze ledger (NOTE) **before** any v4 variant is scored. A later
change is made only by appending a new ledger NOTE that names the change and its reason; this file is never
edited silently after hashing.

## 1. Authority and status of every number

- Freeze-ledger NOTE seq 48 (owner, 2026-10-29): every gold-labelled real scan (DEV 125, TEST-B 61, N1 4, FE1 27,
  FRESH-in-DEV inside DEV) is the v4 **DEVELOPMENT** pool. None of it is held-out for v4. Seq 47 authorizes reading
  the sealed gold post-hoc.
- Every v4 number produced under this protocol is an **internal cross-validated estimate on consumed data**. It is
  never presented as held-out accuracy. The only held-out claim v4 can earn is the pre-registered fresh-scan
  promotion test (`PREREG_FRESH.md`, written and hashed before any new scan is seen).
- Production stays on zs_lane_v1. Promotion is the owner's decision after the fresh-scan test.

## 2. Pool, units and scoring (frozen)

- **Pool:** `docs/production/runlog/zs_v4/POOL.json`, sha256 `b64c7872bd008423beacb940a6f681f8bb96a4363817dfb8bdf1e17bbece62ea`
  (built by `tools/build_pool.py` from the coverage-run inventory/manifest and the FE1 gold-root manifest).
  217 scans: 207 line scans and 10 zero-bend controls, 1148 gold lines, 152 families.
- **Family** (the CV group and the bootstrap unit): `family5 = part_id[:5]`; a scan with an unknown part id is
  its own family. Part identity is used only for grouping, never as an input or feature.
- **Matcher:** the frozen research/broad_baseline `score_case` through `research/zeroshot/zs_lane_v1/scoring.score_scan`
  (5 deg / 4 mm / 0.6 / 0.6 of both spans, one-to-one Hungarian, apex locus, raw frame), unchanged. File hashes:
  score.py `2bacc1b7…`, score_broad.py `210d9885…`, bb_common.py `aca4f685…`,
  evaluate_f1_against_manual_gold.py `1e942055…`, zs_lane_v1/scoring.py `778d7e83…`.
- A missing or failed output is scored as an empty prediction (every gold line a FN); no scan is dropped.
- **Micro** = summed TP/FP/FN over the 207 line scans. **Family-macro** = mean over families of the family-micro
  F1. **CIs:** percentile family bootstrap, n 10000, seed 20261018 (`scoring.aggregate` with explicit n and seed).
- **Paired difference** (arm b − arm a): `zs_lane_v2.train.paired_bootstrap` over the families scored in both,
  same n and seed (zs_lane_v2/train.py `1efb055d…`).
- **Controls:** the 10 zero-bend scans (gold `no_bends_present`: rolled shells, rolled parts and flat parts; the
  2 FE1 ones are adjudicated rolled). Every predicted line on a control is a false line, reported separately
  (never folded into micro).

## 3. Leave-family-out cross-validation (the main metric)

- **Outer folds:** K = 10 grouped folds. The 152 families are ordered by `sha256("zs_v4|" + family5)` (hex, ascending);
  family of rank r goes to fold r mod 10. Every scan of a family is in the same fold. Each pool scan is predicted
  exactly once, by a procedure fitted without its family.
- **Inner folds (nesting):** inside each outer-training set, the training families are ordered by
  `sha256("zs_v4_inner|" + family5)` and assigned to 5 inner folds by rank mod 5.
- **Everything learned goes inside the folds:** model fitting, every threshold, every numeric rule parameter and
  every on/off switch among a grid declared in code before the run. Inner selection objective: pooled micro-F1 of the
  inner out-of-fold predictions over the outer-training line scans; ties go to the first (simplest) grid entry.
  The outer-fold procedure is then refit on the whole outer-training set with the inner choice and applied to the
  outer test fold.
- **Training label** of a candidate: it matches some gold line under the frozen gates (5 deg / 4 mm / 0.6 / 0.6) at
  its output placement (not one-to-one). Control scans contribute negatives.
- **The final frozen v4** = the same procedure fitted on the whole pool (inner selection run on all 152 families).

## 4. Arms

| Arm | What | Fitting |
|---|---|---|
| **A0 v1 pinned** | stored zs_lane_v1 outputs (pin `0087fb23…`; coverage run seq 41 + FE1 RUN seq 44) | none (its GBM was fit on DEV 125, so it is **in-sample on DEV** and optimistic there) |
| **A1 v1 retrained** | the pinned v1 blind candidate pipeline (cache `v1`), the same GBM recipe (z3_slicing.selector.GBM, 26 pooled features), fixed threshold 0.5, no post-processing | GBM refit per outer fold on the training folds |
| A2 v3 pinned (reference) | stored zs_lane_v3 rev3 outputs (pin `711bd079…`) | none (in-sample on DEV) |
| A3 v3 retrained (reference) | v3 rev3 candidates/evidence (cache `v4`/`v3x`), both GBMs refit per outer fold, frozen v3 post-processing and fix values | GBMs refit per outer fold |
| **V4** | the v4 procedure (candidates from the v4 cache, selector, post-processing, new generators) | everything per §3 |

A1 separates "more data" from "a better algorithm": a v4 gain over A0 that A1 also obtains is a data gain.
The cache is validated first: the pinned selections replayed on it must reproduce the stored A0 and A2 outputs.

## 5. Selection-aware correction (design decisions made on the pool)

The structure of the v4 procedure (which generators, features, rules and grids exist) is designed by a researcher who
looks at pool errors. That is outside the folds. To keep it visible:

- every v4 variant scored under LOFO is appended to `results/DESIGN_LOG.jsonl` (variant id, description, LOFO
  micro/macro, paired difference vs A1) **before** the next design decision; nothing is deleted;
- the final report gives M = the number of distinct v4 variants scored, and the paired difference v4 − A1 and v4 − A0
  with both the 95 % CI and the Bonferroni-adjusted (1 − 0.05/M) CI;
- an in-pool number is never quoted as held-out; the fresh-scan test (§8) is the confirmation.

## 6. Reported quantities

1. LOFO micro and family-macro with 95 % CIs for A0, A1, A2, A3, V4 (A0/A2 are their stored outputs).
2. Paired differences V4 − A0 and V4 − A1 (micro and macro; 95 % and selection-aware CIs).
3. Per-family wins and losses of V4 vs A0 and vs A1 (TP/FP/FN, F1, mean ΔFP+ΔFN per scan).
4. False lines on the 10 controls, per arm.
5. The FE1 (set C) scans and families, reported separately (their LOFO predictions are made without their family).
6. Hem recall: gold lines whose notes match `180 ?deg[^;|]{0,24}\bhem\b` (case-insensitive) — 11 lines
   (1 DEV, 10 FE1); recall per arm.
7. Lanced-tab and short-ear results: family 31367 (`gb_68cd06`, `gb_959f4a`) and the FE1 short-ear scans
   (`gb_138090`, `gb_1f3fe0`), per arm.
8. Error taxonomy (FN/FP by cause and by component) for A0, A2 and V4.
9. Runtime: wall P50/P95/max and peak RSS of the frozen v4 lane on the pool, vs the stored v3 pinned runs.

## 7. Gates (all decided on the LOFO numbers of §6)

- **G1** LOFO micro(V4) > micro(A0) and > micro(A1), and macro(V4) > macro(A0) and > macro(A1) (strict, point
  estimates).
- **G2** the paired-difference CIs of §6.2 are reported (whether or not they exclude 0; a CI including 0 is
  reported as "not significant").
- **G3** no family worse than A0 by more than one line on average: for every family, mean over its line scans of
  (FP+FN)(V4) − (FP+FN)(A0) ≤ 1.0. A violation passes only with a written, specific justification (e.g. a gold
  convention conflict), listed in the report.
- **G4** zero false lines on the 10 controls (rolled and zero-bend).
- **G5** FE1-family results reported separately (§6.5).
- **G6** hem recall reported (§6.6).
- **G7** runtime: v4 lane wall P95 on the pool ≤ 1.5 × the stored v3-pinned wall P95 on the same pool; peak RSS
  within the slot limits (16 GiB; 32 GiB above 3 M faces).
- **G8** determinism: the frozen v4 lane gives byte-identical lines on two runs of the same scan (checked on a
  sample), and its output equals the cache-replayed output of the frozen procedure.

v4 counts as better than v1 (on the pool) only if G1, G3, G4, G7 and G8 pass; G2, G5, G6 are mandatory reports.
If G1 fails, the report says that v4 does not beat v1 under LOFO, and what was learned.

## 8. After the freeze

- The best v4 is frozen as a research candidate (`research/zeroshot/zs_lane_v4/`, named tests `tests/test_zs_v4*.py`,
  a new pin file, `FREEZE.md`). Production defaults, the v1/v3 lanes and their pins are not touched.
- `PREREG_FRESH.md` pre-registers the promotion test on the next batch of never-seen scans (decision rule: v4 beats
  v1 by a stated margin with a stated CI rule) and is hashed into the ledger before any of those scans is seen.

## 9. Custody and budgets

- Heavy geometry only through `docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh` (one job at a time,
  OMP 1, nice 15, RSS watchdog 16 GiB, 32 GiB above 3 M faces). Candidate features are cached under
  `docs/production/runlog/zs_v4/cache/`; classifier and rule iterations reuse the cache.
- Excluded: 41880000_D, Eyal50 quarantine, rhino-preview face scans, `*_SEALED*` scan paths (sealed gold roots are
  readable per seq 47/48), unverified phone captures. No dependency installs; deterministic code only.
