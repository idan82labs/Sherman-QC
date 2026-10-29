# Gold set v2

Created 2026-10-18 at the owner's request of the same day. The owner asked for more gold-labelled scans so that the NO-CAD (blind) track can be benchmarked broadly. Production will be CAD-assisted, so the remaining CAD scans were added too.

GOLD_SET_V2 is GOLD_SET_V1 plus 45 new vision-gold scans. GOLD_SET_V1 is unchanged and is referenced by sha256. Every new label is a vision label (`claude-vision-ai-v1`, `ai_adjudicated`): each line was checked by eye on renders, close-ups and perpendicular cross-sections, then audited by 3 blind vision auditors.

Files:

- Manifest: `docs/production/runlog/gold_set_v2/GOLD_SET_V2.json`
- Case lists (frozen; same matcher as v1: 5 deg / 4 mm / 60 % of both spans, one-to-one Hungarian, apex locus, raw scan frame):

| List | Purpose | Cases | Lines | Families | case_list_sha256 |
|---|---|---|---|---|---|
| `runlog/gold_set_v2/case_list_gold_v2_all.json` | every gold scan; also the broadest blind benchmark, because the blind engine never reads CAD | 103 | 494 | 77 | `0514996d5517…` |
| `runlog/gold_set_v2/case_list_gold_v2_cad.json` | **production track**: scans with a STEP the CAD lane can read | 43 | 251 | 24 | `707810f3197b…` |
| `runlog/gold_set_v2/case_list_gold_v2_nocad.json` | **blind track**: scans with no usable CAD (no STEP; native x_t only or none) | 60 | 243 | 53 | `656b452b801a…` |

- Rescore: `runlog/gold_set_v2/rescore/RESCORE_SUMMARY.json`
- Build script: `runlog/gold_set_v2/scripts/build_gold_set_v2.py`. It only reads, apart from what it writes under gold_set_v2. It runs no engine, prep, lines or label step.
- Inputs: `runlog/gold_set_v2/outcomes/*.json` (one per scan, all read), `candidates.json`, `write_label_jobs.json`, `write_label_logs/`.
- GOLD_SET_V1: `runlog/gold_set_v1/GOLD_SET_V1.json` sha256 recorded in the manifest. All 58 v1 label files were re-hashed and match.
- New labels are in `data/bend_corpus/annotations/ai_bend_lines_gold_v1/` (45 labels and 45 `provenance/` sidecars).
- These were not touched, checked by hashing before and after: `human_bend_lines_v1`, `ai_bend_lines_v1`, `ai_bend_lines_gold11_relabel_v1` and the Gold11 human catalog.

## Gold rule

A line LANDS only if all of these hold:

- It lies on the OUTER-skin apex of a real formed fold.
- It runs parallel to the fold.
- It covers at least 80 % of the visible fold.

How the rule is applied:

- A joggle counts as 2 bends.
- Formed bends of **10 deg or more** count (owner rule 2026-10-18; the floor was 15 deg). A 10-15 deg bend must be a distinct formed fold between two flat plates.
- A fold interrupted by windows or reliefs that fold the same two plates is ONE line.
- These are not bends: hole walls, thickness and perimeter edges, embosses, gentle kinks in a rolled surface, and artifacts.
- In staged scans, only the bends formed in that scan are labelled.
- Zero-bend scans are valid gold with no lines.

Note: `gold_set_v2/candidates.json` still quotes the old 15 deg floor. The auditors and the fixer worked to the 10 deg rule. Both GOLD_FIXED scans add 10-11 deg folds.

## What was written

- 45 outcome files were read: 43 GOLD_AS_IS and 2 GOLD_FIXED. None was excluded at the outcome stage.
- 45 `write-label` calls ran one at a time under the geometry lock, all with rc 0. There were **no refusals**.
  - Annotator: `claude-vision-ai-v1`.
  - Coordinates: every asset matched its lines by sha256.
  - One zero-bend scan was written with `--confirm-no-bends`: sc_49e5e7a1c973 (48565022).
- No asset was in `~/Downloads`, so no studio copy was needed.
- 6 lines are kept at `uncertain` confidence, though the auditors marked them OK:
  - sc_589688be58d2: B1, B2
  - sc_b3cde5d10544: B6
  - sc_e218b16ed070: B3, B4, B6
- 3 new lines are 10-15 deg folds:
  - sc_24c57f2660b3: B5 (10.3 deg) and B6 (11.3 deg)
  - sc_c1d112196ae5: B4 (10.7 deg)

## CAD / NO-CAD split

| | Scans | Lines | Families | Full / staged | Zero-bend |
|---|---|---|---|---|---|
| **GOLD_SET_V2 total** | **103** | **494** | **77** | 87 / 16 | 4 |
| CAD (production) | 43 | 251 | 24 | 27 / 16 (168 / 83 lines) | 3 |
| NO-CAD (blind) | 60 | 243 | 53 | 60 / 0 (243 / 0 lines) | 1 |
| *GOLD_SET_V1 part* | 58 | 328 | 38 | 42 / 16 | 3 |
| CAD | 41 | 241 | 23 | 25 / 16 | 3 |
| NO-CAD | 17 | 87 | 15 | 17 / 0 | 0 |
| *New in v2* | 45 | 166 | 40 | 45 / 0 | 1 |
| CAD | 2 | 10 | 2 | 2 / 0 | 0 |
| NO-CAD | 43 | 156 | 38 | 43 / 0 | 1 |

- The new scans add 39 part families that are not in v1.
- Only 2 of the 45 new scans have a STEP:
  - sc_2852c2845b94 (45796000_A 5.ply, 8 lines). Its STEP is the same `45796000_A.stp` that v1 uses.
  - sc_203b95d7a753 (49201000_C.ply, 2 lines). Its STEP is `incoming_email_2026-10-17/20260830_1a05196d/BO44666A-C-PD-bo44666a_c.stp`, sha `2720c509…`.
  - The other 43 have native x_t or no CAD, so they serve the blind track only.
- Two v1 scans have `has_step` true in GOLD_SET_V1 but no `cad_path` in its case list, so they are in the NO-CAD list:
  - sc_c27168a2de83 (28289002)
  - sc_fee685eee4db (49125000, x_t only)
- All 16 staged scans are CAD scans from v1. Every new scan is a full scan, including the shaked_xv DEFECTIVE_X / GOOD_V pairs.

## Excluded

None of the 45 audited scans was excluded. These 13 were skipped at the candidate stage (`candidates.json` "skipped"):

| Scan | Reason |
|---|---|
| sc_d45186653eed, sc_8dfdab44605d, sc_927a2fa13067, sc_9b9fd762f361, sc_a0305a6cf787 | STL originals that are same-mesh twins of GOLD_SET_V1 PLYs (0.0 mm) |
| sc_8379cebef9e3 | OBJ, same mesh as Gold11 sc_7a1773eca9fa |
| sc_4d277fbe7854 | CAD_RECIPE proxy PLY, an exact re-export of v1 gold sc_1a5377ac7c15 |
| sc_90257b128d80 | 17139000_01: eval_tier FRESH_SCAN_OF_CONSUMED_PART, kept out by custody |
| sc_186463746daa | 23553000_B_SCAN_2: failed the v1 2-verifier pass (B3 on the inner skin); already decided |
| sc_2e584e9d1ffe, sc_7a253806b10b, sc_f098dbceaa6d | possible CAD reference meshes (forbidden) |
| sc_9c890e76934f | `part 2.stl`: STL only and no PLY twin, so write-label (PLY-only) cannot take it |

Custody: only CONSUMED scans were used (BENCHMARK_V3 split CONSUMED, V2 eval_tier CONSUMED, sha re-hashed). The 18 aug-30 daily scans are TRAIN18 (A30-xxx), not HOLDOUT10. None of these was used: FRESH, `*_SEALED*`, HOLDOUT10, N001-N016, EHO4, LIDAR_DEMO, CAD reference meshes.

## How good were the prior labels?

All 45 new scans had a prior label. None was human. The frozen matcher scores each prior against the new gold, and a prior "needed a fix" if it would not score F1 = 1.0.

| Prior tier | Audited | Prior → gold lines | Scored as gold unchanged | Needed fix |
|---|---|---|---|---|
| AI_GOLD_ACCEPTED (ai_bend_lines_v1) | 5 | 22 → 22 | **5** (endpoints identical) | **0** |
| GOLD_LINES_AI_LEGACY (Codex manual multiview, relabelled from scratch) | 40 | 128 → 144 | 6 | **34 (85 %)** |
| Human | 0 | - | - | - |

What was wrong in the 34 legacy priors:

- **94 of 144 gold lines** had no matching legacy line. The diagnostic nearest-line causes:
  - 37 had no legacy line within 5 deg on that fold: the fold was missing or the line was drawn on another fold.
  - 55 had a parallel legacy line more than 4 mm away: another skin, a neighbouring parallel fold, or possibly a legacy-frame offset. These were not adjudicated one by one. In 14 scans every missed line is of this kind. For example, sc_e8c946b837aa has both lines parallel and about 16 mm off, which is worth a look if the legacy catalog is ever reused.
  - 2 had a span overlap under 60 %.
- 3 of the missed lines are 10-15 deg folds that the 10 deg rule now counts.
- **78 of 128 legacy lines** match no gold line.
- The line count changed in 10 scans. Examples:
  - 1 → 0: sc_49e5e7a1c973 is now a zero-bend scan.
  - 2 → 4: sc_cca7d7680a51
  - 7 → 10: sc_e218b16ed070
  - 8 → 10: sc_d741e1d515bf
- In 6 scans the legacy line count and positions were right within tolerance (F1 = 1); only the endpoints were redrawn.

Per scan: `GOLD_SET_V2.json` → `prior_label_quality.GOLD_SET_V2_NEW.*.per_scan`. Each new_gold row also has `prior.missing_line_causes`.

Audit route of the new vision labels:

- 43 were accepted as-is: 3/3 blind auditors OK, nothing missing.
- 2 were fixed:
  - sc_24c57f2660b3 (49191054), 4 → 6 lines. It adds 2 inward-toed end tabs of 10.3 and 11.3 deg. B2/B3 were trimmed to the start of the bend-relief slit.
  - sc_c1d112196ae5 (48340007), 3 → 4 lines. It adds a 10.7 deg bend.

For reference, GOLD_SET_V1 audited 44 AI_GOLD_ACCEPTED labels:

- 33 were unchanged.
- 7 were fixed (APEX_OFFSET 6, EXTENT 5, count 6→9 once).
- 3 were zero-bend scans re-verified with no change.
- 1 was excluded.
- 4 human labels were replaced by vision gold.

## Rescore of the existing predictions

The rescore used `broad_baseline/runs/best_f1` and `runs/runtime` with `research/broad_baseline/score.py broad`. No engine was run.

The pre-audit column scores the same cases and the same predictions against the truth that existed before the audit. For v1 cases that is the GOLD_SET_V1 pre-audit spec. For the new cases it is the prior AI label. The scan counts below are complete-label line scans; the zero-bend controls are reported separately.

Coverage (scans lacking predictions are not scored):

| List | Gold scans | With predictions | Lacking (v1 / new) |
|---|---|---|---|
| all | 103 | 57 (54 line + 3 zero-bend) | 46 (5 / 41) |
| cad | 43 | 37 (34 line + 3 zero-bend) | 6 (4 / 2) |
| nocad | 60 | 20 (20 line) | 40 (1 / 39) |

- Only 4 of the 45 new scans have predictions: the 4 UNKNOWN-part AI_GOLD_ACCEPTED uploads, all NO-CAD.
- None of the 41 new legacy-relabelled scans has been run, and neither have the 2 new CAD scans.
- The broad blind benchmark therefore needs blind-lane engine runs on these 41 scans (not done here, by rule).

| List | Lane | Scans / fam | Gold lines | Micro P / R / **F1** | **Family-macro F1** | Pre-audit F1 / fam-macro | Δ micro / Δ fam |
|---|---|---|---|---|---|---|---|
| all | best_f1 | 54 / 40 | 310 | 0.595 / 0.394 / **0.4738** | **0.4350** | 0.4516 / 0.4283 | +0.0222 / +0.0067 |
| all | runtime | 54 / 40 | 310 | 0.422 / 0.087 / **0.1444** | **0.1399** | 0.1183 / 0.1259 | +0.0261 / +0.0140 |
| cad | best_f1 | 34 / 22 | 212 | 0.574 / 0.458 / **0.5092** | **0.4971** | 0.4822 / 0.4871 | +0.0270 / +0.0100 |
| cad | runtime | 34 / 22 | 212 | 0.422 / 0.127 / **0.1957** | **0.2544** | 0.1615 / 0.2289 | +0.0341 / +0.0255 |
| nocad | best_f1 | 20 / 18 | 98 | 0.694 / 0.255 / **0.3731** | **0.3591** | 0.3664 / 0.3564 | +0.0067 / +0.0027 |
| nocad | runtime | 20 / 18 | 98 | 0 predicted lines / **0.0** | **0.0** | 0.0 / 0.0 | 0 / 0 |

- The v1 part of `all` reproduces the GOLD_SET_V1 rescore exactly: best_f1 50 scans, 296 lines, F1 0.4848, family-macro 0.4694; runtime 0.1500 / 0.1555.
- The 4 new scored scans (14 lines) give best_f1 F1 0.20 (2 TP), and the same before the audit because those AI labels were unchanged. On these scans runtime predicts 0 lines.
- The runtime lane predicts no lines on any NO-CAD scan, so as run it is effectively a CAD-only lane.
- Zero-bend controls: best_f1 puts 5 FP lines on 2 of the 3 v1 controls (sc_3bde9996f14d, sc_b4719db3dd69); runtime puts 3 FP on sc_3bde9996f14d. The new control, sc_49e5e7a1c973, has no predictions.
- Cohorts in RESCORE_SUMMARY:
  - Full scans: all/best_f1 F1 0.489, family-macro 0.458
  - Staged scans: 0.415 / 0.425
  - Also split by GOLD_SET_V1 part vs new part

## Blind-track baseline (2026-10-18)

The best_f1 lane (the old research lane, the bar to beat) has now been run on all 46 GOLD_SET_V2 scans that had no predictions. It ran one scan at a time under the shared geometry lock, with OMP_NUM_THREADS=1 and a 600 s budget per scan. The runtime lane is dropped: the owner discarded it as a comparator. Every GOLD_SET_V2 scan now has a best_f1 run record.

- Run list: `runlog/gold_set_v2/rescore_full/case_list_gold_v2_missing46.json`. Driver: `research/broad_baseline/gold_v2_fill_driver.sh` (calls `run_broad.py --only <id>` once per lock). Logs: `rescore_full/run_logs/`. Predictions: `broad_baseline/runs/best_f1/<case_id>/`.
- Scoring: `research/broad_baseline/gold_v2_blind_score.py`, which calls `score.py broad` with the frozen matcher (5 deg / 4 mm / 60 %, apex locus, raw frame). CIs are a 95 % percentile bootstrap that resamples families (10 000 reps). Results: `rescore_full/BLIND_BASELINE_SUMMARY.json` and `rescore_full/{all,cad,nocad}/scores_best_f1*.json`.
- Check: the 54 previously scored scans reproduce the earlier rescore exactly (122/83/188, F1 0.4738).
- **Failures (3, counted as all-FN):**
  - Two new scans hit the 600 s timeout in `a0_typed_union`: sc_28a25c37a928 (49191023, 8 lines) and sc_d741e1d515bf (49191038, 10 lines).
  - One earlier failure is unchanged: sc_85a7f8cca6bc.
  - Two CAD scans ran but emitted no lines (NO_PREDICTIONS, from the broad run).
- Fill runtime (46 scans): median 159 s, P95 494 s, max 601 s (the two timeouts), 2.4 h of compute in total. Peak RSS: median 0.96 GiB, max 1.60 GiB.

Lanes in the table:

- `best_f1` is the A0 scan-only lines united with the gated CAD half where a STEP exists.
- `A0` is the scan-only half alone. It is the strictly blind number. On NO-CAD scans the two lanes are identical.

| List | Lane | Scans / fam | Gold lines | Micro P / R / **F1** (95 % CI) | **Family-macro F1** (95 % CI) | Full F1 | Staged F1 |
|---|---|---|---|---|---|---|---|
| all | best_f1 | 99 / 75 | 494 | 0.551 / 0.330 / **0.413** (0.344–0.468) | **0.338** (0.261–0.419) | 0.415 (85) | 0.403 (14) |
| all | A0 (blind) | 99 / 75 | 494 | 0.628 / 0.233 / **0.340** (0.274–0.398) | **0.273** (0.206–0.345) | 0.349 | 0.291 |
| cad | best_f1 | 40 / 23 | 251 | 0.567 / 0.458 / **0.507** (0.444–0.568) | **0.528** (0.399–0.654) | 0.552 (26) | 0.403 (14) |
| cad | A0 (blind) | 40 / 23 | 251 | 0.744 / 0.267 / **0.393** (0.311–0.469) | **0.317** (0.214–0.420) | 0.442 | 0.291 |
| nocad | best_f1 = A0 | 59 / 52 | 243 | 0.516 / 0.198 / **0.286** (0.196–0.377) | **0.254** (0.168–0.345) | 0.286 (59) | - |

Notes on the table:

- Scan counts are complete-label line scans. The 4 zero-bend controls are reported separately: best_f1 puts 5 FP lines on 2 v1 controls and 0 on the new control sc_49e5e7a1c973.
- On the NO-CAD list the new scans are much harder than the v1 part:
  - 42 new scans, 156 lines: F1 0.220, family-macro 0.183, P 0.387.
  - 17 v1 scans, 87 lines: F1 0.407, P 0.774.
  - The blind lane's recall is the main gap: 0.20 on NO-CAD.

**Legacy catalog offset check.** For the 14 scans whose missed legacy lines were all "parallel but >4 mm off", the check (`research/broad_baseline/catalog_offset_probe.py`, output `rescore_full/catalog_offset_probe.json`) found no constant translation. These are genuine wrong-line errors, not a frame bug. The evidence:

- On 5 of the 14 scans, a least-squares translation over all parallel pairs is only 0.6–1.2 mm, because the other legacy lines sit within 1–2 mm of gold.
- A search over every candidate single shift recovers at most one extra line on 13 of the 14 scans. The exception is sc_e104a6f42d31, where a 4.0 mm shift matches both lines. Its offsets are 4.6 and 4.8 mm, just past tolerance: a locus difference, not a frame error.
- sc_e8c946b837aa: both lines are about 16 mm off, but in nearly orthogonal directions, so no single shift fits (residual 11.4 mm).
- The legacy lines lie on the scan surface (median distance to raw vertices 0.1–1.4 mm for 49 of 52 lines, at most 2.6 mm). The catalog also snapped their endpoints to raw vertices.

No labels were changed.
