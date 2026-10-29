# Zero-shot programme, phase 1 (2026-10-25): inventory, lessons, baseline, plan

Files:
- `INVENTORY.json`: one row per scan, built by `tools/build_inventory.py` from metadata only.
- `SPLIT_FROZEN.json`: built by `tools/freeze_split.py`. sha256 `d25ffd4f4d509624bb70387722d9a8814922cf85fa5d3bf821634db708468642`; INVENTORY sha `d8bd6b9c…`.
- `LESSONS.md`.
- `BASELINE.md` and `BASELINE.json`.
- `PLAN.md`.
- `RESEARCH.md`: the literature, written by the research agent.
- `d1/`: the D1 agent's work.
- `baseline/`: runs, logs and per-case DEV scores. The `*_TESTB_SEALED*` entries are sealed.
- `tools/`.

## Inventory summary

**Where we searched.**
- The repo: `data/bend_corpus/**`, `live_scans`, `test_files`, `uploads`, `docs/production/runlog`.
- `~/ShermanQC/source-assets`.
- `~/Downloads`, `~/Desktop`, `~/Documents`, including the email-attachment and SharePoint copies already on disk.
- Nothing was downloaded.

**What we found.** 289 unique mesh files, by sha256, after the custody exclusions:
- 206 BENCHMARK_V3 rows, which include the excluded synthetic fixtures and format alternates;
- 20 STL/OBJ copies of PLY scans;
- 49 Eyal50 CAD STLs (CAD only, no scans);
- 7 CAD reference meshes in `uploads/`;
- 3 derived caches;
- 1 unverified phone capture in `~/Downloads` (no SKU);
- 4 Eyal50-quarantine scans;
- 1 SKU that overlaps HOLDOUT10 (41880000_D.ply).

**Usable physical scans: 186** (147 part families, 132 five-digit family groups).

| Custody | Usable | With complete gold | Gold lines |
|---|---|---|---|
| DEV_CONSUMED (A1 tuning + other consumed) | 95 | 94 | |
| CONSUMED_DEVCHECK | 10 | 10 | |
| CONSUMED_FRESH | 81 | 78 | |
| **Total** | **186** | **182** (5 are zero-bend controls) | **915** |

- **Gold quality.**
  - A (GOLD_SET_V2, audited vision gold, incl. the Gold11 relabel): 103.
  - B (accepted AI vision, i.e. FRESH sealed labels + `ai_bend_lines_v1` 17139000): 79.
  - `human_bend_lines_v1` was not used and not modified.
- **Stage.** 169 full / 17 partial.
- **CAD.**
  - 60 gold scans have a STEP, and 56 of those compile with E1 (3 rolled parts give NO_RECIPE; the zero-bend control 47479000 frozen recipe is BLOCKED).
  - 122 gold scans have no CAD.
- **Usable scans without complete gold: 4.**
  - `sc_186463746daa` 23553000: its label failed the 2-verifier check.
  - `sc_4d5dc2b40b67` 45141000: partial label.
  - `sc_0178c0155934` 48381001: needs rework.
  - `sc_766be04dc456` 48389000: STL only, no studio label.
- **Not usable.**
  - 4 Eyal50-quarantine scans (owner decision).
  - 1 scan whose SKU overlaps HOLDOUT10.
  - 4 possible CAD reference meshes.
  - 9 LIDAR_DEMO synthetic scans and 3 synthetic fixtures.
  - 1 opaque copy.
  - Duplicate twins, counted once.
- **Every usable scan is consumed.** None is unseen by rule.
  - 65 gold scans have never had any engine or lane output.
  - From these, TEST-B = 61 scans / 50 families / 329 lines, group-disjoint from DEV. It was frozen before any new engine output (`PLAN.md` OD-1).
- **How many need new gold to reach 40+.**
  - 0 for DEV, which already has 121 gold scans.
  - For an honest held-out claim under the "consumed is never held-out" rule: **≥ 40 new scans**, none of which are on disk.
  - With the TEST-B interim: 0 new, plus a recommended blind re-audit of 61.
  - Locally, only 4 usable scans lack gold.

## Custody incident (reported, not hidden)

The wide hash census (`sha256` of mesh files, used to dedupe) hashed some forbidden files before they were filtered out:
- the 10 HOLDOUT10 files in `data/bend_corpus/incoming/2026-08-30_shaked_daily_scans/source_download/`, with their `~/Downloads` and `incoming_codex_fetch` copies;
- the two `~/Downloads/38173000_01 (V|X).ply` N001–N014 copies. The EHO4 list names CAD 38173001_01; the scan file is 38173000_01.

What this did and did not touch:
- **No geometry, label or manifest of these files was opened.** Only raw bytes went through sha256, to identify them.
- The hashes were deleted from the scratch output and appear in no deliverable.
- The HOLDOUT10 membership was confirmed only by testing the hashes against the non-sealed allowlist `research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json`.

Future census scripts must prune `2026-08-30_shaked_daily_scans` and `38173000_01` by path.

## Baseline headline (details in `BASELINE.md`)

| Cohort | A0 zero-shot, micro / family-macro | E3 e3-1.12.0, micro / family-macro |
|---|---|---|
| DEV, 116 scans / 91 families | 0.322 / 0.267 (P 0.60, R 0.22) | – |
| DEV E3-eligible, 49 / 34 | 0.356 / 0.287 | **0.951 / 0.820** (P 0.993, R 0.912) |
| DEV A1 + other consumed, 89 | 0.334 / 0.274 | 0.987 (31 scans) |
| DEV consumed DEV-CHECK, 10 | 0.376 / 0.302 | 0.885 (5) |
| DEV consumed FRESH, 17 | 0.224 / 0.235 | 0.864 (13) |
| **TEST-B (sealed, aggregate only), 61 / 50** | **0.310 / 0.252** (P 0.72, R 0.20) | not run |

## Gold setup (2026-10-25, owner OD-1..OD-4 approved: "execute without waiting")

- Freeze-ledger NOTE seq 11 (`frz_8078a8cc0ec4`, entry sha256 `d6c09e0e…`): binds SPLIT_FROZEN sha `d25ffd4f…` and records OD-1 (TEST-B is the programme-sealed test), OD-2 (blind re-mark and adjudication of all 61), OD-3 (the 5 DEV-CHECK never-run cases stay unrun) and OD-4 (the Eyal50 quarantine is not used). The ledger still verifies, now with 11 entries.
- `gold/GOLD_LEDGER.jsonl` is hash-chained and append-only. It holds 65 REGISTERED scans: 4 DEV scans needing gold and 61 TEST-B. Each scan's sha256 prefix equals its id.
- The guard is `research/zeroshot/common/gold_guard.py`, with the wrapper `tools/guarded_slot.sh`. It refuses:
  - a ledger scan that is not FROZEN;
  - TEST-B, unless `--final-testb` is passed;
  - unknown scans;
  - with `--cad-path`, the OD-3 cases.
- `dev_folds.json` holds 5 grouped folds over DEV (25/24/24/24/24 scans, 80 groups), plus the fold of each of the 4 pending scans.
- Batches:
  - `gold/batches.json` is marker-facing: D0 has 4 + 1 canary, T1–T5 have 10 + 1 and T6 has 11 + 1.
  - The private key and the canaries are in `gold/canaries.json`. Markers never read it.
  - Scan copies with uniform metadata are in `gold/<batch>/scans/`.
- Marker and adjudicator commands are in `gold/MARKER_HOWTO.md`. `tools/gold_setup.py` and `tools/stl_same_frame_ply.py` are the setup tools.

## Integration and TEST-B (2026-10-25, integrator)

- `integration/README.md`: zs_lane_v1 (`research/zeroshot/zs_lane_v1/`), DEV out-of-fold on all 125 DEV scans, the choice of primary, and the CAD-path prototype and plan.
  - zs: 0.944 / 0.944. Z3: 0.914. Z2: 0.915.
- `testb/PREREG.json`: the pre-registration. Freeze-ledger NOTE seq 12, plus RUN seq 13.
- `testb/RESULTS.md` / `RESULTS.json`: the one TEST-B run. Freeze-ledger NOTE seq 14.
  - Primary zs_lane_v1: **0.855 micro [0.764, 0.926] / 0.855 family-macro** against the re-audit gold.
  - Against the old FRESH labels: 0.893.
- **TEST-B is CONSUMED.** Never re-run it or tune on it.

## N1 new-family batch (2026-10-25, adjudicator and one-time runner)

- **Gold.** `gold/N1/ADJUDICATION_N1.json`.
  - A/B agreement F1 1.0. Canary `gb_3f9a9c` scored 1.0 / 1.0 / 1.0.
  - Line counts: 47085000 has 26 lines; 41018000, 45351000 and 51101001 are zero-bend rolled shells.
  - Labels are in the new root `data/bend_corpus/annotations/ai_bend_lines_newscans_n1_v1/`.
  - Gold ledger: REGISTER seqs 131–134, FROZEN seqs 135–138.
- **New tool `tools/gold_ledger_newscans.py`.** It is the narrowest mechanism for the `NEW_UNSEEN_FAMILY_TEST` portion: REGISTER from `newscans_<batch>/REGISTRY.json`, and FROZEN only into `ai_bend_lines_newscans_<batch>_v1/`.
  - `gold_guard.py` is unchanged, because it is pinned in both lane fingerprints.
  - The guard admits N1 scans through its existing GOLD_LEDGER_FROZEN basis.
- **Run.** `newscans_N1/PREREG.json` (freeze-ledger NOTE seq 17), RUN seq 18, RESULTS NOTE seq 19.
  - Primary zs_lane_v1: 0.857 on 47085000 and 0 FP lines on the 3 rolled shells. v2: 0.898.
  - **N=4. Too small for a headline claim.** E3 was not run: the customer sent `.SLDPRT` files, not STEP.
  - Visuals: `newscans_N1/visuals/index.json`.

## zs_lane_v3 (2026-10-25, v3 packager)

- **Lane.** `research/zeroshot/zs_lane_v3/` is zs_lane_v2 plus EXT (adopt the absorbed Z2 extent) plus Z3R (rescue strong Z3-only lines). The base is v2.
- **Freeze.** `v3/FREEZE.md`, freeze-ledger NOTE seq 20 (`frz_0590fac51ef3`). The code fingerprint is `e34e1ba7…5b4a`.
- **DEV OOF.** 0.9660 [0.951, 0.979].
  - Against v1: +0.0216 [+0.0034, +0.0458].
  - Against v2: +0.0082 [−0.0020, +0.0248].
  - Control FP lines: 1, the same as v2.
  - The fix parameters are selection-aware; the disclosures are in the lane README.
- **Next.** The one-time fresh-scan test: `v3/PREREG_FRESH_TEMPLATE.json`, with gold frozen first.
- **Production.** The v3 pin is prepared but not live. `pin.DEFAULT_LANE` is still zs_lane_v1. The flip steps are in `v3/REPIN_PLAN.md`.
