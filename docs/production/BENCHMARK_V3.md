# Benchmark V3: broadened bend benchmark with a sealed FRESH split

Generated 2026-10-17T18:55:25Z from metadata only by `docs/production/runlog/benchmark_v3/_build/build_benchmark_v3.py`. Machine-readable manifest: `docs/production/runlog/benchmark_v3/BENCHMARK_V3.json` (sha256 `67cfd374d187a4c6...`). It merges BENCHMARK_V2 (which already includes the review11 verdicts and the LIDAR_DEMO exclusions) with the 91-scan benchmark_v3 queue from the Codex SharePoint/email fetch of 2026-10-17 (292 files, 7.6 GB; 3 `.crdownload` remnants ignored).

## FRESH rule (binding)

- A part is FRESH iff it never appeared in any tuning or evaluation before (not Gold11, Gold37, physical15 or other-consumed per BENCHMARK_V2.json and merged_inventory.json). Split is per part, never per scan.
- FRESH truth is for the single final evaluation only, with one frozen engine config hash. Never for tuning, thresholds, label edits, case selection or any accuracy-loop look. If FRESH truth or results are looked at earlier, or the engine changes after the FRESH run, those parts become CONSUMED and the run is reported as such.
- FRESH truth is sealed under `data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED` and `docs/production/runlog/benchmark_v3/fresh_SEALED`. Labeling agents looked at FRESH scans to create truth (truth creation is not tuning). The engine/accuracy loop must never read FRESH scans, labels or lines dirs.
- This manifest and BENCHMARK_V3.md list FRESH scans with split, truth source and status only. Per-scan FRESH bend counts, line geometry, CAD comparisons and judge details are only in fresh_SEALED/BENCHMARK_V3_FRESH_SEALED.json.

## Summary counts

| Count | Value |
|---|---|
| Unique parts with a benchmark scan | 151 |
| Benchmark scans | 193 |
| Parts with a benchmark scan and a STEP | 40 |
| CONSUMED parts / scans | 82 / 112 (106 from V2, 6 new; 4 PLY twins of V2 scans not counted) |
| FRESH parts / scans on disk | 69 / 81 |
| FRESH scans / parts with accepted truth | 79 / 67 |
| FRESH accepted bend lines (sealed) | 433 |
| FRESH partial truth (label_complete=false) | 1 scan |
| FRESH needs rework (no truth) | 1 scan |
| FRESH parts / scans with a STEP | 13 / 14 |
| Staged or incomplete scans | 20 (V2 16, V3 consumed 2, V3 FRESH 2) |

Truth sources over benchmark scans: AI_GOLD_ACCEPTED 55, AI_GOLD_ACCEPTED_SEALED 79, AI_GOLD_PARTIAL_SEALED 1, CAD_RECIPE 2, GOLD_LINES_AI_LEGACY 42, GOLD_LINES_HUMAN 13, NONE_NEEDS_REWORK 1.

## V3 label writes (2026-10-17)

- FRESH: 79 sealed studio labels written to `data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED` (426 lines), 5 of them replacing stale pre-reconcile labels. None went to `ai_bend_lines_v1`.
- CONSUMED: 10 labels written to `ai_bend_lines_v1` (51 lines).
- Refused by write-label: `sc_766be04dc456` (FRESH): LabelWriteError: the studio inventory indexes .ply files only; label a PLY asset.
- Not written (NEEDS_REWORK): `sc_0178c0155934`.
- ACCEPT_PARTIAL with `label_complete=false` in the provenance sidecar: `sc_4d5dc2b40b67`.
- No drop_labels were requested, so no `_final` lines runs were needed. The BENCHMARK_V2 rebuild for the 42288005 relabel was already done (a dry-run rebuild matched the on-disk V2 json and md exactly), so it was not rerun.

## AI vs CAD bend-count agreement

CONSUMED V3 scans (per scan):

| Scan | Part | Stage | AI real bends | CAD bend lines | Class | Note |
|---|---|---|---|---|---|---|
| `sc_3c93a0bf6f97` | 11979003 | - | 7 | 9 | AI_LT_CAD_CONVENTION | V10 9 keeps the two step folds split at a full-width window; owner window rule (ENGINE_COMPILE 'Windows', names 11979003) merges each into one line -> 7. All 11 V10 segments covered. The V2 STL twin label (sc_d45186653eed) still has 9 lines; owner should align it (merge its B2+B3 and B4+B5). |
| `sc_43a32fe3b4a3` | 15651004 | THIRD_SCAN | 2 | 3 | AI_LT_CAD_STAGED_UNFORMED | 15651004_01 3RD_SCAN mid-bending stage; 1 of 3 CAD bends not formed |
| `sc_1a5377ac7c15` | 15651004 | SECOND_SCAN | 1 | 3 | AI_LT_CAD_STAGED_UNFORMED | 15651004_01 SECOND_SCAN mid-bending stage; 2 of 3 CAD bends not formed |
| `sc_c27168a2de83` | 28289002 | - | 3 | 3 | AGREE |  |
| `sc_8026eedce786` | 38172000 | FULL_SCAN | 12 | 12 | AGREE |  |
| `sc_ef0abac61ae7` | 42288005 | PARTIAL_SCAN_1 | 5 | 8 | AI_LT_CAD_STAGED_UNFORMED | 42288005_01 PARTIAL_SCAN_1; 3 long channel folds not formed |
| `sc_32099fb52638` | 42288005 | PARTIAL_SCAN_2 | 6 | 8 | AI_LT_CAD_STAGED_UNFORMED | 42288005_01 PARTIAL_SCAN_2; C4/C7 not formed |
| `sc_2852c2845b94` | 45796000 | - | 8 | 8 | AGREE |  |
| `sc_db487cbd29a7` | 47266000 | - | 1 | 1 | AGREE |  |
| `sc_4057ba2144fb` | 47959001 | PARTIAL_SCAN_2, PRELIM_REV | 6 | 6 | AGREE | task header said CAD 3; inventory and step_v10/0d25231fea16.json give 6 (v10 = lenient = same-plates-merged = 6); labels match the 6 CAD bends one to one. PARTIAL_SCAN_2 tag, but every CAD bend is formed. |

FRESH V3 scans (counts only; per-scan rows are sealed):

| Class | Scans |
|---|---|
| AGREE | 8 |
| AI_GT_CAD_EXTRACTOR_UNDERCOUNT | 5 |
| AI_LT_CAD_STAGED_UNFORMED | 1 |
| CAD_UNKNOWN | 66 |

Every FRESH mismatch with a known CAD count is explained by the judge (staged/unformed bends, or V10 extractor refusal/under-count confirmed by a lenient STEP census, a direct cylinder count or the drawing); unexplained: 0. V2 reference: `docs/production/runlog/broad_baseline/cad_crosscheck.json` (one confirmed AI error, `sc_2233372b8df7`, fixed 6 -> 8).

## Rejected or excluded scans

| Scan | Part | Role | Basis |
|---|---|---|---|
| `sc_d5d7f0cfa36c` | 11979003 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_43e1dbdce7e3` | 11979003 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_657b87ceeba6` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_573c999ef6ea` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_123f84b5afcb` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_c2ff6d5b6c2f` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_9018b86ccc9a` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_a0b6add3f6e6` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_e632352b37f2` | 47266000 | EXCLUDED_NOT_A_SCAN | LIDAR_DEMO synthetic resample (synthetic_demo_exclusions) |
| `sc_0e4bf3087d81` | UNKNOWN:sc_0e4bf3087d81 | EXCLUDED_SYNTHETIC_FIXTURE | synthetic test fixture (5000-point Open3D cloud, ~50x40x50 mm ideal L), not a Sherman scan |
| `sc_241aa95bfdfc` | UNKNOWN:sc_241aa95bfdfc | EXCLUDED_REDACTED_OPAQUE_COPY | only copy is evaluator-opaque; path redacted |
| `sc_793a86afb5db` | UNKNOWN:sc_793a86afb5db | EXCLUDED_SYNTHETIC_FIXTURE | synthetic test fixture (3500-point Open3D cloud), not a Sherman scan |
| `sc_fb35a9f4c6bf` | UNKNOWN:sc_fb35a9f4c6bf | EXCLUDED_SYNTHETIC_FIXTURE | synthetic test fixture (5000-point Open3D cloud, ~50x40x50 mm ideal L), not a Sherman scan |
| `sc_0178c0155934` | 48381001 | FRESH NEEDS_REWORK | judge: every kept line lands, but the label set is incomplete; fixable without redoing labelers/verifiers (add the round-1 pairs kept in the _cont state, one lines run under the lock, check the new close-ups). Not written; no truth yet. Details sealed. |

V3 inventory files not queued: EYAL50_QUARANTINE 4, FORMAT_ALTERNATE_OF_PLY 28, FORMAT_ALTERNATE_V2 3, HAS_GOLD 84, IN_PROGRESS_label_lines_r2+label_lines_r3 2, IN_PROGRESS_label_lines_r3 3, IN_PROGRESS_label_lines_r3+label_prep_r3 1, OPAQUE_COPY_CUSTODY 1, SKU_OVERLAPS_SEALED_SET 1.

## Source mismatch: 49182000 / 42725000

The SharePoint link listed under email subject '49182000_B' (msg 1a0516d6677aab46, 2026-08-30) served 42725000_B.ply. The part is taken from the filename (42725000). The file is sha256-identical (d82bd4f4...) to the V2 CONSUMED scan sc_d82bd4f417a2 (42725000_B, GOLD_LINES_AI_LEGACY), so it adds nothing new. 49182000 still has no scan on disk and is not in the benchmark; its real scan must be requested again.

## Flags for the owner

- `sc_3c93a0bf6f97`, `sc_d45186653eed`: Same mesh (PLY vs STL). V3 label 7 lines (window rule) vs V2 label 9 lines. Counted once (V2 scan). Owner: align the V2 label (merge B2+B3, B4+B5) or decide the convention.
- `sc_db487cbd29a7`, `sc_9b9fd762f361`: PLY copy of the V2 STL scan (same 131655 points); labels agree (1). Counted once.
- `sc_43a32fe3b4a3`, `sc_927a2fa13067`: PLY twin of the V2 3RD_SCAN STL; labels agree (2). Counted once. Gives the V2 STL-only scan a PLY studio label.
- `sc_1a5377ac7c15`, `sc_8dfdab44605d`: PLY twin of the V2 SECOND_SCAN STL (human gold n=1); AI label agrees (1). Counted once.
- `sc_4057ba2144fb`: Workflow task header gave CAD 3; repo inventory/step_v10 give 6. Other judge prompts may carry wrong CAD counts; V3 uses inventory values.
- `sc_c27168a2de83`: Part 28289002 is CONSUMED_UNVERIFIED; its label now sits in ai_bend_lines_v1 (tuning pool), so moving it to FRESH is no longer possible.
- `sc_766be04dc456`: FRESH STL scan: write-label refuses non-PLY assets; accepted truth exists only as the sealed lines.json. Export a PLY in the same frame to get a sealed studio label.
- Several verifiers saw the labeler/reconciler 'notes' field in lines.json before forming their count, so they were not fully blind; judges checked geometry independently. Strip notes from what verifiers read in future runs.
- One FRESH lines dir had pairs.json overwritten by a killed re-run reconciler; restored byte-identical from the preserved copy (lines.json unchanged). One FRESH lines dir holds a stale render PNG that lines.json does not reference (write-label ignores it).

## Scans by part

| Part | Split | STEP | Scan | Source | Role | Truth source |
|---|---|---|---|---|---|---|
| 10835000 | FRESH | no | `sc_0e5dd5f6490e` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 10835000 | FRESH | no | `sc_ccdb74afbde5` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 10839000 | CONSUMED | yes | `sc_08686dc1d5d4` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 10839000 | CONSUMED | yes | `sc_71cf0b216f43` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 10839000 | CONSUMED | yes | `sc_cec08dfca9ab` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 11979003 | CONSUMED | yes | `sc_3c93a0bf6f97` | V3 | FORMAT_ALTERNATE_PLY_TWIN | AI_GOLD_ACCEPTED |
| 11979003 | CONSUMED | yes | `sc_43e1dbdce7e3` | V2 | EXCLUDED_NOT_A_SCAN | CAD_RECIPE |
| 11979003 | CONSUMED | yes | `sc_d45186653eed` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 11979003 | CONSUMED | yes | `sc_d5d7f0cfa36c` | V2 | EXCLUDED_NOT_A_SCAN | CAD_RECIPE |
| 11979003 | CONSUMED | yes | `sc_f99d883f2754` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 12851000 | FRESH | no | `sc_ba2286ee9c19` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 15651004 | CONSUMED | yes | `sc_03395097cd96` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 15651004 | CONSUMED | yes | `sc_1a5377ac7c15` | V3 | FORMAT_ALTERNATE_PLY_TWIN | AI_GOLD_ACCEPTED |
| 15651004 | CONSUMED | yes | `sc_43a32fe3b4a3` | V3 | FORMAT_ALTERNATE_PLY_TWIN | AI_GOLD_ACCEPTED |
| 15651004 | CONSUMED | yes | `sc_4d277fbe7854` | V2 | benchmark_scan | CAD_RECIPE |
| 15651004 | CONSUMED | yes | `sc_8dfdab44605d` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 15651004 | CONSUMED | yes | `sc_927a2fa13067` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 15658000 | FRESH | no | `sc_216c636d3bcb` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 15658000 | FRESH | no | `sc_41d9f652df57` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 15658000 | FRESH | no | `sc_90688c5ab7be` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 17139000 | CONSUMED | yes | `sc_90257b128d80` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 21763000 | FRESH | no | `sc_e36194d827bf` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 21763000 | FRESH | no | `sc_fb178c492b04` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 22352000 | FRESH | no | `sc_daeb617ca8ae` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 23317000 | FRESH | yes | `sc_afd4de0698d8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 23553000 | CONSUMED | yes | `sc_186463746daa` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 23553000 | CONSUMED | yes | `sc_5a1ecd7a4db3` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 23553000 | CONSUMED | yes | `sc_6eba02723d4b` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 23976000 | FRESH | no | `sc_f7b2a638e246` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 24694000 | CONSUMED | no | `sc_8b72c3ff233f` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 26687021 | FRESH | no | `sc_8700ef964309` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 26687021 | FRESH | no | `sc_be73cab0e96e` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 27796002 | FRESH | no | `sc_077889c8aeab` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 27796002 | FRESH | no | `sc_0e5eca6fd886` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 27796005 | FRESH | no | `sc_c75fb685efcd` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 27796005 | FRESH | no | `sc_c768328a70e6` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 28289002 | CONSUMED | yes | `sc_c27168a2de83` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 28870000 | CONSUMED | yes | `sc_38091ac91d9d` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 28870000 | CONSUMED | yes | `sc_3bde9996f14d` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 34371016 | FRESH | no | `sc_f5655bd1908e` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 34499000 | CONSUMED | yes | `sc_828bc34d9c8b` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 35705010 | CONSUMED | no | `sc_050cba324c35` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 35705010 | CONSUMED | no | `sc_3e4b28f24a9a` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 36192000 | FRESH | no | `sc_3bf77708bb82` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 36192000 | FRESH | no | `sc_4dd0e37dd764` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 37071000 | FRESH | no | `sc_1bfde19a4430` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 37126000 | CONSUMED | yes | `sc_4a4304bb06f7` | V2 | FORMAT_ALTERNATE | CAD_RECIPE |
| 37126000 | CONSUMED | yes | `sc_822f083e3dfc` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 37126000 | CONSUMED | yes | `sc_a7498a51b52a` | V2 | FORMAT_ALTERNATE | CAD_RECIPE |
| 37361005 | CONSUMED | yes | `sc_d1732de18023` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 38169000 | CONSUMED | no | `sc_db1839480738` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 38172000 | CONSUMED | yes | `sc_07062eee17aa` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 38172000 | CONSUMED | yes | `sc_2714fb7e0c0b` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 38172000 | CONSUMED | yes | `sc_4f8188b4d1a8` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 38172000 | CONSUMED | yes | `sc_8026eedce786` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 38172000 | CONSUMED | yes | `sc_f2475286312b` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 38173000 | CONSUMED | no | `sc_3435e688d344` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 38173000 | CONSUMED | no | `sc_b3cde5d10544` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 38539000 | FRESH | no | `sc_8a4a24043800` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 38539000 | FRESH | no | `sc_92f57084d4c5` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 38898000 | FRESH | yes | `sc_ead8e407c174` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 38965002 | CONSUMED | no | `sc_6101a12648a9` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 38965002 | CONSUMED | no | `sc_c60c74f5ae01` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 38968000 | FRESH | yes | `sc_c0ce5ecd2ba3` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 40500000 | CONSUMED | yes | `sc_152f1b1aacb2` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 40500000 | CONSUMED | yes | `sc_eaa89c6d1f1f` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 40851000 | FRESH | no | `sc_c11b80b6d2af` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 41017000 | CONSUMED | no | `sc_43417f858d5a` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 42288005 | CONSUMED | yes | `sc_2233372b8df7` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 42288005 | CONSUMED | yes | `sc_32099fb52638` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 42288005 | CONSUMED | yes | `sc_ef0abac61ae7` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 42426000 | FRESH | no | `sc_fc2657f59d20` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 42725000 | CONSUMED | no | `sc_d82bd4f417a2` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 42821000 | FRESH | no | `sc_f6b7f5e5ae63` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 43161000 | FRESH | no | `sc_5d7b61913ee0` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 43244000 | FRESH | no | `sc_ec663db9d917` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 43780000 | FRESH | no | `sc_1d920838b34b` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 43798000 | FRESH | no | `sc_5a382360b59f` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 43895000 | CONSUMED | no | `sc_589688be58d2` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 43920000 | CONSUMED | no | `sc_633e26e28e56` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 43976000 | CONSUMED | no | `sc_dff6419eacc4` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 44001000 | FRESH | no | `sc_705ec5de123b` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 44211000 | CONSUMED | yes | `sc_0adb7d717b35` | V2 | FORMAT_ALTERNATE | CAD_RECIPE |
| 44211000 | CONSUMED | yes | `sc_361fc1049e39` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 44211000 | CONSUMED | yes | `sc_cd68c1d49a63` | V2 | FORMAT_ALTERNATE | CAD_RECIPE |
| 44556000 | FRESH | no | `sc_c35199ee5039` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 45141000 | FRESH | no | `sc_4d5dc2b40b67` | V3 | benchmark_scan | AI_GOLD_PARTIAL_SEALED |
| 45352000 | FRESH | no | `sc_3f6e22d70e4e` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 45796000 | CONSUMED | yes | `sc_1268c72dcc30` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 45796000 | CONSUMED | yes | `sc_2852c2845b94` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 45796000 | CONSUMED | yes | `sc_9ef2cf4e52cb` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 45796000 | CONSUMED | yes | `sc_c623d2e45bff` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 46045000 | FRESH | no | `sc_d094d32acc76` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 46127000 | FRESH | yes | `sc_147d3123a21d` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 46128000 | FRESH | yes | `sc_741c10b768dc` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 46414000 | CONSUMED | no | `sc_b76452ea6fa7` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 46465000 | FRESH | yes | `sc_4d62195d4e2d` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 46882000 | FRESH | no | `sc_938e948c1d00` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 47040000 | FRESH | no | `sc_23754ce82e06` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 47247004 | CONSUMED | no | `sc_8935b8a8e8aa` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47251002 | CONSUMED | no | `sc_1059a9192687` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47251002 | CONSUMED | no | `sc_9bd69bc6535c` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_123f84b5afcb` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_573c999ef6ea` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_657b87ceeba6` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_9018b86ccc9a` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_9b9fd762f361` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_a0b6add3f6e6` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_c2ff6d5b6c2f` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_db487cbd29a7` | V3 | FORMAT_ALTERNATE_PLY_TWIN | AI_GOLD_ACCEPTED |
| 47266000 | CONSUMED | yes | `sc_e632352b37f2` | V2 | EXCLUDED_NOT_A_SCAN | AI_GOLD_ACCEPTED |
| 47479000 | CONSUMED | yes | `sc_b4719db3dd69` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47480000 | FRESH | no | `sc_5d1715d17af6` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 47555020 | CONSUMED | no | `sc_3b3e8c9f04e3` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47632001 | CONSUMED | no | `sc_cca7d7680a51` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 47959001 | CONSUMED | yes | `sc_4057ba2144fb` | V3 | benchmark_scan | AI_GOLD_ACCEPTED |
| 47959001 | CONSUMED | yes | `sc_a0305a6cf787` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 47959001 | CONSUMED | yes | `sc_cebb070285b5` | V2 | benchmark_scan | CAD_RECIPE |
| 48099000 | FRESH | no | `sc_0412d642d936` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48099000 | FRESH | no | `sc_76b89855765e` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48101000 | FRESH | yes | `sc_1f1d06c24e9f` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48333004 | FRESH | no | `sc_c54444e2c7df` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48333006 | CONSUMED | no | `sc_b7bd53831ee2` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48333018 | CONSUMED | no | `sc_7d28946d9f49` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48334000 | FRESH | no | `sc_20a96de8b712` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48340007 | CONSUMED | no | `sc_c1d112196ae5` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48340008 | CONSUMED | no | `sc_521e2ee69b21` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48381001 | FRESH | no | `sc_0178c0155934` | V3 | benchmark_scan | NONE_NEEDS_REWORK |
| 48384000 | FRESH | no | `sc_2d74c97bb017` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48389000 | FRESH | no | `sc_766be04dc456` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48412000 | CONSUMED | no | `sc_1aeb5ee39ef0` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48555000 | FRESH | yes | `sc_7bde16010a96` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48565005 | CONSUMED | no | `sc_e104a6f42d31` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48565022 | CONSUMED | no | `sc_49e5e7a1c973` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48758000 | FRESH | no | `sc_92207963e00b` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48811000 | CONSUMED | yes | `sc_ff1667a2a80a` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 48900035 | FRESH | no | `sc_bff1526135fb` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48910000 | FRESH | no | `sc_b0fb59a69e99` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48926001 | FRESH | yes | `sc_d4b9d13966bb` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48932000 | CONSUMED | no | `sc_be840127c462` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48933000 | FRESH | no | `sc_1dd687fd8ff8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48934000 | FRESH | no | `sc_deba7f81ceec` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48938000 | CONSUMED | no | `sc_eb1eab182228` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 48963008 | CONSUMED | yes | `sc_2d3e4a623a31` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 48963009 | FRESH | yes | `sc_217cfc0f4c57` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 48991006 | CONSUMED | yes | `sc_6e5d5a2f60f5` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 49001000 | CONSUMED | yes | `sc_7a1773eca9fa` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 49001000 | CONSUMED | yes | `sc_8379cebef9e3` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 49024000 | CONSUMED | yes | `sc_3434b523db47` | V2 | FORMAT_ALTERNATE | CAD_RECIPE |
| 49024000 | CONSUMED | yes | `sc_70b39f530f00` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 49025000 | FRESH | no | `sc_09aaf245becb` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49125000 | CONSUMED | yes | `sc_c748537194da` | V2 | FORMAT_ALTERNATE | NONE |
| 49125000 | CONSUMED | yes | `sc_efd0d987f60b` | V2 | FORMAT_ALTERNATE | NONE |
| 49125000 | CONSUMED | yes | `sc_fee685eee4db` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| 49162000 | CONSUMED | no | `sc_b92f211e6ec0` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49163000 | CONSUMED | no | `sc_469febe1cd7d` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49186000 | FRESH | no | `sc_3f6a4a32ffdc` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49187000 | FRESH | no | `sc_bfabcec897fd` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49188000 | CONSUMED | no | `sc_711eb2a389a4` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191008 | FRESH | no | `sc_cfe91a0336d4` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49191019 | FRESH | no | `sc_5dadc4b00c47` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49191023 | CONSUMED | no | `sc_28a25c37a928` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191026 | FRESH | no | `sc_7c6712aacca8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49191030 | CONSUMED | no | `sc_e218b16ed070` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191032 | CONSUMED | no | `sc_891a8e4a2ee1` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191037 | CONSUMED | no | `sc_9b38b1167834` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191038 | CONSUMED | no | `sc_d741e1d515bf` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191054 | CONSUMED | no | `sc_24c57f2660b3` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49191061 | CONSUMED | no | `sc_e0285cea8c56` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49201000 | CONSUMED | yes | `sc_203b95d7a753` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49204020 | CONSUMED | yes | `sc_e080ae094f35` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 49207000 | FRESH | yes | `sc_6fa053514df8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49227000 | CONSUMED | no | `sc_1d601c46ff9c` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49289000 | FRESH | no | `sc_57f3d5a23f60` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49291000 | FRESH | no | `sc_0f7ac8e49e99` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49292000 | FRESH | no | `sc_2fc703f32b3b` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49293001 | CONSUMED | no | `sc_e8c946b837aa` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 49355000 | FRESH | no | `sc_03b571de167d` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49438000 | FRESH | no | `sc_f86e40f38a37` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49461000 | FRESH | no | `sc_c7aa4f510d75` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49553000 | FRESH | no | `sc_9b47565ad2ac` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49565000 | FRESH | no | `sc_4f72b44afd5a` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49566000 | FRESH | no | `sc_528766342080` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 49783000 | FRESH | yes | `sc_ead69b8181a8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50278001 | FRESH | no | `sc_d02d75eaff89` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50278001 | FRESH | no | `sc_d338ed2dced8` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50482000 | FRESH | no | `sc_d151c2d71804` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50535000 | CONSUMED | no | `sc_9c84cc623d9b` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50537000 | CONSUMED | no | `sc_ce3de3af51b6` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50539000 | CONSUMED | no | `sc_85a7f8cca6bc` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50545000 | CONSUMED | no | `sc_2c807204c472` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50545000 | CONSUMED | no | `sc_d27acfe5468d` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50546000 | CONSUMED | no | `sc_baa71ce6bc5d` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50546000 | CONSUMED | no | `sc_f4bc782eff8b` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50552000 | CONSUMED | no | `sc_9567d0d3f33d` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50553000 | CONSUMED | no | `sc_581c7de10fc6` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50556000 | CONSUMED | no | `sc_c1a1d6de4681` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50556000 | CONSUMED | no | `sc_d891ea8c5891` | V2 | benchmark_scan | GOLD_LINES_AI_LEGACY |
| 50566000 | CONSUMED | no | `sc_8bb74bda6756` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50567000 | CONSUMED | no | `sc_80472dfa8819` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50597000 | FRESH | yes | `sc_2476a49b0278` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50597000 | FRESH | yes | `sc_3baf989b1612` | V3 | benchmark_scan | AI_GOLD_ACCEPTED_SEALED |
| 50757000 | CONSUMED | yes | `sc_752e2bd54aad` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| 50757000 | CONSUMED | yes | `sc_777c64c88967` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_0e4bf3087d81 | CONSUMED | no | `sc_0e4bf3087d81` | V2 | EXCLUDED_SYNTHETIC_FIXTURE | NONE |
| UNKNOWN:sc_241aa95bfdfc | CONSUMED | no | `sc_241aa95bfdfc` | V2 | EXCLUDED_REDACTED_OPAQUE_COPY | NONE |
| UNKNOWN:sc_2e584e9d1ffe | CONSUMED | no | `sc_2e584e9d1ffe` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_793a86afb5db | CONSUMED | no | `sc_793a86afb5db` | V2 | EXCLUDED_SYNTHETIC_FIXTURE | NONE |
| UNKNOWN:sc_7a253806b10b | CONSUMED | no | `sc_7a253806b10b` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_9b930fe58a1b | CONSUMED | no | `sc_9b930fe58a1b` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_9c890e76934f | CONSUMED | no | `sc_9c890e76934f` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_a5e534e6097c | CONSUMED | no | `sc_a5e534e6097c` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_abe194f04ade | CONSUMED | no | `sc_abe194f04ade` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_f098dbceaa6d | CONSUMED | no | `sc_f098dbceaa6d` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_f362d8de507c | CONSUMED | no | `sc_f362d8de507c` | V2 | benchmark_scan | AI_GOLD_ACCEPTED |
| UNKNOWN:sc_fb35a9f4c6bf | CONSUMED | no | `sc_fb35a9f4c6bf` | V2 | EXCLUDED_SYNTHETIC_FIXTURE | NONE |
| ysherman | CONSUMED | no | `sc_36e5d6998b9b` | V2 | benchmark_scan | GOLD_LINES_HUMAN |
| ysherman | CONSUMED | no | `sc_80debe59ce20` | V2 | FORMAT_ALTERNATE | NONE |
| ysherman | CONSUMED | no | `sc_e83b9feb7090` | V2 | FORMAT_ALTERNATE | NONE |

## Custody

Excluded and not opened: HOLDOUT10, sealed truth, N001-N016, EHO4, held-out manifests, and the 11 sealed items Codex skipped (16231000_03, 38168001_02, 38173001_01, 26687013_A00, 31409000_01, 47215000_B, 42637000_B, 50476000_A, 31367000_A00, 30820000_A01). No V3 scan belongs to those parts. Nothing was reset, cleaned or committed.
