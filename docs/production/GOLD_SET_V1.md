# Gold set v1

Created 2026-10-18 at the owner's request of 2026-10-17 to grow the gold set to 50 or more scans. Every label in it is a vision label (`claude-vision-ai-v1`, `ai_adjudicated`). Each line was checked by eye on renders, close-ups and perpendicular cross-sections. No line is a machine draft.

- Machine-readable manifest: `docs/production/runlog/gold_set_v1/GOLD_SET_V1.json`
- Frozen case list: `docs/production/runlog/gold_set_v1/case_list_gold_v1.json`
  - case_list_sha256 `3d55bba1e95b…`
  - 58 cases, 328 lines (325 + 3 from the 2026-10-18 10-deg sweep)
  - matcher frozen at 5 deg / 4 mm / 60 % both spans, one-to-one Hungarian, apex locus, raw scan frame
- Rescore: `docs/production/runlog/gold_set_v1/rescore/RESCORE_SUMMARY.json`
- Build script: `docs/production/runlog/gold_set_v1/scripts/build_gold_set_v1.py`. It reads only, apart from what it writes under gold_set_v1. It runs no engine, prep or lines step.
- 2-verifier pass of 2026-10-18 on the scans the workflow had left EXCLUDED: `docs/production/runlog/gold_set_v1/verifier_pass_2026-10-18.json` (verifier texts verbatim). 3 passed and joined as GOLD_FIXED; 1 failed and stays excluded.
- New gold labels: `data/bend_corpus/annotations/ai_bend_lines_gold_v1/`, with a `provenance/` sidecar for each label. Gold11 stays in `ai_bend_lines_gold11_relabel_v1/`.
- Files left untouched: `human_bend_lines_v1`, the Gold11 human catalog and the existing `ai_bend_lines_v1` labels. Both checks are recorded in the manifest.

## Gold rule

A line counts only if all of these hold:

- It lies on the outer-skin apex of a real formed fold.
- It runs parallel to that fold.
- It covers at least 80 % of the fold's visible length.

How the rule is applied:

- A joggle counts as 2 bends.
- Formed small bends of 10 deg or more count (owner rule 2026-10-18; was 15-30 deg). Sweep: runlog/gold_set_v1/sweep_10deg/results.json.
- A fold interrupted by windows or reliefs that fold the same two plates is ONE line.
- These are not bends: hole walls, thickness and perimeter edges, embosses, gentle kinks in a rolled surface, and artifacts.
- In staged or partial scans, only the bends formed in that scan are labelled.

## Summary

| | |
|---|---|
| Total gold scans | **58** (>= 50: yes) |
| Gold11 (vision relabel, 2 blind verifiers each) | 11 scans / 89 lines. All 11 are AI vision labels, including CG65CD575B1C (sc_07062eee17aa), now AI at label revision 2 |
| New gold this run | 47 scans / 236 lines: 34 accepted as-is (3/3 auditors OK), 13 fixed and then passed by 2 verifiers (3 of them in the 2026-10-18 verifier pass) |
| Part families | 38 |
| Full / staged | 42 full / 16 staged. Staged covers partial, second/third scans and the stage-0 blank |
| Zero-bend controls | 3: sc_f99d883f2754 (flat blank), sc_3bde9996f14d (rolled profile, 28870000 SCAN_1), sc_b4719db3dd69 (rolled half-cylinder) |
| With CAD (STEP) | 43 |
| Total gold lines | **328** |
| Excluded candidates | 1 of 48 (see below) |

Custody:

- Only CONSUMED scans are used (BENCHMARK_V3 split CONSUMED, checked for all 58).
- No FRESH, `*_SEALED*`, HOLDOUT10, N001-N016, EHO4, LIDAR_DEMO synthetic scans or possible CAD reference meshes.
- The CONSUMED PLYs under `incoming_codex_fetch_2026-10-17` are included. Their FRESH siblings are not.

Studio assets:

- The 11 `~/Downloads` scans are labelled through their byte-identical studio copies in `data/bend_corpus/incoming/2026-10-17_benchmark_v2_downloads/`, so the asset sha256 matches.
- STL/OBJ: none was refused, because every gold scan here is a PLY.
- Five STL scans are represented by same-mesh PLY twins. Their vertex sets are identical, with a nearest-neighbour distance of 0.0 mm in both directions (`twin_vertex_check.json`).

| STL scan | PLY twin |
|---|---|
| sc_d45186653eed | sc_3c93a0bf6f97 |
| sc_927a2fa13067 | sc_43a32fe3b4a3 |
| sc_9b9fd762f361 | sc_db487cbd29a7 |
| sc_8dfdab44605d | sc_1a5377ac7c15 |
| sc_a0305a6cf787 | sc_cebb070285b5 |

## How good were the existing AI labels?

44 existing AI labels were audited: 43 that became gold, plus the 1 excluded.

- **33 (75 %)** were accepted unchanged. All 3 blind auditors marked every line OK and found nothing missing.
- **11** went to the vision fixer:
  - **3** were zero-bend controls and came out unchanged. The fixer re-verified that no formed fold exists.
  - **7** were fixed and passed by 2 verifiers. 14 lines changed in total:

| Scan | What changed |
|---|---|
| sc_3c93a0bf6f97 | B1 moved 1.30 mm, from the OML virtual sharp onto the outer-arc apex |
| sc_8935b8a8e8aa | 4 lines moved 0.40-0.53 mm onto the apex. The auditors' "missing bend" claims were rejected: they were lip kinks below 15 deg (three ~10.7 deg tab lips). 2026-10-18: owner lowered the floor to 10 deg; the 3 lips were added as B7-B9 (label revision 2, 2 blind verifiers passed) |
| sc_9c84cc623d9b | B1 moved 0.80 mm, and its extent was trimmed from 707 to 680 mm because it overran onto the flat past the relief step |
| sc_752e2bd54aad | B2 extent extended from 284 to 394 mm to cover the fold |
| sc_c27168a2de83 | B2 moved 1.43 mm, from the inner skin onto the outer-skin apex (2026-10-18 verifier pass) |
| sc_eaa89c6d1f1f | B3 and B4 moved 1.27 / 1.35 mm, from outside the outer arc onto the apex; B8 and B9 extents grew 11.9 to 13.1 and 12.5 to 13.2 mm (2026-10-18 verifier pass) |
| sc_9ef2cf4e52cb | B1 moved 1.83 mm, from the inner corner to the outer-arc apex; B4 extent grew from 94.1 to 100.3 mm (2026-10-18 verifier pass) |

  - **1** was excluded: its fixed lines failed the 2-verifier pass (B3 on the inner skin).

Kinds of error found:

- Every confirmed AI error was **position or extent**: an apex offset of 0.4-1.85 mm, the line on the inner skin or the virtual sharp, or an extent that over- or under-ran.
- **No AI label had a wrong bend count.** No real bend was missing, and no false bend was confirmed.
- In the 43 gold scans, 14 of the 216 AI lines changed (6.5 %). Counting the 1 excluded scan as well, 15 of 225 AI lines had a defect (6.7 %).
- Every changed line would still have matched under the frozen matcher (prior-vs-gold F1 = 1.0 on all 7 fixed scans). The fixes are sub-tolerance apex accuracy, not F1-visible errors.

For contrast, the 4 non-Gold11 human labels:

| Scan | Human lines that match the gold (frozen matcher) |
|---|---|
| sc_cebb070285b5 | 2 of 5 |
| sc_fee685eee4db | 0 of 2. They lie on the smooth roll; the real bends are a 2-bend joggle |
| sc_36e5d6998b9b | 1 of 12 |
| sc_1a5377ac7c15 | 1 of 1. Human = AI, accepted as-is |

The first three were replaced wholesale by vision gold.

## Rescore of existing predictions (no engine runs)

Inputs:

- Existing predictions in `broad_baseline/runs/best_f1` and `runs/runtime`, scored with `research/broad_baseline/score.py broad` and the frozen frame records.
- The "pre-audit" labels are the same cases with their `case_list_broad_v3` gold: the human catalog for Gold11, and the previous AI or human label as on disk for the others.

Coverage:

- 53 of 58 gold scans have predictions: 50 line-scored scans plus the 3 zero-bend controls. The 5 PLY twins are scored on their STL twin's predictions, with the PLY gold copied verbatim as `generic` scan_raw gold under `rescore/twin_gold/`.
- **5 gold scans have no predictions** (32 lines). They were never in the broad run, and no engine run was allowed here:

| Scan | Part / stage | Lines |
|---|---|---|
| sc_8026eedce786 | 38172000 FULL | 12 |
| sc_32099fb52638 | 42288005 SCAN_2 | 6 |
| sc_ef0abac61ae7 | 42288005 SCAN_1 | 5 |
| sc_4057ba2144fb | 47959001 scan2 | 6 |
| sc_c27168a2de83 | 28289002 B01 | 3 |

- No case was excluded before scoring.

Results on the 50 line-scored scans (36 families, 293 gold lines; the pre-audit gold has 277 lines):

| Lane | Gold | TP / FP / FN | micro P | micro R | **micro F1** | **family-macro F1** |
|---|---|---|---|---|---|---|
| best_f1 | gold v1 | 120 / 79 / 173 | 0.603 | 0.410 | **0.4878** | **0.4713** |
| best_f1 | pre-audit | 110 / 89 / 167 | 0.553 | 0.397 | 0.4622 | 0.4620 |
| best_f1 | delta | | | | **+0.0256** | **+0.0093** |
| runtime | gold v1 | 27 / 37 / 266 | 0.422 | 0.092 | **0.1513** | **0.1555** |
| runtime | pre-audit | 21 / 43 / 256 | 0.328 | 0.076 | 0.1232 | 0.1399 |
| runtime | delta | | | | **+0.0281** | **+0.0156** |

By cohort (micro F1, gold v1 vs pre-audit):

| Cohort | Scans | best_f1 | runtime |
|---|---|---|---|
| Gold11 | 11 | 0.5101 vs 0.4580 | 0.1414 vs 0.0741 |
| New gold | 39 | 0.4781 vs 0.4638 | 0.1550 vs 0.1385 |
| New gold, fixed only | 9 | 0.3617 vs 0.3125 | 0.1429 vs 0.0833 |
| Full | 39 | 0.5078 vs 0.4842 | 0.1754 vs 0.1362 |
| Staged | 11 | 0.4151 vs 0.3750 | 0.0556 vs 0.0645 |
| CAD available | 34 | 0.5092 vs 0.4822 | 0.1957 vs 0.1615 |

Gold11 on gold v1 reproduces the official gold11_relabel_v1 rescore exactly (best_f1 38/22/51, runtime 7/3/82).

Zero-bend controls: best_f1 predicts 5 false-positive lines, on sc_3bde9996f14d and sc_b4719db3dd69. runtime predicts 3, on sc_3bde9996f14d.

Reading:

- The audit raises F1 slightly in both lanes.
- Most of the gain comes from Gold11, where the human gold is replaced by the apex relabel.
- On the 39 scored new gold scans the change is small, because 33 of the 43 AI labels that became gold did not change.
- best_f1 recall stays low (0.41). The engine, not the gold, is the bottleneck.

## Excluded candidates (1 of 48)

| Scan | Reason |
|---|---|
| sc_186463746daa (23553000 SCAN_2) | The 2-verifier pass on the r2 fixed lines (`gold_set_v1/lines/sc_186463746daa_r2`) **failed**: both verifiers find B3 (right-arm top-end plate-to-slope fold, ~60 deg) on the inner skin, about one thickness (0.92-1.0 mm) inside the outer-skin apex (lines.json records B3 skin='inner'). The other 8 lines pass. Earlier fixer note: the current tool cannot place B3 on the outer apex, so this needs the **owner's decision**. Not written to the gold root. |

Joined on 2026-10-18 after both verifiers passed their fixed lines (written with write-label, rc=0, logs in `write_label_logs/`): sc_c27168a2de83 (28289002 B01, 3 lines), sc_eaa89c6d1f1f (40500000 SCAN1, 9 lines), sc_9ef2cf4e52cb (45796000 SCAN_2, 4 lines).

Candidates not chosen are recorded in `candidates.json` (`not_chosen`):

- 5 UNKNOWN-part legacy scans
- the second full scan of 45796000, cut by the 3-per-family cap
- 17139000, held back as FRESH_SCAN_OF_CONSUMED_PART
- the OBJ alias of a Gold11 scan
- the STL originals of the PLY twins
- possible CAD reference meshes and LIDAR_DEMO resamples, which are forbidden

## Per-scan list

| Scan | Part | Stage | STEP | Lines | How it became gold | Label (asset id) sha256 |
|---|---|---|---|---|---|---|
| sc_08686dc1d5d4 | 10839000 | PARTIAL_SCAN_1 (staged) | yes | 8 | Gold11 relabel (CGE4B0549E41) | `72bd51c4885d493d805f1108.json` `baf1550d5c7f` |
| sc_cec08dfca9ab | 10839000 | PARTIAL_SCAN_2 (staged) | yes | 10 | Gold11 relabel (CG08506AE97D) | `17293ebcf404be5a87e25b1f.json` `2805d8f3bf58` |
| sc_71cf0b216f43 | 10839000 | FULL (full) | yes | 12 | Gold11 relabel (CG84C8FA12D2) | `c615c4cd283605201afac7e8.json` `2176bb5dbcba` |
| sc_d1732de18023 | 37361005 | FULL (full) | yes | 4 | Gold11 relabel (CG4BDD1371B6) | `05df0a708054946a1d77f7ee.json` `7a5c355a4ec6` |
| sc_2714fb7e0c0b | 38172000 | GOOD_V (full) | yes | 12 | Gold11 relabel (CGF41F49066A) | `c57b956b11cab8f4a8d3dc17.json` `c9739d8a5737` |
| sc_4f8188b4d1a8 | 38172000 | DEFECTIVE_X (full) | yes | 12 | Gold11 relabel (CG1E0BBA2933) | `b7155cc16c9bf0290d331831.json` `be13473b32d9` |
| sc_07062eee17aa | 38172000 | PARTIAL_SCAN_1 (staged) | yes | 6 | Gold11 relabel (CG65CD575B1C) | `a962166c3019434c10628722.json` `b5e5ff8972b7` |
| sc_f2475286312b | 38172000 | PARTIAL_SCAN_2 (staged) | yes | 10 | Gold11 relabel (CGEBA9A869AC) | `a6a0b1c3f8052cb3a45412e8.json` `eb79eada1e29` |
| sc_152f1b1aacb2 | 40500000 | PARTIAL_SCAN_2 (staged) | yes | 7 | Gold11 relabel (CG64C412CC73) | `8370b5e44945adfba2f2fb34.json` `e9c0821a7078` |
| sc_7a1773eca9fa | 49001000 | PRELIM (staged) | yes | 5 | Gold11 relabel (CG191DEC7F81) | `3aa393ad34b7ace90d84985c.json` `7f0545d5651c` |
| sc_70b39f530f00 | 49024000 | FULL (full) | yes | 3 | Gold11 relabel (CG4FAF273093) | `8778c5e57263f92092cf88b4.json` `cdf860e1d284` |
| sc_3c93a0bf6f97 | 11979003 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 7 | fixed + 2 verifiers, PLY twin of sc_d45186653eed | `3b82060d9cf7d3492afd3949.json` `aa1358e82ee8` |
| sc_f99d883f2754 | 11979003 | STAGE0_BLANK (staged) | yes | 0 | fixed + 2 verifiers, zero-bend control | `9fb07af82dbf6c097cd621a1.json` `d24b76f0bed7` |
| sc_03395097cd96 | 15651004 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 3 | as-is, 3/3 auditors | `1c62d21a1fceedefd236259a.json` `8ee1dcbdbc41` |
| sc_1a5377ac7c15 | 15651004 | SECOND_SCAN (full) | yes | 1 | as-is, 3/3 auditors, PLY twin of sc_8dfdab44605d | `d214c5f02688bbc714baf2db.json` `abd786a314cb` |
| sc_43a32fe3b4a3 | 15651004 | THIRD_SCAN (staged) | yes | 2 | as-is, 3/3 auditors, PLY twin of sc_927a2fa13067 | `2a726688d7fb1f9fb769c259.json` `fcf1bd7ced20` |
| sc_6eba02723d4b | 23553000 | FULL (full) | yes | 16 | as-is, 3/3 auditors | `71f5e686304eac1244183772.json` `3e3d3922e1f1` |
| sc_5a1ecd7a4db3 | 23553000 | PARTIAL_SCAN_1 (staged) | yes | 4 | as-is, 3/3 auditors | `be3bccecf0023ae1e1191702.json` `10d98521a823` |
| sc_c27168a2de83 | 28289002 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 3 | fixed + 2 verifiers (2026-10-18) | `f06551a2f3eb9e8e9eba7b30.json` `bf20eb9b7bd7` |
| sc_38091ac91d9d | 28870000 | FULL (full) | yes | 2 | as-is, 3/3 auditors | `8da4836b46c1392161e5fa48.json` `596509813d05` |
| sc_3bde9996f14d | 28870000 | PARTIAL_SCAN_1 (staged) | yes | 0 | fixed + 2 verifiers, zero-bend control | `a92f04f2ef011fc53a09675c.json` `030bbb9869ce` |
| sc_828bc34d9c8b | 34499000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 3 | as-is, 3/3 auditors | `f795a2da509fda73a25e2556.json` `729fd7074722` |
| sc_822f083e3dfc | 37126000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 2 | as-is, 3/3 auditors | `c769e053715d973985c0c85c.json` `76c7d713f8e8` |
| sc_db1839480738 | 38169000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 5 | as-is, 3/3 auditors | `6a4c42d3a3e11f0eed7646a6.json` `77adc4179437` |
| sc_8026eedce786 | 38172000 | FULL_SCAN (full) | yes | 12 | as-is, 3/3 auditors | `385a0d0badade0917d26fea2.json` `f4cf48210fd4` |
| sc_6101a12648a9 | 38965002 | GOOD_V (full) | no | 3 | as-is, 3/3 auditors | `b19061f3e4a792fe31cdd82b.json` `0da44797073c` |
| sc_c60c74f5ae01 | 38965002 | DEFECTIVE_X (full) | no | 3 | as-is, 3/3 auditors | `197d7a88dbe776749a98de84.json` `0dcf9a020bb3` |
| sc_eaa89c6d1f1f | 40500000 | PARTIAL_SCAN_1 (full: every bend formed) | yes | 9 | fixed + 2 verifiers (2026-10-18) | `a0bf9352332d87f9c8dfc8e0.json` `bea27617cd6c` |
| sc_2233372b8df7 | 42288005 | FULL (full) | yes | 8 | as-is, 3/3 auditors | `ae525e4954effa70d87ba8eb.json` `9641fc07a9f5` |
| sc_32099fb52638 | 42288005 | PARTIAL_SCAN_2 (staged) | yes | 6 | as-is, 3/3 auditors | `bd3f29d2ede84ed086a01432.json` `078d419200db` |
| sc_ef0abac61ae7 | 42288005 | PARTIAL_SCAN_1 (staged) | yes | 5 | as-is, 3/3 auditors | `78a5da9c09d1226d20886787.json` `73249f0b5035` |
| sc_361fc1049e39 | 44211000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 1 | as-is, 3/3 auditors | `fbe906a2187e95605c36ef39.json` `fc52e1b04c44` |
| sc_1268c72dcc30 | 45796000 | FULL (full) | yes | 8 | as-is, 3/3 auditors | `60dc0a2751f6320eb6ddf3d5.json` `9a01e56f4219` |
| sc_c623d2e45bff | 45796000 | PARTIAL_SCAN_1 (staged) | yes | 5 | as-is, 3/3 auditors | `922ee6a9ac06f5df20ef4dd8.json` `2a079f2a6a9f` |
| sc_9ef2cf4e52cb | 45796000 | PARTIAL_SCAN_2 (staged) | yes | 4 | fixed + 2 verifiers (2026-10-18) | `b1038070f28cb7792f9d6090.json` `83816253ba4f` |
| sc_8935b8a8e8aa | 47247004 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 6 | fixed + 2 verifiers | `97d359a69811fb0f8e846d43.json` `07f943cf3386` |
| sc_1059a9192687 | 47251002 | DEFECTIVE_X (full) | no | 12 | as-is, 3/3 auditors | `393c99fadda689b9b49b561b.json` `743a305680d5` |
| sc_9bd69bc6535c | 47251002 | GOOD_V (full) | no | 12 | as-is, 3/3 auditors | `e1e41c2d133c34b6b5886eb5.json` `069fa0ea0788` |
| sc_db487cbd29a7 | 47266000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 1 | as-is, 3/3 auditors, PLY twin of sc_9b9fd762f361 | `cc735fa27ac03414cb6fc7a8.json` `f0229cb4de77` |
| sc_b4719db3dd69 | 47479000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 0 | fixed + 2 verifiers, zero-bend control | `af36cde5bd3c635214303345.json` `f65cb7db4c0c` |
| sc_3b3e8c9f04e3 | 47555020 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 1 | as-is, 3/3 auditors | `b238e9f7821479679797b5a9.json` `158c4716bf01` |
| sc_4057ba2144fb | 47959001 | PARTIAL_SCAN_2+PRELIM_REV (staged) | yes | 6 | as-is, 3/3 auditors | `a97252bf6a4f6f2f2eabe2fa.json` `bce446359873` |
| sc_cebb070285b5 | 47959001 | PARTIAL_SCAN_1 (staged) | yes | 5 | fixed + 2 verifiers, PLY twin of sc_a0305a6cf787 | `d190acb5f24b27246619f3bb.json` `0f6a6556f21d` |
| sc_ff1667a2a80a | 48811000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 4 | as-is, 3/3 auditors | `6069f1090f154ea8afffc236.json` `f228b2f216f7` |
| sc_2d3e4a623a31 | 48963008 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 3 | as-is, 3/3 auditors | `febe9a3931a3027bf3799bd2.json` `d7d8c9da39ae` |
| sc_6e5d5a2f60f5 | 48991006 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 5 | as-is, 3/3 auditors | `f1840bb889ab736f1c9efa55.json` `84afd5a9aab4` |
| sc_fee685eee4db | 49125000 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 2 | fixed + 2 verifiers | `d539518a0a75dbbf9332b4be.json` `96dde6f9fcd5` |
| sc_e080ae094f35 | 49204020 | FULL_PRESUMED_NO_STAGE_HINT (full) | yes | 12 | as-is, 3/3 auditors | `99622fd56079834417a0605c.json` `41e5ae69ad4e` |
| sc_9c84cc623d9b | 50535000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 2 | fixed + 2 verifiers | `4528c30cfa5d165d1914cd69.json` `29838caa2ef8` |
| sc_ce3de3af51b6 | 50537000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 2 | as-is, 3/3 auditors | `60b422452bd1e306e19c9168.json` `f23f4080715b` |
| sc_85a7f8cca6bc | 50539000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 3 | as-is, 3/3 auditors | `fb56b098f504e727263bea4b.json` `1e7525239b5c` |
| sc_9567d0d3f33d | 50552000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 7 | as-is, 3/3 auditors | `b51858fca8b50620e192965b.json` `e0a08db3416a` |
| sc_581c7de10fc6 | 50553000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 2 | as-is, 3/3 auditors | `93b79fa0a29863cbf173a7ce.json` `6e63e7d76ed7` |
| sc_8bb74bda6756 | 50566000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 7 | as-is, 3/3 auditors | `52b3fb6fd45c2e2668c6ed50.json` `4431d5434342` |
| sc_80472dfa8819 | 50567000 | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 2 | as-is, 3/3 auditors | `ade6d40a879bd8f065b14cf8.json` `f6f1a2e70a11` |
| sc_752e2bd54aad | 50757000 | DEFECTIVE_X (full) | yes | 9 | fixed + 2 verifiers | `4e984ba78f44ea3f8edae474.json` `7e7b234267ac` |
| sc_777c64c88967 | 50757000 | GOOD_V (full) | yes | 9 | as-is, 3/3 auditors | `de3742c4ce99808e021c2cdb.json` `4145264a5485` |
| sc_36e5d6998b9b | ysherman | FULL_PRESUMED_NO_STAGE_HINT (full) | no | 12 | fixed + 2 verifiers | `0c33a49d0a4aafc114c84cc1.json` `34359a9a3c93` |
