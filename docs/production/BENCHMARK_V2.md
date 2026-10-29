# Benchmark V2: broadened bend benchmark manifest

Generated 2026-10-17T17:43:32Z from metadata only. Machine-readable manifest: `docs/production/runlog/benchmark_v2/BENCHMARK_V2.json` (sha256 `c680e4d8bfb61564…`). Inputs: `merged_inventory.json` (local files plus Sherman-domain email attachments, deduplicated by sha256), `label_prep/selection.json`, and the 58-scan AI labeling and verification results. No scan geometry was read to build it.

Owner authorization (2026-10-17): broaden beyond Gold11 with all Sherman scans and CADs on this computer and the shermanep.com email attachments, and create AI-vision gold where no gold and no CAD exist. HOLDOUT10, sealed truth, opaque N001-N014, EHO4, authorization/one-shot artifacts and held-out manifests remain excluded.

## Headline

- **106 benchmark scans across 81 parts** (73 with a SKU, 8 without one). Gold11 had 11 scans. 64 of these parts have CAD on disk, but only 25 have a STEP file that the V10 recipe compiler can read. The rest are SLDPRT or x_t and need a STEP export.
- **Every part that has a scan on disk is CONSUMED, and FRESH has 0 scans on disk.** All 82 FRESH parts are CAD/drawing-only on disk. 18 FRESH scan files (in 9 parts) are still in email: they were not downloaded because of the MCP attachment size limit or SharePoint authentication. In total, 100 email scan attachments are not on disk. Until they are downloaded, nothing here can measure accuracy on unseen parts.
- **Only 13 scans have human-verified gold lines.** 42 have earlier Codex (AI) labels, 49 have AI gold accepted in this run, 2 have only a CAD recipe, and 0 have no truth at all. Gold11 itself is 9 human label sets plus 2 Codex label sets (38172000 GOOD_V and DEFECTIVE_X).
- **AI labeling: 49 of 49 scans accepted.** 43 of them carry 221 verified bend lines, and 6 are zero-bend controls (rolled shells, one flat blank and one partial scan with no formed fold). On 2026-10-17 48 scans were re-run and re-verified with renders under the strict rule (a line lands only on a real formed fold, parallel to it, covering at least ~80% of its visible length); every verifier in this manifest saw the line overlays, and 0 accepted labels lack images. 0 scans still need review: the owner delegated the final verdict on the last 11 to the AI review on 2026-10-17 (9 AI_GOLD_ACCEPTED, 2 REJECTED_NOT_A_SCAN; see below). 42 accepted scans have an `ai_bend_lines_v1` studio label file; 7 exist only as `.stl` or `.obj` and have none yet.
- **The Gold11 frame must be recovered before scoring.** 9 of the 11 Gold11 label sets were drawn in `reference_mesh_raw`. Plan §10 amendment 6(a) requires each case's raw-to-reference transform (residual ≤ 0.01 mm), with scoring in the raw-scan frame. This manifest marks those 9 as `REQUIRED_PENDING`. The 2 `scan_raw` cases need only a sha256 check of the labelled file.
- **A green Gold11 is expected, not evidence.** The engine was tuned on Gold11, so it should come back green there. That shows the tuning reproduces; it says nothing about unseen parts.

## Splits

A split is assigned per **part** (SKU), never per scan, so a new scan of a tuned part cannot pass as unseen.

| Split | Definition | Parts | Scans on disk |
|---|---|---|---|
| CONSUMED | Any file of the part is Gold11, Gold37, physical15, used in other development or F1 tuning (`other-consumed`), or of unknown provenance | 95 | 106 |
| FRESH | Every known file of the part (local and email) is `prior_use=fresh`, and the part is not in the Eyal-50 delivery | 90 (82 with a SKU; 71 with CAD, 6 with STEP) | 0 (18 pending download) |
| HELD_EYAL50_OVERLAP | Fresh files, but the part is also in the Eyal-50 delivery (locked-holdout split unread) | 1 | 0 |
| EXCLUDED_EYAL50_QUARANTINE | Only Eyal-50 delivery files (20 dev / 30 locked holdout; membership not read) | 51 | 0 |

3 parts are `CONSUMED_UNVERIFIED`: 17139000, 28289002, 41880000. Every file of these parts on disk is fresh. They count as consumed only because email copies that are not on disk were matched to local files by filename, and for 28289002 the only match is a generic drawing name (`שרטוט החלק.PDF`). They stay CONSUMED (conservative). The owner may verify them and move them to FRESH before the FRESH run, never after it.

One scan is tagged `FRESH_SCAN_OF_CONSUMED_PART`: `17139000_01.ply` (`sc_90257b128d80`, part 17139000, one of the CONSUMED_UNVERIFIED parts). It has no recorded use. It is reported separately from FRESH and is also evaluated once. `37126000_D.ply` from email is not in this tier, because an identical-geometry STL of it in `source-assets` is consumed.

### FRESH rule (binding)

1. FRESH is evaluated **once**, at the end of the accuracy loop, with one frozen engine config hash recorded in the ledger before the run.
2. FRESH results never feed back into tuning, thresholds, recipe or label edits, or case selection. Nobody looks at FRESH scans' engine output before that run.
3. If FRESH is opened early, or the engine changes after the FRESH run, the affected parts become CONSUMED, and the report says so.
4. FRESH labels (AI or human) may be drawn before the run, but only from geometry, never from engine output (the Bend Label Studio shows no predictions).
5. CONSUMED results are engineering-ledger development evidence only and are never quoted as accuracy on unseen parts.

## Truth tiers

| Tier | Meaning | Scans (primary) |
|---|---|---|
| GOLD_LINES_HUMAN | `authoritative_user_verified` line references, verified by the owner | 13 |
| AI_GOLD_ACCEPTED | This run's AI vision labels. An independent verifier agent inspected the renders, every line landed, and no visible bend was missed. **Adjudicated by agents, not humans.** | 49 |
| GOLD_LINES_AI_LEGACY | Earlier Codex manual-multiview labels (`codex_manual_multiview_*`). All 38 files in `data/bend_corpus/annotations/human_bend_lines_v1/` carry annotator `Codex multiview gold import`, so none are human studio labels | 42 |
| CAD_RECIPE | The part has a STEP file, so a V10 / DEV_AUTO_CONFIRMED recipe can be compiled. This gives design intent (count, nominal angle), not the scanned line position, and needs a stage decision for partial scans. Not compiled yet | 2 |
| NONE | No usable truth | 0 |

Each scan in the JSON lists all its `truth_sources`. Its `primary_truth` is the first that applies, in this order: human, AI-accepted, AI-legacy, CAD recipe. AI labels that did not pass are recorded as `AI_LABEL_PENDING_REVIEW (not truth)`.

## Summary counts

| Count | Value |
|---|---|
| Unique SKUs with usable files (scan, CAD or drawing) | 158 |
| Parts with a benchmark scan on disk | 81 |
| Benchmark scans (deduplicated by sha256; format alternates, synthetic fixtures and redacted copies removed) | 106 |
| Parts with a scan and CAD | 64 |
| Parts with a scan and STEP (recipe-compilable) | 25 |
| Parts with CAD of any kind (incl. CAD-only) | 142 |
| Staged or incomplete scans (PARTIAL_SCAN_1/2, PRELIM, STAGE0_BLANK) | 16 |
| Scans with human gold lines | 13 |
| Scans with AI gold accepted (this run) | 49 (43 with 221 lines; 6 zero-bend) |
| Scans needing human review (this run) | 0 |
| FRESH scans on disk / pending in email | 0 / 18 |
| Email attachments not on disk (all kinds / scans) | 197 / 100 |
| Scans with partial AI gold (label_complete=false) | 0 |
| Removed rows: format alternates / synthetic fixtures / not a scan (review11 + synthetic LIDAR demos) / redacted opaque copy | 9 / 3 / 9 / 1 |
| Sealed rows dropped upstream (counted, never opened): scan / CAD / drawing | 42 / 16 / 1 |
| Eyal-50 quarantined files | 151 |

## AI gold accepted (this run)

| Scan | Part | File | Role | Lines | Verifier finding | lines.json sha256 | Studio label |
|---|---|---|---|---|---|---|---|
| `sc_d45186653eed` | 11979003 | 11979003_C.stl | bend_lines | 9 | r1 reverify: all 9 lines land for both verifiers (>=80% coverage), no missed bends | `f8cc4ea01412` | none: STL/OBJ only, needs a PLY export |
| `sc_03395097cd96` | 15651004 | 15651004_01.ply | bend_lines | 3 | r1 reverify: all 3 lines land for both verifiers (>=80% coverage), no missed bends | `7c64387dc25f` | `1c62d21a1fceedefd236259a` |
| `sc_927a2fa13067` | 15651004 | 15651004_01_3RD_SCAN.stl | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `bb0adfacab0f` | none: STL/OBJ only, needs a PLY export |
| `sc_90257b128d80` | 17139000 | 17139000_01.ply | bend_lines | 5 | r1 reverify: all 5 lines land for both verifiers (>=80% coverage), no missed bends | `a1d6bcc8baab` | `af451fb41bee7eec19a26302` |
| `sc_6eba02723d4b` | 23553000 | 23553000_B_FULL_SCAN 1.ply | bend_lines | 16 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 16 lines, 16 real bends, all land for both blind verifiers + judge; 0 missed | `218ce0b4292a` | `71f5e686304eac1244183772` |
| `sc_5a1ecd7a4db3` | 23553000 | 23553000_B_SCAN_1.ply | bend_lines | 4 | r1 reverify: all 4 lines land for both verifiers (>=80% coverage), no missed bends | `5db194916162` | `be3bccecf0023ae1e1191702` |
| `sc_186463746daa` | 23553000 | 23553000_B_SCAN_2.ply | bend_lines | 9 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 9 lines, 9 real bends, all land for both blind verifiers + judge; 0 missed | `8e74f90062f1` | `3512f1a354ccad5b70a98083` |
| `sc_38091ac91d9d` | 28870000 | 28870000_A_FULL_SCAN.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `02859a7c36fd` | `8da4836b46c1392161e5fa48` |
| `sc_828bc34d9c8b` | 34499000 | 34499000_B.ply | bend_lines | 3 | r1 reverify: all 3 lines land for both verifiers (>=80% coverage), no missed bends | `84d3deb405a0` | `f795a2da509fda73a25e2556` |
| `sc_822f083e3dfc` | 37126000 | 37126000_D.ply | bend_lines | 2 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 2 lines, 2 real bends, all land for both blind verifiers + judge; 0 missed | `971122d5006b` | `c769e053715d973985c0c85c` |
| `sc_db1839480738` | 38169000 | 38169000_04.ply | bend_lines | 5 | r2 reverify: all 5 lines land for both verifiers (>=80% coverage), no missed bends | `7c6f3c295306` | `6a4c42d3a3e11f0eed7646a6` |
| `sc_6101a12648a9` | 38965002 | 38965002_A_V.ply | bend_lines | 3 | r1 reverify: all 3 lines land for both verifiers (>=80% coverage), no missed bends | `53b8f63c7c0a` | `b19061f3e4a792fe31cdd82b` |
| `sc_c60c74f5ae01` | 38965002 | 38965002_A_X.ply | bend_lines | 3 | r1 reverify: all 3 lines land for both verifiers (>=80% coverage), no missed bends | `b3b5caf0c4b3` | `197d7a88dbe776749a98de84` |
| `sc_eaa89c6d1f1f` | 40500000 | 40500000_0_SCAN1 1.ply | bend_lines | 9 | r1 reverify: all 9 lines land for both verifiers (>=80% coverage), no missed bends | `e630ac769cf6` | `a0bf9352332d87f9c8dfc8e0` |
| `sc_2233372b8df7` | 42288005 | 42288005_01_FULL_SCAN.ply | bend_lines | 8 | r3 owner relabel: 6 r2 lines unchanged + 2 joggle bends (CAD C5/C6) added; all 8 land (fold support 1.0, >=80% coverage) on the r3 renders and own perpendicular sections; no missed bends | `8bd3cb945362` | `ae525e4954effa70d87ba8eb` |
| `sc_361fc1049e39` | 44211000 | 44211000_A.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `45d55d371a36` | `fbe906a2187e95605c36ef39` |
| `sc_1268c72dcc30` | 45796000 | 45796000_A_FULL_SCAN.ply | bend_lines | 8 | r1 reverify: all 8 lines land for both verifiers (>=80% coverage), no missed bends | `05fd0190656a` | `60dc0a2751f6320eb6ddf3d5` |
| `sc_c623d2e45bff` | 45796000 | 45796000_A_SCAN_1.ply | bend_lines | 5 | r1 reverify: all 5 lines land for both verifiers (>=80% coverage), no missed bends | `489447cc0451` | `922ee6a9ac06f5df20ef4dd8` |
| `sc_9ef2cf4e52cb` | 45796000 | 45796000_A_SCAN_2.ply | bend_lines | 4 | r2 reverify: all 4 lines land for both verifiers (>=80% coverage), no missed bends | `8882a2a39464` | `b1038070f28cb7792f9d6090` |
| `sc_8935b8a8e8aa` | 47247004 | 47247004_03.ply | bend_lines | 6 | r1 reverify: all 6 lines land for both verifiers (>=80% coverage), no missed bends | `b370488039e4` | `97d359a69811fb0f8e846d43` |
| `sc_9bd69bc6535c` | 47251002 | 47251002_02_V.ply | bend_lines | 12 | r3 re-review: 11 r2 lines unchanged + B12 window tab added; all 12 land (fold support 1.0, >=80% coverage) on the r3 renders and own perpendicular sections; no missed bends | `5743f6d8d4f1` | `e1e41c2d133c34b6b5886eb5` |
| `sc_1059a9192687` | 47251002 | 47251002_02_X.ply | bend_lines | 12 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 12 lines, 12 real bends, all land for both blind verifiers + judge; 0 missed | `bb65135aad45` | `393c99fadda689b9b49b561b` |
| `sc_657b87ceeba6` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_154559_545707.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `41d6f3a83fe3` | `7590fd22d5d86158a6baf337` |
| `sc_573c999ef6ea` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_154744_233337.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `8c42cd66a9ba` | `5cbf6ce5c255984ca17a4e4b` |
| `sc_123f84b5afcb` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_155246_866937.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `2d35c9798732` | `a11df0bda44de7b856734cd9` |
| `sc_c2ff6d5b6c2f` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_155400_715790.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `b36636a89369` | `7f7dd0eace422ac665b7df31` |
| `sc_9018b86ccc9a` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_165235_383038.ply | bend_lines | 1 | B1 lands on the single ~90 deg flange fold; no missed bends | `d1eed8adfbec` | `c1e4bcd502f8b3993cbe7722` |
| `sc_a0b6add3f6e6` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_211408_450619.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `a8f8c79109c4` | `387a26b4a83c91527b86793d` |
| `sc_e632352b37f2` | 47266000 | 47266000_F_LIDAR_DEMO_20260209_211721_100645.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `9e39e21e4df6` | `c9ac548e818e85512c838483` |
| `sc_9b9fd762f361` | 47266000 | 47266000_F.stl | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `a9a0891d0e40` | none: STL/OBJ only, needs a PLY export |
| `sc_3b3e8c9f04e3` | 47555020 | 47555020_1.ply | bend_lines | 1 | r1 reverify: all 1 lines land for both verifiers (>=80% coverage), no missed bends | `b79c984fee72` | `b238e9f7821479679797b5a9` |
| `sc_ff1667a2a80a` | 48811000 | 48811000_A.ply | bend_lines | 4 | r1 reverify: all 4 lines land for both verifiers (>=80% coverage), no missed bends | `ce0f0e26124d` | `6069f1090f154ea8afffc236` |
| `sc_2d3e4a623a31` | 48963008 | 48963008_B.ply | bend_lines | 3 | r1 reverify: all 3 lines land for both verifiers (>=80% coverage), no missed bends | `2c3963102a54` | `febe9a3931a3027bf3799bd2` |
| `sc_6e5d5a2f60f5` | 48991006 | 48991006_A 4.ply | bend_lines | 5 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 5 lines, 5 real bends, all land for both blind verifiers + judge; 0 missed | `0c8e76fd6e64` | `f1840bb889ab736f1c9efa55` |
| `sc_8379cebef9e3` | 49001000 | 49001000_A-PRELIM.obj | bend_lines | 5 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 5 lines, 5 real bends, all land for both blind verifiers + judge; 0 missed | `e361bc795dc3` | none: STL/OBJ only, needs a PLY export |
| `sc_e080ae094f35` | 49204020 | 49204020_04.ply | bend_lines | 12 | 12 bends all land, angles 89.6-89.9 deg, coverage 1.0 | `0d2ada9a1911` | `99622fd56079834417a0605c` |
| `sc_9c84cc623d9b` | 50535000 | 50535000_04.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `8595524d7844` | `4528c30cfa5d165d1914cd69` |
| `sc_ce3de3af51b6` | 50537000 | 50537000_04.ply | bend_lines | 2 | r2 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `7aaa39769236` | `60b422452bd1e306e19c9168` |
| `sc_85a7f8cca6bc` | 50539000 | 50539000_05.ply | bend_lines | 3 | 3 bends land (B1/B2 collinear split by cutout); pairs.json rewritten to list form | `da1284e8c02d` | `fb56b098f504e727263bea4b` |
| `sc_9567d0d3f33d` | 50552000 | 50552000_04.ply | bend_lines | 7 | tray, 7 bends all land; lines rerun 2026-10-17 on the on-record pairs.json, so pairs/lines hashes now match | `824367b52a4d` | `b51858fca8b50620e192965b` |
| `sc_581c7de10fc6` | 50553000 | 50553000_04.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `e05a617fcd39` | `93b79fa0a29863cbf173a7ce` |
| `sc_8bb74bda6756` | 50566000 | 50566000_04 1.ply | bend_lines | 7 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 7 lines, 7 real bends, all land for both blind verifiers + judge; 0 missed | `af57ebfbee28` | `52b3fb6fd45c2e2668c6ed50` |
| `sc_80472dfa8819` | 50567000 | 50567000_05.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `171728c24371` | `ade6d40a879bd8f065b14cf8` |
| `sc_777c64c88967` | 50757000 | 50757000_02_V.ply | bend_lines | 9 | r1 reverify: all 9 lines land for both verifiers (>=80% coverage), no missed bends | `87efaa585ebe` | `de3742c4ce99808e021c2cdb` |
| `sc_752e2bd54aad` | 50757000 | 50757000_02_X.ply | bend_lines | 9 | 9 bends land; B2 covers ~72% of its fold (scan/patch support ends) | `ee105792b8f3` | `4e984ba78f44ea3f8edae474` |
| `sc_9b930fe58a1b` | UNKNOWN (98072c85/scan.ply) | scan.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `1d7bb707add3` | `d7c45836673fe7a997185af1` |
| `sc_9c890e76934f` | UNKNOWN (02_CAD_QC_3D/part 2.stl) | part 2.stl | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `22bc495bffd4` | none: STL/OBJ only, needs a PLY export |
| `sc_a5e534e6097c` | UNKNOWN (bend_10cdda62/scan.ply) | scan.ply | bend_lines | 5 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 5 lines, 5 real bends, all land for both blind verifiers + judge; 0 missed | `a0e901708970` | `3b3f853405102c379a2672cf` |
| `sc_abe194f04ade` | UNKNOWN (02_CAD_QC_3D/part 1.ply) | part 1.ply | bend_lines | 2 | r1 reverify: all 2 lines land for both verifiers (>=80% coverage), no missed bends | `ddff5b69b0e5` | `89521be3f0aa92b53ff65fe0` |
| `sc_f362d8de507c` | UNKNOWN (c2a47c72/scan.ply) | scan.ply | bend_lines | 5 | r3 owner-delegated review11 final verdict ACCEPT (AI_GOLD_ACCEPTED): 5 lines, 5 real bends, all land for both blind verifiers + judge; 0 missed | `a2eca00d0201` | `f3b6c986ccb0556ebb674843` |
| `sc_f99d883f2754` | 11979003 | BEFORE BENDING  11979003_C.ply | zero_bend_control | 0 | rolled shell or flat blank, no formed bends; pairs.json now zero-bend form and lines.json no_bends_present=true (2026-10-17) | `1fe3d915c069` | `9fb07af82dbf6c097cd621a1` |
| `sc_3bde9996f14d` | 28870000 | 28870000_A_SCAN_1.ply | zero_bend_control | 0 | r2 reverify: zero-bend label, both verifiers see no formed bends and none missed | `26f7d236c6ae` | `a92f04f2ef011fc53a09675c` |
| `sc_b4719db3dd69` | 47479000 | 47479000_C.ply | zero_bend_control | 0 | rolled shell or flat blank, no formed bends; pairs.json now zero-bend form and lines.json no_bends_present=true (2026-10-17) | `a5456f0a667b` | `af36cde5bd3c635214303345` |
| `sc_2e584e9d1ffe` | UNKNOWN (4c28ab8e/scan.ply) | scan.ply | zero_bend_control | 0 | rolled shell or flat blank, no formed bends; pairs.json now zero-bend form and lines.json no_bends_present=true (2026-10-17) | `460b438583f2` | `0ceb0e34e9e03b205a2bf17e` |
| `sc_7a253806b10b` | UNKNOWN (8d713421/scan.stl) | scan.stl | zero_bend_control | 0 | rolled shell or flat blank, no formed bends; pairs.json now zero-bend form and lines.json no_bends_present=true (2026-10-17) | `6a7ce1d02a13` | none: STL/OBJ only, needs a PLY export |
| `sc_f098dbceaa6d` | UNKNOWN (7800229b/scan.stl) | scan.stl | zero_bend_control | 0 | rolled shell or flat blank, no formed bends; pairs.json now zero-bend form and lines.json no_bends_present=true (2026-10-17) | `3512ed3a2cd6` | none: STL/OBJ only, needs a PLY export |

Caveats before these go into scoring:
- Zero-bend controls now carry `no_bends_present=true` in `pairs.json` and `lines.json` (fixed 2026-10-17). Use them only as false-positive controls, not in line-F1 denominators.
- **Possible CAD reference meshes, kept out of scan-accuracy scoring:** `sc_2e584e9d1ffe`, `sc_7a253806b10b`, `sc_f098dbceaa6d`. These rolled shells come from `uploads/*/reference.*` (for `sc_2e584e9d1ffe`, `scan.ply` and `reference.ply` are byte-identical), so they may be CAD meshes rather than physical scans.
- `sc_3bde9996f14d` (28870000_A_SCAN_1) is accepted as zero-bend. Its sibling full scan carries two shallow creases (about 24 and 18 deg), so check that this partial scan really shows no formed fold.
- `sc_9567d0d3f33d`: `lines` was rerun on the on-record `{a,b}` pairs.json; the pairs and lines hashes now match, and the 7 bends are unchanged.
- `sc_752e2bd54aad`: B2 covers about 72% of its fold. That is below the strict ~80% rule used in the 2026-10-17 re-verification; it was accepted earlier and was not re-verified.
- **Synthetic LIDAR demo files removed (2026-10-17):** 9 `live_scans/*_LIDAR_DEMO_*.ply` files are resamples of `test_files/<part>/scan.stl` made by the legacy `/api/live-scan/demo/load` endpoint, not physical scans. They are `EXCLUDED_NOT_A_SCAN` and out of every count above; 7 of them had been accepted as AI gold. Their label files stay on disk and scorers must skip them. Evidence per scan: `docs/production/runlog/benchmark_v2/manifest_build/synthetic_demo_exclusions_2026-10-17.json`. Gold-set hits: none.
- `sc_9bd69bc6535c` relabelled in round 3 (11 -> 12 bends); see `docs/production/runlog/benchmark_v2/manifest_build/relabel_r3_2026-10-17.json`.
- `sc_2233372b8df7` relabelled in round 3 (6 -> 8 bends); see `docs/production/runlog/benchmark_v2/manifest_build/relabel_42288005_2026-10-17.json`.
- No studio label yet (STL or OBJ only; the studio indexes `.ply`): `sc_d45186653eed` (9 lines), `sc_927a2fa13067` (2 lines), `sc_9b9fd762f361` (1 lines), `sc_8379cebef9e3` (5 lines), `sc_7a253806b10b` (0 lines), `sc_9c890e76934f` (2 lines), `sc_f098dbceaa6d` (0 lines). Their gold is the verified `lines.json`; a PLY export with identical vertices is needed before `write-label`.
- Scans that lived only in `~/Downloads` were copied byte-identically to `data/bend_corpus/incoming/2026-10-17_benchmark_v2_downloads/` so their labels have a stable studio asset id. `sc_361fc1049e39` has two label files, one per byte-identical copy (source-assets and the email copy under `data/`).

## Review11: owner-delegated final verdicts (2026-10-17)

The owner delegated the final verdict on the 11 scans that were left in NEEDS_HUMAN_REVIEW. Each got an investigator, two blind verifiers and a judge, all working from their own multi-angle renders and perpendicular sections, plus a final pass. Verdicts: `manifest_build/review11_verdicts_2026-10-17.json`. These are still AI-adjudicated, not human.

| Scan | Part | File | Verdict | Real bends | Lines | Studio label | Reason |
|---|---|---|---|---|---|---|---|
| `sc_186463746daa` | 23553000 | 23553000_B_SCAN_2.ply | AI_GOLD_ACCEPTED | 9 | 9 | `3512f1a354ccad5b70a98083` | Staged scan: the legs are not yet folded off the plate. 9 formed folds land at 89-99.5%: 2 arm-end folds at each right end, 1 tip fold at each left end (wide sections show the arm flat for >45 mm before the tip), plus B7-B9. Top-skin dropouts and a noise blob are not bends. |
| `sc_6eba02723d4b` | 23553000 | 23553000_B_FULL_SCAN 1.ply | AI_GOLD_ACCEPTED | 16 | 16 | `71f5e686304eac1244183772` | 17 flat faces joined in a tree need exactly 16 folds, and all 16 land at 91-98.5% coverage, apex within +-0.2 mm. The crease detector fires only along the 16 lines; the plate bow is smooth (largest local jump 0.54 deg). |
| `sc_822f083e3dfc` | 37126000 | 37126000_D.ply | AI_GOLD_ACCEPTED | 2 | 2 | `c769e053715d973985c0c85c` | Z-joggle strip: two opposite ~41 deg folds with outer radius ~6.5 mm. Both lines lie on the scanned skin within 0.05 mm, are parallel within 0.1 deg and cover 94% wall to wall. The countersink, side walls and square ends are not bends. |
| `sc_1059a9192687` | 47251002 | 47251002_02_X.ply | AI_GOLD_ACCEPTED | 12 | 12 | `393c99fadda689b9b49b561b` | 12 formed folds all land (88-99% coverage). B12 is a 15 deg window tab cut free on 3 sides with a localized ~8-10 mm radius (20-50 mm radii rejected), and the sibling V scan has the same tab, so it is a formed bend, not a gentle kink. The 135 deg wall corners are butt seams or V-relief notches, not folds. |
| `sc_6e5d5a2f60f5` | 48991006 | 48991006_A 4.ply | AI_GOLD_ACCEPTED | 5 | 5 | `f1840bb889ab736f1c9efa55` | 5 folds land at 96-100%. The back-half wall/floor joints (old 3/5, 4/6) are slits: the walls only butt the floor, with cut faces at 27/27 stations and no arc. B4 is a 17 deg crease formed over a 3.5-6 mm width, not a rolled kink. |
| `sc_8379cebef9e3` | 49001000 | 49001000_A-PRELIM.obj | AI_GOLD_ACCEPTED | 5 | 5 | none: LabelWriteError: the studio inventory indexes .ply files only; label a PLY asset (scan exists only as .obj) | 5 folds land at 94-99%. The B5 extent override is justified. The plate bow beside B5 (radius ~90 mm) and the B3 leg drift are gentle deformation, not folds. Physical scan with an open mesh and scan gaps. |
| `sc_8bb74bda6756` | 50566000 | 50566000_04 1.ply | AI_GOLD_ACCEPTED | 7 | 7 | `52b3fb6fd45c2e2668c6ed50` | Welded box-corner bracket. 7 formed folds share one tooling profile (Ro 3.5-3.7 mm) and land at 87-99%. B5 is a fold, not a weld. The miter weld ridges, the 2 deg flange-B warp and the marker rings are not bends. |
| `sc_a5e534e6097c` | UNKNOWN | scan.ply | AI_GOLD_ACCEPTED | 5 | 5 | `3b3f853405102c379a2672cf` | Tray with lip, 2 tall flanges and 2 tabs. 5 folds land at 92-99%. The base dip by B5 and the B3 flange tail are smooth bend deformation. A different physical part/capture from sc_f362d8de507c (B3 70 vs 82 deg incl, clouds 0.6 mm median apart), not a duplicate. |
| `sc_f362d8de507c` | UNKNOWN | scan.ply | AI_GOLD_ACCEPTED | 5 | 5 | `f3b6c986ccb0556ebb674843` | Trapezoidal tray. 5 press-brake folds land at 95-99%. The ~13 deg roll after B3 is gentle (radius at least 57 mm, no step) and belongs to B3. The base dip by B5 is bend deformation. |
| `sc_43e1dbdce7e3` | 11979003 | 11979003_C_LIDAR_DEMO_20260209_155351_508814.ply | REJECTED_NOT_A_SCAN | 6 | 6 | none: rejected: not a physical scan; no label written | OVERRIDES the judge's ACCEPT. The file is a synthetic Open3D resample of sc_d45186653eed (test_files/11979003_C/scan.stl), made by _write_lidar_demo_scan: 'Created by Open3D', xyz only, 13191 points = 14000 default x 0.94, and the mtime equals the name stamp. Residual to the source mesh is median 0.053 / p95 0.156 mm, exactly N(0, 0.08). All 6 lines coincide with sc_d45186653eed's accepted lines within 0.02-0.14 mm and 0.01-0.15 deg. The geometry label is correct (6/6) but this is not a physical scan. |
| `sc_d5d7f0cfa36c` | 11979003 | 11979003_C_LIDAR_DEMO_20260209_154630_251479.ply | REJECTED_NOT_A_SCAN | 6 | 6 | none: rejected: not a physical scan; no label written | Synthetic Open3D demo resample of sc_d45186653eed: 11293 points = 12000 x 0.94, and the residual to the source mesh is median 0.053 / p95 0.156 mm, exactly N(0, 0.08). The 6/6 geometry label is correct, but counting it would double-count sc_d45186653eed with idealised noise. |

- Flag (`sc_657b87ceeba6`, `sc_573c999ef6ea`, `sc_123f84b5afcb`, `sc_c2ff6d5b6c2f`, `sc_9018b86ccc9a`, `sc_a0b6add3f6e6`, `sc_e632352b37f2`): These 7 already-accepted 47266000_F_LIDAR_DEMO files have the same synthetic provenance as the two rejected 11979003 demos: an Open3D header with xyz only, 11284-13196 points (12000/14000 x 0.94), and an mtime equal to the name stamp. They come from _write_lidar_demo_scan on test_files/47266000_F/scan.stl. Recommendation: mark them REJECTED_NOT_A_SCAN or exclude them from physical scan-accuracy scoring. Not changed here: they are outside the 11. **Resolved 2026-10-17: all excluded as EXCLUDED_NOT_A_SCAN via `docs/production/runlog/benchmark_v2/manifest_build/synthetic_demo_exclusions_2026-10-17.json`.**
- Flag (`sc_9bd69bc6535c`): Sibling 47251002_02_V. Its 11-bend accepted label probably misses the 15 deg window tab that is B12 on sc_1059a9192687; the judge found the same 15.1-15.3 deg, 28.8 mm hinge candidate in its prep. It should be re-reviewed. Not changed here.
- Flag (`sc_43e1dbdce7e3`): Judge accepted it as a physical LiDAR capture. The final verdict overrides that on provenance evidence; see reason.
- Flag (`sc_43e1dbdce7e3`): B3 stops short of a ~13 mm collinear stub past the base slot (coverage 0.92 of the visible fold). Moot after the rejection.

## NEEDS_HUMAN_REVIEW: owner check in Bend Label Studio

None. Every scan that was waiting for review now has a review11 verdict (above).

## Parts with scans on disk

The Truth column counts scans by primary truth. Pending = scan attachments still in email for that part.

| Part | Split | Scans | Staged | Stage hints | CAD | Recipe | Flat | Drawings | Truth | Pending |
|---|---|---|---|---|---|---|---|---|---|---|
| 10839000 | CONSUMED (Gold11) | 3 | 2 | FULL, PARTIAL_SCAN_1, PARTIAL_SCAN_2 | stp | STEP (not compiled) | 0 | 3 | 3 human | 3 |
| 11979003 | CONSUMED | 2 | 1 | STAGE0_BLANK | stl,stp | STEP (not compiled) | 1 | 2 | 2 AI-accepted | 4 |
| 15651004 | CONSUMED (Gold37) | 4 | 0 | REPEAT_3RD_SCAN, REPEAT_SECOND_SCAN | stp | STEP (not compiled) | 0 | 0 | 2 AI-accepted, 1 CAD recipe, 1 human | 6 |
| 17139000 | CONSUMED; 1 fresh scan | 1 | 0 | - | stp | STEP (not compiled) | 0 | 0 | 1 AI-accepted | 1 |
| 23553000 | CONSUMED | 3 | 2 | FULL, PARTIAL_SCAN_1, PARTIAL_SCAN_2 | stp | STEP (not compiled) | 0 | 0 | 3 AI-accepted | 3 |
| 24694000 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 28870000 | CONSUMED | 2 | 1 | FULL, PARTIAL_SCAN_1 | stp | STEP (not compiled) | 0 | 1 | 2 AI-accepted | 2 |
| 34499000 | CONSUMED | 1 | 0 | - | step | STEP (not compiled) | 0 | 0 | 1 AI-accepted | - |
| 35705010 | CONSUMED (Gold37) | 2 | 0 | DEFECTIVE_X, GOOD_V | x_t | native only, needs STEP | 0 | 1 | 2 AI-legacy | 1 |
| 37126000 | CONSUMED | 1 | 0 | - | sldprt,stp | STEP (not compiled) | 0 | 1 | 1 AI-accepted | - |
| 37361005 | CONSUMED (Gold11) | 1 | 0 | - | stp | STEP (not compiled) | 0 | 3 | 1 human | 3 |
| 38169000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 38172000 | CONSUMED (Gold11) | 4 | 2 | DEFECTIVE_X, GOOD_V, PARTIAL_SCAN_1, PARTIAL_SCAN_2 | stp,x_t | STEP (not compiled) | 0 | 2 | 2 AI-legacy, 2 human | 4 |
| 38173000 | CONSUMED (Gold37) | 2 | 0 | DEFECTIVE_X, GOOD_V | sldprt | native only, needs STEP | 0 | 1 | 2 AI-legacy | 1 |
| 38965002 | CONSUMED | 2 | 0 | DEFECTIVE_X, GOOD_V | sldprt | native only, needs STEP | 0 | 1 | 2 AI-accepted | - |
| 40500000 | CONSUMED (Gold11) | 2 | 2 | PARTIAL_SCAN_1, PARTIAL_SCAN_2 | stp | STEP (not compiled) | 0 | 0 | 1 AI-accepted, 1 human | 2 |
| 41017000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 42288005 | CONSUMED | 1 | 0 | FULL | stp | STEP (not compiled) | 0 | 0 | 1 AI-accepted | 3 |
| 42725000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 43895000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 43920000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 43976000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 44211000 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 1 | 1 AI-accepted | - |
| 45796000 | CONSUMED (physical15) | 3 | 2 | FULL, PARTIAL_SCAN_1, PARTIAL_SCAN_2 | stp | STEP (not compiled) | 0 | 0 | 3 AI-accepted | 5 |
| 46414000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 47247004 | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | 1 |
| 47251002 | CONSUMED | 2 | 0 | DEFECTIVE_X, GOOD_V | sldprt | native only, needs STEP | 0 | 1 | 2 AI-accepted | - |
| 47266000 | CONSUMED | 1 | 0 | - | stl,stp | STEP (not compiled) | 0 | 1 | 1 AI-accepted | 3 |
| 47479000 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 2 | 1 AI-accepted | 3 |
| 47555020 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 47632001 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 47959001 | CONSUMED (Gold37) | 2 | 2 | PARTIAL_SCAN_1 | stp | STEP (not compiled) | 0 | 1 | 1 CAD recipe, 1 human | 4 |
| 48333006 | CONSUMED (Gold37) | 1 | 0 | - | x_t | native only, needs STEP | 0 | 1 | 1 AI-legacy | - |
| 48333018 | CONSUMED (Gold37) | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-legacy | - |
| 48340007 | CONSUMED (Gold37) | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-legacy | - |
| 48340008 | CONSUMED (Gold37) | 1 | 0 | - | x_t | native only, needs STEP | 0 | 1 | 1 AI-legacy | - |
| 48412000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 48565005 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 48565022 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 48811000 | CONSUMED (physical15) | 1 | 0 | - | stp | STEP (not compiled) | 0 | 0 | 1 AI-accepted | 2 |
| 48932000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 48938000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 48963008 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 1 | 1 AI-accepted | 2 |
| 48991006 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 2 | 1 AI-accepted | 3 |
| 49001000 | CONSUMED (Gold11) | 2 | 2 | PRELIM | stp | STEP (not compiled) | 0 | 2 | 1 AI-accepted, 1 human | 3 |
| 49024000 | CONSUMED (Gold11) | 1 | 0 | - | stp | STEP (not compiled) | 0 | 1 | 1 human | 3 |
| 49125000 | CONSUMED (Gold37) | 1 | 0 | - | x_b,x_t,xdl | native only, needs STEP | 0 | 1 | 1 human | 3 |
| 49162000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 49163000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 49188000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 49191023 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49191030 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49191032 | CONSUMED (Gold37) | 1 | 0 | - | x_t | native only, needs STEP | 0 | 1 | 1 AI-legacy | - |
| 49191037 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49191038 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49191054 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49191061 | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 AI-legacy | 1 |
| 49201000 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 0 | 1 AI-legacy | - |
| 49204020 | CONSUMED | 1 | 0 | - | stp | STEP (not compiled) | 0 | 0 | 1 AI-accepted | 2 |
| 49227000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 49293001 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 0 | 1 AI-legacy | - |
| 50535000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50537000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50539000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50545000 | CONSUMED (Gold37) | 2 | 0 | DEFECTIVE_X, GOOD_V | x_t | native only, needs STEP | 0 | 1 | 2 AI-legacy | 1 |
| 50546000 | CONSUMED (Gold37) | 2 | 0 | DEFECTIVE_X, GOOD_V | x_t | native only, needs STEP | 0 | 1 | 2 AI-legacy | 1 |
| 50552000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50553000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50556000 | CONSUMED (Gold37) | 2 | 0 | DEFECTIVE_X, GOOD_V | x_t | native only, needs STEP | 0 | 1 | 2 AI-legacy | 1 |
| 50566000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50567000 | CONSUMED | 1 | 0 | - | sldprt | native only, needs STEP | 0 | 1 | 1 AI-accepted | - |
| 50757000 | CONSUMED (physical15) | 2 | 0 | DEFECTIVE_X, GOOD_V | step,x_t | STEP (not compiled) | 0 | 1 | 2 AI-accepted | - |
| ysherman | CONSUMED (Gold37) | 1 | 0 | - | - | none | 0 | 0 | 1 human | - |
| UNKNOWN (4c28ab8e/scan.ply) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (8d713421/scan.stl) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (98072c85/scan.ply) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (02_CAD_QC_3D/part 2.stl) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (bend_10cdda62/scan.ply) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (02_CAD_QC_3D/part 1.ply) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (7800229b/scan.stl) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |
| UNKNOWN (c2a47c72/scan.ply) | CONSUMED | 1 | 0 | - | - | none | 0 | 0 | 1 AI-accepted | - |

## FRESH parts (no scans on disk yet)

82 SKUs: 6 with STEP, 65 with native CAD only, and 11 with no CAD on disk. Even without scans, the FRESH STEP files can test the compile gate (at least 80% of STEPs compile) once at the end. There are also 8 FRESH rows with no SKU (email drawings and images), listed in the JSON.

FRESH parts whose scans are still in email:

| Part | Scan attachments not on disk | Blocker | CAD on disk |
|---|---|---|---|
| 23317000 | 23317000_C 1.obj, 23317000_C 1.ply | MCP_SIZE_LIMIT | native only, needs STEP |
| 38898000 | 38898000_A.ply, 38898000_A.obj, 38898000_A.stl | MCP_SIZE_LIMIT, SHAREPOINT_AUTH | none |
| 38968000 | 38968000_A 1.ply, 38968000_A.stl | SHAREPOINT_AUTH | STEP (not compiled) |
| 46465000 | 46465000_D.ply | MCP_SIZE_LIMIT | none |
| 48101000 | 48101000_C.ply, 48101000_C.stl, 48101000_C.obj | MCP_SIZE_LIMIT, SHAREPOINT_AUTH | none |
| 48389000 | 48389000_02.stl | MCP_SIZE_LIMIT | none |
| 48555000 | 48555000_0102.stl, 48555000_0102.ply | MCP_SIZE_LIMIT | none |
| 48926001 | 48926001_1.ply, 48926001_1.stl | MCP_SIZE_LIMIT | none |
| 48963009 | 48963009_A.ply, 48963009_A.stl | MCP_SIZE_LIMIT, SHAREPOINT_AUTH | none |

All FRESH SKUs: 09061531, 10835000, 12851000, 15658000, 21763000, 22352000, 23317000, 23976000, 26687004, 26687021, 27796002, 27796005, 28272000, 34371016, 36192000, 37071000, 38539000, 38898000, 38968000, 40851000, 42426000, 42821000, 43161000, 43244000, 43780000, 43798000, 44001000, 44556000, 45141000, 45352000, 46045000, 46127000, 46128000, 46181000, 46465000, 46882000, 46957010, 47040000, 47480000, 47553000, 47846000, 47848000, 47949013, 48099000, 48101000, 48333004, 48334000, 48381001, 48384000, 48389000, 48555000, 48758000, 48900035, 48910000, 48926001, 48933000, 48934000, 48963009, 49025000, 49160000, 49186000, 49187000, 49191008, 49191019, 49191026, 49205000, 49207000, 49289000, 49291000, 49292000, 49355000, 49438000, 49461000, 49553000, 49565000, 49566000, 49783000, 50278001, 50482000, 50597000, 5548120, 9353735.

## Limits (read before quoting any number)

- **AI labels are adjudicated by agents, not humans.** AI_GOLD_ACCEPTED means one labeler agent and one independent verifier agent agreed on rendered evidence. No human has reviewed these 49 scans. Report results against them separately from GOLD_LINES_HUMAN.
- Legacy Codex labels (42 scans) are also AI. Despite its `human_bend_lines_v1` directory name, that store has no human labels.
- The first pass had 13 cases where accepted=true sat beside "not verifiable" notes. They were rerun and re-verified with renders on 2026-10-17; the strict gate still applies to every row.
- Missed-bend counts taken from prep renders are estimates. `n/a` means not assessed, not zero.
- Prior use comes from metadata (manifests, docs, hash matches). A file with no recorded use is `fresh` only to the extent those records are complete. Unknown provenance is treated as CONSUMED.
- CAD recipes are listed as available but have not been compiled or reconciled with any scan. Staged scans need a per-bend FORMED/NOT_FORMED truth table (plan §10 amendment 7) before recipe truth applies to them.
- Email intake is incomplete: 100 scan attachments are not on disk. By prior use: other-consumed 48, Gold11 31, fresh 21. Only the fresh ones would add unseen evidence.
- Eyal-50 (151 files) stays quarantined until the owner rules on its 20 dev / 30 locked-holdout split.

## Custody

- This manifest was built from metadata, directory listings and sha256 hashes. No scan geometry was read for it, and nothing from HOLDOUT10, sealed truth, opaque N001-N014, EHO4, one-shot/authorization artifacts or held-out manifests was opened. The upstream inventory dropped those rows by name and manifest membership (42 scans, 16 CAD files, 1 drawing counted only), and paths under `/opaque/` or `/evaluator_only/` are redacted.
- Face/body scans and non-Sherman projects are not in the inventory.
- Regenerate with `.venv/bin/python -I docs/production/runlog/benchmark_v2/manifest_build/build_manifest.py docs/production/runlog/benchmark_v2/manifest_build/verifier_results.tsv`, then run `build_doc.py` in the same directory. They read only `merged_inventory.json`, `label_prep/selection.json`, `label_prep/prep_runlog.json`, the `label_lines/` listings and `verifier_results.tsv` (the per-scan verifier verdicts transcribed from the workflow results).

