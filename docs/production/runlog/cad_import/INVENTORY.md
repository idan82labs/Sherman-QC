# CAD import inventory: SLDPRT files with a matching physical scan

Date: 2026-10-28. Built from file names only. No scan, SLDPRT or STEP file was opened, hashed or sampled.

## Method

- SLDPRT list: `find ~/ShermanQC ~/Downloads ~/Documents -iname '*.sldprt'` (164 paths). There are no SLDASM or SLDDRW files.
- Scan names: `nice -n 15 find ~/ShermanQC ~/Downloads ~/Documents \( -iname '*.ply' -o -iname '*.stl' -o -iname '*.obj' \)` (2305 paths). A name counts as physical-scan evidence only when it sits under `data/bend_corpus/`, `uploads/`, `live_scans/`, `~/ShermanQC/source-assets/` or `~/Downloads/`. These locations are ignored: `output/`, `artifacts/`, `research/`, `docs/`, `cache/`, `.venv/`, `.tools/`, the guide-video folder, `~/Documents/Codex` (face scans), and the Eyal50 `extracted/` STLs. The Eyal50 STLs are CAD meshes, not scans; `benchmark_v2/local_inventory.json` says "Eyal50: CAD-only".
- Scan indexes (released docs, read-only): `docs/production/runlog/benchmark_v3/inventory.json` (`files[kind=scan].part_id`), `benchmark_v2/local_inventory.json` (`files[kind=scan]`) and `zeroshot_2026-10-25/INVENTORY.json` (`scans[].part_id`).
- STEP names: `find … \( -iname '*.step' -o -iname '*.stp' \)` (491 paths, any location). "STEP exists" means a STEP filename with the same base number exists somewhere. The location class is shown in brackets.
- Part number: a leading 7–8 digit base, or an alphanumeric code such as MP01736, plus the next token as the revision. RFQ, PRELIM and SCAN suffixes are stripped. Matching uses the base number. "Rev match" says whether any scan carries the same revision token. Index rows without a revision count as base-only.
- Holdouts were resolved from docs. HOLDOUT10, N001–N014 and N015/N016 come from `benchmark_v2/local_inventory.json` (`sealed_set_identification` plus `files[prior_use=EXCLUDED_SEALED]`). The EHO4 list is the owner's list (30820000_A01 is consumed and allowed). The Eyal50 quarantine comes from `zeroshot_2026-10-25/PLAN.md` OD-4, which ledger entry `frz_8078a8cc0ec4` records as "not used". TEST-B parts come from `SPLIT_FROZEN.json` `test_b` scan ids mapped through `INVENTORY.json`. Holdouts are matched on the base number, which is the conservative choice.

## Counts

| Item | Count |
|---|---|
| SLDPRT paths | 164 |
| Distinct SLDPRT basenames | 152 |
| Distinct parts (base numbers, unparseable files counted one each) | 152 |
| Parts with a matching physical scan (base number) | 104 |
| **Worth converting first** | **84** (81 have no STEP yet, 3 already have one) |
| Held out, convert later | 22 (20 have a scan) |
| &nbsp;&nbsp;held, reason HOLDOUT10 (one part can have several reasons) | 10 |
| &nbsp;&nbsp;held, reason EHO4 (one part can have several reasons) | 8 |
| &nbsp;&nbsp;held, reason N001-N014 (one part can have several reasons) | 8 |
| &nbsp;&nbsp;held, reason Eyal50 quarantine (one part can have several reasons) | 3 |
| &nbsp;&nbsp;held, reason 41880000_D (one part can have several reasons) | 1 |
| No scan found | 43 |
| Unparseable names | 3 |
| Parts with a STEP already somewhere (all classes) | 4 |
| Worth-converting parts with rev match / base only | 82 / 2 |
| Worth-converting parts that are TEST-B (programme-sealed test scans, informational) | 43 |

TEST-B is not on the exclusion list. Converting its CAD does not open its scans. The flag is there because TEST-B scans must not be used for development.

## Worth converting (parts without a STEP come first)

| # | Part number | SLDPRT basename | Location dir | Scan match evidence | Rev match | STEP exists | Flags |
|---|---|---|---|---|---|---|---|
| 1 | 10835000_01 | 10835000_01.SLDPRT | data/bc/inc_email/20260922_1a0c7dcd/ | 10835000_01 (x).ply; 10835000_01.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 2 | 12851000_B | 12851000_B.SLDPRT | data/bc/inc_email/20260906_1a075bef/ | 12851000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 3 | 21763000_E | 21763000_E.SLDPRT | data/bc/inc_email/20260909_1a0851f5/ | 21763000_E (V).ply; 21763000_E (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 4 | 22352000_01 | 22352000_01.SLDPRT | data/bc/inc_email/20260726_19f9ef79/ | 22352000_01.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 5 | 23976000_C | 23976000_C.SLDPRT | data/bc/inc_email/20260901_1a05bdf7/ | 23976000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 6 | 26687004_A01 | 26687004_A01.SLDPRT | data/bc/inc_email/20260813_19ffb148/ | 26687004_A01.ply; idx: benchmark_v3 | yes | no |  |
| 7 | 26687021_A01 | 26687021_A01.SLDPRT | data/bc/inc_email/20260809_19fe69bb/ | 26687021_A01 (V).ply; 26687021_A01 (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 8 | 27796002_D | 27796002_D.SLDPRT | data/bc/inc_email/20261004_1a10734b/ | 27796002_D (V).ply; 27796002_D (x).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 9 | 27796005_C | 27796005_C.SLDPRT | data/bc/inc_email/20261008_1a11aa5b/ | 27796005_C (V).ply; 27796005_C (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 10 | 30820000_A01 | 30820000_A01.SLDPRT | data/bc/incoming/2026-08-25_shaked_new_good_bad_pair/authenticated_source/cad/, ~/Downloads/ | 30820000_A01 (V).ply; 30820000_A01 (X).ply; 30820000_A01.ply; idx: benchmark_v2 | yes | no |  |
| 11 | 34371016_A | 34371016_A.SLDPRT | data/bc/inc_email/20260726_19f9ea5b/ | 34371016_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 12 | 36192000_02 | 36192000_02.SLDPRT | data/bc/inc_email/20261001_1a0f7cb5/ | 36192000_02 (V) 1.ply; 36192000_02 (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 13 | 37071000_A | 37071000_A.SLDPRT | data/bc/inc_email/20260901_1a05b944/ | 37071000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 14 | 38169000_04 | 38169000_04.SLDPRT | data/bc/inc_email/20260813_19ffb722/ | 38169000_04.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 15 | 38539000_02 | 38539000_02.SLDPRT | data/bc/inc_email/20260910_1a08aeb7/ | 38539000_02 (V).ply; 38539000_02 (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 16 | 38965002_A | 38965002_A.SLDPRT | data/bc/inc_email/20260811_19ff1101/ | 38965002_A (V).ply; 38965002_A (X).ply; 38965002_A_V.ply (+1); idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 17 | 40851000_C | 40851000_C.SLDPRT | data/bc/inc_email/20260906_1a075a20/ | 40851000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 18 | 41017000_B | 41017000_B.SLDPRT | data/bc/inc_email/20260830_1a051b1a/ | 41017000_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 19 | 42426000_01 | 42426000_01.SLDPRT | data/bc/inc_email/20260726_19f9edb4/ | 42426000_03.ply; idx: benchmark_v3, zeroshot INVENTORY.json | no (base only; scan rev 03) | no | TEST-B part |
| 20 | 42725000_B | 42725000_B.SLDPRT | data/bc/inc_email/20260830_1a0516d6/ | 42725000_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 21 | 42821000_C | 42821000_C.SLDPRT | data/bc/inc_email/20260906_1a075b3f/ | 42821000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 22 | 43161000_A | 43161000_A.SLDPRT | data/bc/inc_email/20260906_1a075842/ | 43161000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 23 | 43244000_B | 43244000_B.SLDPRT | data/bc/inc_email/20260906_1a0757f3/ | 43244000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 24 | 43780000_B | 43780000_B.SLDPRT | data/bc/inc_email/20260901_1a05be86/ | 43780000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 25 | 43798000_A | 43798000_A.SLDPRT | data/bc/inc_email/20260906_1a0758e3/ | 43798000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 26 | 43895000_B | 43895000_B.SLDPRT | data/bc/inc_email/20260830_1a051aba/ | 43895000_B (1).ply; 43895000_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 27 | 43920000_A | 43920000_A.SLDPRT | data/bc/inc_email/20260830_1a051b50/ | 43920000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 28 | 43976000_A | 43976000_A.SLDPRT | data/bc/inc_email/20260830_1a051a82/ | 43976000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 29 | 44001000_B | 44001000_B.SLDPRT | data/bc/inc_email/20260906_1a0757b8/ | 44001000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 30 | 44556000_A | 44556000_A.SLDPRT | data/bc/inc_email/20260906_1a07588a/ | 44556000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 31 | 45352000_B | 45352000_B.SLDPRT | data/bc/inc_email/20260906_1a0756ca/ | 45352000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 32 | 46045000_D | 46045000_D.SLDPRT | data/bc/inc_email/20260906_1a0756f3/ | 46045000_D.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 33 | 46414000_C | 46414000_C.SLDPRT | data/bc/inc_email/20260830_1a051758/ | 46414000_C.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 34 | 47040000_C | 47040000_C.SLDPRT | data/bc/inc_email/20260906_1a075bd0/ | 47040000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 35 | 47247004_03 | 47247004_03.SLDPRT | data/bc/inc_codex/20260813_19ffb7a3/ | 47247004_03.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 36 | 47251002_02 | 47251002_02.SLDPRT | data/bc/inc_email/20260811_19ff11ae/ | 47251002_02 (V).ply; 47251002_02 (X).ply; 47251002_02_V.ply (+1); idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 37 | 47480000_C | 47480000_C.SLDPRT | data/bc/inc_email/20260901_1a05b9f8/ | 47480000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 38 | 47555020_1 | 47555020_1.SLDPRT | data/bc/inc_email/20260813_19ffb0d2/ | 47555020_1.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 39 | 47632001_A | 47632001_A.SLDPRT | data/bc/inc_email/20260830_1a051bcc/ | 47632001_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 40 | 48099000_01 | 48099000_01.SLDPRT | data/bc/inc_email/20260909_1a085174/ | 48099000_01 (V).ply; 48099000_01 (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 41 | 48333018_00 | 48333018_00.SLDPRT | data/bc/inc_email/20260716_19f6b058/ | 48333018_00.ply; idx: benchmark_v2, zeroshot INVENTORY.json | yes | no |  |
| 42 | 48340007_01 | 48340007_01.SLDPRT | data/bc/inc_email/20260716_19f6af64/ | 48340007_01.ply; idx: benchmark_v2, zeroshot INVENTORY.json | yes | no |  |
| 43 | 48412000_B | 48412000_B.SLDPRT | data/bc/inc_email/20260830_1a051b69/ | 48412000_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 44 | 48565005_A | 48565005_A.SLDPRT | data/bc/inc_email/20260830_1a051937/ | 48565005_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 45 | 48565022_A | 48565022_A.SLDPRT | data/bc/inc_email/20260830_1a051ad4/ | 48565022_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 46 | 48758000_A | 48758000_A.SLDPRT | data/bc/inc_email/20260901_1a05ba4b/ | 48758000_a.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 47 | 48910000_A | 48910000_A.SLDPRT | data/bc/inc_email/20260901_1a05beb9/, data/bc/inc_email/20260901_1a05bee1/ | 48910000_A 1.ply; 48910000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 48 | 48932000_B | 48932000_B.SLDPRT | data/bc/inc_email/20260830_1a051bfa/ | 48932000_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 49 | 48933000_B | 48933000_B.SLDPRT | data/bc/inc_email/20260906_1a07541a/ | 48933000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 50 | 48934000_B | 48934000_B.SLDPRT | data/bc/inc_email/20260901_1a05bf4f/ | 48934000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 51 | 48938000_C | 48938000_C.SLDPRT | data/bc/inc_email/20260830_1a051a13/ | 48938000_C.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 52 | 49025000_A | 49025000_A.SLDPRT | data/bc/inc_email/20260901_1a05b75e/ | 49025000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 53 | 49125000_A00 | 49125000_A00.SLDPRT | data/bc/incoming/eyal50/…/ | 49125000_A00 (1).stl; 49125000_A00 (2).stl; 49125000_A00.ply (+1); idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no | Eyal50 pkg: 20/30 split membership unresolved |
| 54 | 49162000_A | 49162000_A.SLDPRT | data/bc/inc_email/20260830_1a05173a/ | 49162000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 55 | 49163000_A | 49163000_A.SLDPRT | data/bc/inc_email/20260830_1a05169c/ | 49163000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 56 | 49186000_A | 49186000_A.SLDPRT | data/bc/inc_email/20260901_1a05b81f/ | 49186000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 57 | 49187000_B | 49187000_B.SLDPRT | data/bc/inc_email/20260906_1a075535/ | 49187000_B.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 58 | 49188000_A | 49188000_A.SLDPRT | data/bc/inc_email/20260830_1a0519aa/ | 49188000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 59 | 49191030_01 | 49191030_01.SLDPRT | data/bc/inc_codex/20260716_19f6ab35/ | 49191030_01.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 60 | 49191037_01 | 49191037_01.SLDPRT | data/bc/inc_codex/20260716_19f6b019/ | 49191037_01.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 61 | 49191038_02 | 49191038_02.SLDPRT | data/bc/inc_codex/20260716_19f6b08c/ | 49191038_02.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 62 | 49191054_02 | 49191054_02.SLDPRT | data/bc/inc_codex/20260716_19f6b2d4/ | 49191054_02.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 63 | 49227000_A | 49227000_A.SLDPRT | data/bc/inc_email/20260830_1a05198a/ | 49227000_A.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 64 | 49291000_A | 49291000_A.SLDPRT | data/bc/inc_email/20260901_1a05be32/ | 49291000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 65 | 49292000_A | 49292000_A.SLDPRT | data/bc/inc_email/20260906_1a0753df/ | 49292000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 66 | 49293001_B | 49293001_B.SLDPRT | data/bc/inc_email/20260830_1a0519e1/ | 49293001_B.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 67 | 49355000_A | 49355000_A.SLDPRT | data/bc/inc_email/20260901_1a05be5c/ | 49355000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 68 | 49438000_A | 49438000_A.SLDPRT | data/bc/inc_email/20260906_1a075572/ | 49438000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 69 | 49461000_A | 49461000_A.SLDPRT | data/bc/inc_email/20260906_1a0754f2/ | 49461000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 70 | 49553000_A | 49553000_A.SLDPRT | data/bc/inc_email/20260906_1a0755d3/ | 49553000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 71 | 49565000_A | 49565000_A.SLDPRT | data/bc/inc_email/20260906_1a075ba7/ | 49565000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 72 | 49566000_A | 49566000_A.SLDPRT | data/bc/inc_email/20260906_1a075660/ | 49566000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 73 | 50278001_A | 50278001_A_1.sldprt | data/bc/inc_codex/20260909_d0880594/ | 50278001_A.1 (V).ply; 50278001_A.1 (X).ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 74 | 50482000_A | 50482000_A.SLDPRT | data/bc/inc_email/20260726_19f9ea03/ | 50482000_A.ply; idx: benchmark_v3, zeroshot INVENTORY.json | yes | no | TEST-B part |
| 75 | 50535000_04 | 50535000_04.SLDPRT | data/bc/inc_email/20260813_19ffb461/ | 50535000_04.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 76 | 50537000_04 | 50537000_04.SLDPRT | data/bc/inc_email/20260813_19ffb58b/ | 50537000_04.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 77 | 50539000_05 | 50539000_05.SLDPRT | data/bc/inc_email/20260813_19ffb68d/ | 50539000_05.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 78 | 50552000_04 | 50552000_04.SLDPRT | data/bc/inc_email/20260813_19ffb333/ | 50552000_04.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 79 | 50553000_04 | 50553000_04.SLDPRT | data/bc/inc_email/20260813_19ffb669/ | 50553000_04.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 80 | 50566000_04 | 50566000_04.SLDPRT | data/bc/inc_email/20260813_19ffb4c9/ | 50566000_04 1.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 81 | 50567000_05 | 50567000_05.SLDPRT | data/bc/inc_email/20260813_19ffb6ce/ | 50567000_05.ply; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | no |  |
| 82 | 23317000_C1 | 23317000_C1.SLDPRT | data/bc/inc_email/20260901_1a05b6e2/ | 23317000_C 1.obj; 23317000_C 1.ply; 23317000_C.ply; idx: benchmark_v3, zeroshot INVENTORY.json | no (base only; scan rev C) | yes (data/) |  |
| 83 | 37126000_D | 37126000_D.SLDPRT | data/bc/inc_email/20260901_1a05b725/ | 37126000_D.obj; 37126000_D.ply; 37126000_D.stl; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | yes (data/) |  |
| 84 | 37361005_01 | 37361005_01.SLDPRT | data/bc/incoming/eyal50/…/ | 37361005_01.obj; 37361005_01.ply; 37361005_01.stl; idx: benchmark_v2, benchmark_v3, zeroshot INVENTORY.json | yes | yes (data/,other) | Eyal50 pkg: 20/30 split membership unresolved |

## Held out, convert later

| Part number | SLDPRT basename | Location dir | Has scan | STEP exists | Reason |
|---|---|---|---|---|---|
| 16231000_03 | 16231000_03.SLDPRT | data/bc/incoming/eyal50/…/ | no | no | EHO4 list (16231000_03) |
| 26687013_A00 | 26687013_A00.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (26687013_A00); N001-N014 (26687013_A00; v2 sealed_set_identification) |
| 28272000_B | 28272000_B.SLDPRT | data/bc/inc_email/20260830_1a0517ab/ | yes | no | HOLDOUT10 (28272000_B; v2 local_inventory) |
| 2927000_E2 | 2927000_E2.SLDPRT | data/bc/inc_email/20260726_19f9e9d4/, data/bc/incoming/eyal50/…/ | yes | no | Eyal50 quarantine (2927000_E2; PLAN.md OD-4, not used) |
| 31367000_A00 | 31367000_A00.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (31367000_A00); N001-N014 (31367000_A00; v2 sealed_set_identification) |
| 31409000_01 | 31409000_01.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (31409000_01); N001-N014 (31409000_01; v2 sealed_set_identification) |
| 38173000_02 | 38173000_02.SLDPRT | data/bc/inc_email/20260728_19fa78c2/ | yes | no | N001-N014 (38173000_01; v2 sealed_set_identification) |
| 38173001_01 | 38173001_01.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | no | no | EHO4 list (38173001_01); N001-N014 (38173001_01; v2 sealed_set_identification) |
| 41880000_D | 41880000_D.SLDPRT | data/bc/inc_email/20260830_1a051a9b/ | yes | yes (data/) | HOLDOUT10 (41880000_D; v2 local_inventory); explicit exclusion 41880000_D |
| 42637000_B | 42637000_B.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (42637000_B); N001-N014 (42637000_B; v2 sealed_set_identification) |
| 44773000_C | 44773000_C.SLDPRT | data/bc/inc_email/20260909_1a0850bb/, data/bc/incoming/eyal50/…/ | yes | no | Eyal50 quarantine (44773000_C; PLAN.md OD-4, not used) |
| 46181000_C | 46181000_C.SLDPRT | data/bc/inc_email/20260830_1a0519c6/ | yes | no | HOLDOUT10 (46181000_C; v2 local_inventory) |
| 46957010_A | 46957010_A.SLDPRT | data/bc/inc_email/20260830_1a05171e/ | yes | no | HOLDOUT10 (46957010_A; v2 local_inventory) |
| 47215000_B | 47215000_B.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (47215000_B); N001-N014 (47215000_B; v2 sealed_set_identification) |
| 47553000_C | 47553000_C.SLDPRT | data/bc/inc_email/20260830_1a0518ff/ | yes | no | HOLDOUT10 (47553000_C; v2 local_inventory) |
| 47846000_A | 47846000_A.SLDPRT | data/bc/inc_email/20260830_1a051a43/ | yes | no | HOLDOUT10 (47846000_A; v2 local_inventory) |
| 47848000_A | 47848000_A.SLDPRT | data/bc/inc_email/20260830_1a051a6e/ | yes | no | HOLDOUT10 (47848000_A; v2 local_inventory) |
| 47949013_B | 47949013_B.SLDPRT | data/bc/inc_email/20260830_1a051883/ | yes | no | HOLDOUT10 (47949013_B; v2 local_inventory) |
| 49160000_A | 49160000_A.SLDPRT | data/bc/inc_email/20260830_1a051afb/ | yes | no | HOLDOUT10 (49160000_A; v2 local_inventory) |
| 49205000_A | 49205000_A.SLDPRT | data/bc/inc_email/20260830_1a051b2f/ | yes | no | HOLDOUT10 (49205000_A; v2 local_inventory) |
| 49773004_B | 49773004_B.SLDPRT | data/bc/inc_codex/20260706_37fcd8cb/49773004_B_unzipped/, data/bc/incoming/eyal50/…/ | yes | no | Eyal50 quarantine (49773004_B; PLAN.md OD-4, not used) |
| 50476000_A | 50476000_A.SLDPRT | data/bc/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/, ~/Downloads/ | yes | no | EHO4 list (50476000_A); N001-N014 (50476000_A; v2 sealed_set_identification) |

Notes: 38173000_02 is held out because its base number matches the N001–N014 scan 38173000_01. Its revision differs, and its own 02 scans are CONSUMED Gold37 (benchmark_v3 has `sealed_sku_overlap: true`), so the owner decides. 30820000_A01 is not in this table: it is consumed and allowed.

## No physical scan found (43)

1094000_03, 1095000_04, 16074000_02, 18604000_01, 20179000_03, 25759002_01, 27924000_01, 34372018_C, 34419011_A, 37361007_02, 37361014_01, 37361031_01, 38547000_03, 4022000_03, 42744000_B, 44122001_0, 44513027_01A, 47435000_01, 47555040_B, 47555060_C, 47563000_1, 47579011_02, 47579017_02, 47780000_A, 48366000_01, 48390000_02, 48833000_B, 49011000_02, 49012000_02, 4917000_02, 49229001_A, 49229002_A, 49229003_A, 49229004_A, 49255000_A, 49260000_A, 49402000_02, 49467010_A, 49800000_A, 50187000_A, 50188000_A, 7335000_04, MP01736_01

## Unparseable names (3)

- `OH_Mec0222SKL PC arm ext.sldprt` in data/bc/inc_email/20260726_19f9eea2/
- `PS SP 05041143 SUPPORT.SLDPRT` in data/bc/inc_email/20260930_1a0f2bb2/
- `1.SLDPRT` in data/bc/incoming/eyal50/…/

## Unresolved

- Eyal50 20/30 dev/holdout membership lives in `data/bend_corpus/evaluation_slices/eyal_50_locked_split_v1.json`, which was not read. Worth-converting parts from the Eyal50 package are flagged. Their scans come from other sources, because Eyal50 has no scans.
- HOLDOUT10 was resolved by name in the v2 inventory. Its identity rule is a hash rule, and hashes were not checked here.
- Scan files under `uploads/` and in opaque folders (`physical15_v1/opaque`, `2026-08-13_..._blind/opaque`) with non-SKU names cannot be matched by filename.
