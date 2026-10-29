# N1 new-family scans: one-time zero-shot run (2026-10-25)

> **Label.** This is a one-time run on 4 never-seen part families (received from the customer on 2026-10-19). N = 4 scans and 4 families, and only one of them has bends (26 gold lines). **That is too small for a headline claim.** Each arm ran exactly once per scan. Nothing was tuned on these scans, and no arm will be re-run.

## Custody chain

| Step | Record |
|---|---|
| Gold | Blind markers A and B, then adjudication: `gold/N1/ADJUDICATION_N1.json`. A/B agreement was F1 1.0 (28/28 lines, 3/3 zero-bend). The canary `gb_3f9a9c` scored F1 1.0 / 1.0 / 1.0 for A, B and the adjudicated lines. |
| Labels | `data/bend_corpus/annotations/ai_bend_lines_newscans_n1_v1/` (a new root; no existing root was touched), with provenance sidecars. |
| Gold ledger | REGISTER seqs 131–134. FROZEN seqs 135–138. Appended by `tools/gold_ledger_newscans.py`; see "Guard" below. |
| Pre-registration | `PREREG.json`, sha256 `a7cd7c5d…dede9`. Freeze-ledger NOTE seq 17 (`frz_e6adfdb22ab8`). |
| Run binding | Freeze-ledger RUN seq 18 (`frz_fc9385a2e218`), appended before the first lane process started. |
| Runs | 4 scans × 4 arms = 16 processes. Each ran through `tools/guarded_slot.sh <sid>` with guard basis GOLD_LEDGER_FROZEN, one geometry slot and OMP 1. Result: **16/16 status OK, pins_ok** (v2 ran with `--strict-pins`). Outputs are in `runs/v1/{zs,z3,z2}/` and `runs/v2/zs/`. |
| Scoring | Done once, with `tools/newscans_n1_score.py`, which calls the unchanged `zs_lane_v1/scoring.py` (frozen `score_case`, 5° / 4 mm / 0.6 / 0.6). Output: `RESULTS.json` (sha256 `39a2da44…2d332`) and `RESULTS_per_case.json`. |
| Results binding | Freeze-ledger NOTE seq 19 (`frz_e59ffe1fd236`, entry `bbacf350…`). It binds the RESULTS.json and per-case shas. This row was filled in afterwards; that is the only edit to RESULTS.md since the NOTE. |

**E3, the CAD-assisted path, was not run.** The customer sent SolidWorks `.SLDPRT` files, not STEP, so no recipe can be compiled. No conversion was attempted.

**Guard.**
- `research/zeroshot/common/gold_guard.py` is **unchanged**. Its hash is pinned in both lanes' code fingerprints, so editing it would have broken the pins.
- The N1 scans are admitted by the guard's existing GOLD_LEDGER_FROZEN rule.
- The one new mechanism is `tools/gold_ledger_newscans.py`. It appends REGISTER entries for the `newscans_<batch>/REGISTRY.json` scans, with portion `NEW_UNSEEN_FAMILY_TEST`. It refuses:
  - a scan already in SPLIT_FROZEN, INVENTORY or the ledger;
  - a scan whose sha256 differs from the registry.
- It also appends FROZEN entries, but only for that portion, and only when the label sits directly under `ai_bend_lines_newscans_<batch>_v1/` and its sha256 matches.
- The entries use the same hash-chained format, and `gold_guard verify` validates the ledger (138 entries).
- No refusal was weakened. A registered but not-yet-frozen N1 scan was checked and refused (GOLD_NOT_FROZEN), and TEST-B is still refused (TEST_B_SEALED).

## Results

**Scan with bends: 47085000** ("part with short tabs"; 26 gold lines):

| Arm | Pred | TP / FP / FN | P | R | **F1** |
|---|---|---|---|---|---|
| **Primary: zs_lane_v1 zs** | 23 | 21 / 2 / 5 | 0.913 | 0.808 | **0.857** |
| zs_lane_v2 zs | 23 | 22 / 1 / 4 | 0.957 | 0.846 | 0.898 |
| zs_lane_v1 Z3 only | 25 | 22 / 3 / 4 | 0.880 | 0.846 | 0.863 |
| zs_lane_v1 Z2 only | 22 | 22 / 0 / 4 | 1.000 | 0.846 | 0.917 |

**Zero-bend rolled shells: 41018000, 45351000, 51101001** ("rounded part"; 0 gold lines). These follow the frozen convention: `score_case` gives them the disposition ZERO_BEND_CONTROL with no F1, and they are reported as FP lines. The display F1 on the owner page is 1.0 when a scan has 0 FP lines and 0.0 otherwise.

| Arm | 41018000 | 45351000 | 51101001 | Control FP lines |
|---|---|---|---|---|
| zs_lane_v1 zs | 0 | 0 | 0 | **0** (3/3 clean) |
| zs_lane_v2 zs | 0 | 0 | 0 | **0** |
| Z3 only | 0 | 0 | 0 | 0 |
| Z2 only | 0 | **16** | 0 | 16 (Z2 cuts the 45351000 roll into flange-graph facets) |

**Micro.**
- Frozen aggregate: complete-label line scans only, so this is 47085000 alone, and it equals the F1 in the first table.
- "Micro incl. control FPs" adds the lines predicted on the zero-bend scans as FP:

| Arm | micro (frozen) | micro incl. control FP | TP / FP / FN (incl. control) |
|---|---|---|---|
| **zs_lane_v1 zs** | **0.857** | **0.857** | 21 / 2 / 5 |
| zs_lane_v2 zs | 0.898 | 0.898 | 22 / 1 / 4 |
| Z3 only | 0.863 | 0.863 | 22 / 3 / 4 |
| Z2 only | 0.917 | 0.688 | 22 / 16 / 4 |

The family bootstrap CI is degenerate here, because only one family has lines.

**Angle agreement on matched lines.** The engine's included angle (`incl`) is compared with the gold included angle from the labelling tool's patch fit.
- zs v1, 21 matched lines: median |Δ| 0.08°, max 0.32°, mean signed +0.03°. All 21 are within 1°.
- v2: the same, over 22 lines.
- Z3: max 0.32°.
- Z2: median 0.15°, max 0.57°.
- None of the matched lines is a joggle line. The only joggle, B25/B26, was missed by every arm.
- The per-line values are in `RESULTS.json` → `arms.*.per_scan[0].matches`.

**Runtime and memory.**
- Per-process wall time ranged from 7 s to 1535 s. The large rolled scans were slow: 41018000 has 1.69 M vertices and took 514 s for zs v1, 1103 s for Z3 and 1535 s for v2.
- Peak RSS was at most 2.6 GiB, with 2 concurrent workers.

## What the primary lane missed on 47085000 (descriptive; no change follows)

| Gold line | What it is | Why it was missed |
|---|---|---|
| B1 | End flange (80 mm, 90°). The plate's outer skin is not scanned near the flange. | SPAN: the engine line is 36 mm long, covering 40 % of the gold span. It is counted as 1 FN plus 1 FP. |
| B4 | Saddle-shaped lanced lip (70 mm, 90°). | SPAN: the engine line is 30 mm, covering 38 %. 1 FN plus 1 FP. v2 recovers it with its join step. |
| B14 | The hook lip on one left leg, where the foot is only scanned on one skin. | Not detected. The nearest engine line is a different fold 13.8 mm away. |
| B25, B26 | The shallow 3 mm edge joggle on a window strip (two ~75–81° folds about 1 mm apart). | Not detected by any arm. |

These are the same classes as the post-hoc TEST-B analysis:
- the span convention on partially scanned or saddle features (class 1);
- tiny or short-wall bends (class 2).

On all three rolled shells, the v1 primary, v2 and Z3 lanes produced **no false lines**. That is the main new-family behaviour seen here.
