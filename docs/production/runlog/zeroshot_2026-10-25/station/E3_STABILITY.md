# E3 verdict stability under resampling (2026-10-26)

> **Measurement only.** No E3, eval, contract, gold or label file was changed. Engine `e3-1.12.0`, code fingerprint `5fbed707…` (the same before and after every run). No installs, no commits. Every heavy run held one geometry slot (`/private/tmp/claude-501/sherman_slot.sh`), one job at a time; peak RSS stayed under 1 GiB. Machine-readable results: `e3_stability/results.json`; tools: `e3_stability/tools/`; figures: `e3_stability/*.png`. (`station/results.json` belongs to DENSE_READINESS and was not touched.)

## Answer first

- **FAIL↔PASS instability is rare but real, and it is concentrated on one part.** Across 88 unique scans (G1 50, plus 38 unique physical scans in A1 and the DEV cohort), **2 scans (2.3 %) and 11 of 1619 characteristics (0.68 %) show both a FAIL and a PASS** across the 7 realizations. 8 characteristics (0.49 %) go FAIL→PASS from the native verdict. Both scans are part 23553000 B: the full scan `sc_6eba02723d4b` and scan 2 `pc_186463746daa`. **No G1 scan has a FAIL↔PASS change.**
- **Any decision change** (most of them PASS↔REVIEW) affects 19 of 88 scans (21.6 %) and 55 characteristics (3.4 %). On physical scans alone it is 17 of 38 scans and 53 of 1066 characteristics (5.0 %).
- **Mechanism: a discrete skin-assignment choice on a short panel, not randomness.** E3 is deterministic: 83 of 83 native reruns that have an earlier reference output are byte-identical in their bends and summary. On 23553000 B, the 10–19 mm panel between B5 and B6 is grown either over both sheet skins or over only the inner skin. With one skin, the inner skin is taken as the OML, so the panel's OML plane is off by **0.98 mm, about one thickness (t = 0.92 mm)**. That moves B5's line by 1.27 mm and every length that ends on it by 1.1–1.3 mm, which is 10–12× U. Registration and the mirror decision do not change.
- **The native FAIL is very likely the correct verdict; the PASS is the error.** In the native run the panel owns both skins at the right separation, and its articulated fit explains more of the scan (total 1.447, against 1.415–1.431 in every perturbation). The perturbed PASS mode measures the inner skin as the outer one. This is a **false-PASS risk**.
- **G1 correctness.** There are no FAIL↔PASS disagreements to judge. The 2 G1 changes are both safe: one native FAIL becomes REVIEW (truth: out of tolerance, so FAIL is correct), and one native UNKNOWN becomes FAIL (truth: out of tolerance, so that FAIL is correct). No perturbed G1 run produced a wrong decided verdict.
- **Recommended fix.** Both fix families reach 0 FAIL↔PASS on every surveyed scan in leave-one-out simulation:
  - the honest-U rule (c, U' = √(U² + (2σ)²) over 3 extra resamples), with the least coverage loss: **1.6 %** of decided characteristics (2.3 % physical, 0 % G1);
  - the cheapest self-check (a, K = 2 resamples, withhold on any disagreement): **2.2 %** loss (3.2 % physical, 0.2 % G1).

  The root-cause fix (b) is needed anyway to turn the 23553000 B characteristics back into decided, correct FAILs.

## Method

**Inputs** (`tools/build_manifest.py`). The runs used no labels and no truth; G1 truth was read only by the scorer.
- **G1:** the 50 card-tier cases of `zsprod/g1c`, with the scan, compiled part and recipe the G1 run used.
- **A1:** the 27 cases of `freeze/cases_A1.json`, with the frozen DEV compiles and recipes. Compiled and recipe sha256 were checked against the freeze.
- **DEV E3 cohort:** the 50 OK scans of `baseline/e3_driver.jsonl`, minus 17 whose INVENTORY custody is `CONSUMED_FRESH` (12) or `CONSUMED_DEVCHECK` (5). Those were excluded under the "never FRESH / DEV-CHECK" rule, leaving 33 `DEV_CONSUMED` scans.
- 21 cohort scans are the same bytes as A1 scans with the same compile, so each was run once and is counted in both tables: **89 unique jobs**.
- **A1 `pc_b4719db3dd69` (47479000_C)** has a BLOCKED compile, and the native E3 run refuses it (rc 2). It is not in the prevalence counts.

**Realizations per scan** (`tools/run_all.py`), 623 E3 runs in 1.9 h:
- native;
- 5 random 95 % subsamples, seeds 9501–9505, using dense_rel `subsample.py`;
- one 0.40 mm thinning, using dense_rel `thin_exp.py`.

Each realization runs through dense_rel's `station/tools/run_step.py e3` (the frozen CLI with the eval-runner environment) at 1 thread, with the **production limits of 180 s and a 170 s budget**. The longest native run took 34 s. E3 ignores mesh faces (`scan_io`), so the point-only PLY of a subsample loses nothing.

**What counts as a change.** For every characteristic, each perturbed decision is compared with the native one. "FAIL↔PASS" means that FAIL and PASS both occur among the 7 decisions. Also recorded:
- the value spread (max − min) over U;
- bimodality: a gap larger than 3·U between sorted values;
- the pose change against native;
- whether the mirror check ran and what it decided.

**Mechanism.** `tools/probe.py` runs the same CLI in-process with pass-through wrappers on `measure_bend_geometry`, `_fit_side`, `_flat_start` and `_reseat_plane`. It records the region and station planes, the panel-owned points, the skin choices and the window shifts. Its result.json is identical to the survey's for all 4 probed runs.

## Prevalence

| Corpus | Scans run | Scans with any change | Scans FAIL↔PASS | Scans FAIL→PASS | Chars | Chars any change | Chars FAIL↔PASS | Chars FAIL→PASS | Validity changed | Pose changed (>1° or >1 mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 (synthetic) | 50 | 2 (4 %) | 0 | 0 | 553 | 2 (0.36 %) | 0 | 0 | 0 | 0 |
| A1 | 26 | 13 (50 %) | 2 (7.7 %) | 1 (3.8 %) | 813 | 44 (5.4 %) | 11 (1.35 %) | 8 (0.98 %) | 0 | 0 |
| DEV cohort | 33 | 15 (45 %) | 1 (3.0 %) | 1 (3.0 %) | 945 | 44 (4.7 %) | 9 (0.95 %) | 8 (0.85 %) | 0 | 0 |
| **All, unique scans** | **88** | **19 (21.6 %)** | **2 (2.3 %)** | **1 (1.1 %)** | **1619** | **55 (3.4 %)** | **11 (0.68 %)** | **8 (0.49 %)** | 0 | 0 |

Directions, counted per characteristic and direction in each corpus:
- **A1:** PASS→REVIEW 13, REVIEW→PASS 9, FAIL→PASS 8, REVIEW→FAIL 6, FAIL→REVIEW 4, PASS→FAIL 3, REVIEW→UNKNOWN 2.
- **DEV cohort:** PASS→REVIEW 13, FAIL→PASS 8, REVIEW→PASS 8, REVIEW→FAIL 5, FAIL→REVIEW 5, PASS/REVIEW→UNKNOWN 2+2, PASS→FAIL 1.
- **G1:** UNKNOWN→FAIL 1, FAIL→REVIEW 1.

Inspection validity never changed, and no pose moved by more than 0.31° / 0.50 mm. Summaries changed on 2 scans: G1 ZS-006 (REVIEW_REQUIRED → OUT_OF_TOLERANCE_FOUND) and `sc_822f083e3dfc` (OUT_OF_TOLERANCE_FOUND → REVIEW_REQUIRED).

**Mirror and registration decision.** The mirrored-part decision (`MIRRORED_PART_SUSPECTED`) never changed.
- On 4 unique scans (`pc_d45186653eed`, `pc_38091ac91d9d`, `pc_ff1667a2a80a`, `sc_3c93a0bf6f97`), whether the mirror check *ran* differed between realizations. That is its "suspicious" trigger, and it changed no outcome.
- On 23553000 B full, the mirrored fit scored 1.429 against a proper fit of 1.447 native, and 1.419–1.434 against 1.415–1.431 perturbed. That is never above the +0.08 refusal margin.
- The `pose_margin` 0.0195 in the log is the gap to the *second proper* hypothesis, not to the mirror. It grows to 0.17–0.37 in the perturbations because the native best pose (two-skin articulation) loses 0.016–0.032 of its total.
- The registered pose itself moves by only 0.07–0.10° / 0.16–0.18 mm. The registration runs on a 0.8 mm-decimated copy, so its points change with any resampling.

Determinism: 83 of 83 native reruns with an earlier reference output (G1 `g1c/work`, cohort `sherman_zeroshot_work/e3`) match it exactly. The 6 without one are the STL A1 cases, which had no earlier run, and the blocked case.

### G1 (50 card-tier cases)

48 of 50 scans have no change in any of the 6 perturbations.

| Case | Native summary | Characteristic | Native | Perturbed | Truth | Correct decision | Wrong decided verdicts |
|---|---|---|---|---|---|---|---|
| E2-DEV-ZS-006 | REVIEW_REQUIRED | B1 flange B | UNKNOWN | FAIL in 2 of 6 (s2, s5) | 10.100 mm, out of tol | FAIL | 0 |
| E2-DEV-TD-001 | OUT_OF_TOLERANCE_FOUND | B1 flange B | FAIL (18.762) | REVIEW in 2 of 6 (s1, s2) | 18.764 mm, out of tol | FAIL | 0 |

### A1 (26 run; `pc_b4719db3dd69` refused, compile BLOCKED)

13 scans have no change. Rows show only the scans with a change.

| Scan | Native summary | Chars | Changed | FAIL↔PASS | Directions | Max spread/U | Max pose Δ (°, mm) | Native pose margin |
|---|---|---|---|---|---|---|---|---|
| **pc_6eba02723d4b** (23553000 B full) | OUT_OF_TOLERANCE_FOUND | 64 | 14 | **9** | FAIL→PASS 8, FAIL→REVIEW 2, PASS→FAIL 1, PASS→REVIEW 1, REVIEW→FAIL 1, REVIEW→PASS 1 | 11.6 | 0.09, 0.18 | 0.0195 |
| **pc_186463746daa** (23553000 B scan 2) | OUT_OF_TOLERANCE_FOUND | 64 | 7 | **2** | PASS→FAIL 2, PASS→REVIEW 3, REVIEW→FAIL 2, FAIL→REVIEW 1 | 5.4 | 0.06, 0.02 | 0.1028 |
| pc_cec08dfca9ab | OUT_OF_TOLERANCE_FOUND | 48 | 5 | 0 | REVIEW→FAIL 2, REVIEW→UNKNOWN 2, PASS→REVIEW 1 | 1.3 | 0.06, 0.16 | 0.0064 |
| pc_71cf0b216f43 | OUT_OF_TOLERANCE_FOUND | 48 | 3 | 0 | FAIL→REVIEW 1, PASS→REVIEW 1, REVIEW→PASS 1 | 0.1 | 0.03, 0.28 | 0.0064 |
| pc_70b39f530f00 | REVIEW_REQUIRED | 12 | 2 | 0 | PASS→REVIEW 1, REVIEW→PASS 1 | 3.3 | 0.31, 0.38 | 0.0036 |
| pc_d45186653eed | REVIEW_REQUIRED | 28 | 2 | 0 | PASS→REVIEW 1, REVIEW→PASS 1 | 0.1 | 0.00, 0.02 | 1.6156 |
| pc_5a1ecd7a4db3 | OUT_OF_TOLERANCE_FOUND | 64 | 2 | 0 | PASS→REVIEW 1, REVIEW→PASS 1 | 0.5 | 0.03, 0.03 | 0.1379 |
| pc_38091ac91d9d | REVIEW_REQUIRED | 4 | 2 | 0 | PASS→REVIEW 2 | 0.8 | 0.21, 0.24 | 0.6751 |
| pc_07062eee17aa | OUT_OF_TOLERANCE_FOUND | 48 | 2 | 0 | PASS→REVIEW 1, REVIEW→PASS 1 | 1.2 | 0.03, 0.02 | 0.1999 |
| pc_f2475286312b | OUT_OF_TOLERANCE_FOUND | 48 | 2 | 0 | REVIEW→FAIL 1, REVIEW→PASS 1 | 0.7 | 0.07, 0.03 | 0.4296 |
| pc_d1732de18023 | REVIEW_REQUIRED | 16 | 1 | 0 | PASS→REVIEW 1 | 0.1 | 0.00, 0.01 | 0.6656 |
| pc_ff1667a2a80a | REVIEW_REQUIRED | 16 | 1 | 0 | REVIEW→PASS 1 | 0.1 | 0.00, 0.01 | 0.3689 |
| pc_777c64c88967 | OUT_OF_TOLERANCE_FOUND | 36 | 1 | 0 | REVIEW→PASS 1 | 0.6 | 0.04, 0.00 | 0.6276 |

### DEV E3 cohort (33 DEV_CONSUMED scans)

18 scans have no change. 12 rows are the same jobs as the A1 rows above (`sc_` = `pc_` with the same hash). The scans that appear only in the cohort are:

| Scan | Native summary | Chars | Changed | FAIL↔PASS | Directions | Max spread/U | Max pose Δ (°, mm) | Native pose margin |
|---|---|---|---|---|---|---|---|---|
| sc_822f083e3dfc | OUT_OF_TOLERANCE_FOUND | 8 | 6 | 0 | FAIL→REVIEW 2, PASS→UNKNOWN 2, PASS→REVIEW 1, REVIEW→FAIL 1 | 11.3 | 0.04, 0.50 | 0.1994 |
| sc_3c93a0bf6f97 | REVIEW_REQUIRED | 28 | 1 | 0 | PASS→REVIEW 1 | 0.2 | 0.00, 0.01 | 1.6583 |
| sc_4057ba2144fb | REVIEW_REQUIRED | 24 | 1 | 0 | PASS→REVIEW 1 | 0.5 | 0.06, 0.01 | 1.193 |
| sc_8026eedce786 | OUT_OF_TOLERANCE_FOUND | 48 | 1 | 0 | PASS→REVIEW 1 | 0.1 | 0.03, 0.01 | 0.5766 |

## The unstable cases

### 23553000 B full scan (`sc_6eba02723d4b`): skin assignment on a short panel

![B full scan modes](e3_stability/e3stab_B_full_scan_modes.png)

- **Two modes, separated by about one sheet thickness.**
  - Native and s4 give B5 flange B 19.45, B6 A/B 72.68 / 19.54 and B7 A 72.71: FAIL.
  - s1, s2, s3, s5 and the thinning give 18.13, 73.75 / 18.21 and 73.81: PASS. The B5 angle reads 61.5° (REVIEW) instead of 59.9°, and the B5 line position −1.02 instead of −2.24 mm (FAIL either way).
- **The modes are 1.1–1.3 mm apart, against U = 0.10–0.12 mm** (spread/U 10–12), and the reported U says nothing about this.
- **The thinning shows the same switch one panel earlier.** It also moves B2, B3 and B4, whose short panel has the same 18.18 mm nominal. B2 A goes PASS→FAIL and B2 B, B3 A/B and B4 B go FAIL→PASS.

![B5 section](e3_stability/e3stab_B5_section_full_scan.png)

**Mechanism, from the probe (native against seed 9501, subsample mapped into the native frame):**
- **The region-wide plane of the short panel moves.** This is panel 13 (B5 parent, B6 child), with flanges of 9.7 and 18.2 mm at a 60° bend. Its `PanelFit` plane tilts 0.29° and shifts **1.05 mm**. The other panels move by 0.02° / 0.01 mm or less.
- **Native panel 13 owns both skins.** It holds 4022 points, at 0 and −0.95 mm.
- **In the subsample it owns only the inner skin** (1759 points at −0.98 mm), and that skin becomes skin 0, the OML.
- **None of the later safeguards repairs it.**
  - `_reseat_plane` returns None (no reseat) for both sides on both runs.
  - The station `_fit_side` fits use the level each run believes is the OML (0 of 10 inner-skin fits in both), so every station sharp follows the region plane.
  - The sharp-line consistency move (0.15 mm) moves the native line by 0.23 mm and the subsample line by 0. It cannot undo a 1 mm plane offset that the stations share.
- **Result.** B5's OML sharp moves 1.27 mm, B6's sharp 0.7 mm, and every ADJACENT_BEND length ending on them (B5 B, B6 A/B, B7 A) jumps.
- **The native mode is the physically consistent one.** Its two skins sit 0.95 mm apart (measured t 0.92), and its articulated total is the highest of all realizations (1.447 against at most 1.431). So the 8 FAIL→PASS flips are **false PASSes**.
- **Where the two-way choice is made.** Upstream, in the articulated fit's region growing and skin labelling of a short, narrow panel next to a 60° and a 120° bend. A 5 % change in the points tips it. It is a choice between two local optima of a deterministic fit (the one-skin labelling explains about 0.02 less), not a RANSAC seed or a randomness leak.

### 23553000 B scan 2 (`pc_186463746daa`): station skin choice and window shift

![B scan 2 modes](e3_stability/e3stab_B_scan2_modes.png)
![B scan 2 stations](e3_stability/e3stab_B_scan2_B5_stations.png)

- **Same part, same short panel, a different branch.** The region plane is stable here (Δ ≤ 0.012° / 0.004 mm), but the B5 station windows flip between two discrete configurations:
  - **native:** `_flat_start` shift 1.00 mm, `_reseat_plane` refused, and the short panel's 5 stations fitted on the **inner skin + t**;
  - **seed 9501:** shift 3.25 mm, the reseat accepted, and all 5 stations on the **outer skin**.
- **Its effect.** B5 flange A moves 10.22 ↔ 10.78 mm (REVIEW ↔ FAIL) and B5 flange B 18.52 ↔ 18.79 (PASS ↔ REVIEW). Seeds 1, 3 and 5 take one branch and seeds 2 and 4 the other.
- **The thinning reaches FAIL** on B5 flange B (18.83) and on B6 line position (0.74 against 0.15, PASS→FAIL).
- **The native PASS on B5 flange B sits 0.04 mm inside the guard band** (18.52 + 0.11 against a 18.68 limit), so a 0.3 mm mode jump crosses it.
- **There is no truth to say which branch is right.** Both stay inside one skin's geometry; the difference is which skin and which window.

### Other scans (no FAIL↔PASS)

- **`sc_822f083e3dfc`, a 135° part with 6.4 mm flanges** (spread/U up to 11). In seed 9502 and in the thinning, the 6.4 mm ADJACENT_BEND flanges go UNKNOWN, and B1 B / B2 A jump about 1.1 mm. That is the same short-panel family, but the decisions only move to REVIEW or UNKNOWN, apart from one REVIEW→FAIL.
- **Everything else is PASS↔REVIEW or REVIEW↔FAIL at spread/U ≤ 1.3.** These are ordinary guard-band noise: a value within about U of a limit.
- **The two exceptions** (`pc_70b39f530f00` B2 line position 3.3×U, `pc_cec08dfca9ab` angles) move only to or from REVIEW.

## What the unstable characteristics have in common (physical scans, 1066 characteristics)

| Feature | Bucket | Chars | Unstable | FAIL↔PASS | Unstable % |
|---|---|---|---|---|---|
| Shortest flange of the bend | < 12 mm | 206 | 24 | 4 | 11.7 |
| | 12–20 mm | 436 | 15 | 5 | 3.4 |
| | 20–40 mm | 212 | 7 | 0 | 3.3 |
| | ≥ 40 mm | 196 | 5 | 2* | 2.6 |
| Nominal included angle | < 75° | 48 | 9 | 4 | 18.8 |
| | 75–105° | 830 | 21 | 2 | 2.5 |
| | > 105° | 188 | 23 | 5 | 12.2 |
| Thickness used | < 1.2 mm | 192 | 23 | **11** | 12.0 |
| | 1.2–2 mm | 710 | 21 | 0 | 3.0 |
| | ≥ 2 mm | 160 | 9 | 0 | 5.6 |
| Native pose margin | < 0.05 | 276 | 24 | 9 | 8.7 |
| | 0.05–0.2 | 238 | 17 | 2 | 7.1 |
| | ≥ 0.2 | 552 | 12 | 0 | 2.2 |
| Kind | FLANGE_LENGTH | 532 | 25 | 10 | 4.7 |
| | INCLUDED_ANGLE | 266 | 19 | 0 | 7.1 |
| | BEND_LINE_POSITION | 266 | 9 | 1 | 3.4 |

\* B6 flange A and B7 flange A are the long sides (73.9 mm) of bends whose far boundary is the short panel's sharp. They are in the same failure.

- **Short panel, thin sheet, non-90° bend.** Every FAIL↔PASS characteristic sits on 23553000 B (t = 0.92 mm, panels 9.7 / 18.2 mm, 60° and 120° bends), or on a length that ends at that panel. Instability is 4–5× more frequent below 12 mm and away from 90°.
- **Low support on the short panel.** In the B full scan, panel 13 owns 4022 points (native) against 1759 (subsample). Its station fits use 1481 against 1950 support points, about 300 per station. B scan 2's short-panel stations hold 85–296 points.
- **Bimodal values.** 10 of 11 FAIL↔PASS characteristics are bimodal (a gap of more than 3·U), against 13 of 55 unstable characteristics overall. FAIL↔PASS is a mode switch, not a slow drift.
- **The native pose margin is a correlate, not the cause.** All 9 FAIL↔PASS characteristics with a margin below 0.05 are on B full, where the margin is small *because* the native two-skin articulation is the better of two near-equal hypotheses.
- **Articulated-stage features did not trigger.** No wrong-way fold, joggle riser or symmetric-rival choice is involved in any unstable case.
- **G1 is almost immune** (0.36 % changed, none on short flanges). Its synthetic short flanges are clean two-skin points, so the skin labelling never has to choose.

## Fixes, ranked (proposals only; none implemented)

**How the fixes were simulated** (`tools/summarize.py`), leave-one-out on the 7 realizations:
- Each realization in turn plays "the production scan", and its self-check uses K of the other 6. Every combination is scored.
- **Residual** = characteristics where some checked output is FAIL and another is PASS.
- **Coverage loss** = native PASS/FAIL decisions withheld, averaged over the combinations, out of 959 decided characteristics.

| Fix | K extra runs | FAIL↔PASS residual | Any-change residual | Coverage loss, all | Physical | G1 | E3 cost |
|---|---|---|---|---|---|---|---|
| none | – | **11** | 55 | 0 | 0 | 0 | 1× |
| (a) self-check: withhold (REVIEW) any PASS/FAIL that a resample disagrees with | 1 | 4 | 46 | 1.5 % | – | – | 2× |
| | **2** | **0** | 41 | **2.2 %** | 3.2 % | 0.2 % | **3×** |
| | 3 | 0 | 38 | 2.8 % | 4.0 % | 0.3 % | 4× |
| | 5 | 0 | 21 | 3.5 % | – | – | 6× |
| (a′) the same, only on scans with any flange < 25 mm (61 of 88 scans) | 2 | 0 | 41 | 2.2 % | – | – | 1 + 2 × 0.69 |
| (c) honest U: U' = U + range over native + K | 2 | 0 | 70 | 2.5 % | 3.7 % | 0 % | 3× |
| (c′) honest U: U' = √(U² + (2σ)²), σ over native + K | 2 | 1 | 57 | 1.4 % | – | – | 3× |
| | **3** | **0** | 56 | **1.6 %** | 2.3 % | **0 %** | **4×** |
| (b) root cause: short-panel skin assignment | 0 | 0 expected on B (see below) | not estimable | about 0, and B becomes decided FAIL | | | about 1× |

**Run time.** Native E3 on this corpus takes a median of 8 s, a P95 of 29 s and a maximum of 34 s (1 thread).
- **Serially:** K = 2 gives a maximum of about 103 s, and K = 3 about 137 s, inside the 170 s budget for every surveyed scan. A scan at 40 M faces (140–160 s native, DENSE_READINESS) would not fit.
- **In parallel** on spare cores (E3 is single-threaded and deterministic): about +10 % wall time and K× CPU, with about 0.6 GiB RSS per extra run.

**Ranking:**

1. **(b) Root-cause fix of short-panel skin assignment.** This is the only option that restores *correct* decided verdicts. Proposed shape:
   - when a panel's region holds only one skin, decide which skin it is from the neighbours (the adjacent panels' two-skin planes and the measured t), not from the panel's own level;
   - require the OML plane to sit about t outside any material the panel's slab contains;
   - extend `_reseat_plane` so a one-skin strip can be moved by −t when the other skin is present in the slab;
   - make the `_flat_start` / `_skin_level` choices hysteresis-free (pick by score margin, and withhold when within a margin).

   On 23553000 B this would keep the two-skin answer (FAIL, physically consistent) in all realizations: 0 FAIL↔PASS and no coverage loss. Its effect elsewhere cannot be estimated without implementing it. It is an E3 version bump (e3-1.13.0) with a G1, A1 and frozen-recompile re-run and a ledger amendment.
2. **(c′) Honest U from 3 resamples**, until (b) lands. It gives 0 FAIL↔PASS on all surveyed scans with the **least coverage loss (1.6 %; 0 % on G1)**. On B it turns the mode-switching characteristics into REVIEW and keeps far-from-limit decisions. It can never create a PASS from a FAIL: U only grows. It costs 4× E3 compute (parallelisable).
3. **(a) Withholding self-check, K = 2.** It also reaches 0 FAIL↔PASS, is cheaper (3×) and simpler to reason about ("any disagreement → REVIEW", which never creates a PASS). It costs 2.2 % of decided characteristics, 3.2 % physical. Gating it to scans with a flange under 25 mm saves about 30 % of the extra runs and gives the same result here.

**Answer to "0 FAIL→PASS with the least coverage loss":** (c′) honest U with K = 3 gives 0 FAIL↔PASS on every surveyed scan at 1.6 % coverage loss. (a) with K = 2 is the cheapest at 0, with 2.2 %.

**Caveats:**
- The residual is measured on only 7 realizations per scan, and the FAIL↔PASS population is 11 characteristics on one part number. Both fixes are tuned on that, so "0" is a lower bound on what a K-resample check would miss on unseen parts.
- On B scan 2, 2 of 6 perturbations land in the "other" mode, so K = 1 misses it with probability of about ⅔. That is why K ≥ 2 is required.
- Neither (a) nor (c) makes the B verdicts *correct*: they withhold them. Only (b) does.

## Figures

- `e3_stability/e3stab_B5_section_full_scan.png`: B5 section, both plane-fit modes and the scan points, plus the skin histogram.
- `e3_stability/e3stab_B_full_scan_modes.png`: B full-scan characteristics over the 7 realizations, with U and the tolerance.
- `e3_stability/e3stab_B_scan2_modes.png`: the same for B scan 2.
- `e3_stability/e3stab_B_scan2_B5_stations.png`: B scan 2 station skin choice and window shift.

## Reproduce

Scratch: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3stab`. Run each step with `.venv/bin/python`.

1. `tools/build_manifest.py manifest.json`
2. `tools/run_all.py manifest.json`
3. `tools/analyze.py manifest.json analysis.json`
4. `tools/summarize.py analysis.json manifest.json summary.json`
5. `tools/probe.py SCAN COMPILED_DIR RECIPE OUT B5,B6`
6. `tools/plots.py OUTDIR`
