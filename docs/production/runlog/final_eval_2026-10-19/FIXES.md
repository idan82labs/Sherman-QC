# Engine fixes after the one-time held-out evaluation (2026-10-19)

> **Post-hoc, consumed.** FRESH and DEV-CHECK were consumed by the one-time evaluation (`REPORT.md`). Every
> number on those sets below is a per-case **diagnostic** of the fixed engine, not a new held-out result,
> and must never be quoted as one. The coverage metric that needs no labels is the compile rate over every
> STEP file we have (section 2). The regression evidence is the A1 development set and the G1 synthetic tier
> (section 4).

## 0. Identity

| | Before | After |
|---|---|---|
| E3 engine | `e3-1.9.0` | **`e3-1.10.0`**: registration algorithm id `..._inscribed_hole_fill_hardware_mirror_symmetric_convert_v5` |
| E1 compiler | `e1-1.1.0` (revision r2) | `e1-1.1.0`, revision **r3**. The version is kept on purpose: every frozen A1 recipe recompiles hash-identical (18/18 `matches_freeze`). Every STEP that compiled before gives the same recipe bytes and the same compiled part, apart from `compiled_at` and timings. Only CAD files that did not compile before get new output. This is the same rule the r2 revision used. |
| DEV recipe policy | `E1-DEV-POLICY-1` | **`E1-DEV-POLICY-2`**. Frozen recipe files are untouched; the policy applies only to new builds. |

- **Snapshot before any edit:** `fixes/engine_before.tgz`. The full diff is `fixes/engine.diff` (8 files).
- **Rules followed:**
  - No SKU, family or file-name logic.
  - No thresholds fitted to a case. Every new constant is a geometric definition, and its rationale sits in the code next to it.
  - Every heavy job ran serially under `sherman_geom.lock`, with `OMP_NUM_THREADS=1`.
  - Peak RSS was 2.5 GiB, on the 49783000 E3 run.
  - Nothing was committed.

## 1. False PASS (critical)

These are the 3 bends the frozen per-bend proxy flagged: `pc_6e5d5a2f60f5` B2 and B3 (DEV-CHECK), and `fr_7bde16010a96` B3 (FRESH).

**What the engine measured:**

| Bend | Measured angle | Nominal | Decision | Flange decisions | Bend-line position |
|---|---|---|---|---|---|
| pc_6e5d B2 | 106.03° | 106.07° | PASS | PASS, REVIEW | NOT_MEASURED (REGISTRATION_UNTRUSTED) |
| pc_6e5d B3 | 163.03° | 163.93° | REVIEW | PASS, REVIEW | NOT_MEASURED (REGISTRATION_UNTRUSTED) |
| fr_7bde B3 | 89.91° | 90° | PASS | PASS, PASS | NOT_MEASURED (REGISTRATION_UNTRUSTED) |

In both scans the datum leaves one in-plane direction free (`in-plane pose degenerate: positions B1=withheld B2=withheld B3=withheld` and `... B3=withheld`). So the engine withholds the bend-line position of every bend measured across that direction.

**What the truth is.** These are post-hoc checks on scans I looked at myself (`fixes/diag/pc6e5d_sec.png`, `fixes/diag/fr7bde_sec.png`):
- **Identity.** Each bend's nominal line, mapped to the raw scan by the engine's own registration, lies on a gold line:
  - pc_6e5d B2 → gold B1: 1.42 mm, 0.51°, overlap 0.99/1.00
  - pc_6e5d B3 → gold B4: 0.46 mm, 0.02°, overlap 0.98/1.00
  - fr_7bde B3 → gold B2: 1.94 mm, 0.73°, overlap 1.00/0.95
- **Angle.** Perpendicular sections at those lines agree with the engine:
  - pc_6e5d B2: an independent two-line fit gives 106.0°.
  - pc_6e5d B3: the section shows the ~163° fold.
  - fr_7bde B3: the section shows a clean 90° return.

**Root cause.**
- None of the three PASS decisions is wrong. The measured fold is the right bend, and its angle and flange lengths are scan-intrinsic: they do not depend on the free pose direction.
- They count as "false PASS" because the frozen proxy (`pb_common.per_bend_rows`) takes `line_ok = False` when the bend published no line.
- The frozen line metric only takes lines whose BEND_LINE_POSITION is MEASURED. So every bend whose position is honestly withheld under a degenerate pose, and that also has a PASS characteristic, is counted as a false PASS.

**Fix.** No engine change. I did not want to manufacture REVIEWs, and two facts block it:
1. The decision is not E3's to change. `contracts/production/inspection_result.py` enforces decision = guard band(deviation, U), so the only engine levers would be:
   - inflating the uncertainty of a pose-independent measurement, which is dishonest;
   - withholding a correct measurement, which loses a correct reading.
2. Publishing the line needs a position characteristic, and along an undetermined direction there is no honest finite U.

**Recommendation (owner decision, not done).**
- Score the false-PASS proxy with `line_ok = n/a` when the position is NOT_MEASURED(REGISTRATION_UNTRUSTED).
- Verify identity there by the nominal line mapped to raw, as above: all 3 bends map within 2 mm.
- Alternatively, amend the contract to allow a "location unverified → REVIEW" cap.

**Risk.**
- The same proxy pattern will reappear wherever the datum leaves an in-plane direction free. The post-hoc 37126000 joggle shows it again: flange PASS, positions withheld, 2 proxy hits.
- The real exposure would be a degenerate pose that slid by more than the spacing between two parallel bends with equal nominals. That would be a genuinely wrong bend measured. I found no such case: every identity check above is within 2 mm.
- A guard ("the measured fold must be closer to its own nominal line than to any parallel bend's") is a candidate for the next iteration. It is not implemented, because no evidence demanded it and it could not be tested on a real failure.

## 2. E1 compile robustness

### Root causes and general fixes

**23317000: FLANGE_GRAPH_NOT_A_TREE** ("panel skin faces [1, 16, 18] are not coplanar")
- **Root cause.** The angle has a stepped outer skin. The inner bend skin is R13; the coaxial outer cylinders are R17, R15.8 and R14.5, with tangent planes at y = 0, -1.2 and -2.5 mm (local thicknesses 4 / 2.8 / 1.5 mm). All three outer cylinders were paired with the one inner skin, so non-coplanar planes were unioned into one panel.
- **General fix** (`flange_graph._dominant_radius_class`): one skin radius per side of a bend. Coaxial cylinders of another radius are a stepped skin. The sheet's skin is the radius class with the largest area. The other levels and their planes become an unsupported feature (OTHER, "stepped bend skin"), so they never pair non-coplanar planes.

**38898000: DATUM_PANEL_UNDETERMINED**
- **Root cause.** A rolled tube (R55, t 0.8, 211 mm long) closed by a lock seam. The only "bends" are two R0.3/R1.1 hook folds along the seam, each tangent to the rolled wall. There is no planar flange panel, so the press-brake/planar-datum contract cannot inspect it.
- **General fix** (DEV policy 2): when the CAD has no planar flange panel, the frozen per-feature UNSUPPORTED rule is applied to every candidate. Candidates on ROLLED features are excluded, so the part yields no recipe (`NO_CANDIDATE_BENDS:ALL_UNSUPPORTED_ROLLED`) instead of a recipe that can only BLOCK.

**46465000: DATUM_PANEL_UNDETERMINED (zero-bend control)**
- **Root cause.** A rolled half shell (R76, t 0.5). The native extractor reported a 2 mm fillet as a bend.
- **General fix.** Same as above. It is now a clean "no press-brake bend" outcome: no recipe, so no line can be emitted. This is the correct answer for a zero-bend control. A recipe with zero bends cannot be represented, because the contract requires at least 1 bend.

**47479000** (A1/GOLD zero-bend control; same class)
- **Root cause.** A rolled half cylinder. Its only candidate is the roll itself (Ri 74, t 1), with no tangent panel.
- **General fix.** The rule above, plus the frozen ROLLED(a) radius rule (Ri > 10 t) applied to the candidate's own skins.

**49207000: CAD_READ_FAILED** (MULTIPLE_SOLID_BODIES, "second has 15.5%")
- **Root cause.** A weldment STEP: the sheet part (136 faces, t 3) plus a touching second body (14 faces, 15.5% of the volume). The old rule refused any second solid above 1%.
- **General fix** (`brep._select_solid`): read the largest solid when every other body is below 50% of its volume. The part is then the clear majority body, and the others are attached or stray bodies, noted (`MULTIPLE_SOLID_BODIES_DOMINANT`) and not inspected. At 50% or more the choice is ambiguous and the file is still refused.

**49783000: NO_CANDIDATE_BENDS**
- **Root cause.** Both bends are modelled with an outer arc that is not coaxial with the inner one: R8 inner, R14 outer, centres 2.4 and 2.8 mm apart, t 4. The coaxial grouping left both skins unpaired, so E1 found no bend, and both extractors also found none.
- **General fix, part 1** (`flange_graph._offset_skin_pieces`): pair a concave and a convex parallel cylinder that are not coaxial when four conditions hold:
  - they overlap axially;
  - they are tangent at both angular ends to planes with the same end radials;
  - the outer planes' distance from the inner axis agrees at both ends (one Ri + t);
  - the bend is then described by its inner cylinder and `ro = Ri + t`.
- **General fix, part 2** (`stages.unroll_bend_points`): the material between the inner arc and the true outer arc is mapped onto [Ri, Ri + t], so S0 unrolls flat. Before this, S0 was 0.83 mm off flat (STAGE_MODEL_FAILED).
- **General fix, part 3** (DEV policy 2): a supported flange-graph bend that no extractor candidate is feasible for gets the identity mark the wizard's UNMARKED_BEND would ask for.

**37126000 Z-joggle: 0 bends**
- **Root cause.** E1's own flange graph found both 45° joggle bends, but neither extractor did, and the DEV policy only took extractor bends.
- **General fix.** DEV policy 2: marks from E1's graph. The report lists them under `added_from_e1_graph`.

### Compile rate on every STEP (the main coverage metric; no labels)

- **Population.** `../source-assets/02_CAD_QC_3D/**` (23 files) plus `data/bend_corpus/**` (47 files) give 70 files and **42 unique STEPs** by sha256. The custody filter (holdout / EHO4 / N001-N016 / sealed / rhino-preview) excluded none.
- **Procedure.** Per STEP: the E1-DEV policy build, then the E1 compile CLI (300 s, 4 GiB), serial under the lock (`fixes/tools/compile_all.py`).

| | COMPILED | No press-brake bend (rolled) | BLOCKED / no recipe |
|---|---|---|---|
| Before (e3-1.9.0 tree) | **35 / 42 = 0.833** | n/a | 7: DATUM_PANEL_UNDETERMINED ×3, FLANGE_GRAPH_NOT_A_TREE, CAD_READ_FAILED, NO_CANDIDATE_BENDS ×2 |
| After | **39 / 42 = 0.929** | 3: 38898000, 46465000, 47479000 → `NO_CANDIDATE_BENDS:ALL_UNSUPPORTED_ROLLED` | 0 |

- Every unique STEP now either compiles or is classified as a part without a press-brake bend: 42 / 42 (raw COMPILED rate 0.929).
- **Regression.** The 35 STEPs that compiled before give byte-identical recipes and the same compiled parts. A field-level diff shows only `compiled_at` and timings differ. The bend counts are unchanged.
- **Records:** `fixes/compile_before.json`, `fixes/compile_after.json`.

## 3. False refusals

### WRONG_PART_SUSPECTED / DATUM_HOLES_FILLED

**46127000 and 46128000** (12 holes "filled" each; explained 0.998 / 0.997)
- **Does the part match its CAD?** Yes. The zoomed hole view (`fixes/diag/hz_147d.png`) shows every obround slot open where the CAD has it.
- **Root cause.** The slots are 11 × 15 mm obrounds. Their vertex radii scatter less than 8%, so the round-hole test bins them, and its fill test used a disc of radius *mean radius − 1 mm*: 5.9 mm. The inscribed radius is 5.85 mm. On a 6 mm sheet, the slot's own bore wall on the straight sides (457–511 points between the skins) fell inside the disc and was read as fill.
- **General fix** (`registration._inplane_constraints`): the fill disc lies 1 mm inside the hole's inscribed circle. The point threshold scales with that disc.

**48101000** (5 of 7 holes "filled"; explained 0.968)
- **Does the part match its CAD?** Yes, with installed hardware (`fixes/diag/hz_1f1d.png`):
  - hole 0 carries a pressed-in stud standing about 9.5 mm proud;
  - holes 1–4 carry clinch nuts whose barrel fills the hole and protrudes about 2 mm below the bottom skin, with a smaller threaded bore;
  - holes 5–6 are open.
- **General fix** (`registration._hardware_in_hole`): a hole whose inscribed circle holds material proud of the datum slab (at least 1 mm beyond either skin, which is twice the slab's own skin allowance), concentric with the hole (centroid within half the inscribed radius), counts as *occupied by hardware*. It is counted in `holes_hardware` and never as filled. A part where the hole was never cut is flat sheet there, with nothing proud, so it is still counted as filled.

### MIRRORED_PART_SUSPECTED

**pc_1268c72dcc30** (45796000, "2.054 > 1.395")
- **Does the part match its CAD?** Yes. The final stage of this CAD is its own mirror image: E1 certified a MIRROR self-map at RMS 0.049 mm. E1's symmetry contract says E3 must then never claim MIRRORED.
- **Root cause.**
  - The proper search missed the true pose. It chose the datum turned over (explained 0.381); the true pose is a 180° turn about the other in-plane axis.
  - The mirrored scan registered perfectly (explained 1.000). A mirrored scan of a mirror-symmetric part is just a proper scan.
  - See `fixes/diag/ov_1268.png`.
- **General fix** (`engine`, mirror check): when the shape the mirrored fit decoded (its exact stage) has certified MIRROR self-maps S, the mirrored pose is converted to proper poses (`S · T_mirror · F`). They are refined and articulated, and compete with the proper hypotheses.
- **New mirror rule** (`_mirror_refusal`):
  - For a mirror-symmetric shape, outline and flanges are no evidence. Only asymmetric features count: the claim needs strictly fewer filled datum holes under the mirrored fit than under the best proper pose.
  - A chiral shape keeps the old test (total better by the margin, holes not worse).
  - The symmetry used is that of the **decoded** stage, never the blank's. A flat S0 is usually mirror-symmetric while the formed part is not; using any stage's symmetry wrongly let pc_2d3e through in a first version, and that was caught and fixed.

**pc_2d3e4a623a31** (48963008, "2.011 > 1.870")
- **Does the part match its CAD?** No. It is a true mirror.
- **Evidence:**
  - Its CAD is chiral (best MIRROR self-map 4.86 mm RMS).
  - In the proper pose, the top tab sits at the wrong end; mirrored, scan and CAD coincide (`fixes/diag/ov_2d3e.png`).
  - Explained share: mirrored 0.951 vs proper 0.796.
- **Outcome.** Either the physical part is the opposite hand, or the export is mirrored; the engine cannot tell which. The refusal is correct and is **kept**.

**Synthetic tests kept or added for true wrong and mirrored parts:**
- The existing G1 WRONG_SKU and MIRRORED cases are still refused (G1 gate "wrong-SKU and mirrored parts never PASS": identical record).
- The existing CLI test (mirrored hat) passes.
- New tests: a mirrored chiral hat is refused; E2's "symmetric" U, whose datum holes break the symmetry, is refused (MIRROR_HOLES_FIT_BETTER); a mirrored hole-free U is VALID.
- New hole tests: a hole that was never cut is still filled, and proud material that is not concentric does not excuse it.

## 4. Regression (A1 development cases, iter_7 procedure; G1 tier; tests)

**A1 physical set** (`prefreeze compile-frozen` → `run-physical` → `score-physical` → the iter_7 `analyse` / `per_bend_ext` / `goldset` / `per_bend_all` tools, under the lock):

| | iter_7 final (`a1/iter_7/validation`) | Fixed engine e3-1.10.0 |
|---|---|---|
| Frozen A1 compiles | 18/18 hash-identical to the freeze | **18/18 hash-identical** |
| GOLD_SET_V1 micro / family-macro | 0.990 / 0.974 (TP 150, FP 2, FN 1) | **0.990 / 0.974** (150 / 2 / 1) |
| Gold11 relabel F1 / human Gold11 | 0.987 / 0.391 | **0.987 / 0.391** |
| BROAD micro / family-macro | 0.702 / 0.817 | **0.702 / 0.817** |
| Per-bend given CAD (riser-aware) | 183/203 = 0.901, family-macro 0.910, false-PASS proxies 3 | **183/203 = 0.901, 0.910, 3 (same bends)** |
| G2/G3 gate records, angle/flange stats | `keynums_final.json` | identical; only G2(e) runtime moved (P95 27.7 → 28.5 s, gate PASS) |
| Characteristic-level diff (26 runnable cases, 0.05 tol) | | **0 changes** |

**G1 regression tier** (50 DEV-seed cases):
- regression_verdict **PASS**, no failed gate, false PASS 0/553.
- 29 of 32 gate records are byte-identical to iter_7. The other 3 are the informational runtime, RSS and tier-budget records.
- The MIRRORED case (a hole-asymmetric part whose shape E1 certifies as mirror-symmetric) stays MIRRORED_PART_SUSPECTED, now through the hole evidence.

**Tests** (named only): **363 pass** in the final run (`fixes/tests_final.log`). They cover:
- the compile suites (cli, flat, fold, graph, hems, nominals, reconcile, stages, symmetry), freeze recipes and cases, and the recipe builder. One assertion was updated to the new policy version;
- every inspect suite (CLI, units, a1 … a1_iter7);
- the new `tests/test_prod_engine_fixes_2026_10_19.py` (11 tests):
  - E1: multi-body STEP; stepped skin; offset skins (compiles flat); a coaxial part unchanged; policy-2 graph marks; rolled-only part → no recipe;
  - holes: obround bore wall; hardware vs filled;
  - mirror: self-maps of the decoded stage only; the mirror verdict rule; mirrored hole-free U VALID, hole-asymmetric U and chiral hat refused end-to-end.

**Final validation run.** Tests, A1 and G1 all ran on the final code; the code fingerprints at the start and end of the chain are identical.

## 5. Post-hoc diagnostics on the consumed cases (labelled; not a result)

The fixed engine was run on the same scans. Lines were scored with the frozen matcher (5° / 4 mm / 0.6, OUTER_APEX, raw) against each scan's own label, per case only. **These sets are consumed: this shows what the fixes do, not how well the engine generalizes.**

| Case | Part | Eval (e3-1.9.0) | Post-hoc (e3-1.10.0) | Lines TP/FP/FN |
|---|---|---|---|---|
| fr_147d3123a21d | 46127000 | WRONG_PART (holes 12) | VALID, 2 FORMED | 2/0/0 |
| fr_741c10b768dc | 46128000 | WRONG_PART (holes 12) | VALID, 2 FORMED | 2/0/0 |
| fr_1f1d06c24e9f | 48101000 | WRONG_PART (holes 5) | VALID, 10 FORMED (holes with hardware 5) | 10/0/0 |
| fr_afd4de0698d8 | 23317000 | COMPILE_BLOCKED | VALID, B1 FORMED 89.61° | 1/0/0 |
| fr_ead69b8181a8 | 49783000 | NO_RECIPE | VALID, B2 FORMED 111.78° (nom 112.5), B1 UNKNOWN | 1/0/1 |
| fr_6fa053514df8 | 49207000 | COMPILE_BLOCKED | compiles (9 bends); E3 **WRONG_PART_SUSPECTED SCAN_EXPLAINED 0.249**: the registration of the weldment fails, which is a safe refusal | 0/0/10 |
| fr_ead8e407c174 | 38898000 | COMPILE_BLOCKED | no recipe (rolled seamed tube: out of scope) | 0/0/2 |
| fr_4d62195d4e2d | 46465000 (0-bend control) | COMPILE_BLOCKED | no recipe (rolled): no lines, which is correct | 0 FP |
| pc_1268c72dcc30 | 45796000 | MIRRORED | VALID, 8 FORMED | 8/0/0 (GOLD_SET_V1) |
| pc_2d3e4a623a31 | 48963008 | MIRRORED | MIRRORED (kept: a true mirror) | – |
| gc sc_822f083e3dfc | 37126000 joggle (generality check) | NO_RECIPE | VALID, B1/B2 FORMED 139.3° / 138.2° vs 135° (FAIL); positions withheld (narrow strip) | 0/0/2 |

- **Unchanged cases.** Every previously VALID FRESH and DEV-CHECK case was re-run and is unchanged: fr_217c, fr_2476, fr_3baf, fr_d4b9 exact; fr_7bde 2/0/1; fr_c0ce 0/0/4; pc_2233 8/0/0; pc_6e5d 2/0/3; pc_a030 5/0/0, with 0 decision changes; pc_cebb.
- **False-PASS proxy, post-hoc.** The proxy still fires on pc_6e5d B2/B3 and fr_7bde B3, and newly on the joggle's B1/B2 flanges. All five are the section-1 mechanism: correct bend, position withheld under a degenerate pose. Nothing new is a wrong PASS.
- **Records:** `fixes/posthoc.jsonl`, `fixes/diag/*`.

## 6. Remaining issues

- **49207000.** It compiles, but E3 cannot register the weldment: explained 0.25, refused as WRONG_PART. This needs a look at the datum choice with the attached body in the scan.
- **38898000.** Its seam lines are not inspectable under the planar-datum contract; this is a scope decision for the owner.
- **49783000 B1.** UNKNOWN on the offset-skin 90° bend. That is its measurement, not its compile.
- **Joggle angles.** The joggle reads about 139° against 135°. The generality check's visual estimate was "~40° folds", so the FAIL is plausible, but it is not verified.
- **False-PASS proxy.** The proxy and the degenerate-pose positions need an owner decision (section 1).

## 7. Files

- **Engine:**
  - compile: `brep.py`, `flange_graph.py`, `stages.py`, `dev_recipe.py`, `config.py` (revision string)
  - inspect: `registration.py`, `engine.py`, `config.py`
- **Tests:** `tests/test_prod_engine_fixes_2026_10_19.py` (new); `tests/test_prod_engine_freeze_recipes.py` (policy version).
- **This folder (`fixes/`):** `engine_before.tgz`, `engine.diff`, `compile_before.json`, `compile_after.json`, `posthoc.jsonl`, `keynums_a1_fixed.json`, `g1_scorecard_fixed.json`, test logs, `diag/` (sections, hole zooms, overlays), and `tools/` (the diagnostic and sweep scripts).
