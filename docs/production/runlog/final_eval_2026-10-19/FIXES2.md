# Fixes, round 2 (2026-10-19): scoring amendment V2 and coverage work

> **Engine unchanged.** The E3 engine is still `e3-1.10.0`. Every engine edit tried in this round (49207000) was reverted, because it did not fix the case and it slowed the run. The code fingerprints equal the last validated chain (`fixes/a1` f5; see `fixes2/code_fingerprints_now.json`). That is why the A1 and G1 records of round 1 stand without a rerun.
>
> **Custody.** No DEV-CHECK or FRESH run was made, and no RUN was added to the ledger. No `*_SEALED*` path was opened. The FRESH gold labels (`ai_bend_lines_fresh_v1_SEALED/`) and every DEV-CHECK input (`devcheck_SEALED.json`, `.../devcheck_SEALED/`) carry that name. Because of this, section 1.4 can only re-use per-bend identity results already recorded in round 1, and it says so.

- **Snapshot:** `fixes2/engine_before.tgz`, taken before any edit. The engine tree now equals it.
- **Heavy jobs:** serial under `sherman_geom.lock`, with `OMP_NUM_THREADS=1`. Peak RSS was 0.92 GiB.
- **Tests run** (named only):
  - `tests/test_prod_per_bend_scoring_v2.py`: 7 passed (new).
  - `tests/test_prod_engine_fixes_2026_10_19.py`: 11 passed.

## 1. Task A: scoring definition V2 (owner decision, option 1)

### 1.1 What changed

- **New, versioned definition `PER_BEND_GIVEN_CAD_V2`:**
  - rules: `scripts/production/per_bend_scoring_v2.py`
  - A1 runner: `fixes2/tools/per_bend_all_v2.py`
  - single-scan post-hoc runner: `fixes2/tools/posthoc_v2.py`
- **V1 is untouched and still runs.** That is `a1/iter_7/measure/tools/per_bend_all.py` and `tools/pb_common.py`. It was re-run in this round and gives 183/203, 170 strict, 0.910, 3 proxies.

The V2 rules:
- **Position withheld.** This means `BEND_LINE_POSITION` is `NOT_MEASURED(REGISTRATION_UNTRUSTED)`, which is exactly what the contract mandates for a degenerate in-plane pose. The line criterion is then **not applicable**: it is neither a line miss nor a false PASS.
- **Identity.** The recipe's CAD nominal line is mapped to the raw scan frame through E3's own `raw_to_frame`. The line used is E1 `loci.outer_apex` of the decoded stage. It is matched with the frozen matcher (5° / 4 mm / 0.6, Hungarian, apex locus, raw frame), and only against gold lines **not already taken by a published line**, so each gold line is used once.
- **Fallback.** No match, or no nominal line, means a wrong bend. A PASS on it stays a false PASS.
- **Unchanged.**
  - Published lines are matched first, alone, exactly as in V1, so their results are identical by construction.
  - Bends expected NOT_FORMED are unchanged.
- **Ledger.** `freeze/LEDGER.json` seq 6, `frz_68d9dc5274a9`, kind NOTE, `note_type: AMENDMENT`. It records the owner decision, the date 2026-10-19, the rule, and the sha256 of the new files:
  - `per_bend_scoring_v2.py`: `2c231ec7…586f`
  - `per_bend_all_v2.py`: `6608c425…1767`
  - `posthoc_v2.py`: `90c4a0ea…8bf3`
  - the test: `b308d6ef…270`

  It also records the sha256 of the two V1 files it keeps, and the A1 rescore. `verify_ledger` passes: 6 entries, chain intact.

### 1.2 Tests (`tests/test_prod_per_bend_scoring_v2.py`, 7 pass)

The tests use the frozen `line_metric.match_lines`. They cover:
- withheld + correct identity → no false PASS, and the bend counts as correct;
- withheld + wrong identity (nominal half-way between two gold lines) → false PASS;
- withheld + no nominal line → false PASS (fallback);
- a gold line already claimed by a published line cannot confirm a withheld bend's identity;
- published lines: `line_ok` and the false-PASS proxy equal V1. This includes an unpublished bend whose position is NOT_MEASURED for another reason (`EDGE_NOT_SEEN`), which stays a miss;
- expected NOT_FORMED is unchanged;
- "withheld" requires the `REGISTRATION_UNTRUSTED` reason.

### 1.3 Re-report on V2

| | V1 | V2 |
|---|---|---|
| **A1 per-bend** (iter_7 procedure, e3-1.10.0 run f5) | 183/203 = 0.901 riser-aware, 170 strict, family-macro 0.910, false-PASS proxy 3 | **183/203 = 0.901, 170, 0.910, 3**: the same rows. No A1 bend has a withheld position (`fixes2/per_bend_all_v2_a1.json`, `changed_rows: []`). |
| A1 GOLD_SET_V1 / Gold11 relabel / BROAD | 0.990 / 0.974; 0.987; 0.702 / 0.817 | same. V2 does not touch the line metrics, and the engine is unchanged. |
| **G1** | PASS, false PASS 0/553 | **PASS, 0/553**. G1.FALSE_PASS is characteristic-level against E2 truth, with no line criterion. No G1 bend (0/135) has a withheld position, so V2 = V1 on G1. |

The 3 remaining A1 proxies are **not** a withheld-line effect, so V2 rightly keeps them:
- **pc_07062eee17aa B1.** It published a line, and the line matches no gold.
- **pc_cec08dfca9ab B6 and B11.** The truth is NOT_FORMED, but the engine says FORMED / UNEXPECTED_FORMED with a PASS.

### 1.4 DIAGNOSTIC: consumed sets, not a held-out score

What each row is based on:
- **Engine.** The e3-1.10.0 post-hoc results from round 1 (`fixes/posthoc.jsonl`).
- **Withheld flags.** Read from those `result.json` files.
- **Identity.** The **recorded** round-1 measurement of each nominal line, mapped to raw against the nearest gold line (FIXES.md §1).
- **Not re-run with the V2 one-to-one script.** That would need the `*_SEALED*` labels.
- **Generality case.** The 37126000 joggle is not DEV-CHECK or FRESH. Its label is non-sealed (`ai_bend_lines_gold_v1`), so V2 was run on it in full.

| Case (set) | V1 false-PASS proxy | Withheld + PASS | Identity (mapped CAD nominal → gold) | V2 false PASS |
|---|---|---|---|---|
| pc_6e5d5a2f60f5 (DEV-CHECK) | B2, B3 | B2, B3 (B1 withheld, no PASS) | B2→gold B1 1.42 mm / 0.51° / 0.99,1.00; B3→gold B4 0.46 mm / 0.02° / 0.98,1.00 (recorded) | **0** |
| fr_7bde16010a96 (FRESH) | B3 | B3 | B3→gold B2 1.94 mm / 0.73° / 1.00,0.95 (recorded) | **0** |
| other FRESH cases run (11) | none | none | n/a | 0 |
| gc_sc_822f083e3dfc, 37126000 joggle (generality, dev) | B1, B2 | B1, B2 | **computed by V2:** B1→B1 0.83 mm / 0.19°; B2→B2 0.24 mm / 0.26° | **0**; per-bend 0/2 → 2/2 |

- **Per-bend with these identities** (eval engine e3-1.9.0, engine-ran view; DIAGNOSTIC):
  - DEV-CHECK: 14/25 → 16/25. pc_6e5d B1 is left incorrect, because its identity was never recorded.
  - FRESH: 29/38 → 30/38.
  - False PASS: 2 → 0 on DEV-CHECK and 1 → 0 on FRESH.
- **Not computable here.** The full one-to-one V2 rescore of these sets needs the sealed labels. If the owner wants it, reading the consumed FRESH/DEV-CHECK labels must be authorized explicitly (see section 4).

## 2. Task B: coverage

### 2.1 49207000 weldment: root cause found; no safe fix yet (engine reverted)

**Facts:**
- The part matches its CAD. An independent check (PCA seeds plus ICP of the CAD final stage, `tools/icp_pose.py`, `diag/icp49207.png`) puts CAD and scan together, apart from the welded plate.
- Seeded at that pose, E3's own refine + articulated fit explains **0.906** of the scan, with all 9 bends FORMED and total 1.454. The engine's chosen pose explains 0.249 (total 0.717).
- So the refusal is a **registration-search failure**, not the explained-share metric and not a multi-body CAD problem.

**Root cause.** These are the four general mechanisms, traced with `tools/regdiag*.py`, `seeddiag.py`, `seed2.py` and `violdiag.py`:

1. **Seeding crowd-out.** The datum P4 (210 × 249 mm) has no usable free edges and no holes. Its outline correlates on the raster about as well with the two large side walls (segments 0 and 1) as with its own plane. The welded plate also hides part of the datum skin: segment 2 is 29k mm² against 52k mm². As a result:
   - all 14 global coarse slots go to the walls;
   - the true pose is the **best peak of the datum's own plane**, but ranks 205th overall, and never gets a fine search.
2. **Free-edge band normalisation.** The datum's only free edges are three 2.8 mm × 0.4 mm marks on a bend line. Their forbidden band (39 cells) lies in the bend zone of the part itself. The violation term is normalised by the forbidden area, so 9 hit cells give violation 0.23, which turns datum score 0.51 into 0.05 at the true pose.
3. **Undetermined in-plane pose.** With no free edges and no holes, all three in-plane directions are undetermined. `refine_datum` never moves along them, so the in-plane pose is whatever the raster search found. That search ran at the hypothesis' guessed skin height, and it is biased by the occluded datum. It lands about 5 to 11 mm off, and the articulated fit there explains only 0.3 to 0.5.
4. **Symmetric datum.** The bare rectangle is 2-fold symmetric. The 180° image of the true pose ties on every datum score, and only the flanges can tell them apart.

**What was tried.** The diffs are in `fixes2/attempt_49207/`; all are general, with no part logic:
- (a) every candidate plane's best peak gets a fine search, besides the global top 14;
- (b) a free-edge run whose outward strip is mostly the part's own S0/SN material forbids nothing;
- (c) a fit with undetermined directions is searched again after the plane refinement;
- (d) the symmetric images of each refined fit are refined too.

The results: refused at 0.249 → 0.514 → 0.427, never the true pose. E3 wall time went from 25 s to 62 s on this scan. **All of it was reverted.** It does not resolve the case, it adds runtime on every part, and (b)/(c) change the kernel and fits for other parts.

**What is still missing.** For a datum that fixes none of its in-plane directions, the in-plane pose must come from the flanges for **each** hypothesis. Today the flange-ends step (`_refine_free_directions`) runs only on the finally chosen pose. That is a larger change to the ranking. It needs its own A1, G1 and runtime validation, and it is listed under owner decisions.

**Wrong parts.** Wrong parts stay refused. No engine behaviour changed: the G1 WRONG_SKU and MIRRORED gate records and the round-1 tests still hold (11/11 pass).

### 2.2 Joggle 49783000 / 37126000 and the shared-angle-bias question

- **37126000 (the measured joggle).** This is an independent cross-section of the registered scan (z band 1.0–6.2 mm of the 7.2 mm strip, PCA line fit per skin; `diag/joggle37126_section_xy.png`, `angle_bias_check.json`):
  - The flats lie at 0.00° / 0.04° (P0) and −0.08° / −0.17° (P2).
  - The riser is straight, at **41.2° (lower skin) and 41.6° (upper skin)**, with fit RMS ≤ 0.008 mm.
  - So both folds are about 41.5° against a nominal 45°, which is an included angle of about **138.5° against 135°**.
  - The engine reads 139.3° / 138.2°, within 0.8°.
  - **The FAIL is real.** There is no engine bias.
- **49783000.** Its CAD calls the bends 112.5° / 90°, not 135°: it is not a joggle. B2 reads 111.78° against 112.5° (FORMED). B1 is UNKNOWN (OCCLUDED: the 20 mm flange on 4 mm stock, with an R14 outer arc that is not coaxial with the inner one, was not found by the articulated fit). This costs 1 bend. No general fix was attempted.
- **Shared angle bias.** These are signed deviations over all 127 measured FORMED angles in A1 (`angle_bias_check.json`):
  - The 90° readings depend on the part: −1.06° on 38172000 and −1.18° on 10839000, but −0.01° on 48811000, −0.06° on 37361005, +0.12° on 15651004 and −0.06° on 23553000.
  - The same 38172000 B5 reads from −1.3° to +4.7° on different scans.
  - Iteration 7's skin profiles already showed that the 10839000 FULL 88.0–89.7° readings are the part's geometry (overbent lips, a bowed base), and that scan_2 of the same part reads nominal.
  - **Conclusion: there is no shared systematic angle bias, so none was "fixed".**
- **Joggle short flanges UNKNOWN (10839000 E2E).** This is by design. A riser with no flat material is placed by its far flange's step under the riser's nominal length (`metrology.py`, `UNSUPPORTED_FEATURE`: "its own length is not measured"). Its length would be an assumption, not a measurement. The per-bend riser-aware metric already accepts it; the strict metric counts it, which is the 13-bend gap between 170 and 183.
  - **How to measure it:** measure the riser directly, from the S slope at its inflection and its two sharps.
  - **Evidence so far:** the 37126000 section shows such a riser is measurable when it has a straight middle. That is a second joggle family, which iteration 7 lacked.
  - **What it still needs:** validation on a riser **without** flat material. That is a feature, not a fix, so it is listed in section 4.

### 2.3 38898000 seam lines: scope decision with options (nothing built)

- **The part.** A rolled tube (R55, t 0.8, 211 mm) closed by a lock seam. Its only folds are two hook folds (R0.3 / R1.1) along the seam, tangent to the rolled wall. There is no planar panel, so the planar-datum contract (datum panel, flange lengths, OML sharps) does not apply.
- **Today.** E1 says `NO_CANDIDATE_BENDS:ALL_UNSUPPORTED_ROLLED`. The part is a clean "no press-brake bend" with no recipe, and its 2 gold seam lines count as misses.
- **In the corpus.** 3 of 42 unique STEPs are rolled (38898000, 46465000, 47479000), and only 38898000 has seam folds.

Options:
- **A. Keep it out of scope (recommended for now).** Report it as "rolled / seamed part: not inspectable by the press-brake engine". No line is emitted, and the misses stay visible in coverage. Cost: none.
- **B. Seam locator only.** A cylinder datum (axis + radius fit) in E1 and E3, and seam-fold lines published as positions only, with every characteristic UNKNOWN. This needs a contract amendment (a datum kind other than a panel) and a new registration path. Cost: medium (days). Risk: a new registration family with no validation set (one part).
- **C. Rolled-part metrology.** Roll radius, seam hook width and overlap. This is new product scope with a new characteristic vocabulary and contract. Cost: large.

### 2.4 Coverage losses on allowed sets, ranked by bends lost (labels not needed)

| # | Loss | Bends | Class | Status |
|---|---|---|---|---|
| 1 | 49207000 weldment refused (FRESH, consumed) | 9 compiled (10 gold) | registration search (2.1) | root cause found; fix needs the per-hypothesis flange-ends step (owner) |
| 2 | 38968000 AMBIGUOUS_POSE (FRESH, consumed): two E1-certified symmetric twins tie (total 1.532 vs 1.532, holes 6/0 both) | 6 UNKNOWN (+2 NOT_YET_FORMED) | contract: an ambiguous pose measures nothing | owner/contract decision (section 4) |
| 3 | A1 flanges `ADJACENT_BEND_NOT_MEASURED`: **all 11** have their adjacent bend NOT_YET_FORMED (staged scans) | 5 FORMED-truth bends counted incorrect (08686 B2/B4, 8dfd B1, 927a B1, cec08 B2) | contract: FLANGE_LENGTH may not be NOT_APPLICABLE_AT_STAGE | owner/contract or scoring decision |
| 4 | Joggle riser flanges UNSUPPORTED (10839000) | 13 strict-only (riser-aware already counts them) | feature (2.2) | owner |
| 5 | 50757000 `NOMINAL_UNAVAILABLE`: far boundary half adjacent-bend / half free edge, or a stepped edge with no dominant level | 4 (752e B1/B4, 777c B1/B4) | E1 nominal definition; changing it changes frozen compiled parts | owner (versioned E1 change) |
| 6 | 38898000 seam | 2 | scope (2.3) | owner |
| 7 | 49783000 B1 OCCLUDED | 1 | articulated fit on a thick, short, offset-skin flange | not attempted |

I found **no general fix that is safe and within the rules for the top items.** The A1 dev numbers and G1 are therefore unchanged. Items 2, 3, 5 and 6 are bounded by the contract or the frozen E1 definitions, not by engine bugs.

## 3. Before / after

| | Before (round 1) | After (this round) |
|---|---|---|
| E3 / E1 | e3-1.10.0 / e1-1.1.0 r3 | **unchanged** (fingerprints equal; no version bump needed) |
| Frozen A1 recompile | 18/18 hash-identical | unchanged (no compile code touched) |
| GOLD_SET_V1 | 0.990 / 0.974 | 0.990 / 0.974 |
| Gold11 relabel | 0.987 | 0.987 |
| BROAD | 0.702 / 0.817 | 0.702 / 0.817 |
| Per-bend A1 (V1 → V2) | 183/203, 3 proxies | 183/203, 3 proxies (V2 = V1 on A1) |
| G1 | PASS, 0/553 false PASS | PASS, 0/553 (V2 = V1) |
| False PASS, consumed sets (DIAGNOSTIC) | 3 (+2 on the 37126000 generality case) | 0 under V2 (recorded identities; 37126000 computed) |

## 4. Owner decisions needed

1. **Consumed sealed labels.** Do you allow reading the consumed FRESH/DEV-CHECK labels (`*_SEALED*` paths) for post-hoc diagnostics only? Without that, the V2 diagnostic in 1.4 relies on identities recorded in round 1.
2. **49207000-class datums** (no free edges, no holes). Approve a ranking change: run the flange-ends free-direction fit on every hypothesis whose datum leaves directions undetermined, together with the seeding, band and symmetry pieces (`attempt_49207/`). Expect a measurable runtime cost; it needs A1, G1 and runtime revalidation.
3. **Symmetric-twin ambiguity (38968000).** Should the result publish the readings under one labelling, with an identity-ambiguous flag, when twins differ only by an E1-certified symmetry? That is a contract amendment.
4. **Flange length to a not-yet-formed adjacent bend.** Choose one:
   - allow `NOT_APPLICABLE_AT_STAGE` for FLANGE_LENGTH in the contract; or
   - score it as n/a in the per-bend metric.

   Either one would turn the 5 A1 bends correct (183 → 188/203 = 0.926).
5. **Joggle riser length.** Develop a direct riser measurement (a feature), or keep riser-aware scoring.
6. **38898000.** Choose option A, B or C (section 2.3).

## 5. Files (`fixes2/`)

- `engine_before.tgz`: the snapshot.
- `code_fingerprints_now.json`: equals `fixes/a1` f5.
- `per_bend_all_v2_a1.json`: A1 V1 and V2 summaries, rows and changed rows.
- `posthoc_v2_joggle_37126000.json`.
- `angle_bias_check.json`: per-part signed deviations and the joggle section fits.
- `attempt_49207/`: the reverted diffs and the engine log of the last try.
- `diag/`: `joggle37126_section_xy.png`, `v49207.png` (scan vs CAD views), `icp49207.png` (ICP overlay coloured by distance), `datum49207.png` (datum at the true pose).
- `tools/`: `per_bend_all_v2.py`, `posthoc_v2.py`, `icp_pose.py`, `icp_view.py`, `regdiag.py`, `regdiag2.py`, `seeddiag.py`, `seed2.py`, `violdiag.py`, `views.py`.
- **Outside this folder:** `scripts/production/per_bend_scoring_v2.py`, `tests/test_prod_per_bend_scoring_v2.py`, and `freeze/LEDGER.json` seq 6.
