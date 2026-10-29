# A1 iteration 0: baseline of the production engine as built (2026-10-18)

The production engine is measured exactly as waves 1-3 built it: `e3-1.1.0`, default config
(`config_sha256 960f5f3e…`), no changes. All numbers come from consumed development scans (the
engineering ledger). None of them is unseen-family accuracy.

Machine-readable results are in `scorecard.json`. Raw scorer outputs are in `raw/`. The scripts that
derive the report-only numbers are in `tools/`. Overlays are in `overlays/` (indexed by
`overlays/INDEX.json`).

## Headline

The engine is very precise where it publishes a line, but it publishes very few lines. That makes it
worse than the best F1 lane (the bar) on every line metric.

On the BROAD scans it ran, the published lines are 24 of 26 correct (P 0.92). On the gold set the
precision is 26 of 26 (P 1.00). Recall is 0.31, against 0.56 for the best lane on the same scans.

Every lost line traces to one of three engine refusals, plus one process gate:

1. **The amendment-7 gate (process, not the engine).** E3 is refused on 13 of 26 BROAD cases and on
   5 of 9 G2(a) Gold11 cases. The G3 visual truth table is not frozen or recorded yet
   (`g3_truth_material/truth_table_FROZEN.json` is missing). The frozen scorer counts those cases
   as misses.
2. **False `WRONG_PART_SUSPECTED` (`DATUM_HOLES_FILLED:1-2`).** This hits 5 of the 13 scans that
   ran: 38172000 X and V, 37361005, 49024000 and 34499000. The whole inspection is INVALID and no
   line is published, which loses 30 BROAD gold lines. The datum-hole identity check fires on real
   Zeiss exports.
3. **`REGISTRATION_UNTRUSTED`.** The engine sets `BEND_LINE_POSITION NOT_MEASURED`, so no line is
   published even though the bend is FORMED and its angle and flanges are MEASURED. This affects
   15651004 (3 bends) and 49204020 (12 bends), and loses 15 gold lines. These bends still receive
   angle and flange PASS decisions.
4. **50757000 (sc_752e):** 5 of 9 bends are UNKNOWN (OCCLUDED or BEND_STATE_UNKNOWN). B1 is
   PARTIALLY_FORMED at +70.5° from nominal, which is spurious. Its twin scan sc_777c is 9 of 9
   exact.

## Side by side with the bar (same scans, frozen matcher 5° / 4 mm / 0.6, apex locus, raw frame)

| Population | Engine (E3) | Best F1 lane |
|---|---|---|
| BROAD, all 26 A1 cases (frozen scorer; refused cases count as misses), family-macro / micro | **0.319 / 0.282** (TP 24, FP 2, FN 120) | **0.543 / 0.508** (TP 65, FP 47, FN 79) |
| BROAD, the 13 scans E3 ran (11 families), family-macro / micro | **0.406 / 0.462** (P 0.923, R 0.308) | **0.649 / 0.620** (P 0.688, R 0.564) |
| G2(a) Gold11 relabel v1 (gold of record), 9 A1 cases | **0.000** (0 / 77) | **0.547** |
| G2(a) Gold11 relabel v1, the 4 cases E3 ran | **0.000** (all 4 were WRONG_PART_SUSPECTED and published no line) | **0.561** |
| G2(a) human Gold11 (historical diagnostic), 9 cases / 4 ran | 0.000 / 0.000 | 0.473 / 0.717 |
| GOLD_SET_V1, 13 A1 scans with STEP that E3 ran, micro / family-macro | **0.491 / 0.440** (P 1.00, R 0.325) | **0.597 / 0.653** |

The owner's bar is 0.458 broad-57 and 0.510 Gold11 relabel. On the frozen A1 BROAD gold, the same
best lane scores 0.543 family-macro. The engine is below the bar on every population. The legacy
arm's G2(a) F1 is 0.088 (informational only, not a bar).

The recipe ceiling on the A1 BROAD gold is 0.967 family-macro. It is capped below the target by
11979003 (gold-convention conflict) and 28870000 (an UNSUPPORTED rolled feature).

## Gates

| Gate | Result |
|---|---|
| **G1** (synthetic `regression` DEV tier, 50 cases, 7 families, scanner model card) | `regression_verdict` **PASS**. Every accuracy gate passes:<br>- flange bias 0.009 mm, P95 0.103 mm<br>- angle bias 0.0015°, P95 0.019°<br>- position P95 0.073 mm<br>- coverage 0.997<br>- false PASS 0 of 553<br>- false FAIL 0 %<br>- U calibration 0.984<br>- median U 0.106 mm / 0.116°<br>- decisiveness 0.983<br>- stage state 1.0<br>- 0 engine failures<br>`G1.SUITE` FAILs by construction (the regression tier is below 200 cases). The runtime gates are NOT_EVALUABLE on this tier. The run is not qualifying: TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL. |
| **G2** | **FAIL** (`final: false`; the truth table is missing):<br>- (a) line metric: 0.0<br>- (b) coverage: angle 0.264 (24/91), flange 0.258 (46/178), gates 0.9 / 0.7<br>- (c) known-good angle: MAE 0.461° on 10 bends; on the 5 common bends 0.556° vs legacy 1.115°; FAILs on coverage, 0.31 vs legacy 0.50<br>- (d) known-good flange: P95 1.22 mm (> 1.0); large deviations on 11979003 side B ±1.1-1.2 mm and on 15651004 B3 side B 1.45 mm<br>- (e) runtime: P95 20.8 s and 0 crashes, but `all_cases_run` FAILs (5 refused, 1 compile BLOCKED)<br>- (f) repeatability: NOT_EVALUABLE (2 of 3 15651004 scans refused)<br>- (4d) compile rate: 24/26 = 0.923, PASS |
| **G3** | **NOT_EVALUABLE**: no frozen visual truth table (amendment 7). |
| **G5** | **FAIL**:<br>- manifest, tsc, lint and the vendor scripts PASS<br>- backend: 10 NEW failures in `tests/test_vendor_unfold_api_contract.py`<br>Cause: a concurrent workflow edited `apps/api/services/vendor_setup_service.py` at 02:05 to call a new module helper, `_natural_label_key`. The test extracts `get_vendor_setup` alone into an isolated scope, so the call raises NameError. Not engine-related. Reproduced twice. |

The ledger check is OK and the anchor `cf01e709…` was checked. The engine and compile code changed
after the freeze, which is allowed. The eval fingerprint is unchanged. `compile-frozen` produced
18/18 compiled parts byte-identical to the freeze.

## Per-bend correctness given CAD (report-only proxy)

Population: 80 recipe bends on the complete-label, complete BROAD scans that E3 ran. No physical
dimensional truth exists, so these are proxies.

| Measure | Result |
|---|---|
| Correct: FORMED + apex line matches gold + angle and both flanges MEASURED | 20 / 80 = **0.25** |
| FORMED | 0.51 |
| Line matches gold | 0.30 |
| Angle measured | 0.51 |
| PASS decisions | 116 in total, 69 of them on gold-confirmed bends |
| False-PASS proxy: PASS on a published line that matches no gold line | 2. Both are on 11979003 B3/B4, where the BROAD gold's window-split convention conflicts. On GOLD_SET_V1 the same 7 lines all match. |
| PASS on bends with no published line | 15 (REGISTRATION_UNTRUSTED). These need a decision rule. |

## Runtime and RSS

| Run | P95 wall | Peak RSS |
|---|---|---|
| Physical E3 | 20.8 s (G2), 22.9 s (BROAD) | 0.88 GiB |
| Synthetic E3 | 10.9 s | 0.81 GiB |

Every heavy step ran serially under `/private/tmp/claude-501/sherman_geom.lock` with
`OMP_NUM_THREADS=1`.

## Jev A/B

The commands from `JEV_AB.md` were run on the engine lane, restricted to the A1 BROAD cases
(`jev_ab_e3/case_list_broad_A1.json`).

- The export and the dry-run completed: 26 candidates (24 TP, 2 FP), estimated cost $0.0014.
- The live calls were refused with **HTTP 402** (the TypeSafe organisation has no credits), so the
  A/B is BLOCKED.
- Even when it runs, a filter only removes lines. With 2 FP, the possible gain is at most about
  +0.004 micro F1.

## Overlays (20 worst bends, at most 3 per case)

`overlays/NN_<case>_<bend>_closeup.png` is the frozen G4 close-up: scan, recipe CAD, section.

`overlays/NN_<case>_<bend>_context.png` is the raw-frame view with:
- gold apex lines in green
- the engine's line for this bend in red
- the engine's other lines in orange
- the best lane's lines dotted blue

It also states when the engine published no line. Order: e080 B7/B9/B2, 03395 B3/B2/B1, d4518
B3/B4, 2714 B1-3, 4f81 B1-3, 70b3 B1-3, 752e B4-6.

## Custody

Only the cases in `cases_A1.json` were run or scored, and only their BROAD, Gold11 and gold-set
labels were read.

Never read: DEV-CHECK, `*SEALED*`, FRESH, HOLDOUT10, N001-N016, EHO4.

The 13 truth-relevant cases were not run on E3 (amendment-7 gate). No file under
`domains/production/**`, `contracts/**` or the frozen eval was edited. The work directory lives
outside the repository: `…/scratchpad/a1_iter0`.

## What iteration 1 should attack, in order of lines lost

1. **Freeze and record the G3 truth table.** This is the owner/reviewer step that unblocks 13 BROAD
   cases and 5 Gold11 cases.
2. **Fix the datum-hole wrong-part check on real scans.** Filled holes in Zeiss exports must not
   invalidate the whole inspection; they should at most downgrade identity confidence.
3. **When registration is untrusted, still publish the scan-fitted apex line** (the line comes from
   scan geometry). Gate the PASS decisions instead.
4. **Recall on occluded or unknown bends,** and reject the spurious PARTIALLY_FORMED on 50757000.
5. **Investigate the flange nominal-relative outliers above 1 mm** on 11979003 and 15651004.
