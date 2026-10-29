# Generality check of the production engine (E1 compile + E3 inspect), 2026-10-19

> **Families never used for engine tuning; labels are Claude vision gold (CONSUMED dev split); NOT the sealed FRESH/DEV-CHECK final evaluation.**

Engine: E3 `e3-1.9.0` (the same at the start and end of every case), E1 compiler `e1-1.1.0`. The engine code was only read, never edited. Machine-readable results: `results.json`. Overlays: `overlays/*.png`. Per-case work: `work/<case_id>/`.

## Headline: the eligible set is too small to support a generality claim

| | cases | families |
|---|---|---|
| GOLD_SET_V2 CAD list | 43 | |
| not one of the 14 A1 families | 17 | 10 |
| dropped as sealed DEV-CHECK (count-only check) | **14** | |
| dropped by other custody rules (FRESH / SEALED / HOLDOUT10 / N001-N016 / EHO4 / not CONSUMED) | 0 | |
| **kept** | **3** | **3** |

Of the 4 cases under `incoming_codex_fetch_2026-10-17`, all 4 fell in the DEV-CHECK drop, so none was kept and the CONSUMED check never came into play. The 3 cases kept are all BENCHMARK_V3 CONSUMED. The DEV-CHECK draw took 7 families, and they cover almost every non-A1 family that has CAD. **The check that was asked for therefore collapses to 3 scans with 4 gold lines (2 line scans + 1 zero-bend control).** The numbers below describe those scans and nothing more. They are not a generality estimate, and the real held-out answer can only come from the one-time DEV-CHECK/FRESH run.

Custody method: a script checked the sha256 of `devcheck_SEALED.json` against `LEDGER.json` (it matched), then compared family, scan sha and CAD sha in memory and printed only the count. No sealed identifier was printed, stored or reported.

## Line-F1 (frozen matcher: 5 deg / 4 mm / 0.6, Hungarian, apex locus, raw frame)

Scored with `research/broad_baseline/score.py cases`. E3 lines come from `line_metric.engine_lines(OUTER_APEX, raw)`. The best-F1 lane figures are GOLD_SET_V2 `rescore_full/cad/scores_best_f1.json` on the same scans.

| arm | micro P | micro R | micro F1 | family-macro P / R / F1 (2 fam) | FP lines on zero-bend control |
|---|---|---|---|---|---|
| **E3 e3-1.9.0** | 1.000 | 0.500 | **0.667** | 0.50 / 0.50 / **0.50** | **0** |
| best F1 lane | 0.400 | 0.500 | 0.444 | 0.50 / 0.50 / 0.50 | 2 |

| case | family | gold | E3 TP/FP/FN | best_f1 TP/FP/FN |
|---|---|---|---|---|
| sc_203b95d7a753 | 49201000 | 2 | 2/0/0 | 2/0/0 |
| sc_822f083e3dfc | 37126000 | 2 | 0/0/2 (no recipe) | 0/3/2 |
| sc_b4719db3dd69 | 47479000 | 0 (control) | 0 lines (refused) | 2 FP lines |

## Per-bend formed / not-formed agreement with the gold
- 49201000: E3 rated B1 and B2 FORMED, and so does the gold (2/2). Angles read 89.65 and 89.69 deg against a nominal 90; flanges are within 0.12 to 0.45 mm.
- 37126000: neither gold bend got an engine state because there was no recipe (0/2).
- 47479000: the gold has 0 bends. E3 refused and claimed no formed bend.
- Total: 2/4 gold formed bends agree (0.50). Among bends the engine reported, 2/2 agree (1.00). The engine called no gold-formed bend not formed and called no bend formed on the control.

## Compile rate, refusals, runtime, RSS
- DEV recipe rule: the same as the freeze's A1 recipes, `E1-DEV-POLICY-1` DEV_AUTO_CONFIRMED (amendment 4d). That means every V10/native CAD bend, both sides REQUIRED, and no hand edits or part-specific exclusions.
- **Compile 1/3 (33%).** Failures stay in the denominators:
  - 37126000: **NO_RECIPE**. Both the V10 and the native extractor found 0 bends in the STEP. GOLD_SET_V1 records 2 CAD bend lines for this part, and the scan shows them clearly: a 7 mm strip with a Z-joggle of two ~40 deg folds whose bend zones are only ~6.7 mm long. **This is an E1 extractor gap (joggle on a narrow strip), not a label problem.**
  - 47479000: **BLOCKED** `DATUM_PANEL_UNDETERMINED`. The native extractor proposed 1 candidate on a rolled half-cylinder that has no planar flange. E3 then refused, as expected (exit 2, COMPILE_BLOCKED). For a part with no bends this is the safe outcome, but it is a refusal and not a "no bends, OK" result.
- E3: 1 OK, 1 explicit refusal, 0 crashes, 0 timeouts.
- Runtime and RSS:
  - recipe build: 1.6 to 1.9 s and ≤0.35 GiB;
  - compile: 1.0 to 2.3 s and ≤0.50 GiB;
  - E3 on 49201000: 14.7 s and 0.93 GiB peak;
  - whole case: 18 s;
  - for comparison, the best-F1 lane took 8.6 / 379.5 / 87.5 s on these scans (peak 1.40 GiB).
- All heavy steps ran serially under `/private/tmp/claude-501/sherman_geom.lock`, one case per lock hold, OMP_NUM_THREADS=1.

## Visual review (all 3 scans; there are fewer than 5)
I looked at each overlay myself (`overlays/<case>.png`: gold in lime, E3 in red, best_f1 as blue dashes).
- sc_822f083e3dfc (37126000): the Z-joggle is plain to see in the side view. Both gold lines run across the full strip width at the two folds. **The gold is correct and the miss belongs to the engine (E1).** The best-F1 lines are diagonal and off the folds (3 FP).
- sc_203b95d7a753 (49201000): a U-channel. E3 and the gold coincide along both ~208 mm long edges. The result is correct.
- sc_b4719db3dd69 (47479000): a smoothly rolled half-cylinder with two cutouts and no folds. **The zero-bend gold is correct.** The best-F1 lines are FPs along the roll.

**Label problems found: none.** No score was adjusted.

## What this does and does not say
- On the one generic sheet-metal part that compiled (a U-channel), E3 matched the gold exactly with no FP.
- The weak point outside the tuned families is E1 compile coverage. A narrow-strip joggle produced no CAD bends, and a rolled part blocks. Both mean "no inspection", which caps recall whatever E3 does.
- With n=3, no confidence interval is meaningful. The real held-out generality figure has to come from the sealed DEV-CHECK + FRESH final run.
