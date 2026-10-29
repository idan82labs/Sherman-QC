# E4a: evaluation core (scorer, truth loaders, tiered runner, G2 line metric, G4 renders, freeze, model card)

Owner: E4a. Code: `domains/production/engine/eval/`. Tests: `tests/test_prod_engine_eval_*.py`.
Plan: `docs/production/PRODUCTION_LINE_PLAN.md` §5 (G1–G4) and §10 amendments 3, 4, 5, 6, 8.
Contracts it reads: `contracts/production/ENGINE_IO.md` (E3 command line and output directory,
`ShopfloorInspectionResultV1`, `CompiledPartV1`, `check_result_against_compiled`), the E2 case
directory (`docs/production/ENGINE_SYNTHETIC.md` §2, §4) and the E1 compile command line.

**Custody.** Everything here runs on synthetic E2 cases and their truth, except the scanner model
card (§10), which read five authorized development scans from `../source-assets/02_CAD_QC_3D`
(plan §9; all CONSUMED parts named in the plan's G2/G3 lists) for light statistics, one scan at a
time. Nothing reads HOLDOUT10, sealed truth, N001–N014, EHO4, email, Gold labels or authorization
artifacts (the G2 aggregate parity test runs the Gold script on synthetic Gold files only).
Results on E2 cases are synthetic engineering evidence and are never quoted as physical or
unseen-family accuracy; every scorecard says so in its `label`.

## 1. Package and command line

| Module | Role |
|---|---|
| `thresholds.py` | Frozen, hashed gate documents: `G1`, `G2_LINE`, `G2_PHYSICAL`, `G3`, `G4` |
| `truth.py` | Typed loaders for E2 `case.json` / `truth.json` / `manifest.json` (hash-verified) |
| `engine_output.py` | Validated E3 output directories, frame helpers, characteristic lookup by kind/side, the measured outer apex |
| `scorer.py` | Rows per (scan, bend_uid, characteristic), per bend and per case; registration identity check; the G1 scorecard |
| `runner.py`, `procmon.py` | Serial tier runner: E1 compile CLI + E3 inspect CLI subprocesses with wall time, process-group RSS and hard limits |
| `line_metric.py` | G2 line metric (5° / 4 mm / 0.6 Hungarian) in the raw-scan frame |
| `render.py`, `g4.py` | G4 overlay renders, secret canaries, reviewer packets, verdict schema, the G4 gate |
| `freeze.py` | Hash-chained freeze ledger (amendment 4): FREEZE / RUN / WAIVER / NOTE entries |
| `scanner_card.py` | Amendment-4b scanner model card |
| `oracle.py` | **TRUTH ORACLE, not an engine**: writes contract-valid E3 outputs from E2 truth for harness self-tests |

```
PY="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python"
$PY -m domains.production.engine.eval run --tier-dir E2_TIER --out RUN [--ledger L --ledger-anchor SHA] [--engine-config C]
$PY -m domains.production.engine.eval run --tier-dir QUAL_TIER --out RUN --ledger L --ledger-anchor SHA --final-qual-run
$PY -m domains.production.engine.eval score --run-dir RUN [--ledger L --ledger-anchor SHA]   # rebuild / re-score
$PY -m domains.production.engine.eval blindness --tier-dir E2_TIER --out D --case ID ...
$PY -m domains.production.engine.eval render --engine-dir E3_OUT --compiled C/compiled_part.json --recipe R --out D
$PY -m domains.production.engine.eval packet --jobs jobs.json --packet P --sealed S
$PY -m domains.production.engine.eval g4 --packet P --sealed S --verdicts g4_verdicts.jsonl --out report.json [--no-synthetic-requirement]
$PY -m domains.production.engine.eval freeze --ledger L --label "pre-A1" --seeds regression:DEV --seeds full:DEV \
    --seeds full:QUAL --seeds runtime:QUAL --allow-qual --case-list G2=g2.json --case-list G3=g3.json \
    --case-list DEV_CHECK=devcheck.json --recipe R1.json ... --scanner-model-card docs/production/runlog/scanner_model_card.json
$PY -m domains.production.engine.eval verify-ledger --ledger L
$PY -m domains.production.engine.eval waive --ledger L --scope SCANNER_MODEL_PARTIAL --approved-by OWNER --reason TEXT \
    --scanner-model-card CARD --field grazing_start_deg ...        # OWNER ONLY; agents never run it
$PY -m domains.production.engine.eval model-card SCAN1.ply SCAN2.ply SCAN3.ply --out card.json --evidence ev.json --measured-on DATE
$PY -m domains.production.engine.eval model-card --from-evidence docs/production/ENGINE_EVAL.md --measured-on 2026-10-17 --out card.json
$PY -m domains.production.engine.eval line-metric --gold gold.json --predictions preds.json [--legacy l.json] --out r.json
```

Exit codes: 0 the command ran (a gate verdict is inside its output), 2 usage, input or policy error
(e.g. QUAL seeds without `--final-qual-run`), 3 internal error. `run` prints `PROGRESS\t{json}`
per case and a final `RESULT\t{json}` with `verdict`, `regression_verdict` and `qualifying`.
`packet` has no seed option: its canary salt always comes from the OS (reproducible seeds exist
in the Python API for tests, and such packets never count, §8).

## 2. Truth and the scoring rule

`truth.load_truth_case(dir)` verifies every file hash listed in `case.json` (a tampered or partial
case raises `TruthError`) and types the truth: per characteristic `nominal`, `truth`,
`true_deviation`, `truth_in_tolerance`, `tolerance`, `applicable`, `well_defined`,
`truth_undefined_reason`; per bend the contract-vocabulary state (`bend_states_contract`, so an
out-of-order formed bend is `UNEXPECTED_FORMED`) and the as-built OML / outer-apex loci; the stage
block; the expectation lists; `raw_to_cad` (None when absent, never silently the identity). It
imports nothing from E1, E2 or E3.

**Scorable** (the E2 contract): `applicable` and `truth` not null and `well_defined` not false. A
backgauge carries truth for both sides; only the side E1 attached the characteristic to is scored
(the engine row's `side`, else the compiled nominal's). A nominal is never used as truth.

## 3. Scorer and the G1 gates

`scorer.score_case(truth, run, result, compiled)` builds three row kinds (all JSON-serialisable,
persisted by the runner under `rows/`; loading rows written by another eval version raises):

- `CharRow` per truth characteristic: engine state/decision/value/U next to the truth, with
  `error = actual − truth`, `within_u = |error| ≤ U`, `false_pass` (+ `false_pass_kind`),
  `false_fail`, `false_pass_tol_plus_u` (the §5 wording, informational), `ambiguity_excused`,
  `nominal_disagreement = expected(E1) − nominal(E2)`.
- `BendRow` per recipe bend: truth vs engine state, binary formed/unformed agreement (an engine
  `UNKNOWN` or a missing row is wrong), exact-vocabulary agreement, a **truth-frame line check**
  (the measured OML line mapped raw-ward with the engine's registration, then into the truth frame
  with `raw_to_cad`: `ON_OWN_BEND` when its angle to the bend's true OML sharp is ≤ 5° and its
  transverse offset ≤ max(2 mm, Ri + t), `WRONG_ID` on another bend, else `MISPLACED`), and the
  bend's **identity** verdict (below).
- `CaseRow`: run status, wall time, peak RSS, validity, summary, stage decode, symmetry source,
  pose check.

**False PASS** (amendment 5, judged on E2 truth, never on the engine's U): a PASS is false when
the characteristic is scorable and its truth is out of tolerance (`OUT_OF_TOLERANCE`), when it is
on E2's never-PASS list (every characteristic of a wrong or mirrored part, out-of-tolerance
backgauge sides: `NEVER_PASS_LIST`), or when its bend is truly NOT_YET_FORMED (`NOT_YET_FORMED`:
the true angle deviation is the whole sweep; the flange and line do not exist yet). The last rule
closes the gap the review found: a result that claims a flat bend FORMED at nominal values made
four PASS decisions while every zero-tolerance gate stayed green.

**Identity from the registration (pose check).** `D = raw_to_frame · diag(unit_factor) ·
inv(truth.raw_to_cad)` maps the true part to where the engine believes it is. Reference points are
the panel outer boundaries at the stage pose. `POSE_OK` when D rotates ≤ 5° and moves the points
≤ 4 mm RMS; `POSE_SYMMETRIC_FLIP` when D matches an E1 ROTATION self-map of the stage model
(≤ 5°, ≤ 4 mm), whose `bend_permutation` names the physical bend each row carries; `POSE_WRONG`
otherwise; `NOT_EVALUATED` without a result, a transform or registration truth. A bend's identity
is `WRONG_ID` (line check or flip), `WRONG_POSE`, `RIGHT_ID`, `NO_MEASUREMENT` or `NOT_EVALUATED`.
`G1.SYMMETRIC_NO_WRONG_ID` counts `WRONG_ID` + `WRONG_POSE` on E1-symmetric cases; a bend with
MEASURED values and no identity evidence makes the gate NOT_EVALUABLE, never PASS. So a confident
180° flip is caught even when E3 withholds every line (the review's probe: PASS before, FAIL now).

**Engine failures stay in every denominator.** A run status other than `OK` (non-zero exit,
crash, timeout, RSS kill, missing/invalid/inconsistent `result.json`, an engine that modified its
input) means the result is not used: every characteristic of the case counts as not measured
(`ABSENT`) for coverage and stage accuracy, nothing is scored as an error, and
`G1.ENGINE_FAILURES` counts it. A compile failure is reported separately but also stays in the
denominators.

**Symmetric vs unambiguous (fixed definition, amendments 5 and 8).** A case is *symmetric* when
E1's compile-time symmetry test of the stage model whose formed set equals the truth formed set
has a non-empty `ambiguous_bend_groups` (rotation orbits). Only when no E1 stage model has that
formed set (out-of-order formed sets) does E2's `expected.symmetric` decide; the source is recorded
per case (`ambiguity_source`) and E1/E2 disagreements are listed in the scorecard counts.

**Gates** (thresholds in `G1_THRESHOLDS`, version `E4-G1-2`; every gate is computed on the point
estimate over its population, with a family-cluster bootstrap 95 % CI reported, not gated):

| Gate | Population | Statistic and threshold |
|---|---|---|
| `G1.ACC.<kind>.BIAS` / `.P95` | MEASURED scorable FLANGE_LENGTH / INCLUDED_ANGLE / BEND_LINE_POSITION on complete parts, FULL plan, native density | flange \|bias\| ≤ 0.05 mm, P95 \|error\| ≤ 0.20 mm; angle ≤ 0.10° / ≤ 0.30°; position P95 ≤ 0.30 mm |
| `G1.COVERAGE` | scorable characteristics of defined parts, FULL plan, native density; failed runs included; NOT_MEASURED(AMBIGUOUS_POSE) on E1-symmetric cases excused (raw rate reported) | MEASURED share ≥ 0.90 |
| `G1.FALSE_PASS` | every characteristic, native density, every plan | false PASS count (rule above) = 0, with the count per kind |
| `G1.FALSE_FAIL` | MEASURED scorable characteristics whose truth is in tolerance | FAIL share ≤ 0.5 % |
| `G1.U_CALIBRATION` | MEASURED scorable characteristics | share with \|actual − truth\| ≤ U ≥ 0.95 |
| `G1.MEDIAN_U.mm` / `.deg` | MEASURED scorable characteristics on FULL-plan cases, per unit | median U ≤ 0.15 mm / ≤ 0.25° |
| `G1.DECISIVENESS` | MEASURED rows with a guard-band decision whose true deviation is within 0.5 × or beyond 1.5 × the tolerance limit on its side | PASS-or-FAIL share ≥ 0.90 |
| `G1.SLOPE.<kind>` | MEASURED scorable rows per kind; needs ≥ 10 rows of which ≥ 5 have \|true dev\| ≥ 0.5 × tol, else NOT_EVALUABLE | OLS slope of measured deviation on true deviation in [0.95, 1.05]; BACKGAUGE informational |
| `G1.STAGE_STATE` | bends of truth-unambiguous incomplete-part cases (failed runs count as wrong) | binary formed/unformed correct ≥ 0.98; UNKNOWN/AMBIGUOUS is an error |
| `G1.SYMMETRIC_NO_WRONG_ID` | bends of E1-symmetric cases | confident wrong identities (line check or pose check) = 0 |
| `G1.BACKGAUGE_DETECTION` | scorable backgauge characteristics with \|true dev\| ≥ 0.5 mm | decided FAIL or REVIEW, share = 1.0 |
| `G1.NOT_YET_FORMED_NEVER_FAIL` | characteristics of truly NOT_YET_FORMED bends | FAIL count = 0 (PASS decisions there are false PASSes) |
| `G1.WRONG_OR_MIRRORED_NEVER_PASS` | every characteristic of wrong-SKU / mirrored cases | PASS count = 0 |
| `G1.SUMMARY_NEVER` | every case with a result | summary never in `expected.summary_never` |
| `G1.MUST_BE_UNKNOWN` | characteristics without defined truth | MEASURED count = 0 |
| `G1.ENGINE_FAILURES` | every compiled case | engine failures = 0 |
| `G1.OCCLUSION_RAISES_UNKNOWN` | TOP_ONLY / SPARSE / SINGLE plans | no engine failure, no false PASS, U calibration ≥ 0.95 |
| `G1.RUNTIME.P95` / `.RSS` | runtime tier, cases with ≥ 0.9 M points | P95 wall ≤ 45 s, max peak RSS ≤ 4 GiB (binding only on the runtime tier) |
| `G1.SUITE` | the scored tier | ≥ 200 cases, ≥ 7 families, every amendment-5 feature per family (binding on `full`) |
| informational | `G1.LINE_ON_OWN_BEND`, `G1.POSE_CHECK`, `G1.AMBIGUOUS_ON_UNAMBIGUOUS`, `G1.TIER_BUDGET` | automated LANDS analogue; registration identity on every defined case; AMBIGUOUS_POSE on unambiguous cases; amendment-3 20-minute budget |

Every empty population is `NOT_EVALUABLE`, never PASS. Each gate also records
`truth_population_n`, the size of its population on the truth side alone; when that is zero in
the scored tier (no degraded scan plan in the regression tier, too few injected errors for a
slope, no symmetric case) the gate is flagged `not_applicable_to_tier`. **`regression_verdict`**
(A1's per-change signal) is FAIL when any applicable binding gate fails, PASS when every
applicable binding gate passes, INCOMPLETE otherwise; an engine-side empty population (nothing
MEASURED) is never "not applicable". The strict `verdict` keeps every binding gate. On a
point-count override tier (`runtime`, `smoke`) only the runtime and engine-failure gates bind.

**Qualifying.** A scorecard is qualifying only when all of these hold; otherwise
`not_qualifying_reasons` says why and the verdict is at best `PASS_NOT_QUALIFYING`:
`tier = full` (or `runtime` for its gates); QUAL seeds; a `MODEL_CARD` scanner model whose
amendment-4b quantities are all measured (`SCANNER_MODEL_PARTIAL` otherwise, unless an owner
WAIVER for exactly that model and those fields is in the ledger; the unmeasured fields are listed
in the provenance); standard run conditions (`RUN_CONDITIONS_NOT_RECORDED`,
`NON_STANDARD_LIMITS` unless 180 s / 4 GiB / 170 s, `NON_STANDARD_ENGINE_CMD` unless
`-m domains.production.engine.inspect`, `NON_STANDARD_ENGINE_ENV` for any `--engine-env`); no
oracle results; a passing suite composition; and a ledger check that matches
(`FREEZE_LEDGER_MISMATCH`), is anchored (`LEDGER_NOT_ANCHORED`) and saw no second engine on the
QUAL seeds (`QUAL_SEEDS_REUSED`). `scorer.g1_verdict(full, runtime)` is PASS only when both
scorecards are qualifying and PASS with one engine fingerprint, one engine config and one ledger.

Threshold document hashes at hand-off (canonical JSON sha256; the freeze ledger pins them):

| Document | Version | sha256 |
|---|---|---|
| G1 | E4-G1-2 | `d475df3a97e2a458ebbe59e4591e82efee4d36094a6a0f2865929532419ade17` |
| G2_LINE | E4-G2-LINE-2 | `eb00d448cacee65d797645948d9de20d11e8bed03fc1785cf726524f626e9902` |
| G2_PHYSICAL | E4-G2-PHYS-1 | `148d0440b0b56c52e5e8f34379d892ef0201e6a680f5eaaa072359a074b0a340` |
| G3 | E4-G3-1 | `c67d15278b530a5ef1b0b73659cefcd595f693728cc4bde1c1d19664df1285f6` |
| G4 | E4-G4-2 | `cfe774d0d70a9843aa48deb9461f58cc83c368d8f43db6e3c45cc6ec521dd41f` |

## 4. Tiered runner

`runner.run_tier(RunnerConfig)` (CLI `run`) processes an E2 tier directory serially:

1. load and hash-verify the case;
2. compile with `python -m domains.production.engine.compile`, cached under
   `RUN/compile_cache/<sha256(recipe | cad | flat | E1 code fingerprint | compile config)>/`; the
   recipe, STEP and flat are staged there as read-only copies under fixed names. A transient
   compile failure (timeout, signal, RSS kill) is recorded but never reused: the next run compiles
   again;
3. inspect with `python -m domains.production.engine.inspect` (or `--engine-cmd`), passing
   `--scan-sha256`, `--compiled-sha256`, `--recipe-sha256` and `--time-budget-s 170`, under the
   runner limits (180 s hard stop: SIGTERM, 3 s, SIGKILL on the whole process group; 4 GiB RSS:
   immediate SIGKILL). **Blindness**: the workspace is `RUN/work/<opaque hash>/`, the scan
   `input/scan.ply`, the recipe `input/recipe.json`; no case id, family code, ZEISS-like name,
   truth or count reaches E3. The inputs are **read-only copies**, never hard links into the E2
   tier, and are re-hashed after the run: an engine that changed them is `RESULT_INCONSISTENT`
   (`ENGINE_MODIFIED_INPUT:scan`);
4. validate `result.json` (model) and run `check_result_against_compiled`; any problem is
   `RESULT_INVALID` / `RESULT_INCONSISTENT`; a result for another scan sha is inconsistent;
5. score and write `cases/<case_id>.json` (schema `sherman_e4_run_case_v2`).

Two identities make reruns cheap and honest. The **resume key** (input hashes, compile key,
outcome and compiled-part sha, E3 code fingerprint, engine command, config, extra environment,
bundle id and limits) decides whether the engine runs again. The **score key** (eval version,
eval code fingerprint, G1 thresholds hash) decides whether a stored engine output is re-scored:
after a scorer fix the stored results are scored again without rerunning the engine (`score`
does the same for a finished run directory, refusing a record whose engine output is gone).
`run_manifest.json` records the fingerprints, code roots, limits, run conditions, config shas and
paths; `score` copies them into the scorecard provenance and passes the engine config sha to the
ledger check, so a rescored card qualifies exactly like the original.

**QUAL seeds** (amendment 4c) run only with `--final-qual-run` and `--ledger` (the policy applies
when the tier manifest or any selected case's own `case.json` spec says QUAL, so an edited manifest
cannot hide QUAL cases); before the first engine call the runner appends a RUN entry (tier, case-list sha, engine fingerprint, engine
config sha, eval fingerprint). A later QUAL run with another engine identity sets
`QUAL_SEEDS_REUSED`.

Measurement (`procmon.run_monitored`): wall time from spawn to exit; peak RSS = max of the
process group's summed RSS sampled every 0.25 s with `ps -A -o pgid=,rss=` (the C02 supervisor's
probe) and the leader's exact `ru_maxrss` from `os.wait4`. Outputs: `run_manifest.json`,
`cases/*.json`, `rows/{cases,chars,bends}.jsonl`, `scorecard.json`, `scorecard.md`.
`runner.blindness_check` (CLI `blindness`) runs each case twice with the scan staged under
different opaque names and compares the canonical results (amendment 4).

## 5. G2 line metric (amendment 6a)

`line_metric` re-implements the matcher of `scripts/evaluate_f1_against_manual_gold.py` (so the
package never imports `scripts/`); tests load the script and check identical `pair_metrics` and
`match_lines` outputs on random line sets, and identical `evaluate()` totals, precision, recall,
F1 and exact-part rate on synthetic Gold files (the script's `ROOT` and case table are redirected
to a temporary directory). Rules:

- scoring is in the **raw-scan frame**: engine lines are mapped back with the result's own
  `raw_to_frame` and `unit_factor`; Gold lines drawn in another frame need a recovered
  raw-to-reference transform with residual ≤ 0.01 mm, otherwise the case is **listed and excluded
  before scoring** (`NO_RECOVERABLE_TRANSFORM`, `TRANSFORM_RESIDUAL_TOO_LARGE`);
- the declared locus is the visible-surface **apex at the measured angle**:
  `engine_output.measured_outer_apex` moves the measured OML sharp along the measured bisector
  (θm/2 from the parent flange direction) by (Ri + t)(1/sin(θm/2) − 1), E1's own loci formula, so
  a partially formed bend (150° instead of 90°) is scored at its real apex instead of 0.38·(Ri + t)
  off it. Without a measured angle the nominal offset is used and the record says
  `apex_source = NOMINAL_ANGLE_FALLBACK`; `OML_VIRTUAL_SHARP` is a diagnostic locus;
- only bends whose BEND_LINE_POSITION is MEASURED emit lines (selected by kind, not position);
  NOT_YET_FORMED/UNKNOWN bends and invalid inspections emit none; a Gold case without predictions
  is all false negatives;
- the aggregate is the script's: precision and recall are 0.0 on an empty denominator and F1 is
  0.0 when precision + recall = 0, so an engine that emits nothing on usable Gold scores F1 = 0 and
  FAILs; only a run without any usable Gold case is NOT_EVALUABLE;
- the report carries TP/FP/FN, precision, recall, F1, the exact-part count and rate, gates F1 ≥
  0.80 and precision ≥ 0.90, `exact_part_rate_beats_legacy` (strictly greater; NOT_EVALUABLE
  without legacy predictions) and the label "recipe-guided vs blind union, not like-for-like".

The Gold transform recovery itself is wave 3 (not E4a); `GoldCase` takes its output.

## 6. G2 physical and G3 threshold documents

`G2_PHYSICAL_THRESHOLDS` (G2 b–f, amendments 4d/4e/4f/6) and `G3_THRESHOLDS` (amendment 7) hold the
numbers the wave-3 real-part runner must gate on, so the pre-A1 freeze pins them now: compile rate
≥ 80 % of authorized STEPs; ≥ 90 % of bends with a MEASURED angle and ≥ 70 % of flange sides with a
MEASURED length; nominal-relative angle MAE ≤ 1.0° on known-good parts and no worse than legacy on
bends both arms measure, engine coverage ≥ legacy; flange P95 |Δ| ≤ 1.0 mm with nothing excluded and
a zoomed crop per |Δ| > 1 mm; runtime P95 ≤ 45 s with zero crashes; repeatability P95 between
repeats ≤ 0.3 mm / 0.3° and better than legacy (15651004 ×3, 42288005 ×2, 47959001 PLY+STL); the
DEV-CHECK rule; and for G3 a frozen two-reviewer truth table, every bend matching, UNKNOWN only
against NOT_DETERMINABLE, no unformed bend reported as a defect, the blank decoded as stage 0, and
the formed count equal to `expected_completed` where it exists. E4a evaluates none of these.

## 7. G4 visual verification (amendment 8)

**Renders** (`render.py`, matplotlib Agg, CPU). Global: ISO plus three orthographic views along the
scan's principal axes (camera on the outside of the bends), a DIMENSIONS view and a HEATMAP view
(signed deviations, ±band legend). The DIMENSIONS arrows run from the measured OML sharp along the
flange direction **fitted to the scan points** at mid-span (`fit_flange_direction`; nominal when
fewer than 20 points, counted in the record), and reviewer packets render them **without lengths**
(`show_values=False`; amendment 8: reviewers see no metrics). Per bend a close-up at **exactly
10 px/mm** (never below 8; a test measures the 10 mm scale bar at 100 ± 3 px and a 14 mm canary
shift at 140 ± 4 px):

- **SCAN** panel: camera on the outer bisector, points coloured by segment (E3's
  `segment_labels.u16`, else nearest stage-model face, stated on the image); the measured line
  **solid on the outer apex at the measured angle** (as in §5; without a measured angle the solid
  line is the OML sharp and the header says so), the measured OML sharp **dashed**, the nominal
  stage-model line **grey dashed**, labels `<seq> · <label> · <uid6>`; the header states the
  engine state and, without a line, the position reason;
- **RECIPE CAD** panel (stacked under the SCAN panel, same camera, window and scale): the decoded
  stage model's display mesh with the labels of every bend in view **from the recipe**;
- **SECTION** at mid-span: scan points, CAD section, measured apex (dot) and OML sharp (circle),
  nominal (x), flange-length arrows along the fitted direction.

**Close-up classes and questions** (`g4.closeup_class`): MEASURED (LANDS + TAG_OK); NOT_YET_FORMED,
UNKNOWN and FORMED_NO_LINE (the reviewer answers whether the missing line is justified: really flat,
really occluded or missing; **gated**); AMBIGUOUS_E1_JUSTIFIED / AMBIGUOUS_NOT_E1 (an
AMBIGUOUS_POSE bend is justified **only** by E1's symmetry test: it must be in an
`ambiguous_bend_groups` orbit of the decoded stage model; when the engine decoded none, of the stage
model whose formed set is the bends it did not report NOT_YET_FORMED, and only when no stage model
matches, of any stage model; no reviewer question, and an AMBIGUOUS outside every orbit fails the
gate);
REGISTRATION_RULE (position withheld as REGISTRATION_UNTRUSTED, amendment 9) and UNSUPPORTED_RULE
(E1's per-feature rule) are reported with TAG_OK judged; INVALID_INSPECTION scans are reported and
left to G1 / G2(e).

**Packets and canaries** (`g4.build_review_packet`). At least 10 % of close-ups (rounded up) are
canaries, chosen with a secret OS salt. LINE_SHIFT moves the solid and dashed lines and the section
markers across the bend by |s| drawn from [max(8, LANDS tolerance + 3), 15] mm, LANDS tolerance =
max(2, Ri + t): a reviewer applying the written rule must answer NO (the review found a 10.2 mm
shift on a 10.8 mm-tolerance bend). A bend without such a range (Ri + t > 12 mm) gets a LABEL_SWAP
instead, which exchanges the labels of two bends of one scan **everywhere in the SCAN panel** (so no
image shows two identical labels; only the truthful RECIPE CAD panel reveals it). Ambiguous-pose
close-ups are never canaries. Images get opaque names and opaque titles (a jobs.json title is never
drawn); `review_manifest.json` lists image id, sha256, kind, opaque scan reference, the subject as
displayed and the questions, plus the sha256 commitment of the sealed key. The key
(`sealed/canary_key.json`, schema v2, mode 0400, refused inside the packet) maps every image to its
true subject, case, arm, population, class, LANDS tolerance, E1 orbit membership and corruption
with its margin, and records `salt_source` (OS_RANDOM or TEST_SEED). A prebuilt `RenderScene` can
replace the E3 directory so the legacy arm is rendered the same way.

**Verdicts** (`G4VerdictV1`, one JSONL line each in `g4_verdicts.jsonl`): `image_id`,
`image_sha256`, `bend_uid` (as listed), `reviewer_id`, `role` PRIMARY|TIEBREAK, `lands`
YES|NO|OCCLUDED|NOT_APPLICABLE, `tag_ok` YES|NO|UNDETERMINABLE, `unknown_justified` (only when no
measured line is shown), the derived `verdict` (`…_MISTAG` whenever tag_ok is NO), `reason`,
`recorded_at`. **Evaluation** (`evaluate_g4`): the key must match its commitment and every image its
sha256; a test-seeded packet is refused (`TEST_SEED_PACKET`) unless the caller is a test; the two
primaries' agreement or else the tie-break decides; the review **counts only if every primary
reviewer flagged every canary**. The gate runs on the non-canary close-ups of the **E3 arm only**
(LEGACY is the reported visual baseline, ORACLE a reported self-test): 100 % of MEASURED bends land,
0 mis-tags, every gated no-line close-up justified, no AMBIGUOUS outside E1's orbits. The synthetic
part of the review needs ≥ 20 E3 scans including an incomplete one (`SYNTHETIC_SAMPLE_INSUFFICIENT`;
`--no-synthetic-requirement` for a physical-only packet, recorded in the report). Undecided images
make the gate NOT_EVALUABLE, never PASS.

## 8. Freeze ledger (amendment 4)

`freeze.append_entry` appends a hash-chained JSONL entry (`entry_sha256` over the canonical entry,
`previous_sha256`, `seq`). FREEZE entries hold the sha256 of every threshold document, the planned
case ids and seeds per (tier, seed set) (QUAL only with `--allow-qual`; planning builds no geometry
and reads no output), the DEV_AUTO_CONFIRMED recipes, the G2/G3/DEV-CHECK case lists, the engine
config, the scanner model card (and E2's model sha256) and the engine/compile/eval/contracts code
fingerprints. RUN entries log QUAL runs; WAIVER entries record an **owner** decision on one waivable
mismatch (`SCANNER_MODEL_PARTIAL`, `EVAL_CODE_CHANGED_SINCE_FREEZE`,
`CONTRACTS_CODE_CHANGED_SINCE_FREEZE`; `approved_by` and `reason` required; never a threshold, seed
or case-list change; agents never write waivers). `verify_ledger` detects edits, reordering and
broken links; appending to a broken ledger is refused.

`check_against_ledger` ("thresholds never change mid-run"): the **first** FREEZE entry that
defines a pre-A1 item (thresholds, seeds, recipes, case lists, scanner card, eval and contracts
code, extra files) is authoritative; a later FREEZE entry that redefines one is reported
(`THRESHOLDS_REFROZEN:G1`, `SEEDS_REFROZEN:…`, …) and the current values are compared with the
authoritative ones (`THRESHOLDS_CHANGED:G1`, `CASE_LIST_DIFFERS_FROM_FROZEN`,
`SCANNER_MODEL_DIFFERS_FROM_FROZEN`, `EVAL_CODE_CHANGED_SINCE_FREEZE`, …). The engine config is
A1's final knob and is compared with the **latest** FREEZE entry that defines it. The chain alone
proves only internal consistency, so the sha256 of the first FREEZE entry must be recorded outside
the ledger (the run log / `docs/CURRENT_STATUS.md`) and passed back as `--ledger-anchor`; a
rewritten ledger then fails with `LEDGER_ANCHOR_MISMATCH`, and an unanchored check is never
qualifying. The review's probe (FREEZE, lower `coverage_min` to 0.5, FREEZE again) now reports
`THRESHOLDS_REFROZEN:G1` and `THRESHOLDS_CHANGED:G1`. The integrator/A1 owns the ledger file; E4a
created none in the repo.

## 9. Truth oracle (self-test only)

`oracle.py` writes contract-valid E3 outputs (result, registered scan, deviations and segment
labels from the stage model, display meshes, bend geometry) from E2 truth plus an error model
(noise, bias, gain, U, per-characteristic overrides, forced bend states, ambiguous pose, swapped
lines, a symmetric pose flip, ignored part identity), checked with `check_result_against_compiled`.
Its CLI mirrors E3's and finds truth by scan sha under `SHERMAN_E4_ORACLE_CASES` (passed with
`run --engine-env`; the runner strips `SHERMAN_E4_*` from its own environment), with failure
switches in `SHERMAN_E4_ORACLE_MODE` (`sleep=`, `alloc_mb=`, `crash`, `exit2`, `exit3`, `garbage`,
`no_result`, `name_leak`, `modify_input`). Every oracle result carries `engine_version e4-oracle-…`
and the warning `E4_TRUTH_ORACLE_NOT_AN_ENGINE`; a scorecard containing one is never qualifying,
and oracle runs are also non-qualifying by their engine command and environment. Its numbers are
never engine evidence.

## 10. Scanner model card (amendment 4b)

`docs/production/runlog/scanner_model_card.json`, E2 `ScannerModel` JSON, `provenance =
MODEL_CARD`, E2 model sha256 **`cf55a20c23f37e9076426f0be559584fa9f00941db5d856ddc109bac4cabf8ce`** (validated with E2's own `ScannerModel.from_dict`).
Measured 2026-10-17 by `scanner_card.py` (measurement code `e4a-scanner-card-1.0.0`) on five
authorized development scans (Geomagic-exported ZEISS meshes with vertex normals), one at a time in
one process: 13 s wall, 0.72 GB peak RSS. Light statistics only (no registration, no CAD, no
labels). The fix stage rebuilt the card from the saved per-scan evidence (no scan was re-read):
the numbers are unchanged, the notes now carry the field status.

| Scan (sha256 prefix) | Vertices | NN spacing mm | Noise floor mm | 5 / 10 mm residual | Skin sep. median / sigma mm | Two-layer patches | Edge samples (excluded) | Flare share / p95 mm | Outliers |
|---|---|---|---|---|---|---|---|---|---|
| 15651004_01.ply (03395097cd96) | 104,543 | 0.277 | 0.0036 | 0.0041 / 0.0054 | 1.930 / 0.024 | 0.000 | 2396 (12) | 0.026 / 0.888 | 0.0000 |
| 28870000_A_FULL_SCAN.ply (38091ac91d9d) | 175,182 | 0.306 | 0.0144 | 0.0095 / 0.0125 | 1.460 / 0.014 | 0.000 | 1955 (0) | 0.128 / 0.166 | 0.0000 |
| 23553000_B_FULL_SCAN 1.ply (6eba02723d4b) | 473,764 | 0.503 | 0.0048 | 0.0058 / 0.0063 | 0.921 / 0.042 | 0.000 | 919 (1087) | 0.146 / 0.811 | 0.0000 |
| 38172000_01_SCAN_1.ply (07062eee17aa) | 715,498 | 0.501 | 0.0046 | 0.0051 / 0.0053 | 1.487 / 0.078 | 0.000 | 695 (158) | 0.069 / 0.924 | 0.0000 |
| 10839000_01_full_scan.ply (71cf0b216f43) | 863,115 | 0.401 | 0.0044 | 0.0058 / 0.0072 | 1.441 / 0.041 | 0.000 | 548 (375) | 0.073 / 0.947 | 0.0000 |

Card (median over scans) and **field status** (the machine-readable `FIELD_STATUS={…}` object at
the end of the card notes, parsed by `scanner_card.field_status`):

| Field | Value | Status |
|---|---|---|
| `point_spacing_mm` | 0.401 | MEASURED |
| `noise_sigma_mm` | 0.0046 | MEASURED (2 mm same-skin floor of the merged exports, about 5 × below E2's assumed 0.025 mm) |
| `duplicate_skin_offset_mm` | 0.041 | UPPER_BOUND (the skin-separation spread also holds the thickness tolerance) |
| `duplicate_skin_fraction` | 0.0 | MEASURED (no two-layer patch on any scan) |
| `edge_flare_probability` / `edge_flare_mm` | 0.073 / 0.934 | LOWER_BOUND (flares above 1 mm are not measurable; 0–54 % of the edge samples per scan continue into another surface and are excluded) |
| `outlier_fraction` | 0.0 | MEASURED |
| grazing (`grazing_start_deg`, `grazing_cutoff_deg`), specular (`specular_cone_deg`, `specular_dropout_probability`), random dropout, erosion, incidence gain, depth jump, duplicate-skin rotation and patch, view jitter, triangulation angle, working distance | E2 defaults | ASSUMED |

**The card is PARTIAL.** Amendment 4b asks for grazing and specular dropout, which an exported,
merged mesh without sensor poses cannot identify. The G1 scorecard is therefore non-qualifying
(`SCANNER_MODEL_PARTIAL`, listing `grazing_start_deg`, `grazing_cutoff_deg`, `specular_cone_deg`,
`specular_dropout_probability`) until those are measured from raw per-view exports or the cell's
sensor poses, or the owner records a WAIVER for this model sha and those fields. The low noise floor
and the flare lower bounds make synthetic G1 easier than an assumed sensor would; that is what the
line receives from the cell, not an idealised sensor, but A1 must read the G1 numbers with it.

Method (validated on synthetic two-skin plates with known noise, spacing, doubled skins, flare,
outliers and an upstand; `tests/test_prod_engine_eval_card.py`): spacing = median NN distance (E2's
statistic); noise floor = robust σ of point-to-plane residuals in 2 mm same-skin patches (vertex
normals within 14°, normal spread p90 ≤ 4°), area-uniform seeds; skin separation = opposite-normal
points within 2 mm laterally; two-layer patches = gap > 6 pooled σ with minority ≥ 15 %; flare at
free edges (a wall vertex whose two skins lie behind it), following only wall points contiguous
with the skin slab, a column running more than 1 mm beyond the slab being another surface
(excluded and counted). The 5/10 mm residuals stay close to the 2 mm floor: no large-scale
waviness beyond white noise.

**Evidence (verbatim; rebuild without reading any scan):**
`model-card --from-evidence docs/production/ENGINE_EVAL.md --measured-on 2026-10-17 --out card.json`
reproduces the committed card byte for byte (a test checks it through both entry points).

```json scanner-card-evidence
{
 "assumed_fields": [
  "noise_incidence_gain",
  "edge_erosion_probability",
  "depth_jump_mm",
  "grazing_start_deg",
  "grazing_cutoff_deg",
  "specular_cone_deg",
  "specular_dropout_probability",
  "random_dropout_probability",
  "outlier_range_mm",
  "duplicate_skin_rotation_deg",
  "duplicate_skin_patch_mm",
  "view_jitter_mm",
  "triangulation_angle_deg",
  "working_distance_mm"
 ],
 "card_build": "e4a-scanner-card-build-1.1.0",
 "card_sha256": "cf55a20c23f37e9076426f0be559584fa9f00941db5d856ddc109bac4cabf8ce",
 "field_status": {
  "depth_jump_mm": "ASSUMED",
  "duplicate_skin_fraction": "MEASURED",
  "duplicate_skin_offset_mm": "UPPER_BOUND",
  "duplicate_skin_patch_mm": "ASSUMED",
  "duplicate_skin_rotation_deg": "ASSUMED",
  "edge_erosion_probability": "ASSUMED",
  "edge_flare_mm": "LOWER_BOUND",
  "edge_flare_probability": "LOWER_BOUND",
  "grazing_cutoff_deg": "ASSUMED",
  "grazing_start_deg": "ASSUMED",
  "noise_incidence_gain": "ASSUMED",
  "noise_sigma_mm": "MEASURED",
  "outlier_fraction": "MEASURED",
  "outlier_range_mm": "ASSUMED",
  "point_spacing_mm": "MEASURED",
  "random_dropout_probability": "ASSUMED",
  "specular_cone_deg": "ASSUMED",
  "specular_dropout_probability": "ASSUMED",
  "triangulation_angle_deg": "ASSUMED",
  "view_jitter_mm": "ASSUMED",
  "working_distance_mm": "ASSUMED"
 },
 "measured": {
  "duplicate_skin_fraction": 0.0,
  "duplicate_skin_offset_mm": 0.041438,
  "edge_flare_mm": 0.934258,
  "edge_flare_probability": 0.072993,
  "noise_sigma_mm": 0.004564,
  "outlier_fraction": 0.0,
  "point_spacing_mm": 0.401182
 },
 "measured_on": "2026-10-17",
 "scans": [
  {
   "duplicate_layer_fraction": 0.0,
   "duplicate_layer_gap_mm": null,
   "edge_flare_fraction": 0.02587646076794658,
   "edge_flare_p95_mm": 0.8875452023445365,
   "edge_samples": 2396,
   "edge_samples_complex": 12,
   "face_count": 208905,
   "has_normals": true,
   "name": "15651004_01.ply",
   "nn_spacing_mm": 0.27664625937750287,
   "noise_patches": 1538,
   "noise_sigma_mm": 0.0035993023992898243,
   "outlier_fraction": 0.0,
   "peak_rss_bytes": 96583680,
   "seconds": 1.98,
   "sha256": "03395097cd9662030e15f84411ad37365e903325a757cbc24b3ff2ac6bf4a6fc",
   "skin_separation_median_mm": 1.92974847804207,
   "skin_separation_p5_p95_mm": [
    1.874018937674293,
    1.9901974188568767
   ],
   "skin_separation_patches": 1527,
   "skin_separation_sigma_mm": 0.023543538486224258,
   "vertex_count": 104543,
   "waviness_10mm": 0.005365457389765448,
   "waviness_5mm": 0.004123227719783408
  },
  {
   "duplicate_layer_fraction": 0.0,
   "duplicate_layer_gap_mm": null,
   "edge_flare_fraction": 0.1278772378516624,
   "edge_flare_p95_mm": 0.166414929309429,
   "edge_samples": 1955,
   "edge_samples_complex": 0,
   "face_count": 350004,
   "has_normals": true,
   "name": "28870000_A_FULL_SCAN.ply",
   "nn_spacing_mm": 0.30575355653213127,
   "noise_patches": 2356,
   "noise_sigma_mm": 0.014446691875282396,
   "outlier_fraction": 0.0,
   "peak_rss_bytes": 124420096,
   "seconds": 3.45,
   "sha256": "38091ac91d9de3da6a8bc175fc340824b1f4d7c11b1c74e5a39c9af7c43ba2c8",
   "skin_separation_median_mm": 1.4600721620209693,
   "skin_separation_p5_p95_mm": [
    1.4373386569706172,
    1.491258523479832
   ],
   "skin_separation_patches": 2357,
   "skin_separation_sigma_mm": 0.014212585274767256,
   "vertex_count": 175182,
   "waviness_10mm": 0.012527126621207968,
   "waviness_5mm": 0.009494582094261874
  },
  {
   "duplicate_layer_fraction": 0.0,
   "duplicate_layer_gap_mm": null,
   "edge_flare_fraction": 0.1458106637649619,
   "edge_flare_p95_mm": 0.8114118604036591,
   "edge_samples": 919,
   "edge_samples_complex": 1087,
   "face_count": 943568,
   "has_normals": true,
   "name": "23553000_B_FULL_SCAN 1.ply",
   "nn_spacing_mm": 0.503489337564914,
   "noise_patches": 2175,
   "noise_sigma_mm": 0.004811393343871126,
   "outlier_fraction": 0.0,
   "peak_rss_bytes": 216858624,
   "seconds": 2.28,
   "sha256": "6eba02723d4b6994630fc9210497c5841f373e87397e132974bf7acdea5fe511",
   "skin_separation_median_mm": 0.9211423134114053,
   "skin_separation_p5_p95_mm": [
    0.7993686306580181,
    1.0326180157573863
   ],
   "skin_separation_patches": 2171,
   "skin_separation_sigma_mm": 0.04181976994403089,
   "vertex_count": 473764,
   "waviness_10mm": 0.006318688938047829,
   "waviness_5mm": 0.005752669418121544
  },
  {
   "duplicate_layer_fraction": 0.0,
   "duplicate_layer_gap_mm": null,
   "edge_flare_fraction": 0.06906474820143885,
   "edge_flare_p95_mm": 0.9244561610969995,
   "edge_samples": 695,
   "edge_samples_complex": 158,
   "face_count": 1429620,
   "has_normals": true,
   "name": "38172000_01_SCAN_1.ply",
   "nn_spacing_mm": 0.5011141940730781,
   "noise_patches": 2557,
   "noise_sigma_mm": 0.004564299799492988,
   "outlier_fraction": 0.0,
   "peak_rss_bytes": 321617920,
   "seconds": 2.37,
   "sha256": "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880",
   "skin_separation_median_mm": 1.4872636655043776,
   "skin_separation_p5_p95_mm": [
    1.3391650123000058,
    1.583355485590636
   ],
   "skin_separation_patches": 2557,
   "skin_separation_sigma_mm": 0.07775354464930087,
   "vertex_count": 715498,
   "waviness_10mm": 0.005346034051235607,
   "waviness_5mm": 0.0050558519121521855
  },
  {
   "duplicate_layer_fraction": 0.0,
   "duplicate_layer_gap_mm": null,
   "edge_flare_fraction": 0.072992700729927,
   "edge_flare_p95_mm": 0.9472294587405287,
   "edge_samples": 548,
   "edge_samples_complex": 375,
   "face_count": 1701535,
   "has_normals": true,
   "name": "10839000_01_full_scan.ply",
   "nn_spacing_mm": 0.4011820358244799,
   "noise_patches": 1995,
   "noise_sigma_mm": 0.004375668903701452,
   "outlier_fraction": 0.0,
   "peak_rss_bytes": 449265664,
   "seconds": 2.39,
   "sha256": "71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215",
   "skin_separation_median_mm": 1.4413514475870182,
   "skin_separation_p5_p95_mm": [
    1.3224389623882014,
    1.486448889666389
   ],
   "skin_separation_patches": 1454,
   "skin_separation_sigma_mm": 0.04143811665188206,
   "vertex_count": 863115,
   "waviness_10mm": 0.007226667574262468,
   "waviness_5mm": 0.00579655413647181
  }
 ],
 "schema": "sherman_e4_scanner_card_evidence_v1",
 "tool": "e4a-scanner-card-1.0.0"
}
```

## 11. Verification (2026-10-17, fix stage)

- **Tests** (named files only, `PYT` of plan §4, final code): `test_prod_engine_eval_scorer.py` (33),
  `_runner.py` (17), `_render.py` (21), `_line_metric.py` (6), `_freeze.py` (9), `_card.py` (8):
  **94 passed in 109 s**, peak RSS 1.17 GB. Every gate is shown to PASS on exact truth and to FAIL on
  the defect it guards against. The fix stage added a mutate-and-check test for each review
  finding: a PASS on a truly unformed bend and on a mirrored part counted as false PASS; a
  symmetric flip caught with and without published lines; measured bends without identity evidence
  NOT_EVALUABLE; a 25 mm misregistration POSE_WRONG; `regression_verdict` PASS on an all-FULL subset
  and FAIL for a silent or timed-out engine; SCANNER_MODEL_PARTIAL and its owner waiver;
  NON_STANDARD_* run conditions; LEDGER_NOT_ANCHORED and QUAL_SEEDS_REUSED; `g1_verdict`'s PASS
  path and its engine/ledger identity checks; the measured-angle apex (90° vs 150°) and its
  fallbacks; aggregate parity with the script's `evaluate()` including a silent engine (F1 0,
  FAIL); LINE_SHIFT ≥ LANDS tolerance + 3 mm on 10.8 mm and 13.6 mm bends; E1-only ambiguity
  (justified on a symmetric U, unjustified on an asymmetric hat and on a U whose implied stage S1
  is unambiguous); unjustified NOT_YET_FORMED and UNKNOWN close-ups, a non-canary mistag; E3-only
  gating, the synthetic-sample rule and refused test-seed packets; opaque titles; metric-free
  DIMENSIONS with fitted arrows; label swaps that never show two identical labels; re-freeze,
  frozen-code change and waiver, external anchor, QUAL RUN entries; read-only staged inputs and an
  engine that writes to its input; transient compile failures not reused (this test found a resume
  bug, fixed: the resume key now includes the compile outcome); QUAL policy from the manifest or
  any case spec; `score` provenance and re-scoring after an eval change without an engine rerun;
  the committed card rebuilt byte for byte from §10's evidence.
- **Regression tier** (50 DEV cases, E2 1.1.0, 7 families, final code) through `run` with the
  oracle engine and an anchored scratch freeze ledger: 50/50 runs OK in 194 s including 50 E1
  compiles (engine wall sum 24.5 s, slowest run 1.5 s, at most 0.24 GB per run; parent 0.70 GB).
  Every applicable binding gate PASS on exact truth, `verdict` INCOMPLETE because
  `G1.OCCLUSION_RAISES_UNKNOWN` has no degraded-plan case in this tier (flagged not applicable),
  `regression_verdict` PASS. Pose check: 48 POSE_OK, 2 undefined parts not evaluated. Correctly
  **not qualifying**: TIER_REGRESSION, SEEDSET_DEV, SCANNER_ASSUMED, NON_STANDARD_ENGINE_CMD,
  NON_STANDARD_ENGINE_ENV, ORACLE_RESULTS, FREEZE_LEDGER_MISMATCH (the tier was generated with the
  ASSUMED scanner model, not the frozen card). `score` rebuilt the same card (0 records re-scored).
- **Re-scoring and freeze protection, live:** `score` on the same tier's earlier run directory
  (scored by the pre-final eval code) re-scored all 50 records from their stored outputs without
  rerunning the engine, and its ledger check reported `EVAL_CODE_CHANGED_SINCE_FREEZE` and
  `THRESHOLDS_CHANGED:G4`, because that freeze predated the last eval edits.
- **Review probes on the final code:** symmetric case with every line withheld: PASS before, now
  FAIL (2 wrong ids) under a 180° flip and PASS with the right pose; a flat bend claimed FORMED at
  nominal: `G1.FALSE_PASS` 0 before, now 4 (`NOT_YET_FORMED`); every no-line close-up answered
  unjustified: G4 PASS before, now FAIL (`NOT_YET_FORMED_NOT_JUSTIFIED`); LINE_SHIFT margins over
  8 packet seeds ≥ 3.05 mm (before: 10.23 mm on a 10.8 mm-tolerance bend); FREEZE / move
  `coverage_min` / re-FREEZE: now `THRESHOLDS_REFROZEN:G1` + `THRESHOLDS_CHANGED:G1`; zero
  predictions on usable Gold: now F1 0, FAIL.
- **Integration probe with the in-development E3** (`python -m domains.production.engine.inspect`)
  on three DEV regression cases: three valid, consistent results (2.7 s, at most 0.25 GB each),
  POSE_OK on all three, renders of a real E3 output with the apex at the measured angle. That
  three-case card FAILs `G1.ACC.FLANGE_LENGTH.BIAS` (0.050 mm on 6 rows) and both median-U gates
  (0.158 mm, 0.256°): E3's state on three cases, a harness check, not engine evidence.
- G5 manifest check (`scripts/production/g5_check.py --only manifest`): PASS, no change under
  `backend/**`, `domains/bend/**` or `scripts/**`. AST unused-import check clean; every file
  compiles; no bytecode left in the eval package. The scanner card was rebuilt from the saved
  evidence (no scan was re-read in the fix stage).

## 12. Open issues

- **Scanner card is PARTIAL (owner/C12).** Grazing and specular dropout (and random dropout,
  erosion, incidence gain, sensor geometry) cannot be identified from exported merged meshes. Every
  G1 scorecard stays non-qualifying (`SCANNER_MODEL_PARTIAL`) until raw per-view exports or the
  cell's sensor poses are measured, or the owner records a WAIVER for model sha
  `cf55a20c23f37e9076426f0be559584fa9f00941db5d856ddc109bac4cabf8ce` and those four fields.
- The card's noise floor (0.0046 mm, about 5 × below E2's assumed 0.025 mm), the flare lower
  bounds and the duplicate-skin upper bound make synthetic G1 easier than the assumed sensor; A1
  must read G1 numbers with that in mind (pass the card with `--scanner-model` for every G1 tier).
- **Freeze before A1 (integrator/A1).** Nobody has frozen yet. Run `freeze` (all five threshold
  documents, DEV and QUAL seeds, DEV_AUTO_CONFIRMED recipes, G2/G3/DEV-CHECK lists, the card, code
  fingerprints) and record the first FREEZE entry's `entry_sha256` outside the ledger. R1 owns
  `docs/CURRENT_STATUS.md`; E4a cannot write it. Every qualifying run then passes it as
  `--ledger-anchor`; an unanchored check never qualifies.
- After the freeze, any edit under `domains/production/engine/eval/**` or `contracts/production/**`
  makes scorecards non-qualifying (`EVAL_` / `CONTRACTS_CODE_CHANGED_SINCE_FREEZE`) unless the owner
  waives that exact fingerprint. Freeze only after wave-3 eval work is done, or plan for a waiver.
- **Wave 3 (not E4a):** the legacy control arm must build a `RenderScene` and line records
  (source LEGACY); the Gold transform recovery produces `GoldCase`; the real-part runner evaluates
  `G2_PHYSICAL_THRESHOLDS` and `G3_THRESHOLDS`; the G4 jobs.json must carry `population` and
  `incomplete` for every scan, and the synthetic G4 sample needs at least 20 E3 scans including an
  incomplete one.
- G4 rule classes REGISTRATION_RULE (in-plane degenerate pose), UNSUPPORTED_RULE and
  INVALID_INSPECTION are reported but not reviewer-gated: a picture cannot check them. G1 covers
  them on synthetic data; physical scans rely on G2(b), G2(e) and G3 coverage and failure counts.
  When the engine decodes no stage, AMBIGUOUS falls back to "any stage model" only if no stage
  matches its implied formed set.
- The pose check uses fixed identity limits (5°, 4 mm RMS over the panel outlines); it detects
  identity flips and gross misregistration, not accuracy. `G1.POSE_CHECK` is informational on
  non-symmetric cases.
- Carried over: E1 and E2 symmetry flags disagree on 6 of the 50 regression cases (E1 decides);
  the regression tier holds FULL plans only, so the occlusion gate binds on the full tier;
  `G1.SLOPE.BACKGAUGE_LENGTH` is informational; the family-cluster bootstrap has only 7 clusters;
  the G4 close-up stacks the SCAN and RECIPE CAD panels vertically (same camera and scale).
