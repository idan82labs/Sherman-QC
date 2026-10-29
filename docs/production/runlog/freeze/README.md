# Pre-A1 freeze (E4b, 2026-10-17, wave 4)

Plan of record: `docs/production/PRODUCTION_LINE_PLAN.md` §5 and §10 (amendments 4, 6, 7, 8 bind
here), plus the owner's wave-4 directions of 2026-10-17 (`owner_directions.json`). Code:
`domains/production/engine/eval/freeze/` (CLI `python -m domains.production.engine.eval.freeze.prefreeze`,
docstring = full description). Everything in this directory is **consumed development** material
(engineering ledger); no number derived from it is unseen-family accuracy.

This is the freeze as fixed after the independent review of 2026-10-17 (findings F1-F9). The ledger
was replaced before A1 (`ledger --replace-before-a1`: it held no RUN or WAIVER entry), so
`ANCHOR.txt` holds a new anchor.

## Binding rules for A1 (the accuracy loop)

1. **Work in a fresh work directory of your own** (outside the repository). Never use the freeze
   builder's work directory (`/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e4b_work`):
   everything in it lives in its work area `builder_SEALED/` (pre-split and superseded material,
   legacy worker outputs, reviewer-blind placements, earlier passes, logs). The A1-facing commands
   (`compile-frozen`, `run-physical`, `score-physical`) refuse a work directory that holds
   `builder_SEALED/` and, without `--include-sealed`, a `*SEALED*` one or one holding
   `devcheck_SEALED/`.
2. Run, read and score **only** the cases of `cases_A1.json` (and its id lists). Every other
   authorized scan may belong to DEV-CHECK. The A1-facing commands read the A1 audience only
   (`cases_A1.json`, `recipes/`); a DEV-CHECK case document is refused before it is opened.
3. **Never read a path containing `SEALED`**: `devcheck_SEALED.json`,
   `gold_transforms_devcheck_SEALED.json`, `broad_reference_lanes_devcheck_SEALED.json`,
   `recipes_devcheck_SEALED/`, `../legacy_control/devcheck_SEALED/`,
   `../g3_truth_material/devcheck_SEALED/`, the builder's `builder_SEALED/`, any `devcheck_SEALED/`
   of a work directory. DEV-CHECK runs **once**, at the end, after the engine is frozen (amendment
   4f); G2 and G3 must hold on it too. Only the final DEV-CHECK run and R1 pass `--include-sealed`.
4. Do not edit `domains/production/engine/eval/**` or `contracts/production/**`: their code
   fingerprints are frozen in `LEDGER.json`; a change makes every scorecard non-qualifying unless
   the owner waives that exact fingerprint (E4a `waive`, owner only).
5. Thresholds never change (amendment 3). The engine config is A1's final knob: freeze the final
   config with a later FREEZE entry (`python -m domains.production.engine.eval freeze --ledger
   docs/production/runlog/freeze/LEDGER.json --label ... --engine-config C.json`); it is compared
   with the latest FREEZE entry that defines one.
6. Every qualifying run passes `--ledger-anchor <first line of ANCHOR.txt>`. The anchor must also be
   recorded outside this directory (R1: `docs/CURRENT_STATUS.md`).
7. **Amendment 7 order, enforced.** The G3 visual truth table (`../g3_truth_material/`) is filled by
   two fresh reviewers plus a tie-break, frozen and recorded in the ledger
   (`prefreeze record-truth-table`) **before any engine output on the truth-relevant scans exists**:
   the G3 and BROAD staged scans and the provisional-complete G2 scans (13 A1 cases, every one with
   section images). Until then `run-physical` refuses E3 on those cases (they are listed, exit 2)
   and runs the others; the scorer rejects a table frozen or recorded after the first E3 record of
   a truth-relevant case (G3 NOT_EVALUABLE, `final: false`). The `STATE_CHECK` rows of the other
   complete scans are a completeness check that no score reads. The engine never receives stage
   keys, expected counts, file names or part numbers (amendment 4; owner: no
   SKU/family/part-id/filename-dependent code, thresholds or special cases).
8. G2(a) is reported against **both** the human Gold11 (`gold_g2a_A1.json`, the gate of record)
   and the Gold11 audit relabel (`gold_g2a_relabel_A1.json`, report-only); scoring re-reads only
   the A1 relabel files listed in `cases_A1.json` (`relabel_files`) and records whether they or the
   count of the other files changed. The relabel never replaces the human set without the owner.

## How A1 evaluates the physical gates

```
PY="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze"
W=<a fresh directory of your own, outside the repository>
$PY compile-frozen --work $W        # once: the 18 A1 recipes into $W/compiled/dev/<sku>, sha256 checked against the freeze
$PY run-physical --work $W --cases docs/production/runlog/freeze/cases_A1.json [--list G2_COMPLETE] [--case pc_...] [--engine-config C]
$PY score-physical --work $W --cases docs/production/runlog/freeze/cases_A1.json \
    --ledger-anchor <first line of ANCHOR.txt> --out $W/physical_scorecard.json
```

Every heavy command runs under the shared lock (`until mkdir /private/tmp/claude-501/sherman_geom.lock
2>/dev/null; do sleep 3; done; <cmd>; rmdir /private/tmp/claude-501/sherman_geom.lock`), one at a time.
`run-physical` drives the E3 command line under the standard conditions (180 s, 4 GiB, budget 170 s,
opaque read-only inputs) after the amendment-7 gate; `score-physical` applies the frozen G2 (b-f,
4d compile rate, 6a line metric with the legacy arm and the relabel dual report) and G3 rules (the
frozen truth table `../g3_truth_material/truth_table_FROZEN.json` is used when it exists) and adds
the report-only BROAD block (family-macro line-F1 of the engine and of the reference lanes, the
recipe ceiling, per-bend checks; owner aim 0.80). Without the frozen truth table the scorecard says
`final: false` and G3 is NOT_EVALUABLE; `qualifying` also needs the ledger check and the anchor.

Recording the frozen truth table (whoever assembles it; DEV-CHECK's at the final run or by R1 only):
`$PY record-truth-table --truth-table docs/production/runlog/g3_truth_material/truth_table_FROZEN.json`
(add `--include-sealed` for the DEV-CHECK table). It refuses an unfrozen table, a wrong label or
`final`, and a table that misses or adds a template row; it appends one FREEZE entry (the A1 table
by path and sha256, the DEV-CHECK table by sha256 only).

## Contents

| File | What |
|---|---|
| `LEDGER.json`, `ANCHOR.txt` | E4a hash-chained ledger with the pre-A1 FREEZE entry (items, decisions, open items); the anchor (sha256 of the first FREEZE entry) |
| `thresholds_plan_verbatim.md` | Plan §5 and §10 byte for byte, with the plan's sha256 and line ranges |
| `thresholds_documents.json` | E4a's five threshold documents (G1, G2_LINE, G2_PHYSICAL, G3, G4) with their sha256 |
| `owner_directions.json` | The owner's wave-4 directions that bind the freeze, as relayed to E4b |
| `inventory.json` | Every file of `../source-assets/02_CAD_QC_3D` plus the external CONSUMED files, hashed and checked against the manifest (A1 and DEV-CHECK alike: it seeds the draw; see "DEV-CHECK") |
| `gold_transforms.json` | Amendment 6a, A1 part: the Gold rows of A1 families, every count, the excluded list, and the sha256 and time of the full list |
| `compile_report.json` | Amendment 4d: recipe build + E1 compile of every unique authorized STEP (three recipe variants, two denominators), the gate under rule E4B-4D-GATE-2 and the per-variant floor check |
| `recipes/` | The 18 frozen DEV_AUTO_CONFIRMED recipes (canonical JSON, opaque SKU `DEV-<cad sha>`) of A1 families and CAD without scans |
| `cases_A1.json` | Every A1 physical case (lists, recipe, compile status, scoring-only stage keys, BROAD rows), repeatability sets, Gold / relabel / BROAD gold (with recipe-count flags), the A1 relabel files, the recipe ceilings, excluded scans |
| `g2_complete_A1.json`, `g3_staged_A1.json`, `known_good_A1.json`, `repeatability_A1.json`, `broad_A1.json`, `broad_staged_A1.json` | Case-id lists |
| `gold_g2a_A1.json`, `gold_g2a_relabel_A1.json`, `gold_broad_A1.json` | GoldCase records (raw-frame scoring): human Gold11 (gate of record), the relabel (report-only, with its A1 file snapshot), the BROAD gold (with the recipe ceiling) |
| `broad_reference_lanes_A1.json` | The broad baseline's own line predictions on the A1 BROAD cases (shipped runtime and old best lane), scored as reference arms |
| `devcheck_SEALED.json`, `gold_transforms_devcheck_SEALED.json`, `broad_reference_lanes_devcheck_SEALED.json`, `recipes_devcheck_SEALED/` | DEV-CHECK and the full amendment-6a list (sealed; in the ledger by sha256 only) |

## Decisions and findings recorded at the freeze

- **Custody (owner wave-4 directions).** Authorized: the source tree; the Gold11 dev cases and line
  refs (the nine script-located jobs and the two `shaked_xv` C05/C06 label files); the scans and
  STEP CAD of parts marked CONSUMED in `BENCHMARK_V2.json` and the BROAD case list; the AI gold of
  `ai_bend_lines_v1`; the Gold11 relabel; corpus stage keys (scoring only). Refused by the guard:
  HOLDOUT10, sealed truth, N001-N016, EHO4 parts, authorization / one-shot material, other
  workflows' `*SEALED*` paths, `incoming_codex_fetch_2026-10-17`, rhino-preview.
  `17139000_01.ply` is BENCHMARK_V2 eval tier `FRESH_SCAN_OF_CONSUMED_PART`: it is excluded from
  every list and its geometry was never read or used, but like every source-tree file the inventory
  hashed it and read its PLY header (`inventory.json` records its sha256, vertex and face counts and
  exporter comment). It is not re-inventoried, because the inventory sha256 seeds the DEV-CHECK draw.
  (The first report's "never read" was wrong; corrected here.)
- **Gold (6a).** 11 of 11 Gold11 cases recovered, 71 lines, residual 0.0 mm (nine display clouds are
  exact raw-vertex subsets; the two 38172000 X/V label sets are sha256-bound `scan_raw`, Codex
  multiview, drawing-backed, flagged `GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN`); 0 excluded. The
  script-located human non-Gold11 label jobs whose scans are BROAD cases are BROAD gold, never G2(a)
  gold. The full list (`gold_transforms_devcheck_SEALED.json`, written 2026-10-17T19:14:37Z, before
  any list, lane or scoring) names every Gold case, so it is sealed; `gold_transforms.json` is its A1
  part, derived after the draw, with the full list's sha256 and time (it lists the 10 rows of A1
  families; 3 rows are sealed). G2(a) for A1: 9 cases / 59 lines. The relabel (status
  COMPLETE_PENDING_OWNER_SIGNOFF) covers the 9 A1 cases with 77 lines; one A1 relabel file was
  revised by the audit workflow after the reviewed freeze and is frozen as it stands now.
- **Compile rate (4d), rule E4B-4D-GATE-2, decided before A1 and flagged for the owner's /
  orchestrator's confirmation before the anchor is published.** Amendment 4d puts the 80 % floor on
  "the DEV_AUTO_CONFIRMED recipes: every V10/native bend": the frozen DEV recipes (E1-DEV-POLICY-1,
  the V10/native union the evaluation runs on) are gated on both denominators: **24 / 26 = 0.923**
  (every authorized unique STEP) and **19 / 20 = 0.95** (plan §9 source tree), PASS. The strict
  single-source figures are reported as diagnostics, never gated: strict V10 20 / 26 = 0.769 (16 /
  20 = 0.80 on the source tree), native 18 / 26 (14 / 20). The first pass gated strict V10 too, which
  made G2 FAIL by construction on A1 and on DEV-CHECK whatever the engine does (strict V10 counts as
  "the amendment-4d figure" only in E1's own ENGINE_COMPILE.md §10); that alternative is recorded in
  the ledger as not taken. Failures stay in every denominator: 47479000_C (rolled shell,
  DATUM_PANEL_UNDETERMINED), 37126000_D (no candidate bend from either extractor, NO_RECIPE). All
  compiled parts are byte-identical to the first pass.
- **Lists (A1).** G2_COMPLETE 15, G3_STAGED 11, KNOWN_GOOD 6, REPEATABILITY 3 (15651004 x3,
  evaluable), GOLD_G2A 9, BROAD 26 (14 families, 144 gold lines, 2 zero-bend controls), BROAD_STAGED
  0. 38172000 X/V joined G2 (complete parts, Gold11). 15651004 SECOND/3RD are `provisional_complete`
  (corpus stage keys say partial): they stay in G2 only if the frozen visual truth table marks every
  bend FORMED. The fix pass re-ran `cases`: the draw (seed, strata, families), every list, every
  case record, every human Gold11 and BROAD gold line and all 25 recipes are identical to the
  reviewed freeze. New: the recipe-count flags and ceilings, the per-audience relabel file lists.
  Changed: the A1 relabel snapshot (the one revised file) and the wording of the `17139000_01.ply`
  exclusion reason and of the superseded draws' notes.
- **BROAD.** 36 of the 60 broad cases qualify (STEP that compiles, PLY/STL scan); excluded and
  listed: 21 without STEP CAD, one OBJ copy (the line takes PLY/STL), 37126000_D (CAD does not
  compile), 47479000 (blocked). Gold kinds: label jobs via 6a, AI gold sha256-bound to the scan; a
  gold file that changed since the case list is used as it is today and flagged
  `GOLD_FILE_CHANGED_SINCE_CASE_LIST`. Reference lanes on the A1 BROAD gold (no engine involved):
  the old best research lane scores family-macro line-F1 0.543 (micro 0.508), the shipped runtime
  0.279 (micro 0.121); owner aim 0.80. The runtime lane ran with the repo-default config, not the
  legacy-arm conditions (labelled as such).
- **Recipe ceiling (report-only, before any run).** A recipe-guided engine publishes at most one line
  per recipe bend. One rule for every gold case (no part is named): gold with more lines than the
  frozen DEV recipe has bends is flagged `GOLD_LINES_EXCEED_RECIPE_BENDS` with its ceiling and E1's
  recorded cause, `GOLD_CONVENTION_CONFLICT` (the policy merged extractor pieces of one fold, the
  owner's one-line convention, while the gold labels the pieces) or `RECIPE_EXCLUDED_FEATURE` (E1's
  frozen UNSUPPORTED rule excluded a feature); a complete scan whose gold has fewer lines than the
  recipe has bends is flagged `GOLD_LINES_FEWER_THAN_RECIPE_BENDS` (no hard ceiling, a precision
  exposure). On the A1 BROAD gold the best reachable family-macro line-F1 is **0.967** (micro
  0.990): 11979003 (9 gold lines, 7-bend recipe, convention conflict) is capped at 0.875 and
  28870000 (2 lines, 1-bend recipe, an UNSUPPORTED rolled-feature joint) at **0.667, below the 0.80
  aim, whatever the engine does**. The Gold11 G2(a) gold is not capped (ceiling 1.0); three A1 Gold11
  cases have fewer gold lines than recipe bends (10839000 full: 8 of 12; 38172000 X and V: 10 of 12),
  which bounds the precision of an engine that reports every bend. The cases stay scored as they
  are (comparable with the reference lanes); the scorer reports the ceiling next to the aim.
- **DEV-CHECK (4f), rule V4.** 7 of 22 families (0.32), stratified GOLD / STAGED / REPEATABILITY /
  OTHER with at least one Gold11 family and one staged family, seeded by
  `sha256("E4B-DEVCHECK-V4" | inventory sha256 | strata)`. 10839000 is never held out (the app E2E
  real part, amendment 10). Superseded before the freeze, with no E3 run on any physical case (the
  legacy control arm had run in the first pass): V2 (first pass; the BROAD families were added by
  the owner's directions) and V3 (its draw held no Gold11 family, so G2(a) would have been
  NOT_EVALUABLE on DEV-CHECK and amendment 4f unsatisfiable).
- **What the seal covers (corrected).** Every A1-facing file of this directory and of
  `../legacy_control/` and `../g3_truth_material/` names only A1 cases, with one documented
  exception: `inventory.json` hashes every authorized file, A1 and DEV-CHECK alike (scan and CAD
  sha256, BROAD ids), because it is the input that seeds the draw. Beyond the run logs, the plan
  text (`thresholds_plan_verbatim.md`) and the freeze code name the whole population: the plan's G3,
  known-good and repeatability lists and the 11 pinned Gold11 records (scan sha256, label jobs),
  all defined before the draw. DEV-CHECK identities are therefore inferable by omission from the
  public tree, as they are from the source tree and the broad case list themselves; the protection
  is procedural (rules 1-3). The first report's claim that no A1-facing file named a sealed scan
  sha256 was false (`gold_transforms.json` and `inventory.json` did); `gold_transforms.json` is now
  the A1 part. The reviewer-blind display placements of the G3 material moved out of
  `../g3_truth_material/` into the builder's work area (moved, not re-rendered; images, manifests and
  templates are unchanged).
- **Work directories.** The builder's work directory holds only `builder_SEALED/`: the full gold
  list `gold_cases_ALL.json`, `compile_report_full.json`, the recipe builds and compiles of every
  STEP, the legacy worker's outputs, the reviewer-blind placements, the earlier passes
  (`prev_pass1/`, `prev_pass2/` = the reviewed freeze), logs and dry runs. A1's and the final
  DEV-CHECK run's products live in their own fresh work directory (DEV-CHECK's under
  `devcheck_SEALED/`).
- **Legacy control arm.** 12 GiB process-group RSS (the line contract's limit); the one run the
  earlier 4 GiB cap had killed was re-run at 12 GiB and completed; the unary count model condition
  is recorded per run (applied in 0 reports). See `../legacy_control/README.md`.
- **Repeatability (6f).** The A1 set, 15651004 x3, is three meshes (evaluable). The other plan sets
  are recorded with their cases: a set of byte-identical copies is NOT_EVALUABLE, and one mesh
  exported as PLY and STL is a format pair.
- **Recipes** use `inspection_stage` FINAL for every scan (no stage hint reaches E3; staged scans'
  unformed bends must be decoded from geometry).
- **Scanner model card (4b).** E4a's card (5 authorized scans) is PARTIAL: grazing and specular
  dropout are ASSUMED. G1 qualification needs them measured or an owner `SCANNER_MODEL_PARTIAL`
  waiver (open item in the ledger).

## Custody of the passes

Read (hashes, headers, labels; geometry only through the legacy worker, the G3 display registration
and the Gold display-cloud checks of the first pass): the source tree; the nine Gold11 and two human
non-Gold11 label jobs; the C05/C06 label files; the CONSUMED BROAD scans and STEP CAD listed in
`BENCHMARK_V2.json` / `case_list_broad_v3.json`; `ai_bend_lines_v1` and
`ai_bend_lines_gold11_relabel_v1` label + provenance files; the broad baseline's frame records and
prediction files; corpus `metadata.json` stage keys. The fix pass read only run logs, the relabel
and AI gold files, the manifest and case list, the hashes of the BROAD scan and CAD bytes, and ran
the unchanged legacy worker on one scan and its STEP. Not read: HOLDOUT10, sealed truth, N001-N016,
EHO4, FRESH (codex fetch, `*_SEALED` labels), the geometry of `17139000_01.ply` (reserved),
authorization artifacts, the Gold catalog container, `output/improvement_loop`,
`output/unary_models`.
