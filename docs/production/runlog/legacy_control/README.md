# Legacy runtime control arm (E4b, 2026-10-17, wave 4)

The unchanged legacy worker (`backend/bend_analysis_worker.py` → `run_progressive_bend_inspection`),
run serially, under the shared geometry lock, on every frozen G2, G3 and repeatability scan (A1 and
DEV-CHECK), as the comparison arm of plan §5 G2(a) exact-part rate, G2(c) angle, G2(e) runtime and
G2(f) repeatability, and as the G4 visual baseline (amendment 8). Code:
`domains/production/engine/eval/freeze/legacy.py`. Consumed development material; never
unseen-family accuracy.

## Run conditions

- Survey contract §6a command line, `--tolerance-angle 1 --tolerance-radius 0.5` (the DEV recipes'
  defaults); opaque `--job-id/--part-id/--part-name` and opaquely named read-only input copies
  (blind mode: `use_part_truth_hints` stays False in the worker).
- `BEND_INSPECTION_CONFIG`, `SHERMAN_QC_F1_PROGRAM_STATUS_PATH`, `BEND_EXPECTED_OVERRIDES` and
  `BEND_FEATURE_POLICIES` point at absent files: built-in defaults (contract `config/NONE`); from
  the repo the worker would otherwise read `output/improvement_loop/best_config.latest.json`,
  which is outside this chunk's custody.
- One process at a time, `OMP_NUM_THREADS=1`, 600 s wall, **12 GiB process-group RSS** (immediate
  kill, recorded `RSS_LIMIT`): the line contract's limit for this worker (`SHERMAN_LINE_RSS_LIMIT_MB`),
  under the 16 GiB machine rule. The first passes ran with 4 GiB. A run of theirs that the 4 GiB cap
  did not kill and that peaked below 12 GiB is the run a 12 GiB cap gives (the cap only kills, and
  the worker is deterministic under fixed conditions), so `reusable_run` accepts it; the one run the
  4 GiB cap had killed (a DEV-CHECK case) was **re-run at 12 GiB in the freeze fix pass and completed
  (OK)**, so no run of the arm is a cap artefact any more. `SUMMARY.json` records the cap of every run
  (`conditions.rss_limit_gib_of_runs`).
- **Unary count model.** The worker loads `<repo>/output/unary_models/latest.json` by itself when that
  file exists (no environment override; the file is outside this chunk's custody and is never opened
  here). Whether it was applied is read back from every report (`structured_context` or per-match
  `unary_predictions`) and recorded per case (`legacy_case.json` `unary_count_model_applied`) and in
  `SUMMARY.json` (`conditions.unary_count_model`): it was applied in **0** of the successful reports
  of either audience.
- A failed run (`WORKER_ERROR`, `RSS_LIMIT`, `TIMEOUT`) measures nothing: every planned bend of
  the case counts as not measured by the legacy arm.
- Legacy code fingerprint (`backend` + `domains/bend`, every `*.py`):
  `3f445d5c8618b3b2229282edc8fb6b83f90b9ec5d3972f0e97f85fecf59f7c30`, unchanged since the first pass.

## Reuse (no work redone) and what was run

- **Freeze fix pass (2026-10-17 evening): 30 of the 31 runs reused, 1 re-run.** Each reuse is checked
  by `reusable_run`: the same scan sha256, the same CAD sha256, the same legacy code fingerprint, the
  same conditions (tolerances, wall limit, config absent, blind, opaque inputs), an RSS cap that never
  bound (not killed, peak below 12 GiB), inputs intact, and for a successful run the worker's report
  still in the builder's work area with its recorded sha256. Every re-parsed `legacy_case.json` is
  identical to the reviewed freeze apart from the new `unary_count_model_applied` field, and the
  renders were copied, not redrawn (every compiled part is byte-identical). `run.json` `reused.from`
  keeps the original run (29 runs of the first E4b pass, 2026-10-17 ~15:05 local; 2 of the second).
- **Second pass: 2 runs** (the Gold11 38172000 X and V scans, which joined G2 under the wave-4
  custody), both OK.
- **Determinism probe.** One first-pass case (`pc_03395097cd96`) was re-run under identical inputs
  and conditions into a separate directory: every bend match (status, angles, detected lines,
  assignment) and the raw-to-reference transform are identical (only run-identity fields of the
  report differ). The legacy worker is deterministic under fixed conditions, so a reused run is
  the run.
- **The broad baseline's runtime outputs were not reused for this arm.** They come from the same
  worker and command line but ran as the API runs it: the repo's default config resolution (not
  the absent-config built-in defaults of this arm) and a 9 GiB limit. On the 16 scans both ran
  successfully, the bend matches differ on 12 (e.g. a different measurement method on one bend,
  different detected lines on others) although the transforms agree, which, with the determinism
  probe, attributes the difference to the configuration. Mixing them would make the control arm
  non-uniform. They are used, labelled with their conditions, as the BROAD reference lane
  (`docs/production/runlog/freeze/broad_reference_lanes_A1.json`), where this arm did not run.

## Files (per case `pc_<scan sha12>/`; DEV-CHECK under `devcheck_SEALED/`)

- `run.json`: status, worker error, wall time, peak RSS, conditions (with the RSS cap of that run),
  scan and CAD sha256, output paths (the worker's outputs, in the freeze builder's work area
  `builder_SEALED/legacy/<case>`, which A1 never reads), the report's sha256 and, for a reused run,
  `reused.from` / `reused.why`.
- `legacy_case.json` (`parse_legacy_report`): alignment, frame check (worker reference mesh vs
  E1's final sample cloud), every legacy match, and per recipe bend the mapped legacy bend, the
  legacy angle (`legacy_delta_vs_e1_deg` = measured minus E1's included nominal folded like the
  worker, `max(theta, 180 - theta)`; `legacy_delta_deg` against the worker's own CAD target is a
  diagnostic), whether the legacy line is a scan fit (placeholders never count), and the
  tip-to-tip edge distance (a proxy, not an OML flange length); `line_predictions_raw` are the
  scan-fit lines in the raw-scan frame for the G2a line metric; `compiled_part_sha256` of the frozen
  compile it was mapped to; `unary_count_model_applied` (see the run conditions).
- `renders/`: the E4a G4 renderer's global views and close-ups of the LEGACY scene
  (`legacy_render_scene`), palette re-encoded; `render_manifest.json` lists them with sha256.
  For a reviewer packet build `RenderJob(engine_dir=None, compiled_json=None, recipe_json=None,
  case_ref=..., source="LEGACY", population="PHYSICAL_DEV", scene=legacy_render_scene(...))` so
  the legacy arm is rendered and reviewed exactly like E3 (it is reported, never gated).
- `SUMMARY.json`: one row per case (with its RSS cap and unary-model flag), status counts, reused-run
  count, P95 wall time, max peak RSS, the run conditions, and the arm's own baseline on the frozen
  lists (G2a line metric against the human Gold11, G2(b) angle coverage, known-good angle deviation).

## A1 numbers (no engine involved)

23 runs (all reused, all under the earlier 4 GiB cap, none killed): 15 OK, 8 WORKER_ERROR
(registration: "Invalid raw scan to reference transform", "insufficient shared registration
fitness"); P95 wall 41.6 s; max peak RSS 2.74 GiB. Baseline on the A1 lists: G2(b) angle coverage
31 / 91 = 0.34; known-good angle MAE vs E1 nominal 1.29° on 16 bends; G2(a) against the 9 A1 Gold11
cases: line-F1 0.088 (3 TP, 6 FP, 56 FN), precision 0.33, exact-part rate 0.
