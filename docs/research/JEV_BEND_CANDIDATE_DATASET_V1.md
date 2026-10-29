# Jev bend-candidate dataset and offline harness, v1 (C10)

Date: 2026-10-17. Code: `research/jev/`. Tests: `tests/test_prod_jev_*.py`.

**Status: research only.** Jev output is **not wired into any runtime decision**. Nothing in
`apps/`, `backend/`, `domains/`, `contracts/` or `frontend/` imports `research.jev`. No live call
was made while building this harness, and none may be made without the egress approval described
under [Egress policy](#egress-policy).

## 1. Purpose

[Jev](https://docs.typesafe.ai/models.md) (TypeSafe System One) returns typed, calibrated
judgments over a structured `state`. This harness prepares clean data to evaluate Jev **offline**
as a decision model for one question:

> Is this candidate a real press-brake bend line?

It is asked of two candidate families. CAD candidates are judged at onboarding. Scan and engine
candidates are judged at inspection. The harness produces leakage-free, custody-safe, grouped
datasets, request bodies with cost estimates, a dry-run-by-default client, and an evaluator with
frozen operating points and baselines.

Three later hooks are documented here but **not built**:

1. **Onboarding mark review.** Flag CAD candidates the operator kept or removed against a
   confident Jev answer, for a second look in the wizard.
2. **Unexplained creases.** Send E3 unplanned creases (the residual lane) to REVIEW with a Jev
   opinion attached. They already force REVIEW, and Jev can never clear one.
3. **Borderline formed/unformed calls.** Send these to REVIEW with a Jev opinion attached.

Each hook can only add a REVIEW. None can remove a measurement or turn UNKNOWN into PASS or FAIL.

## 2. Families and sources

| Family | `candidate_source` | Input | Frame |
|---|---|---|---|
| A. CAD onboarding candidate | `CAD_ONBOARDING` | `PartRecipeV1` JSON, or an onboarding bundle `{recipe, draft_marks?, compiled_part?, provenance_flags?}` (JSON or JSONL) | `cad_model_mm` |
| A'. Unmarked E1 flange-graph bend | `E1_FLANGE_GRAPH` | `compiled_part.unmarked_graph_bends` of a bundle | `cad_model_mm` |
| B1. Production engine (E3) candidate | `SCAN_E3` | `<job>/engine/result.json` (`ShopfloorInspectionResultV1`): planned bend rows and `unplanned_creases` | `cad_model_mm` |
| B2. Legacy worker candidate | `SCAN_LEGACY_REPORT` | `<job>/engine/bend_inspection_report.json` or `<job>/bend_inspection_report.json`: `matches` with a detection, and `unmatched_detections` | `reference_mesh_raw` |
| B3. Legacy F1 pool candidate | `SCAN_F1_POOL` | JSONL of F1 candidate-pool rows: the selector-v2 `BASE_FEATURE_NAMES` plus `span_endpoints` (or `anchor` with `axis_direction` and `span_mm`), `representative_source`, `candidate_id`, `case_key`, `group_sku` and `coordinate_frame` | declared per row |

`draft_marks` are the vendor-setup `BendMark`s of the released draft. Their CAD candidate geometry
is immutable, so confirmed and removed candidates are described from the same source. This is the
shape C05's benchmark export is expected to provide. C05 is documented here, not imported.

## 3. State schema (`jev_bend_candidate_state_v1`)

JSON Schema: `research/jev/schemas/jev_bend_candidate_state_v1.schema.json`. The code check is
`research.jev.state_v1.check_state`. A test asserts that both reject the same mutations.

```json
{"schema": "jev_bend_candidate_state_v1", "candidate_source": "CAD_ONBOARDING",
 "units_convention": "MM_DEG",
 "candidate":    {"bend_line_length_mm": 152.3, "nominal_sweep_deg": 90.12, "...": "..."},
 "neighbours":   {"parallel_2deg_within_5mm_count": 0, "nearest_parallel_offset_mm": 40.1},
 "context":      {"sheet_thickness_mm": 2.0, "cad_candidates_count": 7},
 "cad_relation": {"nearest_offset_mm": 3.0, "nearest_graph_sweep_deg": 90.12}}
```

The four blocks:

- **`candidate`** holds what the detector or engine observed, using only fields that exist for
  every candidate of that source.
- **`neighbours`** is a coordinate-free summary of the other candidates of the same part or scan:
  parallel-neighbour counts within 2 or 5 deg and 5 or 25 mm, the nearest parallel offset, and
  length ratios.
- **`context`** holds part- or scan-level facts. The CLI flag `--context-free` drops it.
- **`cad_relation`** holds everything whose value **or presence** depends on the CAD or on the
  binding to a recipe bend: nominal angles, deviations, assignment kind and scores, pipeline
  observability around a bound CAD bend, the nearest planned bend and `engine_confidence_score`
  for matched legacy rows. `--cad-free` drops this block, and that drop is the CAD-free ablation.

The deny-list is fail-closed: `build_state` raises instead of dropping. It applies these rules:

- The legacy `INFERENCE_FORBIDDEN_KEYS` are matched exactly at any depth. Geometry keys
  (anchors, line endpoints, axis directions, polylines, transforms) are matched the same way.
- Keys are denied when any `_`-separated token is an identity token: id, uid, sku, part, job,
  recipe, scan, file, path, timestamp, date, at, serial, station, sha, hash, name, label, case,
  seed, split, family, operator, truth, gold, manual, review, endpoint and others (`DENY_TOKENS`).
- Lists are rejected, so no 3-vector or coordinate array can appear.
- Strings must be short enum tokens matching `[A-Za-z][A-Za-z0-9_]{0,63}` with no run of four or
  more digits. That excludes SKUs, filenames, hashes, uuids and timestamps.
- Key classes:
  - a numeric key ends with a unit suffix (`_mm`, `_mm2`, `_deg`, `_count`, `_fraction`,
    `_ratio`, `_score`, `_pct`, `_per_mm`, `_eigenvalue`, `_quantile`, `_probability`, `_log1p`);
  - a boolean key starts with `is_` or `has_`;
  - a string key ends with a category suffix (`_kind`, `_state`, `_source`, `_method`, ...).
- Finite numbers are rounded to 4 significant digits. NaN and missing values become `null` with
  a sibling `<key>_missing: true`.
- Nesting is limited to 8 levels.
- The record id is `sha256(source_sha256 + "|" + candidate_key)[:24]` and is kept outside the
  state.

### Leakage guards (deliberate amendments to the survey feature list)

- **Onboarding.** Recipe-only fields exist only for confirmed bends, so their presence would
  encode the label. These are sequence, machine move, tolerances, `flange_inspection`,
  `extent_semantics` and side boundary kinds, and all are excluded. With `draft_marks`, every
  candidate is described from its CAD geometry: line length, target angle (sweep and included,
  per `cad.candidate_angle_convention`) and target radius. Thickness and the CAD candidate count
  are part-level.
  - The E1 relation uses the **whole** flange graph (marked and unmarked bends), which does not
    depend on the marks.
  - Unmarked graph rows never change a CAD candidate's neighbours or count. Their presence
    depends on the marks.
  - A test checks that confirmed and removed candidates have identical key sets.
- **Without `draft_marks`**, removed candidates have no geometry. Only confirmed bends are
  exported, and the manifest warns `NO_NEGATIVES`.
- **Legacy reports.** The `candidate` block (folded angle and both unfold candidates, radius and
  line length) is identical for matched and unmatched detections. Everything that differs goes
  into `cad_relation`.
  - **Placeholder matches are skipped.** A match with `assignment_source = CAD_LOCAL_NEIGHBORHOOD`
    or `detected_line_geometry_source = CAD_REFERENCE_PLACEHOLDER` carries the CAD line as its
    "detected" line (`backend/feature_detection/bend_detector.py`), and older reports may carry
    the target radius as `measured_radius`. So does any match whose detected line equals its CAD
    line. These rows are not exported, since that would copy nominal into actual. They are counted
    as `legacy_matches_cad_placeholder`.
- **E3 results.** The `candidate` block holds only `measured_included_deg` and
  `measured_line_length_mm`, the scan evidence a planned row and an unplanned crease both have.
  - Every per-bend engine field that exists only for planned rows goes into `cad_relation`:
    support counts and balance, coverage, angle fit RMS and support, bend and characteristic
    states, reasons and the flange-measured count.
  - The crease reason also goes there, because EXTRA_BEND_SUSPECTED and UNPLANNED_CREASE are
    decided against the recipe.
  - Planned bends without a measured polyline are skipped and counted
    (`e3_bends_without_measured_line`).
  - A test asserts that after the CAD-free drop, planned rows and creases have identical key paths
    and enum values.
- **The pipeline's observability or bend state** (`observability_state`, `engine_bend_state`) is
  a feature, never a label.

## 4. Questions (`research/jev/questions_v1.py`)

| id | type | content |
|---|---|---|
| `is_real_bend` (primary, always asked) | Noul | "Is this candidate a real press-brake bend line?" The `false` criteria list the confusers: outline, hole wall, rolled band, step, duplicate skin, tongue silhouette, artifact. |
| `candidate_kind` (optional) | Choice | bend, outline, hole_wall, rolled_band, step, duplicate_skin, tongue_silhouette, artifact, other |
| `evidence_quality` (optional) | Score | 4 levels, from almost no usable evidence to strong evidence |

There are two variants, `v1` and `verbalized_v1`, and each is hashed separately:
`question_set_sha256(questions, variant)`. Question ids are not sent to the model, so every
instruction stands on its own and refers to state fields by backticked path. Only the Noul is
asked by default. Add the others with `--questions is_real_bend,candidate_kind`.

## 5. Labels and weights (`labels.jsonl`, a sidecar never sent)

| `label_source` | label | weight | origin |
|---|---|---|---|
| `operator_confirmed` | 1 | 1.0 | active CAD candidate of a **released** (`released_at` set) recipe whose `confirmation` is `OPERATOR_CONFIRMED` (the contract default when absent) |
| `operator_removed_not_a_bend` | 0 | 1.0 | removed with issue `not_a_bend` |
| `operator_wrong_location` | null | 0 | removed with `wrong_location`: a real bend with wrong geometry, so it is excluded |
| `operator_removed_issue_missing` | null | 0 | removed with `missing` (ambiguous), so it is excluded |
| `operator_manual_missing` | 1 | 1.0 | a manual bend no candidate represents: a **recall row with no state** |
| `human_marker_real_bend` / `human_marker_not_bend_outline` / `human_marker_uncertain` | 1 / 0 / null | 1.0 | explicit marker reviews in frame `reference_mesh_raw` only |
| `gold_line_matched` / `gold_line_unmatched` / `gold_line_unmatched_incomplete` | 1 / 0 / null | 1.0 / 0.8 / 0 | F1 pool against gold lines (5 deg, 4 mm, 0.6/0.6 overlap). An unmatched candidate is a 0 only when the gold set is complete. |
| `synthetic_truth_real` / `synthetic_truth_not_bend` | 1 / 0 | 0.5 | E2 synthetic truth (engineering ledger only) |
| `unreviewed`, `dev_auto_confirmed`, `e1_unmarked_graph_bend` | null | 0 | not adjudicated, or self-referential |

Truth fails closed. Any other `confirmation` value, or a recipe without `released_at`, makes
every row of that recipe `unreviewed` and emits no recall rows. The manifest then warns
`UNTRUSTED_CONFIRMATION` or `RECIPE_NOT_RELEASED`. Bundle `provenance_flags` are limited to
`consumed` and `retrospective`. `holdout`, `sealed` and `one_shot` refuse the export with a
custody error, and any other value is an export error.

Weak labels live in their own column (`weak_label`, `weak_label_source`) and are **never
evaluation truth**:

- `recipe_binding_bound_formed` = 1: bound to a recipe bend, FORMED and measured.
- `recipe_binding_unbound_far` = 0: unbound and more than 10 mm from every recipe bend. The
  distance is the true segment-to-segment distance, so a long line that crosses a recipe bend is
  never "far". The state feature `is_within_10mm_of_reference` uses the same distance, with an
  axis angle of at most 10 deg.
- `dev_auto_confirmed_active` = 1: an active bend of a DEV_AUTO_CONFIRMED recipe. These recipes
  are built from the same detector, so their labels are circular.

`evaluation_truth(row)` is the only accessor the evaluator uses. Human labels for scan jobs are
read only from an explicit `--labels-root DIR`, from `DIR/<job>/bend_instance_labels.json` or
`DIR/<job>/jev_candidate_labels.json`, and the document's frame must match the source frame. A
label file inside a job workspace is ignored.

## 6. Splits (`jev_split_manifest_v1`)

```json
{"schema": "jev_split_manifest_v1",
 "families": {"<family>": {"split": "dev|calib|eval|excluded",
                            "flags": ["holdout", "sealed", "one_shot", "consumed", "retrospective"]}},
 "sku_to_family": {"<SKU>": "<family>"}}
```

- The family is the unit of assignment, so no family spans two splits. A SKU that maps to two
  families after canonicalisation is an error.
- A family flagged `holdout`, `sealed` or `one_shot` is **excluded**, whatever split the manifest
  names. The override is recorded.
- `consumed` and `retrospective` families keep their split, but any result that touches them is
  flagged `exploratory_not_unseen`.
- An unmapped SKU is `excluded`.
- Split and family names never appear in a state. A test scans the states for them.

## 7. Custody

- Inputs come only from explicit paths.
- Inputs under the repo's `data/`, `output/`, `outputs/`, `artifacts/`, `uploads/`,
  `live_scans/` or `research/` (except `research/jev/`) are refused unless `--authorization FILE`
  is given. That file is `{"schema": "jev_input_authorization_v1", "authorized_roots": [...]}`.
  The input must lie under a listed root, and the file's sha256 is recorded in the manifest.
- Path components matching holdout, sealed, EHO4, one-shot or the opaque N001–N014 names are
  refused even with an authorization.
- The check has three layers:
  1. It is lexical first, so a refused path is never opened or stat'ed. Protected directory names
     and the repo-root prefix are compared **case-insensitively**, because macOS volumes are
     case-insensitive and `DATA/x` is `data/x`. The `research/jev` exemption and
     `authorized_roots` match only exactly, so a case variant can only be refused, never let
     through.
  2. It is repeated on the resolved path, so a symlink cannot escape it.
  3. Every ancestor of the resolved path is compared with the protected directories by filesystem
     identity (`st_dev`, `st_ino`), which catches aliases the strings miss.
- The pattern match is deliberately broad: it also refuses an innocent directory whose name
  contains "holdout".
- The `--authorization` file must itself pass custody **without** an authorization before it is
  opened. It may not lie under a protected repo directory (`research/` siblings hold other
  protocols' one-shot and authorization artifacts) or have a never-readable path component. An
  `authorized_roots` entry with a never-readable component is refused too.
- `--labels-root` and `--jobs-root` are checked before anything inside them is listed or
  stat'ed.
- Exports go to a **new** `--out` directory. They are written into a temp sibling and renamed,
  so a failed export leaves nothing.
- Two runs over the same inputs are byte-identical. The manifest holds input shas and basenames,
  never timestamps.
- Tonight's build used synthetic records only. No file under `data/` or a real corpus was read,
  and no exporter was run on real data.

## 8. Requests, client and cost

Request shape (docs verified 2026-10-17, <https://docs.typesafe.ai/api.md>):
`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`, body
`{state, model, questions}`.

`request_v1.validate_request` enforces these limits:

- 32 MiB, UTF-8 without BOM, finite numbers only;
- nesting of at most 64 in each of `state`, `instructions` and `criteria`;
- Noul criteria with only `true` and `false`; Choice with 1 to 255 options; Score with 2 to 10
  levels;
- 64k tokens per request and 32k per question (state plus one question), estimated
  conservatively as `ceil(len(json)/3)`.

The client (`client_v1`):

- **Dry run is the default.** It writes `requests/<record_id>.json` (the exact bytes) and
  `summary.json` with `{count, est_tokens, est_cost_usd}` at $0.042 per million input tokens.
  Output tokens are free. A synthetic onboarding request is about 1k estimated tokens with the
  Noul alone and about 1.5k with the Choice added. 10,000 candidates are therefore about 10 to
  15 M tokens, or about $0.42 to $0.63 at the conservative estimate.
- **Live** needs both `live --live` **and** `TYPESAFE_API_KEY` in the environment. Otherwise it
  exits with code 2.
  - The key is read at call time only. It is never written, logged or put in a fixture.
  - Only `dev` and `calib` records are sent, and a split manifest is required. `eval` is sent
    only with `--include-eval`, and `excluded` never is.
  - Each attempt has a 10 s timeout.
    - 429 and 529 are retried with exponential backoff that honours `Retry-After`.
    - Other 5xx and transport failures (`httpx.TransportError`: timeouts, connection errors) are
      retried with the same backoff.
    - Every other 4xx fails fast.
    - Exhausted retries end with exit 3, after `answers.jsonl` and `summary.json` are written.
  - Responses are cached by `sha256(body)`. Each `answers.jsonl` row records the response
    `model`, `usage` and the request identity: `body_sha256`, `request_model`,
    `question_variant`, `question_ids`, `question_set_sha256` and `ablation`.
- Only versioned model ids such as `jev-1.13.0` are accepted. `jev-latest` (currently
  jev-1.13.0) and `jev-preview` (currently jev-1.14.0) move between releases, so they are refused
  unless `--allow-alias`.
- Jev is text-only and is not fine-tuned on customer data. Domain knowledge goes in the state and
  the criteria.

## 9. Evaluation (`evaluate_v1`)

- **Truth** is `evaluation_truth` only. Recall rows are added to the recall denominator as misses
  ("end-to-end recall").
- **Answers must belong to the current export.** Each row's request body is rebuilt from the
  current state with the row's model, variant, questions and ablation. The evaluator refuses:
  - a row whose `body_sha256` or `question_set_sha256` differs, which covers stale answers, a
    changed state, a different variant and a false ablation claim;
  - a row without identity (an older harness);
  - an answer for a record that is not in the states, and a duplicate record;
  - a file that mixes identities.

  The identity is recorded in `evaluation.json` and `operating_point.json`.
- **The operating point** is frozen on dev and calib and written as `operating_point.json`.
  - Its sha256 covers the whole document: the points, `split_manifest_sha256`,
    `answers_identity` and `cad_free_baseline`.
  - A given operating point is reused unchanged and checked for tampering.
  - It is refused when its split manifest, answer identity or `--cad-free-baseline` setting
    differs from the current run. It is also refused when it was frozen without Jev answers and
    answers are now given.
  - `prune_below` is the highest threshold whose dev+calib recall is at least 0.99.
  - `accept_at` is the lowest threshold at or above it whose precision is at least 0.99.
  - Scores in between fall in the abstention band and map to **review**.
- **Metrics** per split: AUC, Brier, log-loss, ECE with 10 bins, precision and recall at
  `prune_below`, pruning rate, the share of negatives pruned, the decision mix, the
  recall-preserving pruning rate with group-bootstrap 95 % CIs over SKU families, and the
  worst-family recall.
- **Baselines** use the same records and split:
  - the engine confidence, where a state carries `engine_confidence_score`. It is not
    necessarily calibrated.
  - scikit-learn `LogisticRegression` on the numeric state fields, fitted on `dev` only. The
    CAD-free variant is `--cad-free-baseline`.
- **Flags:**
  - `exploratory_not_unseen`: a consumed or retrospective source is involved;
  - `in_sample_only`: no eval record exists;
  - `synthetic_only`: every truth label is E2 synthetic, so the result is an engineering ledger,
    never accuracy evidence.

## 10. Commands

```bash
export PYTHONPATH=.:backend
# synthetic inputs and their exports (no real data)
.venv/bin/python -m research.jev.cli make-synthetic --out "$TMPDIR/jev-ws"
# exports from explicit inputs
.venv/bin/python -m research.jev.cli export-onboarding --input BUNDLES.jsonl --out "$TMPDIR/jev-onb"
.venv/bin/python -m research.jev.cli export-scan --jobs-root JOBS --labels-root LABELS \
    --f1-pool POOL.jsonl --gold GOLD.json --out "$TMPDIR/jev-scan"
# dry run (default; no network)
.venv/bin/python -m research.jev.cli dry-run --states "$TMPDIR/jev-ws/onboarding/states.jsonl" \
    --out "$TMPDIR/jev-dry" [--questions is_real_bend,candidate_kind] [--cad-free]
# live: NOT run in this session; needs owner approval + ZDR terms, --live and TYPESAFE_API_KEY
.venv/bin/python -m research.jev.cli live --live --states S --labels L --splits SPLITS --out DIR
# evaluation
.venv/bin/python -m research.jev.cli evaluate --labels L --splits SPLITS --states S \
    --answers DIR/answers.jsonl --out "$TMPDIR/jev-eval"
```

Tests (no network; `httpx.Client.send` is patched to raise):

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider \
  tests/test_prod_jev_state_request.py tests/test_prod_jev_labels_splits.py \
  tests/test_prod_jev_exporters.py tests/test_prod_jev_client_harness.py tests/test_prod_jev_evaluate.py -q
```

## Egress policy

States are coordinate-free and identity-free, but they still describe the owner's parts. Before
**any** live call:

1. The owner gives explicit written approval for the specific export: its manifest sha, record
   count and estimated cost.
2. Zero-data-retention or enterprise terms with TypeSafe are in place
   (<https://docs.typesafe.ai/legal>). Jev is not trained on customer requests, but retention
   still matters.
3. Only dev and calib records are sent. Eval records are sent once, only after the operating
   point is frozen, and only with `--include-eval`. Held-out, sealed and one-shot families are
   never exported.
4. The key stays in the environment (`TYPESAFE_API_KEY`). It is never put in a file, test, log
   or commit.

Jev answers stay research data. Promoting any hook in §1 into the line needs its own plan,
gates and review. Even then a hook could only route to REVIEW.
