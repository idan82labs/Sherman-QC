# Production engine I/O contract (ENGINE_IO)

Status: binding from wave 0 (plan `docs/production/PRODUCTION_LINE_PLAN.md` §10, amendments 1
and 2). Owner: C01. Consumers: E1 (compile), E2 (synthetic suite), E3 (inspect), E4 (evaluation),
C02 (runner), C05 (release gate), C06 (assets and PDF), C07 (line API), C08 (station UI),
C09 (wizard) and C11 (integration).

The pydantic models in `contracts/production/*.py` are normative. This document explains them
and fixes the command lines, directory layouts and file formats that the models cannot express.
If this text and a model disagree, the model wins, and the disagreement is a bug to fix in
this document.

| Model | Module | Schema tag |
|---|---|---|
| `PartRecipeV1`, `ReleaseRequestV1` | `part_recipe.py` | `sherman_part_recipe_v1` |
| `CompiledPartV1` | `compiled_part.py` | `sherman_compiled_part_v1` |
| `ShopfloorInspectionResultV1`, `BendGeometryFileV1`, `StationFeedV1` | `inspection_result.py` | `sherman_shopfloor_inspection_result_v1`, `sherman_bend_geometry_v1`, `sherman_station_feed_v1` |
| `ScanFileV1`, `LineJobV1`, `LineJobSummaryV1` | `line_job.py` | `sherman_scan_file_v1`, `sherman_line_job_v1` |
| `LineEventV1` | `events.py` | `sherman_line_event_v1` |
| `check_result_against_compiled` (pure cross-document check) | `consistency.py` | — |

Worked examples for the synthetic asymmetric hat part `SYN-HAT-1` are in `examples/`, produced by
`python -m contracts.production.examples.generate_examples`. The JSON schemas are exported with
`python -m contracts.production.schema_export --out DIR`.

---

## 1. Frame, units and conventions

- **One frame, `cad_model_mm`.** It is the STEP model coordinate system of the recipe CAD, in
  millimetres. Every geometry artifact uses it: CompiledPart loci, stage models, the registered
  scan, deviations, display meshes, bend polylines and label anchors. A raw scan reaches the frame
  through `result.inspection_validity.registration.raw_to_frame`, a 4×4 rigid matrix in
  row-major order, applied after `unit_factor`. The validator requires a *proper* rigid
  transform: last row `[0, 0, 0, 1]`, `RᵀR = I` within 2e-5 (the canonical 6-decimal rounding
  stays far below this; any real scale or shear does not) and `det R = +1`. A registration that
  only fits mirrored is never published as a transform; validity is `MIRRORED_PART_SUSPECTED`.
- **Units.** Lengths are in mm and angles in degrees. A raw scan is scaled by an explicit
  `unit_factor` of 1, 25.4 or 1000 only. There is never a continuous auto-scale. If a unit
  mismatch is suspected, validity is `UNIT_MISMATCH_SUSPECTED`.
- **Angle.** Every angle is the *included* angle (`INCLUDED_DEG`): 180 means flat, and acute
  angles are allowed. Native STEP candidates store the sweep, so `included = 180 − sweep`.
- **Flange length** (`OUTSIDE_MOLD_LINE`). For bend *b* and side *S*, it is the distance from
  *b*'s OML virtual sharp to the far boundary of *S*. It is measured in *S*'s OML plane,
  perpendicular to the bend direction, inside the bend's axial corridor. The far boundary is
  either `FREE_EDGE` (a robust edge fit) or `ADJACENT_BEND` (measured OML sharp to OML sharp).
  Expected and actual values use the same definition.
- **Side naming.** Side A is the flange whose in-plane outward unit vector (perpendicular to the
  bend axis, pointing away from the bend) has the larger dot product with +X. Ties
  (|Δ| < 1e-6) are broken by +Y, then +Z. The rule is implemented in
  `compiled_part.assign_sides`, and the validators enforce it.
- **Skins.** Skin 0 of the datum panel is the face whose normal has the larger dot product with
  +Z (ties are broken by +Y, then +X). Skin 0 continues across every bend, and skin 1 is the
  other face. `CompiledBendV1.oml_skin` says which skin is on the outside of that bend, so its
  virtual sharp is the OML.
- **Bend line.** The bend line is the **OML virtual sharp**: the intersection of the two fitted
  OML-skin planes. The cylinder axis (the centre of curvature) is stored for stage models and
  diagnostics only, and is **never** published as the bend line. The line-localization locus
  declared for G2 is `loci.outer_apex`, the visible-surface apex.
- **Bend-line position** (`BEND_LINE_POSITION`). `expected` is 0 (`NOMINAL_LOCUS_REFERENCE`).
  `actual` is the signed offset of the measured OML sharp from the nominal OML sharp of the
  decoded stage. It is measured perpendicular to the bend axis, in side A's OML plane, and is
  positive along side A's `outward_dir`. The station median is used.
- **Backgauge length** (`BACKGAUGE_LENGTH`, mid-sequence only). It is measured from the OML of
  the just-formed bend (sequence index *k* at stage S_k), across still-flat material, to the far
  boundary. The nominal comes from stage model S_k (`STAGE_MODEL_NOMINAL`). Its characteristic
  id is `<bend_uid>/BACKGAUGE_LENGTH/S<k>`, and it is attached to the side that leads toward the
  unformed bends.
- **General measurement** (`GENERAL_MEASUREMENT`). It is the straight-line distance between the
  operator-tagged anchors. The nominal is `nominal_mm` (the operator target, or else the CAD
  distance).
- **Characteristic ids** (`characteristics.make_characteristic_id`):
  `<bend_uid>/FLANGE_LENGTH/A|B`, `<bend_uid>/INCLUDED_ANGLE`, `<bend_uid>/BEND_LINE_POSITION`,
  `<bend_uid>/BACKGAUGE_LENGTH/S<k>` and `gm/<measurement_id>`. The CompiledPart nominals and the
  result rows share these ids.
- **Stable bend identity.** `bend_uid = "bu_" + sha256("<cad_sha256>|<source>|<sorted endpoints
  at 1e-3>|<radius at 1e-3 or none>")[:20]`, computed by `domains.production.recipe.bend_identity`
  from the operator-confirmed mark line. The recipe assigns the uids, and E1 binds its flange
  graph to them; E1 never invents its own uids.
- **Canonical JSON.** UTF-8, sorted keys, separators `(",", ":")`, `ensure_ascii=False` and no
  NaN/inf. Floats are rounded to 6 decimals and −0.0 becomes 0.0 (`canonical.py`). Every hash in
  these contracts is sha256 over those bytes.

### 1.1 Decisions (guard band)

Let `d = actual − expected`, let U be the expanded uncertainty (strictly greater than 0), and let
the tolerance band be `[−minus, +plus]`:

| Decision | Rule (asymmetric form) | Symmetric form |
|---|---|---|
| PASS | `d + U ≤ plus` and `d − U ≥ −minus` | `|d| + U ≤ tol` |
| FAIL | `d − U > plus` or `d + U < −minus` | `|d| − U > tol` |
| REVIEW | otherwise | the U interval straddles a limit |

`characteristics.guard_band_decision` implements the rule. The result validator recomputes the
decision for every MEASURED characteristic that has both a nominal and a tolerance, and rejects
the result if the stored decision differs.

**Rounding.** Result floats (`expected`, `actual`, `deviation`, `uncertainty`, station stats and
tolerances) are stored already rounded to 6 decimals, the canonical-JSON precision. Deviation is
`round6(actual − expected)`, and the decision is computed on the stored values. A canonical
round trip is therefore lossless, and a reloaded result still validates. Producers build rows only
through `CharacteristicResultV1.measured(...)`, `.not_measured(...)` and `.not_applicable(...)`,
which take the CompiledPart `CharacteristicNominalV1` and derive deviation, decision, units and
tolerance. They never compute a decision themselves.

UNKNOWN is not produced by the guard band. It is the decision of every NOT_MEASURED
characteristic, and of a MEASURED value that has no nominal or no tolerance. UNKNOWN never becomes
PASS or FAIL, and the nominal is never copied into `actual`. A MEASURED characteristic must carry
U > 0.

### 1.2 Vocabulary map (survey/legacy → production)

| Legacy term | Production term | Display |
|---|---|---|
| IN, IN_TOLERANCE | PASS | green `#16A34A` |
| OUT, OUT_OF_TOLERANCE | FAIL | red `#DC2626` |
| *(none)* | **REVIEW** | **amber `#F59E0B`** |
| NOT_MEASURED | UNKNOWN (with a reason) | neutral grey `#6B7280`, label "Undetermined" / "לא הוכרע" plus the reason (UNKNOWN also covers a MEASURED value without nominal or tolerance, so the label never says "not measured") |
| UNFORMED, OBSERVED_NOT_FORMED | NOT_YET_FORMED (bend state) | muted slate `#94A3B8`, greyed row |
| NOT_APPLICABLE | NOT_APPLICABLE | muted slate `#94A3B8` |
| FORMED, OBSERVED_FORMED | FORMED | neutral |
| *(summary)* | ALL_MEASURED_IN_TOLERANCE | green: the only green summary |
| *(summary)* | STAGE_IN_TOLERANCE | blue "info" `#2563EB`, never green: the part is not finished |

The code is in `contracts/production/vocabulary.py` (`LEGACY_TO_PRODUCTION`, `DISPLAY_TOKENS`,
with Hebrew and English labels). The station UI and the PDF mirror these tokens. UNKNOWN and
NOT_YET_FORMED are never shown green or red. The Hebrew labels still need review by a native
speaker.

### 1.3 Reason vocabularies

- **UNKNOWN reasons** (`UNKNOWN_REASONS`):
  - from plan §3.2: OCCLUDED, EDGE_NOT_SEEN, LOW_DENSITY, AMBIGUOUS_SEGMENTATION,
    REGISTRATION_UNTRUSTED, AMBIGUOUS_POSE, TIMEOUT, UNSUPPORTED_FEATURE;
  - contract additions: INVALID_INSPECTION, BEND_STATE_UNKNOWN, ADJACENT_BEND_NOT_MEASURED,
    NOMINAL_UNAVAILABLE, NO_TOLERANCE.
- **NOT_APPLICABLE reasons**, each allowed only for the listed kinds:

  | Reason | Allowed kinds |
  |---|---|
  | BEND_NOT_YET_FORMED | bend characteristics of a NOT_YET_FORMED bend |
  | NOT_INSPECTED_BY_RECIPE | flange sides and general measurements |
  | NOT_APPLICABLE_AT_STAGE | backgauge and general measurements; flange sides whose adjacent bend is NOT_YET_FORMED (amendment ENGINE_IO-A1, §7.1) |

---

## 2. CompiledPartV1 (E1 output, E3 input)

`compiled_part.py` holds the full field list. Its structure:

- **Identity.** `status` is COMPILED or BLOCKED. The file also holds the recipe reference
  (`recipe_id`, `version`, `recipe_sha256`, `content_sha256`), `cad_sha256`, `flat_sha256?`,
  `compiler {name, version, config_sha256?}`, `compiled_at` and `frame = cad_model_mm`.
  `compiled_part_sha256` is the canonical sha of everything except itself, `compiled_at` and
  `diagnostics`, so identical inputs give an identical hash. Use `CompiledPartV1.finalize(...)`
  to build one.
- **Material.** `thickness {t_mm, source MEASURED_BOTH_SKINS|RECIPE, spread_mm}` and
  `k_factor {value, source BOUND_FLAT|SHOP_DEFAULT, uncertainty}`. With no bound flat, K
  defaults to the shop K, 0.33, with an explicit uncertainty, and E1 adds the warning
  `K_FACTOR_DEFAULTED`.
- **Flange graph.**
  - `panels[]`: `panel_id` `P<n>`, `skin0_plane`, `skin1_plane?`, `thickness_mm?`, `area_mm2`,
    `centroid_mm`, `outer_boundary_mm` (the skin-0 polygon), `free_edges[]` with
    `registration_usable`, `holes_mm[]`, `is_datum`, `supported` and `unsupported_feature?`.
  - `bends[]`, recipe bends in sequence order: `bend_uid`, `sequence_index`, `label`,
    `machine_move_index?`, `graph_bend_id` `G<n>`, `parent_panel_id`, `child_panel_id`,
    `inner_radius_mm` (Ri), `thickness_mm`, `sweep_deg`, `nominal_included_deg = 180 − sweep`,
    `direction?`, `oml_skin`, `axis {point_mm, direction}`, `loci`, `bend_length_mm`,
    `neutral_length_mm = sweep_rad·(Ri + K·t)`, `sides (A, B)` and `supported`. The axis is
    oriented so that rotating the child by +sweep (right-hand rule) goes from flat to formed.
  - The graph is a tree rooted at `datum_panel_id`: the panel that never moves in the sequence,
    or the largest one if several qualify. The validator enforces one parent per panel, no
    cycles, and exactly one datum.
  - `unmarked_graph_bends[]`: flange-graph bends that no recipe mark matched (see
    reconciliation below).
- **Loci** per bend, at the final pose: `cylinder_axis`, `inner_tangent_parent|child`,
  `outer_tangent_parent|child`, `iml_virtual_sharp`, `oml_virtual_sharp`, `inner_apex` and
  `outer_apex`. Each one is a `LineLocusV1 {start_mm, end_mm}` spanning the bend length.
- **Nominals per characteristic** (`characteristics[]`). Each entry holds:
  - the characteristic id, `kind`, `side?`, `stage_index?`, `measurement_id?` and `unit`;
  - `nominal`, or `unavailable_reason` (`NOMINAL_UNAVAILABLE` / `UNSUPPORTED_FEATURE`);
  - `convention` (for example `OML_SHARP_TO_FREE_EDGE`, `OML_SHARP_TO_OML_SHARP`,
    `INCLUDED_DEG`, `OML_SHARP_OFFSET_ALONG_SIDE_A`, `OML_SHARP_TO_FAR_BOUNDARY_AT_STAGE` or
    `CAD_STRAIGHT_LINE`);
  - `boundary_kind?`, `adjacent_bend_uid?`;
  - `tolerance` (resolved: the per-bend override, or else the recipe default, with its `origin`
    and `standard`);
  - `recipe_requirement` (REQUIRED, NOT_INSPECTED or OPTIONAL).

  The band and the requirement are fixed by the recipe (`compiled_part.recipe_policy_for`), and
  the `CharacteristicNominalV1` validator enforces the per-kind vocabulary:

  | Kind | Tolerance (recipe) | `recipe_requirement` |
  |---|---|---|
  | FLANGE_LENGTH A/B | `flange_a_mm` / `flange_b_mm` override, else `flange_mm`; always carried | REQUIRED or NOT_INSPECTED (`flange_inspection`) |
  | INCLUDED_ANGLE | `angle_deg`; always carried | REQUIRED |
  | BEND_LINE_POSITION | `bend_line_position_mm` (may be null) | REQUIRED iff a tolerance resolves, else OPTIONAL |
  | BACKGAUGE_LENGTH | `backgauge_mm`, else `flange_mm` | REQUIRED iff a tolerance resolves, else OPTIONAL |
  | GENERAL_MEASUREMENT | measurement override, else `general_mm`, else `flange_mm` | REQUIRED or OPTIONAL (`required`) |

  A result row built from a nominal is required exactly when `recipe_requirement == REQUIRED`
  and a tolerance exists (`CharacteristicResultV1.required_for`, the factories' default).

  **The compiled nominal is authoritative.** The included angle E3 compares against is the
  flange-graph value in the CompiledPart (`nominal_included_deg = 180 − sweep`), not the recipe's
  `nominal_angle`. When the two disagree beyond 0.5°, E1 emits the WARNING
  `ANGLE_CONVENTION_DISAGREES_WITH_RECIPE` (naming the bend), and the wizard shows both values
  before release, so the operator can correct the mark or the convention.

  A COMPILED part has FLANGE_LENGTH A and B, INCLUDED_ANGLE and BEND_LINE_POSITION for every
  bend. It has BACKGAUGE_LENGTH for each mid-sequence stage k = 1..N−1, and every recipe
  general measurement.
- **Stage models S_k** (`stages[]`).
  - A COMPILED part has exactly one `SEQUENCE_PREFIX` stage for each k = 0..N, named `S<k>`,
    where the formed set is the first k bends. Optional `OUT_OF_ORDER` stages are named
    `S<k>-<tag>`. Stages are defined per bend; the order is only a soft prior.
  - Each stage carries:
    - `method`: `METHOD_A_ROTATE_BACK` (always available) or `METHOD_B_FOLD_FLAT` (when a flat
      is bound);
    - `k_factor_used`;
    - `panel_transforms {panel_id → 4×4}`, the transforms from the final pose to the stage
      pose. For each unformed bend on the path from the datum, the transform rotates back by
      the sweep about the axis, then translates by the neutral strip length away from the
      parent;
    - `bend_lines {bend_uid → line}`: the OML sharp for formed bends, and for unformed bends the
      centre line of the strip on the OML skin;
    - `sample_cloud`, an `.npz` file;
    - `display_mesh`, a `.ply` file;
    - `symmetry`.
  - **Sample cloud `.npz`**, loaded with `allow_pickle=False`, with a spacing of about 0.5 mm:

    | Array | Type | Content |
    |---|---|---|
    | `points` | float32 (M,3) | stage pose, `cad_model_mm` |
    | `normals` | float32 (M,3) | unit normals, pointing out of the material |
    | `segment_kind` | uint8 | 0 = panel, 1 = bend, 2 = edge wall |
    | `segment_index` | int32 | index into `panels` (kind 0 or 2) or `bends` (kind 1) |
    | `skin` | uint8 | 0, 1, or 2 = edge wall |

    E3 builds its own KD-tree on load. Nothing pickled is ever stored.
  - **Display mesh `.ply`**: binary little-endian, float32 `x y z` vertices,
    `list uchar int vertex_indices` triangles, in the stage pose.
- **Symmetry test** (one per stage).
  - The stage model is registered onto itself, both rotations and mirrors.
  - `is_symmetric` is true if and only if some non-identity candidate reaches
    `rms_mm ≤ threshold_rms_mm`, which is 0.3 mm.
  - `ambiguous_bend_groups` are the bend orbits under those candidates. Only these bends may ever
    be reported as AMBIGUOUS, and AMBIGUOUS is justified only by this test.
  - Each candidate carries its `transform` and `bend_permutation`.
  - A flat blank (S0) is usually symmetric even when no bend line maps onto another bend line.
- **Flat** (`flat`): `provided`, `format`, `sha256`, `parse_status`, `bend_lines_found`,
  `binding_state` (UNBOUND, AUTO_PROPOSED or OPERATOR_CONFIRMED, as released in the recipe) and
  `developed_length_check {status OK|MISMATCH|NOT_EVALUATED, threshold_mm, max_abs_delta_mm}`.
- **Size caps.** At most 999 panels (`MAX_PANELS`) and 999 bends (`MAX_BENDS`), so the
  `segment_labels.u16` encoding of §4.1 cannot collide.
- **Unsupported geometry** is declared per feature (`unsupported_features[]`: ROLLED, HEM,
  SPLINE_WALL, STEPPED_BOSS or OTHER, with the characteristic ids it affects), never per part.
  E3 reports the affected characteristics as UNKNOWN(UNSUPPORTED_FEATURE).

### 2.1 Reconciliation and compile blockers (shown in the wizard)

E1 matches the recipe's confirmed bend lines (`RecipeBendV1.line_*_mm`, mid-radius identity lines)
one-to-one to flange-graph bends. A pair is feasible when the directions are parallel within 2°
and the recipe line midpoint lies inside the bend region (within `Ri + t + 1 mm` of the cylinder
axis), and the match uses the Hungarian algorithm over feasible pairs. Issues are
`CompileIssueV1 {code, severity, bend_uid?, graph_bend_id?, detail}`. Severity is fixed per code:

| Code | Severity | Meaning and wizard action |
|---|---|---|
| `MARK_NOT_IN_FLANGE_GRAPH` | BLOCKER | A recipe bend (`bend_uid`) matches no graph bend. Remove the mark or correct its location. |
| `UNMARKED_BEND` | BLOCKER | A graph bend (`graph_bend_id`, listed in `unmarked_graph_bends`) has no mark. Add or confirm the mark. |
| `MARK_AMBIGUOUS` | BLOCKER | One mark matches several graph bends, or one graph bend matches several marks. |
| `DATUM_PANEL_UNDETERMINED`, `FLANGE_GRAPH_NOT_A_TREE` | BLOCKER | The flange graph is unusable. |
| `CAD_READ_FAILED`, `CAD_HASH_MISMATCH`, `RECIPE_INVALID` | BLOCKER | Input integrity. |
| `THICKNESS_INCONSISTENT` | BLOCKER | The skins disagree beyond tolerance. |
| `FLAT_PARSE_FAILED`, `FLAT_BINDING_INCONSISTENT`, `FLAT_DEVELOPED_LENGTH_MISMATCH` | BLOCKER | Flat QA. |
| `STAGE_MODEL_FAILED` | BLOCKER | A prefix stage could not be built. |
| `UNSUPPORTED_FEATURE`, `NOMINAL_UNAVAILABLE`, `K_FACTOR_DEFAULTED`, `ANGLE_CONVENTION_DISAGREES_WITH_RECIPE`, `FLAT_NOT_BOUND` | WARNING | Shown; the release is allowed. |

`status` is BLOCKED exactly when at least one BLOCKER is present. `CompiledPartV1.blocker_codes`
lists them. The C05 release gate refuses a BLOCKED compile and returns its codes. A BLOCKED
compile still writes `compiled_part.json`, because the wizard needs `unmarked_graph_bends` and the
issue list.

`check_compiled_against_recipe(compiled, recipe)` returns cross-document problems (sorted):
RECIPE_REF_MISMATCH, CAD_SHA_MISMATCH, FLAT_SHA_MISMATCH, BEND_INVENTORY_MISMATCH,
GENERAL_MEASUREMENT_MISMATCH, NOMINAL_UNKNOWN_REFERENCE, TOLERANCE_MISMATCH (a nominal's band
differs from the recipe's resolved band) and REQUIREMENT_MISMATCH (its `recipe_requirement`
differs from the recipe). E3 and the C05 release gate refuse to proceed on any of them, so a
compile-side tolerance bug cannot reach the line.

---

## 3. E1 command line: onboarding compile

```
python -m domains.production.engine.compile \
    --recipe RECIPE.json          # canonical PartRecipeV1 (finalized; hashes verified)
    --cad PART.step               # bytes must hash to recipe.cad.sha256
    [--flat FLAT.dxf|FLAT.step]   # bytes must hash to recipe.flat.sha256 when the recipe has a flat
    --out DIR                     # created; must be empty or absent
    [--shop-k 0.33]               # K when no flat is bound
    [--config CONFIG.json]        # pinned compile config (hash recorded in compiler.config_sha256)
```

- **Environment.** Launched with `cwd = <engine_root>`, `PYTHONPATH=<engine_root>:<engine_root>/backend`
  and the thread caps (`OMP_NUM_THREADS=1` and friends). The budget is 120 s wall time and
  32 GiB RSS (4 GiB before ENGINE_IO-A5). The compile is CPU-only and deterministic: the same inputs give the same
  `compiled_part_sha256`.
- **Stdout**: one tab-separated record per line: `PROGRESS\t{"stage": str, "progress": 0-100}`,
  `RESULT\t{"status": "COMPILED"|"BLOCKED", "compiled_part_sha256": str, "blocker_codes": [..],
  "out": str}`, or `ERROR\t{"code": str, "message": str}`. Stderr is free-form log text.
- **Exit codes:**

  | Code | Meaning | Files written |
  |---|---|---|
  | 0 | COMPILED | |
  | 2 | BLOCKED, or invalid input | BLOCKED: `compiled_part.json` with blockers. Invalid input (unreadable or mismatched-hash inputs): `compile_error.json {code, message}` instead. |
  | 3 | internal error | nothing is guaranteed |
- **Output directory** (`layout.COMPILE_OUTPUT_FILES`):

  ```
  DIR/compiled_part.json       CompiledPartV1, canonical JSON, written last (tmp + rename)
  DIR/stages/S<k>.npz          sample cloud per stage model (layout.stage_cloud_rel_path)
  DIR/stages/S<k>.ply          display mesh per stage model (layout.stage_mesh_rel_path)
  DIR/compile.log              human-readable, never parsed
  ```

  Every `rel_path` inside `compiled_part.json` is relative to `DIR`, and every referenced file's
  sha256 is recorded. C05 stores `DIR` at `recipes/compiled/<recipe_sha256>/` under the data root
  (`CompiledPartRefV1`).

---

## 4. E3 command line: per-scan inspection

```
python -m domains.production.engine.inspect \
    --compiled DIR/compiled_part.json   # its directory holds stages/
    --recipe RECIPE.json                # canonical PartRecipeV1
    --scan WORKSPACE/input/scan.ply     # PLY (mesh or points) or STL, staged under an opaque name
    --out WORKSPACE/engine              # created; must be empty or absent
    --scan-sha256 HEX                   # sha256 the job pinned for the scan bytes
    [--compiled-sha256 HEX]             # the job's compiled_part_sha256 (the runner always passes it)
    [--recipe-sha256 HEX]               # the job's recipe_sha256 (the runner always passes it)
    [--declared-stage K --declared-stage-source SIDECAR|OPERATOR|API]
    [--unit-factor 1|25.4|1000]         # default 1; no auto-scale
    [--time-budget-s 170]               # internal soft budget (runner hard stop is 180 s)
    [--engine-bundle-id ID] [--config CONFIG.json]
```

**Blindness** (plan §10 amendment 4):

- E3 never receives filename tokens, serials, metadata counts or `expected_completed`, by
  construction: no argument carries them. The runner stages the scan as `input/scan.<ext>`, and
  E3 has no `--context` option (an earlier draft had one; it is withdrawn because it handed the
  filename and serial to the engine).
- E3 writes `result.context` as an empty `ResultContextV1` (every field null). The job identity
  in `input/context.json` is added **after** the engine, when C06 (or the runner) writes
  `result/shopfloor_result.json` (section 5). `context` is never read for measurement, and none
  of the result validators look at it.
- Renamed copies of a scan must give identical `engine/result.json`, apart from `created_at`
  and the engine timing fields (`engine.duration_s`, `engine.peak_rss_bytes`).

**Integrity.** Before measuring, E3 checks:

- the compiled file's recomputed `compiled_part_sha256`, and that it equals `--compiled-sha256`
  when given (COMPILED_PART_HASH_MISMATCH);
- the recipe's recomputed hashes, and that `recipe_sha256` equals `--recipe-sha256` when given
  (RECIPE_SHA_MISMATCH);
- `check_compiled_against_recipe` returns no problems (COMPILED_PART_HASH_MISMATCH);
- `status == COMPILED` (COMPILE_BLOCKED);
- the sha256 of the scan bytes equals `--scan-sha256` (SCAN_HASH_MISMATCH).

Any failure here is exit 2. The runner checks the same hashes itself before launching E3; the
E3 checks make the engine safe to run on its own (E4, V1).

**Limits.** P95 ≤ 45 s, a 180 s hard stop enforced by the runner (process group kill →
`WORKER_TIMEOUT`), RSS ≤ 32 GiB (ENGINE_IO-A5; 4 GiB before), CPU only. When `--time-budget-s` expires, E3 stops measuring
and still writes a valid result. Characteristics it has not measured become
UNKNOWN(TIMEOUT). If registration did not finish, validity is `TIMEOUT`.

**Stdout**: the same tab-separated protocol as E1. The RESULT payload is
`{"summary_state", "inspection_validity", "stage_status", "result": "result.json"}`.

**Exit codes:**

| Code | Meaning | Files written |
|---|---|---|
| 0 | `result.json` written and valid | Every inspection outcome, including INVALID_INSPECTION, is a result, not an error. |
| 2 | invalid input | `engine_error.json {code, message}`. Job input codes (members of `line_job.FAILURE_CODES`): COMPILED_PART_MISSING, COMPILED_PART_HASH_MISMATCH, COMPILE_BLOCKED, RECIPE_SHA_MISMATCH, SCAN_MISSING, SCAN_HASH_MISMATCH. Invocation codes (a runner or developer bug, not a job input problem): OUT_DIR_NOT_EMPTY, CONFIG_INVALID, DECLARED_STAGE_INVALID. |
| 3 | internal error | |

The runner maps exit 2 to that code when it is in `FAILURE_CODES` and to WORKER_ERROR otherwise
(the invocation codes), exit 3 to WORKER_ERROR, a crash to WORKER_CRASH, and exit 0 without a
valid `result.json` to WORKER_NO_RESULT or ENGINE_OUTPUT_INVALID.

### 4.1 E3 output directory (`layout.ENGINE_OUTPUT_FILES`)

All geometry is in `cad_model_mm`. Files are written as tmp + rename, with `result.json` last.

| File | Format | Content |
|---|---|---|
| `result.json` | canonical JSON | `ShopfloorInspectionResultV1`. Must validate, or the job fails with ENGINE_OUTPUT_INVALID. |
| `registered_scan.ply` | PLY `binary_little_endian 1.0`, `element vertex N`, `property float x/y/z`, no faces | Every ingested scan point after `unit_factor` and `raw_to_frame`. Absent when registration abstained. |
| `deviations.f32` | N × float32 LE | Signed point-to-surface deviation against the decoded stage model `S_k̂` (`viewer_assets.deviation_reference_stage`, a stage model id that must equal `stage_decision.stage_model_id`), positive along the stage model's outward normal, NaN where none. Index-aligned with `registered_scan.ply` vertices. Absent when the heatmap is unavailable. |
| `segment_labels.u16` | N × uint16 LE (optional) | 0 = unassigned or outlier; `1 + i` = `compiled.panels[i]`; `1001 + j` = `compiled.bends[j]`; 65535 = unplanned crease. Index-aligned. Collision-free because a CompiledPart has at most 999 panels and 999 bends. |
| `display/stage_mesh.ply` | as in §2 | Display mesh of `S_k̂`. |
| `display/final_mesh.ply` | as in §2 | Display mesh of `S_N`, the finished CAD. |
| `bend_geometry.json` | `BendGeometryFileV1` | Per-bend nominal and measured polylines, label and side anchors, and unplanned creases (`BendGeometryFileV1.from_result`). |
| `engine.log` | text | Never parsed. |

`result.viewer_assets` names these files, relative to the result's directory, and declares
`frame`, `point_count`, `deviation_encoding = float32_le_nan_missing`,
`deviation_sign = SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL` and `heatmap_band_mm`. The band is the
recipe's `surface_profile_mm`, and the legend is signed (±band). When there is no heatmap,
`heatmap_unavailable_reason` is required.

### 4.2 Result semantics (enforced by validators)

- **Rows.** There is one `BendRowV1` per recipe bend, sorted by `sequence_index` 1..N. Its
  characteristics come in a fixed order: FLANGE_LENGTH A, FLANGE_LENGTH B, INCLUDED_ANGLE,
  BEND_LINE_POSITION, then optionally BACKGAUGE_LENGTH. `general_measurements` follow the bends.
- **States** (`state` ∈ FORMED, NOT_YET_FORMED, PARTIALLY_FORMED, UNEXPECTED_FORMED, UNKNOWN):
  - **NOT_YET_FORMED**: every characteristic is NOT_APPLICABLE. Angle and position carry
    BEND_NOT_YET_FORMED. Such a bend is never FAIL and never a defect.
  - **UNKNOWN**: needs UNKNOWN reasons and has no MEASURED characteristic.
  - **PARTIALLY_FORMED**: the angle can never be PASS.
  - **UNEXPECTED_FORMED**: a formed bend that the order prior did not expect. It forces the
    stage flag OUT_OF_ORDER_SUSPECTED.
  - Evidence: `state_evidence` holds the measured and nominal included angle, `formed_fraction`,
    support points and coverage. The tagged order and מהלך groups are only a soft prior, carried
    in `prior_formed_probability`.
- **Requiredness is fixed by the contract.** An applicable FLANGE_LENGTH or INCLUDED_ANGLE is
  required and toleranced. BEND_LINE_POSITION and BACKGAUGE_LENGTH are required exactly when they
  have a tolerance. A NOT_APPLICABLE characteristic is never required. A flange side that the
  recipe marks NOT_INSPECTED is NOT_APPLICABLE(NOT_INSPECTED_BY_RECIPE). A required
  GENERAL_MEASUREMENT always has a tolerance.
- **NOT_APPLICABLE_AT_STAGE** (backgauge and general measurements; flange sides under amendment
  ENGINE_IO-A1) needs a stage that leaves something unformed: the validator rejects it unless some
  bend is NOT_YET_FORMED. On a FLANGE_LENGTH it needs `adjacent_bend_uid` to name a bend whose row
  is NOT_YET_FORMED; a flange next to an UNKNOWN bend stays UNKNOWN(ADJACENT_BEND_NOT_MEASURED). The
  cross-check (4.3) narrows it further, so a finished part can never skip a required general
  measurement or backgauge and still go green.
- **Measurement states:**
  - **MEASURED**: needs `actual` and `uncertainty > 0`, and `deviation = actual − expected`. The
    decision follows the guard band.
  - **NOT_MEASURED**: no `actual`, `deviation`, `uncertainty` or `stations`. The decision is
    UNKNOWN, with a reason.
- **Ambiguous pose** (`registration.pose_ambiguous`): measures nothing, unless `registration.identity_ambiguous`
  is set (amendment ENGINE_IO-A2, §7.1): then only the readings identical under every tied symmetric twin are
  published, never a bend-line position, heatmap or stage model.
- **Display.** `display.measured_polyline_mm` exists exactly when BEND_LINE_POSITION is MEASURED,
  that is, when the line was fitted from scan planes. Only these lines may be scored or drawn
  solid; NOT_YET_FORMED bends emit no measured line. `nominal_polyline_mm` is the bend line of
  the decoded stage model, drawn grey and dashed.
- **Registration and validity:**
  - Validity other than `VALID` requires reasons. In that case nothing is MEASURED, every bend is
    UNKNOWN and no heatmap is published.
  - `VALID` requires a non-abstained registration with `raw_to_frame`.
  - `pose_ambiguous` is set when the top pose hypotheses lie within the margin **and** would
    change some bend identity, state or value. Then nothing is MEASURED, and each bend is either
    UNKNOWN with AMBIGUOUS_POSE or NOT_YET_FORMED (allowed only when every hypothesis agrees on
    it). A symmetric part never gets a confident wrong id.
  - Under `pose_ambiguous` no heatmap is published either (`viewer_assets.deviations` is null,
    `heatmap_unavailable_reason` e.g. AMBIGUOUS_POSE): the deviations would be coloured under
    one arbitrary pose hypothesis.
  - `in_plane_degenerate`: only BEND_LINE_POSITION and the heatmap are withheld. On every bend
    whose state is known and not NOT_YET_FORMED, BEND_LINE_POSITION must be
    NOT_MEASURED(REGISTRATION_UNTRUSTED). Angle and flange length do not depend on the pose.
- **Stage decision** (`derive_stage_decision`, a pure function). `inferred_stage` is the number
  of formed-like bends, or null if any bend is UNKNOWN.
  - Order rule: all bends of one מהלך share a rank, so order inside a group is free. The formed
    set must not leave a lower-rank bend unformed.
  - Flags, in precedence order:

    | Flag | Decision |
    |---|---|
    | NOT_EVALUATED (invalid inspection) | UNKNOWN |
    | AMBIGUOUS_POSE | UNKNOWN |
    | PARTIAL_BEND_FOUND | REVIEW |
    | OUT_OF_ORDER_SUSPECTED | REVIEW |
    | STAGE_MISMATCH (declared ≠ inferred) | REVIEW |
    | FINAL_INCOMPLETE (recipe `inspection_stage` FINAL and some bend not yet formed) | REVIEW |
    | UNDETERMINED (some bend UNKNOWN) | UNKNOWN |
    | CONSISTENT | PASS |

  UNDETERMINED ranks below the REVIEW flags, as REVIEW ranks above UNKNOWN for
  characteristics: a partial bend, an order violation or a missing FINAL bend is positive
  evidence that needs the operator, and an unknown bend elsewhere must not hide it (the summary
  is then REVIEW_REQUIRED, not INCOMPLETE). `flags` always lists every applicable flag.

  - A declared stage is never trusted or overridden silently.
  - `stage_model_id` (`S<k>` or `S<k>-<tag>`) names the CompiledPart stage model the scan was
    decoded against. It is null for an invalid inspection. When `inferred_stage` is known, k
    equals it; without OUT_OF_ORDER_SUSPECTED it is the prefix model `S<inferred_stage>`. A
    published heatmap references the same id.
- **Summary** (`derive_summary`, a pure function, mirrored by the station UI). It is evaluated in
  this order:

  | Condition | Summary | Reason added |
  |---|---|---|
  | validity ≠ VALID | INVALID_INSPECTION | `VALIDITY:<state>` |
  | any required FAIL | OUT_OF_TOLERANCE_FOUND | REQUIRED_FAIL |
  | any required REVIEW, a REVIEW stage decision, or any unplanned crease | REVIEW_REQUIRED | REQUIRED_REVIEW, `STAGE:<status>`, UNPLANNED_CREASES |
  | any required UNKNOWN, or an UNKNOWN stage decision | INCOMPLETE | REQUIRED_UNKNOWN, `STAGE:<status>` |
  | no required characteristic at all (nothing measured, e.g. every bend NOT_YET_FORMED) | INCOMPLETE | NO_REQUIRED_MEASURED |
  | some planned bend is not FORMED / UNEXPECTED_FORMED (unfinished part, IN_PROCESS) | STAGE_IN_TOLERANCE | PART_NOT_COMPLETE |
  | otherwise | ALL_MEASURED_IN_TOLERANCE | |

  `worst` is the required MEASURED characteristic with the largest `(|d|+U)/tol_side`.

  Optional characteristics (recipe `OPTIONAL`, so `required = false`) never change the summary
  state, but a valid result that has an optional FAIL or REVIEW says so: the reasons gain
  `OPTIONAL_FAIL` and/or `OPTIONAL_REVIEW`. The operator never sees a red or amber row under a
  green banner without a reason. The station UI and the PDF show these reasons next to the
  banner.

  ALL_MEASURED_IN_TOLERANCE therefore needs every planned bend formed and every required
  characteristic MEASURED and PASS, under either `inspection_stage`:
  - A missing bend in a FINAL scan is FINAL_INCOMPLETE, a REVIEW stage decision, so the summary
    is REVIEW_REQUIRED.
  - The same scan under IN_PROCESS is STAGE_IN_TOLERANCE at best. It is never green.
  - An extra bend is an unplanned crease (REVIEW_REQUIRED) or an UNEXPECTED_FORMED bend
    (OUT_OF_ORDER_SUSPECTED, REVIEW_REQUIRED).
  - Recipes default to `inspection_stage = FINAL` (`ReleaseRequestV1`). IN_PROCESS is an explicit
    operator opt-in.
- **Authority** is fixed: `OPERATOR_ASSIST` / `NOT_AUTHORIZED`. `metrology_qualified` is
  always false.

### 4.3 Cross-document check (result vs CompiledPart vs recipe)

The result validator sees only the result. `contracts.production.consistency.check_result_against_compiled(result,
compiled, recipe)` checks the rest and returns sorted codes (`RESULT_CONSISTENCY_CODES`):

- references: RECIPE_REF_MISMATCH, SKU_MISMATCH, COMPILED_PART_REF_MISMATCH, COMPILE_BLOCKED,
  INSPECTION_STAGE_MISMATCH, and every `check_compiled_against_recipe` code;
- inventory: BEND_INVENTORY_MISMATCH (uids, sequence, labels and moves must equal the compiled
  bends), GENERAL_MEASUREMENT_MISMATCH (ids and order must equal the recipe's);
- per characteristic: CHARACTERISTIC_UNKNOWN, CHARACTERISTIC_IDENTITY_MISMATCH,
  EXPECTED_MISMATCH (`expected` must be `round6(nominal)`), TOLERANCE_MISMATCH (the nominal's
  tolerance, unchanged) and REQUIREMENT_MISMATCH (NOT_INSPECTED_BY_RECIPE only where the recipe
  does not require the value; a NOT_INSPECTED nominal is always NOT_APPLICABLE; `required`
  equals `required_for(nominal)`);
- stage: BACKGAUGE_STAGE_MISMATCH, BACKGAUGE_MISSING (an order-consistent mid-sequence stage S_k
  must report bend k's compiled backgauge) and STAGE_MODEL_MISMATCH (the decoded stage model
  must exist and agree with every known bend state);
- STAGE_APPLICABILITY_MISMATCH: a NOT_APPLICABLE_AT_STAGE row the stage does not justify.
  - A `bend_pair` general measurement may be NOT_APPLICABLE_AT_STAGE only when one of its two
    bends is NOT_YET_FORMED. A `points` general measurement needs some NOT_YET_FORMED bend.
  - A backgauge row may be NOT_APPLICABLE_AT_STAGE only when the S_k nominal does not describe
    the part. It is rejected when the formed set is exactly the first k bends by sequence and
    `stage_model_id` is `S<k>`, because then the nominal applies.

Who calls it:
- E3 runs it on its own result before writing `result.json`. A non-empty list is an internal
  error (exit 3).
- The runner (C02) and C06 run it when they ingest `engine/result.json`. Any problem fails the
  job with ENGINE_OUTPUT_INVALID.
- C07 may re-run it before serving a result.

---

## 5. Runner integration (C02 → E3 → C06)

Job workspace `<data_root>/jobs/<job_id>/` (`layout.workspace_paths`):

```
job.json            LineJobV1
recipe.json         canonical recipe bytes (sha checked)
compiled/           copy of recipes/compiled/<recipe_sha256>/ (compiled_part_sha256 checked)
input/scan.<ext>    archived scan, opaque name
input/context.json  ResultContextV1 {job_id, station_id, scan_file_name, serial, scan_timestamp, received_at};
                    read by C06/the runner only, never passed to E3
engine/             E3 --out (section 4.1)
result/             C06: shopfloor_result.json, viewer_scan.ply, viewer_deviations.f32,
                    stage_mesh.ply, final_mesh.ply, bend_geometry.json, overview.png,
                    report.pdf, postprocess_error.json, postprocess.log
```

1. The runner (C02) leases the job, materialises the workspace, verifies the hashes, and runs E3
   with a hard timeout of 180 s and an RSS limit of 32 GiB (ENGINE_IO-A5; `line_job.ENGINE_HARD_TIMEOUT_S`,
   `ENGINE_RSS_LIMIT_BYTES`).
2. C06 validates `engine/result.json` (model validation plus `check_result_against_compiled`) and writes the decimated viewer copies into `result/`. The
   decimation keeps `viewer_scan.ply` and `viewer_deviations.f32` index-aligned by using one
   shared index selection. C06 rewrites `viewer_assets` and `artifacts` to point at `result/`,
   sets `context` from `input/context.json` (the engine's copy is empty), writes
   `result/shopfloor_result.json`, and renders the PDF from the result only. Nothing else in the
   result changes, so the summary and every decision are the engine's.
3. Served artifacts are only those in `layout.ARTIFACT_ALLOWLIST`, at the URL
   `/api/line/jobs/{job_id}/artifacts/{name}`.

Execution status never encodes conformance: an engine failure is a job failure code, never a
part FAIL.

## 6. Scan filenames (C01 `sku.py`, C05 resolver)

`domains.production.recipe.sku.parse_scan_filename(name, known_skus, pattern=None)` parses names
such as `BEFORE BENDING  11979003_C.ply`, `23553000_B_FULL_SCAN 1.ply`,
`38172000_01_SCAN_2 1.ply` and `(1)` copies.

- The SKU token may appear anywhere in the name. The longest SKU on token boundaries wins, and
  two distinct SKUs give AMBIGUOUS. The parser never guesses.
- `known_skus` may be a mapping from SKU to the recipe's expected revision. A revision reading
  that differs from it gives **REVISION_MISMATCH**, and the resolver quarantines the scan with
  reason `REVISION_MISMATCH`. Readings:
  - explicit markers anywhere after the SKU: `REV D`, `-REV-D`, `REVISION 01`, glued `REVD` or
    `REV01`; with a known expected revision, also a bare `R` right after the SKU (`_R_D`);
  - otherwise the revision-shaped token right after the SKU (`[A-Z]{1,2}` or two digits);
  - a single digit counts when the expected revision is numeric and the digit is not a
    whitespace-separated export counter (`38172000_1_SCAN_2 1` against `01`).
  Numeric revisions compare by value (`1` = `01`). With a known expected revision, two different
  readings in one name also give REVISION_MISMATCH (at least one of them differs). Without an
  expected revision nothing can mismatch: the status is OK with `revision_status` UNVERIFIED,
  and `revision` is the reading when there is one, or **null** when two readings conflict (the
  parser never picks one).
- Words that label a serial or an order (`SN`, `ID`, `NO`, `NR`, `PN`, `PO`, `WO`, `OK`, `QC`)
  are never a bare revision: `11979003_SN_0042` has no revision and serial `0042`. They are
  still accepted after an explicit marker (`REV NO` is unusual but explicit) and when they equal
  the expected revision.
- Copy markers are stripped: `(N)`, `copy`, `copy N` and Windows ` - Copy` / ` - Copy (N)`.
- Parsed descriptors (`scan_label`, such as FULL_SCAN, SCAN_2 or BEFORE_BENDING) are for display
  only and never reach the engine.

## 7. Versioning

Every document carries a literal schema tag. Any change that alters existing meaning needs a new
tag (`…_v2`). Additive optional fields are allowed only when an older reader still validates
older documents. C11's `tests/test_prod_contract_parity.py` imports the frozensets in these
modules (`EXECUTION_STATUSES`, `FAILURE_CODES`, `JOB_KINDS`, `STAGES`, `EVENT_KINDS`,
`QUARANTINE_REASONS`, `RESOLVE_STATUSES`, `PARSE_STATUSES`, `UNKNOWN_REASONS`,
`COMPILE_BLOCKER_CODES`, `ARTIFACT_ALLOWLIST`, …) and compares every duplicated copy against
them.

### 7.1 Amendments within `_v1`

An amendment here only relaxes or narrows a rule in a way every document valid before it stays
valid; readers of this repository share the one model module, so the schema tag is kept. Each
amendment is recorded as a NOTE / AMENDMENT in `docs/production/runlog/freeze/LEDGER.json`.

| Id | Date | Owner decision | Change | Engine |
|---|---|---|---|---|
| ENGINE_IO-A1 | 2026-10-19 | FIXES2.md §4 item 4 (option: allow NOT_APPLICABLE_AT_STAGE for FLANGE_LENGTH) | A FLANGE_LENGTH side bounded by an adjacent bend that is NOT_YET_FORMED is NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE): its length is undefined at that stage. Validator: such a row needs `adjacent_bend_uid` naming a NOT_YET_FORMED bend row. A side whose adjacent bend is FORMED is measured as before; one whose adjacent bend is UNKNOWN stays UNKNOWN(ADJACENT_BEND_NOT_MEASURED). | e3-1.11.0 |
| ENGINE_IO-A2 | 2026-10-19 | FIXES2.md §4 item 3 (D3) | `RegistrationV1.identity_ambiguous` (default false; needs `pose_ambiguous`). When certified symmetric twin poses tie, the result may publish the readings identical under every tied twin: the matched characteristic (the twin's label of the same physical bend, through E1's certified bend permutation) has the same nominal and tolerance, every twin reads the characteristic, the values agree within the published U, and the guard-band decision is the same under each twin. Everything else is UNKNOWN(AMBIGUOUS_POSE); a bend whose state differs between the twins is UNKNOWN(AMBIGUOUS_POSE). Bend-line position, backgauge, heatmap and stage model stay withheld (they depend on which bend is which). The AMBIGUOUS_POSE stage flag keeps the stage decision UNKNOWN, so the summary is never green. Without the mark an ambiguous pose still measures nothing. | e3-1.11.0 |
| ENGINE_IO-A5 (**IN FORCE since 2026-10-26**; proposed 2026-10-25) | 2026-10-25 / 2026-10-26 | Owner station spec (2026-10-25): one local PC (NVIDIA RTX 5000 Pro, 256 GB RAM) running ONE job at a time; production scanner a ZEISS high-end scanner on a cobot (ScanBox), about 5–50 M points/faces per scan; "raise the per-step memory cap from 4 GiB to 32 GiB, for both the E3 engine and the zero-shot overlay step" | Execution limit only, no document changes: `line_job.ENGINE_RSS_LIMIT_BYTES` and its C11 mirror `store.vocabulary` 4 GiB → **32 GiB**; the supervisor ceiling `MAX_RSS_LIMIT_BYTES` (16 GiB) and `SHERMAN_LINE_RSS_LIMIT_MB` (default 4096, max 16384) follow at 32 GiB. The runner invariant stays: zs limits never exceed the engine's (zs 32 GiB / 90 s, E3 32 GiB / 180 s; the two run in parallel, ≤ 64 GiB together on the 256 GB station). Time limits unchanged (E3 180 s, `--time-budget-s` 170 s; zs 90 s; merge 60 s). Effect: a job that previously died with `WORKER_RSS_LIMIT` (or a zs step killed for RSS, leaving no overlay) can run to completion; its output goes through the unchanged validators, guard band and summary rules, so A5 **cannot create a PASS by itself** (an RSS kill was never a part FAIL; a completed run is decided like any other). The frozen E4 G1 harness keeps its own standard run condition and gate (180 s / 4 GiB / 170 s; RSS ≤ 4 GiB at ~1M points, thresholds E4-G1-2); changing those is a separate thresholds amendment. **Blocker:** `contracts/production/line_job.py` is one of the pinned source files of every zs lane pin (zs_lane_v1 live `af87df2f…`, zs_lane_v3 rev 2 `ab31d061…`); changing its bytes makes `pin.verify` fail, so the zs step refuses and the overlay goes dark until both lanes are re-pinned, and the v3 re-pin changes the fingerprint bound by the owner-confirmed PREREG_FRESH_TEMPLATE. A5 takes effect only with an owner decision on that re-pin (see `docs/production/runlog/zeroshot_2026-10-25/station/DENSE_READINESS.md` §A). Ids A3 and A4 are reserved by the roll proposal (`docs/production/runlog/cad_roll_solver/DESIGN.md` §4.1). **In force 2026-10-26:** owner approval of the bundled re-pin ("1a, approve the bundle, station is Windows"; freeze-ledger seq 25); both zs lane pins re-pinned in the same change set (lane code and models unchanged), which resolves the blocker. Applied as written: `ENGINE_RSS_LIMIT_BYTES` and the `store.vocabulary` mirror 32 GiB, `MAX_RSS_LIMIT_BYTES` 32 GiB, `SHERMAN_LINE_RSS_LIMIT_MB` default and max 32768. The eval CLI (`--rss-limit-mb` 4096) and the G1 thresholds keep the frozen 4 GiB run condition. Same change set: e3-1.13.0's density guard -- a scan above the validated envelope (> 15 points per 0.63 mm cell or > 3 M points) is not measured (validity UNSUPPORTED_INPUT, reason `DENSITY_ABOVE_VALIDATED_RANGE`), so A5 does not let an over-dense scan reach a decision. **e3-1.14.0 (2026-10-28, owner "extend the waiver, thin dense scans, go ahead"):** a cloud above 15 points per 0.63 mm cell is thinned to one actual point per cell (the point nearest the cell mean) instead of refused; the 3 M-point refusal still applies after thinning (`repin_2026-10-28/REPIN.md`). Record: `docs/production/runlog/zeroshot_2026-10-25/repin_2026-10-26/REPIN.md`. | e3-1.13.0 (A5 itself changes no engine output) |
