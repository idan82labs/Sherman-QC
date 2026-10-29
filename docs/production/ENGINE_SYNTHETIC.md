# E2: synthetic sheet-metal suite and virtual ZEISS scanner

Owner: E2. Code: `domains/production/engine/synthetic/`. Tests: `tests/test_prod_synth_*.py`.
Plan: `docs/production/PRODUCTION_LINE_PLAN.md` §4 (E2), §5 (G1), §10 amendments 3, 4b, 4c, 4d, 5.
Contracts it follows: `contracts/production/ENGINE_IO.md` §1 (frame, conventions), C01
`PartRecipeV1` and the C01 recipe builder.

The suite builds parametric multi-bend parts as **real OCP STEP BReps**, together with the matching
flat DXF, a synthetic drawing PDF, a recipe draft and a finalized `PartRecipeV1`, and a virtual
robotic structured-light scan (PLY). For each part it also writes **exact analytic truth** for every
bend and characteristic, at any stage of the tagged bending order.

**Independence.** This package imports nothing from E1 (`engine.compile`) or E3 (`engine.inspect`).
Truth comes from the part parameters, never from measuring generated meshes or scans
(amendment 5). The only cross-package imports are the C01 contracts and the C01 recipe builder.

**Blindness** (amendment 4). Nothing E3 receives names the case or its scenario: SKUs, part ids
and part names are opaque hashes, the recipe carries no case id, scan files carry no label in their
point order, and the recipe's inspection stage is drawn by policy rather than copied from the
scenario (§5). The case id to SKU mapping lives only in `case.json` and `manifest.json`.

Revision: generator `e2-synthetic-1.1.1` (2026-10-17).

- 1.1.1 changes one definition: `features_present.stage_1` (and so the manifest coverage that
  E4's binding G1.SUITE gate reads) is now a property of the scanned part. It means exactly the
  first bend of the tagged order is formed, fully. It no longer also requires the recipe's
  inspection stage to be IN_PROCESS. That stage is a 25 % policy draw (§5), so under 1.1.0 a
  family could lose its stage-1 coverage on seeds nobody had generated, such as the final QUAL run.
- Only `case.json` and `manifest.json` differ from 1.1.0. Every other case file is byte-identical
  (checked on all 50 `regression` cases, §9).
- A 1.1.0 directory is refused on resume (`CaseConflictError`). Regenerate it, or use `--force`.
- 1.1.0 was the fix stage after the independent review. Every one of its case files differs from
  1.0.0.

## 1. Quick start

```
PY="env PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python"
$PY -m domains.production.engine.synthetic generate --tier regression --seedset DEV --out /tmp/e2_reg
$PY -m domains.production.engine.synthetic plan --tier full --seedset DEV            # list, no geometry
$PY -m domains.production.engine.synthetic part-set --family HAT --seedset DEV --out /tmp/e2_ps   # one part, many scans
$PY -m domains.production.engine.synthetic export-inbox --cases /tmp/e2_reg --out /tmp/inbox
$PY -m domains.production.engine.synthetic scanner-model --out card.json             # model-card template
$PY -m domains.production.engine.synthetic generate --tier full --seedset QUAL --allow-qual \
    --scanner-model CARD.json --out ...                                               # final G1 run only
```

The amendment-4b model card is `docs/production/runlog/scanner_model_card.json` (E4,
`provenance: MODEL_CARD`, model sha256 `cf55a20c23f37e90...`). Pass it with `--scanner-model` for
every run that feeds a G1 scorecard. Without it the scans use the ASSUMED defaults, and E4 counts
those runs as non-qualifying.

`generate` behaviour:

- **Resume identity.** An existing case directory is skipped only when its `case.json` spec,
  generator version **and** generation identity all match the request. The generation identity
  (`case.json.generation`) holds the sha256 of the scanner model, the sha256 of the scan plan, the
  density policy and the file options (`--write-stage-steps`, `--write-as-built-step`). Any other
  existing directory is refused (exit 2, `CaseConflictError`) unless `--force` regenerates it. So
  re-running with the frozen amendment-4b model card can never silently reuse scans of another model.
- **One model per manifest.** `manifest.json` records `scanner_model` and `scanner_model_sha256`,
  and every case entry carries its own `scanner_model_sha256`. A chunked run whose model differs
  from cases already listed in the directory's manifest is refused before anything is generated.
  Use a fresh `--out`, or regenerate every listed case with `--force`.
- A case does not depend on its tier. `regression` cases are byte-identical to the same ids in
  `full`, so the tiers can share one output directory.
- Chunked runs (`--cases`, `--families`, `--limit`) merge into the directory's `manifest.json`, in
  plan order.
- Each case is built in a hidden temporary sibling directory and renamed into place, so a crash
  never leaves a half-written case.
- Stdout uses the engine protocol: `PROGRESS\t{json}` lines and a final `RESULT\t{json}`. The RESULT
  includes per-family coverage gaps (empty for `regression` and `full`).
- `--threads N` sets the Open3D/Embree raycasting threads. The default follows `OMP_NUM_THREADS`
  (all cores when it is unset), so a generator run beside an E3 runtime measurement stays
  single-threaded under the usual `OMP_NUM_THREADS=1`. Outputs do not depend on the thread count
  (tested).
- Exit codes: 0 success; 2 usage or policy error (bad arguments, QUAL without `--allow-qual`, a
  directory or manifest that does not match, an invalid scanner-model card); 3 internal error,
  including any failure inside geometry generation (traceback on stderr).
- `--write-stage-steps` adds `stages/S<k>.step` for every prefix stage S0..SN of the tagged order.
  These are real STEP BReps for any stage, useful for checking E1's stage models.
- `--write-as-built-step` adds the scanned as-built BRep (`as_built.step`) as truth.

`export-inbox` copies each scan under its ZEISS-like name the way the cell does it: write to a
temporary name, then rename. Given a part-set directory, it exports that set's scans in drop
order. Use it for hotfolder tests and the §6 E2E test.

**Part sets** (`part-set`, `cases.generate_part_set`) serve onboarding and E2E flows (plan §6 and
amendment 10), which need several scans of **one** released part. A part set writes the part
files once (STEP, flat, drawing, FINAL recipe) and `part_set.json` with the drop order and
per-scan expectations. Scans use the model's native spacing unless `--target-points` is given (an
explicit, recorded override). Its scans, under `scans/<name>/`:

- `nominal`: a known-good complete part.
- `errors`: exactly **+1.5 mm on a free flange side of B2 and +2.0° on B3**, with no other
  variation. The physically coupled small effects of the angle change are part of the truth, and
  `expected.injected` lists every non-zero true deviation.
- `stage_k`: an incomplete scan under the FINAL recipe (FINAL_INCOMPLETE, so NOT_YET_FORMED bends
  plus a backgauge).
- `stage_0`: the blank.
- `unknown_sku`: the nominal scan bytes under an unknown (opaque) SKU. Expect quarantine
  (`UNMATCHED_SKU`), then an admin assigns the SKU.
- `wrong_part`: another part scanned under this SKU.

A part set needs a family with >= 3 bends (HAT, CHANNEL_WITH_RETURNS, TWO_DIRECTION_TAB,
PANEL_WITH_HOLES_NOTCHES).

`cases.load_case(dir)` reads `case.json` + `truth.json` and verifies every listed file hash. A
tampered or partially copied case raises.

## 2. Case directory

| File | Content | Give to E3? |
|---|---|---|
| `part.step` | Nominal folded CAD: one valid solid, planes and true cylinders only, AP214, mm, byte-deterministic. Product name = the opaque SKU. | via E1 |
| `flat.dxf` | Nominal flat pattern (§5) | via E1 |
| `drawing.pdf` | Synthetic drawing: flat pattern, numbered bend lines, bend table. Labelled SYNTHETIC. | no |
| `recipe_draft.json` | C01 builder inputs (part, CAD spec rows, Vendor-Setup-shaped draft, release request, CAD/drawing/flat refs) plus a human summary: SKU, bends in order with nominal OML lengths and well-definedness, מהלך groups, tolerances, the flat's bend lines with their `bend_uid`. No case id. | no |
| `recipe.json` | Finalized `PartRecipeV1`, canonical JSON, built by `domains.production.recipe.builder.build_part_recipe` from the draft (reproducible byte for byte). No case id or scenario word. | yes |
| `scan.ply` | Virtual scan in the **raw scanner frame**. Binary LE, float32 xyz, plus triangles in mesh mode. Point order is a random permutation. | yes, staged under an opaque name |
| `truth.json` | Analytic truth (§4) | **never** |
| `truth_labels.npz` | Per scan point: `segment_kind`, `segment_index`, `skin` (contract skins of the physical part), `view`, `setup`, `flare`, `ideal_cad` (the noise-free hit in `cad_model_mm`). Index-aligned with `scan.ply`. | **never** |
| `case.json` | Spec, seed, generation identity, scenario, injections, ZEISS-like file name, SKU, features present, sha256 of every file | **never** |

A tier directory also holds `manifest.json` (cases, hashes, scanner model, per-family coverage)
and `generation_log.jsonl` (timings). Timings are kept out of the case files so the case files stay
deterministic.

## 3. Seed sets, tiers and scan density

- **DEV** is for iteration (the A1 loop, regression runs, debugging). **QUAL** is **final-run only**:
  G1 is gated on QUAL (amendment 4c).
  - The two sets use different salts (`cases.SEEDSETS`). Their seeds are disjoint, and a test
    asserts it.
  - Planning or generating QUAL needs an explicit opt-in: `allow_qual=True` in the Python API
    (`plan_cases`, `generate_case`, `generate_tier`, `generate_part_set`) or `--allow-qual` on the
    CLI (`generate`, `plan`, `part-set`). Otherwise `SeedsetLockedError` (CLI exit 2); `plan` prints
    nothing. The salt is in the source, so this is a policy guard, not secrecy.
  - Never tune anything on QUAL outputs. See the custody record at the end of this file.
- Case seed: `sha256("<salt>|<family>|<group>|<index>")` (63 bits). Every random draw comes from
  per-purpose streams seeded by `SeedSequence([seed, i])`, one each for geometry, errors, scan,
  pose, tolerance, wrong part, placement, naming, symmetry and policy. Changing one aspect therefore
  does not move the others.
- SKU: `SYN-` + 10 upper-case hex characters of `sha256("<salt>|sku|case|<seed>")`. It carries no
  family, template index or scenario. Part sets use the kind `partset`.

**Scan density.** `ScannerModel.point_spacing_mm` is the median nearest-neighbour distance of the
exported points, the statistic a model card measures on a real export (the 0.46 mm default is the
0.463 mm documented for the local physical references). For `regression`, `full` and part sets the
generator scans at that **native spacing**: it calibrates the merge voxel per case (secant
iteration, every pass restarting the scan stream from the same seed) until the exported cloud's
median NN spacing is within 3 % of the model's. The point count is then whatever the part's size
gives. An explicit point-count override exists only for `smoke` (fast tests) and `runtime` (the
runtime/RSS gate); it is recorded as `truth.scan.density.policy = "TARGET_POINTS_OVERRIDE"` with a
note never to score accuracy on it. `truth.scan.density` always reports the measured NN spacing,
the voxel, the passes and the policy.

| Tier | Cases | Density | Points per scan | Notes |
|---|---|---|---|---|
| `smoke` | 7 (one nominal per family) | override ~30 k | ~30 k | tests only |
| `regression` | 50 = 7 families x 7 + 1 symmetric | native | see §9 | **a subset of `full`** (same ids and seeds); runs on every A1 change |
| `full` | 216 = 7 x 30 + 6 symmetric | native | see §9 | G1 suite (>= 200 cases, >= 7 families) |
| `runtime` | 10 fixed cases, large parts (a flange >= 300 mm) | override ~1 M | ~1.0 M | runtime and RSS gate only; spacing is whatever 1 M points gives (§9) |

Coverage (amendment 5). Every family's `regression` and `full` lists contain:

- a stage-0 (flat blank) case and a stage-1 case;
- an acute bend (included < 90°);
- `Ri >= 4t`;
- a flange <= 12 mm and a flange >= 300 mm.

`manifest.coverage` verifies this on the generated truth, and the RESULT line prints any gaps. E4's
G1.SUITE gate is binding on the qualifying tier (`full`, QUAL, MODEL_CARD), and it fails on any gap.
So every feature is a property of the scanned part that a template forces, never a random draw:

| Feature | Definition (`case.json.features_present`) | Forced by |
|---|---|---|
| `stage_0` | no bend formed | STAGE_0 templates (2 and 19) |
| `stage_1` | exactly the first bend of the tagged order formed, fully, under either recipe stage | STAGE_1 templates (3 and 20) |
| `acute` | some nominal included angle < 90° | `FeatureRequest(acute)`, checked on the design |
| `large_radius` | some `Ri >= 4t` | `FeatureRequest(large_radius)`, checked on the design |
| `short_flange` / `long_flange` | some nominal OML flange length <= 12 mm / >= 300 mm | the family builders' requested ranges |

A test plans the DEV **and** QUAL template lists (seeds only, no geometry) and checks that every
family requests every feature. A second test releases a STAGE_1 scan FINAL and checks that it
still counts as stage 1.

Scan plans used by the templates:

- `FULL`: two setups (top and turned over), 11 views each.
- `TOP_ONLY`: one side only. Whole skins are hidden.
- `SPARSE`: 4 views.
- `SINGLE`: 1 view.

The more occlusion, the fewer points: G1 requires that occlusion raises UNKNOWN, not error.

## 4. Part model, frames and truth

**Families** (`families.py`):

| Family | Bends | Shape |
|---|---|---|
| `L_BRACKET` | 1 | two legs |
| `U_CHANNEL` | 2 | web + two flanges, same direction; a symmetric variant (equal flanges, 180°-symmetric holes) for the symmetry gate |
| `Z_STEP` | 2 | flange / riser / flange, opposite directions |
| `HAT` | 4 | brim / wall / top / wall / brim |
| `CHANNEL_WITH_RETURNS` | 4 | C channel with inward returns (lips) |
| `TWO_DIRECTION_TAB` | 3 | base with flanges about non-parallel axes (corner relief) + a narrow tab (10 mm or 8–13 % of the edge, at most 15 %) on one flange |
| `PANEL_WITH_HOLES_NOTCHES` | 3 | large panel, hole grid, top-edge notch outside F3's span; F1 covering 86–94 % of the +X edge; full-span F2; partial-span F3 with a notched free edge |

Draws:

- `t` ranges from 0.8 to 4 mm; `Ri` is 0.7–2 t, or 4–6 t on request.
- Included angles are 90° (60 %) or 95–150°. Acute angles are 40–85°.
- `K = 0.33`. It equals the shop default, so stage models match whether or not E1 binds the flat.
- Profiles are asymmetric and hole patterns chiral unless symmetry is requested.

**Conditioning.** Every flange side and backgauge run is laid out so that its far boundary has one
dominant level, under E1's station rule as well as E2's (see *well-definedness* below), or is
clearly mixed, in which case the recipe rule makes it NOT_INSPECTED. In practice: partial-span
flanges and tabs cover well under 20 % or well over 80 % of the edge a neighbouring bend marches
across, and notches stay out of corridors whose far boundary they would split near 50/50. (The
first release put the PANEL top notch over 39–56 % of F3's corridor and made F1 span 32–73 % of the
base edge; E1 then disagreed with E2 by up to 14 mm on REQUIRED nominals. The review found this.)

**Model** (`geometry.py`). A part is a flat blank made of panel polygons (with holes) plus
straight bends.

- Each bend owns a strip of neutral length `BA = sweep·(Ri + K t)`.
- Forming a bend to an actual sweep α consumes `α(Ri + K t)` of its strip, centred on its line;
  the remainder stays flat. Folding is exact rigid kinematics: `T_b = Rot(axis, α)·Trans(−BA_act·m)`.
- An unformed bend is the identity, so a stage-k model is exactly the contract's Method A stage
  model (a test checks this).
- The blank and its holes are fixed. Errors act physically:
  - a bend-line shift moves material from one side to the other;
  - a cut error moves a cut line (all collinear free edges of one straight cut together);
  - an angle error changes the consumed neutral length.
- Cut errors never move a free edge that touches a bend attachment edge, because the bend's strip
  would have to widen with the blank. The geometry refuses such an offset, and a test checks that
  every as-built surface is watertight (zero vector area).

**BRep** (`brep.py`):

- Panels are prisms of their effective polygons, so hole walls are true cylinders.
- Formed bends are `MakeRevol` sectors, giving true cylinders of radius `Ri` and `Ri + t`.
- Pieces are fused and unified into one valid solid, whose volume equals the analytic volume to
  1e-6 relative; this is also the self-intersection check.
- Mirrored parts are never built directly: the unmirrored solid or tessellation is reflected.
- The STEP writer normalises the header time stamp and the process-numbered PRODUCT name, so
  bytes depend on content only.

**Frame.** All truth geometry is in `cad_model_mm`, the STEP frame.

- The **datum** is the largest panel; truth records the rule and the panel.
- The as-built datum panel's cut features (plane, free edges, holes) sit at their nominal CAD
  position. That is the registration truth.
- `scan.raw_to_cad` is the exact rigid transform from scan to CAD.
- Half the parts have a random CAD placement, so nothing may assume Z-up.
- Per-panel transforms from the reference stage to the as-built pose are included, so truth can
  be re-expressed if E1 picks another datum.

**Conventions** (ENGINE_IO §1):

- Angles are included angles.
- Flange lengths are OML. Per station along the bend corridor (span trimmed by `min(2 mm, 5 %)`,
  41 stations) the far boundary is the **farthest crossing** of the station line with the material:
  the contract's per-bin maximum extent, so a notch or hole the line passes over does not end the
  flange. The crossed boundary segment says whether it is a free edge or a bend. The value is the
  median over the stations of the majority boundary kind. The minimum, maximum, per-kind groups and
  levels are also reported.
- Side A follows the +X rule at the nominal final pose. The duplicated rule is tested against
  `compiled_part.assign_sides`.
- Contract skins and UP/DOWN are mapped from the flat frame through the placement.
- Backgauge (stage k, the just-formed bend k) is the same measurement from the OML sharp across the
  unformed bends to the farthest material of the flat island. Truth gives both sides when both lead
  to unformed material.

**Well-definedness.** Station values are grouped into *levels*: sorted values chained within
0.05 mm inside one boundary kind, the grouping E1 uses. A far boundary is `well_defined` when its
largest level holds at least 80 % of the valid stations. Its median is then that level under any
reasonable station placement: E1 samples 21 stations with a `min(0.5 mm, 2 %)` trim and E2 41 with
`min(2 mm, 5 %)`. A test checks that both rules give the identical value on every REQUIRED flange
side and every well-defined backgauge side of 112 seeded designs at the nominal pose, and on every
well-defined side of 42 designs as built (process variation, shifts, angle errors, partial stages).
A stepped edge or a mixed boundary near a 50/50 split is not well defined: its median flips with
station placement, for the nominal and for a scan alike.

- Flange sides: the recipe rule (§5) never makes an ill-defined side REQUIRED.
- Backgauge sides cannot be excluded per side by a recipe, so truth flags them
  (`per_side[side].well_defined`, `ill_defined_reason`). The current families produce none on the
  tested seeds.
- `ill_defined_reason`: `MIXED_FAR_BOUNDARY` (the majority kind holds < 80 %),
  `STEPPED_FAR_BOUNDARY` or `NO_FAR_BOUNDARY`, prefixed `NOMINAL_` or `TRUTH_`.
- An ill-defined side keeps its station statistics for diagnostics, but **must not be scored**.

**Bend-line position.** The contract defines it as the projection of the actual sharp's offset
onto side A's outward direction. That projection is blind to a displacement within side B's
plane. Example: a backgauge shift where side A is the vertical flange reads 0.0 by contract. Truth
therefore also reports `offset_vector_mm` and `normal_component_mm`. Open issue for C01/E3/E4.

**`truth.json`** (`sherman_synthetic_truth_v1`):

- `bends[]`: per recipe bend, the `nominal` block (sweep, included angle, Ri, t, K, neutral length,
  contract direction and OML skin, axis, all final-pose loci, and sides A/B with outward direction,
  boundary kind and OML length) and the `actual` block (state, sweep, included angle, formed
  fraction, shift, loci, bend-line offset).
- `characteristics{}` uses contract ids: `<bend_uid>/FLANGE_LENGTH/A|B`, `/INCLUDED_ANGLE`,
  `/BEND_LINE_POSITION`, and `/BACKGAUGE_LENGTH/S<k>` (per side). Each carries `nominal`, `truth`,
  `true_deviation`, the recipe `tolerance` and `recipe_requirement`, `truth_in_tolerance` (the
  false-PASS reference: true |dev| against the tolerance), `applicable` with
  `not_applicable_reason`, station statistics (`nominal_stations` / `truth_stations` with
  `dominant_level_fraction`, `well_defined`, `groups`, `levels`), `well_defined` and
  `observability`.
- `stage`: formed set, partial bends, `is_sequence_prefix`, `order_consistent` (מהלך groups share a
  rank), out-of-order formed bends, `expected_stage_model_id`, the recipe's `inspection_stage` and
  the missing bends of a FINAL recipe.
- `expected`:
  - `bend_states`, plus `bend_states_contract`, where an out-of-order formed bend becomes
    `UNEXPECTED_FORMED`;
  - `summary_never` (for example ALL_MEASURED_IN_TOLERANCE for an incomplete, missing, extra,
    mirrored or wrong part);
  - `characteristics_never_pass`: applicable, well-defined characteristics with a true deviation
    out of tolerance, plus every characteristic of a wrong or mirrored part. **A backgauge entry is
    `<characteristic id>#A` or `#B`**: truth gives both sides, E1 attaches the characteristic to one
    of them, so match on E1's side;
  - `characteristics_never_fail`: characteristics of NOT_YET_FORMED bends;
  - `characteristics_must_be_unknown`: applicable characteristics without a defined truth
    (`ADJACENT_BEND_NOT_FORMED`: a flange whose far boundary is a still-flat bend; or
    `NO_FAR_BOUNDARY`). The engine must report them NOT_MEASURED, never PASS or FAIL;
  - `characteristics_ill_defined`: never scored (backgauge sides as `#side`);
  - `symmetric` (bend identity ambiguous: a rotation self-map or a non-trivial bend orbit),
    `ambiguous_bend_groups`, and `mirror_image_rms_mm` / `mirror_distinguishable_by_rms`.
- Scoring rule for E4: score a characteristic only when `applicable`, `truth` is not null and
  `well_defined` is true (for a backgauge, the side E1 chose must be well defined).
- `symmetry`: the truth-side self-registration test on the reference stage model. Candidates are
  the PCA 180° rotations and reflections, refined by ICP against the surface; RMS is measured
  point to surface; the threshold is 0.3 mm. `is_symmetric` has the contract's meaning, so it is
  true for most profile parts through the mirror plane perpendicular to the bends.
- `stage_models_nominal`: S0..SN with formed sets, final→stage panel transforms and stage bend
  lines, to cross-check E1.
- `scan`: point and face count, `spacing_mm` (the measured median NN spacing), `voxel_mm`,
  `density` (§3), scanner model and its sha256, plan, statistics including camera positions and
  duplicate-skin counts, `cad_to_raw` / `raw_to_cad`, the ZEISS name and the label legend.

**Undefined truth** is explicit (`truth: null` + `truth_undefined_reason`); a nominal is never
copied into an actual:

- `WRONG_PART_SCANNED` and `MIRRORED_PART`: every characteristic;
- `BEND_NOT_FORMED`: an unformed bend; NOT_APPLICABLE / `BEND_NOT_YET_FORMED`;
- `ADJACENT_BEND_NOT_FORMED`: a flange whose far boundary is still flat. The stage backgauge
  covers it.

**Labels.** `truth_labels.npz` `skin` holds the contract skins of the **physical** part: skin 0 is
the face of the scanned part's datum whose CAD-frame normal has the larger +Z (ties +Y, then +X).
For a mirrored part that is the reflected datum normal, for a wrong part its own datum. A test
checks it geometrically, on a mirrored case where the flat and contract skins differ.

`observability` is a heuristic aid computed from the labelled noise-free points. Per side it gives
OML-skin support points in the corridor and far-edge coverage over `max(5, span/5)` bins within
`max(1 mm, 2·spacing)`. It is `observable` when there are >= 200 points and coverage >= 0.6.
Angle and position characteristics get hinge-band support. It is not the engine's rule.

## 5. Flat DXF and recipe

**DXF** (ezdxf R2010, `$INSUNITS = 4`), viewed from flat +Z:

- `OUTLINE`: the blank's outer loop as one closed LWPOLYLINE (an Euler tour of the panel tree, with
  collinear vertices merged), and holes as CIRCLE entities.
- `BEND`: one dashed LINE per bend (the strip centre, spanning the bend) and one TEXT note per
  bend, `UP 90.00° R2.00`. **The note prints the bend (sweep) angle**, as CAD flat exports do;
  included = 180 − sweep.
- A test checks that the blank length equals Σ OML − Σ BD, with `BD = 2 Ro tan(α/2) − α(Ri + K t)`.

**Recipe:**

- Marks lie on the native mid-radius line (`Ri + t/2` from the axis, mid-sweep), and
  `target_angle` is the sweep (`NATIVE_SWEEP`). The recipe's included angle therefore equals the
  compiled value.
- `confirmation = DEV_AUTO_CONFIRMED`: the recipe is built from truth, not by an operator.
- SKU, part id and part name are opaque (`SYN-<hex>`, `part-syn-<hex>`, `Synthetic sheet-metal
  part SYN-<hex>`); the release note is a fixed string. A test checks that no recipe-side file
  (recipe, draft, flat, STEP) contains the case id, the template index or a scenario word.
- The flat is released `UNBOUND`, because its bend-line ids belong to E1's parser. The draft
  summary lists the generator's flat bend lines with their `bend_uid` for any binder.
- Default tolerances (SHOP_STANDARD): angle ±1°, flange ±0.5 mm, bend-line position ±0.5 mm,
  backgauge ±0.3 mm.
- About 25 % of cases override one bend's tolerance (flange A ±0.3 mm or angle ±1.5°).
- **Exclusion rule** (pre-written, per feature; amendment 4d): a flange side is NOT_INSPECTED when
  its far boundary has no value, mixes boundary kinds (the majority kind holds < 90 % of the
  stations) or is not well defined (no 0.05 mm level holds >= 80 % of the stations). Those sides have
  no station-placement-independent nominal, so a careful operator would not tolerance them.
- **Inspection stage policy** (blindness): complete-part, mirrored and wrong-part cases are
  released FINAL, except 25 % IN_PROCESS; staged, partial and order-swap cases IN_PROCESS, except
  25 % FINAL (a mid-sequence scan of a FINAL recipe: FINAL_INCOMPLETE); missing-bend and
  extra-bend cases are always FINAL, as G1 requires. So the recipe's stage does not tell E3 the
  scenario class. The ZEISS-like name follows the physical scan, not the recipe.

## 6. Scenarios and injected errors (`injection.py`)

| Kind | Effect |
|---|---|
| `PROCESS_VARIATION` | known-good scatter: angle σ 0.15°, bend-line shift σ 0.03 mm, cut σ 0.03 mm per non-datum cut line (collinear free edges of one straight cut move together; edges touching a bend attachment never move) |
| `ANGLE` | included-angle delta on one bend, ±0.2–3.0° absolute (the default angle tolerance is ±1°; ~25 % of cases override one bend's tolerance) |
| `SPRINGBACK_GROUP` | the same +0.5–2.5° on every bend of a מהלך group |
| `FLANGE_LENGTH` | cut error on a non-datum free far end, ±0.1–1.5 mm: every free edge parallel to the bend in the side's far half (the far edge segments and any notch bottom) moves, so every level of the far boundary shifts by exactly δ |
| `BEND_LINE_SHIFT` | backgauge error towards side B. Side A grows and side B shrinks by exactly δ. Mid-stage cases inject ±0.5–1.5 mm on the just-formed bend, so the backgauge error is >= 0.5 mm. |
| `PARTIAL_BEND` | one bend at 30–70 % of its sweep. Values outside that range are refused. |
| `ORDER_SWAP` | stage k with bend i <= k left flat and bend j > k formed instead, drawn only among swaps that are really out of order (j ranks above the lowest unformed rank). A swap inside one מהלך group is a legal reorder, never labelled ORDER_SWAP; a part with no out-of-order swap falls back to MISSING_BEND. |
| `MISSING_BEND` | a FINAL part with one bend flat |
| `EXTRA_BEND` | an unplanned 15–40° crease splitting a free rectangular flange |
| `MIRRORED` | the physical part is the mirror image |
| `WRONG_SKU` | another part of the same family is scanned under this SKU |

`COMPLETE_NOMINAL` template 0 of every family is exact nominal (zero deviation). The other cases
carry process variation.

## 7. Virtual scanner (`scanner.py`)

- Setup: Open3D `RaycastingScene` (Embree). The pinhole camera covers the part's projected
  bounding box, and only first hits count, which gives exact self-occlusion.
- The projector sits on a 20° triangulation baseline. A shadow ray means no point.
- Dropouts: grazing (linear between the start and cutoff angles), specular (projector mirror
  direction into the camera) and random.
- At range-image depth discontinuities: edge erosion, and edge flare (mixed pixels pushed up to
  `edge_flare_mm` away from the camera).
- Range noise along the ray grows with incidence. Each view also gets a small residual
  registration jitter.
- Merging, like the cell software fusing all measurements into one cloud:
  - within a setup, overlapping views are merged by voxel ownership (the best-incidence view keeps
    the voxel);
  - across setups the same rule applies, so overlapping setups do not double the surface, except
    inside **duplicate-skin patches**: cells of `duplicate_skin_patch_mm` (25 mm) in the raw frame,
    each doubled with probability `duplicate_skin_fraction` (0.2) by a position hash. Every setup
    after the first carries a rigid misregistration (`duplicate_skin_offset_mm`,
    `duplicate_skin_rotation_deg`), so those patches show doubled, offset skins. (The first release
    doubled every overlap, 3–63 % of the second setup's points.)
  - The merge-level randomness (misregistrations, the patch hash seed) is drawn before any per-view
    draw, so the density calibration sees a smooth response to the voxel size;
  - sparse outliers are added last, and the point order is randomly permuted: outliers, setups
    and views interleave in the file, and mesh faces are re-indexed.
- Each scan gets a uniformly random part pose.
- Output is `points` (one point per voxel) or `mesh` (range-grid triangles, facing the camera).
  The voxel is calibrated per case to the model's spacing (§3); `stats.nn_spacing_mm` is the
  measured median NN spacing of every scan.

`ScannerModel` is a JSON-loadable dataclass (`schema sherman_synthetic_scanner_model_v1`,
validated, unknown fields rejected). A `MODEL_CARD` must state every field, so nothing silently
falls back to an assumed default. `ScannerModel.sha256()` is its identity in `case.json` and the
manifest. The defaults are **ASSUMED** (provenance field), except the 0.46 mm spacing, which follows
the median nearest-neighbour spacing documented for the local physical references in
`scripts/generate_f1_calibrated_raw_range_synthetic_corpus_v1.py` (0.463 mm). E2 read no real scans.
The amendment-4b **model card** (E4, measured on >= 3 authorized real scans) replaces the defaults
with `--scanner-model card.json` (`provenance: MODEL_CARD`); measure `point_spacing_mm` the same way
(median distance to the nearest other point over the exported cloud). `ScannerModel.idealised()` is
for diagnostics only and must never be used for G1. The plan forbids quoting idealised accuracy.

| Parameter | Default | Meaning |
|---|---|---|
| `point_spacing_mm` | 0.46 | median nearest-neighbour distance of the exported points (the generator calibrates to it) |
| `noise_sigma_mm` / `noise_incidence_gain` | 0.025 / 1.0 | range σ at normal incidence; σ·(1 + g(1/cosθ − 1)) |
| `edge_flare_mm` / `edge_flare_probability` | 0.4 / 0.25 | mixed-pixel push at depth discontinuities |
| `edge_erosion_probability` | 0.2 | boundary pixels dropped |
| `grazing_start_deg` / `grazing_cutoff_deg` | 65 / 82 | dropout ramp |
| `specular_cone_deg` / `specular_dropout_probability` | 6 / 0.6 | hot-spot dropout |
| `random_dropout_probability` | 0.002 | |
| `outlier_fraction` / `outlier_range_mm` | 0.0015 / 4 | half sky, half near-surface |
| `duplicate_skin_offset_mm` / `duplicate_skin_rotation_deg` | 0.04 / 0.01 | per-setup misregistration σ |
| `duplicate_skin_fraction` / `duplicate_skin_patch_mm` | 0.2 / 25 | share of cross-setup overlap (in patches) kept doubled |
| `view_jitter_mm` | 0.008 | per-view residual registration σ |
| `triangulation_angle_deg` | 20 | projector–camera angle (shadowing) |
| `working_distance_mm` | 650 | sensor standoff |

**ZEISS-like names** (`case.json.zeiss_filename`, informational only; E3 never sees them,
amendment 4):

- `BEFORE BENDING  <SKU>_<REV>.ply` (stage 0);
- `<SKU>_<REV>_SCAN_<k> 1.ply` (k = bends the operator has worked on);
- `<SKU>_<REV>_FULL_SCAN 1.ply` (a part meant to be complete, missing bend included);
- occasionally with a ` (1)` copy marker, or without the export counter.

Revisions are drawn from A, B, C, 01 and 02. All names parse OK with the C01 parser (tested).

## 8. Determinism, performance, custody

- Every case file is byte-deterministic for a given case id, seed, scanner model and options,
  **across processes**. A test regenerates a case under two different `PYTHONHASHSEED` values. This
  covers STEP, DXF, PDF (ReportLab invariant), JSON, PLY and the npz (fixed zip timestamps). ezdxf
  writes CLASS records in set order, which made DXF bytes depend on the hash seed; the writer pins
  that order for its own document.
- The raycasting thread count does not change any byte (tested with 1, 4 and all threads).
- Measured performance is in §9 (single process, `OMP_NUM_THREADS=1`, so raycasting is
  single-threaded too; Embree may still build its BVH with its own threads).
- Synthetic only: no real scan, CAD, HOLDOUT or sealed truth is read. Results on E2 cases are
  synthetic engineering evidence and are never quoted as physical accuracy.
- Generate outside the repo (for example under `$TMPDIR` or the data root): the full tier is about
  1–2 GB.

## 9. Measured numbers (2026-10-17, DEV, this machine)

### Generator 1.1.1 with the frozen amendment-4b model card (evening re-run)

Conditions: a single process, `OMP_NUM_THREADS=1`, generated under the session scratch directory
(outside the repo), `/usr/bin/time -l` peak RSS. Other agents were running on the same machine.
`--scanner-model docs/production/runlog/scanner_model_card.json` (MODEL_CARD, point spacing 0.401 mm,
noise 0.0046 mm, no duplicate skins and no outliers measured). The `full` and `runtime` runs used
the 1.1.0 code (fingerprint `a41d11077b9b7a7a`); under 1.1.1 only their `case.json` files would
differ (shown on `regression` below). The 1.1.1 code fingerprint is `0023e758e51007ab`: the first
16 hex of the sha256 of the package's `shasum -a 256 *.py` listing.

| Run | Cases | Wall time | Peak RSS | Points per scan (min / median / max) | NN spacing (min / median / max) | Disk |
|---|---|---|---|---|---|---|
| `full` DEV, card | 216 (7 families, no coverage gaps) | 746 s (3.5 s per case) | 2.29 GB | 29 k / 221 k / 918 k | 0.390 / 0.401 / 0.413 mm | 1.5 GB |
| `regression` DEV, card | 50 (no coverage gaps) | 171 s | 1.91 GB | 36 k / 241 k / 679 k | 0.390 / 0.401 / 0.412 mm | 0.39 GB |
| `runtime` DEV, card | 10 (override) | 147 s | 2.53 GB | 0.9997 M / 1.0003 M / 1.0009 M | 0.148 / 0.263 / 0.350 mm | |
| `regression` DEV, ASSUMED, 1.1.0 code | 50 | 133 s | 1.63 GB | 24 k / 163 k / 520 k | ratio to model 0.973 / 1.002 / 1.027 | 0.27 GB |
| `regression` DEV, ASSUMED, 1.1.1 code | 50 | 137 s | 1.54 GB | identical scans | identical | |

- The card's finer spacing gives about 1.4 to 1.5 times the ASSUMED point count. The regression
  median is 241 k points, which is amendment 3's "about 200 k". Every native-spacing scan is within
  3 % of the card's spacing, after 1 or 2 calibration passes (full: 59 and 157 cases).
- **1.1.0 to 1.1.1 byte check** (`regression`, ASSUMED): all 400 non-`case.json` files are
  identical. `case.json` differs in `generator_version` everywhere and in `features_present.stage_1`
  in 6 cases:
  - `LB-000` and `LB-001`: complete L-brackets released FINAL (a one-bend part's stage 1 is the
    finished part), now counted;
  - `ZS-005`: a two-bend stage-1 scan released FINAL, now counted;
  - `LB-005`: a partially formed single bend, no longer counted;
  - `PN-006` and `UC-006`: one-bend order swaps, no longer counted.
- On the `full` DEV tier the new definition counts 2 to 18 stage-1 cases per family (at least the
  two STAGE_1 templates). Under the old definition HAT's second stage-1 case was an order swap
  (`E2-DEV-HT-012`, one bend formed out of order), because `E2-DEV-HT-020` (STAGE_1) was drawn
  FINAL.
- The scenario mix, inspection stages and characteristic counts equal the 1.1.0 numbers below.
  The truth does not depend on the scanner model.

### Generator 1.1.0 with the ASSUMED model (afternoon acceptance runs)

These come from the acceptance runs (a single process, `OMP_NUM_THREADS=1`, on the development Mac;
generated outside the repo). The runs used code fingerprint `c2eb380f500489f8` (the first 16 hex of
the sha256 of the package's `shasum -a 256 *.py` listing). The code after them only adds an
up-front existing-case check to `generate_tier` (`eead340a06d756bc`, which regenerated `regression`
byte for byte, all 450 files and `manifest.json`, in 128 s with a 1.59 GB peak) and then drops an
unused import (final fingerprint `a41d11077b9b7a7a`).

| Run | Cases | Wall time | Peak RSS | Points per scan (min / median / max) | NN spacing (min / median / max) | Disk |
|---|---|---|---|---|---|---|
| `full` DEV | 216 (7 families, no coverage gaps) | 525 s (about 2.4 s per case) | 1.85 GB | 20.5 k / 156 k / 658 k | 0.447 / 0.460 / 0.473 mm | 1.0 GB |
| `regression` DEV | 50 (no coverage gaps) | 129 s | 1.60 GB | 24 k / 163 k / 520 k | 0.448 / 0.461 / 0.472 mm | 0.27 GB |
| `runtime` DEV | 10 (large parts, override) | 131 s | 2.47 GB | 0.9996 M / 1.0004 M / 1.0019 M | 0.148 / 0.256 / 0.348 mm | 0.23 GB |
| part set (HAT) | 1 part, 6 scans | 24 s | 1.04 GB | native (stage_k: 267 k) | 0.457 mm (stage_k) | |

- Density: every native-spacing scan is within 3 % of the model's 0.46 mm (1–3 calibration passes;
  61 cases needed 1, 150 needed 2, 5 needed 3). Points per scan follow the part size: median per
  family from 77 k (L_BRACKET) to 275 k (PANEL); 10th / 90th percentile 62 k / 303 k. The runtime
  tier reaches 1 M points by densifying (NN 0.15–0.35 mm); at native spacing its parts would give
  about 0.1–0.6 M points (estimated as 1 M x (NN / 0.46)^2). It is a runtime/RSS gate only.
- Duplicate skins (FULL plan): median 4.5 %, 90th percentile 11 %, maximum 18 % of the points.
- Scans: 174 point clouds and 42 meshes; 187 FULL, 22 TOP_ONLY and 7 SPARSE plans.
- Scenario mix: COMPLETE_ERRORS 50, COMPLETE_NOMINAL 38, STAGE_K 24, PARTIAL_BEND 18, STAGE_0 15,
  STAGE_1 15, COMPLETE_SPRINGBACK 14, MISSING_BEND 11, EXTRA_BEND 8, MIRRORED 8, WRONG_SKU 8,
  ORDER_SWAP 7. Every ORDER_SWAP is out of order; 2 ORDER_SWAP specs fell back to MISSING_BEND
  (an L bracket, and a U channel whose two bends form one מהלך group).
- Recipe inspection stage: FINAL 124, IN_PROCESS 92. Of the complete-part cases, 26 of 102 are
  IN_PROCESS; of the staged, partial and order-swap cases, 19 of 79 are FINAL; mirrored and wrong
  parts are 6 of 16 IN_PROCESS; all 19 missing- and extra-bend cases are FINAL.
- Characteristics: 2328 bend characteristics; 1618 applicable after NOT_YET_FORMED and
  NOT_INSPECTED are removed. 278 applicable characteristics are truly out of tolerance (the
  false-PASS reference set). 1581 (98 %) are observable by the coverage heuristic. 60 are
  must-be-unknown (flanges ending at a still-flat bend). 0 are ill-defined; all 47 backgauge sides
  are well defined.
- 17 cases are bend-identity symmetric. 5 of the 8 mirrored cases are distinguishable by RMS; the
  other 3 differ only by holes.

Determinism:

- `regression` and `full` ran in separate processes: all 450 files of the 50 shared cases are
  byte-identical, `case.json` included.
- 77 cases regenerated with `--write-as-built-step --write-stage-steps` have byte-identical main
  files (616 files); only `case.json` (its generation options) and the extra STEP files differ.
- Re-running `regression` on its directory skips all 50 cases; re-running it with another
  scanner-model card exits 2 (`CaseConflictError`) and changes nothing.
- Robustness: the `full` tier generated on a scratch seed salt that is neither DEV nor QUAL (see §11):
  exit 0, 216 cases, no coverage gaps, 549 s, 1.61 GB peak; NN spacing 0.447–0.474 mm; 22.6 k–692 k
  points; 0 ill-defined sides (46 backgauge sides); 277 truly out-of-tolerance applicable
  characteristics; 62 must-be-unknown.

### Independent BRep cross-check (2026-10-17)

The reviewer's own OCP script (it locates bends only by the truth axis and measures everything from
the BRep faces and edges), switched to the farthest-crossing far boundary that 1.1.0 implements, on
`as_built.step` / `part.step` of 77 full-tier cases (templates 1, 4, 5, 8, 10, 11, 12, 14, 17, 24, 27
of every family): 645 quantities (included angles, station-median OML flange lengths including
extra-bend boundaries, backgauge truth, bend-line position on complete parts with side A by the +X
rule on the measured nominal BRep), all within 1e-5 mm or degrees.

**Committed version (`tests/test_prod_synth_crosscheck.py`, generator 1.1.1).** The same
independent measurement now runs as a test on generated case directories read back from disk:

- **STEP files.** A bend is a coaxial cylinder pair; the OML sharp is the intersection of the two
  tangent planes. Lengths are taken per station (the contract's 41 stations) as the farthest
  crossing of the station line with the outer wire of the side's planar face. The exit edge's
  neighbouring face names the boundary kind.
- **DXF file.** Read with ezdxf only. Every length is walked across the blank and developed with
  the bend deductions printed in the bend notes.

There are nine cases: `LB-001`, `UC-003`, `ZS-007`, `HT-005`, `CR-011`, `TD-008`, `TD-023`,
`PN-010` and `PN-014`. Together they cover every family, the angle, flange and bend-line-shift
errors, springback, prefix stages with backgauges (including notched and partial-span islands), a
partial bend and an unplanned crease. The test compares:

- `part.step`: 50 nominal flange sides, every nominal sharp and included angle;
- `as_built.step`: 29 defined as-built flange truths, the formed-bend set and the crease;
- `stages/S<k>.step`: the stage bend lines, 5 backgauge sides (nominal and truth) and 18 bend-line
  positions (the value and the 3D offset vector);
- `flat.dxf`: 50 developed flange sides and the 5 developed backgauges.

The largest disagreement is 1.2e-10 mm on the STEP files and 1.4e-14 mm on the DXF (the test allows
1e-6). A mutation test confirms that each check fails on a 1e-3 truth error, on `part.step`
substituted for the as-built part, on a DXF bend line moved by 0.01 mm and on a wrong bend note.
Sides whose far boundary is partly a still-flat bend are skipped, because that boundary leaves no
edge in a BRep.

The same nominal-STEP and DXF checks were also run over every case of the `full` DEV tier (216
cases, model card): 1164 flange sides on `part.step`, plus 1164 developed flange sides and 47
developed backgauges on `flat.dxf`. There were no failures; the worst disagreement was 1.5e-10 mm
(STEP) and 1.1e-13 mm (DXF), in 51 s.

### Cross-check against E1 (2026-10-17, informational)

`python -m domains.production.engine.compile` (the E1 code in the tree at about 06:10, while its
owner was still editing it) on all 50 `regression` cases and all 30 `full` PANEL cases (73 distinct
cases; 7 PANEL cases are in both sets): every case COMPILED. Compared with E2's independent analytic
truth (the reviewer's comparison script):

- REQUIRED flange sides (267 in `regression`, 179 in the PANEL set): identical requirement and
  boundary kind, nominals within 3.4e-7 mm (PANEL: 5.7e-14 mm); no `NOMINAL_UNAVAILABLE`;
- backgauge nominals on E1's chosen side (13 and 8): within 4.6e-7 mm;
- included angles identical; OML virtual sharps within 7.7e-7 mm; stage bend lines within 1e-3 mm;
- symmetry: 8 of 46 regression stage rows differ on the contract `is_symmetric` flag (mirror-only
  planes perpendicular to the bends, parts symmetric except small chiral holes, E1 near its 0.3 mm
  threshold), none on ambiguous bend groups. **E1's test is the contract authority for AMBIGUOUS**;
  treat E2's symmetry block as a diagnostic.

Before the fix the same comparison found PANEL REQUIRED nominals off by up to 14.4 mm and
`NOMINAL_UNAVAILABLE` on 4 REQUIRED flange sides and 3 backgauges; those were definitional, and the
layout and rule changes in §4 and §5 removed them.

## 10. Known limitations

- The symmetry candidate set (PCA 180° rotations and reflections) does not enumerate 90°
  symmetries of square-like frames. No family relies on them.
- Small chiral features (holes) barely move the RMS. A mirrored profile part can be
  RMS-indistinguishable (`mirror_distinguishable_by_rms = false`), so detecting it needs
  feature-level checks (holes or notches against the CAD). E4 should stratify mirrored cases by this
  flag.
- Bend regions are ideal circular arcs (no springback-induced radius change, no thinning, no
  hole distortion near bends). Notches never touch bend strips.
- Flats are released UNBOUND (see §5). E1 parses them (all bend lines found) and proposes a
  binding in its diagnostics. The developed-length check stays NOT_EVALUATED until a binding is
  confirmed.
- Bend-line position follows ENGINE_IO §1: it is the projection on side A's outward direction.
  A sharp displacement within side B's plane reads as 0 (see §4). The truth carries the full
  offset vector.
- General measurements are not generated, so recipes have none.
- The voxel grid of the merge is axis-aligned in the raw frame, so the local point density varies
  somewhat with surface orientation (the median NN spacing of the whole cloud is what is
  calibrated).
- A swap inside one מהלך group (order-consistent, not a sequence prefix) is never generated: the
  contract has no stage model for it (open issue for C01/E1).
- The sheet thickness of the scanned part always equals the CAD's (E3 notes this in
  `ENGINE_INSPECT.md` §10.16). Thickness tolerance (stock about ±0.05–0.1 mm) is therefore not
  exercised, and neither is E3's measured-thickness path. Adding it changes every case's bytes and
  belongs to a new generator version with its own truth rules: the OML moves with t while the blank
  and the tooling stay fixed.
- The `regression` tier is the first 7 templates of `full`, and all of them use the FULL scan plan.
  Occluded plans (TOP_ONLY, SPARSE) appear only in `full` (22 + 7 cases), so per-change regression
  runs never exercise "more occlusion raises UNKNOWN". The plan is part of the case identity, so
  changing it would change the regression ids that E3/E4 tests and the freeze use.
- An unplanned crease (EXTRA_BEND) turns the split flange's truth into OML-to-crease
  (ADJACENT_BEND, `boundary_changed: true`), while the recipe keeps its FREE_EDGE nominal. That
  side's true deviation is large and it is in `characteristics_never_pass`.

## 11. QUAL custody record

- 2026-10-17, generator 1.0.0: the implementer generated the full QUAL tier once as a crash check
  (exit 0, 216 cases, no coverage gaps). The outputs were deleted immediately; only the exit code,
  case count and gaps were read. The independent review flagged this as contrary to amendment 4c
  ("QUAL used only on the final run"). This record is for the engineering ledger.
- Generator 1.1.0 produces different bytes from 1.0.0. The fix stage generated no QUAL cases: it
  verified robustness on DEV and on a scratch seed salt that is neither DEV nor QUAL (§9). The tests
  plan QUAL specs (seeds only, with the explicit opt-in) to check seed-set disjointness and the
  guard; they never build QUAL geometry. The final G1 run should be the first 1.1.0 QUAL generation
  and must record the scanner-model sha256 it used.
- 2026-10-17 evening, generator 1.1.1: no QUAL geometry was generated and no QUAL output was read.
  The new coverage test plans the QUAL template rows (seeds only, with the opt-in) to show that every
  family requests every amendment-5 feature. The 1.1.1 coverage change was derived from the
  construction (a policy draw must not decide coverage) and checked on DEV only. The final G1 run is
  the first QUAL generation of 1.1.1. It should use the frozen card
  (`docs/production/runlog/scanner_model_card.json`, model sha256
  `cf55a20c23f37e9076426f0be559584fa9f00941db5d856ddc109bac4cabf8ce`).

## 12. Tests and acceptance checks

Run with `env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider
tests/test_prod_synth_*.py -q`. That is 127 tests in about 50 s on the development Mac.

| File | What it asserts |
|---|---|
| `test_prod_synth_geometry.py` | analytic truth against the written STEP for every family (sharps, angles, flange lengths to 1e-6), as-built stages with errors, a 1e-3 truth error detected; flat plate at S0; Method A stage models; closed-form developed length; DXF determinism and area; side rule = contract; design validation; mirrored parts by reflection; watertight as-built surfaces |
| `test_prod_synth_crosscheck.py` | truth.json of generated cases against `part.step`, `as_built.step`, `stages/S<k>.step` and `flat.dxf` per station (§9), mutations detected; every planned DEV SKU parses OK in every ZEISS-like name shape and revision with the C01 parser, a wrong revision is REVISION_MISMATCH, unknown-SKU drops are UNMATCHED |
| `test_prod_synth_truth.py` | injected errors are exact (angle, cut, shift, springback, partial, order swap, missing, extra, mirrored, wrong SKU), backgauge closed form, NOT_YET_FORMED never FAIL, undefined truths explicit |
| `test_prod_synth_conditioning.py` | REQUIRED flange and well-defined backgauge nominals do not depend on the station rule (E1's or E2's), farthest-crossing far boundary, stepped-edge flagging, collinear cuts, partial-span layouts |
| `test_prod_synth_scanner.py` | first-hit occlusion, hidden skins, noise and grazing statistics, flare at true depth jumps only, duplicate-skin misregistration, outliers, determinism, thread independence, model JSON validation, PLY round trip, NN spacing statistic |
| `test_prod_synth_cases.py` | tier sizes and subset relation; amendment-5 features requested for DEV and QUAL plans; stage coverage independent of the recipe stage; hash-bound case files; recipe rebuilt from the draft; registration truth; determinism across processes and `PYTHONHASHSEED`; QUAL guards; resume identity; part sets; blindness; import independence from E1/E3 |

Acceptance runs on 2026-10-17 are in §9: `generate --tier regression|full|runtime --seedset DEV`,
with and without the model card. All exited 0 with no coverage gaps.
