# CAD roll solver: E1 today, prototype, validation, production proposal (2026-10-25)

> **Ledger.**
> - **Data.** DEV data only:
>   - the 23 DEV_CONSUMED STEP files in `zeroshot_2026-10-25/INVENTORY.json`;
>   - synthetic STEP files built in-run.
>
>   No scans, gold, FRESH, TEST-B, DEV-CHECK, sealed or forbidden data was opened. The two FRESH rolled STEP files appear below only as INVENTORY metadata.
> - **Code.** No production engine or contract code was edited, and nothing was installed or committed.
> - **Execution.** All geometry ran through `sherman_slot.sh`, one job at a time. Peak RSS was 462 MB.
> - **Kind of evidence.** These numbers show engineering capability on consumed DEV CAD. They are not accuracy evidence.

Code: `research/cad_roll_solver/`
- `solver.py`
- `run.py`
- `render.py`
- `synth.py`
- `validate_synth.py`
- `screen_report.py`
- `SCHEMA.md`

Tests: `tests/test_prod_cad_roll_solver.py`, 21 passed.

## 1. What E1 does today with rolls

**How E1 reads CAD (`domains/production/engine/compile/brep.py`).**
- It reads the STEP with OCP (OpenCascade).
- It takes the dominant solid and unifies same-domain faces.
- It recognises B-spline planes and cylinders canonically, then records per face the family, the analytic frame, the convexity and the angular span.

**How E1 finds bends (`flange_graph.py`).**
1. **Candidates.** A bend candidate is a cylinder that is not closed and meets a **plane** smoothly along a generator. The edge must be a line parallel to the axis, with a dihedral ≤ 1°.
2. **Grouping.** Candidates are grouped coaxially (0.25°, 0.02 mm).
3. **Pairing.** An inner (concave) and an outer (convex) skin are paired when they share a tangent end, overlap axially, and differ in radius by 0.05–30 mm.
4. **Panels.** The planes at each angular end become one panel: both skins, grown through coplanar splits.
5. **Graph bend.** A graph bend is such a pair tangent to exactly two panels. Its values:
   - **sweep:** from the end radials, with the reflex case resolved by the cylinder span;
   - **Ri and Ro;**
   - **axial extent;**
   - **bend line:** the OML virtual sharp, from `oml_sharp()`.
   - The flange graph is the tree of panels and bends; the largest admissible panel is the datum.

**How the frozen UNSUPPORTED rule (`unsupported.py`, rule 2) classes curved features:**
- **ROLLED (a):** a graph bend with `Ri > 10 t`.
- **ROLLED (b):** a cluster of cylinders with one tangent end or no partner skin.
- **HEM:** sweep ≥ 170°.
- **OTHER:** the remaining cases.

**What happens to large-radius cylinders.** There are three cases, all seen on DEV:

| Case | What E1 does | DEV example |
|---|---|---|
| Roll with **planar run-outs on both ends** | It becomes a graph bend, ROLLED (a), `supported=False`. All its characteristics, and the flange sides it bounds, are UNKNOWN(UNSUPPORTED_FEATURE). Its "bend line" is the OML virtual sharp of the two flange planes, far outside the material for a large R. | none on DEV. Synthetic B, C and K reproduce it. |
| Roll **tangent to a bend or free at its ends** (no planar flange) | It is **never a candidate**: candidates need a cylinder–plane tangency. Its shells fall through every pass and end up with face role **`wall`**, so E1 treats the roll shell as a cut-edge or thickness face. The small bend joining a flange to the roll has only one plane end, so it becomes a ROLLED (b) feature ("bend cylinders meet 1 tangent ends"). | 28870000_A: shells R30 and R31.5 are `wall`. J1 is ROLLED (b). |
| **No planar panel joined by a bend at all** | The graph has 0 panels and the part model raises `DATUM_PANEL_UNDETERMINED`. Under E1-DEV-POLICY-2 the candidates on rule-declared ROLLED features are dropped first, so the result is NO_RECIPE `NO_CANDIDATE_BENDS:ALL_UNSUPPORTED_ROLLED`. | 47479000_C. Per INVENTORY metadata only, also 38898000 and 46465000 (both FRESH, not opened). |

**Unchanged E1 rerun through the slot** (`compile_some.py`, `e1_unchanged_compile_4files.json`). Every result is identical to the stored records in `zeroshot_2026-10-25/baseline/compile_some.log`, including the recipe sha256 values.

| File | E1 DEV compile | Bends | Notes |
|---|---|---|---|
| 28870000_A | COMPILED | 1 (G1, Ri 0.5, 25°) | Roll R30/31.5 → `wall`. Junction bend → ROLLED (b) feature. 2 flange sides are UNSUPPORTED_FEATURE (ENGINE_COMPILE.md). The owner's second bend, the flange-to-roll junction, is missing. |
| 47479000_C | NO_RECIPE (NO_CANDIDATE_BENDS) | 0 | Its flange graph gives DATUM_PANEL_UNDETERMINED. The half-shell R74/75 is `wall`. |
| 11979003_C | COMPILED | 7 | No rolls. Has a base window (cycle break). |
| 23553000_B | COMPILED | 16 | No rolls. The one curved face with R/t > 10 is a hole wall, R 36.25 with an axial extent of t. |

The full 23-file E1 rerun (`e1_unchanged_compile_some.json`) gives 22 COMPILED and 1 NO_RECIPE (47479000). It is the same as the stored log.

## 2. The prototype solver

It is built like E1's bend compile and reuses E1's reader and flange graph read-only. It runs in 0.1–2.8 s per part.

1. **Curved surfaces.**
   - Every cylinder face, and every analytic **cone** face, which E1 reads as kind `other` and the solver reads from the OCC adaptor.
   - Each is described as `r(z) = r0 + k z` about a canonical axis.
2. **Skin pairing.**
   - Coaxial grouping with E1's tolerances, plus 1e-5·R for very large R.
   - Radius classes within 0.01 mm, split per convexity.
   - A concave class and a convex class are paired when:
     - the normal offset is 0.05–30 mm, choosing the offset closest to E1's t;
     - they overlap angularly;
     - they overlap axially by more than 1.5 t, which rejects hole walls.
   - The connected components of the paired faces are the features.
3. **Rule ROLL-RULE-0** (SCHEMA.md):
   - An E1 graph bend keeps E1's class, unless E1 itself said ROLLED (a).
   - A paired feature with `Ri > 10 t` is a **roll**. The `10 t` is E1's frozen `ROLLED_RI_OVER_T`, reused and not tuned.
   - A feature with `Ri ≤ 10 t` that is tangent to a roll is a **junction bend**.
   - Everything else is left exactly as E1 made it.

   **Why this rule.**
   - It moves only what E1 already declares out of press-brake scope (ROLLED), plus curved skins E1 never saw.
   - Every bend E1 inspects today stays a bend by construction.
   - The open question is whether 10 t is the right physical boundary. E1 froze it before evaluation. The scan lane uses `max(12 mm, 5 t)`. See SCHEMA.md, "Alignment".
4. **Per-roll values:**
   - axis point and direction, and the frame;
   - Ri, Ro and t;
   - sweep (union of the skins' angular cover);
   - axial length;
   - chord and opening;
   - taper and cone flag;
   - convexity, as the bulge direction;
   - **run-out lines** for both ends and both skins, from the actual B-rep generator edges at the angular extremes. For a cone the generator is the slant line. Each line carries:
     - neighbour class: FLANGE, BEND, ROLL, FREE_EDGE or EDGE_ROUND;
     - neighbour ids: E1 panel `P#`, solver flange `F#`, roll `R#` or junction `J#`;
     - coverage;
   - the nominal arcs at 3 stations.
5. **Graph.** E1's panels and bends, with ROLL nodes, RUNOUT edges and JUNCTION_BEND edges added. An E1 ROLLED (a) bend edge is replaced by its roll node.
6. **Checks:**
   - skins coaxial;
   - t consistent with an independent part thickness;
   - skin sweeps agree;
   - axial overlap;
   - cone semi-angles agree;
   - run-outs tangent within 0.02 mm;
   - flange-plane dihedral vs sweep within 1°.

## 3. Validation

**Synthetic STEP: exact recovery.**
- **Cases.** 12 cases (`synth.py`). Each is built in OCP, written to STEP and read back. Truth comes from the construction parameters only.
- **Result.** All 12 pass (`synth/validate_synth.json`).
- **Error.** The worst error over axis, Ri, Ro, t, length, chord and run-out positions is **5.2e-7 mm** and **1.5e-6°**. The cone taper error is < 1e-6 mm/m.

| Case | Content | Result |
|---|---|---|
| A | Free-ended shell, R40, 120° | 1 roll, 2 FREE_END |
| B | Roll R50, 90°, with flat run-outs | E1 ROLLED (a) → roll. Run-outs on E1 panels P0/P1. Tangency and dihedral checks PASS. 0 bends. |
| C | 90° bend (R2) + roll (R60, 70°) | 1 roll; solver bends = E1's supported bend |
| D | Reverse junction R0.5 –27° + roll R30, 154° (the 28870000 topology) | 1 roll + junction J1; graph coherent |
| E | Same-turn junction + roll + flange | 1 roll, J1, solver flange F2 |
| F | **Cone** shell, semi-angle 10°, 140° | `is_cone`, taper 176.327 mm/m, run-outs along the slant |
| G | Slit tube, 350° | 2 FREE_END |
| H | Closed tube, Boolean | `closed`, 360°, no run-outs |
| I, J | Ordinary bends; hem | 0 rolls; bends = E1 |
| K | Bump, R300 / 8° | Roll (300 > 10 t) |
| L | **Ri = 10 t exactly** | Bend (boundary) |

**DEV screen** of all 23 DEV_CONSUMED STEP files (`screen_report.json`, `out/*.cad_roll_nominal.json`):
- **Rolls in 2 files:** 28870000_A and 47479000_C.
- **Other curved faces with R/t > 10:** 4 faces in 3 files (23553000, 48811000, 49204020). All are hole or cut-out walls with axial extent = t. The solver correctly does not make them rolls.

**Regression** (21 files with ordinary bends, 126 E1 bends):

| Result | Value |
|---|---|
| Solver bend list equals E1's flange-graph bend ids | 21/21 |
| Solver bend list equals the bends of the unchanged E1 compiled part | 21/21 |
| Rolls / junction bends reported | 0 / 0 |
| E1 bends whose skins the solver paired independently | 126/126 |
| Ri, t and sweep difference to E1 | 0.0 |

**Per rolled DEV part:**

| Part / feature | Ri / Ro / t (mm) | Sweep | Axial length | Chord (inner) | Run-outs | Checks | Visual |
|---|---|---|---|---|---|---|---|
| 28870000 **R1** (roll) | 30.000 / 31.500 / 1.500 | 154.076° | 76.5 mm | 58.471 mm | **start**: TANGENT → J1. Line at angle 150° in xy, z −76.5…−7.5 (coverage 0.90: above z −7.5 the outline is stepped). **end**: FREE_END; r0.5 edge rounds, then the cut face. | all PASS (t vs E1 1.5) | **OK** |
| 28870000 **J1** (junction bend) | 0.5 / 2.0 / 1.5 | 27.000° | 69.0 mm | n/a | R1 ↔ E1 panel P0 | all PASS | **OK** |
| 28870000 E1 **G1** (unchanged) | 0.5 / 2.0 / 1.5 | 25.000° | 69.0 mm | n/a | P0 ↔ P1 | E1 | **OK** |
| 47479000 **R1** (roll) | 74.000 / 75.000 / 1.000 | 178.000° | 414.0 mm | 147.977 mm | both FREE_EDGE (cut ends), coverage 1.0 | PASS; t and tangency NOT_APPLICABLE: E1 has no thickness and no smooth run-out | **OK** |

The coherent recipe for 28870000 is `P0 -G1- P1`, `R1 -J1- P0`: one component, which matches the owner's "2 bends" plus the roll.

**Mesh residual.** The solved radii are confirmed independently against E1's CAD tessellation: the vertices of the roll faces lie within ≤ 1.1e-4 mm of Ri and Ro, over the full axial range.

**Visual check.**
- **Method.** Every PNG was opened and inspected. The CAD mesh is drawn with:
  - the roll axis (magenta dash-dot);
  - run-out lines in bend-line style (orange; outer skin solid, inner skin dashed);
  - the nominal arcs at 25, 50 and 75 % (cyan);
  - E1 bend lines (red);
  - junction tangent lines (green).
  Overlays hidden by material are dotted, using a z-buffer.
- **End-on views.** The arcs trace the section boundary on both skins. The run-out markers sit exactly where the arc meets the flange or the cut end, and E1's G1 marker sits on the outer corner of the two flanges.
- **Method fixes made during the check.** None changed any value.
  - The renderer drew E1's ROLLED (a) "bend line" far outside the part. It is now drawn as the roll it became.
  - A cone's axis marker was clipped.
  - A grazing-angle visibility test marked visible lines as hidden.
  - The cone generator test had treated slant run-outs as missing; that was fixed in the solver.

## 4. Production path proposal (not implemented)

### 4.1 Contract amendment ENGINE_IO-A3 (additive, within `_v1`, review-only)

ENGINE_IO §7.1 allows a change within `_v1` when every document valid before it stays valid. A3 adds only optional fields and new enum members.

**`compiled_part.py`:**
- **New field on `CompiledPartV1`.** `rolls: tuple[CompiledRollV1, ...] = ()`. A `CompiledRollV1` carries:
  - `roll_uid`, `graph_roll_id`;
  - `axis: AxisV1`, `ri`, `ro`, `t`, `sweep_deg`, `axial_length_mm`, `chord_mm`, `taper_mm_per_m`, `is_cone`, `closed`;
  - `loci: RollLociV1`, with the inner and outer start/end run-out lines (the `LineLocusV1` counterparts of `inner_tangent_parent` etc.) and the axis segment;
  - `neighbours`: panel ids, bend uids and roll uids at each end, with `end_type`;
  - `supported`, and the `unsupported_feature` kind as for bends.

  `GraphBendV1` stays as it is. An E1 ROLLED (a) graph bend is listed once as a roll, never as a bend.
- **New characteristic kinds** in `characteristics.py`, each tied to a `roll_uid`:

  | Kind | Unit |
  |---|---|
  | `ROLL_RADIUS` | mm, inner skin. `side` INNER/OUTER optional. |
  | `ROLL_SWEEP` | deg |
  | `ROLL_OPENING` | mm, chord between the run-outs |
  | `ROLL_RUNOUT_POSITION` | mm. Side A = start, B = end. `expected_source` NOMINAL_LOCUS_REFERENCE, expected 0, the same convention as BEND_LINE_POSITION. |
  | `ROLL_TAPER` | mm/m, cones only |

  Ids look like `<roll_uid>/ROLL_RADIUS`. The owner asked for "a ROLL characteristic". A single `ROLL` kind with a measure sub-field is the alternative, but it needs a new identity field. Separate kinds reuse the existing identity fields.
- **Review-only clause** (the core of A3). A ROLL_* row may be MEASURED with value and U, but its decision is **REVIEW**, or UNKNOWN when not measured. It is **never PASS or FAIL** in A3. The consistency validator rejects PASS or FAIL on a ROLL_* row.
  - Rolls cannot make a summary green, so the zero-false-PASS gate holds by construction.
  - An out-of-band roll value shows up as REVIEW with its exceedance ratio.

**`part_recipe.py`:**
- `rolls: tuple[RecipeRollV1, ...] = ()`. Each entry carries:
  - `roll_uid`, `label`, `cad_feature_id` (from `cad_roll_nominal`), `source: cad_roll_candidate | manual`;
  - `tolerances: RollTolerances(radius_mm, sweep_deg, opening_mm, runout_mm, taper)`;
  - `requirement: OPTIONAL` (default), `REQUIRED` only after A4.
- `DefaultTolerances` gains optional `roll_*` entries.

**`consistency.py` / `check_compiled_against_recipe`:**
- `ROLL_INVENTORY_MISMATCH`: result roll rows ≠ compiled rolls.
- `ROLL_MARK_NOT_IN_CAD`: a recipe roll uid has no compiled roll.
- `ROLL_DECISION_NOT_REVIEW_ONLY`: a ROLL_* row with PASS or FAIL.
- The existing identity, expected and tolerance checks extend to roll rows unchanged.

**`layout.py`.** No change. Roll data travels inside `compiled_part.json` and `result.json`.

### 4.2 E1 changes (compiler `e1-1.2.0`, new `COMPILER_REVISION`)

1. **Port the solver.** Port it into `compile/rolls.py` after `build_flange_graph`. It is pure numpy on `BrepModel` and `FlangeGraph`, as here.
2. **Change the UNSUPPORTED rule** (rule 3, a new `UNSUPPORTED_RULE_VERSION` and a ledger note):
   - A ROLLED (a) bend, or a ROLLED (b) cluster, that the roll rule recognises becomes a **roll feature** instead of an unsupported bend.
   - A flange side whose far boundary is a roll run-out keeps `UNSUPPORTED_FEATURE` in A3. Measuring a flange to a run-out tangent line is A4.
   - Junction bends become graph bends between a panel and a roll node. This needs the tree builder to accept roll nodes.
3. **Stage models.**
   - In A3, rolls are **formed in every stage** (rolling precedes the press brake). A marked roll formed mid-sequence is out of scope.
   - Roll skin samples get their own `segment_kind`, so they never enter a plane fit. Today they are labelled as walls.
4. **Datum.**
   - A part with no planar panel (47479000) stays `DATUM_PANEL_UNDETERMINED` in A3.
   - A cylindrical datum (axis + run-out lines + holes) is a later amendment. Such parts get the roll nominals in diagnostics only, for the zs/roll display lane.
5. **Gate.**
   - Every frozen A1 and DEV-CHECK recipe, and every compiling corpus STEP **without rolls**, must recompile **hash-identical** (`rolls=()` is omitted from the canonical JSON when empty, or the schema reader defaults it).
   - Parts with rolls change, and only by the added `rolls` and roll characteristics.

### 4.3 What E3 needs (the scan side, from the roll-measure agent)

1. **Input.** E3 registers on the datum panel as today. It then maps each `CompiledRollV1` axis and run-out loci into the scan frame through its own pose T.
2. **Measurement.** It runs `research/zeroshot/roll_measure` **seeded by the CAD axis**: it slices perpendicular to the mapped axis instead of doing a Hough search, and keeps only the region whose axis lies within a few degrees and mm of the nominal.
   - **Measured values:** R_in, R_out, sweep, chord, run-out line positions and taper.
   - **U:** from the station bootstrap, plus a registration term for the position-type rows (ROLL_RUNOUT_POSITION) only.
3. **Withholding rules,** reused from today:
   - if the pose is untrusted or ambiguous, positions are withheld (UNKNOWN(REGISTRATION_UNTRUSTED) / AMBIGUOUS_POSE);
   - if coverage is low (`coverage.arc_fraction` and `length_fraction` below policy), the row is UNKNOWN(LOW_DENSITY / OCCLUDED).

   R, sweep, chord and taper are invariant, so they can be reported even under an in-plane degenerate pose. Run-out positions cannot.
4. **No PASS or FAIL.** The decision is always REVIEW under A3.
5. **Display.** Roll axis, run-out lines and the measured vs nominal arcs, in the same overlay style as these PNGs.

### 4.4 Tolerance sources

1. **Drawing first.** The wizard reads roll callouts into `RecipeRollV1.tolerances` with `source: DRAWING`:
   - the radius (`R30 ±0.5`);
   - the opening and chord dimension;
   - the run-out position, as a flange or edge dimension;
   - any profile tolerance.
2. **Shop default otherwise** (`source: SHOP_STANDARD`). These are placeholders for the owner to decide, flagged as a DEV assumption:
   - radius ±max(0.5 mm, 1 % R);
   - sweep ±1°;
   - opening ±1.0 mm;
   - run-out ±1.0 mm;
   - taper ±2 mm/m.
3. **Optional ISO 2768-m** for untoleranced linear dimensions, when the drawing's title block cites it.
4. **No tolerance, no measurement.** A roll with no tolerance stays UNKNOWN(NO_TOLERANCE). It never measures against an invented band.

### 4.5 Test and gate plan

1. **Unit.** `tests/test_prod_cad_roll_solver.py`, already written (21 tests): exact synthetic recovery; the rule boundary at Ri = 10 t; E1 ROLLED (a) → roll; E1 bends preserved; junction bend and graph coherence; cone; closed tube.
2. **Contract.**
   - Schema round-trip for `CompiledRollV1` and `RecipeRollV1`.
   - `test_prod_contract_parity.py` for the new kinds and codes.
   - Consistency codes (ROLL_INVENTORY_MISMATCH, ROLL_DECISION_NOT_REVIEW_ONLY).
   - Old documents validate unchanged.
3. **E1 hash gate.** Recompile every frozen recipe and every corpus STEP. Parts without rolls must be byte-identical. On the DEV screen that is 21/23 files.
4. **E2 synthetic.**
   - Add a rolled family: rolls with flat run-outs, junctions, free ends and cones.
   - Inject errors: radius ±0.2–2 mm, sweep ±0.5–3°, opening, and run-out shift.
   - Gate: recovery bias < tol/10, U coverage 90–98 %, and **0 PASS or FAIL on roll rows**.
5. **DEV physical** (consumed; capability only). Run on the DEV scans of 28870000 and 47479000 with CAD-seeded roll_measure. Report the CAD-vs-scan R, sweep and chord against the operator's caliper or radius-gauge readings.
6. **Promotion beyond review-only** (a later amendment, A4). This needs:
   - a frozen protocol on **unseen** rolled parts;
   - a gauge study (radius gauge or CMM) on ≥ 5 rolled parts;
   - U ≤ tol/4;
   - zero false PASS at the frozen decision rule.

   No consumed DEV, FRESH or TEST-B part can serve as held-out evidence. Note that the FRESH rolled STEP files (38898000, 46465000) are consumed.

## 5. Artifacts (`docs/production/runlog/cad_roll_solver/`)

**Visual-check PNGs.** Each is 1200×900 and between 54 KB and 152 KB.
- DEV:
  - `28870000__28870000_A_iso.png`, `28870000__28870000_A_R1_end.png`
  - `47479000__47479000_C_iso.png`, `47479000__47479000_C_R1_end.png`
- Synthetic illustrations, in `synth/`:
  - `B_roll_flat_runouts_{iso,R1_end}.png`
  - `C_bend_and_roll_{iso,R1_end}.png`
  - `D_junction_reverse_{iso,R1_end}.png`
  - `F_cone_shell_{iso,R1_end}.png`

**Solver output.**
- `out/<name>.cad_roll_nominal.json`, one per DEV STEP (23).
- `out/solver_run_summary.json`.

**Reports.**
- `screen_report.json`: screen and regression.
- `synth/validate_synth.json` and `synth/truth.json`.
- `e1_unchanged_compile_4files.json` and `e1_unchanged_compile_some.json`: the unchanged E1 reruns.

**Reproduce** (one slot):
```
S=sherman_slot.sh
$S .venv/bin/python -B -m research.cad_roll_solver.run --out OUT --list NAMES.json
$S .venv/bin/python -B -m research.cad_roll_solver.synth --out SYN
$S .venv/bin/python -B -m research.cad_roll_solver.validate_synth --dir SYN
$S .venv/bin/python -B -m research.cad_roll_solver.render --pkl OUT/*.render.pkl --out PNG
$S .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_prod_cad_roll_solver.py
```
