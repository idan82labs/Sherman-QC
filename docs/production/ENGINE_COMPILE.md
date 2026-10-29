# E1 onboarding compile (`domains/production/engine/compile`)

Status: implemented 2026-10-17 (overnight run, chunk E1), revised the same night after an
independent review (§9, items 9–20), and extended in the evening re-dispatch (§11: Method B,
the per-bend developed-length QA, wizard and robustness fixes). Plan of record:
`docs/production/PRODUCTION_LINE_PLAN.md` §3.1 and §10 amendments 1, 2, 4d, 5 and 9.
Binding I/O: `contracts/production/ENGINE_IO.md` §2–3 and the models in
`contracts/production/compiled_part.py` (the models win on any disagreement).

The compile runs once per recipe release, offline. It turns three inputs into a
`CompiledPartV1`:

- the folded STEP;
- the operator-confirmed recipe;
- the optional flat.

The `CompiledPartV1` holds:

- the flange graph;
- the surface loci;
- the nominals with resolved tolerances;
- one stage model per sequence prefix (sample cloud, display mesh, symmetry test);
- the reconciliation issues the wizard shows.

The per-scan engine (E3) never reads the STEP again.

## 1. Entry points

```
python -m domains.production.engine.compile --recipe RECIPE.json --cad PART.step \
    [--flat FLAT.dxf|FLAT.step] --out DIR [--shop-k 0.33] [--config CONFIG.json]
```

- Stdout carries only `PROGRESS\t{...}`, `RESULT\t{...}` or `ERROR\t{...}` records.
  Native OCC output is redirected to stderr at the file-descriptor level.
- Exit codes:
  - 0: COMPILED.
  - 2: BLOCKED (`compiled_part.json` is written, with its blockers), or invalid input
    (`compile_error.json {code, message}`).
  - 3: internal error, including the soft time budget (`COMPILE_TIMEOUT`; nothing is
    guaranteed).
- `compile_error.json` codes: `RECIPE_INVALID`, `CAD_MISSING`, `CAD_HASH_MISMATCH`,
  `FLAT_MISSING`, `FLAT_HASH_MISMATCH`, `FLAT_UNEXPECTED`, `CONFIG_INVALID`,
  `SHOP_K_INVALID`. For `OUT_DIR_NOT_EMPTY` only the `ERROR` record is printed, and nothing
  is written into a directory that is not empty.
- A STEP whose bytes match but that OCC cannot turn into one solid is **not** invalid
  input. It is a BLOCKED compile with `CAD_READ_FAILED`, so the wizard can show it.
- Geometry the code does not handle is not a crash either. An unexpected exception inside a
  compile phase becomes a BLOCKER that names the phase and the exception
  (`FLANGE_GRAPH_NOT_A_TREE` up to reconciliation, `FLAT_PARSE_FAILED` in the flat phase,
  `STAGE_MODEL_FAILED` afterwards). The traceback goes to `compile.log`, and
  `compiled_part.json` is still written. Exit 3 is left for the time budget and for a
  failure while assembling the part itself.
- Output files:
  - `compiled_part.json`: canonical JSON, written last (tmp + rename);
  - `stages/S<k>.npz` and `stages/S<k>.ply`;
  - `compile.log`.
- The effective configuration (`config.py`, unknown keys refused) is hashed into
  `compiler.config_sha256`. The compiler version is `e1-1.1.0` (1.0.0 before the review fixes).
  The evening revision is recorded as `e1-1.1.0-r2` in `diagnostics.compiler_revision`, which is
  outside `compiled_part_sha256`. It changes the output only of compiles with a bound flat, so the
  version, and every pre-A1 freeze hash, is kept (§11).

Library (`from domains.production.engine.compile import ...`):

| Function | Use |
|---|---|
| `compile_part(recipe, cad_path, out_dir, flat_path=None, shop_k=0.33, config=None, progress=None)` | The CLI's behaviour. Returns `CompileResult(compiled, out_dir, files)`. Raises `CompileInputError(code, message)` or `CompileTimeout`. |
| `analyze_cad(cad_path)` | Flange-graph summary for the wizard before marks exist. |
| `reconcile_draft(cad_path, marks)` | Preview of MARK_NOT_IN_FLANGE_GRAPH / UNMARKED_BEND / MARK_AMBIGUOUS for draft marks (`Mark(bend_uid, line_start, line_end)`). |
| `propose_flat_binding(cad_path, marks, flat_path, fmt=None, cfg=None, shop_k=0.33)` | Flat parse plus binding proposal: `bend_line_map`, `mirrored`, `confidence`, `needs_operator_review`, `fit_choice` (how the fit was chosen, `ambiguous`), K (`k_factor` inside the plausible band, `k_flat_fit`, `k_plausible_band`, `k_clamped`) and the developed-length check. Also `fold` (§5, Method B): whether a release with that map would fold the flat with its own per-bend strips, the strips, and the reason when not. Unmarked unsupported bends stay rigid and restrict the datum exactly as in the compile. For C05/C09 before release. |
| `parse_flat(path, fmt=None)` | Parsed flat: outer and inner loops, and bend lines `L<n>` with source, direction and angle. |
| `stage_transforms_from_compiled(compiled, formed_bend_uids)` | Panel transforms of **any** formed set from a `CompiledPartV1` alone, for E3's out-of-order hypotheses. |
| `stage_points_from_compiled(compiled, points, normals, segment_kind, segment_index, formed_bend_uids)` | Maps the S_N sample cloud to any formed set: panels move rigidly, unformed bend regions are unrolled. It reproduces every prefix stage cloud (tested), so E3 can build out-of-order stage clouds without the CAD. |
| `dev_recipe.dev_candidates(cad)`, `dev_recipe.dev_recipe(cad, candidates, ...)` | DEV_AUTO_CONFIRMED recipes (§10). |

The package never imports OCP at import time.

## 2. Pipeline

1. **Inputs** (`compiler._verify_inputs`). The CAD sha256 must equal `recipe.cad.sha256`.
   A recipe with a flat needs `--flat` with the matching sha256, and a flat without a
   recipe flat is refused. The flat is parsed first, so a BLOCKED compile still reports
   the real `parse_status`.
2. **B-rep** (`brep.py`). OCP reads the STEP and the single solid is kept. A second
   solid holding more than 1 % of the volume gives `MULTIPLE_SOLID_BODIES`, reported as
   `CAD_READ_FAILED`. Coplanar and co-cylindrical face splits are merged by
   `ShapeUpgrade_UnifySameDomain` (the geometry is unchanged; the original faces are kept
   if unification fails).

   Per face the reader keeps:
   - the surface family;
   - the outward normal (OCC `du × dv`, negated for REVERSED faces);
   - the analytic plane or cylinder frame;
   - the boundary edge uses with B-rep vertex ids;
   - a tessellation.

   Per edge it keeps the dihedral between the two faces' outward normals (0° means a
   smooth tangent or coplanar junction).

   **Canonical recognition.** Some exporters write every plane, bend cylinder and straight
   edge as a B-spline. A free-form face that `GeomLib_IsPlanarSurface` proves planar
   (1e-4 mm) is a plane. One that `ShapeAnalysis_CanonicalRecognition` proves cylindrical
   within 1 µm is a cylinder: its axis, radius and convexity come from the recognised
   surface, its own (u, v) parametrisation is kept for sampling, and its angular span comes
   from its boundary (2π when it has a seam). A free-form edge whose samples lie within
   1e-4 mm (or the edge tolerance) of their chord is a line. Recognition changes only the
   family the flange graph sees, never the geometry. An all-B-spline L and hat compile to
   the same graph and nominals as the analytic ones (tested).
3. **Flange graph** (`flange_graph.py`). The analysis is topological.
   - A *bend skin* is a partial cylinder that meets a plane smoothly along a
     **generator**: a straight edge parallel to the axis, with outward normals within 1°.
   - Bend skins are grouped coaxially (0.25°, 0.02 mm). Concave (inner, `Ri`) and convex
     (outer, `Ro`) faces are paired when they share a tangent end and overlap axially.
   - The planes the inner and outer skins touch at the same angular end are the two
     skins of one **panel**.
   - A single cylinder without a partner joins the skin-connected cluster pass. Tangent to
     a *panel* skin it is a curl (ROLLED (b)) when its cluster holds only cylinders, or part
     of the formed feature around it (a boss's foot fillet is in its STEPPED_BOSS record).
     One tangent only to walls, such as an outline corner round or a slot end, is a wall.
   - **Sweep.** The end radials give the sweep only up to the reflex ambiguity (acos lies in
     [0°, 180°]). The cylinder faces' angular span decides it, so a teardrop hem reads
     200°, not 160°. The material's angular middle (the radial toward the centroid of the
     bend's cylinder faces) is kept for hems at and beyond 180°.
   - Collinear pieces of one bend between the same two panels (a bend interrupted by a
     relief) are one graph bend.
   - **Windows**: a panel cut into coplanar pieces by a full-width window makes a cycle.
     Each piece is joined to the same neighbours through collinear bend pieces. When the
     far panels are coplanar and still connected without those pieces, they are merged
     into one windowed panel, and the pieces into one bend (one press stroke). Collinear
     tabs connected only through their base stay separate bends. The windowed panel's
     `outer_boundary_mm` joins the pieces with a zero-width keyhole slit; that slit is
     never a free edge.
   - Faces that touch a skin smoothly but are neither panel nor bend skin are
     **features**, classified by the UNSUPPORTED rule (§4). Everything else is a
     **wall**.
   - Panel skin boundaries are chained into closed loops whose segments are labelled
     FREE, BEND (with the bend) or FEATURE.
   - Ids are deterministic. Panels are `P0..`, ordered by mid-thickness centroid. Graph
     bends are `G1..`, ordered by OML-sharp midpoint (the outer apex for a hem).
4. **Part model** (`part_model.py`).
   - **Datum**: every panel is rigid at every stage, so all qualify, and the largest
     (mean skin area) wins, with ties to the lowest id.

     When unmarked unsupported bends are kept rigid, the contract's tree rule applies: a
     compiled bend's parent chain must reach the datum through compiled bends only. The
     datum is then restricted to the component holding every recipe bend. If the recipe
     bends are split across rigid bends, the result is `FLANGE_GRAPH_NOT_A_TREE`, with a
     detail asking the operator to mark them.
   - **Tree**: built breadth-first from the datum. A cycle or a disconnected panel gives
     `FLANGE_GRAPH_NOT_A_TREE`.
   - **Skin 0** of the datum is the side whose outward normal has the larger +Z component
     (ties +Y, then +X). It continues across every bend on the same cylinder side.
   - `oml_skin` is the skin index of the bend's convex side. `direction` is UP when
     `oml_skin` = 1, else DOWN.
   - **Axis**: `unit(cross(−p_out, c_out))`. A +sweep right-hand rotation about it takes
     the child from flat to formed. At sweeps ≥ 179° (near-reverse hems, 180° hems, reflex
     teardrops) that cross product vanishes or flips, so the axis is oriented toward the
     material's angular middle and `p_out = −(axis × r_parent)`, `c_out = axis × r_child`,
     which agrees with the regular construction wherever both are defined.
   - **Hem loci.** For a hem (sweep ≥ 170°, always unsupported) the virtual sharps lie at
     `R / cos(sweep/2)` ≥ 11.5 R from the axis, or at infinity at 180°. The inner and outer
     apex lines are published in `iml_virtual_sharp` / `oml_virtual_sharp` instead, and the
     bend's feature note says so. No measurement depends on them: such a bend has no
     BEND_LINE_POSITION nominal, and a flange side ending on it is UNSUPPORTED_FEATURE.
5. **Reconciliation** (`reconcile.py`, ENGINE_IO §2.1).
   - **Feasibility**: the mark must be parallel within 2°. Its midpoint must lie within
     `Ri + t + 1 mm` of the axis, inside the axial extent (±1 mm) and inside the angular
     sector (±10°).
   - **Cost**: the distance to the mid-radius apex line plus 0.5 mm per 2°. An exact
     native mark costs about 0 mm.
   - **Ambiguity**: a mark or graph bend whose runner-up is within 1 mm of its best is
     MARK_AMBIGUOUS and is never auto-assigned. The remaining pairs are matched by the
     Hungarian algorithm. A mark that is feasible only for graph bends another mark already
     holds is a **duplicate**, also MARK_AMBIGUOUS (the graph bend "matches several marks"),
     whatever the cost gap; it is not MARK_NOT_IN_FLANGE_GRAPH.
   - An *unsupported* unmarked graph bend is not UNMARKED_BEND. It stays rigid (formed)
     in every stage and is recorded as a feature with the note
     `RIGID_FOLLOWER_OF:<carrier panel>`, which `stage_transforms_from_compiled` follows.
     That is wrong for a hem formed mid-sequence: the operator should mark it. A marked
     unsupported bend is staged like any recipe bend, with every characteristic
     UNKNOWN(UNSUPPORTED_FEATURE).
   - **Marked hems.** `CompiledBendV1.sweep_deg` must stay below 180°. A marked 180° hem
     (open or closed) is published *and staged* at 179.999°, its child's outward direction
     rotated to match. That is 0.001° off, about 0.3 µm over a 15 mm return flange, and it
     also gives the side-naming rule two distinct flange directions (a 180° hem's flanges
     point the same way). A marked reflex hem (teardrop, > 180°) cannot be represented: the
     compile is BLOCKED with `STAGE_MODEL_FAILED`, and the detail asks for the mark to be
     removed (the hem is then rigid).
6. **Flat** (`flat.py`, `flat_binding.py`, §5). This step yields K and the stage method:
   Method B (the bound flat folded with its own per-bend strips) when the flat determines
   every strip, else Method A.
7. **Nominals** (`nominals.py`, §3).
8. **Stage models** (`stages.py`, `sampling.py`, `symmetry.py`, §6–7). They are built
   only when no blocker exists before this step, so a BLOCKED part carries no stages.

   **Self-check**: at S_0 every skin point reachable through recipe bends must lie on the
   datum's skin planes (skin 0 at height 0, skin 1 at −t) within 0.05 mm; otherwise the
   result is `STAGE_MODEL_FAILED`. A deliberately wrong unfold sign trips it (see the
   tests). On the real corpus the worst value is 5e-8 mm; a marked 180° hem staged at
   179.999° leaves 3e-4 mm.
9. **Assembly**. `CompiledPartV1.finalize`. If the assembled part fails contract
   validation (an internal inconsistency), a `STAGE_MODEL_FAILED` blocker is recorded
   and a BLOCKED part without bends or stages is written instead. A second failure is an
   internal error (exit 3).

## 3. Nominal conventions

- **INCLUDED_ANGLE** = 180 − sweep, the flange-graph value.
  - It is authoritative over the recipe's nominal.
  - A disagreement above 0.5° adds the WARNING `ANGLE_CONVENTION_DISAGREES_WITH_RECIPE`
    (naming the bend). The wizard shows both values.
- **FLANGE_LENGTH** side S is measured on the OML.
  - There are 21 section stations over the bend's axial corridor. Each station line runs
    from the OML virtual sharp along S's outward direction, in S's OML plane.
  - The farthest crossing of S's skin boundary is the far boundary at that station. Its
    segment label classifies it:
    - FREE: the length is that distance;
    - BEND (another formed bend b′): ADJACENT_BEND, with the length measured to where the
      station line meets b′'s OML virtual sharp (OML sharp to OML sharp). Every station
      agrees when b′ is parallel. When it is not (an angled or trapezoidal panel), the
      values vary and their median is used;
    - FEATURE, or the tangent line of an **unsupported** bend (marked or rigid): no nominal
      (`UNSUPPORTED_FEATURE`). An unsupported bend's OML sharp is no measurable reference;
      the characteristic is listed by that bend's feature record.
  - The class found at ≥ **60 %** of the valid stations wins. A FREE_EDGE nominal is the
    dominant *level*: station values are grouped within 0.05 mm, and the level must hold
    ≥ 60 % of the stations. A continuously varying (tapered) edge uses the median.
  - When ≥ 30 % of the FREE stations end on a far-edge segment parallel to the bend
    (within 5°: the straight far edge, or a notch bottom), only those stations count. Corner
    rounds and chamfers at the corridor ends therefore do not move the nominal.
  - A boundary without a 60 % class, or a stepped edge without a dominant level, gets no
    nominal (`NOMINAL_UNAVAILABLE`, a WARNING). A near 50/50 split would make the median
    flip with station placement, for the nominal and for the scan alike, so the value is
    never guessed. A minority notch or tab never moves the nominal, and the spread is in
    diagnostics.
  - `FlangeSideV1.boundary_kind` requires a value (`BoundaryKind` has no UNDETERMINED).
    A side with a nominal names the nominal's boundary. A side without one names the class
    most of its stations end on: ADJACENT_BEND with the compiled bend, or FREE_EDGE. A
    plurality on an unmarked (rigid) bend, which has no `bend_uid`, reads FREE_EDGE. The
    characteristic nominal carries the truth (`boundary_kind` null and
    `unavailable_reason`).
  - **E3 note**: measure ADJACENT_BEND sides per station the same way and take the station
    median. Measure free edges on the same clean stations.
  - **E2 parity** (DEV seeds, `part-set` for HAT, CHANNEL_WITH_RETURNS, TWO_DIRECTION_TAB
    and PANEL_WITH_HOLES_NOTCHES): every nominal that both define agrees within 4.4e-7 mm
    (327 comparisons, flange, angle, position and backgauge). The exception is two sides of
    the notched panel where the stations split 50/50, which E1 withholds and E2's truth
    still values.
- **BEND_LINE_POSITION**: nominal 0 (`OML_SHARP_OFFSET_ALONG_SIDE_A`).
- **BACKGAUGE_LENGTH**, for k = 1..N−1, at the S_k pose.
  - It is measured from the OML sharp of bend k to the far boundary: a cut edge, or the
    OML sharp of an already formed bend.
  - It crosses the side panel and every panel and flat strip reachable through unformed
    bends.
  - The side is the one leading toward more unformed bends. Ties go to the longer
    developed run, then to side A.
  - The id is `<bend_uid>/BACKGAUGE_LENGTH/S<k>`.
  - **K sensitivity.** Unlike a final-pose flange length, a backgauge nominal moves with K
    through every flat strip it crosses (`sweep · t` per unit K). `diagnostics.backgauge_nominals`
    records per stage the crossed bends (`crossed_bend_uids`), `d_nominal_d_k_mm` (finite
    difference at K + 0.01; nominals are piecewise linear in K) and
    `k_sigma_mm = |dN/dK| · σK`. On the 4-bend hat S1 crosses three strips: 3π mm per unit
    K, 0.47 mm at σK = 0.05 against a ±0.3 mm DEV tolerance. E3 must carry this into U; the
    contract has no field for it yet (open issue for C01).
- **GENERAL_MEASUREMENT**: the recipe's `nominal_mm` (`CAD_STRAIGHT_LINE`).
- **Tolerances**: every nominal carries the tolerance and `recipe_requirement` of
  `compiled_part.recipe_policy_for`, with `origin` and `standard`. This includes
  NOT_INSPECTED flange sides.
- **Thickness**: a recipe `thickness_mm` that differs from the CAD skin separation by more
  than 0.05 mm + 2 % is the BLOCKER `THICKNESS_INCONSISTENT` (wrong material or wrong CAD).
  It used to be a diagnostics note the wizard never showed.

## 4. Per-feature UNSUPPORTED rule (frozen before evaluation)

Rule `E1-UNSUPPORTED-RULE-2 (2026-10-17; rule 1 thresholds, review fixes)`, in
`unsupported.py`. Rule 1 was written in code and in this document **before** the compile
was run on any real STEP. Its thresholds are not configurable and have not changed. Rule 2
(after the independent review, before A1) makes the implementation follow the rule text;
the changes and their corpus effect are in §9. Unsupported geometry is declared per
feature, never per part. The rules apply in order, and the first match wins.

| Kind | Rule |
|---|---|
| HEM | A graph bend with sweep ≥ 170° (included angle ≤ 10°): open hem (180°), closed hem (180°, tiny Ri) or teardrop (reflex sweep > 180°, read from the cylinder faces' angular span). |
| ROLLED | (a) A graph bend with inner radius `Ri > 10·t`: a rolled or bump-formed radius, not a press-brake bend. (b) A skin-connected cluster made only of cylinders tangent to a panel without a coaxial partner on the other skin, or without a tangent panel at their other end (curl, open roll). A lone cylinder inside a larger cluster takes that cluster's class. |
| SPLINE_WALL | A *skin* face (smooth-connected to a panel or bend skin) that is free-form (B-spline, Bezier, offset, extrusion or revolution) and not proven planar. Free-form *thickness walls* (cut outline faces) are supported. |
| STEPPED_BOSS | A skin-connected cluster with toroidal, conical or spherical faces, or with planar faces that belong to no panel: emboss, dimple, louver, extruded hole or stepped boss. |
| OTHER | Anything the graph cannot explain. This covers bend cylinders that do not meet exactly two tangent ends; bend/panel tangency error > 0.02 mm; a cylinder sweep ≠ panel dihedral by > 1°; `Ri < 0.01 mm`; and walls that are not thin (> 20 % of samples farther than t + 1 mm from every skin, rigid-bend or feature face; the 1 mm is the fixed sampling slack of the 0.5 mm contract grid, `THIN_WALL_ALLOWANCE_MM`). |

Effects:

- **Bend feature**: the bend becomes `supported = false`. FLANGE_LENGTH A/B,
  INCLUDED_ANGLE, BEND_LINE_POSITION and its BACKGAUGE_LENGTH carry
  `unavailable_reason = UNSUPPORTED_FEATURE`, and their tolerances are still carried. So
  does every flange or backgauge side of another bend whose far boundary is that bend.
- **Panel feature**: a flange side is affected only when the feature is its far boundary
  at the majority of stations. The panel stays supported unless the features' one-sided
  footprint (the projected area of the faces facing one side of the panel plane, the larger
  side) covers more than 50 % of the panel (planar skin plus footprint). Rule 1 compared the
  area of every feature face (both skins and side walls) with one skin's area, which
  declared a panel unsupported at about 25 % coverage.
- **Unsupported panel**: a wizard and display flag only. Its characteristics stay available,
  because E3 fits planes on the panel's remaining planar skin samples (feature faces are
  never panel samples), and feature far boundaries are already handled per side.
- **Labelling**: feature faces and the skins of unmarked unsupported bends are edge-wall
  points (`segment_kind` 2, `skin` 2). They never enter a skin plane fit.
- **Warnings**: one `UNSUPPORTED_FEATURE` WARNING per feature. E3 reports the affected
  characteristics as UNKNOWN(UNSUPPORTED_FEATURE).

## 5. Flat parse, binding, K and developed-length QA

- **DXF** (ezdxf 1.4.5).
  - Units come from `$INSUNITS`; inch, cm, m and others are scaled, and a unitless file
    is taken as mm and noted.
  - INSERTs are exploded.
  - **Bend lines** are straight lines found by:
    - layer name: BEND, FOLD, BIEG, KANT, BENDLINE, כיפוף or קיפול;
    - linetype (on the entity or by layer): DASH, CENTER, PHANTOM, HIDDEN, DOT, DIVIDE or
      BORDER families;
    - annotation: a TEXT or MTEXT such as `UP 90°` or `90° DN` within
      max(15 mm, 5 % of the size) of a straight solid line.

    Annotations give the direction and angle. Solid lines on outline layers are never
    bend lines.
  - The remaining curves are chained into closed loops. The largest is the outer loop;
    loops inside it are holes.
  - Ids `L1..Ln` are ordered by midpoint. They are deterministic, so a released
    `bend_line_map` stays valid.
  - A short line centred on a hole (within 1.5 hole radii, at most 4 radii long) is the
    hole's centre mark, not a bend line. It is dropped and noted.
- **Flat STEP**: a planar solid. Its largest planar face and the parallel partner give
  the loops and the thickness. Bend lines are inferred by the binding (`I1..In`).
- **Development**: the compiled tree unfolded to S_0 for a trial K. It is represented as
  the cut boundary (panel FREE segments plus strip ends), sampled every 0.5 mm with the
  segment each sample lies on, together with each bend's strip centre line. Because it is
  generated from the panel tree, assigning flat lines to development lines is a
  graph-consistent match.
- **Fit**:
  - **Seeds and ICP**: a 2D rigid fit, optionally **mirrored**, from 8 seeds on the
    principal axes. Trimmed ICP pairs each flat sample with its closest point on the
    development *polyline*, and the score is the symmetric trimmed RMS of point-to-segment
    distances. Point-to-point pairs between two 0.5 mm samplings pull along the edges by up
    to a quarter spacing; the pull changes with K (segment lengths move the sample phase),
    which had biased the pose and K\* on parts with few bends (K\* = 0.357 for a single-bend
    L drawn with 0.33; now 0.330).
  - **Choosing among outline-equivalent fits**. A rectangular blank fits four ways equally
    well, and the old code took whichever tied RMS sorted first, which changed with a mere
    translation of the drawing (7 of 12 placements gave an empty map). The fits within
    max(0.05 mm, best RMS) of the best are now chosen between by the drawing's evidence:
    1. with a released map (compile check): agreement with the released mirror flag, the
       most released lines within tolerance, then their mean offset;
    2. with drawn lines: the most lines assigned, then the smallest line residual (ties
       within 0.05 mm), then the most UP/DOWN and angle annotations that agree with the
       compiled direction and sweep (UP is toward the viewer: skin 0 on an unmirrored fit,
       skin 1 on a mirrored one);
    3. then unmirrored before mirrored, then the smaller rotation angle.

    When candidates with different maps survive every step, `fit_choice.ambiguous` is set,
    the confidence is halved and review is required. This is the honest outcome for a part
    with a real symmetry (a point-symmetric Z drawn without annotations) and for any flat
    STEP of a rectangular blank, which cannot tell its ends apart.
  - **K search**: on [0.05, 0.75], a 0.02 grid then golden-section refinement, over the
    whole outline-equivalent family. The objective is the outline RMS plus a line term: the
    RMS over *every* drawn line of its offset, or of 1.5 mm when it is unassigned. The old
    objective ignored unassigned lines, so a wrong fit could score better by assigning none.
  - **Plausible K band**: shop K ± 3 × `shop_k_uncertainty`, by default [0.18, 0.48]. A
    uniform length error in a flat is indistinguishable from a different K. Before the
    review, K\* was fitted over the whole search range and the QA was evaluated at that same
    K\*, so the error was absorbed: a single-bend L drawn 1 mm too long passed with
    K = 0.67. K\* is now used only inside the band. Outside it, the band edge is used, its
    uncertainty covers the clamp, and the proposal is flagged (`k_clamped`, review).
  - **Line assignment**: Hungarian, with offset ≤ 1.5 mm and angle ≤ 3°.
- **Proposal**:
  - Contents: `bend_line_map`, `mirrored`, `confidence` ∈ [0, 1],
    `needs_operator_review`, `fit_choice` (`equivalent_fits`, `narrowed_by`, `decided_by`,
    `ambiguous`, `mirror_determined`, `alternative_maps`), `k_factor` (the K a binding
    would use), `k_flat_fit` (K\*), `k_plausible_band`, `k_clamped`, `line_offsets_mm` (at
    K\*) and `qa_line_offsets_mm` (at the K used).
  - Review is required for an incomplete map, an ambiguous fit, inferred lines, a clamped
    K, a failed QA or confidence < 0.9.
  - Release still requires operator confirmation in the wizard (plan §3.1).
- **Compile**:
  - **Binding check**: with an `AUTO_PROPOSED` or `OPERATOR_CONFIRMED` binding, every map
    entry is checked against the fit at K\*. An unknown id, an offset over tolerance or a
    mirror flag that disagrees gives `FLAT_BINDING_INCONSISTENT`.
  - **Developed-length QA**: the largest of the bound lines' offsets and the outline
    extents' differences across every bend direction, at the K actually used (the nearest
    plausible K). It is the smallest disagreement any plausible K leaves, not the residual
    at a K fitted to absorb it. Above 0.5 mm the result is `FLAT_DEVELOPED_LENGTH_MISMATCH`
    (BLOCKER). A K\* at the search bound is also a mismatch.
  - **Per-bend QA** (evening revision, §11). When the Method B fold below *explains* the
    flat, its strips are known. The QA then compares the flat with the nearest plausible
    per-bend development (every strip clamped into the plausible K band) instead of a single
    K, and the single-K value stays in the detail. Otherwise the single-K check stands.
    - A wide but plausible bend table is no mismatch. A t = 4 hat drawn with K alternating
      0.20 / 0.47 fails every single K at 0.505 mm, yet it is exactly a plausible blank (0.0 mm).
    - One implausible strip is not hidden by being spread over every strip; a strip error
      moves its own drawn line by only half of it. A hat drawn with K = 0.66 at one bend
      passed the single-K check at 0.356 mm (K\* = 0.434). Its strip lies 0.565 mm beyond
      K = 0.48, so it is now `MISMATCH`.
    - A flat drawn uniformly at K = 0.60 is still `MISMATCH`: its clamped strips accumulate to
      1.508 mm.
  - **K from the flat** (`BOUND_FLAT`, uncertainty from the residuals and any clamp) only
    when the binding is consistent, the QA passes, and the binding is confirmed:
    `OPERATOR_CONFIRMED`, or `AUTO_PROPOSED` in a `DEV_AUTO_CONFIRMED` recipe (the DEV
    stand-in for operator confirmation). An unconfirmed `AUTO_PROPOSED` binding in an
    operator recipe is still checked, but K stays the shop default, and `FLAT_NOT_BOUND`
    says why. C05 should not release such a binding (open issue).
  - **Stage method: Method B, fold the bound flat** (`flat_binding.fold_flat`, policy
    `E1-FOLD-1`, evening revision §11). A shop flat carries the shop's own bend allowance per
    bend; a single K matches it only to within the QA threshold per strip. Once a binding is
    bound, every recipe bend's strip is measured from the flat itself.
    - **Model**: the S_0 development is linear in the strip lengths. A longer strip of bend *a*
      moves every panel, strip and drawn line beyond *a* along *a*'s in-plane strip direction,
      and *a*'s own drawn line by half as much.
    - **Solve**: Gauss-Newton over the strips and a small correction of the flat's 2D pose.
      The residuals are the normal residuals of every outline sample against the development
      segment of the panel that carries it, plus the bound drawn lines (strip centre lines)
      at weight 100. The strips follow the drawn lines; the outline fixes the pose and what
      the lines leave open.
    - **Used only when**:
      - every strip is observable: the strip block, with the pose projected out, has full
        rank (smallest singular value ≥ 1e-3). A flat STEP has no drawn lines, and a chain
        with only its outer lines bound cannot be split, so both stay Method A;
      - the fold converged;
      - the fold explains the flat to drafting precision: afterwards every bound line and
        every extent across a bend direction agree within 0.03 mm, and 99 % of the outline
        lies within 0.05 mm.
    - **Drafting errors**: a panel or line drawn off the CAD is never absorbed into the strips.
      A last flange drawn 0.05 or 0.1 mm long, or one line 0.1 mm off its strip centre, leaves
      a 0.05–0.19 mm extent residual after the fold. Such a flat stays Method A, with uniform
      strips within 0.03 mm of the truth. At line weight 10 the same errors had moved a strip
      by up to 0.16 mm.
    - **Implausible strips**: a strip whose equivalent K, `(L/sweep − Ri)/t`, lies outside
      the plausible band is the flat's defect, not the shop's process. Its stage strip is the
      band edge (the clamp is reported), every other strip is the flat's own, and one beyond
      the QA threshold blocks (above).
    - **Published**: the stages are `METHOD_B_FOLD_FLAT`, and each bend's `neutral_length_mm`
      is its stage strip. `k_factor` stays the bound uniform K (`k_factor_used`, the same K),
      and its `uncertainty` is what the fold leaves unexplained, floored at 0.005, not the
      uniform fit's residual.
    - **Reporting**: `diagnostics.flat.method_b` holds `used`, `explained`, `applied`, the
      reason, the before/after residuals and, per bend, `strip_mm`, `stage_strip_mm`,
      `delta_mm`, `k_equivalent` and `excess_mm`. The developed-length check's detail ends with
      the stage method and its reason.
    - **Results** on a hat drawn with the bend table K = 0.30/0.38/0.42/0.35: every strip is
      within 1e-3 mm (1e-6 in practice) in all placements tested (0°, 27°, 180° mirrored). The
      backgauge nominals follow the table within 1e-3 mm; the uniform K is off by up to
      0.2 mm per strip and more than 0.1 mm per backgauge.
  - **UNBOUND flat**: it is parsed and its proposal goes to diagnostics. K is then the
    shop default (`K_FACTOR_DEFAULTED`, `FLAT_NOT_BOUND`).
- **Default K**: `--shop-k` (0.33), with standard uncertainty 0.05.
- **Synthetic check** (tests; K tolerances are fixed, never the reported uncertainty):
  - An analytic DXF of a 4-bend hat drawn with K = 0.42 gives K\* = 0.420 ± 0.003 in every
    placement tested: 4 rotations × mirror × 2 offsets plus the original ones, always with
    the same, correct map. Drawn with 0.45 or 0.20 it gives 0.45 / 0.20. Drawn with 0.60
    it gives K\* = 0.60, clamped to 0.48, and the QA at 0.48 is 1.52 mm, `MISMATCH`.
    (Since r2 the reported value is the per-bend QA, 1.51 mm: §5 "Per-bend QA".)
  - A single-bend L, the weakest case, drawn 1.0 mm too long is `MISMATCH` (0.53 mm at the
    band edge) as a DXF, with or without the line moved, and as a flat STEP. Drawn 0.6 mm
    long as a flat STEP, 0.47 mm is absorbed up to the band edge: K = 0.48, QA 0.13 mm,
    `k_clamped`, review. That is the documented residual: up to `3σK · Σ sweep·t` of
    uniform error can pass as K.
  - A point-symmetric Z drawn with UP/DOWN annotations binds correctly whether or not it is
    turned half round, and reports the remaining mirror ambiguity. Without annotations it
    is reported ambiguous.
  - The existing cases still hold: a 1.2 mm error on one flange is `MISMATCH`, swapped
    lines are `FLAT_BINDING_INCONSISTENT`, and a flat STEP of a Z drawn with K = 0.38 gives
    K\* = 0.380 ± 0.005 from the outline alone (was 0.395), ambiguous and for review.

## 6. Stage models (Method A)

For every bend unformed at a stage:

`T(child) = T(parent) · Shift(−p_out · L_n) · Rot(axis, −sweep)`

where `L_n = sweep_rad · (Ri + K·t)` and the datum transform is the identity. K is the shop
K or a bound flat's K, applied uniformly (`METHOD_A_ROTATE_BACK`). Under Method B (§5) the
same composition uses each bend's strip measured from the bound flat as `L_n`
(`METHOD_B_FOLD_FLAT`), so the stage models are the flat folded at the formed bends.

- **Stage bend lines**: the OML sharp for formed bends (the outer apex for a hem). For
  unformed bends, the strip centre on the OML skin, `outer_tangent_parent + (L_n/2)(−p_out)`.
- **Unrolling**: an unformed region's samples and mesh vertices, skins and end caps, are
  unrolled onto the strip. A point at φ and ρ goes to
  `axis + z·a + ρ·r_parent + (φ/sweep)·L_n·(−p_out)`. The map is continuous with both
  panels. φ is measured in [0, sweep] also at and past 180° (a marked hem), where atan2
  alone would wrap the child half of the bend to negative angles.
- **Samples**: each face is sampled on a regular grid of its own parameters at
  h = 0.5 mm, analytically for planes and cylinders. h grows only above 1.6 M points per
  stage, and the actual spacing is recorded in `spacing_mm`.
- **Labels**:
  - panels: kind 0;
  - compiled bends: kind 1;
  - walls: kind 2, or (kind 1, skin 2) for a bend end cap, from the nearest skin;
  - features and rigid bends: kind 2 / skin 2 of their panel.
- **Files**:
  - The `.npz` is written deterministically: fixed zip timestamps, deflate level 1
    (about 10× smaller than stored), no pickles.
  - Points are rounded to 1e-5 mm before the float32 cast.
  - The same inputs give the same `compiled_part_sha256` (tested).

## 7. Symmetry test (amendment 5)

Method `SELF_REGISTRATION_PCA_ICP_V1`, run on each stage cloud voxel-decimated to 1 mm.

- **Candidates** (about the centroid):
  - ROTATION: 180° about each principal axis;
  - MIRROR: the reflection across each principal plane, plus the point inversion;
  - when two moments agree within 8 %: rotations by ±60°, ±90° and ±120° about the third
    axis (orders 3, 4 and 6; a rotation axis of order ≥ 3 forces two equal moments), and
    the 180° flips and reflections about in-plane directions every 15° (the in-plane
    principal axes are arbitrary there, and ICP refines the nearest one).

  Coverage: every symmetry of order 2, and orders 3, 4 and 6 about the axis of two equal
  moments (triangle, square and hexagon plates are tested). Rule 1's text claimed an exact
  symmetry is always a candidate; that was false for orders 3 and 6, which it did not
  generate. Other orders (5, 8, …) are still not generated. Such a part would report fewer
  symmetries than it has, which makes a stage look less ambiguous than it is; there is none
  in the authorized corpus.
- **Scoring**: each candidate is refined by point-to-plane ICP and scored by RMS. A
  point counts point-to-plane only where it lands on the surface (in-plane offset
  ≤ 1.5 spacings, agreeing normals), and point-to-point elsewhere. A map into a hole or
  beyond an edge is a miss. Distances are capped at 10 mm: a capped RMS above 0.3 mm is
  a lower bound, and one at or below it is exact. ICP abandons a candidate once fewer
  than half of its points reach the surface.
- `is_symmetric` iff some non-identity candidate has RMS ≤ 0.3 mm.
- `ambiguous_bend_groups` are the orbits under **ROTATION** candidates within the
  threshold. E3 publishes only proper rigid registrations, so a mirror-only symmetry
  never makes a bend identity ambiguous. A MIRROR candidate within the threshold means a
  mirrored part is indistinguishable from the CAD, so E3 must not claim
  MIRRORED_PART_SUSPECTED.
- Most extruded parts are mirror-symmetric across their mid-length. E3 should therefore
  test `ambiguous_bend_groups` and ROTATION candidates, not `is_symmetric` alone.
- **Scope limits** (open issues for E3/E4):
  - Each stage S_k is registered onto *itself*. Congruence *between* hypotheses is not
    tested. An equal-wall U with one wall formed (S_1) has no self-symmetry, yet it is
    congruent to the out-of-order hypothesis {B2}, so a scan of either is ambiguous between
    them. E3 must test competing formed sets itself (`stage_points_from_compiled` builds
    any of them).
  - The 0.3 mm RMS is a whole-cloud average. A difference confined to a small region passes
    it: a Z whose OML flanges are 31.5 and 30.0 mm reports its two bends as one group,
    although a 1.5 mm flange difference is measurable. AMBIGUOUS is then justified by the
    plan's definition while the nominals could still tell the bends apart.
- **Tests**:
  - An equal-wall U gives a ROTATION candidate with B1↔B2 and the group (B1, B2).
  - The asymmetric hat gives none (best ROTATION > 1 mm).
  - An off-centre hole breaks the U's symmetry.
  - A flat blank is symmetric with no permutation.

## 8. Engine bundle imports

The compile imports only:
- `contracts.production`;
- `domains.production.recipe` (DEV recipes);
- numpy, scipy, matplotlib (`Path.contains_points`), OCP and ezdxf.

These are pinned in `requirements.production.txt`.

The DEV candidate helper (`dev_recipe.native_candidates`) additionally imports, on demand:
- `scripts/extract_step_topology_bend_axes.py` (native);
- `scripts/extract_step_topology_bend_axes_v10.py`, with its
  `scripts/extract_step_topology_bend_axes_v9.py`, `_v7.py`, `_v2.py` and
  `backend/cad_finite_native_curve_graph_domain_v2.py` / `_v3.py` (V10).

The production compile path does not need them.

## 9. Changes made after the first real-STEP run (honesty log)

The UNSUPPORTED rule did not change. After the first authorized run, the following
changed:

1. **Bug fix**: single cylinders tangent only to walls (outline corner and slot rounds)
   had been reported as ROLLED. Rule ROLLED (b) requires tangency to a *panel*, so they
   are now walls. The first run had produced spurious features, and FEATURE far
   boundaries on some sides.
2. **Bug fix**: rigid-bend and feature skins are now references in the thin-wall test.
   An unmarked rolled bend's end caps had been reported as OTHER.
3. **Capability**: window cycles (§2, step 3). 11979003 has a full-width base window.
4. **Convention**: ADJACENT_BEND is now measured per station, including far bends that
   are not parallel (§3). It had been NOMINAL_UNAVAILABLE. This matches ENGINE_IO's
   "OML sharp to OML sharp, perpendicular to the bend, inside the corridor".
5. **Contract-forced**: the datum is restricted when bends are kept rigid (§2, step 4).
6. **Added**: the S_0 flatness self-check, the validation fallback, and deflate for
   `.npz`.
6b. **Convention robustness**, found by the E2 parity check: a boundary class or edge
   level now needs a 60 % station majority instead of 50 %, and corner rounds or chamfers
   at the corridor ends are excluded (clean parallel-edge stations). On the real corpus
   this withholds 7 nominals with a genuinely mixed or 50/50-stepped far boundary
   (40500000, 42288005). It also restores 23553000's corner-rounded flanges to their
   straight-edge value.
7. **Performance only**: the symmetry ICP queries are bounded (4 mm in ICP, 10 mm cap
   in scoring), and a candidate is abandoned below 50 % surface inliers. The symmetric
   stages found are identical on all 20 real parts before and after the change.
8. **DEV recipe candidate policy** (§10).

Changes after the independent review (same night, before A1). The compile was re-run on
the whole authorized corpus afterwards (§10): every nominal of every compiled part is
identical to the previous run (max |Δ| = 0), the unavailable count is unchanged (13), and
every symmetry group and ROTATION-symmetric stage is unchanged.

9. **Bug fix, hems at and past 180°** (major). An exact 180° hem has antiparallel end
   radials, so `unit(r1 + r2)` was a zero vector; `analyze_cad`, the compile and the CLI
   crashed (exit 3, no `compiled_part.json`). The bisector, orientation and outward
   directions now come from the material's angular middle at sweeps ≥ 179°, the apex
   lines stand in for the virtual sharps of every hem, a teardrop's reflex sweep is read
   from its cylinder span (HEM, not OTHER), and unrolling measures angles in [0, sweep].
   A marked 180° hem compiles (published and staged at 179.999°); a marked teardrop is a
   BLOCKER with a reason. Exact open, closed and teardrop hems are tested, marked and
   unmarked, including the CLI.
10. **Robustness**: any unexpected exception inside a compile phase is a BLOCKER that
    names the phase (§1), never exit 3.
11. **Bug fix, circular developed-length QA** (major). K\* is used only inside the
    plausible band, and the QA is evaluated at the K used (§5). Before, uniform errors of
    0.6–1 mm passed with K = 0.54–0.67.
12. **Bug fix, binding depended on the drawing's position** (major). The choice among
    outline-equivalent fits is now made by the drawing's own evidence, and unassigned
    lines count in the K objective (§5). Before, 7 of 12 placements of the hat DXF gave an
    empty map.
13. **Accuracy, flat fit**: closest-point ICP and point-to-segment scoring (§5). K\* for a
    single-bend L drawn with 0.33 went from 0.357 to 0.330.
14. **Honest labels**: stages are `METHOD_A_ROTATE_BACK` also with a bound flat (§5), and an
    unconfirmed `AUTO_PROPOSED` binding in an operator recipe gives no K.
15. **UNSUPPORTED rule 2** (§4, no threshold changed): fixed thin-wall allowance, one-sided
    feature footprint, lone cylinders clustered with the feature around them, teardrops as
    HEM, unsupported far bends. Corpus effect: 47266000_F's 20 ROLLED + 1 STEPPED_BOSS
    records are one STEPPED_BOSS and its panel P1 is supported again (the old area test
    over-counted both skins); 28870000_A and its duplicate family part Default have
    1 ROLLED + 1 STEPPED_BOSS instead of 3 ROLLED. No nominal changed.
16. **Side boundary names**: a side without a nominal names the class most of its stations
    end on (§3). On the corpus, 4 sides of 40500000 and 42288005 changed from FREE_EDGE to
    ADJACENT_BEND (their nominals stay unavailable).
17. **Visible thickness check**: a recipe thickness that disagrees with the CAD is the
    BLOCKER `THICKNESS_INCONSISTENT` (§3). DEV recipes carry no thickness.
18. **Backgauge K sensitivity** exported to diagnostics (§3).
19. **Reconciliation**: a duplicate mark is MARK_AMBIGUOUS, not MARK_NOT_IN_FLANGE_GRAPH (§2).
20. **Capability**: B-spline planes, cylinders and lines are recognised (§2, step 2);
    hole centre marks in DXF flats are not bend lines (§5); the symmetry candidates cover
    rotation orders 3 and 6 (§7).

## 10. Real-STEP compile rate (development ledger, 2026-10-17)

**Inputs and custody**:
- The authorized `../source-assets/02_CAD_QC_3D/*.stp` files: 22 files, 20 unique by
  sha256. `11979003_C (1).stp` duplicates `11979003_C.stp`, and `Default (1).stp`
  duplicates `Default.stp`.
- These are development-consumed CAD files. The numbers are engineering evidence, not
  unseen-family accuracy. No scan was read.
- The runs were serial, one process at a time, with `OMP_NUM_THREADS=1`.

**DEV_AUTO_CONFIRMED recipes** (`dev_recipe.dev_candidates`, amendment 4d) carry "every
V10/native bend":
- the **union** of the V10 and native extractor candidates, with geometric duplicates
  dropped;
- candidates feasible for the **same** graph bend merged. These are a bend reported
  piecewise because a relief interrupts it, or two extractors' markers on one bend. This
  is the resolution an operator makes for MARK_AMBIGUOUS duplicates;
- candidates whose own cylinder faces are an unsupported feature under the frozen rule
  **excluded** (a graph feature, or walls the compile's thin-wall test declares OTHER).

The policy never adds a bend that no extractor found. A candidate that matches nothing
and is not a rule-declared feature stays and blocks the compile. Both flange sides are
REQUIRED (default recipe policy), with shop-standard tolerances.

**Amendment-4d figure (strict, every V10 bend as reported): 16 / 20 COMPILED (80 %)**,
exactly the ≥ 80 % floor. Native marks only: 14 / 20 (70 %).

**DEV policy, with E1-graph dedupe: 19 / 20 COMPILED (95 %).** The policy merges extractor
pieces of one bend and excludes rule-declared features with E1's own flange graph, so
MARK_AMBIGUOUS cannot arise from extractor duplicates; the 95 % is therefore partly
circular and is reported as "with E1-graph dedupe". The policy is pinned as
`E1-DEV-POLICY-1`, sha256 `2ea6e87a73183cc0d3c1db2e75f59172d21ed0e0ce93c4765233c91f95158d5b`
(`dev_recipe.py` plus the UNSUPPORTED rule version), recorded in every `dev_candidates`
report, so frozen 4d recipes name the policy that built them.

Re-run after the review fixes (the table): the same statuses in all three columns, and in
the DEV column every nominal identical to the first run (max |Δ| = 0, 13 unavailable).
Strict-source blockers now read MARK_AMBIGUOUS alone where duplicates had also shown as
MARK_NOT_IN_FLANGE_GRAPH (§9, item 19).

**Re-run with revision r2 (2026-10-17 evening, §11).** The run was serial, one CLI
subprocess per compile, on a shared machine while other agents ran.

- Every cell of the table has the same status as before: DEV 19 / 20, V10 marks only
  16 / 20 (the amendment-4d figure, 80 %), native marks only 14 / 20.
- 13 nominals are unavailable, as before.
- Compile wall time is ≤ 8.6 s per part, including interpreter start-up.
- Peak RSS is ≤ 664 MB and output is ≤ 26.8 MB.
- All 73 compiled parts of the run (59 recipe variants and the 14 public frozen recipes)
  validate, hash included, with no `check_compiled_against_recipe` problem.
- Each of the 14 public pre-A1 frozen DEV recipes (`docs/production/runlog/freeze/recipes/`)
  recompiles to its frozen `compiled_part_sha256` byte for byte.
- No sealed file was read.

- All 19 pass `check_compiled_against_recipe`.
- S_0 flatness ≤ 5e-8 mm.
- Compile time ≤ 7.8 s (≤ 8.9 s including candidate extraction), against the 120 s
  budget.
- Peak RSS ≤ 0.70 GB, against the 4 GiB budget.
- Output ≤ 28 MB per part.

The strict-source failures are piecewise duplicates (MARK_AMBIGUOUS), bends one extractor
misses (UNMARKED_BEND), and native marks on rolled geometry (MARK_NOT_IN_FLANGE_GRAPH).

| File (sha256 prefix) | DEV policy | Bends | Unavailable nominals | Compile s | Total s | Peak RSS MB | Out MB | V10 marks only | Native marks only |
|---|---|---|---|---|---|---|---|---|---|
| 10839000_01.stp (ea3e4820) | COMPILED | 12 | 0 | 7.8 | 8.9 | 697 | 28.1 | COMPILED | COMPILED |
| 11979003_C.stp (bde878d2) | COMPILED | 7 | 0 | 3.3 | 4.7 | 490 | 5.5 | BLOCKED MARK_AMBIGUOUS | BLOCKED MARK_AMBIGUOUS |
| 15651004_01.stp (1f88d309) | COMPILED | 3 | 0 | 1.1 | 1.6 | 412 | 0.8 | COMPILED | COMPILED |
| 23553000_B.stp (6e71609f) | COMPILED | 16 | 0 | 6.7 | 8.5 | 565 | 19.0 | COMPILED | COMPILED |
| 28870000_A.stp (aab8c67c) | COMPILED | 1 | 2 | 0.9 | 1.8 | 411 | 0.4 | COMPILED | BLOCKED MARK_NOT_IN_FLANGE_GRAPH |
| 37361005_01.stp (81c428a0) | COMPILED | 4 | 0 | 2.6 | 3.4 | 610 | 6.6 | COMPILED | COMPILED |
| 38172000_01.stp (4d512b5c) | COMPILED | 12 | 0 | 7.4 | 8.6 | 666 | 20.2 | COMPILED | COMPILED |
| 40500000_0 1.stp (f9b8965c) | COMPILED | 9 | 5 | 5.4 | 8.5 | 579 | 12.7 | BLOCKED MARK_AMBIGUOUS | BLOCKED MARK_AMBIGUOUS |
| 42288005_01.stp (7afb000c) | COMPILED | 8 | 2 | 4.2 | 5.1 | 646 | 13.5 | COMPILED | BLOCKED UNMARKED_BEND |
| 44211000_A.stp (2139ca09) | COMPILED | 1 | 0 | 0.4 | 0.8 | 391 | 0.1 | COMPILED | COMPILED |
| 47266000_F.stp (fdf1a1c7) | COMPILED | 1 | 2 | 3.2 | 6.6 | 509 | 0.9 | COMPILED | COMPILED |
| 47479000_C.stp (ada91a64) | **BLOCKED DATUM_PANEL_UNDETERMINED** | 0 | – | 0.6 | 1.9 | 367 | 0.0 | no candidates | BLOCKED DATUM_PANEL_UNDETERMINED |
| 47959001_01PREL.stp (0d25231f) | COMPILED | 6 | 0 | 4.4 | 8.0 | 566 | 5.0 | COMPILED | COMPILED |
| 48811000_A.stp (8ce6789e) | COMPILED | 4 | 0 | 1.6 | 2.2 | 415 | 1.3 | COMPILED | COMPILED |
| 48963008_B.stp (8b6fbb18) | COMPILED | 3 | 0 | 1.9 | 2.7 | 563 | 4.8 | COMPILED | COMPILED |
| 48991006_A 4.stp (9aa36653) | COMPILED | 5 | 0 | 2.1 | 3.2 | 449 | 4.2 | COMPILED | COMPILED |
| 49001000_A-PRELIM.stp (d8c1042f) | COMPILED | 5 | 0 | 3.3 | 4.5 | 654 | 11.8 | BLOCKED UNMARKED_BEND | COMPILED |
| 49024000_A.stp (3cdfcba8) | COMPILED | 3 | 0 | 1.1 | 1.9 | 405 | 0.7 | COMPILED | COMPILED |
| 49204020_04.stp (7a0e6689) | COMPILED | 12 | 0 | 5.3 | 6.5 | 568 | 15.2 | COMPILED | COMPILED |
| Default.stp (85940820) | COMPILED | 1 | 2 | 1.0 | 1.9 | 414 | 0.5 | COMPILED | BLOCKED MARK_NOT_IN_FLANGE_GRAPH |

Per-file reasons:

- **47479000_C** is a rolled part. About 99 % of its area is cylinders, and no planar
  panel is joined by a bend. V10 finds no bend and the native extractor finds one
  cylinder pair, so the result is BLOCKED `DATUM_PANEL_UNDETERMINED`. This is the
  intended outcome: the press-brake line cannot inspect it.
- **28870000_A / Default** are the same family. Each has one press-brake bend compiled.
  The rolled shell (r 30/31.5 mm) and the bend that joins a panel to it are unsupported
  features under the rule. Two flange sides are unavailable (UNSUPPORTED_FEATURE, the
  far boundary is the feature).
- **47266000_F**: one bend is compiled. Two flange sides are unavailable because the far
  boundary is a feature: the part has many toroidal and extrusion faces. Under rule 2 the
  formed feature is one STEPPED_BOSS record (68 faces) and its panel is supported.
- **11979003_C** has a base window (§2, step 3). **40500000** has bends interrupted by
  reliefs (pieces merged).
- **49001000**: V10 misses 2 of the 5 bends that the native extractor and the graph find.
  **42288005**: native misses a 15° bend that V10 finds.

## 11. Evening revision r2 (2026-10-17, E1 re-dispatch)

**Starting point.** The E1 code was re-verified against the contracts as of C01's 19:08 fix
stage. `compiled_part.py` was unchanged; consistency, layout and result rules changed only on
the E3 side.
- All 129 existing E1 tests passed unchanged.
- The changes below then went in. Each is confined to a path that no pre-A1 frozen DEV recipe
  takes (none has a flat or a machine move), so every frozen compile hash is unchanged (§10).
- `compiler.version` therefore stays `e1-1.1.0`. The revision is recorded as
  `diagnostics.compiler_revision = e1-1.1.0-r2 …`, which is outside the hash. The next change
  that alters any other output must bump the version.

**Changes:**

1. **Method B, folding the bound flat** (§5): `flat_binding.fold_flat`, policy `E1-FOLD-1`.
   Plan §3.1 asks for it ("Method B (when a flat is bound): fold the flat"); until this revision
   the stages were always Method A.
   - A bound flat now gives each recipe bend its own strip when the flat determines it.
   - **Validation is synthetic only.** The authorized corpus holds no real flat, so Method B
     has never seen a shop DXF.
   - The gates were set by experiment on the analytic E1 fixtures, before the tests were
     written, and the tests use the same fixtures:

     | Flat | Outcome |
     |---|---|
     | bend table K = 0.30/0.38/0.42/0.35 (hat; 0°, 27°, 180° mirrored) | Method B, every strip exact (< 1e-3 mm; the uniform K is off by up to 0.2 mm) |
     | L, Z, U and two-direction parts with one K per bend | Method B, exact |
     | uniform K = 0.42 | Method B, strips = uniform within 1e-3 mm |
     | last flange 0.05 / 0.1 / 0.3 mm long; one line 0.1 mm off | Method A: the fold does not explain the flat (extent residual 0.05–0.28 mm); uniform strips within 0.03 mm |
     | one strip at K = 0.52 / 0.58 | Method B, that strip clamped to K = 0.48 (0.13 / 0.31 mm), the rest exact; Method A would leave 0.54 mm |
     | one strip at K = 0.66 | BLOCKED `FLAT_DEVELOPED_LENGTH_MISMATCH` (0.565 mm beyond the band; the uniform check alone passed at 0.356 mm) |
     | t = 4 hat, bend table K = 0.20 / 0.47 alternating | Method B, strips exact; COMPILED (per-bend QA 0.0 mm; the single-K check alone blocked it at 0.505 mm) |
     | uniform K = 0.60 | BLOCKED (per-bend QA 1.508 mm: four strips clamped to K = 0.48) |
     | outer lines only bound; flat STEP | Method A: the strips are not observable |

   - Weighting: the line weight is 100. At weight 10, a 0.1 mm end error was absorbed into a
     strip as 0.16 mm and passed a 0.1 mm residual gate. The extent and line residuals after
     the fold are the discriminating statistics, so they are the gate (0.03 mm). A trimmed
     outline RMS dilutes an error confined to one end and is not used.
2. **Per-bend developed-length QA** (§5). When the fold explains the flat, the QA is the
   flat against the nearest plausible per-bend development, replacing the single-K check.
   - This removes a false block: a t = 4 bend table K = 0.20 / 0.47 measured 0.505 mm under
     a single K and 0.0 mm per bend.
   - It exposes one implausible strip that the single-K check spread out: K = 0.66 measured
     0.356 mm under a single K and 0.565 mm per bend.
3. **K uncertainty under Method B**: what the fold leaves unexplained (floor 0.005), instead of
   the uniform fit's residual. On the bend-table hat that is 0.005, where the uniform fit gave
   0.065. E3's backgauge U, `σK·Σ sweep·t`, would otherwise have overstated measured strips
   about 13×.
4. **Wizard `propose_flat_binding`**:
   - Bug fix: with an unmarked unsupported (rigid) bend, the wizard fitted the flat to a
     development with a different datum from the compile's. On the rolled-L fixture the datum
     was P0 instead of P2, K\* 0.07 instead of 0.75, and outline RMS 14.8 mm instead of 10.6 mm.
     It now rebuilds the rigid model exactly as the compile does.
   - Addition: the new `fold` key previews Method B for the proposed map.
5. **Robustness**:
   - An `--out`, or library `out_dir`, that is a file is now `OUT_DIR_NOT_EMPTY`. It used to be
     an uncaught `NotADirectoryError`.
   - A fold failure falls back to Method A with its reason. It never blocks under an unrelated
     code.
   - On an early `STAGE_MODEL_FAILED` return, `unsupported_features` now lists the wall features
     that its warnings name.
6. **`requirements.production.txt`**:
   - The installed transitive closure is pinned, and `anyio` is added.
   - `psutil` is noted as Windows-only and not installed.

**Not done on purpose (contract-gated, open issues):**

- **Stage models for reorders inside one מהלך group** (E3 limitation 8, E2 §10).
  `StageDecisionV1` requires an order-consistent decode to use the prefix model `S<k>`.
  E3's `stage_for_formed` would pick up any `S<k>-<tag>` stage E1 published for such a formed
  set, and its result would then fail validation. C01 must first allow an `OUT_OF_ORDER`-kind
  stage whose formed set equals the order-consistent formed set. Then E1 can publish the
  2^g − g − 1 in-group sets per group of size g (capped).
- **A correspondence rule for `points` general-measurement anchors** (E3 limitation 9). This
  needs a contract field, for example the anchor's panel, skin and in-plane coordinates, which
  E1 would fill.
- **Congruence between competing formed sets** (§7 scope limit, E3 limitation 12). It needs a
  contract field; diagnostics are outside the hash, and E3 rightly ignores them.

**Tests added:**
- `tests/test_prod_engine_compile_fold.py` (16 tests, each positive check paired with the
  Method A value it must beat);
- a library and CLI test for an out path that is a file;
- the bound-flat test rewritten for Method B.

The full E1 suite is 146 tests (129 before the revision). The consumers' tests still pass:
- C05 `test_prod_flat_intake.py`, `test_prod_release_service.py`;
- E3 `test_prod_engine_inspect_cli.py`;
- E4 `test_prod_engine_eval_scorer.py`, `_render.py`, `test_prod_engine_freeze_g3.py`,
  `_legacy.py`, `_physical.py`.
