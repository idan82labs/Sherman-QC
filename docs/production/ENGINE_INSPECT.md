# E3 per-scan inspection engine (`domains/production/engine/inspect`)

Status: implemented 2026-10-17 (overnight run, chunk E3); review fixes applied the same day (engine
`e3-1.1.0`, §12 lists what changed). Plan of record:
`docs/production/PRODUCTION_LINE_PLAN.md` §3.2 and §10 amendments 1, 4 (no file-name or
metadata inputs), 5 (G1 integrity) and 9 (stage-independent registration, articulated fit,
theta search). Binding I/O: `contracts/production/ENGINE_IO.md` §4 and the models in
`contracts/production/inspection_result.py` (the models win on any disagreement). Inputs come from
E1 (`docs/production/ENGINE_COMPILE.md`); the qualification data from E2
(`docs/production/ENGINE_SYNTHETIC.md`).

The engine takes one scan, the released recipe and its CompiledPart, and answers, for every planned
bend in sequence order, *how far* each planned characteristic deviates, or why it cannot say. It
never discovers bends blindly, never reads the scan's file name, and never turns an unobserved value
into a pass, a fail or a nominal.

## 1. Entry points

```
python -m domains.production.engine.inspect \
    --compiled DIR/compiled_part.json | DIR   # its directory holds stages/
    --recipe RECIPE.json --scan SCAN.ply|SCAN.stl --out OUT --scan-sha256 HEX
    [--compiled-sha256 HEX] [--recipe-sha256 HEX] [--context CONTEXT.json]
    [--declared-stage K --declared-stage-source SIDECAR|OPERATOR|API]
    [--unit-factor 1|25.4|1000] [--time-budget-s 170] [--engine-bundle-id ID] [--config CONFIG.json]
```

- **Stdout** carries only `PROGRESS\t{"stage","progress"}`, one `RESULT\t{"summary_state",
  "inspection_validity", "stage_status", "result": "result.json"}` and `ERROR\t{"code","message"}`
  records. Logs go to `engine.log` (and stderr on a crash).
- **Exit codes.** 0: `result.json` written and valid. Every inspection outcome is a result,
  INVALID_INSPECTION included. 2: invalid input. The ENGINE_IO §4 integrity failures write
  `engine_error.json {code, message}` with `COMPILED_PART_MISSING`, `COMPILED_PART_HASH_MISMATCH`
  (the compiled file does not validate, differs from `--compiled-sha256`, disagrees with the
  recipe, or a stage file is missing or altered), `COMPILE_BLOCKED`, `RECIPE_SHA_MISMATCH`,
  `SCAN_MISSING` or `SCAN_HASH_MISMATCH`. Usage problems print only an ERROR record and write
  nothing: `OUT_DIR_NOT_EMPTY`, `CONFIG_INVALID`, `CONTEXT_INVALID`, `DECLARED_STAGE_INVALID`,
  `USAGE`. 3: internal error, including a result that fails its own
  `check_result_against_compiled` (ENGINE_IO §4.3) and any unexpected failure outside the engine (an
  input file that cannot be read, an output directory that cannot be written): `ERROR
  INTERNAL_ERROR`, never a bare traceback with exit 1.
- **Output directory** (`layout.ENGINE_OUTPUT_FILES`, every geometry file in `cad_model_mm`,
  tmp + rename, `result.json` last): `result.json`, `registered_scan.ply` (all ingested points after
  `unit_factor` and `raw_to_frame`; absent when no proper pose is published), `deviations.f32`
  (index-aligned, NaN where none; only with a trusted heatmap), `segment_labels.u16` (0 unassigned,
  `1+i` panel `i`, `1001+j` bend `j`, 65535 unplanned crease; under an ambiguous pose only the datum
  is labelled), `display/stage_mesh.ply` (S_k̂), `display/final_mesh.ply` (S_N), `bend_geometry.json`
  (`BendGeometryFileV1.from_result`) and `engine.log`. "Ingested" excludes non-finite vertices and
  gross outliers (a vertex more than `gross_outlier_extent_factor` = 10 robust extents from the
  trimmed bounding box, such as a 3e38 sentinel); both are counted in a warning
  (`SCAN_NONFINITE_POINTS_DROPPED:n`, `SCAN_GROSS_OUTLIERS_DROPPED:n`).
- **Library.** `load_inputs(...)` (the integrity checks; raises `InspectInputError`),
  `inspect_scan(inputs, out_dir, cfg, InspectOptions(...))` (returns `InspectOutcome(result,
  files, log)`; `write_outputs=False` keeps it in memory) and `load_config(path, **overrides)`.
- **Configuration.** Every threshold is a field of `InspectConfig` (`config.py`). `--config`
  overrides fields; unknown keys are refused, and every value is checked on construction: finite,
  inside its documented range, whole where a count is expected, and the cross-field orders
  (`voxel_min_mm <= voxel_max_mm`, `stations_min <= stations_max`, `flat_fraction_max + 2
  prior_band < formed_fraction_min`, ...). A NaN or a zero station count is a `CONFIG_INVALID`
  usage error, never a hashing crash or a division by zero mid-measurement. The effective
  configuration's canonical sha256 is `result.engine.config_sha256`, so one hash names one engine
  behaviour (amendment 4). Engine version `e3-1.1.0`; `result.engine.algorithm_versions` names
  every stage's algorithm.
- **Blindness** (amendment 4). The format is decided from the bytes; the name is never read.
  `--context` is copied into `result.context` and read nowhere else. A renamed copy of a scan gives
  an identical result once `created_at`, `context`, `engine.duration_s` and `engine.peak_rss_bytes`
  are set aside (tested).

## 2. Pipeline

| Step | Module | What it does |
|---|---|---|
| 1 Ingest | `scan_io`, `cloud` | PLY (ASCII, binary LE/BE, points or mesh) or STL (binary/ASCII, shared vertices merged); non-finite and gross-outlier vertices dropped and counted. Explicit unit factor only. Voxel cloud (0.8–2.5 mm, never finer than the scan spacing) with PCA normals, a planarity measure and a radius neighbour graph; the full-resolution cloud is kept for every measurement. |
| 2 Datum registration | `registration` | Stage independent: the datum plane, its free (cut) edges and holes only (§3); the sheet thickness from the datum's two skins (§3.1). |
| 3 Articulated fit | `articulated` | Flange-tree walk with the hinge theta search and region-grown planes (§4). |
| 4 State decode | `decode` | FORMED / NOT_YET_FORMED / PARTIALLY_FORMED / UNEXPECTED_FORMED / UNKNOWN, order as a soft prior, stage model (§5). |
| 5 Metrology | `metrology` | 5–11 section stations per formed bend: angle, OML sharp, bend-line position, flange lengths, backgauge, U (§6). |
| 6 Decisions | `result` | Contract factories only (guard band, UNKNOWN reasons, requiredness) (§7). |
| 7 Heatmap, labels, creases | `engine` | Signed deviation vs S_k̂, per-point segment labels, unplanned creases (§8). |
| 8 Result | `result`, `engine`, `outputs` | Validation, `check_result_against_compiled`, files. |

### 2.1 Thin sheets and the point spacing

A PCA neighbourhood that is not much wider than the sheet sees both skins at once and is nearly
isotropic, so its normal is noise. When `t < 1.2 x` the neighbourhood radius (*slab mode*), normals
are computed on a coarser voxel cloud (`1.8 t`), where the sheet reads as one slab, and every voxel
takes the normal of its nearest coarse voxel; planarity is judged against the slab half-thickness.
Least-squares planes are never fitted through two skins at once: a thin sheet whose skins are seen
in different areas would tilt the fit by about `t / distance` (0.6° on a 300 mm datum).

## 3. Registration (stage independent)

The registration runs on the scan thinned to one point per 0.55 mm cell when it is denser
(deterministic); the datum fit gains nothing from more density, and every measurement uses the full
cloud.

1. **Planes.** Planar voxel points are binned by normal (a 4° Fibonacci grid); the densest
   direction's offset histogram gives a plane; its inliers are split into connected segments over
   the voxel graph; repeated. A segment's normal is the trimmed mean of its voxel normals.
2. **Candidates.** The largest segments with at least 12 % of the datum skin area. A parallel
   segment one thickness away is the other skin and fixes the material side.
3. **Coarse in-plane search.** For both ways the plane can map onto the datum (face up or face
   down, which mirror the outline as seen in the plane), the segment is rasterised at 2 mm and
   FFT-correlated, every 3°, with the datum *kernel*: +1 inside the outline (holes excluded), a
   penalty inside every hole and in an 8 mm band outside every free edge, and nothing beyond the
   datum's bend edges, where an unformed bend leaves a flat continuation. A free edge E1 marks
   `registration_usable = false` is treated like a bend edge (no band, no edge rows). The best peaks over all
   candidates and angles are pooled, angle-polished and turned into 3D poses (both skin heights
   unless the other skin was seen).
4. **Fine search.** Each peak is searched again over ±20 mm / ±2° on a 0.5 mm raster whose kernel
   weights hole cores separately (a 5 mm hole is invisible at 2 mm), which ranks the peaks.
5. **Fine fit.** The datum's skin heights are assigned *locally* (12 mm tiles: two height levels
   one thickness apart = both skins). Tilt and height come from the tiles that show both skins
   when there are enough of them; a datum seen on one skin only takes its side from the cut walls
   (points between the skin levels near the free edges and holes). Hole walls stay out of the
   height points. Then the in-plane pose is solved by weighted least squares on per-bin free-edge
   observations (point to edge line) and per angular-bin hole boundaries (radial), iterated,
   Huber-weighted; only constrained directions are updated.
   - **Walls and erosion.** A bin whose cut wall is seen (at least 3 points between the skins near
     the extreme; 2 per hole bin) is observed at the wall's median: the wall *is* the edge. A bare
     skin edge sits at a range-image depth discontinuity, where a structured-light scanner erodes
     it inward (on the DEV suite: walled bins +0.02 mm, bare bins -0.15 mm median, -0.36 mm at the
     10th percentile; TOP_ONLY scans have ~20 % bare bins, FULL scans < 1 %). Bare rows get weight
     0.15. Their erosion scale is measured on the scan itself (walled median minus bare median,
     separately for edges and holes, clipped to 0.2–1.5 spacings; 0.5 spacings when not
     measurable).
   - **Pose uncertainty.** The covariance is the weighted least-squares sandwich with the residual
     scatter of the walled rows (each hole's radius bias removed). The gain of every bare row on
     the pose, times its erosion scale, gives a worst-case **pose bias** at any point and direction
     (`InPlaneFit.position_bias_mm`: the larger of the positive and negative sensitivity sums), which
     enters the bend-line position U.
6. **Score and degeneracy.** The refined datum score combines outline coverage, the share of
   free-edge and hole bins that agree with the CAD and the forbidden-zone violation. The 3x3
   information matrix of the constraints gives the degeneracy test: `in_plane_degenerate` is set
   when the weakest direction has fewer than 3 effective observations or the position sigma at the
   outline exceeds 0.25 mm; only bend-line position and the heatmap are then withheld (amendment 9).

### 3.1 Sheet thickness

The real sheet is rarely exactly the CAD thickness (stock tolerances of a few per cent), and every
single-skin construction ("the other skin is `t` away") inherits the difference. After the fine fit,
the datum's points inside its outline (1.5 mm away from edges and holes) are cut into 12 mm tiles;
a tile shows both skins when, besides its densest 0.02 mm level, a second level at least 15 % as
dense lies within ±25 % of the compiled `t` from it (the registration's own skin search only looks
±0.15 mm around `t`, and a 2 mm sheet that is 10 % thicker escapes it). Each skin level is the median
within 0.12 mm of its peak, so a residual tilt or bow does not mix the skins. With at least 5 such
tiles, the median separation is the **measured thickness** (`tile_thickness`), its standard error
of the median floored at 0.01 mm. It replaces the compiled `t` in the articulated fit (every child's
two-skin mid-plane and single-skin offsets), in the station fits and in the crease sharps; a measured
`t` more than 8 % from the compiled one is a warning (`THICKNESS_DIFFERS_FROM_CAD:...`). When the
datum shows one skin only, the compiled `t` is used and its stock tolerance (`0.04 t` standard, a
uniform ±7 %) enters U; if that one skin is skin 1, the frame itself was placed `t` below it, and the
same tolerance enters every bend-line position along the datum normal (`frame_height_mm`).

On the E2 DEV suite the measured separation scatters by 0.036 mm (σ) around the exact CAD thickness:
the FULL plan's TOP and BOTTOM setups are rigidly misregistered (the ASSUMED scanner model's
duplicate-skin offset), so the two skins of one panel are offset by that motion. Using the measured
`t` therefore costs some accuracy on DEV, where the CAD `t` is exact by construction (ADJACENT_BEND
lengths, all scenarios, P95 0.055 → 0.071 mm on the full tier, 0.068 → 0.092 mm on the regression
tier), and
makes the engine robust where it is not: with a compiled `t` 10 % thinner than the sheet, every
MEASURED row of a TOP_ONLY scan stays inside U (it did not before; tested). The skin-level
misregistration enters U through `u_skin_level_mm` (§6).

**Hypotheses, twins, ambiguity, mirror.** The best refined poses (up to 6) are kept. The best three
get the articulated fit, and so does every *symmetric twin* of one of them: a pose whose motion to
it matches a ROTATION candidate of some compiled stage's symmetry test (rms within E1's 0.3 mm
threshold, rotation within 6°, median displacement within 6 mm). The pose with the best datum +
explained-scan score wins, except among twins: a twin maps the datum onto itself, so its
datum-score difference is noise, and a twin whose total or explained shares are within 0.03 of the
best competes with it. Among competing twins the planned **order is the soft prior**: an
order-consistent decode wins (a symmetric U-channel with one flange formed is S1, not the
out-of-order image of it). `pose_ambiguous` is declared only when E1's symmetry test of the *chosen
decode's own stage* matches the motion to a competing twin and the two disagree on a bend state or
on the identity of a formed bend (the candidate's bend permutation). Then nothing is measured,
bends are UNKNOWN(AMBIGUOUS_POSE), and NOT_YET_FORMED survives only where every rival agrees. A part
E1 does not declare symmetric is never ambiguous: the best pose is taken. The mirrored scan
(x -> -x) is registered too when the best proper pose looks suspicious (a datum hole filled with
material, a low datum score or a poorly explained scan); if it fits clearly better
(`mirror_margin` 0.08) the result is MIRRORED_PART_SUSPECTED and no transform is published. A part
whose mirror image is congruent to itself fits both equally and is never called mirrored.

## 4. Articulated fit

The flange tree is walked from the datum, parents first. For every bend, full-resolution points
near the hinge (the bend's axis carried to the scan by the parent's fitted pose) are expressed in
the cross-section frame `u` (the parent's flat continuation) / `v = axis x u` (the side the child
rotates toward).

- **Theta search** (amendment 9). For every angle from -10° through flat (0°) to the nominal
  sweep + 25° (1° steps, then 0.1°), points on the child side inside the band `Ri + 2t + 5 mm` of
  the axis are histogrammed by their line offset `rho`; the score is the best two-skin line count
  (inner and outer skin one thickness apart, near the nominal tangent offsets within ±3 mm).
  Support starts past the bend's own arc tangent zone (`sqrt(2 Ro window) + 0.3 mm` from the
  tangent point): every line between the parent's and the child's tangent touches the arc there,
  which otherwise lets a short formed flange lose to a spurious "flat" plateau. The fine angle may
  look 2.5 band radii along the child for a sharper peak. A ±25° window around nominal would miss
  unformed flanges; this search sees them at 0°.
- **Thickness.** Every child inherits the sheet thickness the registration measured on the datum
  (§3.1; the compiled `t` when the datum showed one skin).
- **Region growing**, not a distance-to-nominal gate. Seeds are the winning line's points. The
  child's material slab (mid-plane ±(t/2 + 0.35 mm), voxel normals within 12°) is grown over the
  voxel graph from the seeds, the two skins are refitted robustly and the growth is repeated until
  it stops changing. Points owned by the parent are never taken. A flange that is too long, too
  short or bent wrong is therefore measured, not dropped.
- **Skin levels.** On the converged slab the skins are its height levels (two 0.05 mm levels one
  thickness apart, or one), not its two halves: cut walls, flare and fused setups fill the slab
  between the skins and would pull a split-at-the-mid-plane estimate by a few tenths (seen on
  TOP_ONLY scans, where it pushed station fits off their skin).
- **Pose: fold kinematics** (`fold_transform`). For a found angle `theta` the arc consumes
  `theta * R_n` of the bend's neutral strip `BA`, centred on the bend line: the tangent line moves
  toward the child by half the unconsumed strip and the other half stays flat on the child (the E2
  fold model; E1's Method A at `theta = 0`; the identity at the nominal sweep). Rotating the
  final-pose child about the nominal axis instead misplaces a half-formed flange by ~2 mm, enough
  to hide its own children's hinges. The pose is then moved along the child normal onto the
  fitted slab. Grandchildren are searched from it.
- A child with neither a formed flange nor a flat continuation near its hinge is UNKNOWN
  (OCCLUDED, or LOW_DENSITY when its line covers under 12 % of the bend length); everything below
  it is UNKNOWN (BEND_STATE_UNKNOWN).

The **explained shares** of the scan (voxels within 1 mm, and within a tight `0.35 t` in
[0.25, 0.5] mm, of the articulated model: the final-stage sample cloud mapped to the decoded formed
set with E1's `stage_points_from_compiled`, then moved panel by panel onto the fitted poses) rank
the pose hypotheses (the tight share separates the two skin heights of a datum seen on one side)
and flag wrong parts.

## 5. State decode and stage model

`formed_fraction = measured sweep / nominal sweep` (the dihedral of the fitted parent and child OML
planes): at most 0.12 (or within 4° of flat) is NOT_YET_FORMED, at least 0.82 FORMED, in between
PARTIALLY_FORMED (a partial bend whose angle passes the guard band is FORMED by definition). Inside
±0.04 of a threshold the order prior breaks the tie; it never overrides clear evidence and is
reported as `prior_formed_probability` (the share of lower-ranked bends that are formed). A formed
bend that ranks above a not-yet-formed one (all bends of one מהלך share the group's first rank) is
UNEXPECTED_FORMED, so the stage decision flags OUT_OF_ORDER_SUSPECTED.

A sequence-prefix formed set decodes to `S<k>`; any other formed set to a compiled stage with
exactly that formed set (`S<k>-<tag>`) when E1 provides one, else no stage model
(`stage_model_id` null, no heatmap; nominal lines come from E1's `stage_transforms_from_compiled`
for that formed set). With UNKNOWN bends the smallest compiled stage consistent with every known
state is named for display only (no heatmap). A declared stage (`--declared-stage`) that disagrees
with the inferred one is STAGE_MISMATCH and goes to review; it is never used to decode.

## 6. Section metrology

Per formed-like bend, the axial extent where both flanges are seen (trimmed by min(2 mm, 5 %) at
each end, so lateral cut walls and corner rounds stay out) is cut into 5–11 contiguous stations of
about 12 mm. In each station the parent's and child's OML planes are fitted on the flat part of
each flange only (past this bend's tangent line `Ro tan(sweep/2)` from the sharp, short of the
side's far boundary), robustly (Tukey IRLS). The panel plane only says where to look: each skin's
level is taken within ±min(0.6 mm, 0.4 t) of where the plane puts it (the densest 0.05 mm level when
under half of its points sit within 0.15 mm of the plane), so a panel plane a few tenths off near
one bend does not empty the station.

- **Angle**: `180 - angle(n_parent_out, n_child_out)`.
- **OML sharp**: the two station planes intersected at the station's axial position. A flange whose
  OML skin is not seen uses its inner skin moved by `t`: the thickness measured on the datum (§3.1),
  else the compiled `t`. The share of stations that did so is recorded per side
  (`inner_skin_share_parent/child` in the row diagnostics, with `thickness_used_mm`), and the
  offset's uncertainty enters the U of every characteristic whose sharp it moves (table below).
- **Bend-line position**: the station sharps' offsets from the decoded stage's nominal OML sharp,
  perpendicular to the nominal line, projected on side A's stage-pose outward direction.
- **The bend's OML sharp line** (the far boundary of its neighbours' ADJACENT_BEND lengths) is the
  intersection of the region-wide panel planes: fitted over whole flanges it is more precise than
  the station sharps (DEV: P95 0.058 vs 0.078 mm). When the station sharps sit more than 0.15 mm
  from it a panel plane is misfitted near this bend and the line follows the station median.
- **Flange length**: from the station sharp along the side's in-plane direction away from the bend.
  `FREE_EDGE`: the farthest material of the side panel at that station: the grown region, then
  contiguous material-slab points (gaps up to 2.5 spacings, so the run reaches the real cut edge
  past the region's blurred last voxel), and the median of the visible cut-wall points between the
  skins when at least 5 are there (the wall *is* the edge), else the 3rd-largest extent (a bare skin
  edge, eroded: see U). **Truncated runs**: when at least 8 points of the panel's grown region lie
  more than max(2 mm, 1.5 voxels) beyond the run's end *outside* the material slab, the panel
  continues where its plane does not (a bow, a kink nothing unfolded): that station is dropped, and
  too few stations left make the length UNKNOWN(AMBIGUOUS_SEGMENTATION), never a short value.
  **Bare far edges** (no cut wall at half the stations or more) are a lower bound of the length: a
  scanner erodes such an edge by a few tenths (in U), but a clamp or an out-of-view edge can end the
  visible material millimetres short of it. They can prove a side too long, never too short: a
  short-side FAIL there becomes UNKNOWN(EDGE_NOT_SEEN) with the shortfall in the note; PASS, REVIEW
  and long-side FAIL stand. `ADJACENT_BEND`: to the adjacent bend's OML sharp line (OML sharp to OML
  sharp); an adjacent bend that is NOT_YET_FORMED gives NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE): the
  length is undefined at that stage (contract amendment ENGINE_IO-A1, e3-1.11.0, owner decision
  2026-10-19). An UNKNOWN or unmeasured adjacent bend still gives UNKNOWN(ADJACENT_BEND_NOT_MEASURED).
  **Unplanned crease** (§8) running along the bend (within ~37°) on a FREE_EDGE side: the side ends
  at the crease's OML virtual sharp (method `OML_SHARP_TO_UNPLANNED_CREASE_SHARP`), as the contract
  measures to any bend's OML sharp and as E2's truth does; UNKNOWN(AMBIGUOUS_SEGMENTATION) when that
  sharp cannot be resolved (with one skin of the creased segment seen, the panel's material must end
  within 30 % of the two candidate lines' separation of one of them, and 25 % nearer it than the other).
- **Backgauge** (decoded order-consistent stage S_k, 1 ≤ k ≤ N−1, on bend k with the compiled
  side): the same far-edge measurement across the side panel and every panel reached through
  unformed bends (E1's flat region), or to the OML sharp of an already formed bend. A not-yet-formed
  bend may carry a kink of up to 12 % of its sweep (or 4°), so the island is first **unfolded**
  (`unfold_island`): walking out from the side panel, each island panel is rotated onto its
  predecessor's plane about the line where their fitted OML-skin planes meet, then moved toward that
  line by the kink's developed-length correction `2 r_skin tan(θ/2) − θ R_n` (`r_skin = Ri + t` when
  the measured skin is the kink's convex side, `Ri` when concave; `R_n = Ri + K t`), so the far edge
  is found on developed material, as S_k has it. The line lies on both fitted planes, so an error in
  where it sits moves the unfolded points by a second-order amount only; planes parallel within
  0.05° are only shifted along the normal. A point belongs to a panel when it lies in that panel's
  material slab, on its side of the line and within its grown extent along it. The evidence notes
  name every unfolded kink (`unfolded_B4_deg=4.249`). When the formed set is order-consistent but
  not the sequence prefix (a reorder inside one machine move) the S_k nominal does not describe the
  part: NOT_APPLICABLE_AT_STAGE.
- **General measurements** (`gm/<id>`, recipe-tagged): `bend_pair` anchors are carried from the
  nominal OML sharps onto the measured ones (each anchor's offset in its bend's own frame), so the
  distance is taken between the same physical loci; a pair with a not-yet-formed bend is
  NOT_APPLICABLE_AT_STAGE. `points` anchors stay UNKNOWN(UNSUPPORTED_FEATURE) (no defined scan
  correspondence yet).

Stations are aggregated by the median, or by the dominant level when at least 60 % of the stations
share one and a minority sits on a notch or tab; min/max are reported in `stations`.

**Uncertainty** (`metrology.u_budget`): `U = 2 sqrt(sum (f_i c_i)^2)` with unit components

| Component | Applies to | Unit value |
|---|---|---|
| station | all | standard error of the median, `1.2533 MAD-sigma / sqrt(n)` |
| tilt | angle | `hypot(rms_parent / extent_parent, rms_child / extent_child)` (rad -> deg) |
| sharp | position, lengths | `hypot(rms_other, u_skin_level) / sin(sweep)`: where the OML intersection moves when the other plane sits off, by its residual (duplicate skins, fusion steps) or by the skin-level misregistration between setups (`u_skin_level_mm` 0.015) |
| far_sharp | ADJACENT_BEND lengths, crease-ended lengths | the same for the far bend's (crease's) sharp |
| edge | FREE_EDGE lengths, backgauge | the scan's median NN spacing |
| erosion | FREE_EDGE lengths, backgauge | the scan's erosion scale, in full once half the stations show no cut wall (linear below) |
| thickness | single visible skin | `u_t / sin(sweep)` along the other flange, times `min(1, 2 x)` for the share `x` of stations that used the inner skin there; `u_t = hypot(0.02 t, sigma_t)` with the measured `t`, `hypot(0.02 t, 0.04 t)` with the compiled one (factor 1: already a standard contribution) |
| frame_height | position | `0.04 t` along the datum normal when the datum showed only its skin 1 (the frame placed `t` below it) |
| pose | position | the datum frame's in-plane sigma (sandwich covariance x walled-row scatter) |
| pose_bias | position | the worst-case datum shift at the nominal line along side A from eroded bare rows (§3) |
| k_sigma | backgauge | `sigma_K \|dN/dK\|` from hashed fields only: `sigma_K` = `k_factor.uncertainty`; `dN/dK` = `sum(sweep t / \|cos phi\|)` over the crossed strips that at least half of 11 stations laid along bend k's stage-model line cross short of the nominal far boundary (phi between the station and the strip's width direction). E1's `diagnostics.backgauge_nominals` (outside `compiled_part_sha256`) is only compared, in the log; an edited diagnostics block cannot change a result (tested). |
| floor | all | 0.03 mm / 0.03° / 0.04 mm |

Factors (`InspectConfig`): tilt 1.0, sharp 1.0, edge 0.15, thickness 0.02, pose 1.0 and the
skin-level 0.015 mm were chosen on the E2 DEV regression tier only (never QUAL) as the smallest values
that keep at least 99 % of DEV characteristics inside U, i.e. with margin over the 95 % gate (§9);
the skin level is the one added by the review fixes (ADJACENT_BEND lengths 95.0 % → 98.3 % inside U
on the regression tier once the measured thickness carries the setup misregistration). The stock
tolerance 0.04 t is a sheet-metal assumption, not a DEV fit (E2 never varies `t`). The pose term is
analytic (the in-plane fit's sandwich covariance), not a perturbation probe. The two bias bounds enter
with 1/sqrt(3) = 0.577 by construction (a bound `b` of a uniform error is `b / sqrt(3)`); the full
tier checked them, it did not tune them. The backgauge's `k_sigma`
term follows the contract (`KFactorV1.uncertainty` "carried into stage-model nominals' U"): with
the shop K's 0.05 standard uncertainty a backgauge across several strips is usually REVIEW, never
a false PASS.

## 7. Decisions and honesty rules

Rows are built only by `CharacteristicResultV1.measured / not_measured / not_applicable`, so the
guard band (PASS only if `|d| + U <= tol`, FAIL only if `|d| - U > tol`, else REVIEW), units,
expected values, tolerances and requiredness come from the compiled nominal. The engine:

- never copies a nominal into `actual` (an UNKNOWN row has no actual, deviation, U or stations);
- reports NOT_YET_FORMED bends as NOT_APPLICABLE (BEND_NOT_YET_FORMED; NOT_INSPECTED_BY_RECIPE for
  a side the recipe does not inspect), so they are never FAIL;
- reports compile-unavailable nominals as UNKNOWN (UNSUPPORTED_FEATURE / NOMINAL_UNAVAILABLE);
- measures a PARTIALLY_FORMED bend normally (its angle fails the guard band);
- never takes the visible end of a far edge without its cut wall for proof that a side is too short
  (UNKNOWN(EDGE_NOT_SEEN), §6), and never measures to a run its plane could not follow
  (UNKNOWN(AMBIGUOUS_SEGMENTATION));
- never uses a value from outside `compiled_part_sha256` (`diagnostics`) in a measurement or a U;
- self-checks every result with `check_result_against_compiled` and exits 3 on any problem.

## 8. Validity, heatmap, labels, creases

Validity states, in order: `UNSUPPORTED_INPUT` (unreadable scan), `INSUFFICIENT_CAPTURE` (fewer than
2000 points), `UNIT_MISMATCH_SUSPECTED` (scan extent off both the compiled blank's and the final
shape's by more than 6x; the reason names a likely inch or metre factor; nothing is rescaled),
`UNTRUSTED_REGISTRATION` (no datum plane matched, abstained; or the datum barely covered),
`TIMEOUT` (the budget ran out before the registration finished: the datum fit, or the pose
selection, i.e. the articulated fit of every competing pose and, when the best looked suspicious,
of the mirrored scan's), `MIRRORED_PART_SUSPECTED` (§3), `WRONG_PART_SUSPECTED` (a datum hole filled
with material, or under 60 % of the scan explained). A cut articulated fit explains less of the
scan, so the explained-share checks only ever see complete fits: a budget cut inside the walk is
TIMEOUT, never WRONG_PART (tested by expiring the budget at every point of the walk). A budget that
runs out later, in the metrology, leaves a VALID result whose unmeasured characteristics are
UNKNOWN(TIMEOUT); FREE_EDGE sides are then UNKNOWN(TIMEOUT) too when the unplanned-crease search could
not run (a crease would end them). Any non-VALID state measures nothing, leaves every bend UNKNOWN
and publishes no heatmap; a proper pose of a wrong part is still published with its registered scan.

- **Heatmap**: signed point-to-surface deviation of every registered point against the decoded
  stage model's sample cloud (nearest sample within 6 mm and 1.5 mm laterally, positive along the
  outward normal), only with a trusted, exactly decoded, unambiguous pose. The legend band is the
  recipe's `surface_profile_mm`.
- **Display**: `nominal_polyline_mm` is the decoded stage model's bend line (under an ambiguous
  pose, the chosen decode's stage: every twin puts its bends at the same places, so the grey lines
  stay on the part although no identity is claimed); `measured_polyline_mm` is the measured OML
  virtual sharp (exactly when BEND_LINE_POSITION is MEASURED), shifted onto the station median. The
  measured **outer apex** (the visible-surface line G2/G4 draw solid on the material) is in each
  row's `diagnostics` as `outer_apex_{start,end}_{x,y,z}`, shifted with the same station median; see
  open issues.
- **Segment labels**: fitted panels `1+i`, bend zones `1001+j`, unplanned creases 65535. Under an
  ambiguous pose only the datum (which every twin maps onto itself) is labelled: a flange label would
  be the confident identity the contract forbids on a symmetric part.
- **Unplanned creases** (found before the metrology, §6): unexplained planar regions (outside every
  bend zone, at least 60 mm², not a thin cut-wall strip) hinged within 6 mm on a fitted panel at 6°
  or more, whose material leaves the panel's slab (90th-percentile height at least max(1 mm, t/2)
  outside it: an unowned strip of the panel itself fits a tilted plane but stays inside) are
  reported as `EXTRA_BEND_SUSPECTED` with the crease line and its included angle (REVIEW_REQUIRED).
  The two skins of a thick creased segment are often separate components; parallel components one
  thickness apart are merged. The crease's OML virtual sharp is the intersection of the convex
  skin's planes (skin 0 continues across a crease; a segment bending toward `-n0` has skin 0
  outside); with one skin of the segment seen, the skin whose crease line meets the panel's
  material end is taken (clearly: §6), else the sharp is unresolved. The crease is drawn on its OML
  sharp. The search runs only when every bend state is known.

## 9. Measured numbers (E2 DEV seeds only; synthetic engineering evidence)

All numbers below are synthetic, on E2 generator 1.1.0 DEV seeds with the ASSUMED scanner model
(amendment 4b's model card was not yet available), compiled by E1 1.1.0. QUAL seeds were never
generated or read by this chunk. They are not physical accuracy and not a G1 qualification (G1 is
gated on QUAL, E4 scores it).

One engine configuration for every number below: the default `InspectConfig`, `engine.config_sha256`
`960f5f3e2e009da4b348fe5aabfe9065641d30fafe49558746cbb60a951568a9` (engine `e3-1.1.0`, after the
review fixes). Tiers were generated with E2's CLI into `$TMPDIR` (`regression` 50 cases ~200k points,
`full` 216 cases, `runtime` 10 cases ~1M points), compiled by E1's CLI and run through E3's CLI with
`--compiled-sha256` and `--recipe-sha256`. Two independent scratch scorers (the implementer's and the
reviewer's, both written from plan §5, amendment 5 and E2's scoring rule) agree on every number
here. Timings were taken on the shared development machine while other chunks were running (load
average 5-12; 6-7 cases in parallel for the full and regression tiers, the runtime tier serially),
so they are upper bounds. The `e3-1.0.0` value is in brackets where the fixes moved it.

**G1 table** (complete parts, FULL plan):

| Characteristic | Gate | full (216) | regression (50) | runtime (10, ~1M pts) |
|---|---|---|---|---|
| Flange length bias | ≤ 0.05 mm | +0.0004 (n 412) [-0.0010] | +0.0025 (n 113) [+0.0015] | -0.0013 (n 49) |
| Flange length P95 | ≤ 0.20 mm | 0.074 (max 0.179) [0.071] | 0.103 [0.085] | 0.070 [0.069] |
| Angle bias | ≤ 0.10° | +0.0023 (n 209) | -0.0021 (n 57) | +0.0035 (n 25) |
| Angle P95 | ≤ 0.30° | 0.045 [0.046] | 0.028 | 0.031 [0.028] |
| Bend-line position P95 | ≤ 0.30 mm | 0.062 (max 0.116) | 0.074 | 0.041 |

**Other G1 requirements and amendment-5 integrity** (full tier unless noted):

| Requirement | Gate | Result |
|---|---|---|
| Coverage with realistic occlusion (MEASURED share, truth-defined, non-symmetric) | ≥ 90 % | 98.8 % of 1571 [99.0 %] (FULL 98.0 %, TOP_ONLY 95.6 %, SPARSE 78.7 %); regression 100 %; runtime 100 % |
| False PASS (PASS while true \|dev\| > tol) | 0 | 0 of 1552 (regression 0 of 322; runtime 0 of 106) |
| False FAIL (FAIL while true \|dev\| ≤ tol) | ≤ 0.5 % | 0.00 % of 1261 (regression 0.00 %, runtime 0.00 %) |
| U calibration (\|measured − truth\| ≤ U) | ≥ 95 % | 99.81 % of 1552 [99.87 %] (FULL 99.85 %, TOP_ONLY 100 %, SPARSE 98.3 %); regression 99.69 % [100 %]; runtime 99.06 % |
| Median U at full visibility | ≤ 0.15 mm, ≤ 0.25° | 0.116 mm, 0.154° [0.108 / 0.154] (complete parts 0.111 / 0.154); regression 0.120 / 0.146; runtime 0.108 / 0.149 |
| Decisiveness (PASS/FAIL when true \|dev\| ≤ 0.5 tol or ≥ 1.5 tol) | ≥ 90 % | 98.4 % of 1412 [98.6 %] (regression 98.0 %, runtime 100 %) |
| Slope of measured vs true deviation (\|true dev\| > 0.2) | [0.95, 1.05] | 0.9999 (n 430); regression 1.0002; runtime 0.989 |
| Per-bend state on truth-unambiguous cases (AMBIGUOUS/UNKNOWN count as errors) | ≥ 98 % | 100.00 % (498/498); regression 111/111; runtime 29/29 |
| Symmetric cases: AMBIGUOUS, never a confident wrong ID | — | 4 symmetric complete U-channels AMBIGUOUS_POSE (nominal lines drawn on the decoded stage, only the datum labelled); 13 symmetric stage-0 / missing-bend cases decoded with no formed-bend identity at stake; 0 confident wrong IDs |
| Injected backgauge errors ≥ 0.5 mm detected | all | 21/21 measured and not PASS (regression 2/2, runtime 1/1) |
| No NOT_YET_FORMED bend ever FAIL | 0 | 0 |
| More occlusion raises UNKNOWN, not error | — | every case exits 0 (216 + 50 + 10); unmeasured rows carry LOW_DENSITY / EDGE_NOT_SEEN / AMBIGUOUS_* / ADJACENT_BEND_NOT_MEASURED reasons |
| Wrong SKU and mirrored never PASS | — | 8/8 wrong SKU invalid (5 WRONG_PART_SUSPECTED, 3 UNTRUSTED_REGISTRATION), 8/8 mirrored MIRRORED_PART_SUSPECTED |
| Missing / extra bend in a FINAL scan never ALL_MEASURED_IN_TOLERANCE | — | 11 missing-bend cases REVIEW_REQUIRED; 8/8 extra-bend creases found (7 OUT_OF_TOLERANCE_FOUND, 1 REVIEW_REQUIRED); the split flange measured to the crease sharp in 7/8 (error ≤ 0.07 mm, inside U), UNKNOWN in 1 |
| Runtime P95 at ~1M points | ≤ 45 s | engine 17.2 s P95, 18.9 s max, P50 12.6 s (process wall 19.2 s max; load average 5-8); runs at load 5-12 gave 16.5-18.1 s P95 [15.5 s; the fixes add ~0.3 s of thickness tiles] |
| Peak RSS at ~1M points | ≤ 4 GiB | 1.04 GiB max |

All scenarios with defined truth (full tier): flange length P95 0.072 mm (max 0.380: a SPARSE far
edge outside its U of 0.22 mm, decided REVIEW: one of the three U misses, the others an
ADJACENT_BEND length on a 0.8 mm sheet whose skins are 0.063 mm apart from the CAD thickness, -0.175
vs U 0.134, and a bend-line position, +0.140 vs U 0.133, both PASS and in tolerance), angle P95
0.048°, bend-line position P95 0.067 mm, backgauge P95 0.084 mm (median U 0.26 mm: the K term).
Decisions over all MEASURED rows: 1209 PASS, 258 FAIL, 85 REVIEW. ADJACENT_BEND lengths P95 0.071 mm
[0.055], FREE_EDGE 0.074 mm [0.075].

What the review fixes moved (DEV only, `e3-1.0.0` → `e3-1.1.0`):

- The measured sheet thickness (§3.1) costs accuracy on DEV, where the CAD thickness is exact by
  construction and the measured one carries the ASSUMED setup misregistration: ADJACENT_BEND P95
  0.055 → 0.071 mm (full), complete-part flange P95 0.085 → 0.103 mm (regression). The skin-level U
  term keeps those rows inside U (median U at full visibility 0.108 → 0.116 mm). In return a sheet off
  its CAD thickness no longer leaves rows outside U.
- Four FREE_EDGE FAILs on far edges seen without their cut wall (injected short flanges on a TOP_ONLY,
  two SPARSE and one FULL scan, measured within 0.21 mm of truth) are now UNKNOWN(EDGE_NOT_SEEN):
  without a wall the visible end cannot prove a short side (§6). Coverage 99.0 → 98.8 %.
- The one-skin crease of E2-DEV-CR-014 stays unresolved (UNKNOWN), as before; a looser rule would
  have measured it to the wrong skin line (2.2 mm off).
- The backgauge K term from the stage geometry equals E1's diagnostic value on every DEV backgauge.

Before the review (the implementer's night): wall-aware datum rows and the pose-bias / erosion U terms
took TOP_ONLY bend-line positions from 80 % to 100 % inside U (errors up to 0.29 mm became ≤ 0.08
mm); the converged-slab skin levels and the local station skin search took TOP_ONLY coverage from
82 % to 96 % and SPARSE from 49 % to 81 %; the fold kinematics fixed the children of partially formed
bends; symmetric twins with the order prior fixed the symmetric stage-1 case; crease-ended flanges
removed the 8 extra-bend U misses (up to 12 mm).

## 10. Known limitations and open issues

1. **DEV only, synthetic only.** No QUAL seed was generated or read; G1 is gated on QUAL (E4). The
   scanner model is E2's ASSUMED model (amendment 4b's model card is not frozen), so the erosion,
   flare, duplicate-skin and setup-misregistration behaviour the U terms were sized on is assumed,
   not measured (`u_skin_level_mm`, the erosion scale, the 0.04 t stock tolerance). Real scans were
   not read by this chunk.
2. **Amendment 9's articulated ICP is not implemented as a joint refine (deviation, for owner / R1
   sign-off).** The datum pose comes from the datum's own plane, free edges and holes only, and each
   flange is fitted independently (hinge theta search, region-grown plane, fold kinematics), not by
   an ICP over the datum and all fitted flanges together. Rationale: a pose refined on the flanges
   absorbs their errors. A flange cut long or a bend line placed off would pull the datum frame
   toward it and hide part of the very bend-line position and flange-length deviations the engine
   must report; with the datum-only pose every flange is measured against a frame it cannot move.
   What amendment 9 protects (angle and flange length independent of the in-plane pose, bend-line
   position and heatmap withheld when the pose is degenerate) holds and is tested.
3. **Measured line locus.** `measured_polyline_mm` is the OML virtual sharp (the contract
   docstring). G2/G4 draw the visible-surface outer apex solid; the engine reports it in each row's
   `diagnostics` as `outer_apex_{start,end}_{x,y,z}`. Promoting it to a typed field is a C01/C08/E4
   decision.
4. **Measured thickness vs setup misregistration.** The two skins of a panel often come from
   different setups; their scanned separation is the real thickness plus that misregistration
   (0.036 mm σ on DEV). Using it (§3.1) is right for a sheet off its CAD thickness and costs ~0.02 mm
   of ADJACENT_BEND P95 on DEV, where E2 makes the CAD thickness exact. E2 never varies `t`, so the
   DEV suite cannot show the benefit; the 10 %-off test is the only evidence. A part whose panels'
   normals span opposite directions (a U-channel's walls) could separate the two effects; not done.
5. **Bare far edges are lower bounds.** A short-side FAIL on a far edge without a seen cut wall is
   UNKNOWN(EDGE_NOT_SEEN): 4 correct DEV FAILs (TOP_ONLY / SPARSE / one FULL) became UNKNOWN. On
   QUAL a backgauge error that shortens a bare far edge would be UNKNOWN, i.e. "not detected" by the
   G1 backgauge rule; on DEV every backgauge far edge with an error had its wall seen (21/21).
6. **Occluded datum edges bias the registration.** A datum free edge hidden by a clamp (4 mm) is
   observed as bare edge rows at the visible end, which pull the pose: on the smoke L-bracket with
   three datum holes the pose moved 4.6 mm and the holes read as filled (WRONG_PART_SUSPECTED
   instead of VALID). With every datum edge hidden and no holes the pose is degenerate (handled). A
   lower-bound treatment of bare datum rows (as for flange far edges) belongs to A1, with the real
   fixture geometry.
7. **Backgauge U with a shop-default K.** `sigma_K |dN/dK|` (contract) makes most backgauges across
   several strips REVIEW (median U 0.26 mm on the full tier; the runtime tier's one backgauge has
   U 0.79 mm). Correct but not decisive until K is calibrated.
8. **Stage models for formed sets that are not prefixes.** A reorder inside one machine move, or an
   out-of-order formed set E1 has no `S<k>-<tag>` stage for, gets no stage model: no heatmap and the
   backgauge is NOT_APPLICABLE_AT_STAGE (E1/C01 scope).
9. **General measurements of kind `points`** stay UNKNOWN(UNSUPPORTED_FEATURE): no scan
   correspondence rule for surface anchors is defined yet (C01/E1).
10. **Sparse scans.** SPARSE coverage is 79 % (LOW_DENSITY stations, bare short far edges); acute
    bends seen from one side only can stay UNKNOWN. Honest, but a denser scan plan is the remedy.
11. **Single-skin creases.** When only one skin of an unplanned crease's segment is seen and the
    panel contact cannot tell which, the split flange is UNKNOWN (1 of 8 DEV extra-bend cases).
12. **E1 symmetry test scope.** Ambiguity relies on E1's per-stage self-symmetry candidates; twins of
    a stage E1 did not test (a formed set with no compiled stage) fall back to every stage's
    candidates.
13. **Contract (C01).** A FLANGE_LENGTH side bounded by a not-yet-formed bend must be a *required*
    UNKNOWN(ADJACENT_BEND_NOT_MEASURED) (NOT_APPLICABLE_AT_STAGE is not allowed for that kind), so
    STAGE_IN_TOLERANCE is rarely reachable mid-sequence (11 of 22 IN_PROCESS regression results carry
    such a row). Allowing NOT_APPLICABLE_AT_STAGE there (the backgauge covers it) is a C01 decision.
    **Resolved 2026-10-19** (owner decision, amendment ENGINE_IO-A1, e3-1.11.0): such a side is now
    NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE); only an UNKNOWN neighbour keeps the UNKNOWN row.
14. **Real-scan risks no DEV case covers** (for E4/A1 on authorized scans before G2/G3): a table or
    fixture plane exported with the part lowers the explained share (a reviewer probe read 0.127:
    WRONG_PART_SUSPECTED); a filled or pinned datum hole is WRONG_PART_SUSPECTED by design; the
    mirror check runs only when the best pose looks suspicious, so chirality carried only by
    features off the datum is never tested.
15. **Timing on a shared machine.** The runtime numbers were measured while other agents ran (load
    average 12); a dedicated rerun belongs to E4's qualification.
16. **E2 observations (not E3 defects; reported for E2).** E2's bend-line truth is the offset
    projected on side A's outward direction only, so an in-plane registration error along the bend
    normal of a perpendicular flange is invisible to it; the extra-bend truth measures the split
    side to the crease's OML sharp (ADJACENT_BEND), while the recipe keeps its FREE_EDGE nominal; E2
    never varies the sheet thickness against the CAD.

## 11. Tests

- `tests/test_prod_engine_inspect_units.py` (45): scan ingest (PLY ASCII/LE/BE, STL binary,
  non-finite vertices, unreadable bytes), config hashing, refusal of unknown and removed keys, and
  of non-finite, out-of-range, non-integer and inconsistent values (also on direct construction),
  robust plane fit with outliers (this test found an IRLS centring bug), plane intersection and
  rotation helpers, normal averaging, two-skin peak detection, voxel cloud mapping, gross-outlier
  removal (and the voxel-key overflow it prevents), decode thresholds, dominant-level aggregation,
  the U budget's monotonicity, floors and bias terms; the fold kinematics against an explicit
  neutral-path construction (and the rotation-only model it replaces failing it); the station skin
  search when the panel plane is 0.27 mm off; converged-slab skin levels with wall clutter between
  the skins; wall-observed vs bare-skin datum rows and their weights; the erosion scale and the
  worst-case pose-bias bound; crease crossings only for creases along the bend; the tile thickness
  of a sheet 15 % off its CAD thickness (which the CAD-window skin search misses or mislocates); the
  single-skin and frame-height U terms; a kinked strip unfolded to its developed length (and the raw
  run flagged truncated, 50 mm short); bare far edges proving too long but never too short; the
  backgauge K term from stage geometry, deaf to edited diagnostics.
- `tests/test_prod_engine_inspect_cli.py` (32): E2 smoke-density cases (DEV seeds) compiled by the
  real E1 compile, run through the CLI: a complete part (valid, consistent, close to truth, files
  index-aligned, bend geometry equal to `from_result`), determinism and blindness to the scan name,
  context passthrough, a mid-sequence scan (states equal truth, backgauge on bend k, unformed bends
  never FAIL, never green under FINAL), declared-stage mismatch, injected errors never PASS, a
  mirrored scan never green and never published improper, a suspected inch scan (invalid, nothing
  rescaled) and the explicit `--unit-factor 25.4`, heavy occlusion (UNKNOWN, not an error), timeout,
  an unreadable scan, every ENGINE_IO §4 exit-2 code (tampered stage file, wrong hashes, missing
  files), a non-empty `--out`, usage errors, the library API agreeing with the CLI, a bend-pair
  general measurement; a symmetric U-channel at stage 1 (not ambiguous, the order-consistent twin,
  inside U), an extra-bend crease (reported, the split flange measured to the crease sharp inside U
  or UNKNOWN, never PASS), a TOP_ONLY scan (U covers truth, no decision contradicts truth) and a
  partially formed bend with formed bends beyond it (states equal truth). Added by the review fixes:
  a not-yet-formed bend kinked by 6 % of its sweep on the backgauge's path (unfolded, measured to the
  nominal within U, never FAIL); diagnostics edited under an unchanged hash (identical result); the
  budget expiring at four points of the articulated walk (TIMEOUT every time, never WRONG_PART) and
  in the metrology (VALID, UNKNOWN(TIMEOUT)); a degenerate in-plane pose from unusable datum edges
  (only position and heatmap withheld); a far edge occluded by 4 mm (UNKNOWN(EDGE_NOT_SEEN), measured
  on the full scan); a binary STL with a `solid` header (the PLY's decisions); a `points` general
  measurement (UNSUPPORTED_FEATURE) and a pair with an unformed bend (NOT_APPLICABLE_AT_STAGE); a
  3e38 sentinel vertex (dropped, otherwise identical); an unreadable input outside the engine (exit 3,
  INTERNAL_ERROR); a compiled `t` 10 % thinner than the sheet on a TOP_ONLY scan (measured `t` used,
  every row inside U; with the compiled `t` rows fall outside U); free edges marked unusable (not
  used); an ambiguous symmetric part (nominal lines on the decoded stage, only the datum labelled).

## 12. Review fixes (2026-10-17, `e3-1.0.0` → `e3-1.1.0`)

| Review finding | Fix |
|---|---|
| Backgauge far-edge run truncated by a kinked not-yet-formed bend (a 12.8 mm false FAIL) | island unfolded about each kink with the developed-length correction; truncated runs detected and dropped (UNKNOWN(AMBIGUOUS_SEGMENTATION) when too few stations remain) (§6) |
| Backgauge U read unhashed `diagnostics` | `sigma_K \|dN/dK\|` from hashed stage geometry only; diagnostics compared in the log (§6) |
| Budget cut inside the articulated walk read as WRONG_PART | an unfinished pose selection is TIMEOUT; metrology timeouts stay VALID with UNKNOWN(TIMEOUT) (§8) |
| One-skin handling: measured t unused, no thickness term in position U, term zeroed by `t_known` | datum-measured thickness used everywhere; thickness and frame-height terms per side and per share of inner-skin stations; skin-level term (§3.1, §6) |
| Ambiguous pose: nominal lines from S0, flange labels | lines of the chosen decode's stage; only the datum labelled (§8) |
| Occluded FREE_EDGE far edge gave a confident FAIL | bare far edges cannot prove a side short: UNKNOWN(EDGE_NOT_SEEN) (§6) |
| Amendment 9 joint articulated ICP not implemented, deviation unrecorded | recorded with its rationale (§10.2) |
| Dead config fields; NaN / negative / zero counts accepted | `pose_probe_*` and `coarse_peaks_per_family` removed (new config hash); every value range-checked on construction (§1) |
| A 3e38 sentinel vertex overflowed the voxel keys | gross outliers dropped at ingest and counted (§1) |
| `FreeEdgeV1.registration_usable` ignored | unusable edges never drive the registration (§3) |
| Exceptions outside `inspect_scan` exited 1 with a traceback | catch-all: `ERROR INTERNAL_ERROR`, exit 3 (§1) |
| Untested branches (degenerate pose, timeout in the walk, `points` and not-applicable general measurements, STL, diagnostics tamper) | tests added (§11) |
| Apex not shifted with the measured line; crease points unlabelled; config doc reference | shifted; label 65535; reference fixed |
| One-skin crease sharp resolved to the wrong line when the thickness moved (found while re-running the full tier) | the contact must lie clearly at one line (§6) |
| Contract-level: required FLANGE_LENGTH UNKNOWN next to an unformed bend; untyped outer apex | routed to C01 (§10.3, §10.13) |
