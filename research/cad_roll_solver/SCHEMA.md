# `cad_roll_nominal.json` schema (`cad_roll_nominal_v0`)

This file is written by the CAD roll solver (`research/cad_roll_solver/`, research prototype, 2026-10-25). It holds the nominal **roll** features of one STEP file. It also lists E1's bends next to them, so a part with both bends and rolls reads as one recipe.

**Consumer.** The scan-side roll measurement in `research/zeroshot/roll_measure/` (see "Alignment with the scan side" below).

**Status.** DEV research output. It is not a production contract. The design is in `docs/production/runlog/cad_roll_solver/DESIGN.md`.

## Producer and conventions

```
.venv/bin/python -B -m research.cad_roll_solver.run --out DIR [--name NAME] PART.step   # through the geometry slot
```

- **Reader.** The STEP is read by E1's own reader (`domains/production/engine/compile/brep.py`, OCP). E1's flange graph (`flange_graph.py`) is built from it. Both are imported read-only and unchanged.
- **Frame.** `cad_model_mm`: the STEP frame with no transform applied. Lengths are in mm and angles in degrees. Values are rounded to 1e-6.
- **Axis direction.** `canonical_direction`, the same convention as E1: the dominant component is positive.
- **Angles about a roll axis.** The angle is `atan2((p - axis.point)·e2, (p - axis.point)·e1)`, using `frame.e1` and `frame.e2` from the record, with `e1 × e2 = axis.direction`. A roll's **start** and **end** are the two angular extremes of its skins. The sweep runs counter-clockwise about `axis.direction` from start to end.
- **Skins.**
  - The **inner skin** is the concave one (radius `r_inner_mm`). The **outer skin** is the convex one.
  - Thickness is the normal offset of the two skins: `(r_outer - r_inner)·cos(semi-angle)`. For a cylinder this is just `r_outer - r_inner`.
- **Cones.** A cone has `r(z) = r0 + k·z` along the axis. Every radius in the record is given at the axial middle of the roll. `taper` gives `k` and the radii at both axial ends.

## Top level

| Field | Meaning |
|---|---|
| `schema` | `"cad_roll_nominal_v0"` |
| `rule_version` | The bend/roll rule text (see "The bend/roll rule"). |
| `source` | `file` and `sha256` of the STEP, and `name`. |
| `e1` | What E1 makes of the part:<br>• `flange_graph_error` and `part_model_error`, for example `DATUM_PANEL_UNDETERMINED: ...`;<br>• `thickness_mm`, `panels` and `graph_bends`;<br>• `features`: E1's UNSUPPORTED records (kind, reason, faces). |
| `thickness` | `t_mm` and `source`. The source is `E1_FLANGE_GRAPH`, or, when E1 has no thickness, the area-weighted median of the paired skins. |
| `summary` | Counts: `rolls`, `junction_bends`, `e1_bends`, `solver_bends` and `cones`.<br>`all_checks_pass`.<br>`roll_faces_overlap_e1_bend_faces`: must be `[]`. |
| `rolls[]` | One record per roll (see "Roll record"). |
| `junction_bends[]` | Bends E1 cannot place in its graph because one end is tangent to a roll. Same record shape as a roll; `kind` = `BEND`, `subtype` = `JUNCTION_BEND`. |
| `e1_bends[]` | Every E1 flange-graph bend, verbatim:<br>• `graph_bend_id`, `class` (E1 UNSUPPORTED class or null), `parent` and `child` panel ids;<br>• `inner_radius_mm`, `thickness_mm`, `sweep_deg` and `length_mm`;<br>• `oml_virtual_sharp`, the E1 bend line;<br>• `faces`;<br>• `solver_cross_check`: the solver's own Ri, t and sweep differences for that bend. |
| `solver_bends[]` | The part's bend list after the rule. It holds E1's graph bends except those E1 itself classed ROLLED (a), plus the junction bends. `source` is `E1` or `ROLL_SOLVER_JUNCTION`. |
| `graph` | E1's flange graph with rolls added (see "Graph view"). |
| `unclassified_paired_features[]` | Paired coaxial skins that are neither an E1 bend, nor a roll, nor a junction bend, for example the formed collar of an extruded hole. They are reported with their E1 role and never re-classified. |
| `unpaired_large_curved_skins[]` | Curved faces with `R > 10 t` that have no partner skin. These are reported, not rolls. |
| `large_radius_curved_faces[]` | Screening census of every curved face with `R > 10 t`: its radius, `r_over_t`, axial extent, what the solver made of it, and E1's role for it. |
| `curved_faces` | Counts of cylinder and cone faces. |

## The bend/roll rule (`ROLL-RULE-0`)

The rule is evaluated on **paired** features. A paired feature is a concave and a convex skin that are coaxial (E1's tolerances: 0.25° and 0.02 mm, plus 1e-5·R), share the same taper, are offset by `0.05 ≤ t ≤ 30 mm`, and overlap angularly. They must also overlap axially by more than 1.5 t, which rejects hole walls, whose axial extent is t.

1. **E1 graph bend.** An E1 flange-graph bend keeps E1's class and stays a bend (including HEM and OTHER). The exception is a bend that E1 itself classed **ROLLED (a)** (`Ri > 10 t`): that one becomes a roll, with E1's two panels as its run-out flanges.
2. **Other large-radius features.** Any other paired feature with `Ri > 10 t` is a **roll**, whether a cylinder or a cone. This includes closed tubes and rolls without planar flanges, which E1 cannot represent at all.
3. **Junction bends.** A paired feature with `Ri ≤ 10 t` that is not an E1 graph bend but is smoothly tangent to a roll is a **junction bend**: flange-to-roll or roll-to-roll.
4. Everything else stays as E1 made it.

The threshold `10 t` is E1's frozen `ROLLED_RI_OVER_T` (`unsupported.py`, rule 2). It is reused, not re-tuned. `Ri = 10 t` exactly is a bend; the synthetic case `L_control_ri_10t` covers this.

## Roll record (`rolls[]`, `junction_bends[]`)

| Field | Meaning |
|---|---|
| `id` | `R<n>` for a roll, ordered by size. `J<n>` for a junction bend. |
| `kind`, `subtype` | `ROLL` with `ROLL` or `E1_ROLLED_A_GRAPH_BEND`; or `BEND` with `JUNCTION_BEND`. |
| `surface`, `is_cone` | `cylinder` or `cone`. The **cone flag**. |
| `axis` | `point_mm`: on the axis, at the axial middle of the roll. `direction`: a unit vector. `segment`: the axis over the roll's axial extent. |
| `frame` | `e1` and `e2`, which define the angles above. |
| `r_inner_mm`, `r_outer_mm` | Inner and outer skin radius, at the axial middle. |
| `thickness_mm` | The skin offset `t`, measured normal to the skins. |
| `r_inner_over_t`, `r_mid_mm` | `Ri / t` and the mid-surface radius. |
| `sweep_deg` | Angular extent of the union of both skins: 360 minus the largest angular gap. It is measured to the skins' own boundary. At a free end with edge rounds, the round is not counted. |
| `sweep_inner_deg`, `sweep_outer_deg` | The sweep of each skin. |
| `start_angle_deg` | Angle of the start end, in `frame`. |
| `closed` | True for a full 360° tube. A closed roll has no run-outs. |
| `axial_length_mm` | Extent along the axis, as the union of both skins. Per-skin values are in `axial_length_inner_mm` and `axial_length_outer_mm`. |
| `chord_inner_mm`, `chord_outer_mm` | `2 R sin(sweep / 2)` on that skin: the straight distance between the two run-out lines. |
| `opening_mm` | Equal to `chord_inner_mm`. |
| `opening_gap_deg` | `360 - sweep`. |
| `arc_length_mid_mm` | Mid-surface arc length, `r_mid · sweep`. |
| `taper` | `dr_dz`, `mm_per_m`, `semi_angle_deg`, and `r_inner_at_axis_ends_mm` / `r_outer_at_axis_ends_mm`. All are 0 or constant for a cylinder. |
| `convexity` | The inner skin is concave and the outer skin convex. `bulge_direction` is the unit radial from the axis to the angular middle of the sweep, which is the convex side. |
| `faces` | B-rep face indices of the inner and outer skins. |
| `e1` | `graph_bend_id` and `class` when the roll is an E1 ROLLED (a) graph bend. `face_roles`: E1's role for the skin faces, for example `wall` for the roll shells on today's DEV parts. |
| `runouts[]` | Two entries, `which` = `start` and `end`. Empty when `closed`. See "Run-out entry". |
| `neighbours` | Ids of the neighbours at the run-outs: E1 panel `P<n>`, solver flange `F<n>`, roll `R<n>`, junction bend `J<n>` or E1 bend `G<n>`. |
| `stations[]` | The nominal arcs at 25, 50 and 75 % of the axial extent: `z_mm`, `centre_mm`, `r_inner_mm`, `r_outer_mm`, and the start angle and sweep. |
| `checks[]`, `checks_pass` | See "Checks". |
| `interior_generator_edges` | Straight edges parallel to the axis that lie inside the sweep. For example, a step in the outline where part of the length starts later. |

### Run-out entry (`runouts[i]`)

- `which`: `start` or `end`.
- `angle_deg`: the angle of this end.
- `end_type`:
  - `TANGENT_RUNOUT`: both skins continue smoothly into a flange, bend or roll;
  - `FREE_END`: both skins end at a cut edge, for example a sharp wall, an edge round or no generator edge;
  - `MIXED`: the two skins differ.
- `class`: the class shared by both skins, or `inner|outer`.
- `inner` and `outer`, one per skin, each holding:
  - `start_mm`, `end_mm`: the **run-out line** of that skin. It sits at the end angle and radius, and covers the axial span of the actual tangent edges;
  - `length_mm`;
  - `class`, one of:
    - `FLANGE`: a planar skin, either an E1 panel or a solver flange;
    - `BEND`;
    - `ROLL`;
    - `FREE_EDGE`: a sharp edge to a wall;
    - `EDGE_ROUND`: an unpaired small round of radius ≤ 1.5 t, then a wall;
    - `WALL_TANGENT`;
    - `CURVED_UNPAIRED`;
    - `FREEFORM`;
    - `NO_GENERATOR_EDGE`;
  - `neighbour_ids`, `neighbour_faces`, `edges`;
  - `coverage`: the fraction of the skin's axial extent covered by the edges. Less than 1 means the run-out exists only over part of the length.

The run-out line is the tangent line between the roll skin and its neighbour. It is the roll's counterpart of E1's `inner_tangent_*` and `outer_tangent_*` bend loci.

### Checks

Each check is `{check, status: PASS|FAIL|NOT_APPLICABLE, value, tol, detail}`.

| Check | Rule |
|---|---|
| `SKINS_COAXIAL_ANGLE_DEG`, `SKINS_COAXIAL_OFFSET_MM` | Every skin face's axis lies within 0.25° and 0.02 mm (+1e-5·R) of the roll axis. |
| `THICKNESS_CONSISTENT_MM` | The roll's t is within 0.05 mm + 2 % of an **independent** part thickness: E1's thickness, else the median of the other paired features. When neither exists it is NOT_APPLICABLE. |
| `SKIN_SWEEPS_AGREE_DEG` | Inner and outer sweep agree within 1°. |
| `SKIN_AXIAL_OVERLAP_FRACTION` | The skins overlap over at least 90 % of their union along the axis. |
| `CONE_SEMI_ANGLES_AGREE_DEG` | Cones only: the semi-angles of all faces agree within 0.25°. |
| `RUNOUT_TANGENT_MM` | Every smooth run-out is tangent to its neighbour, within 0.02 mm (E1's `TANGENCY_TOL_MM`):<br>• a plane: its outward normal equals the skin normal (inner skin: the negated normal), and the generator lies in the plane;<br>• a cylinder: parallel axis, and the axis distance equals `R1 ± R2`.<br>NOT_APPLICABLE when both ends are free. |
| `SWEEP_VS_RUNOUT_PLANES_DEG` | When both ends run out to planes: the dihedral of the flange planes equals the skin sweep within 1° (E1's `SWEEP_CONSISTENCY_TOL_DEG`). |

## Graph view (`graph`)

- **Nodes.** Each node has an `id` and a `type`:
  - `PANEL`: an E1 panel;
  - `FLANGE`: a solver flange, which is a planar skin pair at a roll run-out and not an E1 panel;
  - `ROLL`.
- **Edges.** Each edge has an `id`, a `type` and `ends`:
  - `BEND`: an E1 bend, minus E1 ROLLED (a) bends, which are replaced by their roll;
  - `RUNOUT`: a roll tangent directly to a flange or another roll;
  - `JUNCTION_BEND`.
- **`components` and `coherent`.** `coherent` is true when the view is a single component and every edge has two ends.
- **Rolls and bends together.** A part with both, such as 28870000, gives one coherent graph: `P0 -G1- P1`, `R1 -J1- P0`.

## Alignment with the scan side (`research/zeroshot/roll_measure/measure.py`)

| Scan measurement | CAD field |
|---|---|
| `axis.point_mid`, `axis.direction` | `axis.point_mm`, `axis.direction`. Both are in their own frame; comparing them needs E3's registration pose. |
| `R_inner.mean`, `R_outer.mean` | `r_inner_mm`, `r_outer_mm` (cone: at the axial middle; use `taper` to evaluate them at the scan's stations). |
| `thickness.t_used_mm` | `thickness_mm` |
| `sweep_deg.mean` (main skin), `sweep_by_skin_deg` | `sweep_deg`, `sweep_inner_deg`, `sweep_outer_deg` |
| `chord_runout_mm` (between the arc ends on the main skin, inner preferred) | `chord_inner_mm` (`opening_mm`), or `chord_outer_mm` when the scan measured the outer skin |
| `runout_lines.start` / `.end` (line fit of the arc end points over the stations, `type_majority` tangent or not) | `runouts[which].inner` / `.outer` lines, with `end_type`: `TANGENT_RUNOUT` corresponds to tangent, `FREE_END` to not. Start/end ordering differs per frame, so match the ends by position after registration. |
| `length_mm` (region length along the axis) | `axial_length_mm`. Where the run-out covers less than the full length (`coverage` < 1), use the run-out `length_mm`. |
| `taper_mm_per_m` | `taper.mm_per_m` |
| cone flag | `is_cone` |
| `ends` per station (tangent or not) | `runouts[*].end_type` |

**Two rules for the consumer.**

- **Radius threshold.** The scan lane accepts rolls from `R ≥ max(12 mm, 5 t)`, but the CAD rule is `Ri > 10 t`. So a scan "roll" with Ri between 5 t and 10 t is a CAD **bend**. Look the feature up in `solver_bends` before reporting it as a roll.
- **Frame.** Invariant values can be compared without a pose: R, t, sweep, chord, length and taper. Axis and run-out positions need the registration pose.
