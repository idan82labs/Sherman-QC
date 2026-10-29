# bench_labeling: per-bend overrides

Use these when the default `lines` result is cut short or a flange has no patch.
They never change a bend that does not use them. Two benchmark scans were re-run
without overrides: every existing per-bend key and every render hash came out
identical, and the new keys are additive only.

All coordinates are `source_ply_raw`, in mm.

## 1. Support-zone margin

`LineParams.margin_mm` (default 3.0) sets the support-zone radius
`D = (Ri + t) * max(1, tan(bend/2)) + margin`, which also controls which points
the plane refit uses (`D + fit_extra_mm`). You can change it in two places:

- Default for the whole run: `python -m research.bench_labeling lines ... --margin 5`
- One bend: `"margin_mm": 5.0` (range 0 to 50). This wins over `--margin`.

lines.json records the margin used on each bend (`margin_mm`). It also records the
run default under `line_params.margin_mm`, and per-bend overrides under
`overrides`.

## 2. Explicit extent along the fold

### What t means

Every ok bend in lines.json now reports these fields:

| key | meaning |
|---|---|
| `line_origin` | `[x,y,z]`. This is the reference point. It is the foot of the perpendicular from the centroid of the refit flange points onto the plane-intersection (sharp) line. |
| `direction` | Canonical unit direction of the line. Its largest-magnitude component is positive. |
| `t_range` | `[t0, t1]` actually used. `sharp.start = line_origin + t0*direction` and `sharp.end = line_origin + t1*direction`. |
| `t_range_auto` | What the automatic clipping produced, or `null` if it found no overlap. |
| `extent_source` | `auto` or `override` |

So `t = (X - line_origin) . direction`. The apex line is the sharp line shifted along
`bisector`, which is perpendicular to `direction`, so the same `t` applies to both loci.

The plane fit does not depend on the extent. That means `line_origin` and
`direction` from an automatic run are exactly the ones an `extent_t` run will use,
provided the sides and `margin_mm` are the same. Changing the sides or
`margin_mm` moves `line_origin`. In that case, read it again from a run with the
new settings, or use `extent_xyz`.

### Examples

To extend a fold across a hole or slot to the fold's real ends:

```json
{"labeler": "claude-vision-ai-v1",
 "bends": [
   {"patches": [4, 9], "extent_t": [-25.9, 34.1],
    "note": "slot at t 14..20 splits flange 9; fold runs full width"}
 ]}
```

Or give the same thing as two 3D points (for example, picked from a render or a
cross-section). Each point is projected onto the line and the projections are
sorted:

```json
{"bends": [
   {"patches": [4, 9], "extent_xyz": [[-212.4, 301.0, -280.2], [-170.9, 288.3, -301.7]]}
 ]}
```

You can't use `extent_t` and `extent_xyz` together on one bend. `t1` must be
greater than `t0`, and the length must be at least `min_length_mm` (2 mm).

### Validation (an override is never accepted silently)

For every ok bend, the tool computes the fold support of the final `[t0, t1]`
from the prep's downsampled cloud (`points_ds.npz`). It uses all points, not
just the chosen patches:

- **Stations:** there are `n = max(1, round(t1 - t0))` stations of about 1 mm,
  placed at step centres. Each station uses a window of +-`max(0.5, 0.75*voxel)` mm.
- **Points counted:** points within `Ri + t + 1` mm of the label line (the apex
  line when an apex was found, otherwise the sharp line).
- **Two sides:** points are split by the bisector plane into a flange-A side and a
  flange-B side. The test is `w . unit(side_dir_a - side_dir_b) > 0` or `< 0`.
- **Supported station:** both sides have at least `fold_support_min_points` (3)
  points. A station also counts if the fold-arc band (`|w.e| <= max(0.5,
  0.5*voxel)` mm) has at least 3 points that straddle the bisector plane. A lone
  flange edge reaching the plane from one side does not count.

These keys are recorded on each bend:

- `fold_support_frac`, recorded for all bends and informational for auto bends.
- `fold_support_gaps_t`, the unsupported `[t_a, t_b]` intervals.
- `override_support_frac`, which equals `fold_support_frac` for an override and
  is `null` for auto.

If `override_support_frac < 0.8` (`LineParams.fold_support_min_frac`), the tool
does three things:

- It sets the bend's `confidence` and `auto_confidence` to `uncertain`. This
  happens even if the labeler wrote `"confidence": "certain"`, and a
  "downgraded" reason is added in that case.
- It adds a reason giving the support fraction and the gaps.
- It adds `extent override (fold support X)` to the bend note in the written
  label.

A hole that takes up less than 20% of the fold passes, and the line spans it. An
extent that runs past the end of the part fails.

With an override, the patch-pair `coverage < 0.60` check is replaced by the
fold-support check. `coverage` is still reported. All other automatic checks
still apply: apex found, support of at least 100 points, plane rms, mixed skin,
and widened zone.

## 3. A flange side as a raw plane (segmentation missed the flange)

A side can be an object instead of patch ids:

```json
{"bends": [
   {"patches": [7, {"plane_from_points": {"seed_xyz": [-206.2, 294.7, -291.0],
                                            "radius_mm": 10.0}}],
    "note": "flange B not segmented (too small); seed picked on its outer skin"}
 ]}
```

| field | required | meaning |
|---|---|---|
| `seed_xyz` | yes | A point on the flange skin you mean. It must be within max(2 mm, 3 voxels) of a downsampled point, or the bend fails. |
| `radius_mm` | yes | Radius of the ball around the seed used for the plane fit, from 1 to 500. Keep it inside the flange: away from the fold arc, holes and other flanges. |
| `grow` | no, default `true` | Flood-fill the fitted plane over connected downsampled points so the whole flange is used, not only the ball. |
| `tol_mm` | no, default `min(0.6, 0.45 t)` | Distance-to-plane gate. |

The fit works as follows:

1. Start from the ball points whose normal is within 20 deg of the normal at the
   point nearest the seed, and that lie within `0.45 t` of that point's tangent
   plane. This keeps one skin only.
2. Do a trimmed plane fit on those points.
3. Refit on the ball points that lie within `tol_mm` of the plane and have
   consistent normals.
4. Grow the region if `grow` is true.
5. Map to raw points the same way patches are mapped: nearest downsampled point
   within 1.5 voxels, plane gate, and normal gate on oriented scans.

The skin side (outward) comes from scan normals when the scan has oriented
normals. On unoriented clouds it comes from the other skin, found at about `t`
inside the ball. If neither is available, skin is `unknown` and the bend is
marked `uncertain` with a reason.

lines.json records the following for each bend:

- `plane_sides.{a|b}` holds the normalised spec plus `fit`: `normal`, `point`,
  `rms_mm`, `tol_mm`, `seed_surface_dist_mm`, `ds_ball_points`, `ds_inliers`,
  `ds_region_points`, `raw_points` and `outward_source`.
- `patches` shows such a side as the string `plane@x,y,z/r<radius>`.

In the close-up the plane side is drawn in red for side a or blue for side b.

You can combine all of these on one bend:

```json
{"bends": [
   {"patches": [[3, 12], {"plane_from_points": {"seed_xyz": [10.0, 4.5, 88.0], "radius_mm": 15,
                                                 "grow": false, "tol_mm": 0.4}}],
    "margin_mm": 5.0, "extent_t": [-40.0, 42.5], "confidence": "auto"}
 ]}
```

## 4. Apex on a bowed fold (`apex_mode`)

By default (`"envelope"`) the apex offset is one number for the whole line: the
median of the nearest-surface heights along the bisector within 0.45 mm of their
2nd percentile. On a straight fold this is the apex. On a fold whose outer arc
bows along its length (a warped flange edge), it is the outer envelope: the line
touches the skin where the bow comes closest and sits in free air elsewhere.

`"apex_mode": "station_median"` (per bend) measures the nearest-surface height in
every 1 mm station instead (median of the points within 0.3 mm of that station's
minimum, at least 3 points per station) and uses the median over stations. A
straight line on a bowed fold then runs through the middle of the apex. On a
straight fold both modes agree within about 0.1 mm. The plane fit, direction,
extent and validation are unchanged. lines.json records `apex_mode` and
`overrides.apex_mode` only on bends that use it; `apex_points` is then the number
of stations.

```json
{"bends": [{"patches": [2, [3, 11]], "apex_mode": "station_median"}]}
```

## Workflow

1. Run `lines` without overrides. Read `line_origin`, `direction`, `t_range_auto`
   and `fold_support_gaps_t` for the bend that was cut short.
2. Find the real fold ends on the multi-angle renders and cross-sections.
   Convert them to `t`, or give them as `extent_xyz`.
3. Re-run `lines`. Check that `extent_source` is `override`, that
   `override_support_frac` is at least 0.8, and that the gaps are where the holes
   actually are. Then check the close-up.
4. Misspelled keys are rejected with a hint: `margin`, `extent`, `extent_mm`,
   `t_range`, `plane` and `extent_points`.
