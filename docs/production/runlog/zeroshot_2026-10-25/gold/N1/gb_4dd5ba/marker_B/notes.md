# gb_4dd5ba — marker B

**Result: 26 bend lines.** All are `ok` with outer-skin apex. B16 and B19 are flagged `uncertain` by the tool; see Doubts.

Part: a flat perforated plate (t ~0.97 mm, ~450 x 94 mm) with formed legs, an end flange, window tabs and window lips. I used my own frame: L = along the plate, W = across, N = plate normal (+N = top).

## Bends
Every bend listed below is a localized formed fold of at least 10 deg, shown in my own L/W/N sections (sample at L=-418) and in each closeup. All lines run parallel to the fold.

**Legs: 6 legs, 3 lines each = 18.** The legs sit in pairs at L = -418, -348 and -144, one on each long edge. Each leg is a separately formed tab about 29-30 mm wide, with three folds:

| fold | angle | lines |
|---|---|---|
| plate to leg | ~90 deg | B1, B4, B7, B10, B13, B16 |
| knee, turning inward | ~60 deg | B2, B5, B8, B11, B14, B17 |
| hairpin foot | ~115 deg | B3, B6, B9, B12, B15, B18 |

- **Left-leg hairpins (B3, B9, B15).** The upper face of the knee web is barely scanned (about 11 points), so I seeded that side with `plane_from_points`. The fitted plane catches part of the arc, so the reported angle is 85-93 deg instead of about 116 deg. The apex dot still sits on the outer skin at the hairpin crest in all three closeups, and the direction matches the other legs.
- **B13 and B16** (top folds of the L=-144 legs). The plate's top skin is missing near the right end of the plate. I set the extent to the leg width (extent_t from the leg patch), which gives fold support of 1.0 and 0.97.

**End flange B19 (2|3), 90 deg, 80 mm.** The plate's top skin is not scanned within about 30 mm of this fold. I used margin_mm 32 so the fit reaches plate 2, and set extent_xyz to the flange width (W 85-165). The tool flags it uncertain ("support < 100 points on a side"). The closeup apex sits on the outer corner.

**Window tabs and lip, 3 lines:**
- B20 (2|8), tab at L=-306, 90 deg down, about 57 mm deep and 37 mm wide.
- B21 (2|19), tab at L=-463, 90 deg down, 62 mm wide.
- B22 (2|21), lip at L=-202 on a window edge, 90 deg down, about 4.5 mm deep and 69 mm long. It has both skins and a radius, so it is a formed lip, not a hole wall.

**Z-tab at L=-365, 2 lines:**
- B23 (2|32): plate to web, 90 deg.
- B24 (41|31): web to foot, 90 deg.
- The foot is offset about 11 mm. Patch 41 is short, so I copied the extent from B23; fold support is 0.92.

**Joggled window lip at L=-135, 2 lines (joggle = 2):**
- The lip is about 38 mm long and stepped down about 3 mm (3t) through a steep S (web ~75-80 deg).
- B25 (48|web top skin) and B26 (web bottom skin|36). Each web side is seeded with plane_from_points, radius 15, grow.
- Patch 48 is the only top-skin piece of the plate there. B25's extent is copied from B26.

## Rejected
- **Hook-tip curls.** There is a curl of about 1 mm at the tip of each foot. That is edge rounding, not a fold.
- **Hole and slot walls.** Many round holes and rectangular windows, all hole walls.
- **Square patch near L=-500.** It shows bottom skin in the top view, but that is only a top-skin scan dropout. My section shows the plate is flat there.
- **Patch 49.** A sliver on a leg fold arc.

## Partial?
Not partial in geometry. Top-skin coverage is missing near the right end (around L=-131 to -100) and in a patch near L=-500. The bottom skin is complete.

## Doubts
- **Joggled lip (B25/B26).** The web is very short (~1-2 mm), so the two lines sit about 1 mm apart. If the adjudicator reads this as an emboss or offset rather than a joggle, it would be 0 lines instead of 2.
- **Left-leg hairpin angles (B3/B9/B15).** The angles are understated, but the line positions are good.
- **B19 support and B16.** B19 has low support on the plate side. B16 needed a widened zone. Both closeups look correct.
