# gb_4dd5ba - marker A

**Result: 26 bend lines.**

## What the part is
A flat perforated base plate (t about 1 mm, about 450 x 94 mm; plate outer skin = patch 2). Formed features:
- an end flange,
- 3 lanced tabs bent up at window edges,
- 6 side legs (3 per long side). Each leg ends in a hook: the leg, a bend of about 60 deg into a foot about 13 mm long, then a bend of about 116 deg into a lip about 5 mm long.
- a lanced offset (Z) tab,
- a small edge joggle (about 3 mm offset) along one window edge near the end flange.

All folds were checked in the close-ups. The dot sits on the outer-skin arc in every one, and each line covers the full width of its fold.

## Bends (index in lines.json / closeup number = index + 1)
- **B1 (0)**: plate to end flange (4), 89.5 deg, 79.9 mm. Near this flange the plate's outer skin and the outer arc are **not scanned**; only the inner plate skin reaches the fold. The plate side is therefore a plane_from_points fit on the inner skin (seed -67.6, -108.0, 257.9) and the pair classifies as inner. The close-up shows the dot at the top of the corner. Confidence set to uncertain.
- **B2-B4 (1-3)**: plate (2) to lanced tabs 8, 19 and 21 (21 is saddle-shaped; 33/34 are its other skin). Each about 90 deg, lengths 36.7, 61.1 and 69.6 mm.
- **B5-B10 (4-9)**: plate (2) to the 6 legs (6, 7, 10, 12, 15, 17). Each about 90 deg, about 30 mm (the leg width). For B5 (leg 6) and B6 (leg 7), the plate's outer-skin patch stops short of the leg, so extent_t was set to the leg width. Fold support is 1.0.
- **B11, B13, B15, B17, B19, B21 (10, 12, 14, 16, 18, 20)**: leg to foot, about 58-61 deg, about 29-30 mm.
- **B12, B14, B16, B18, B20, B22 (11, 13, 15, 17, 19, 21)**: foot to lip, about 116-118 deg bend (about 62-64 deg included), about 25-29 mm. For the 3 left-side legs (feet 29, 27, 28) only one foot skin is scanned. That skin is the inner skin at this fold, so a plane_from_points on the same skin was used with the lip's matching skin (46, 43, 40), and lines.json labels these pairs "inner". The close-ups B14, B20 and B22 still show the apex dot on the convex outer arc of the hook.
- **B23, B24 (22, 23)**: lanced offset (Z) tab. Plate (2) up to the riser (32/41), 90 deg, then the riser back to the top (31/35), 90 deg, about 38 mm each. These are 2 separate formed folds. For B24, extent_t was set to the tab width.
- **B25, B26 (24, 25)**: edge joggle about 3 mm deep along the window edge next to the end flange (strip 36/42, about 4 mm wide, about 38 mm long). The joggle wall is not segmented, so plane_from_points (radius 7, grow false) was fitted on the wall's outer face for each corner. Lower corner: plate bottom skin 48 to wall, 75.6 deg. Upper corner: strip top 36 to wall, 81.2 deg. extent_t was set to the strip length. Both are outer skin, fold support 1.0. Confidence set to uncertain because the wall is only about 1.5 mm tall.

## Rejected
- Patch 49: a thin strip at the arc of the leg-10 fold.
- Patch 48 on its own: part of the plate's bottom skin.
- Holes and windows (19/20, 8/9, 21, 31-35 sit in windows; the windows themselves are not bends).
- A small flat scan artefact near the left end, checked in `sweep/v_sq.png`: flat, no fold.
- The window outline edges.

## Partial?
The scan is mostly complete. The outer skin is missing near the end flange (B1).

## Doubts
1. **B1**: the line comes from inner-skin planes. The true outer-apex line may sit about 1 mm outboard.
2. **B25/B26 joggle**: shallow (3 mm) and short-walled. It could be read as an emboss or coined edge instead of a joggle. If so, drop both lines.
3. **Hook lips (B12-B22, even numbers)**: the lips are only about 5 mm long and have a fairly large radius. I count them as formed folds of 10 deg or more between two plates.
