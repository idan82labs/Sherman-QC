# gb_3f9a9c — marker B

**Result: 2 bend lines (one joggle).**

Part: thin sheet (t ~1.46 mm), ~75 mm long. Its cross-section is a large rolled trough. One straight edge ends in a short Z-offset (joggle) and then a flat end plate with two holes and a lug.

## Bends
- **B1 (17|16, outer skin), 24.2 deg, 67.6 mm.** Joggle fold 1 runs between the flat end plate (patches 17/12) and the inclined joggle web (16/14). The own-tool section profile at three stations along the axis (t = 10, 25, 60) gives a normal angle of 93 deg on the plate and about 69 deg on the web. The change is localized over about 2 mm.
- **B2 (14|2, outer skin), 17.4 deg, 67.6 mm.** Joggle fold 2 runs from the web (69 deg) back to about 88-90 deg on plate 2, which is where the roll starts. Like fold 1, it is localized over about 3 mm.
- The plates on either side of the joggle are near parallel (93 vs 88-90 deg), with an offset of about 3.6 mm. Under GOLD_SET_V2 a joggle is 2 lines.
- Both lines are `ok` and `certain`, with fold_support 1.0 and no gaps. Each spans the full length of the straight flank (about 66-68 mm, about 100% coverage). In both close-ups the apex dot sits on the convex skin.
- On the first run B1 used inner-skin patches 12/14 (skin = inner). I swapped it to the twins 17/16, and it now reports outer.

## Rejected
- **The large curved body (patches 3-11, 13, 15, 18-28).** This is a constant-curvature roll: about 1.6 deg/mm along the arc, R ~36 mm, roughly 140 deg in total over about 95 mm of arc. I sampled it every 2 mm and found no localized change of slope. The free edge has no flat plate on the far side. Segmentation cut it into many planar strips, but those are facets of a rolled surface, not formed folds, so I marked no lines on it.
- The far free edge of the roll has a slight lip at the edge. That is an edge, not a bend.
- The two holes in the end plate and the hole near the roll edge are hole walls.
- The lug and notch at the end-plate corner are edge geometry.

## Partial?
The scan is closed along the length. There are scan dropouts on the convex underside of the roll (iso_under view), but they do not touch the joggle.

## Doubts
- If the adjudicator counts large-radius rolls as bends, the roll could be argued as one bend. I think it is not one: it has no localized fold and no second plate.
