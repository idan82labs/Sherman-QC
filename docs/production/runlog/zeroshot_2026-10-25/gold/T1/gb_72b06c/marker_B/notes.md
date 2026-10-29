# gb_72b06c, marker B

Long W-profile channel. Full scan with both skins captured, t about 1.5 mm. 7 bends. Confidence: medium-high.

Outer-skin sides were chosen per bend. Where the outer skin was not segmented, I used plane_from_points on the back skin.

Profile bends, each running the full length (about 286 mm):
- B1 [1,4]: plate 1 to flange 4 (valley).
  - apex_mode station_median; the default envelope picked up see-through noise at the window.
  - The window in plate 1 at one end cuts near the fold (fold-support gap at t 81-144). It is the same two plates, so 1 line.
- B2 [back-skin plane of 4, 11/14/27]: flange 4 to strip 5/6 (peak).
  - The outer skin is the back side, which is only partly segmented. I used extent_xyz taken from the inner-skin run's fold ends.
  - Fold support is 0.95, with the gap at the square cut-out. station_median apex sits on the outer arc in section.
- B3 [2,5/6]: strip 5/6 to plate 2 (valley). The square cut-out splits 5/6; it is the same two plates, so 1 line.

Tabs bent 90 deg off plate 2 (outer side = the back skin of 2, via plane_from_points; extents from the inner-skin run):
- B4: side tab 7/8 with holes, 33.5 mm.
- B5: end tab 12/16, 19 mm.
- B6: end tab 13/15, 18 mm.
  - 12 and 13 sit side by side on the left end edge. They are about 2 mm apart in fold position and separated by a slit, so I marked them as two separate folds.
- B7: corner tab 17/18 on the slanted edge, 19 mm.

Doubts:
- B5 and B6 use a small back-skin region near the plate edge, so the plane fit is tilted. The reported bend angle is about 78 deg (true is about 90).
  - The line direction agrees within 0.1 deg with the inner and mixed-skin variants, and the position within 0.3-0.7 mm. The apex offset equals the expected outer value.
- 12 and 13 could be read as one fold split by a relief. I kept them as 2 because they are offset by about 2 mm.

Rejected:
- 13-17 meet at an angle across a slit, with a sharp gap in section. That is a corner between two separate tabs, not a fold.
- Strips 25, 26 and 27 are edges or arc facets.
- 20 and 24 are small back-skin pieces of plate 1.
