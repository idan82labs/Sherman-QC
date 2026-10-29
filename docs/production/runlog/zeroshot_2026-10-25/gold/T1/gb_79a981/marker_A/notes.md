# gb_79a981 — marker A

**Result: 4 bends.** Confidence: medium.

- Part: open box-like bracket, t ~1.9.
  - Top plate 1/2: 255 x 203, with a large window.
  - Side wall 3/4: ~137 mm.
  - End frame 6/7: full width, with a large window.
  - Bottom return flange 9/10 on the frame.
  - End flange 5/8 on the side wall.
- Bends:
  - B1: top 1/2 to side wall 3/4. 184 mm, the full wall length. Large-radius fold: the inner radius is ~10 mm and the segmentation made 8 facet strips (11–18). Needs margin_mm 12. The section shows a smooth 90 deg arc, and the dot is on the outer apex.
  - B2: top 1/2 to end frame 6/7. 272 mm.
    - extent_t override: the auto run clipped it to 96 mm.
    - My own raw-cloud sections at t-70..+186 show the formed arc continuously across the whole frame top bar. At one station a window cuts one leg.
  - B3: end frame 6/7 to bottom return flange 9/10. 276 mm.
    - extent_t override: sections at t-60..+190 show the arc along the full bottom bar.
    - Uses station_median, because material at 2 stations (t~70, ~170, a tab or edge) pulled the envelope apex ~1.6 mm off the arc.
  - B4: side wall 3/4 to end flange 5/8. 140 mm, the full wall height.
- Rejected:
  - The corners [1,5] (top to end flange) and [3,7] (side wall to frame; included 86 deg). Their sections show a gap or a sharp joint, not an arc.
  - Thin strips 19–26: edges.
  - Window and hole walls.
- Not partial.
- Doubts:
  - The B2/B3 extents depend on my reading of the raw sections. Check override_support_frac in lines.json.
  - B1 is a large-radius fold, but it is formed: a continuous arc, not rolled over a long length.
