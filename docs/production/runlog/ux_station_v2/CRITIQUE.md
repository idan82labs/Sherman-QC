# Station result view: critique before the v2 redesign (2026-10-19)

Evidence: real part 10839000 run through the app copy, real ZEISS-style hot-folder drops,
1920x1080 Chromium (`node scripts/production/run_e2e_real.mjs --keep --run ux-station-v2-before`).
Screenshots are in `before/`: `full-en.png` (FULL scan), `full-worst-closeup-en.png`,
`partial-scan1-en.png` and `partial-scan1-table-end-en.png` (stage scan, 8 of 12 bends done),
and `partial-scan2-he.png` (Hebrew, RTL).

From the brake, at about 1.5 m, a bender cannot tell these things at a glance:

1. **Which bend is which.** The 3D tags say `8 23e7` (sequence number plus a hex id). The table
   says `8 / B12` and the strip says `8`. The studio name (B12) appears only in small print in the
   table. The hex ids mean nothing on the shop floor.
2. **What to do.** The banner reads "18 pass 4 fail 22 review 4 undetermined of 48 required checks".
   That is a count of characteristics, not of bends, and it never says which bends to fix or what
   "review" asks of the operator.
3. **Expected versus actual for one bend.** There is no per-bend view. Each table cell packs the
   actual, the Δ, the nominal, the tolerance and a verdict word into about 180 px, at four type sizes.
   Clicking a bend only moves the camera, and nothing on the part shows the numbers.
4. **Whether a value is near the limit.** "Review" appears on a flange that is off by −0.1 mm with a
   ±0.5 tolerance (B1). The reason, measurement uncertainty, is invisible, so the verdict looks
   arbitrary and erodes trust.
5. **Flange length comes first only in name.** "Not measured / Feature not supported / 3.6 ±0.5"
   takes three to four lines per cell. Only about 3 bends fit on screen, so the operator scrolls past
   12 bends of noise to find the red ones.
6. **The heatmap is the default.** A red and blue point cloud plus a 5-stop legend is engineer
   tooling. It hides the CAD and the bend lines and competes with the verdict colours (red heatmap
   points versus red FAIL tags).
7. **Too many colours.** Heatmap red and blue, plus status green, amber, red, blue and grey, plus
   amber tags on almost every bend. Amber is everywhere, so it carries no signal.
8. **Partial scans.** The greyed bends exist, but "Stage 8 of 12" sits on a second strip below a red
   slab. "Final check, but bends are missing" appears twice. Nothing says "8 of 12 bends done, carry on".
9. **Hebrew/RTL.** The nominal and tolerance pairs reorder ("±0.5 13.0"), the part name stays English,
   and the leader dots of the 3D tags point at the label anchors, not at the bend line.
10. **Branding.** There is no Sherman logo. The app sidebar shows the gradient text "Sherman QC" with a
    generic activity icon.

What the bender needs: one big verdict for the part (OK / CHECK / FIX) with "8 / 12 bends done"; a
list of bends using the studio names, each with one status word and icon, flange Δ and angle Δ, and
a bar showing where each value sits in its tolerance; and a tap on any bend that puts expected → actual
on the part itself, with one sentence saying what to do.

## After (station v2, 2026-10-19)

- `after/`: the same real 10839000 results (FULL, scan_1 at 8/12, scan_2 at 10/12) re-rendered by the
  new station at 1920x1080, in English and Hebrew, including callouts (`*-callout-*`), a not-bent bend
  and the heatmap toggle.
- `after-e2e/`: screenshots from the real app-copy E2E `e2e/ux-station-v2-after2`. Every station
  assertion passed there. That run is still 10/13 overall: the three scan steps stop at the PDF header
  check, after the PDF report changed (that work is in parallel and out of scope here).
