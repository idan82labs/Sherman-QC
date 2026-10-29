# gb_a0bfde — marker A

Part: a multi-flange bracket (t ~1.47 mm). The base plate A is patches 1/2.
- Three up-walls (6/8, 7/15, 9/10) and an up-tab 17/18. Each has an outward top flange at +37 mm from A (16, 5/24, 19/20, 21). These four flanges are coplanar, which confirms the reading.
- A big down-flange B (3/4, 100 mm) carries a bottom lip 12/14 and a side flange 11/13.
- There are two small lanced down-tabs (22, 23) at slotted cut-outs in A.
13 lines, all ~90 deg, all ok/certain, fold support 1.0, all on the outer skin:
- B1 [1,3] A -> B, 193 mm. The fold is interrupted in the middle by the up-tab 17, which is coplanar with B but bent the opposite way. Each side is the same A-B fold, so I made it 1 line (GOLD_SET_V2 split-fold rule). apex_mode station_median: the envelope apex was pulled to -0.7 mm by the middle stations; now 1.21 vs 1.23 expected.
- B2 [2,17] A -> up-tab 17, 50 mm.
- B3 [2,6] A -> wall 6/8, 133 mm. B4 [6,16] wall -> flange 16, 94 mm.
- B5 [2,7] A -> wall 7/15, 113 mm. B6 [7,5] wall -> flange 5, 113 mm.
- B7 [2,10] A -> wall 9/10, 87 mm. B8 [9,20] wall -> flange 19/20, 87 mm.
- B9 [17,21] up-tab -> flange 21, 49 mm.
- B10 [4,13] B -> side flange 11/13, 101 mm. B11 [4,12] B -> bottom lip 12/14, 191 mm.
- B12 [1,22], B13 [1,23]: lanced tabs, 44 mm each, ~90 deg. The section shows a real formed radius.
Rejected: hole/slot walls, the thin strip 25 (bend arc of B2), and the 11-12 corner (no shared fold, only a corner relief).
Doubts: B1 as one line versus two segments. B2 lies in the gap of B1, offset only by the skin side.
Sweep: checked against the raw iso, front, right, top and under views; no other folds. Full scan.
Confidence: medium-high.
