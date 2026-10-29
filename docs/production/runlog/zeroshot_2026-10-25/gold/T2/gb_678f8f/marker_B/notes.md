# gb_678f8f marker B
Large formed bracket (t~1.47). Profile (seen along the main fold axis): plate 3 -> web 7 -> plate 5 (Z/hat, 3 and 5 parallel), plate 1 folded ~86 deg out of plate 5 along the edge of a big window in 5, plus end flanges.
- B1 [3,7] 66 deg, 218 mm. B2 [5,8] 65 deg, 218 mm.
- B3 [1, plane on plate 5] 86 deg, 152 mm. Auto pair 1/6 failed: plate 5 extends on both sides of this fold (window), which flips the side direction and gives mixed skin; the 5-side flange at the fold is only a ~5 mm stub. Fixed with plane_from_points seeded on plate 5 outer skin below the fold (grow=false, r=25) and extent_xyz from section scan (fold present s=-48..104 along axis). fold_support 0.99, outer, certain. Apex sits ~1 mm from the arc top on this tight-radius fold (sharp corner almost on skin); acceptable but worth a look at adjudication.
- B4 [2,11] 90 deg 220 mm return flange on plate 1 long edge.
- B5 [2,10] 91 deg 190 mm end flange of plate 1; B6 [2,19] 91 deg 39 mm short end flange of plate 1.
- B7 [5,15] 92 deg 147 mm end flange of plate 5; B8 [5,23] 90 deg 14 mm small eared tab of plate 5 (with hole).
- B9 [3,14] 92 deg 101 mm, B10 [3,17] 90 deg 88 mm end flanges of plate 3.
Rejected: 20/21 small patches = transition in the 1-5 fold region at the window end (part of B3); 24 = other skin of 17; 25 thin edge; hole walls.
Full scan. Confidence medium-high (B3 needed overrides; B8 small).
