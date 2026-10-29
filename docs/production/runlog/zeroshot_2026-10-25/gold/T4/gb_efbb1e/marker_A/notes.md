# gb_efbb1e — marker A

Part: small U-channel (~290 x 20 x 9 mm, t ~1.2 mm). The web has a lengthwise slot; the web half on the y- side has its slot edge turned up (~3 mm upstand, patches 7/8). Legs are split into 5 segments each by relief slots; the y- leg is joined to the web only through a comb of short bridges (windows along the fold line).
- B1: web half (y-) top skin [plane_from_points] -> y- leg [4,17,10,22,13]. 91.3 deg. extent_xyz over the leg span (pca0 -132..132), 264 mm. One line per GOLD_SET_V2 (fold split by windows). Override fold support 0.48 (windows) -> tool flags uncertain; apex offset 0.47 vs expected 1.03 because many stations are windows.
- B2: web half (y+) top skin -> y+ leg [6,23,12,24,16]. 91.1 deg, 264 mm, support 0.96.
- B3: web half (y-) underside -> upstand 7 (outer = underside + slot-facing face). 90.9 deg, 248 mm, support 1.0.
Why per-half web planes: a single web patch 1/2 spans both halves, which are ~7 deg apart (plane rms 0.30 mm), so each fold uses a plane seeded on its own half.
Rejected: end tab 25/26 (7.6 deg to the web, below 10 deg); the slot, holes and window walls; leg ends.
Sweep: 8 cross-sections along the length; no other folds. Full scan.
Confidence: medium-high (B2, B3 high; B1 extent and apex rely on override through windows).
