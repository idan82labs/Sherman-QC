# gb_bc4b8c marker B

Large shallow frame (~690 x 275 mm), t ~1.45 mm, central window (~490 x 110). 11 bends.

| line | patches | what |
|---|---|---|
| B1 | 2-[40,46,52] | -e2 long edge joggle, upper bend (~31 deg wall, ~1.5 mm offset) |
| B2 | [39,42,43,48]-[13,17,21,38] | -e2 joggle, lower bend |
| B3 | [14,18]-5 | -e2 outer down-flange (~10 mm) |
| B4 | 2-[19,20,24,30] | +e2 joggle, upper bend (~30 deg) |
| B5 | [22,23,25,29,36,44]-4 | +e2 joggle, lower bend |
| B6 | 3-12 | +e2 outer down-flange |
| B7 | 2-16 | -e1 short end down-flange (~20 mm), spans the base zone (221 mm) |
| B8 | 2-8 | window long side down-flange (+e2 side) |
| B9 | 2-11 | window long side down-flange (-e2 side) |
| B10 | 7-28 | foot tab at the bottom of window flange 7/11, near the +e1 window end |
| B11 | 6-31 | foot tab at the bottom of window flange 6/8 |

Overrides:
- B1/B2: the -e2 joggle walls are only segmented over e1 95..318, but the sections at e1 -130/-100/100 show the step all along. I set extent_xyz to e1 -301..368 (fold support 1.0).
- B1: apex_mode station_median, because a ghost double top skin (registration artefact, around e1 -170..-105, e2 -110..-60) lifted the envelope apex.

Not bends:
- the +e1 outer end and both window short ends (plain edges);
- the ghost double-skin area near the -e1 window corner (patches 32-37, 41, 49-51, 54-60): scan artefact, not a formed feature;
- hole walls and the rectangular cut-out.

Doubts: the joggle offset is ~1 t. Per the rule, a joggle counts as 2. B3 apex offset 0.73 vs 1.24 expected (noise on the strip top skin).
