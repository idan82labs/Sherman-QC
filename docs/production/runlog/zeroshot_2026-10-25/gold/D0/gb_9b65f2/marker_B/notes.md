# gb_9b65f2 (STL), marker B

Final marking: `lines/lines.json` (copy in `lines.json`). 7 lines, apex locus, raw frame. Full scan.

Part: a 90 deg bracket made of plate A (1/2) and plate B (3/4). Each plate has a ~7.5 mm 90 deg flange on its free edge and on both side edges.

| bend | pair | what |
|---|---|---|
| B1 | 2-4 | main 90 deg fold between plate A and plate B. `extent_t` [-58.3, 59.2] is the full plate length, because the auto extent was cut where plate A's outer skin near the fold is unscanned. Override support is 1.0. The tool adds "support zone widened" and sets confidence to uncertain; the section puts the line on the apex. |
| B2 | 2-5 | plate A free-edge flange, 119 mm |
| B3 | 2-7 | plate A side flange, `extent_t` to the full flange length, 84 mm, support 1.0 |
| B4 | 2-9 | plate A other side flange, `extent_t`, 84 mm, support 1.0 |
| B5 | 4-6 | plate B free-edge flange, 97 mm. 14/17 are pieces of 6's other skin. |
| B6 | 4-11 | plate B side flange, 54 mm |
| B7 | 4-15 | plate B other side flange, 53 mm |

Every pair is on the outer skin. The line-to-scan distance is 0.03 to 0.39 mm at every station.

Rejected:
- the rows of small dimples or louvres on both plates (embossed features);
- the two long slots in plate B (holes);
- the corner seams where side flanges 9/11 and 7/15 meet. These pairs are coplanar: each plate has its own flange and they butt at the corner, so this is not a bend.
- 18 is a thin edge strip.

Doubts: none major. Side flanges 9 and 11 lie in one plane. I read them as two flanges, one bent from each plate, meeting at the corner. If they were one wall bent from one plate, one of B4/B6 would be wrong. The face-on closeups show each one with its own fold arc along its own plate edge.
