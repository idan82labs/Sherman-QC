# gb_21068a, marker B

Final marking: `lines/lines.json`. 3 lines, apex locus, raw frame. Full scan.

Profile: lug flange 3/4 (two lugs with holes) -> sloped web 5/8/9 -> top plate 1/7 (large holes) -> end flange 2/6.

| bend | pair | angle | what |
|---|---|---|---|
| B1 | 3-[8,9] | 37 deg | lug flange to sloped web. 8 and 9 are the same outer-skin plane, split by an unsegmented area. 138 mm. |
| B2 | [8,9]-7 | 53 deg | web to top plate. 7 is the only segmented outer-skin piece of the top plate. `extent_t` [-40, 97] is the full web length, override support 1.0. |
| B3 | 1-2 | 90 deg | top plate to end flange, 99 mm |

On B2, the line sits 1.4 mm from the nearest scan points over its far half. My own sections at t = 40 and 80 show why: the outer skin is unscanned there and only the inner skin, about t = 1.48 mm in, is present. At t = -30 and 10 the line sits on the outer arc. The plate is flat: the inner-skin offset from plane 7 holds at 1.37 to 1.49 mm along the whole length. So the line is placed correctly in the gap where the outer skin was not scanned.

Rejected:
- lugs on 3/4 (coplanar with the flange);
- the holes and slots in 1 and 2;
- 10 and 11, thin strips on the fold arcs.

Doubts: none.
