# gb_44b168 marker A

Full scan, t~1.48 mm. This is an assembly, not one sheet:
(S1) a stepped channel: plate C, wall97, plate B, wall44, with a 132 mm tab on wall44 ending in an angled lip;
(S2) plate 1, with a closed hem on its top edge, two end walls bent from it, and a Z-offset on the right end wall;
S2's plate-1 bottom edge sits on wall44.
9 bends:
- B1 [4,15] outer: plate C to wall97, 89.6 deg, 169 mm.
- B2 [6,14] outer: wall97 to plate B, 90.4 deg, 175 mm.
- B3 [2,11] outer: plate B to wall44, 91.1 deg, 172 mm.
- B4 [8,19] outer: wall44 tab to angled lip 7/8, 122 deg, 134 mm.
- B5 [1,5] outer: plate 1 to left end wall 5/10, 89.8 deg, 51 mm. The outer arc radius is larger than the default estimate, but the apex sits on the arc in the section.
- B6 [1,13] outer: plate 1 to right end wall 13/22, 89.9 deg, 52 mm.
- B7 [17,22] outer: right end wall to offset flange 16/17, 90.1 deg.
- B8 [16,21] outer: offset flange to outer wall 20/21, 90.4 deg (B7+B8 = Z-offset, 2 bends).
- B9 [1, plane on the hem top arc] (uncertain): closed 180 deg hem on the top edge of plate 1, return ~11 mm (patch 9), 175.5 mm, extent_xyz over the full hem length. Plate 1 and its return are parallel, so the tool cannot intersect them. Side b is a plane fitted on the hem's top arc, which puts the label point on the top of the hem arc (checked in the section). The reported 99 deg angle is an artefact; the real fold is ~180 deg. If the adjudicator does not count hems under GOLD_SET_V2, drop B9.
Rejected:
- the plate 1 / wall44 junction: sharp outer corner, plate edge resting on or abutting the wall, weld-type joint, no formed arc;
- plate B against the end walls (butt contact, not bent);
- thin strips 25-28 (arc facets / edges);
- hole walls in wall44.
Missed-bend sweep: 6 cross-sections along the length plus 3 longitudinal sections, and zoomed sections at the hem and at the wall44 joint. Nothing else found.
