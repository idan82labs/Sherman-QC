# Case03 native eligibility and stepped-hole diagnosis

Date: 2026-09-30. Scope: source-only engineering and pure processing of the
explicitly released retained diagnostic at
`artifacts/vendor-unfold-geometry-diagnostic-v1-2026-09-30-attempt1/case-03-42288005/GEOMETRY_DIAGNOSTIC.json`.
No raw CAD, native import, new physical run, Gold, held-out, or historical
attempt mutation occurred. Diagnostic source SHA declaration:
`7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837`.
This is a mechanism diagnosis, not manufacturing approval or a benchmark score.

## First refusal: a representation boundary loses one forming carrier

The retained graph has four tree components: the two main skin layers carry
six extracted features each; two isolated skin layers carry B5. Coplanar chart
stitching cannot join them, because their missing connection is cylindrical.

Native source faces 98 and 99 are paired cylindrical skins with radii 1.2 and
2.7 mm and axial spans of 45 mm. Their retained native sweeps evaluate as:

| Zero-based source face | Raw sweep in degrees | Tangent flange source faces |
| --- | --- | --- |
| 98 | 14.9999999999981 | 45, 86 |
| 99 | 15.00000000000273 | 14, 47 |
| 102, existing B5 | 15.000000000004983 | 39, 47 |
| 103, existing B5 | 15.000000000007272 | 41, 45 |

The incumbent extractor filters individual faces with a raw minimum of 15.0
degrees, while reporting accepted sweeps rounded to six decimal places. It
drops face98 but keeps face99, then rejects the resulting unpaired group. This
removes the entire forming carrier and leaves B5 disconnected on both skins.

New `scripts/extract_step_topology_bend_axes_v2.py` applies the same existing
six-decimal degree representation at the minimum eligibility boundary. Its
200-degree upper limit and 8-mm axial minimum still use raw values. It retains
raw and published sweeps linked to each source face, and explicitly declares
that the representation rule is not a manufacturing tolerance. Coaxial
grouping, radius bounds, candidate geometry and canonical marker construction
remain the incumbent mechanism.

Verification: 27 focused pure-row tests pass, including 15 degrees plus/minus
1e-7, a genuinely below-boundary case at minus1e-6, raw upper cutoff, axial
minimum, missing radius partners, coaxial separation, raw provenance, and
native signed V coordinates. No native geometry job was used.

Source pins at verification:

- Frozen incumbent extractor: `f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314`.
- V2 extractor: `8217ffb09c46e04f3840b8dc687ad72014be6101ad2d9e8f6f85d8b95be13f65`.
- V2 tests: `9beacf06cbba2b8b4d9f05fb254f90b0cfc8407fb2ea94feb6ca74337384dac7`.

## Residual refusal: extra annular material is not a through-hole wall

A pure retained-chart probe restored the omitted pair using the same native
axis/mid-turn construction. After stitching, the graph contained 16 links,
and both skin trees covered all eight native forming carriers. The unchanged
material boundary proof then refused `UNKNOWN_HOLE_WALL_BOUNDARY`.

The exact first failing boundary is source skin face23 to source cylinder17,
native edge61. Bend sheet thickness is 1.5000000000000002 mm. Cylinder17 spans
2.0 mm and extends outward from the underside, rather than through the sheet.
The local source topology at approximately x=-92.965, y=2 is:

| Source face | Geometry and axial position | Native neighbors relevant here |
| --- | --- | --- |
| 25 | Planar skin, z=0 | top bore27 |
| 23 | Planar skin, z=-1.5 | outer annular wall17 |
| 27 | Radius2.7 bore, z=-1.5 to0 | skin25, shoulder28 |
| 28 | Annular shoulder, z=-1.5 | bore27, inner bore38 |
| 38 | Radius1.65 bore, z=-3.5 to-1.5 | shoulder28, annulus29 |
| 29 | Bottom annulus, z=-3.5 | inner bore38, outer wall17 |
| 17 | Radius3.95 wall, z=-3.5 to-1.5 | annulus29, skin23 |

There are analogous stepped annular components near (108,4.3) and (-58,4.3).
Their manufacturing origin is not established: these may encode formed or
added material. The different upper/underside rail radii and material outside
the sheet slab defeat the incumbent straight-extrusion through-hole proof.
Increasing the hole-wall tolerance or dropping these source faces would hide
actual geometry and is not justified.

## Next engineering boundary

Promote V2 extraction only through matching fresh-upload and unfolding source
bundles, so the operator sees the restored forming carrier as a marked bend.
Re-run only a separately frozen and independently audited successor; the
diagnostic and earlier attempts remain terminal.

Complete preview support for this case additionally needs an explicit native
auxiliary-feature/material role, with all source surfaces accounted for and
no assertion that such geometry develops into a uniform sheet blank. A
manufacturing blank requires the feature's forming or assembly provenance and
an appropriate material model. Until that new role is independently proved,
the complete selected-skin envelope must continue to abstain.
