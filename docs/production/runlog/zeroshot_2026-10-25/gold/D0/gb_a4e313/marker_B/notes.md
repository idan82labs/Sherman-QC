# gb_a4e313, marker B

Final marking: `lines/lines.json`. 9 lines, apex locus, raw frame. Full scan. t about 0.92 mm.

Part: a large flat perforated plate (1/2) with two strips crossing it. Each strip has a hook at one end and a sharp lip at the other. A downstand tab hangs from one window edge, and an offset pocket tab hangs from another window.

| bend | pair | what |
|---|---|---|
| B1 | 2-4 | 90 deg downstand tab with an arch cut-out. Outer skin. `extent_t` [-32.5, 45.9] is the full tab width, 78 mm, because the auto extent was cut where the plate skin is missing near the window edge. Support 1.0. |
| B2 | 1-7 | pocket tab, plate to drop wall (89 deg). Wall 7 has no scanned other skin, so this is an inner-skin pair, and the apex is about t = 0.9 mm inside the true outer apex. |
| B3 | 3-7 | pocket tab, drop wall to lowered floor (90 deg). B2 and B3 together form an offset (joggle) and count as 2. |
| B4 | plane(strip 1, +end)-9 | 60 deg first hook fold. `extent_t` is the full strip width, about 30 mm. |
| B5 | 8-12 | 118 deg second hook fold, strip 1 |
| B6 | plane(strip 2, +end)-10 | 60 deg first hook fold. Lip 10 has no twin, so this is an inner-skin pair, about t off. `extent_t` is the full strip width. |
| B7 | 10-11 | 118 deg second hook fold, strip 2 |
| B8 | plane(strip 1, -end)-13 | 119 deg end lip. `extent_t` is the full width. Confidence is uncertain: "support < 100 points on a side". |
| B9 | plane(strip 2, -end)-14 | 119 deg end lip, `extent_t` |

The strip-end bends use a local `plane_from_points` (radius 10, no grow), seeded on the strip about 8 to 16 mm from the lip. The global plate patch 2 or 1 does not fit the strip ends: the 2-9 pair failed with no support, and the other pairs gave lines a few mm long.

Rejected:
- the window and hole edges, and 18 (a thin strip);
- the strip/plate crossings, which are coplanar;
- 19 (a small patch on the plate).

Doubts:
- B2 and B6 are inner-skin only, because the outer skin was not scanned. Expect an offset of about 0.9 mm.
- If the adjudicator reads the 8/9/12 hook or the 10/11 hook as a rolled kink, it removes B5 and B7. I count them: both are discrete straight folds with flat legs.
