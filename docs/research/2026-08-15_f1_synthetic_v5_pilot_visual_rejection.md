# F1 Synthetic V5 Finite-Part Pilot — Native Visual Rejection

Date: 2026-08-15

## Decision

**REJECT BEFORE INFERENCE.** The six-case V5 pilot is a useful correction in topology, but it is not visually close enough to the physical corpus to fund a detector run. It must not be used to promote, retire, or compare V61/V62.

V4 remains an adversarial rounded-profile/parallel-transition stress set. V5 remains a morphology-development pilot only. Neither is confirmatory physical evidence.

## What improved over V4

- Parts are finite in both sheet directions instead of constant-cross-section extrusions.
- Bend lines can terminate at local tab endpoints.
- Cases contain unequal flanges, free edges, apertures, and one explicit smooth-roll negative control.
- Truth density fell to 2–5 manufacturing folds per part rather than the V4 average of 10.6 transitions per part.
- Fused source sizes of 41,879–61,080 points are within the observed scanner-density regime and well below the 200,000-point inference budget.

## Why the pilot is still rejected

Side-by-side native review against physical C006, C019, C030, and 48811000_A found five material visual gaps:

1. **Zero-radius manufacturing seams.** Most positive folds are razor-sharp patch junctions. Physical parts show finite bend radii and a thickness-dependent transition zone.
2. **Coarse punched geometry.** Circular and slotted cutouts are cell-masked on a rectangular grid, creating stepped or plus-shaped boundaries. Those boundaries are artificial high-curvature false-positive attractors.
3. **No material thickness.** The pilot is primarily a collection of single-sided surfaces. Physical free edges, returns, hole walls, and folded corners expose finite sheet thickness.
4. **Overly clean rectangular silhouettes.** Real parts use rounded/chamfered corners, relief notches, asymmetric outlines, patterned perforations, and local mounting ears. V5 still reads as a CAD exercise rather than a scanned production part.
5. **Insufficient compound topology.** Several cases remain one carrier plate plus a few simple flanges. O005 in particular reproduces an extrusion-like full-span step pattern in finite form.

These defects would teach or test the wrong behavior: sensitivity to meshing artifacts and idealized seams rather than discrimination between supported manufacturing folds, free edges, hole boundaries, rounded forms, and scan damage.

## Case-by-case visual disposition

| Case | Visual finding | Disposition |
|---|---|---|
| O001 | Local unequal tabs are useful, but the parent plate is too rectangular; openings are stepped; all three folds are zero-radius. | Retire geometry; keep concept only. |
| O003 | The tray silhouette is closer to a physical part, but walls and ears are too clean and disconnected at relief corners; apertures remain coarse. | Revise from concept. |
| O005 | Full-span parallel steps dominate the part and recreate the V4 extrusion bias. | Retire entirely. |
| O007 | Best local-topology concept: finite doghouse and terminated folds. Still fails on zero-radius seams, single-sided material, and idealized rectangular opening. | Revise first. |
| O009 | Best negative-control concept: the broad roll correctly remains unlabeled while two short tabs are positive. Plate outline and holes are still synthetic-looking. | Revise and retain concept. |
| O011 | Sparse carrier plate plus two parallel returns is visually under-complex and likely to reinforce parallel-edge false positives. | Retire geometry; keep small-crease idea. |

## Required replacement design

The next pilot must be accepted visually before inference and must include:

- explicit 1–2 mm sheet thickness at free edges and apertures;
- 1.5–4 mm manufacturing bend radii with truth at the bend centre, not at a sharp patch seam;
- smooth circular/slot boundaries and non-rectangular rounded/chamfered outer contours;
- relief notches and physically joined corner geometry;
- at least one compound bracket, one perforated cover, one local raised feature, and one rounded-form negative control;
- 1–5 true bends per part with every smooth tangent continuation left unlabeled;
- no constant-cross-section or mostly full-span parallel-fold case.

## Execution gate

No V61/V62 or later engine invocation is authorized on this pilot. The next action is a replacement four-case high-fidelity visual pilot, followed by native-depth and fused-cloud inspection of every case. Only a passing visual morphology audit may open blind source-only inference.
