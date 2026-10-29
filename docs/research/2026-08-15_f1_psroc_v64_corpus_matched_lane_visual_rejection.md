# PSROC V64 prospective lane — native visual rejection

Date: 2026-08-15  
Lane: `output/research/f1-psroc-v64-corpus-matched-prospective-v1-20260815`  
Decision: **REJECTED BEFORE DETECTOR EXECUTION**

## Why this lane is rejected

The TSDF replacement corrected the earlier stacked-view ghost skins and brought
triangle spacing into the numerical range of the real U01–U04 scans.  That was
necessary, but it was not sufficient.  Native-resolution inspection shows that
the underlying authored parts still do not match the morphology or acquisition
character of the corpus closely enough for a credible prospective F1 lane.

The real scans contain irregular stamped boundaries, rounded transitions,
localized ears and returns, small press marks, surface damage/roughness,
incomplete observation, and nonuniform feature density.  The synthetic parts
remain dominated by ideal planar polygons, exceptionally straight perimeter
edges, clean cutouts, abrupt flange endpoints, and repeated rectangular design
grammar.  TSDF fuzz cannot turn an unrealistic CAD design into a realistic
part.

## Case-by-case native visual ruling

| Case | Ruling | Native visual critique | Likely false-positive mechanism |
|---|---|---|---|
| O001 | reject | Large rectangular window plus four crisp, independently authored flanges; too many long, regular boundaries and abrupt tab endpoints. | Window perimeter, sheet perimeter, and flange endpoints form strong parallel/corner alternatives to the true folds. |
| O003 | reject | Plausible tray class, but far more rectilinear and uniform than U02; walls have nearly constant height and ideal corners, while ears and reliefs read as CAD appendages. | Four equally dominant wall boundaries plus crisp corner/ear junctions overstate fold evidence. |
| O005 | reject | Superficially resembles a broad U03-like panel, but lacks the corpus' press marks, local damage, asymmetric shallow features, and boundary irregularity. | One perfectly long outer edge and a clean parallel flange concentrate evidence on silhouette lines. |
| O007 | reject | Broad polygon with two separated lips reads as a constructed test coupon, not a manufactured/scanned part. | Artificial lip endpoints and opposite-facing local returns create isolated high-contrast turn candidates. |
| O009 | reject | Near-duplicate design grammar of O001: another long rectangular window bracket with short tabs.  It does not add independent morphology. | Repeats the same inner/outer parallel-edge confound and would overweight one synthetic failure mode. |
| O011 | reject | Idealized finite rib on a clean cover.  The rib's two symmetric shoulders are more regular and higher contrast than comparable corpus stamp features. | Paired shoulders and their crisp termination points invite double counting and spurious ownership. |

## Acquisition mismatch visible at native resolution

- Raw depth views show smooth, almost piecewise-linear depth gradients over
  perfectly clean faces.  Real scans show spatially varying roughness,
  incomplete returns, local dropouts, and surface blemishes.
- The fused TSDF has realistic *spacing* but not realistic *information
  content*.  Most irregularity is reconstruction fuzz laid over ideal geometry.
- Thin sheet/cutout boundaries remain unusually complete and straight across
  views.  These are precisely the non-bend edges that can inflate false
  positives in a line detector.
- O001 and O009 are not sufficiently independent to count as two prospective
  families.

## False-positive evidence boundary

The predecessor V5R3B lane was executed and produced `9 TP / 9 FP / 25 FN`
under the frozen V62 baseline.  The present lane was **not** run through V52,
V62, or V64.  Its false-positive mechanisms above are visual preflight findings,
not post-hoc scores.  Rejecting it before inference prevents another knowingly
confounded execution.

## Replacement requirement

The next lane must not be another set of clean handcrafted CAD coupons.  It must
be scan-derived or scan-conditioned at the morphology level:

1. use real source-only scan surfaces as morphology references without opening
   labels or detector outputs;
2. reproduce rounded stamped transitions, boundary irregularity, local press
   marks, nonuniform holes/ears, and partial observation—not merely point
   spacing;
3. limit near-duplicate templates to one case;
4. compare every candidate side-by-side with its nearest real corpus class at
   native resolution;
5. reject any candidate whose strongest visible lines are ideal cutout or
   silhouette edges rather than manufactured folds;
6. keep detector execution disabled until all six cases pass that visual gate.

No inference output, detector feedback, or evaluator truth was used for this
decision.  The evaluator-only truth remains unopened.
