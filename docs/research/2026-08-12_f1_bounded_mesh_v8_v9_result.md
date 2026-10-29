# F1 bounded-mesh V8/V9 result — 2026-08-12

## Outcome

V8 finite-span completion is **retired as a general selector**. On 35 bounded-mesh
development scans it increased recall but reduced micro F1:

- V7: TP 83, FP 236, FN 93, precision 0.260188, recall 0.471591, F1 0.335354.
- V8: TP 94, FP 313, FN 82, precision 0.230958, recall 0.534091, F1 0.322470.

The V8 completion is now restricted to bounded-mesh pools and fails closed on
ordinary pools. `fragment_merge` representatives are excluded. These changes
prevented the demonstrated ordinary-pool regression but do not justify funding
V8 as a universal route.

V9 is a source-only complexity router:

- non-decimated pools retain V6;
- bounded-mesh pools with at most 400 line objects retain V7's recall route;
- larger bounded-mesh pools use a family-grouped ExtraTrees precision selector.

Five-fold part-family-grouped out-of-fold validation on 35 scans / 25 part
families produced TP 99, FP 140, FN 77, precision 0.414226, recall 0.562500,
and F1 0.477108. This is a development result, not an unseen promotion claim.

## Transfer checks

The previously untouched 472 and 389 scans were rerun without count, identity,
CAD, or expected-count input:

- 472 H01: 12 TP, 0 FP, 0 FN.
- 472 H02: 10 TP, 0 FP, 0 FN.
- 389 N01: 3 TP, 0 FP, 0 FN.
- 389 N02: 3 TP, 0 FP, 0 FN.

These matches use frozen manual precommit lines and therefore demonstrate
transfer against that precommit, not an independent new labeling event.

On the elaborate authoritative `ysherman` case, V9 produced 16 lines: 9 TP,
7 FP, 3 FN, precision 0.5625, recall 0.75, F1 0.642857. V7 had 4 TP, 6 FP,
8 FN (F1 0.363636); corrected V8 had 8 TP, 6 FP, 4 FN (F1 0.615385).

## Manual native-resolution visual critique

The four-view `ysherman` overlay was opened at original resolution. V9 is
visibly cleaner than V7 and covers most long perimeter and formed-end bends.
It still draws parallel aliases along the outer perimeter and duplicates several
vertical/edge transitions. Three trusted interior bends remain missing. The
visual therefore supports funding V9 as the next research baseline, but vetoes
promotion to runtime/main or production.

The existing 472 good/bad overlays were also reopened at original resolution.
They remain unchanged because both scans route through the small-pool V7 path.
They retain useful bend placement but also show that proposal count and clean
visual geometry alone do not establish defect classification.

## Reproducibility and limits

- Seven focused selector tests pass.
- Repeated V9 inference on `ysherman` is byte-identical.
- Two historical STL cases (`C004`, `C016`) failed closed because bounded
  decimation still exceeded the point budget; no threshold or resource rescue
  was performed.
- No runtime/main or production files were changed.

## Decision

Fund V9 as a research-only baseline and use the new bounded-mesh labeled corpus
for the next precision/recall cycle. Retire universal V8. Do not claim production
generalization yet; the next evidence should be a genuinely new manually
adjudicated part or scan pair evaluated once with this frozen V9 artifact.
