# Root visual inspection: native-curve graph successor

2026-09-30. Derived-only gallery:
`artifacts/cad-bend-native-curve-gallery-v2-2026-09-30-attempt1/`.
Pinned visual bundle SHA-256
`36af48093a16137d41f73319ffcee437b8d4dde7f5bbe69b92f273bd67e03bdb`.
Renderer completed once: outer exit 0, cleanup known, 6.8666 seconds,
283,888 KiB peak sampled RSS; fixed 120-second/2-GiB budget.

Root opened and inspected each of the twenty case PNGs, with four orthographic
views per image. No part was omitted. Whole native surface context and holes are
present in every panel; manifest reports no unrendered faces. Labels are crowded
on complex cases, and marker lines deliberately overlay occluded geometry, so
these are diagnostic panels, not an operator-view acceptance test.

- Cases 04 and 17: zero-output baseline becomes two finite bend markings each.
  Remaining long/upper bend regions are visibly red reference overlays, not
  inferred output. No claim of complete part coverage.
- Case 11: two new long/short bend regions visibly follow the sheet transitions,
  alongside the retained third region. All three known regions are marked.
- Case 13: two long bend regions recovered; two side bend regions remain missed.
- Case 19: all fourteen known regions now marked, including seven recovered
  short flange/long rail regions. Openings and disconnected tabs are retained.
- Other fifteen cases: retained output visually consistent; missing rolled
  regions (10, 14, 15) and remaining ordinary regions (12, 16) remain exposed.

Blue/green whole-face colors are frozen-reference matching overlays. Dashed
green/cyan lines are actual challenger canonical endpoints, never synthesized
from reference width. New graph markers certify a strict interior core only;
they do not establish full finite extent, flange dimensions, calibrated units,
angle accuracy, unfolding correctness or manufacturing approval. Static views
are a supporting sanity check, not independent geometric measurement truth.

All original baseline predictions and six unfolded previews are preserved by
the independent candidate POST; this visual review does not expand unfolding
eligibility. Numeric score must come from the unchanged frozen scorer.
