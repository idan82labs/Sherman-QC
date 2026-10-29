"""C06 reporting: offscreen 3D snapshots and the bilingual PDF inspection report.

- :mod:`.typefaces`: the bundled Heebo font (OFL), sha256-verified.
- :mod:`.labels`: English / Hebrew report vocabulary (decision colours come
  from ``contracts.production.vocabulary``).
- :mod:`.snapshots`: matplotlib/Agg iso + front/top/right views from the
  viewer assets (heatmap, bend lines, labels).
- :mod:`.shopfloor_pdf`: the A4-landscape PDF (PyMuPDF Story; RTL Hebrew is
  shaped and bidi-ordered by MuPDF), deterministic for the same input.
- :mod:`.render_cli`: the runner's render step and the stand-alone re-render.

The report is rendered from a ``ShopfloorInspectionResultV1`` only; it never
computes or changes a decision. Heavy libraries are imported lazily.
"""
from __future__ import annotations

__all__: list[str] = []
