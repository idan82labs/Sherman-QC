"""C06 inspection adapter: E3 output directory -> station/PDF-ready job result.

The production engine (E3, ``python -m domains.production.engine.inspect``)
writes its output directory (ENGINE_IO §4.1): ``result.json``
(``ShopfloorInspectionResultV1``), ``registered_scan.ply``, ``deviations.f32``,
display meshes and ``bend_geometry.json``, all in ``cad_model_mm``. This
package *wraps* that directory. It never measures, never decides and never
changes a verdict (plan §10 amendment 1; the legacy worker is not on this
path):

- :mod:`.engine_output` loads ``result.json``, validates it as the contract
  model and runs ``contracts.production.consistency.check_result_against_compiled``
  against the job's CompiledPart and recipe (any problem is
  ``ENGINE_OUTPUT_INVALID``).
- :mod:`.viewer_assets` writes the decimated, index-aligned viewer copies
  (``viewer_scan.ply`` + ``viewer_deviations.f32`` from one shared index
  selection), the display meshes and ``bend_geometry.json`` into ``result/``.
- :mod:`.adapter` orchestrates ingest -> viewer assets -> snapshots -> PDF
  (:mod:`domains.production.reporting`) -> ``result/shopfloor_result.json``.
  Only ``viewer_assets``, ``artifacts`` and appended ``warnings`` differ from
  the engine's result (the runner enforces this as ``RESULT_ALTERED``).
- :mod:`.postprocess_cli` is the command line; the runner calls the
  equivalent ``python -m domains.production.reporting.render_cli``.

Heavy libraries (numpy, matplotlib, PyMuPDF) are imported inside functions so
importing this package stays cheap.
"""
from __future__ import annotations

__all__ = ["ADAPTER_VERSION"]

#: Version of the C06 adapter (recorded in postprocess.log and the PDF footer).
ADAPTER_VERSION = "c06-1.0.0"
