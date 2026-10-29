"""Production line contracts (single source of truth).

Pydantic v2 models, ``extra="forbid"``, ``frozen=True``; document models carry
a literal ``schema`` tag. Modules:

- :mod:`.canonical` - canonical JSON bytes and sha256 hashes
- :mod:`.common` - shared types (Vec3, Tolerance, timestamps, frame name)
- :mod:`.characteristics` - characteristic kinds, ids, guard-band decision
- :mod:`.states` - bend / validity / summary / stage vocabularies
- :mod:`.part_recipe` - PartRecipeV1, ReleaseRequestV1, build reason codes
- :mod:`.compiled_part` - CompiledPartV1 (E1 output, E3 input)
- :mod:`.inspection_result` - ShopfloorInspectionResultV1, StationFeedV1, derivations
- :mod:`.consistency` - result vs CompiledPart vs recipe cross-check
- :mod:`.line_job` - ScanFileV1, LineJobV1, LineJobSummaryV1 and literal sets
- :mod:`.events` - LineEventV1 and EVENT_KINDS
- :mod:`.layout` - data root, workspace and engine output layout
- :mod:`.vocabulary` - legacy -> production terms, display tokens (REVIEW amber)
- :mod:`.schema_export` - ``python -m contracts.production.schema_export --out DIR``

Engine I/O (CLIs, output directories, frames) is specified in
``contracts/production/ENGINE_IO.md``. This package is pure: it imports only
the standard library and pydantic.
"""
