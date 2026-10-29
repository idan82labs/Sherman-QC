"""Production recipe back end (C05): immutable recipe store, release gate, flat intake.

Modules:

- :mod:`.store` - SQLite recipe store (SKU registry, immutable versioned
  recipes, release drafts, compile records, expectations, benchmark
  references, flat intake rows).
- :mod:`.catalog_port` - the narrow port onto the legacy part catalog and the
  Vendor Setup draft validator (imported lazily, never at module import).
- :mod:`.release_service` - the release gate: draft + catalog files ->
  ``PartRecipeV1`` -> supervised E1 compile -> hash-bound CompiledPart ->
  stored recipe + benchmark reference.
- :mod:`.flat_intake` / :mod:`.flat_intake_cli` - flat (DXF / flat STEP)
  upload, parse and E1 binding proposal for operator confirmation.
- :mod:`.onboarding` - the per-part onboarding checklist.
- :mod:`.resolver` - scan filename -> active recipe (C02 ports), recipe
  provider and compiled-part sink for the runner.
- :mod:`.export_benchmark` - JSONL export of released benchmark references.
- :mod:`.hashing` - file sha256 with a stat-keyed cache for the read-only gate checks.
- :mod:`.synthetic_onboarding` - puts an E2 synthetic case into a part catalog
  (tests, E2E, demos).

Importing this package imports nothing heavy (no OCP, no legacy server).
"""
