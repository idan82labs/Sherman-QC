"""Sherman QC production line domain.

- :mod:`domains.production.recipe` - pure recipe domain (C01): SKU rules and
  real scan-filename parsing, stable bend identity, PartRecipeV1 builder.
- :mod:`domains.production.engine` - onboarding compile (E1), per-scan
  inspect (E3), synthetic suite (E2) and evaluation (E4).

Contracts live in :mod:`contracts.production`; see
docs/production/PRODUCTION_LINE_PLAN.md and contracts/production/ENGINE_IO.md.
"""
