"""Sherman QC production engine. Plan: docs/production/PRODUCTION_LINE_PLAN.md (§3, §10).

Separate entry points, no shared CLI (plan §10 amendment 2):

- :mod:`domains.production.engine.compile` - onboarding compile (E1):
  ``python -m domains.production.engine.compile`` turns the folded STEP, the
  confirmed recipe and the optional flat into a ``CompiledPartV1`` (flange
  graph, loci, nominals, stage models, symmetry tests). Design notes:
  docs/production/ENGINE_COMPILE.md.
- :mod:`domains.production.engine.inspect` - per-scan engine (E3):
  ``python -m domains.production.engine.inspect``.
- :mod:`domains.production.engine.synthetic` - synthetic suite and virtual scanner (E2).
- :mod:`domains.production.engine.eval` - evaluation and visual verification (E4).

The binding I/O is contracts/production/ENGINE_IO.md. Importing this package
imports nothing heavy (no OCP).
"""
