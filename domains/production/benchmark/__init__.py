"""Benchmark case registry and per-bend evaluator (C12).

Two small modules on the accuracy-qualification path (plan §5, §8 and
``docs/research/2026-10-17_production_line_accuracy_protocol.md``):

- :mod:`.case_registry`: an append-only, hash-chained JSONL registry of
  benchmark cases. Each row binds one inspection (job id, recipe sha, scan
  sha) to independent truth: synthetic, metrology-measured or an operator
  verdict.
- :mod:`.evaluator`: per-quantity errors (reported actual minus truth),
  NOT_MEASURED rates and conformance confusion matrices with explicit
  denominators, plus per-SKU-family aggregation and the worst family. It also
  adapts E4 synthetic score rows (:mod:`domains.production.engine.eval.scorer`)
  to the same shape. It does not duplicate E4's G1 gates: E4 owns the scorer
  and the gate scorecards (plan §10 amendment 2), and this package only
  summarises their rows for line-level reports.

Nothing here is promotion evidence by itself. The protocol decides which
registered cases may be scored and when.
"""
from __future__ import annotations

__all__ = ["case_registry", "evaluator"]
