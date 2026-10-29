"""E4 evaluation core: scorer, truth loaders, tiered runner, G2 line metric, G4 renders.

Plan: docs/production/PRODUCTION_LINE_PLAN.md §5 (gates G1-G4) and §10 amendments
3-8; design notes and gate definitions: docs/production/ENGINE_EVAL.md.

Modules (import what you need; importing the package imports nothing heavy):

- :mod:`.thresholds` -- frozen, hashed gate thresholds (G1, G2 line metric, G2
  physical, G3, G4).
- :mod:`.truth` -- typed loaders for E2 ``truth.json`` / ``case.json`` / manifests.
- :mod:`.engine_output` -- validated E3 output directories and frame helpers.
- :mod:`.scorer` -- per (scan, bend_uid, characteristic) rows and the G1 scorecard.
- :mod:`.runner` / :mod:`.procmon` -- serial tier runner driving the E1 and E3
  command lines with wall-time and RSS capture.
- :mod:`.line_metric` -- G2 Gold line-metric adapter (5 deg / 4 mm / 0.6, raw frame).
- :mod:`.render` and :mod:`.g4` -- G4 overlay renders, canaries, review packets,
  verdict schema and the G4 gate.
- :mod:`.freeze` -- the freeze ledger (amendment 4): FREEZE / RUN / owner WAIVER
  entries, re-freeze and frozen-code detection, the external anchor.
- :mod:`.scanner_card` -- the amendment-4b scanner model card from real scans.
- :mod:`.oracle` -- a TRUTH ORACLE for harness self-tests only (never an engine).

Command line: ``python -m domains.production.engine.eval --help``.

Custody: synthetic E2 cases and their truth only, except :mod:`.scanner_card`,
which reads authorized development scans (plan §9) for light statistics.
"""
