# ADR-003: Report Contract

## Decision
Operator/UI/report payloads must be assembled through explicit contracts and mappers rather than ad hoc dictionaries emitted from runtime code.

## Consequence
JSON responses, UI summaries, and PDF/report exporters should share one report model with domain-specific payload sections.
