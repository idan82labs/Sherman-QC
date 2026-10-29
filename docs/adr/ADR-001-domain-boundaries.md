# ADR-001: Domain Boundaries

## Decision
Reorder the repo around application, domain, infrastructure, contract, and research layers.

## Consequence
Feature code moves behind stable contracts first; old entrypoints survive temporarily as compatibility shims.
