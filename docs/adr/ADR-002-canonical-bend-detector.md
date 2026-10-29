# ADR-002: Canonical Bend Detector Ownership

## Decision
The bend-inspection runtime is being centralized under `domains/bend/*`, with temporary adapters back to the existing pipeline while the canonical path is extracted.

## Consequence
Multiple historical bend-entry paths may exist temporarily, but must be marked as canonical, compatibility shim, or dead code before deletion.
