# Conformity Decision Model Phase 1 — 2026-04-12

## What changed
- Added compatibility-safe top-level `release_decision` alongside legacy `overall_result`
- Split bend reporting into explicit:
  - `completion_state`
  - `metrology_state`
  - `positional_state`
- Added first-class debug/inference fields:
  - `correspondence_state`
  - `correspondence_confidence`
  - `correspondence_margin`
  - `observability_state_internal`
  - per-axis observability booleans
- Added `position_evidence` payload with trusted-frame handling

## Release semantics
- `AUTO_FAIL`:
  - any countable bend confidently `MISLOCATED`
  - any countable bend confidently `UNFORMED`
- `HOLD`:
  - ambiguous or unresolved correspondence
  - insufficient observability
  - unmeasurable metrology
  - unknown position
  - untrusted alignment / datum frame
- `AUTO_PASS`:
  - all countable bends formed, in tolerance, on position
  - correspondence-confident
  - sufficient observability
  - trusted frame

Legacy compatibility:
- `AUTO_PASS -> PASS`
- `HOLD -> WARNING`
- `AUTO_FAIL -> FAIL`

## Benchmark changes
- `evaluate_bend_corpus.py` now emits four scoreboards:
  - completion
  - metrology
  - position
  - abstention
- countable bends remain the main scored population
- rolled/process features remain separate guard metrics
- contradiction cases are exported separately for review

## Current limitation
- Position trust still falls back to line-center deviation when richer live per-bend heatmap evidence is unavailable.
- This is intentionally compatibility-first, not the final release architecture.
