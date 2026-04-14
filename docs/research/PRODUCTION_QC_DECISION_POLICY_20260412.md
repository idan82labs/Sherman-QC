# Production QC decision policy — 2026-04-12

## Goal
Separate bend evaluation into three independent axes so production release logic does not overload a single bend status field.

## Per-bend axes
1. `completion_state`
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

2. `metrology_state`
- `IN_TOL`
- `OUT_OF_TOL`
- `UNMEASURABLE`

3. `positional_state`
- `ON_POSITION`
- `MISLOCATED`
- `UNKNOWN_POSITION`

## Release doctrine
- `FORMED` means geometrically completed, not automatically acceptable.
- False positives are the worst error class.
- A bend in the wrong position is a harder failure than a bend with the right position but wrong angle.
- Heatmap / positional evidence is the release veto layer.
- Until live per-bend heatmap contradiction is available in the product report, bend-line center deviation is the deterministic positional proxy.

## Part-level release logic
- `FAIL` if any countable bend is `MISLOCATED`
- `FAIL` if any countable bend is `UNFORMED`
- `WARNING` if there are no release blockers but any bend is `OUT_OF_TOL`, `UNMEASURABLE`, `UNKNOWN`, or `UNKNOWN_POSITION`
- `PASS` only if all countable bends are formed, in tolerance, and on position

## Current implementation status
Implemented in the bend report contract on the clean QC branch.

Current deterministic positional source:
- per-bend `line_center_deviation_mm`
- threshold aligned with existing matcher policy: `> 2.5 mm` => `MISLOCATED`

Future upgrade path:
- when per-bend heatmap contradiction/support is surfaced in live product output, it should override the positional proxy:
  - `CONTRADICTED` => `MISLOCATED`
  - `SUPPORTED` => `ON_POSITION`
  - otherwise fall back to geometric position deviation
