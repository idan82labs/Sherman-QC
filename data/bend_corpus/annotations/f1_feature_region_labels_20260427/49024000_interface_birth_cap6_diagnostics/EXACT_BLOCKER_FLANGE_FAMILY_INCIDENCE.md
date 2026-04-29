# F1 Exact-Blocker Flange Family Incidence Probe

Generated: `2026-04-27T16:05:25`

## Family Grouping

- Flanges: `18`
- Families: `6`
- Normal threshold: `4.0`
- Plane threshold: `12.0`

| Family | Flanges | d | Normal |
|---|---|---:|---|
| `FF1` | `['F1', 'F2', 'F3', 'F4', 'F5']` | `-285.928236` | `[0.098795, -0.42187, 0.901257]` |
| `FF2` | `['F11']` | `310.535555` | `[0.002111, 0.514113, -0.85772]` |
| `FF3` | `['F12', 'F20', 'F25', 'F26', 'F27', 'F29', 'F30']` | `261.451169` | `[0.685769, -0.396432, -0.610379]` |
| `FF4` | `['F14', 'F15', 'F22']` | `166.717783` | `[0.787714, -0.568122, -0.23821]` |
| `FF5` | `['F17']` | `134.309705` | `[0.769129, -0.623538, -0.140142]` |
| `FF6` | `['F32']` | `132.403532` | `[0.710016, -0.674751, -0.201464]` |

## Incidence Decision

- Status: `hold_exact_promotion`
- Hold reasons: `['no_uncovered_family_pair']`
- Candidates: `0`
- Uncovered family pairs: `0`

| Corridor | Bend | Raw pair | Family pair | Family uncovered | Family leakage | Reasons |
|---|---|---|---|---|---:|---|

## Use

This is diagnostic-only. It tests whether raw uncovered flange pairs are actually aliases of already-covered physical flange families.
