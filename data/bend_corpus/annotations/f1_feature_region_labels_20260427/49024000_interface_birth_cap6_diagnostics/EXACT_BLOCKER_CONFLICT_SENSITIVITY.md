# F1 Exact-Blocker Conflict Sensitivity Sweep

Generated: `2026-04-27T16:05:25`

## Summary

- Interpretation: `no_reasonable_sweep_promotes_exact_12`
- Dominant runs: `0` / `24`
- Status counts: `{'no_admissible_candidate': 24}`
- Dominant candidate counts: `{}`
- Label costs: `[0.3, 0.42, 0.55, 0.7, 0.85, 1.0]`
- Conflict weights: `[0.5, 0.85, 1.2, 1.6]`

## Runs

| Label cost | Conflict weight | Status | +1 | +1 score | +2 | +2 score | +3 | +3 score |
|---:|---:|---|---|---:|---|---:|---|---:|
| `0.3` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.3` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.3` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.3` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.42` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.42` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.42` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.42` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.55` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.55` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.55` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.55` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.7` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.7` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.7` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.7` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.85` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.85` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.85` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `0.85` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `1.0` | `0.5` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `1.0` | `0.85` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `1.0` | `1.2` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |
| `1.0` | `1.6` | `no_admissible_candidate` | `None` | `None` | `None` | `None` | `None` | `None` |

## Use

If exact +1 dominance appears only under extreme label costs, the missing piece is a better object likelihood or corridor coverage model, not parameter tuning.
