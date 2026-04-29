# Hard Known F1 Latent Support Gap Summary

Generated: `2026-04-29T17:40:34`

## Cases

| Case | Target | Selected | Safe | Gap Class | Main Reasons | Next Experiment |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 10839000 | 8 | 4 | 4 | `latent_support_creation_with_contact_recovery_gap` | raw_bridge_safe_support_under_covers_target, many_candidates_die_on_flange_contact_or_attachment, fragment_pruning_present | prototype contact-aware latent support creation: recover candidate support first, then re-score incidence instead of pruning early |
| ysherman | 12 | 2 | 2 | `latent_support_creation_with_contact_recovery_gap` | raw_bridge_safe_support_under_covers_target, many_candidates_die_on_flange_contact_or_attachment, birth_recovery_blocked_by_active_count_cap, selected_pair_duplicate_pressure_present, fragment_pruning_present | prototype contact-aware latent support creation: recover candidate support first, then re-score incidence instead of pruning early |

## Conclusion

- Raw sparse arrangement under-covers both hard known cases.
- Decomposition diagnostics show strong contact/incidence pruning pressure, so the next tranche should recover support/contact before pruning.
- Runtime/main promotion remains blocked.
