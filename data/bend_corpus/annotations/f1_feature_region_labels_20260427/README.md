# F1 Feature Region Labels - 2026-04-27

These files encode manual owned-region feedback from the 1080p F1 visual review.

- `49024000_feature_region_labels.json`: complete region labels. User validated the part as 3 conventional bends; only `OB2` is accepted as a conventional bend. This proves raw F1 has a missing-support/split-recovery problem, not just duplicate overcount.
- `48991006_feature_region_labels.json`: partial region labels. User identified `OB4` and `OB5` as the same physical bend zone; the remaining regions need a cleaner manual pass because the scan/process quality is poor.

Use `scripts/summarize_f1_feature_region_labels.py` to regenerate the `*_summary.json` files.

## Follow-up Diagnostics

- `49024000_split_candidate_diagnostics.json`: label-aware split-recovery diagnostic. It uses `valid_conventional_region_deficit` and `valid_counted_feature_deficit` as recovery targets. Current result: `missing_region_recovery_needed`, with no strong split candidate among existing owned regions.
- `49024000_interface_birth_cap6_diagnostics/`: exact-blocker/interface-birth diagnostic comparing the visual-QA baseline against the `cap_6` overcomplete pool. Current result: no admissible clean missing-bend candidate, so this case still needs a stronger residual/flange-interface support proposal step.
- `49024000_residual_interface_support_candidates.json`: diagnostic-only residual/flange-interface proposal search. Current result: residual/interface signal exists, but all admissible candidates overlap into one ambiguous component, so corridor-family selection or sub-clustering is required before promotion.
- `49024000_residual_interface_subclusters.json`: diagnostic-only sub-clustering of the ambiguous residual/interface support. Current result: one plausible broad cluster, not two separated missing bend supports.
- `49024000_local_ridge_candidates.json`: diagnostic-only local curvature/normal ridge trace inside the residual/interface domain. Current result: one sharper ridge candidate, still not two separated missing bend supports.
- `49024000_residual_interface_visual_check_summary.md`: visual verdict from the 1080p candidate and sub-cluster overlays.
