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
- `49024000_junction_bend_arrangement.json`: diagnostic-only professor-guided junction-aware bend-line arrangement. Current result: selected `H3_locked_plus_2_new`, keeping `OB2` (`F11-F14`), proposing `F11-F17` and `F1-F14` as the two missing conventional bend-line instances, and rejecting `F1-F17` as a possible chord.
- `48991006_residual_interface_support_candidates.json`: diagnostic-only residual/flange-interface proposal search. Current result: partial candidate pool, with only 3 admissible candidates for the unresolved manual-label deficit.
- `48991006_local_ridge_candidates.json`: diagnostic-only local curvature/normal ridge trace. Current result: partial ridge signal, with 2 admissible ridge candidates.
- `48991006_junction_bend_arrangement.json`: diagnostic-only junction-aware arrangement. Current result: `two_bend_arrangement_conflicted`; selected new line candidates overlap heavily and are not promotion-safe.
- `48991006_uncovered_bend_corridors.json`: scan-wide uncovered-corridor diagnostic after manual feedback that a whole middle bend is unmarked. Current result: too permissive; broad top/lower bands dominate and the missed middle bend is not isolated.
- `48991006_interior_bend_gaps.json`: diagnostic-only interior crease/ridge search after manual feedback that the middle bend is totally unmarked. Current result: two admissible adjacent fragments (`IBG1`, `IBG2`) merged into one interior-gap hypothesis (`IBH1`), visually located on the interior panel rather than the earlier lower-edge alias zone.
- `48991006_junction_bend_arrangement_with_interior.json`: junction-aware arrangement rerun with `IBH1` fed in as an interior birth candidate. Current result: safe rejection; `36` interior flange-pair attachments were tested and all were rejected by incidence/axis/normal validation, so the interior support remains diagnostic evidence only.
- `48991006_interior_flange_contact_recovery.json`: local proximity-based flange/contact recovery around the interior missed-bend fragments. Current result: `fragment_contact_recovered`; `IBG1` validates as a local `F1-F5` contact candidate (`axis_delta=3.9°`, `mean_normal_arc_cost=0.106`, nearby counts `F1=134`, `F5=38`), while `IBG2` and merged `IBH1` remain rejected.
- `47959001_count_context_region_labels.json` / `47959001_count_context_summary.json`: count-context diagnostic inputs, not per-region manual truth. They encode expected visible observed bends `5`, raw F1 owned regions `4`, and one missing-support recovery target.
- `47959001_residual_interface_support_candidates.json`: diagnostic-only residual/flange-interface proposal search. Current result: one admissible candidate covers the one-missing-bend target.
- `47959001_local_ridge_candidates.json`: diagnostic-only local curvature/normal ridge trace. Current result: one admissible ridge candidate covers the one-missing-bend target.
- `47959001_junction_bend_arrangement.json`: diagnostic-only junction-aware arrangement. Current result: `single_bend_arrangement_candidate`, selecting `JBL7` / `F1-F20` with no conflicts.
- `49024000_residual_interface_visual_check_summary.md`: visual verdict from the 1080p candidate and sub-cluster overlays.
- `49024000_junction_arrangement_visual_check_summary.md`: visual verdict for the junction-aware bend-line arrangement overlays. Current decision: promising diagnostic proof-of-concept, not promotion-ready.
- `48991006_junction_arrangement_visual_check_summary.md`: visual verdict for the `48991006` junction-aware overlays. Current decision: reject as promotion candidate; keep as duplicate/poor-scan fail-safe case.
- `48991006_uncovered_corridor_visual_check_summary.md`: visual verdict for the scan-wide uncovered-corridor overlays. Current decision: reject as promotion logic; use as evidence that this case needs interior-support discovery.
- `48991006_interior_gap_visual_check_summary.md`: visual verdict for the interior-gap overlays. Current decision: keep as a promising diagnostic signal for the missed middle bend, but do not promote until it is tied to flange-pair incidence and converted into a countable bend-line instance.
- `48991006_junction_arrangement_with_interior_summary.md`: incidence-gated verdict after feeding the interior signal back into the arrangement solver. Current decision: promotion remains blocked; next work is flange-pair/contact recovery around `IBH1`, not more raw interior detection.
- `48991006_interior_flange_contact_recovery_summary.md`: verdict for the local contact-recovery diagnostic. Current decision: use `IBG1/F1-F5` as a diagnostic interior birth candidate in the next arrangement pass, but keep it offline-only until it is reconciled with existing raw F1 `F1-F5` regions and duplicate support.
- `47959001_junction_arrangement_visual_check_summary.md`: visual verdict for the `47959001` junction-aware overlays. Current decision: promising one-missing-bend recovery candidate, diagnostic-only.
