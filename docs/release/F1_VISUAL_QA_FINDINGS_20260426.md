# F1 Visual QA Findings - 2026-04-26

This note records assistant visual inspection of the 1080p review renders after manual feedback. These observations are not a replacement for human bend-count truth; they identify render/owned-region failure modes to drive the next F1 tranche.

## Important Render Semantics

- Several contact sheets show `no accepted dots` because the accepted result is a range, not an exact count.
- In those cases, the visible lines/markers are raw F1 debug owned regions, not accepted final bend dots.
- Current final-dot suppression is correct for range outputs; the visible issue is mainly raw owned-region quality and decomposition semantics.

## `47959001_01PREL_scan1`

- Known/manual target in current annotations: visible observed conventional bends = `5`.
- Current accepted output: range `[4,6]`, exact `None`.
- Visual result: raw F1 shows `4` owned support regions.
- Assessment: raw supports mostly sit on plausible bend zones, but the scan is missing at least one visible conventional bend compared with the 5-bend human target.
- Status: range-safe but not exact-promotion-safe.
- Failure mode: under-separation / missing compact-neighbor bend.

## `10839000_01_full_scan`

- Manual semantics: `8` conventional bends plus `2` raised/form features.
- Current raw F1: `9` owned support regions.
- Visual result: raw markers cluster around the major frame/corner zones. Several labels appear to share the same local bend/feature zone rather than representing distinct conventional bends.
- Assessment: raw F1 is not reliably separating the conventional bend pair structure from raised/form or fragment features. It finds bend zones, but not the correct conventional-object decomposition.
- Status: keep protected by control/accepted logic; do not promote raw F1 exact count.
- Failure mode: conventional-vs-raised feature ambiguity plus duplicate/merged support at corner clusters.

## `10839000_01_scan_1`

- Manual target in current annotations: `4` visible conventional bends plus `2` raised/form features.
- Current accepted output: range `[2,4]`, exact `None`.
- Visual result: raw F1 shows `4` owned support regions.
- Assessment: accepted range contains the conventional target, but final dots are correctly suppressed because the output is not exact. Raw marks are plausible bend-zone evidence, not promotion evidence.
- Status: range-safe; exact promotion not allowed.
- Failure mode: limited partial-scan confidence and raised/form ambiguity.

## `10839000_01_scan_2`

- Manual target in current annotations: `6` visible conventional bends plus `2` raised/form features.
- Current accepted output: range `[6,8]`, exact `None`.
- Visual result: raw F1 shows `10` owned support regions with visible duplicate clusters around the same physical bend/feature areas.
- Assessment: accepted range contains the conventional target, but raw F1 over-fragments/duplicates support and cannot be promoted.
- Status: range-safe; raw exact not safe.
- Failure mode: duplicate support and conventional-vs-raised feature ambiguity.

## Next Algorithmic Targets

1. Keep accepted final-dot suppression for range outputs.
2. Add a clearer visual distinction between accepted final bends and raw debug support.
3. Improve duplicate support collapse where multiple OB labels sit on the same physical bend zone.
4. Add a feature-class gate for raised/form features so they do not compete directly with conventional bend count.
5. Improve compact-neighbor separation on `47959001`, where the accepted range is safe but raw F1 still misses one human-counted visible bend.

## Follow-up Patch

- Raw F1 render titles now explicitly say `Debug Raw F1` and `not final dots`.
- Owned-region manifests now include `render_debug_semantics=debug_raw_owned_regions_not_final_accepted_bends`.
- Render/export now suppresses conservative same-selected-pair alias markers before screenshots and manifest export. This targets cases like `48991006`, where two nearby OB labels can sit on the same physical bend.
- The suppression is artifact-only. It does not change solver ownership, F1 candidate ranges, accepted exact/range logic, runtime zero-shot, or product APIs.
- Focused regression: `82 passed` across patch-graph decomposition, marker audit, accuracy validation, and F1 runner tests.

## Regenerated Review Pack

- Generated focused 1080p review pack at `/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/visual_qa_after_debug_alias_patch_20260426`.
- `49024000`: accepted range remains `[3,16]` with corrected manual truth `3`; raw F1 still solves as `4`, but the review reason is now `raw_f1_stable_singleton_conflicts_with_known_truth`, not a promotion candidate.
- `48991006`: accepted range remains `[8,12]`; raw F1 still solves as `9`, but render/export suppresses three raw marker aliases/fragments and shows `6` debug support regions.

## Rejected Solver Cleanup Probe

- A no-render probe forced selected-pair duplicate cleanup into the solver/count path with relaxed alias thresholds.
- Result: rejected. It did not fix `49024000` (`4` stayed `4`) and damaged protected/sensitive cases: `48991006` collapsed to `6`, `47959001` collapsed to `1`, and `10839000` narrowed to `9`.
- Decision: keep alias/cluster cleanup in render/export artifacts only for now. Solver-count cleanup needs a stronger physical split/merge/support model before it can be trusted.

## Region-Level Label Encoding

- Added owned-region feature labels under `data/bend_corpus/annotations/f1_feature_region_labels_20260427/`.
- `49024000`: complete labels show only `OB2` is a validated conventional bend; `OB1`, `OB3`, and `OB4` are fragments/false supports. Summary reports `valid_conventional_region_deficit=2`, proving this is missing-support/split-recovery, not only duplicate overcount.
- `48991006`: partial labels encode the user finding that `OB4` and `OB5` sit on the same physical bend zone, with `OB5` marked duplicate and other regions pending because scan/process quality is poor.

## Rejected Support-Birth Cap Probe

- A `49024000` no-render probe relaxed the birth-rescue active-count cap.
- Result: rejected. Counts climbed with the cap (`5`, `6`, `8`, then `15-16`) instead of converging to the validated `3`.
- Decision: do not loosen birth caps. The next solver work needs object-level support competition that can choose the right missing supports while rejecting large false supports.

## Split-Recovery Diagnostic

- Updated `scripts/analyze_f1_owned_region_split_candidates.py` so manual/feature-label deficits drive recovery diagnostics, not just raw owned-region count.
- `49024000` now reports `capacity_deficit=2` from `valid_conventional_region_deficit` / `valid_counted_feature_deficit`, while `owned_region_capacity_deficit=0`.
- Result: no existing owned region is a strong balanced split candidate. `OB1` has enough span but no large absolute along-axis gap; `OB2`, `OB3`, and `OB4` are too small or unbalanced.
- Decision: reject “split existing owned region” as the next fix for `49024000`. This case needs missing support discovery from residual/flange-interface evidence, not splitting the current OB set.

## Interface-Birth Candidate Diagnostic

- Ran the existing exact-blocker/interface-birth diagnostic using the raw visual-QA result as baseline and the `cap_6` overcomplete run as candidate pool.
- Result: rejected. The diagnostic found no admissible clean missing-bend candidate: `selection_status=no_admissible_candidate`, `corridor_coverage_status=no_corridor_gap`, and `family_residual_repair_eligible=false`.
- Decision: current overcomplete interface-birth candidates are not sufficient for `49024000`. The next algorithmic tranche should create new candidate support from flange-interface/residual geometry, then score it against manual-label deficits before any exact promotion.

## Residual / Interface Support Proposal Diagnostic

- Added `scripts/analyze_f1_residual_interface_support_candidates.py`, a diagnostic-only search over residual/flange-interface atoms. It excludes manually accepted conventional support and treats manually rejected raw OB atoms as evidence only, not as countable bends.
- `49024000` result: `recovery_deficit=2`, `candidate_count=237`, `admissible_candidate_count=4`, `status=ambiguous_overlapping_candidate_pool`.
- The four admissible candidates all conflict into one overlapping component (`RIC1`-`RIC4`), with high atom overlap / near-centroid conflicts. This means the residual/interface signal exists, but it is not yet separated into two physical missing bend objects.
- Decision: next solver tranche should add corridor-family selection or sub-clustering over residual/interface support before any birth promotion. The current diagnostic should not be used to accept exact `3` yet.

## Residual / Interface Sub-Cluster Visual Gate

- Added `scripts/cluster_f1_residual_interface_candidates.py` to test whether the ambiguous residual/interface support separates into smaller physical corridor families.
- `49024000` result: `candidate_atom_count=319`, `connected_component_count=1`, `cluster_count=1`, `plausible_cluster_count=1`, `status=partial_subcluster_signal`.
- Generated 1080p overlays in iCloud under `feature_region_labels_20260427/49024000_residual_interface_subcluster_visual_check/`.
- Visual inspection confirms the sub-cluster remains one broad same-zone support patch, not two separated missing conventional bends.
- Decision: reject this sub-cluster method as promotion logic. The next game plan needs a different support partition cue, likely local normal/curvature ridge tracing or manual/professor-guided interface-line constraints, before exact recovery on `49024000`.

## Local Ridge-Tracing Visual Gate

- Added `scripts/trace_f1_local_ridge_candidates.py` to trace high curvature / high normal-contrast atoms inside the residual/interface candidate domain.
- `49024000` result: `domain_atom_count=319`, `ridge_atom_count=118`, `candidate_count=1`, `admissible_candidate_count=1`, `status=partial_ridge_signal`.
- Generated 1080p overlays in iCloud under `feature_region_labels_20260427/49024000_local_ridge_visual_check/`.
- Visual inspection: ridge tracing is sharper than broad residual/interface support, but the output is still one U/corner-shaped ridge around the same zone. It does not separate into the two missing conventional bend lines.
- Decision: reject local ridge tracing alone as promotion logic. The next useful step is likely professor guidance or an explicit interface-line/support assignment formulation that can separate connected corner ridges into distinct conventional bend objects.

## Junction-Aware Bend-Line Arrangement Diagnostic

- Added `scripts/solve_f1_junction_bend_arrangement.py` as a professor-guided diagnostic-only local arrangement solver for the connected `49024000` U/corner ridge.
- The countable object is now tested as a maximal flange-pair-conditioned conventional bend-line instance inside an uncounted junction/corner complex, not as a connected support patch.
- `49024000` result: selected `H3_locked_plus_2_new`, keeping locked accepted `OB2` (`F11-F14`), proposing `F11-F17` and `F1-F14` as the two missing conventional bend-line instances, and rejecting `F1-F17` as a possible chord.
- Generated 1080p overlays in iCloud under `feature_region_labels_20260427/49024000_junction_arrangement_visual_check/`.
- Visual inspection: this is a meaningful topology-level improvement over residual support, sub-clustering, and plain ridge tracing. It separates the connected ridge into a sparse three-line arrangement plus leftover uncounted junction support.
- Follow-up refinement: added connected-component/path selection, robust line-core atoms, explicit line endpoints, exported `junction_transition_atom_ids`, and hard rejection of chord-labeled candidates unless a future unique-support proof is added.
- Decision: keep as a diagnostic proof-of-concept only. The selected lines are now more compact, but support is sparse and endpoints are still diagnostic, so this must be turned into a reusable local arrangement module with real skeleton/path/endpoints before any F1 promotion.

## Cross-Scan Junction Diagnostic: `48991006`

- Ran the same residual-interface, local-ridge, and junction-arrangement chain on `48991006`.
- Residual-interface result: `partial_candidate_pool`, with only `3` admissible candidates for the unresolved manual-label deficit.
- Local-ridge result: `partial_ridge_signal`, with `2` admissible ridge candidates.
- Junction-arrangement result: `two_bend_arrangement_conflicted`, selecting locked accepted `OB4` plus `F1-F5` and `F31-F5` new line candidates, but those two candidates overlap heavily (`atom_iou=0.636364`).
- Visual inspection confirms the selected new line candidates mostly sit on the same lower-edge support, so this is not a valid separated-bend arrangement.
- Manual follow-up: a whole middle bend is also totally unmarked, so this case is not only duplicate/alias cleanup.
- Added `scripts/analyze_f1_uncovered_bend_corridors.py` and `scripts/render_f1_uncovered_bend_corridors.py` to test scan-wide missed-corridor discovery.
- Scan-wide uncovered-corridor result: rejected. It finds broad top/lower bands and aliases, but does not isolate the missed middle bend as a clean countable line instance.
- Added `scripts/analyze_f1_interior_bend_gaps.py` and `scripts/render_f1_interior_bend_gaps.py` to test an interior-only missed-bend search that excludes already-owned bend atoms.
- Interior-gap result: promising diagnostic signal. It finds two adjacent admissible fragments (`IBG1`, `IBG2`) on the interior panel area and merges them into one candidate hypothesis (`IBH1`), instead of repeating the lower-edge alias failure.
- Fed `IBH1` back into the junction-aware arrangement as an interior birth input. Result: safe rejection, with `36` tested flange-pair attachments and `0` incidence-validated candidates.
- Decision: do not promote. Keep `48991006` as a poor-scan / duplicate-support / missed-interior-bend fail-safe case. The next blocker is local flange/contact recovery around `IBH1`, not more raw interior detection.

## Cross-Scan Junction Diagnostic: `47959001`

- Ran the residual-interface, local-ridge, and junction-arrangement chain on the partial-scan `47959001` compact-neighbor blocker.
- This case has count-level truth but no per-OB manual labels, so a count-context input was added instead of pretending region labels exist: expected visible observed bends `5`, raw F1 owned regions `4`, recovery target `1`.
- Residual-interface result: `candidate_pool_covers_deficit`, with exactly `1` admissible candidate.
- Local-ridge result: `ridge_candidates_cover_deficit`, with exactly `1` admissible ridge candidate.
- Junction-arrangement result: `single_bend_arrangement_candidate`, selecting `JBL7` / `F1-F20` with no conflicts.
- Visual inspection: the selected `F1-F20` compact line core sits on a visible diagonal bend/ridge zone and is plausible as the missing visible bend.
- Decision: promising diagnostic-only recovery candidate. The next implementation step should aggregate current raw owned regions plus non-conflicting junction-line candidates while preserving partial-scan observed-bend semantics.

## Remaining Before Promotion

- Keep `49024000` treated as a 3-bend manual-truth case; only `OB2` was manually accepted in the previous raw debug view.
- Keep `48991006` as poor-scan / duplicate-owned-region evidence. The regenerated render is less noisy, but still not clean enough for exact promotion.
- Do not promote raw F1 exact output from range cases. Promotion remains blocked until accepted final markers match a validated exact count.
