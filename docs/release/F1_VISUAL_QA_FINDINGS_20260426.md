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

## Remaining Before Promotion

- Keep `49024000` treated as a 3-bend manual-truth case; only `OB2` was manually accepted in the previous raw debug view.
- Keep `48991006` as poor-scan / duplicate-owned-region evidence. The regenerated render is less noisy, but still not clean enough for exact promotion.
- Do not promote raw F1 exact output from range cases. Promotion remains blocked until accepted final markers match a validated exact count.
