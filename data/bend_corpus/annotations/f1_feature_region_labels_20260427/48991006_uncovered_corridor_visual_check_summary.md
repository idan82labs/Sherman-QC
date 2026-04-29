# `48991006` Uncovered Corridor Visual Check

Date: 2026-04-29

## Trigger

Manual review correction:

```text
48991006 is missing a whole bend in the middle of the part.
It is totally unmarked.
```

This changes the interpretation of the previous `48991006` result. The issue is not only duplicate same-zone support (`OB4` / `OB5`) and lower-edge aliasing. There is also a missed interior bend that the current residual/ridge/junction diagnostic did not recover.

## Inputs

- Source decomposition: `data/bend_corpus/evaluation/f1_visual_qa_after_debug_alias_patch_20260426/known/02_48991006_48991006_A_4/decomposition_result.json`
- Existing junction arrangement: `48991006_junction_bend_arrangement.json`
- New scan-wide corridor diagnostic: `48991006_uncovered_bend_corridors.json`

## Result

The scan-wide uncovered-corridor diagnostic found many broad candidates:

```text
status = candidate_pool_has_uncovered_corridors
candidate_count = 18
admissible_candidate_count = 18
```

Top candidates are dominated by broad top/lower flange corridors and aliases:

```text
UBC1: F5-F9, 79 atoms
UBC2: F31-F8, 150 atoms
UBC3: F32-F8, 156 atoms
UBC4: F5-F8, 98 atoms
UBC5: F1-F5, 246 atoms
```

## Visual Verdict

The 1080p render confirms this diagnostic is too permissive:

- top candidates cover broad top and lower edge bands;
- multiple candidates are aliases of the same large supports;
- the missed middle bend identified by manual review is not isolated as a clean countable line instance;
- this should not be used for count recovery or promotion.

## Decision

Reject scan-wide uncovered-corridor ranking as promotion logic.

Keep the output as evidence for the next formulation gap:

```text
48991006 requires interior-support discovery / middle-bend targeting.
```

The next useful diagnostic should not search every flange pair globally. It should first identify under-explained interior ridge/crease evidence inside the main panel, then fit a local bend-line object there. Candidate ranking must include:

- interior-vs-boundary location;
- uniqueness relative to existing top/lower edge corridors;
- compactness around the missing middle bend;
- explicit rejection of broad flange-surface bands;
- optional user-marked region of interest if the middle bend remains visually ambiguous.

## Artifact Paths

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/48991006_uncovered_corridor_visual_check/uncovered_bend_corridors_top_candidates_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/48991006_uncovered_corridor_visual_check/uncovered_bend_corridors_render_sidecar.json
```
