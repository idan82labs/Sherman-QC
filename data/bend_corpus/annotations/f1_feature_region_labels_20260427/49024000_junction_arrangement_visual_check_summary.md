# `49024000` Junction Arrangement Visual Check

Date: 2026-04-29

## Inputs

- `49024000_feature_region_labels.json`
- `49024000_local_ridge_candidates.json`
- `49024000_junction_bend_arrangement.json`
- Source decomposition: `data/bend_corpus/evaluation/f1_visual_qa_after_debug_alias_patch_20260426/known/01_49024000_49024000_A/decomposition_result.json`

## Result

The local junction-aware arrangement diagnostic selected:

```text
H3_locked_plus_2_new
```

Selected countable bend-line candidates:

```text
JBL4: F11-F14, locked accepted OB2
JBL5: F11-F17, proposed missing bend-line instance
JBL2: F1-F14, proposed missing bend-line instance
```

Rejected/chord candidates:

```text
JBL3: F1-F17, rejected as possible chord
JBL1: F1-F11, rejected as axis mismatch / possible chord
```

This matches the professor-proposed sparse topology:

```text
F17 -- F11 -- F14 -- F1
```

with `F1-F17` treated as a diagonal/chord explanation rather than a physical conventional bend.

## Visual Verdict

The close-up render shows a useful diagnostic signal:

- the method no longer treats the connected U/corner ridge as one countable bend;
- it keeps manually accepted `OB2`;
- it proposes two additional flange-pair-conditioned line instances;
- it leaves leftover ridge atoms as uncounted junction/corner support;
- it rejects the broad diagonal/chord candidate.

This is not promotion-ready yet:

- candidate support remains broad around the central corner;
- fitted line endpoints are approximate;
- the selected lines need stronger path skeleton / endpoint ownership before becoming final F1 owned bend regions;
- this has only been tested on `49024000`.

## Artifact Paths

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_junction_arrangement_visual_check/junction_arrangement_overview_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_junction_arrangement_visual_check/junction_arrangement_closeup_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_junction_arrangement_visual_check/junction_arrangement_render_sidecar.json
```

## Decision

Keep as a diagnostic proof-of-concept.

Next algorithmic work should convert this from a one-case diagnostic into a reusable local arrangement module with:

- ridge skeleton path segmentation;
- explicit `junction_transition` ownership;
- line endpoint fitting;
- selected-pair duplicate/chord suppression;
- local MDL comparison across H1/H2/H3/H4 arrangements;
- visual gates on `48991006`, `10839000`, `47959001`, `ysherman`, and `49204020_04`.
