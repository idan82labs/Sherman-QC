# 48991006 Interior Bend Gap Visual Check

Manual feedback changed the blocker interpretation for `48991006`: the case is not only duplicate raw support around the side/lower edge. A whole middle bend is unmarked.

## Diagnostic

- Input: current F1 decomposition from the known-10 review pack.
- Method: scan interior-only crease/ridge search that excludes already-owned bend atoms and scores remaining atoms by curvature plus local normal contrast.
- Output: `48991006_interior_bend_gaps.json`.
- Render: `48991006_interior_gap_visual_check/interior_bend_gaps_top_candidates_1080p.png`.

## Result

- `selected_atom_count=26`
- `candidate_count=7`
- `admissible_candidate_count=2`
- `merged_hypothesis_count=1`
- Admissible fragments: `IBG1`, `IBG2`
- Merged hypothesis: `IBH1`

## Visual Verdict

The admissible fragments do not reproduce the earlier lower-edge alias failure. They sit on the interior panel area where the missed middle bend is expected, but appear as two adjacent fragments of one structure. The merged `IBH1` line is therefore the useful next diagnostic object.

This is promising, but not promotion-ready. It still lacks:

- flange-pair incidence,
- junction/corner ownership semantics,
- endpoint support tied to visible bend boundaries,
- integration with existing owned bend regions and duplicate suppression.

## Decision

Keep as a diagnostic signal for the missed middle bend on `48991006`. The next implementation step is to feed `IBH1` into the junction-aware bend-line arrangement as an interior birth candidate, then require flange-pair/incidence validation before it can affect count or rendering.
