# F1 strict manual overlay review — 2026-07-15

## Corrected conclusion

The previous `10/13` hard-visual checkpoint was too permissive. It credited
colored lines that sat on a visible 2-D edge without proving that the edge was a
two-flange dihedral bend. Under the corrected rule, only case 12 is presently a
clean visual pass from the supplied orthographic audits.

```text
reviewed:      13
perfect:        1
questionable:   4
failed:         8
perfect rate:  1/13 = 7.69%
```

Questionable cases are 03, 05, 08, and 09. Clear failures are 01, 02, 04, 06,
07, 10, 11, and 13. The machine-readable line-by-line review is in
`output/experiments/f1-strict-manual-review-20260715/review.json`.

## Corrected acceptance rule

A colored line is not a bend merely because it overlaps a visible outline.
Count a marker only when rotatable 3-D evidence or a local orthogonal
cross-section establishes two adjacent sheet flanges and a sustained dihedral
transition. Reject:

- free perimeter and silhouette edges;
- hole, cutout, notch, or emboss outlines;
- face-interior ridges without two incident flanges;
- contained or fragmented aliases of a longer physical bend;
- unsupported extensions across scan gaps or empty space;
- partial lines when the physical bend is a longer maximal instance.

## Labeling workflow

The Bend Label Studio is embedded in the 3-D overlay for every pipeline job,
including jobs with zero F1 markers. A reviewer can rotate the model, click two
surface points to create each human bend axis in `reference_mesh_raw`, classify
every F1 marker as `real_bend`, `not_bend_outline`, or `uncertain`, save the
review atomically with the job, and export the typed JSON. A job cannot be
marked review-complete in the UI until every displayed F1 marker has a decision.

These labels are evaluation data. They must remain outside blind inference and
be split by part before any future held-out accuracy claim.
