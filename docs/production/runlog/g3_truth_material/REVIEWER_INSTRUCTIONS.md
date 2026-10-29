# G3 visual truth table: reviewer instructions (plan §10 amendment 7)

You build the per-bend truth table **before anyone has seen engine output**. You see only the
raw scan, the recipe CAD and these instructions. Do not open any other run log, result,
engine directory or `*SEALED*` path unless you are a DEV-CHECK reviewer told to review the
`devcheck_SEALED/` material (the display registration's fitted placements are not in this
tree: they are never shown to reviewers).

For every row of `truth_table_TEMPLATE.json` (one row per case and recipe bend):

- `FORMED` — the scan shows this bend bent (it follows the blue FORMED reference in
  `bend_<seq>.png`, or the global views clearly show the flange folded).
- `NOT_FORMED` — the scan shows the flange still flat where this bend will be (it follows
  the orange NOT FORMED reference, or the flange lies in the plane of its parent).
- `NOT_DETERMINABLE` — the scan does not show the bend region or its flange well enough to
  decide (occluded, cut off, too sparse). Never guess.

Images per case (opaque case reference only): `raw_views.png` (the scan alone),
`cad_numbered_views.png` (the recipe CAD with numbered bend lines `<seq> · <label> · <uid6>`),
`overlay_views.png` (the scan registered to the CAD; the registration is a display aid, not
an engine — if it says LOW CONFIDENCE or looks wrong, judge from the raw views) and, for
staged cases, `bend_<seq>.png` sections.

Two reviewers fill `reviewer_1` and `reviewer_2` independently (do not look at the other
reviewer's answers). A third reviewer fills `tiebreak` only where they disagree. `final` is
the agreed label, else the tie-break. Then set `frozen_at` (UTC ISO) and list the reviewers.

Freezing (whoever assembles the table, still before any engine output is compared with it):
save it as `truth_table_FROZEN.json` next to the template (DEV-CHECK: inside
`devcheck_SEALED/`) and record it in the freeze ledger with
`python -m domains.production.engine.eval.freeze.prefreeze record-truth-table --truth-table PATH`
(the DEV-CHECK table, at the final DEV-CHECK run or by R1 only: add `--include-sealed`). The
command refuses an unfrozen table, an unknown label, a `final` that is not the agreement or the
tie-break, and a table that misses or adds a template row; it records the A1 table's path and
sha256 and the DEV-CHECK table's sha256 only, in one FREEZE entry of
`docs/production/runlog/freeze/LEDGER.json`.

Order (amendment 7, enforced): until a frozen table is recorded, `prefreeze run-physical` refuses
E3 on every truth-relevant case (groups `G3_STAGED`, `BROAD_STAGED`, `PROVISIONAL_COMPLETE`), and
the scorer rejects a table frozen or recorded after the first E3 record of such a case.

Groups (`MANIFEST.json` `group`): `G3_STAGED` (plan §5 G3; every bend has a section image),
`BROAD_STAGED` (BROAD scans outside the G3 list whose stage hints say partial; sections too),
`PROVISIONAL_COMPLETE` (corpus stage keys disagree with "complete"; sections too) and
`STATE_CHECK` (complete scans: confirm every bend is formed from the global views; mark any
bend that is not; a completeness check that no score reads, filled in the same pass).
