# Vendor machine moves — 2026-09-27 (revised for optional moves)

Latest vendor UI update: [persistent group visibility, general references and PDF, 2026-09-29](vendor-general-measurements-2026-09-29.md), published as version 8. The version-7 evidence below remains historical.

## Source state

- The vendor setup draft stores only explicitly tagged `machine_moves`; `[]` is a valid and visible no-move plan. A move is an ordered, contiguous subset of the bend sequence. Bends outside moves remain ungrouped.
- In order-tagging mode, the worker taps bends in execution order, then taps `+ מהלך` to group the bends just tagged. `בלי מהלך` leaves that chunk ungrouped before tagging another chunk. Applying the order preserves only explicit groups; editing the order can replace or remove them.
- Selecting a move highlights every member bend line, marker, and available CAD flange surface. Changing bend geometry or removing a bend clears moves rather than silently preserving an invalid plan.
- The API and public static demo reject overlapping, noncontiguous, empty, or out-of-order groups. Legacy version-6 automatic singleton partitions are normalized to no moves; legacy multi-bend groups are retained.
- Public demo: `https://sherman-vendor-bend-demo.idan82labs.chatgpt.site/#/parts/vendor-setup/corpus-37361005`, version 7, Site source commit `54efbea06c23981a14eeb98859d6a4b1af31b5e2`.

## Verified evidence

- Frontend TypeScript check passed; static Sites build completed.
- Five-part browser regression passed, including tag-B2/B4-then-move grouping, dual-marker highlighting, save/reload, removal of all moves, and legacy singleton migration.
- Touch regression passed for orbit, pinch, and pan.
- Eleven isolated API contract tests passed; Python service compiled.
- Version 7 production deployment succeeded. No additional live-page fetch was needed for publication verification.

## Boundary and next action

This is a CAD planning demo, not machine-feasibility validation or scan-based quality inspection. The worker remains responsible for deciding which bends share a move. Future work may add machine/tooling constraints after those inputs are specified; do not infer them from the CAD alone. No physical data, held-outs, email parts, or QC verdicts were used for this change.
