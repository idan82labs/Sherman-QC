# 49125000 Sparse Arrangement Visual Check

Generated from:

- `49125000_raw_crease_sparse_arrangement.json`
- `49125000_multiscale_raw_point_crease_bridge_validation.json`
- Render: `49125000_sparse_raw_crease_arrangement_visual_check/raw_crease_sparse_arrangement_1080p.png`

## Verdict

Reject as a conventional bend-line promotion candidate.

## Reasoning

Manual validation marks `49125000` as a rounded/rolled mixed-transition part with two counted bends. The sparse raw-crease arrangement reports `1 selected / 2 safe`, selecting `FINE_RPC2` and suppressing `STANDARD_RPC2` as the same `F10-F2` family.

Visual inspection of the 1080p render shows the selected raw-crease support riding a broad curved/rolled surface, not two clean maximal conventional bend-line instances. This makes the case useful as a rounded-geometry negative control, but not as evidence that sparse raw-crease arrangement is promotion-safe.

## Current Gate

- Quality decision: `promotion_needs_review`
- Sparse status: `sparse_arrangement_underfit`
- Offline recommendation: `hold_underfit_support_creation_needed`

## Next Use

Keep `49125000` in the validation set as a router/semantics test:

- rounded or rolled surfaces must not be counted by the sharp conventional bend-line gate;
- sparse raw-crease support may be diagnostic evidence, but should not override the existing F1/control result;
- promotion should require a separate rolled/mixed-transition semantic class, not a conventional bend-line shortcut.
