# F1 Merge Readiness - 2026-04-26

## Summary

This branch adds the offline F1 patch-graph research lane, renderable owned-bend diagnostics, and product-path structural profiling/scan-overlay diagnostics.

F1 is not promoted to runtime in this branch. The current production/control zero-shot path remains the safety gate for accepted counts.

## Recommended Review Slices

### Slice A: Offline F1 Research Lane

Commits:

- `0da1aea Add offline F1 patch graph research lane`
- `c9ea78b Improve owned bend render annotations`
- `6d53743 Suppress tiny clustered owned bend fragments`
- `784e461 Limit clustered bend suppression to render export`

Review note: `6d53743` and `784e461` should be reviewed together. The final behavior is render-only clustered marker suppression; solver ownership and count metrics are not suppressed.

Scope:

- Offline patch-graph latent decomposition prototype.
- Offline scan-family router for F1 solver profile selection.
- Known-count and visual-review validation scripts.
- Renderable owned bend regions for manual inspection.

Runtime stance:

- Offline/research only.
- No product API replacement.
- Raw F1 counts are not authoritative runtime counts.

### Slice B: Product Diagnostics / Scan Overlay

Commit:

- `8529a4d Add part profiling and scan overlay diagnostics`

Scope:

- CAD/scan structural profiling payloads.
- Additional bend inspection run details for structural diagnostics.
- Scan-only renderable bend object API types.
- Frontend scan-overlay viewer/card support and local demo route.

Runtime stance:

- Diagnostic visibility only.
- Review separately from F1 algorithmic promotion.

## Validation Snapshot

Local validation before opening the draft PR:

- Full backend regression: `849 passed, 3 skipped, 35 warnings`.
- Frontend production build: passed.
- F1 known10 accepted exact accuracy: `1.0`.
- F1 known10 accepted range contains truth: `1.0`.
- F1 known10 accepted broadened cases: `0`.
- Strict marker audit after render-only suppression: `4/4` pass, `0` issues.

GitHub PR check:

- `frontend-build`: passed.

## Known Blockers

- `47959001` raw F1 still undercounts; accepted output remains protected by control guard.
- `10839000` raw F1 still undercounts; accepted output remains protected by control guard.
- `48991006` and `49024000` are raw F1 singleton promotion candidates, but still require manual visual review before exact promotion.
- Rounded, raised, and non-conventional bend semantics still need a separate policy/validation pass.
- Partial scans must continue to target visible observed bends, not inferred full-part totals.

## Main Merge Gate

Before this branch is merged as anything beyond offline diagnostics:

- Confirm F1 remains offline-only and control-gated.
- Confirm no runtime path treats raw F1 as authoritative count.
- Confirm product diagnostics/scan-overlay changes are accepted as a separate review slice.
- Re-run backend tests and frontend build after conflict resolution.

## Post-Merge Work

- Improve raw F1 support ownership on `47959001` and `10839000`.
- Generate review packs for `48991006` and `49024000`.
- Expand strict visual marker audit beyond the current focused four-case pack.
- Add a partial-scan observed-bend validation lane.
- Decide policy for rounded and raised bend features before runtime promotion.
