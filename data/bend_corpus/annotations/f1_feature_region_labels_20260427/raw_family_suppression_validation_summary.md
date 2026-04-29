# Raw-Family Suppression Validation

This note validates the recovered-contact candidate suppression added after the `48991006` visual review.

## Purpose

The new rule suppresses a generic `ridge_pair_support` candidate when:

- an admissible recovered-contact candidate exists,
- the generic candidate is comparable in score,
- and the generic candidate is geometrically covered by an already-owned raw bend family.

The goal is to let recovered missing-middle support win on poor scans without letting raw edge-family aliases dominate the local arrangement.

## Case Results

### 48991006

- Result: positive diagnostic improvement.
- `FEIG2/RCB1` remains admitted as `offset_parallel_same_pair_candidate`.
- `JBL1`, `JBL6`, and `JBL7` are now suppressed as raw-family-covered generic ridge candidates.
- Unconstrained arrangement: `underfit_arrangement`, selecting `JBL11 + RCB1`.
- Target-one arrangement: `single_bend_arrangement_candidate`, selecting `JBL11 + RCB1`.
- Decision: keep diagnostic-only; this is a poor-scan stress case and still needs broader validation.

### 49024000

- Result: stable control.
- This case has no recovered-contact candidate in the arrangement pass, so the new suppression path is inactive.
- Rerun selected the same known diagnostic arrangement: locked `OB2/F11-F14` plus `F11-F17` and `F1-F14`.
- Decision: no observed regression from the suppression rule.

### 47959001

- Result: not a valid direct rerun control from the current 10-part decomposition snapshot.
- Stored arrangement uses flange IDs including `F20/F21`.
- Current available 10-part decomposition for `47959001` contains a different flange-ID set and does not include `F20/F21`.
- A rerun with the current decomposition therefore changes the candidate pool for input reasons, not because of raw-family suppression.
- Decision: exclude from this suppression-control conclusion until the exact original decomposition snapshot is recovered or the count-context labels are regenerated against the current flange IDs.

## Current Decision

The suppression rule is safe enough to keep in the offline diagnostic lane, but not promotion-ready. The next validation step is to regenerate or realign the `47959001` count-context labels against the current decomposition snapshot, then run the local arrangement pass again.

## Next Work

- Regenerate `47959001` local arrangement inputs against the current decomposition flange IDs.
- Run the same suppression-control check on any cleaner case that has both a recovered-contact candidate and a true direct transition candidate.
- Keep all changes offline-only until the cleaner validation set shows no hidden missed-bend regression.
