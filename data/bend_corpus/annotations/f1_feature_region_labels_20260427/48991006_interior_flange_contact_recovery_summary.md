# 48991006 Interior Flange Contact Recovery

This diagnostic tests the next blocker after `IBH1` was visually found but rejected by the standard graph-adjacency incidence gate.

Instead of relying only on atom-graph boundary contact, it searches for nearby flange atoms around each interior missed-bend fragment and scores flange pairs using:

- two-sided nearby flange atom support,
- fragment axis agreement with `normalize(n_a x n_b)`,
- fragment normal-arc agreement with the two flange normals,
- side-balance and local line-distance checks.

## Result

- Output: `48991006_interior_flange_contact_recovery.json`
- Status: `fragment_contact_recovered`
- Candidate rows evaluated: `3`
- Validated local pairs: `1`

Validated pair:

- Source: `IBG1`
- Pair: `F1-F5`
- Axis delta: `3.895884°`
- Mean normal-arc cost: `0.105921`
- Nearby flange counts: `F1=134`, `F5=38`
- Median line distance: `F1=10.03 mm`, `F5=13.90 mm`

Rejected:

- `IBG2`: same best pair `F1-F5`, but only one nearby `F5` atom and weak side balance.
- `IBH1`: enough nearby `F1/F5` atoms, but merged-axis PCA is wrong (`axis_mismatch`), proving the merged interior hypothesis should not be the countable object.

## Decision

This is the first useful local recovery signal for the missed middle bend on `48991006`.

Do not promote it directly. The next diagnostic arrangement pass should add `IBG1/F1-F5` as a recovered interior birth candidate and test whether it can coexist with existing raw `F1-F5` regions without duplicating the same physical bend.
