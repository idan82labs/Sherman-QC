# Hard Known Raw-Crease Underfit Visual Check

Generated from:

- `10839000_raw_crease_sparse_arrangement.json`
- `10839000_multiscale_raw_point_crease_bridge_validation.json`
- `ysherman_raw_crease_sparse_arrangement.json`
- `ysherman_multiscale_raw_point_crease_bridge_validation.json`
- 1080p renders copied to the iCloud review folder:
  - `10839000_sparse_arrangement_1080p.png`
  - `ysherman_sparse_arrangement_1080p.png`

## Verdict

Reject as hard-case recovery logic. Keep both cases as underfit controls.

## 10839000

Manual target under current semantics is `8` conventional bends, with `2` raised/form features tracked separately.

Raw-crease result:

- Raw admissible before dedupe: `29`
- Deduped raw candidates: `9`
- Bridge-safe candidates: `4`
- Sparse selected families: `4`
- Gate: `limited_observed_support`

Visual check: selected supports sit on real local crease/edge evidence, but they cover only one side/subset of the rectangular repeated-bend structure. They do not cover the visible eight conventional bend target and should not be used to override F1/control semantics.

## ysherman

Manual target is `12` conventional bends.

Raw-crease result:

- Raw admissible before dedupe: `18`
- Deduped raw candidates: `11`
- Bridge-safe candidates: `2`
- Sparse selected families: `2`
- Gate: `limited_observed_support`

Visual check: selected supports are localized on one interior/central repeated feature family, not the full set of twelve bend-line instances. This confirms raw-crease sparse arrangement is not yet a repeated-near-90 recovery method.

## Conclusion

The raw-crease sparse lane is currently useful as:

- a compact clean-case diagnostic (`37361005`);
- an abstain/negative-control lane (`48963008`, `44211000`, `15651004`);
- a scan-support quality gate for poor or partial scans (`47959001`, `49001000`);
- a semantics blocker for rounded/rolled surfaces (`49125000`).

It is not yet a promotion path for hard repeated-near-90 cases. The next solver work for hard cases should focus on support creation / latent object decomposition, not sparse raw-crease arrangement promotion.
