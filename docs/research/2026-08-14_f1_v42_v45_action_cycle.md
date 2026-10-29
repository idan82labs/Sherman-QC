# F1 V42–V45 action cycle

Date: 2026-08-14

## Outcome

This cycle produced one retired recall rescue, two retained precision mechanisms,
and one corrected research-only paired-return completion.

- **V42 is retired.** It restored one U02 candidate that was geometrically
  plausible in the topology certificate but visibly lay on an open
  rim/thickness edge. Evaluator-only evidence confirmed that U02 already had
  the correct four bends and that the fifth line was false.
- **V43 is retained as a V42-specific precision guard.** It requires at least
  0.25 source support on each transverse side of a V42 restoration. It vetoes
  the U02 rim candidate at 0.155086 and preserves the U01 true-but-duplicate
  candidate at 0.403276. U01 is still suppressed by incumbent-first NMS.
- **V44 is retained.** It composes V43 with the already retained V27 collinear
  fragment stitch. U04 changes from six selected fragments to three physical
  axes; U01/U02/U03 remain at 5/4/3.
- **V45 is a corrected research successor to retired V28.** It recovers the
  fourth U04 paired-return axis while canonically sorting all source rows and
  running mandatory incumbent-first output NMS. U01/U02/U03 remain unchanged;
  U04 becomes four physical axes.

The final batch counts are 5/4/3/4, matching the evaluator-only reported counts.
Visual inspection shows supported structural folds for all outputs. This is
strong mechanism evidence, but U04 remains a post-hoc target case rather than a
new independent promotion measurement.

## Visual adjudication

Every final scan overlay was inspected at native resolution.

- U01: five visible structural folds; no V43/V44/V45 addition.
- U02: four base-to-wall tray folds. The V42 fifth line on the open top rim is
  absent under V43.
- U03: three close, distinct return-fold axes; unchanged.
- U04: two folds on each long edge. V44 merges four collinear fragments into
  the full right-base axis; V45 adds the adjacent right wall-to-return axis.
  No broad-face, hole, or perimeter alias is visible.

Primary visual:
`output/research/f1-v45-deterministic-paired-return-v1/visuals/U04_v45_overlay_4096.png`

## Verification

- Combined focused suite: **21 passed** in 60.83 seconds.
- Real U04 679-row permutation check: exact V45 result hash under original and
  reversed source-row order.
- V43 real executions: zero resource stops on U01–U04.
- V45 real result hashes:
  - U01: `3055a3f85eee4099663dbb0ba10373bad69d6c714683eef07f419fcde2c61086`
  - U02: `c3f37042a1b045fa279f7879e95b6a12d1293aa859a4cec2ab2779af5d7b9ba1`
  - U03: `22a008b21bf7f51c71235f65b812782c7f63ea3986f7dfdde5fe7bd63bae1652`
  - U04: `ea46d35b4fbc78a6fb337ee7a2db19bc025138eae5d39aa62b6393b45c0bdd47`

Artifact SHA-256 values:

- V43 engine: `c3521fd7552fdcc24956b137635717e61e07ceca204cdf469fedb9f5b5a72f6b`
- V44 engine: `fffd8d6e1c6597648e6dafb1e028bc83c5dabfbdea10bfea4735fa3a3249a1dc`
- V45 engine: `a9327807a10c5cba0da1fbfd91c4eec4a613ccc3d92e89150607df652628dcd5`
- U04 V45 JSON: `1e17b8244b4f91b46607a38ff2c8b01d3773d6a3fe1d49a8ed7f4085fb102721`
- U04 V45 visual: `c96b6511540fb2a0bdc3495bc80db13e3ced71ee6f70c80decbe9e4d0ab0e5b5`

## Scientific status and next gate

V43 and V44 are funded research components: each removes a demonstrated false
positive mechanism, and V44 already has positive 35-case LOFO evidence through
its retained V27 primitive. V45 fixes the old V28 execution-contract defects
and succeeds visually on U04, but it still needs one genuinely unseen
paired-return part before promotion. No tuning or rerun on U04 is authorized or
needed.
