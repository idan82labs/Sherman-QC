# CIBEC_V1 neutral authority packet — visual audit

Date: 2026-08-22

Scope: source-free synthetic fixture geometry only. No physical scan, model prediction, gold label, expected event count, or proposed event marker was opened or added.

## Iteration ruling

- V1: **REJECTED FOR AUTHORITY REVIEW**. S03 did not visibly preserve the parent flanges; S08/S09 read as generic tubes in the PCA views; S11's parent sheet was too faint.
- V2: **INSUFFICIENT ALONE**. The affected geometries improved, but the six PCA views still did not distinguish open hem, closed hem, and rolled edge reliably enough for a manufacturing-authority decision.
- V3: **ACCEPTED AS A NEUTRAL AUTHORITY-REVIEW PACKET**, not as a semantic qualification PASS. It retains the exact pinned six-view renderer and adds answer-free mid-span Y/Z profiles. The profiles make the ambiguous cross-sections legible while the six-view renders preserve span and interruption context.

## Native-resolution case review

| Case | Visual ruling | Native-resolution observation |
|---|---|---|
| S01 | PASS_FOR_REVIEW | Single positive-axis 90-degree bend and both parent legs are clear. |
| S02 | PASS_FOR_REVIEW | Mirror/negative-axis control is visually distinct from S01. |
| S03 | PASS_FOR_REVIEW | Parent flange plus paired inner/outer cylindrical arcs are both visible. |
| S04 | PASS_FOR_REVIEW | Multiple tangent radii and the compound transition are legible. |
| S05 | PASS_FOR_REVIEW | Two close, parallel formed transitions are distinct. |
| S06 | PASS_FOR_REVIEW_WITH_SIX_VIEW | The mid-span profile resembles S01; the six-view artifact is required to see non-overlapping axial spans. |
| S07 | PASS_FOR_REVIEW | Return lip and parent bend are visibly separable. |
| S08 | PASS_FOR_REVIEW | Open hem has a large, unmistakable separation between returning and parent legs. |
| S09 | PASS_FOR_REVIEW | Closed hem has a small approximately 0.75 mm separation and is visually distinct from S08. |
| S10 | PASS_FOR_REVIEW | Raised bead is centered and continuous with the parent sheet. |
| S11 | PASS_FOR_REVIEW | Rolled edge is a full large-radius loop attached to a long parent sheet, distinct from both hems. |
| S12 | PASS_FOR_REVIEW | Compound rounded transition contains two visibly different curvature regimes. |
| S13 | PASS_FOR_REVIEW_WITH_SIX_VIEW | Cross-section matches a single bend; the six-view artifact is required to judge shorter contiguous axial support. |
| S14 | PASS_FOR_REVIEW_WITH_METADATA | Geometry intentionally matches S01; the distinction is the untrusted-overlay abstention condition in the fixture metadata, not a geometric difference. |

## Presentation limitations

- Profile titles overlap the neutral disclaimer line. Geometry, axes, and case identity remain readable, so this is a presentation defect rather than a semantic ambiguity.
- The pinned six-view renderer displays `0 predicted bends` because it consumes the deliberately empty prediction file. That text must not be interpreted as an expected answer; the worksheet and manifest explicitly state that no event count or marker is proposed.
- S06, S13, and S14 cannot be adjudicated from the mid-span profile alone; reviewers must use the six-view artifact and fixture description.

## Scientific status

This visual audit establishes only that V3 is clear enough to collect independent manufacturing-authority decisions. It does not define the bend-event contract, sign fixture expectations, qualify the CIBEC implementation, or provide any new generalization/F1 evidence.
