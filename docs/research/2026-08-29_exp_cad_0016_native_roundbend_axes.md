# EXP CAD 0016 — native SolidWorks RoundBend axes

## Outcome

Retain the source-native event-birth mechanism for further unchanged testing. On authenticated CAD `50476000_A` and independent paired scans N011/N012, it produced exactly four finite manufacturing-root axes and passed both native-resolution visual reviews.

Generalization progress indicator: `GPI-C_POSITIVE_SOURCE_NATIVE_MECHANISM_NOT_F1_PROGRESS`.

This is material architecture progress, not an F1 or generalization claim.

## What changed

The retired unrestricted mesh proposal produced 11 candidates per scan and failed 0/2 visual gates on this family. The new selector uses only source-native evidence:

1. Four active unsuppressed SolidWorks `RoundBend` features.
2. Native custom property `Bends=4`.
3. Native custom property `Sheet Metal Thickness=2 mm`.
4. Exactly four folded-body inner/outer cylindrical surface pairs that are coaxial and whose radii differ by 2 mm.
5. Finite axis support taken from the source face-owned tessellations.

The resulting four axes are the two forming roots on each side. The prior diagonal, stacked, crossing, short-fragment, free-edge, hole, and inner/outer duplicate hypotheses disappear before scan processing.

## Visual ruling

Both 3000×1800 three-view overlays were inspected at original resolution. N011 and N012 each pass. The stable side structures visibly support the four CAD axes; the large central-panel departure remains visible and is not used to invent or suppress events.

## Integrity boundary

The decoder does not expose a complete name-level mapping from each `RoundBendN` feature to a specific B-rep face pair. The experiment therefore binds the feature count and source thickness to the complete qualifying geometry set, and explicitly does not fabricate a feature-name-to-pair mapping.

No gold, labels, expected evaluator count, benchmark feedback, filename identity, or score influenced the selector. F1, cross-family generalization, production readiness, and defect classification remain closed.

## Next gate

Audit the remaining authenticated SLDPRTs for the same native integrity tuple. Execute the selector unchanged wherever active feature count, native bend count, sheet thickness, and qualifying finite cylinder-pair count agree. Retire or route parts that lack that native evidence rather than falling back silently to unrestricted mesh event birth.
