# EXP CAD 0017 — patch-aware native RoundBend axes

## Outcome

Retain for broader unchanged testing. The patch-aware source-native selector passes 3/3 native-resolution visual gates across two authenticated CAD families: four bends on 50476000_A (N011/N012) and 20 bends on 31367000_A00 (N014).

Generalization progress indicator: `GPI-C_POSITIVE_CROSS_FAMILY_SOURCE_NATIVE_MECHANISM_NOT_F1_PROGRESS`.

This moves the architecture needle: it is positive cross-family mechanism evidence, but it is not yet measured F1 or physical defect generalization.

## Successor change

EXP16 assumed one inner and one outer face per cylindrical bend carrier. EXP17 preserves disconnected face patches before pairing. Within each coaxial axis cluster, it:

1. Groups source B-rep patches by cylindrical radius.
2. Selects inner/outer radius levels separated by the native sheet thickness.
3. Pairs patches only when their finite axial intervals match.
4. Keeps disconnected collinear intervals as separate inspection events.
5. Requires the final event count to equal both active `RoundBend` features and the native `Bends` property.

This correctly resolves 31367000_A00 as 15 disconnected bridge/root segments plus five other bends, rather than collapsing the 15 collinear segments into one event.

## Visual result

Candidate-bound orthographic close-ups and three-view overlays were inspected at original 3000×1800 resolution.

- N011: pass, four full-height roots.
- N012: pass, independent reproduction of the same four roots.
- N014: pass, 15 disconnected bridge roots, three full-length longitudinal roots, and two transverse end roots.

No diagonal, crossing, duplicate inner/outer, hole, slot, short edge-fragment, or unsupported free-edge candidates were visible.

## Scientific boundary

The three scans are authenticated and physical, and the mechanism transfers across two substantially different CAD morphologies. However, no trusted bend annotations were used or opened. Therefore:

- event ontology and localization mechanism: positive source-only evidence;
- cross-family robustness: preliminary positive visual evidence;
- F1 improvement: unmeasured;
- defect detection/generalization: unmeasured;
- production promotion: closed.

The next quantitative gate requires trusted event annotations on a sealed physical scan or a production-approved source-native definition that can be scored without circularly deriving truth from the same CAD features.
