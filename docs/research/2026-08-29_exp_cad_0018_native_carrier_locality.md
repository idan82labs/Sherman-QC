# EXP CAD 0018 — native carrier locality

## Outcome

Retain as a prerequisite locality gate, not as bend conformance.

Generalization progress indicator: `GPI-C_POSITIVE_CROSS_FAMILY_LOCALITY_SIGNAL_INSUFFICIENT_FOR_F1`.

Across seven authenticated physical scans, the local source-native method is independent of incomplete global CAD tessellation. N011/N012/N014/N015/N016 support every intended event along every finite axis bin. N003 supports two of three and has one entirely absent intended neighborhood. N004 supports all three, with minimum event coverage 0.625.

## Visual critique

Every native-resolution green/magenta overlay agrees with the reported bin coverage. The result is useful, but green points include planar strips that fall inside the source-radius annulus. Therefore the method measures continuous material locality near the intended carrier; it does not yet prove that those points form a cylinder or the correct bend angle.

## Diagnostic correction

N015/N016 previously appeared to fail because their converted CAD mesh contains face-owned tessellation for only 70 of 206 B-rep faces. Exact distance was independently reproduced with Open3D raycasting, so the math is correct; the global reference coverage is not. EXP18 avoids that invalid global denominator and recovers five visually coherent local roots on both scans.

## Next step

The next bounded action is local carrier search: allow a small source-thickness-scaled offset around each intended axis, maximize unchanged carrier support, and report displacement from nominal. Curvature or surface-normal evidence must then confirm bend geometry before any good/bad classification or F1 claim.
