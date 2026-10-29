# F1 organized-frame capability opening V1

## Why this is the next representation

The retained scan-only parent V51 reaches F1 0.593 on the consumed physical
development set.  Its full candidate pool has a much higher matching ceiling
(recall 0.847), but EXP_SCAN_0002 demonstrated that learned selection over the
same fused-Ply line objects promotes exterior and parallel aliases.  Previous
real-scan resampling also showed that the fused representation is unstable.

An unordered fused PLY has already discarded which camera ray produced each
sample, which samples were invalid or occluded, and which frame observed each
surface.  Those are precisely the observations needed to distinguish a true
two-surface bend from a silhouette, boundary, return skin, or duplicate line.

## Accepted physical bundle

Each case contains:

- one fused PLY for compatibility and custody binding;
- a strict pinhole calibration JSON;
- at least two authentic organized depth/range arrays (`.npy`);
- a matching validity mask per frame;
- strictly increasing capture timestamps;
- a calibrated right-handed `world_from_sensor` transform per frame.

The validator rejects labels, expected counts, CAD, quality class, part
identity, filenames, predictions and other answer-bearing information.  It
hashes every observation and emits a model-safe receipt.

## Capture requirement

Export the scanner's pre-fusion depth/range frames and calibration from the
same acquisition that produces each PLY.  Do not synthesize frames from the
PLY: that cannot restore occlusion or ray history and is not valid evidence.
Capture at least two independent physical instances per morphology family for
the final qualification cohort, under the separate custody protocol.

## Gate

No acquisition-aware detector experiment starts until at least two consumed
development cases validate byte-for-byte through
`scripts/validate_f1_organized_frame_intake_v1.py`.  Untouched qualification
cases remain sealed and are never used to develop the method.

