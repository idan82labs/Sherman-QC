# EXP_CAD_0024 — source-native orientation landmarks

## Executive result

EXP24 breaks the proper-pose 180-degree ambiguity left by EXP23 on two independent physical scans without using labels, expected counts, evaluator data, filenames, scanner-up, or a fixture prior.

The frozen accepted CAD mesh contained only 43 of 67 active-body tessellations and stopped at CAD `z=+7 mm`. The source SLDPRT retained 24 unresolved native display-list tables that cover the omitted crown, asymmetric tab, holes, slots, and upper profile through `z=+53 mm`. EXP24 uses those tables only as an orientation landmark lane; they are forbidden from discovering, classifying, or measuring bends.

Result:

`GPI-C_POSITIVE_CROSS_SCAN_SOURCE_NATIVE_ORIENTATION_DISAMBIGUATION_NOT_F1_PROGRESS`

This is source-only mechanism progress. It is not an F1 or production claim.

## Frozen gate

For each case, EXP24:

1. verifies the SLDPRT, pinned cadmpeg, and frozen mesh-report hashes;
2. requires the decoded unresolved-table inventory to equal the frozen 24-table inventory exactly;
3. evaluates only proper SO(3) candidates that already pass the EXP23 native-event support gate;
4. clusters numerically duplicate transforms;
5. measures deterministic, table-balanced scan support at 2 mm;
6. authorizes orientation only when the best class has minimum table coverage at least 0.95, mean coverage at least 0.98, and a mean-coverage margin of at least 0.20 over the runner-up.

## Measurements

| Case | Selected class | Selected min / mean | Runner-up | Runner-up mean | Margin | Visual |
|---|---|---:|---|---:|---:|---|
| N011 | SO3_PCA_02 | 1.000 / 1.000 | SO3_PCA_21 | 0.19759 | 0.80241 | PASS |
| N012 | SO3_PCA_02 | 1.000 / 1.000 | LEGACY_PROPER | 0.20036 | 0.79964 | PASS |

Both acquisitions independently select the same proper orientation class.

## Native-resolution visual ruling

Both 3000×2400 comparison images were inspected at original resolution. In each selected panel the native landmarks cover the cyan scan continuously over the upper crown, asymmetric tab, hole field, and lower outline. The runner-up panel visibly places the upper asymmetric morphology on the wrong side and leaves most fixed tables unsupported.

This visual ruling is hash-bound in `EXP24_MANUAL_NATIVE_VISUAL_REVIEW.json`.

## Replay and isolation

- Primary and replay full semantic payloads are byte-exact for 2/2 cases.
- Primary and replay rendered PNG hashes are exact for 2/2 cases.
- Focused EXP24 tests: 2 passed.
- Gold, labels, expected counts, and evaluator data remained unopened.
- Scanner-frame inference was separately rejected: the Geomagic PLY files contain no sensor pose, gravity, fiducial, or persistent fixture geometry, and the two acquisitions reverse their raw PCA directions.

## Next gate

Before opening evaluator truth, corroborate the orientation with an independent analytic-B-rep lane built from active-body topology vertices and through-thickness cylindrical cutout carriers. This avoids relying on the same unresolved display-list tessellation representation twice. The corroboration must reproduce the same orientation on both scans and pass native-resolution review; otherwise the system abstains.
