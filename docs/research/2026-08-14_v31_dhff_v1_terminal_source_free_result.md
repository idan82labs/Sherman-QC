# V31 DHFF V1 terminal source-free result

## Status

`V31_DISTRIBUTED_HARMONIC_FOLD_FLUX_V1` is terminally retired after its first immutable source-free invocation.

- Token: `V31_DHFF_SOURCE_FREE_MECHANISM_RETIRE`
- Stage: `base:SF01_FOLD60_MESH_C2`
- Runtime to terminal result: `1.67981645799955` seconds
- Expected: one pre-NMS rescue, one final rescue, station mask `2047`
- Observed: zero pre-NMS rescues, zero final rescues, station mask `953`
- Passing stations: `7/11`
- Longest consecutive run: `3`
- Terminal reason: `AXIAL_STATION_COVERAGE`

## Station result

| Station | Result |
|---:|---|
| 00 | `STATION_PASS` |
| 01 | `ANGULAR_COVERAGE` |
| 02 | `ANGULAR_COVERAGE` |
| 03 | `STATION_PASS` |
| 04 | `STATION_PASS` |
| 05 | `STATION_PASS` |
| 06 | `POTENTIAL_COVERAGE` |
| 07 | `STATION_PASS` |
| 08 | `STATION_PASS` |
| 09 | `STATION_PASS` |
| 10 | `ANGULAR_COVERAGE` |

## Manual native-resolution visual ruling

All three required 4096px post-invocation artifacts were inspected at original resolution. All eleven station slices visibly contain the same clean, continuous 60-degree fold with broad tails and the opposite skin. The four failed stations have no corresponding visible geometric defect or missing material. The final overlay confirms that a physically valid continuous fold was withheld.

The station centers advance by `2.4s` while the analytic sample grid is spaced by `0.5s`. The alternating pass/fail pattern on visually equivalent station sections is therefore evidence consistent with station-phase/discretization sensitivity inside the DHFF certificate. This is a mechanism failure, not a visually credible rejection of the fixture geometry.

## Governance

- V31 may not be rerun, threshold-tuned, rescued, or given a replacement input.
- Metamorphic, replay, stress, sealed-scan, benchmark, gold, runtime/main, and production lanes were not reached.
- The active F1 generalization objective remains unchanged.
- A fresh GPT-5.6 Pro successor ruling was requested at `https://chatgpt.com/c/6a7eec45-2fd4-83eb-8499-ccc6c5c8132e`.
