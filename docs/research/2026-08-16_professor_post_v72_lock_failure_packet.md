# Professor decision request — V72 CASL source-free lock contradiction

We executed your V72 `CANDIDATE_CENTERED_ANNULAR_SECANT_LAW_V1` source-free lock exactly once. It did not lock: **37/43 passed, 6 failed**. We made no post-lock edits and did not execute U01-U04. Please issue exactly one amended lock specification or a formal V72 retirement.

## Immutable evidence

- Your preserved ruling SHA-256: `31ea25b5c572f0611620aedc9a95aa212a77bf8b545a418a3b63429513386b8b`
- JUnit: `output/research/f1-v72-casl-source-free-lock-20260816.junit.xml`
- JUnit SHA-256: `e609c9f6221448e7017d667a49470be8cd7ca41a43b1b9adc2185fc7fbe4cde8`
- Implementation SHA-256: `b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a`
- Fixture SHA-256: `8d7ddece77b4f56b17883e8b8f0ae436ce5a38bc301572bdb7ec9e33e8233147`
- Runner SHA-256: `254a465ceba4bf493dce963e2e35941d7eab970338d1d192ce33a84bfdc4848f`
- One invocation only: `37 passed, 6 failed in 5.52s`
- U01-U04: unopened by V72
- Native-resolution visual gate: unopened
- Reserved physical scan: sealed
- Gold, labels, CAD, expected counts, identities, filenames, benchmark outcomes: unopened

## Failure class A — your analytic generator exceeds your frozen source cap

Your common generator requires half-unit axial samples `z_m=m/2, m=0..340` (341 planes), half-unit profile parameters through radius 75, and `N<=200,000`. Exact materialization gives:

| Required fixture | Exact records | Result |
| --- | ---: | --- |
| G10 two G03 bends centered 24 apart | 205,282 | `SOURCE_POINT_CAP` / scan abort |
| G15 four-ray crossing | 204,941 | `SOURCE_POINT_CAP` / scan abort |
| G16 two unconnected full parallel lines | 205,282 | `SOURCE_POINT_CAP` / scan abort |
| exact twofold profile refinement | 204,941 | `SOURCE_POINT_CAP` / scan abort |

This is a paper-level inconsistency: the required fixture cannot both preserve the stated common sampling and satisfy the required source cap.

## Failure class B — boundary values are not constructible by the naïve fixture arithmetic

### Six-degree direction drift

The unit vector built as `(cos(6*pi/180), sin(6*pi/180), 0)` evaluated through the frozen `acos(clip(dot))` path to `0.10471975511966022`; the frozen `6*pi/180` boundary is `0.10471975511965977`. The mathematically exact boundary therefore appeared several binary64 ulps above the inclusive comparator.

### Effective count

The fixture used `W=Q=nextafter(12,-inf)=11.999999999999998`. Binary64 evaluation of `W^2/Q` rounded to exactly `12.0`, so it did not create the required one-ulp-below computed `n_eff`.

## Passed evidence

Thirty-seven cases passed, including all within-cap ordinary wedge/rounded/return positives; flat, edge, and R=20/R=80 roll negatives; wedge 8°/174°, flat 176°, ratio 1.20, and eta 0.08 boundaries; permutation, endpoint, reflection, signed-axis, translation, 0.8/1.25 similarity invariance; and all nonconflicting cap/atomicity cases.

## Decision requested — choose exactly one

### `AUTHORIZE_V72_FIXTURE_ONLY_CORRECTION`

Provide the exact replacement formulas for:

1. G10, G15, G16, and the twofold refinement so every required fixture remains analytically faithful yet has `N<=200,000`. Specify whether to trim the radial endpoint, reduce axial support, deterministically thin, or explicitly exempt only source-free fixture generation. Give exact boundaries and tie rules.
2. The 6° boundary fixture. Specify whether the production comparator must operate in dot/cosine space or give exact binary64 vector components whose evaluated `acos` equals the inclusive boundary.
3. The one-ulp-below `n_eff` fixture. Give exact binary64 `W,Q` inputs or authorize direct comparator injection on a precomputed binary64 `n_eff` scalar.
4. State whether a single corrected 43-case lock run is authorized after these fixture-only amendments. No CASL thresholds, formulas, station rules, annuli, candidate comparator, real-scan cap, or U01-U04 inputs may change.

### `RETIRE_V72_AND_SELECT_ONE_SUCCESSOR`

Formally retire V72 at the source-free lock and specify exactly one bounded source-only successor/capability opening. Preserve the same strict blind constraints and require native-resolution visual adjudication before any reserved-scan opening.

Do not authorize U01-U04 unless the corrected source-free lock is exactly 43/43 and byte-identical on its internal replay. Do not provide multiple alternatives for local selection.
