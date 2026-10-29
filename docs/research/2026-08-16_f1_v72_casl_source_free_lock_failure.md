# F1 V72 CASL source-free lock failure

Date: 2026-08-16  
Architecture: `V72_CANDIDATE_CENTERED_ANNULAR_SECANT_LAW_V1` (`CASL_V1`)  
Disposition: **NOT LOCKED; NO U01-U04 EXECUTION AUTHORIZED**

## Frozen evidence

- GPT-5.6 Pro ruling: `docs/research/2026-08-16_gpt56_pro_post_v71_successor_ruling_verbatim.md`
  - bytes: `30021`
  - SHA-256: `31ea25b5c572f0611620aedc9a95aa212a77bf8b545a418a3b63429513386b8b`
- Source-free JUnit: `output/research/f1-v72-casl-source-free-lock-20260816.junit.xml`
  - bytes: `25701`
  - SHA-256: `e609c9f6221448e7017d667a49470be8cd7ca41a43b1b9adc2185fc7fbe4cde8`
- Lock result: `37 passed, 6 failed` in `5.52s`; one frozen invocation only.
- Implementation SHA-256: `b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a`
- Fixture SHA-256: `8d7ddece77b4f56b17883e8b8f0ae436ce5a38bc301572bdb7ec9e33e8233147`
- Runner SHA-256: `254a465ceba4bf493dce963e2e35941d7eab970338d1d192ce33a84bfdc4848f`

No threshold, formula, comparator, source-free fixture, or implementation file was changed after the failed lock. U01-U04 were not opened or executed. The reserved physical scan and native-resolution visual gate remain sealed.

## Six failures

### Frozen source-cap conflicts

The Pro ruling simultaneously requires half-unit axial sampling from `z=0` through `z=170`, half-unit profile sampling through radius `75`, and `N <= 200,000`. Four mandated constructions exceed the cap before CASL can make a candidate decision:

| Fixture | Materialized source records | Frozen cap | Observed outcome |
| --- | ---: | ---: | --- |
| G10 two rounded bends 24 units apart | 205,282 | 200,000 | `SCAN_ATOMIC_ABORT` |
| G15 four-ray crossing | 204,941 | 200,000 | `SCAN_ATOMIC_ABORT` |
| G16 two full parallel lines, gap 6 | 205,282 | 200,000 | `SCAN_ATOMIC_ABORT` |
| exact twofold profile refinement | 204,941 | 200,000 | `SCAN_ATOMIC_ABORT` |

The record counts were recomputed from the frozen analytic generators without rerunning CASL.

### Binary64 comparator-fixture ambiguities

1. The fixture vector constructed analytically at exactly `6°` produced:
   - evaluated `acos(dot)`: `0.10471975511966022`
   - frozen `6 * pi / 180` boundary: `0.10471975511965977`
   - outcome: the mathematically exact boundary failed the inclusive binary64 comparator.
2. The naïve one-ulp-below effective-count construction used `W=Q=nextafter(12,-inf)`:
   - input scalar: `11.999999999999998`
   - computed `W^2/Q`: `12.0` after binary64 rounding
   - outcome: the requested one-ulp-below effective-count check did not produce a below-boundary computed value.

## What did pass

All ordinary positive and anti-certificate primitives that remained within the source cap passed, including:

- sharp wedges, rounded bends, return/hem, opposing/reflected and finite-span bends;
- flat, free-edge, and both smooth-roll negatives;
- wedge lower/upper boundaries, flat boundary, ratio boundary, and eta boundary;
- source/candidate permutation, endpoint reversal, reflection, signed-axis permutation, translation, and both similarity scales;
- source/candidate/spacing/length caps, result-byte cap, and atomic pre-rename failure.

These passes do not authorize real-scan execution because the frozen 43/43 lock gate failed.

## Required ruling

V72 requires one fresh GPT-5.6 Pro decision:

1. provide an exact fixture-only correction that resolves all four `N > 200,000` contradictions and both binary64 boundary constructions without changing the CASL scientific architecture, constants, or real-scan caps; **or**
2. retire V72 formally and select one next bounded source-only architecture/capability opening.

No local repair, rerun, threshold edit, or U01-U04 execution is authorized pending that ruling.
