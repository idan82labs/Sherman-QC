# F1 calibrated raw-range synthetic V3 — frozen truth-free ruling

Date: 2026-08-15

## Scope

This ruling was frozen after V51/V52/V60/V61 inference and visual inspection, but before opening the private synthetic bend truth. The V3 corpus is engineering-only, non-confirmatory synthetic evidence. It contains 30 captures from 15 new complex cross-section families, including micro-bends, shallow and close creases, nested returns, compound joggles, tight radii, and broad rounded facets.

## Execution integrity

- V51: 30 primary plus 30 replay executions; 30/30 semantic replay identity; zero failed stages; zero forbidden cache records; truth unopened.
- V52: 30 cases; 55 retained baseline predictions; exact replay identity.
- V60: 19 raw births across 14 cases; 2 V52 duplicate matches suppressed; exact replay identity.
- V61: all 19 raw births admitted; zero ownership suppressions; exact replay identity.

Changed cases: `O001 O003 O004 O007 O011 O012 O013 O019 O020 O021 O022 O023 O027 O029`.

## Manual native-resolution visual audit

Every changed case was inspected in:

1. a three-view fused-source overlay showing retained V52 predictions in green and raw V60 births in cyan; and
2. an eight-frame calibrated 384×288 depth sheet for every proposed birth.

All 19 proposed lines remained attached to persistent geometric transitions across the native views. O004, O013, and O023 were ambiguous in the compressed fused view; their native depth sheets resolved the candidate as a visible fold distinct from an obvious floating line or simple V52 duplicate. No candidate was visually rejected before scoring.

Frozen truth-free ruling: **ADMIT_ALL_19_FOR_SCORING**.

This ruling is not a claim that all 19 are true positives. It records only that none failed the source-only geometric and visual sanity check. The private truth may now be opened once for scoring without changing inference or the audit ruling.

## Frozen artifact hashes

- V51 execution manifest: `a6f9e521b1aee1df481253f55e3b5945007265d2cd96c2b5a9bd7dd60f302052`
- V52 execution manifest: `191b1f50140994ff79fef4e9455bc767827f49e4e3f0d93b02a0d52a02694371`
- V60 execution manifest: `6a9a3ba7f34b8ce6aa56d65abd860634cb4e7143fb1764f741be402c623e55c1`
- V61 execution manifest: `8ce69896941258626e0b5534d938d0c84664b40ad997276280a6af9b4dd0c5ef`
- Three-view audit manifest: `d4051bbf6116c1fce58c32b9998e2976cb8fccd9ae15c12b84618a882adcfb3c`
- Aggregate hash over all three-view and native audit files: `1629304acfdc8c4d81f1df1d32ad4457b40530a91ac660adbe1c2562ab9a6781`

## Evidence locations

- Three-view audit: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815/v61-truth-free-audit/three-view`
- Native-frame audit: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815/v61-truth-free-audit/native`
