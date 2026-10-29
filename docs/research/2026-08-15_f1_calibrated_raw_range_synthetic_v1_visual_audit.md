# F1 calibrated raw-range synthetic V1 visual audit

Date: 2026-08-15  
Corpus: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815`  
Truth opened: no  
Detector invoked: no

## Aggregate ruling

`PASS_TRUTH_FREE_ACQUISITION_VISUAL_GATE`

All 30 opaque cases were inspected in ten high-resolution review sheets. Every
case included all eight native 384x288 depth frames and a three-view 3-D plot
of the colored per-frame back-projections over the gray fused source cloud.

Observed conditions:

- fold/profile topology is coherent across camera viewpoints;
- structured holes and random dropout remain spatially consistent;
- no case shows camera-pose splitting, duplicated surfaces, or a detached
  ghost layer;
- colored view points land on the fused source surface in oblique, top, and
  side projections;
- isolated secondary-depth islands occur where another material surface is
  visible through a hole or opening and remain supported by the 3-D overlay;
- all cases retain at least five views with valid fraction >= 0.05, so every
  case exceeds the three-view mechanism requirement;
- O009 is consistently narrow in projection; O028 includes three near-edge
  views, but both retain coherent geometry and at least five substantial views.

## Per-case rulings

| Cases | Ruling | Notes |
| --- | --- | --- |
| O001-O002 | coherent | L-profile pair; stable two-tail fold and holes; no split layers. |
| O003-O004 | coherent | U-profile pair; secondary surfaces through openings are supported in 3-D. |
| O005-O006 | coherent | Offset profile; shallow side views remain registered. |
| O007-O008 | coherent | Hat/step geometry; O008 has lower-area views but six >=0.10 valid fraction. |
| O009-O010 | coherent | Multi-fold profiles; O009 is narrow but has eight >=0.05 views. |
| O011-O012 | coherent | Return geometry; consistent internal/external surfaces. |
| O013-O014 | coherent | Deep channels; full wall/floor registration. |
| O015-O016 | coherent | Double steps; occlusion holes remain bounded and consistent. |
| O017-O018 | coherent | Double-Z pair; repeated folds remain separate in 3-D. |
| O019-O020 | coherent | Corrugated pair; multiple ridges remain aligned across views. |
| O021-O022 | coherent | Narrow-hat pair; sidewalls and top plate agree. |
| O023-O024 | coherent | Omega returns; inner returns remain visible and registered. |
| O025-O026 | coherent | V-profile pair; acute fold and cutouts remain stable. |
| O027-O028 | coherent | Obtuse pair; O028 has five >=0.05 views and no pose ghosting. |
| O029-O030 | coherent | Asymmetric multistep pair; repeated steps remain aligned. |

## Numerical support checked after visual review

- Bundle preflight: 30/30 PASS.
- Source point range: 31,431-108,952, below the 195,000 cap.
- Maximum 95th-percentile source/back-projection residual: 0.001863 of source
  scale, below the 0.015 gate.
- Median nearest-neighbour spacing: 0.442958 mm versus 0.463105 mm across ten
  physical LiDAR references.
- Synthetic case range: 0.413473-0.519626 mm; physical observed range:
  0.153753-1.039131 mm.
- Every case has at least five views with valid fraction >=0.05.

## Preserved evidence

- Native depth frames and fused sources: `public/O001`-`public/O030`.
- Per-case depth mosaics and 3-D overlays: `visual_audit/O001`-`visual_audit/O030`.
- Review sheets: `review_sheets/review_sheet_01.png` through
  `review_sheet_10.png`.
- Calibration reports: `preflight/O001.json` through `preflight/O030.json`.
- Density report: `density_audit.json`.

This pass funds a synthetic-only mechanism test. It does not establish realism
of Creaform sensor errors, physical generalization, production validity, or the
goal benchmark thresholds.
