# Local scanner-intake capability audit

Date: 2026-08-15  
Scope: read-only local workspace audit; no detector invocation; no truth opened

## Finding

The current Sherman QC ingestion path preserves scan geometry, not authentic
pre-fusion range observations. It cannot currently constitute
`CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1`.

Authoritative code evidence:

- `apps/api/services/bend_inspection_service.py:144-162` accepts scan uploads
  only as STL, PLY, OBJ, or PCD.
- `backend/file_watcher.py:38-42` watches only PLY, STL, OBJ, and PCD.
- `backend/bend_inspection_pipeline.py:49-51` declares the same scan extension
  set.
- `backend/file_watcher.py:55-64` extracts a part number from the VXelements
  export filename, confirming that the watched product is a named geometry
  export rather than an opaque calibrated view bundle.

Local file inventory across `data/bend_corpus/incoming`, `live_scans`,
`uploads`, and `source-assets` found 201 PLY files and no E57, BAG, MCAP, ONI,
EXR, TIFF, H5/HDF5, LAS/LAZ, or other native acquisition container. A wider
search under `/Users/idantal/ShermanQC` likewise found no such capture file.
The 50 ZIP packages in the intake tree were also inspected by member name;
none contains a depth/range-frame, validity, intrinsic/extrinsic, camera-pose,
or recognized native sensor-container member. Those archives are older CAD
intake packages, not hidden raw scanner acquisitions.

The newest blind intake (`2026-08-13_shaked_new_parts_blind`) contains four
opaque PLY scans (`U01`-`U04`) plus inference/evaluator manifests and a drawing.
Those new parts are valid for mesh-only research, but they do not reopen the
calibrated raw-range capability.

## Consequence

No software conversion can reconstruct the discarded view organization,
per-pixel visibility, calibrated intrinsics, or measured camera poses from the
fused PLY. Rendering the PLY from synthetic cameras would create a different
experiment and is explicitly forbidden by the preserved post-V56 ruling.

The next physical capture must follow
`docs/research/2026-08-15_calibrated_raw_range_capture_contract_v1.md`. The
scanner/export station—not the bend detector—is the next system boundary that
must change.

## Accounting

- Existing qualifying physical capture bundles: 0.
- Existing physical mesh scans modified or reclassified: 0.
- V57 detector implemented or invoked: no.
- Truth, CAD, labels, expected counts, and V56 outputs opened: no.
- Runtime/main or production changed: no.
