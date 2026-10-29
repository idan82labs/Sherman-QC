# F1 calibrated raw-range synthetic V1 lock

Date: 2026-08-15  
Token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY`  
Status: generator frozen after truth-free pilot; full corpus not yet scored  
Confirmatory eligible: no

## Purpose

Create the missing calibrated multiview information channel prospectively so
the acquisition-view mechanism can be falsified without pretending to recover
discarded real scanner observations. This is a new synthetic experiment, not a
V57 rerun and not evidence of physical-scan generalization.

## Frozen corpus

- 30 opaque cases, `O001`-`O030`.
- 15 synthetic sheet-metal profile families, two captures each.
- Eight original organized depth views per case at 384x288.
- Depth in millimetres, Boolean validity, finite pinhole `K`, and rigid
  `T_world_from_camera` for every view.
- Public source PLY fused only from those saved views.
- Maximum 195,000 fused points.
- Hidden scene meshes, family map, generation metadata, and bend truth remain
  in `evaluator_only` and are unavailable to preflight and truth-free visual
  audit.

## Frozen acquisition constants

- Horizontal field of view: 50 degrees.
- Camera radius: 1.20 times hidden-scene bounding-box diagonal.
- Eight golden-angle directions with fixed elevation pattern.
- Depth noise sigma: 0.075 mm.
- Random dropout: 0.0025 plus a fixed grazing-incidence loss.
- Initial fusion voxel: 0.62 mm, increased by 8% only if needed to satisfy the
  195,000-point cap.
- No random sampling, family routing, expected count, truth, or detector output
  affects camera placement, noise, dropout, or fusion.

## Pilot history

Attempt 1 used a 0.38 mm fusion voxel. All six bundles passed calibration and
visual alignment, but median nearest-neighbour spacing was 0.274 mm versus
0.463 mm across ten physical LiDAR references. It was not scored.

Attempt 2 changed only the fusion voxel to 0.62 mm. All raw depth and validity
hashes remained byte-identical. Median spacing became 0.422 mm, all six cases
remained inside the observed physical range 0.154-1.039 mm, and all calibration
preflights passed. This density-only adjustment was completed before any bend
detector or truth was opened.

Manual review of every attempt-2 depth mosaic and 3-D back-projection overlay
found coherent part topology, stable holes/dropout, no pose-split surfaces, and
no ghost layers. Secondary-depth islands in O003/O004 were consistent with
opposing surfaces visible through holes and landed on the fused source.

## Frozen implementation hashes

- Generator `scripts/generate_f1_calibrated_raw_range_synthetic_corpus_v1.py`:
  `def0b7f66e3207e6111ab483dd77eae780a8893cee416e1a10eba66bbd0926e4`
- Bundle validator `scripts/validate_f1_calibrated_raw_range_bundle_v1.py`:
  `7ba3614326c8d2d92c162d58eca00c32bee5601aa9785da16096d493359b5f4c`
- Truth-free renderer
  `scripts/render_f1_calibrated_raw_range_synthetic_audit_v1.py`:
  `ad009f002ae6e0fa792ae53f420311d91cc44006e6f1e447318e1d69781af17f`

## Gates before detector work

1. Generate all 30 cases once from the frozen hashes.
2. Require 30/30 bundle preflight PASS.
3. Require every source cloud at or below 195,000 points.
4. Require every median nearest-neighbour spacing inside the physical reference
   range and report aggregate median.
5. Render and manually inspect every depth mosaic and 3-D calibration overlay.
6. Seal public/evaluator manifests and visual rulings.

Only after all six gates pass may a bounded synthetic-only acquisition-view
mechanism test be implemented. No physical benchmark, runtime/main, production,
or generalization claim is opened by this lock.
