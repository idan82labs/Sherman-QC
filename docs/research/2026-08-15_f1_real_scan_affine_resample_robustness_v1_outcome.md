# Real-scan affine/resample robustness V1 — outcome

Date: 2026-08-15  
Outcome: robustness failure retained as a deterministic development regression  
Promotion impact: none; not human gold and not independent F1

## What changed after the synthetic visual critique

The idealized synthetic parts were rejected before detector execution because their manufactured shape grammar did not resemble the real corpus. They were replaced with eight deterministic affine/resampling variants of four real scans. This preserved real holes, tabs, formed boundaries, surface wear, damage, and acquisition artifacts while varying pose, sampling density, and small normal-direction noise.

## Execution result

- V51 primary/replay: byte-identical for 8/8 cases.
- V52 primary/replay: byte-identical for 8/8 cases.
- V52 added 12 candidates across the eight primary cases.
- Every prediction-only native-resolution sheet was inspected before evaluator pseudo-truth was opened.
- Visual decisions: one pass (R008), one conditional pass (R002), six failures.

## Development pseudo-truth accounting

Against transformed, previously visually audited V51 references:

- true positives: 18
- false positives: 22
- false negatives: 14
- exact cases: 0/8
- pseudo-truth precision: 0.4500
- pseudo-truth recall: 0.5625
- pseudo-truth F1: 0.5000

These numbers quantify self-consistency only. They are not human-gold F1.

## Visual/numeric agreement

- R001: 10 predictions for 5 references; 7 extras. The overlay shows long-fold duplicates and junction clutter.
- R004: 5 predictions, 0 matches, 5 extras, 4 misses. The overlay independently shows two axes crossing a flat face.
- R005/R006: the same real parent changes from 3 to 5 predictions, with most axes stacked on one visible fold.
- R008: 2 visually clean opposing-fold axes and 0 extras, but 2 transformed references remain missed.

The failure is therefore not merely threshold sensitivity in the scorer. It is visibly unstable candidate ownership and consolidation under routine mesh resampling.

## Why V64 was not forced onto this lane

The frozen V64 PSROC mechanism certifies two separated interior responses on one smooth ribbon. Most panel duplicates here are closer than its frozen 10h minimum separation, and the R004 flat-face crossings are not a ribbon pair. Moreover, V64's prospective contract requires reference data to remain unopened until its own primary/replay and pre-truth audit are frozen; the development references were already opened after the baseline audit. Applying V64 now would be both scientifically mismatched and procedurally invalid.

## Next funded action

Use this lane only as a regression target. The next intervention must have a new architecture identity and solve final-line cross-route ownership/consolidation with positive source-geometry support. It must be frozen and tested on scientific fixtures, then executed on a fresh reference-unopened lane with native-resolution visual auditing. No case-specific rescue or threshold fitting to R001–R008 is allowed.

True generalization evidence still requires exporting a scan of the newly received physical part, manually selecting its observable bend axes before inference, and locking those annotations independently.

## Bound artifacts

- Visual critique/contact sheet: `output/research/f1-real-scan-affine-resample-v51-v52-v1-20260815/visual_audit_contact_sheet.png`
  - SHA-256: `5c40a66ba41bd55448da708c36cb0382c72cea4ac965c20900da380dab57bd90`
- Execution manifest:
  - SHA-256: `b7045655bc58c5e7315828e739320239771e32d34710a0d9958efd15c791cde6`
- Pre-truth visual audit:
  - SHA-256: `8229e2641cc9acb1bcb38e303e4cac7ccc2f963a5b7d1fdb77286065c105d991`
- Self-consistency score:
  - SHA-256: `53e92cf89fecc5b0c88d738fa967ba744a410d1b23980abdb5b4f4635ba83c37`
