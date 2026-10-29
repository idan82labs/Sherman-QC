# Candidate-local ultrasonic B-scan acquisition contract V1

Status: acquisition-ready; no ultrasonic files were present in the workspace on 2026-08-14.

This lane is intentionally separate from the scan-observable F1 target. Its purpose is to test whether subsurface evidence can distinguish a monolithic fold from a fabricated/ground joint when optical geometry cannot.

## Locked acquisition

- Maximum 16 candidate axes per study batch.
- Candidate axes are selected by a source-only detector before any ultrasonic result is opened.
- For each axis, acquire three perpendicular B-scan strips centered at normalized axis stations 0.25, 0.50, and 0.75.
- Store raw amplitude/time-of-flight arrays and calibration metadata; do not begin with screenshots or a reconstructed volume.
- Co-register each strip to the optical scan with the opaque candidate ID and raw scan-frame endpoints.
- Include matched reference coupons: at least one monolithic bend and one fabricated/ground joint with comparable thickness, material, angle, and surface finish.
- Operator, gain, probe, wedge, velocity, sampling rate, gate, and calibration block are fixed within a batch.

## Pre-registered decision

The first batch is feasibility-only. A candidate-local ultrasonic feature is fundable only if a deterministic rule separates the matched reference classes and repeats across all three stations without using filename, part identity, or optical role. Failure at any station is reported; no post-hoc gain, gate, or threshold rescue is permitted in the same batch.

## Required delivery layout

```text
ultrasonic_batch_v1/
  manifest.json
  calibration/
  candidates/<opaque_candidate_id>/
    station_025.npz
    station_050.npz
    station_075.npz
```

The manifest must include SHA-256 hashes, candidate endpoints, station positions, acquisition settings, coupon class (evaluator-only), and an explicit statement that the detector never received coupon class or expected count.

## Current blocker

The repository contains optical PLY/STL assets but no genuine ultrasonic, PAUT, A-scan, B-scan, or C-scan measurement files. Software can prepare candidate locations and validate a delivered batch, but it cannot manufacture the missing physical measurement.
