# Calibrated raw-range capture contract V1

Status: prospective acquisition protocol; no V57 detector exists or is invoked  
Capability: `CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1`  
Intended successor experiment: new untouched corpus; **not** a V57 rerun

## Why this contract exists

The present Sherman QC intake accepts `PLY`, `STL`, `OBJ`, and `PCD` scan
geometry. The local physical corpus contains fused geometry exports, but no
original organized depth frames, per-pixel validity, camera intrinsics, or
world-from-camera poses. Once those observations are discarded, they cannot be
recovered by rendering the fused mesh from invented cameras.

The post-V56 GPT-5.6 Pro ruling therefore stopped V57 on S001-S030 and opened
one prospective capability only: preserve the authentic calibrated pre-fusion
range views from a **new** capture before detector development.

## Scanner-station capture requirement

For every physical scan, the scanner operator must preserve two products before
cleanup or fusion:

1. the scanner's immutable native acquisition/export package; and
2. a normalized, content-addressed research bundle containing 3-64 original,
   undistorted organized depth views.

Each normalized view contains:

- a two-dimensional depth array in millimetres (`.npy`);
- a same-shaped Boolean validity array (`.npy`);
- a finite 3x3 pinhole intrinsic matrix `K` with positive focal lengths;
- a finite rigid 4x4 `T_world_from_camera` transform;
- SHA-256 hashes of both arrays.

The bundle also contains the fused source geometry used by the bounded detector,
with no more than 200,000 finite points. Valid depths are finite and positive;
invalid depths are exactly zero. No frame may exceed 8192x8192 and the scan may
contain at most 100,000,000 raw pixels.

If the scanner software cannot export the original organized depth plus its
calibration, stop at the station. A PLY mesh by itself is not a qualifying
capture.

### VXelements-specific station check

The local watcher describes the present filenames as VXelements exports.
Creaform's official VXelements 10 release notes say that VXscan's Merge Scan
can retain original data, so the first low-cost test is to enable that option,
save the complete native session, and keep it before PLY/STL export:

<https://www.creaform3d.com/en/about-us/press-releases/creaform-releases-tenth-version-their-vxelements-3d-measurement-software-platform-cloud>

That public statement does **not** prove that VXelements exposes undistorted
organized frames, per-pixel validity, `K`, and `T_world_from_camera`. On a
disposable calibration object, the operator must verify the saved session or
supported SDK/API can export all four items. If it preserves only unmerged
point sets or meshes, it still does not satisfy this contract. Ask Creaform
support for the exact supported raw-acquisition/API export for the installed
scanner model and VXelements version; do not reverse-engineer calibration or
infer poses.

## Custody split

The private acquisition ledger may retain scanner model, firmware, calibration
certificate, operator, timestamps, part number, native filenames, and the
native package hash. It stays outside inference custody.

The normalized research bundle uses only an opaque token such as `O001` and the
exact schema validated by:

`scripts/validate_f1_calibrated_raw_range_bundle_v1.py`

Its `manifest.json` is:

```json
{
  "schema_version": 1,
  "kind": "sherman_qc_calibrated_raw_range_capture_bundle_v1",
  "opaque_case_id": "O001",
  "source_geometry": {
    "file": "source.ply",
    "sha256": "<64 lowercase hex characters>"
  },
  "frames": [
    {
      "ordinal": 0,
      "depth_file": "frames/f000_depth.npy",
      "depth_sha256": "<sha256>",
      "validity_file": "frames/f000_validity.npy",
      "validity_sha256": "<sha256>",
      "K": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
      "T_world_from_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  ]
}
```

`file` fields exist only in this custody manifest so the validator can bind
bytes. They are removed when creating the later content-addressed child input.
No part/family identity, filename hint, CAD, drawing, truth, expected count,
RGB, intensity, texture, benchmark result, or score enters the inference bundle.

## Mechanical preflight

Run from the project root:

```bash
.venv/bin/python scripts/validate_f1_calibrated_raw_range_bundle_v1.py \
  --bundle /absolute/path/to/O001 \
  --output /absolute/path/to/O001/preflight.json
```

The preflight fails closed on malformed or nonrigid calibration, nonfinite valid
depth, nonzero invalid depth, duplicate semantic frames, hash mismatch, path
escape, cap violation, schema extras, or source/calibration disagreement.

Coordinate agreement is measured by back-projecting up to 4,096 valid pixels
per view and querying the fused source geometry. It requires:

- median nearest-source residual <= 0.003 of the source bounding-box diagonal;
- 95th percentile residual <= 0.015 of the diagonal;
- at least 90% of samples <= 0.020 of the diagonal.

The report contains only sizes, counts, hashes, finite-value results,
calibration residuals, and cap margins. It emits no bend hypotheses or visual
artifacts and opens no truth.

## New-corpus order of operations

1. Confirm the scanner can export the required native observations using a
   disposable calibration object, not a benchmark part.
2. Freeze the exporter/converter and this validator by SHA-256.
3. Precommit eligibility and exclusion rules, then capture at least 30 scans
   spanning at least 15 known or near-neighbour families.
4. Seal private identity and acquisition ledgers; expose only opaque bundles.
5. Validate every bundle. One failure prevents corpus execution; do not replace
   it by visual preference.
6. Independently annotate and adjudicate truth without detector outputs.
7. Freeze the baseline source-only predictions and the new experiment protocol.
8. Only after a fresh professor review may a calibrated multiview detector be
   implemented and invoked.

## Non-negotiable exclusions

- Do not retrofit S001-S030.
- Do not render an existing PLY to fabricate raw views.
- Do not infer missing camera poses.
- Do not mix qualifying and mesh-only cases.
- Do not change capture settings after seeing predictions.
- Do not use the new physical part merely because its result looks favourable;
  membership follows a precommitted acquisition-time rule.

This contract removes the current information-loss blocker. It does not claim
that the scanner presently exposes the required export, and it does not itself
authorize V57 or any successor execution.
