# F1 synthetic LiDAR stress V2 — V57 capability-not-opened STOP

Date: 2026-08-15  
Evidence class: synthetic engineering only; non-confirmatory  
Corpus token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`

## Outcome

`V57_CALIBRATED_MULTIVIEW_TURN_CHANGEPOINT_CONSENSUS_V1` (`CMTCC_V1`) was not implemented or invoked. The professor-authorized capability `CALIBRATED_RAW_RANGE_VIEW_BUNDLE_V1` is unavailable for all 30 locked S001–S030 scans.

The terminal status is:

`V57_CMTCC_CAPABILITY_NOT_OPENED_STOP`

The preserved terminal artifact is:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v57-calibrated-multiview-turn-changepoint-consensus-v1/capability_stop.json`

SHA-256: `f259e1b1a813e700db8dcb1307fc49ee2df851d64a18cdb48d3e69d10606c5ca`

## Professor ruling

The complete post-V56 response is preserved byte-for-byte at:

`docs/research/PROFESSOR_RESPONSE_POST_V56_NEXT_ACTION_20260815.md`

SHA-256: `8659ed318a479cd64e5375317a0bc691a27d03000167ef940220c5a802e1c2c8`

The ruling opened exactly one capability: authentic organized, calibrated pre-fusion depth frames. It explicitly forbade new viewpoints, altered renderer settings, inferred poses, rendering the fused mesh to manufacture “raw” views, mesh-only fallback, partial-corpus execution, and V57b rescue. It required the capability STOP before any detector invocation if even one S001–S030 bundle was unavailable.

## Custody evidence

The locked corpus contains its manifest, direct PLY meshes, density metadata, and research results. It contains no S001–S030 depth arrays, validity masks, organized range frames, camera intrinsics, camera poses, acquisition manifest, or original renderer manifest.

The corpus generator is:

`scripts/generate_f1_lidar_like_synthetic_corpus.py`

SHA-256: `55e7ea0ff266bf5cb488b89ca15b6446408eba5f2bafa6b78fbdef61b7a9c328`

Its source constructs a `TriangleMesh` directly from an extruded filleted 2-D profile, applies noise/dropout to mesh vertices and faces, and writes the final PLY with `write_triangle_mesh`. It has no camera model, organized depth lattice, validity image, pinhole intrinsics, sensor pose, ray casting, per-view renderer, or range-image fusion stage. Therefore no original acquisition views exist to reproduce byte-for-byte.

The locked corpus manifest SHA-256 is `3df2614d1c86c64245aa909e081744c877677f9d8dce902dc98689f41e99c88d`.

A workspace-wide filename/type inventory found some unrelated rendered multiview audit frames and deviation arrays, but zero acquisition bundles associated with S001–S030. Those unrelated files cannot satisfy V57 custody and were not opened as inference inputs.

## Execution accounting

- Authentic capability bundles available: 0/30.
- Missing capability bundles: 30/30.
- V57 detector implemented: no.
- Source-free V57 detector fixtures invoked: 0.
- Sealed child invocations: 0/60.
- Predictions produced: 0.
- Score consumed: no.
- Visual gate consumed: no.
- V56 outputs opened: no.
- Mesh rerendering substitute used: no.
- Mesh-only fallback used: no.
- Truth opened for V57 inference: no.

The capability recorder and its tests are:

- `scripts/record_f1_v57_capability_not_opened_stop.py` — SHA-256 `dcf77da5ce3e66875106d56aa59fb608b51a4a55f9ca3fc9f0a037d39b46043d`.
- `tests/test_record_f1_v57_capability_not_opened_stop.py` — SHA-256 `b924b1b6d556a3b1a9da3de7d3e910fd22ff69370de5e922a740db0fd9658074`.
- Verification: `2 passed in 0.02s`.

## Process disclosure

During the post-ruling custody audit, Codex opened `generation_metadata.json`. That file contains family and bend-count fields and therefore should not have been opened inside the V57 pre-score boundary. No V57 detector or fixture existed at the time, the read was not used to select constants or produce predictions, and the missing-capability STOP was independently established by the corpus file inventory and direct-mesh generator source. Nevertheless, the read is recorded as an additional process breach and independently forbids proceeding with V57.

## Scientific disposition

V52 remains the retained frozen research baseline. V53–V57 are terminal. V56 remains closed for any prospective successor design.

Creating camera views now from the frozen PLYs would not recover discarded acquisition evidence; it would introduce a new synthetic observation selected after the benchmark and would violate the causal test. The result would be a rendered-mesh experiment, not CMTCC.

The only legitimate way to reopen this capability family is a fresh, prospectively constituted corpus whose acquisition pipeline natively preserves, for every scan:

1. three or more original organized depth frames;
2. validity masks;
3. finite pinhole intrinsics;
4. rigid world-from-camera transforms;
5. immutable acquisition and simulator/renderer manifests;
6. content hashes and a custody record frozen before detector development or output inspection.

That would be a new experiment on a new corpus and requires fresh user custody authority plus a new professor protocol. It cannot be called a V57 rerun or substitute S001–S030 after the fact.
