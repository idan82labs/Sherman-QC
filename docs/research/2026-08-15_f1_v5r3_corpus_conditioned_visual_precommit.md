# F1 V5R3 corpus-conditioned visual precommit

Status: **SUPERSEDED — POST-EXECUTION CORPUS-SIMILARITY AUDIT FAILED**

> Correction recorded 2026-08-15 after direct side-by-side native visual
> comparison with the real incoming U01–U04 scans.  The earlier pretruth
> acceptance judged mechanical plausibility and capture consistency, but did
> not enforce corpus-level morphology or mesh-spacing similarity.  V5R3B is
> consumed, terminal, and excluded from any claim about corpus performance.
> See `2026-08-15_f1_v5r3b_posthoc_corpus_similarity_rejection.md`.

The user's visual objection to V5/V5R2 was correct. Those pilots used a
repetitive rounded-carrier vocabulary, overly clean long folds, simple
topology, and weak corpus resemblance. They are quarantined as diagnostic-only
and are not reused for validation, tuning, or scientific performance claims.

## Rejected pre-gate attempt

`output/research/f1-calibrated-raw-range-synthetic-v5r3-ltco-pilot-20260815`
was rejected before inference. O005 rendered its local hat as separate
triangular sheets and O009 rendered detached return rails beside the box. The
batch-wide gate therefore failed even though its other four cases were closer
to physical corpus morphology. No detector or truth score was produced.

## Accepted prospective batch

Root: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815`

The two failed constructions were replaced with continuous finite extruded
sheet profiles, including visible thickness and end caps. All six cases were
then inspected in every one of twelve native depth views plus fused oblique,
top, and side views. Private truth was not opened.

| Case | Corpus-conditioned morphology | Visual ruling |
|---|---|---|
| O001 | four-sided service tray, relief corners, mounting ears, large window, unequal edge folds and short creases | ACCEPT |
| O003 | windowed U guard, close parallel walls, mounting ears, broad rolled negative | ACCEPT |
| O005 | continuous finite local stamped hat with rounded/faceted shoulders, end caps, adjacent apertures and partial edge fold | ACCEPT |
| O007 | vented guard, dense slots, short tabs, true bottom wall and broad smooth guard roll | ACCEPT |
| O009 | deep box with continuous integral hem returns, corner relief gaps, center aperture and side walls | ACCEPT |
| O011 | asymmetric stepped instrument bracket, mixed apertures, short creases, partial folds and shallow rounded negative | ACCEPT |

The batch is materially closer to the real corpus references than V5/V5R2:
silhouettes are open or stepped rather than six rounded rectangles; apertures
range from large access windows to mounting holes and vents; folds have finite
lengths and unequal extents; close parallel walls, small creases, corner
reliefs, thickness, view occlusion, and broad rounded facets are present.

## Scan and topology contract

- Public input is an organized-range triangle fusion, not the hidden design
  mesh and not a point-only PLY.
- Six public meshes contain 10,695 to 23,348 vertices and 17,815 to 40,989
  triangles, below the 200,000-point bound.
- Triangle edges are admitted only within a per-frame local spacing bound, so
  occlusion gaps are not bridged.
- Twelve noisy/dropout calibrated range views are retained per case.
- Hidden design meshes and manual truth remain under `evaluator_only` and are
  unavailable to inference.
- No V5/V5R2 geometry, output, label, or outcome is used by V63.

## Frozen hashes before inference

- Generator: `scripts/generate_f1_calibrated_raw_range_synthetic_corpus_v5r3.py`
  - SHA-256 `59e79628e96d3d2c01ac822a228d575ef8f71f6c71e16b402c2351e3091d8d74`
- Public manifest SHA-256: `959db07a198c0e45a2d8cdfaa5f076fdb5fc573d9fb501f0c77e96c23b7dbaf6`
- Renderer manifest SHA-256: `7a38efd721b4a3b394f7a71e22a32881d7b7baaa66f7608bfab2377e393d9d82`
- Review sheets:
  - sheet 01: `5b18430101e2be8eae3edbbce4c08a3da005b387dea68fc64f748753ff83116e`
  - sheet 02: `9e17edb41d2c3bb8e10afd7ab0d2cbd5de2ea01feaad6fa5a4f986d109f0dad4`
  - sheet 03: `0b59c88a79558554269445e0ada67262a69c9eac1121ca9583af0ae92cf5ff18`

Public source SHA-256 values are:

- O001 `7cd9d7dd31086024be38e9858092251d8b815bbed18b2fd525b52767db9a3f0e`
- O003 `3dd4a2d3d15fd90858d0bd94097a55c35f97e93435110852eca571588fac96c7`
- O005 `f3bc1f8aab99a577802e2fbc16c1d2b9666b2804c91dc3ccc12e9197c3000090`
- O007 `15ce2d0275f0fe15021439238f8eece31fb92b669a8c9a937c490df88c7bfd77`
- O009 `99c92ba085f157ed4ac430bb8a2602ea54e887505aa6c5e06424eadacd118901`
- O011 `4e4e7a18738a2bb9c7ff29748b406790a6c6ef617536cd5e3d4a50fc9f7df0a3`

## Execution rule

Run the frozen blind baseline and replay first. If baseline truth activation
later proves `TP_B < 4` or `FP_B < 4`, retire V63 as underactivated. Otherwise
apply frozen V63 once, freeze primary/replay output, open truth exactly once,
score, and audit every changed case at native resolution. No rerun, replacement
input, threshold rescue, or post-hoc variant is authorized after inference
begins.
