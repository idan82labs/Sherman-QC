# Zero-Shot Contour / Fold-Edge Integration Memo

## Purpose

Translate two relevant point-cloud papers into a concrete next tranche for the current zero-shot bend engine:

- Hackel et al., *Contour Detection in Unstructured 3D Point Clouds* (CVPR 2016)
- Lin et al., *Edge Detection and Feature Line Tracing in 3D-Point Clouds by Analyzing Geometric Properties of Neighborhoods* (Remote Sensing 2016, AGPN)

This memo is not a general literature review. It is a repo-specific decision document for what to borrow, where to attach it, and what to reject unless it passes the frozen zero-shot gates.

## Current Repo State

The current zero-shot path is:

1. `BendDetector.detect_bends(...)` generates local bend candidates from single-scale and multiscale plane-pair geometry.
2. `backend/zero_shot_bend_profile.py` builds perturbation bundles on the original scan and voxelized variants.
3. Candidate bends are grouped into line families.
4. Line-family-grouped candidates are converted into `BendEvidenceAtom`s.
5. Atoms are greedily merged into `BendSupportStrip`s.
6. Policy families score the scan and the selector emits:
   - exact count
   - count range
   - or abstain
7. `renderable_bend_objects` are produced from the final strip set.

This path is now good enough for:

- exact outputs on several known-count full scans
- safe abstention on partial scans
- renderable bend objects in the viewer

But it still has a persistent upstream weakness:

- repeated near-90 scans such as `ysherman`
- compact neighboring bends
- cases where voxel perturbations drop or fragment the repeated bend family before strip inference can stabilize it

That means the next gain is unlikely to come from another selector heuristic. The main limiter is still upstream evidence quality.

## What Hackel Contributes

Hackel's key pattern is:

1. compute a pointwise contour likelihood from local geometric features
2. construct an overcomplete graph of contour candidates
3. select a coherent subset with connectivity priors and loose-end penalties

The important lesson for this repo is not the exact learning machinery. It is the decomposition:

- local contour evidence first
- candidate graph second
- global subset selection third

That is directly relevant because the current bend engine still jumps too quickly from detector candidates to line families and support strips.

### Repo mapping

Hackel-like ideas fit best into:

- `backend/feature_detection/bend_detector.py`
- `backend/zero_shot_bend_profile.py`

Specifically:

1. **Point / patch bend-likelihood**
   - Add a bend-likelihood score at the raw detector-support level before policy scoring.
   - Current closest object: `DetectedBend`.
   - Current gap: `DetectedBend.confidence` is mostly plane-pair / radius-fit confidence, not explicit contour-or-fold evidence.

2. **Overcomplete candidate graph**
   - Keep more locally plausible bend traces before dedup, but represent their compatibility explicitly.
   - Current closest objects:
     - line families from `_cluster_line_families(...)`
     - evidence atoms from `_build_evidence_atoms(...)`
   - Current gap: grouping is mostly greedy and pairwise, with no explicit graph-level subset decision.

3. **Subset selection with connectivity prior**
   - Penalize isolated short candidates and reward continuous, repeated support aligned with the same physical bend family.
   - Current closest seam:
     - `_infer_support_strips(...)`
     - `_same_support_strip(...)`
   - Current gap: strip inference is still greedy first-fit grouping.

## What AGPN Contributes

AGPN's key pattern is:

1. detect edge points directly from geometric neighborhood properties
2. distinguish boundary edges from fold edges
3. trace feature lines from the detected edges
4. make thresholds spacing-aware and density-aware

The most relevant AGPN-specific ideas for this repo are:

1. **Fold-edge specific evidence**
   - A bend-support strip should be driven by fold evidence, not only by plane adjacency.

2. **Feature-line tracing**
   - Traced line segments can become a better centerline prior than midpoint-only grouping.

3. **Spacing-aware thresholds**
   - Parameter scales should be tied more directly to local point spacing and density variation.

4. **Robust normals near fold edges**
   - AGPN's RANSAC-normal idea matters because ordinary PCA normals are unstable at intersecting surfaces.

### Repo mapping

AGPN-like ideas fit best into:

1. `backend/feature_detection/bend_detector.py`
   - upstream fold-edge evidence
   - spacing-aware thresholds
   - robust neighborhood handling around narrow adjacent bends

2. `backend/zero_shot_bend_profile.py`
   - trace-informed evidence atoms
   - centerline continuity scoring
   - strip grouping driven by traced support overlap instead of midpoint proximity alone

## What We Should Borrow

### 1. Add contour / fold evidence before strip grouping

Add an intermediate evidence layer ahead of `BendEvidenceAtom` construction:

- `FoldEdgeEvidence`
  - `evidence_id`
  - `support_point_indices` or sampled support points
  - `centerline_points`
  - `mean_axis_direction`
  - `fold_strength`
  - `normal_discontinuity_score`
  - `spacing_mm`
  - `trace_length_mm`
  - `branch_degree`
  - `loose_end_penalty`

This should be produced from raw `DetectedBend`s plus local geometric support, not from final policies.

### 2. Trace line support, not just midpoints

The current strip grouping mostly relies on:

- direction similarity
- orthogonal midpoint distance
- angle compatibility

That is not enough for:

- several short neighboring bends
- repeated near-90 families
- cases where one perturbation fragments a long bend into several short local supports

Add a trace/line-support overlap term to grouping. Two local candidates should count as the same physical strip only if they have:

- compatible axis direction
- compatible flange geometry
- overlapping or near-continuous traced support along the axis

### 3. Make density-aware thresholds explicit

The current engine already estimates local spacing with `_estimate_local_spacing(...)`, but that spacing is underused.

Promote spacing-aware logic into:

- multiscale detector thresholds
- duplicate merge thresholds
- strip overlap / gap thresholds
- candidate rescue conditions for compact bends

The principle from AGPN is:

- when density variation is piecewise, a globally fixed threshold is not good enough
- subtle edges should be detected first, then filtered by traced continuity

### 4. Preserve an overcomplete candidate graph longer

Do not force early collapse into final line families.

Instead:

- keep an overcomplete candidate set from single-scale + multiscale + perturbation variants
- attach compatibility edges
- run a subset selection step before final support-strip formation

This does not need to be a full MRF initially. A deterministic graph pruning step is enough for the next tranche if it is grounded in:

- trace continuity
- duplicate overlap
- loose-end penalties
- compactness penalty

## What We Should Not Borrow Yet

### 1. Hackel's full learned classifier

We do not have the right labeled contour dataset yet.

Borrow the feature logic and staged structure first. Do not introduce a learned contour classifier until:

- we have a stable annotation story for bend-support localization
- we know which points or patches should count as positive fold evidence

### 2. AGPN as a direct replacement for the detector

AGPN is useful as an upstream auxiliary signal, not as a wholesale replacement for the current sheet-metal bend detector.

Our current detector already contributes sheet-metal-specific structure:

- adjacent flange reasoning
- bend angle / radius estimates
- confidence tied to plane-pair geometry

Replacing it wholesale would likely throw away useful domain structure.

### 3. New selector heuristics as the main path

The current selector has already been pushed far.

The next tranche should improve:

- candidate preservation
- fold evidence
- centerline tracing
- strip ownership

before adding more exact/range rescue logic.

## Concrete Code Tranche

### Tranche name

`B8` — contour / fold-edge upstream evidence

### Write scope

- `backend/feature_detection/bend_detector.py`
- `backend/zero_shot_bend_profile.py`
- targeted tests only

### Step 1: diagnostics, not behavior change

Persist raw fold/trace diagnostics per candidate bundle:

- candidate trace count
- trace length summary
- trace overlap summary
- fold-strength summary
- connected-component count

Acceptance:

- new diagnostics appear in repeated-near-90 and small-tight-bend artifacts
- no protected slice changes

### Step 2: trace-informed evidence atoms

Extend `BendEvidenceAtom` construction to include:

- traced support length
- traced support endpoints
- trace continuity score
- fold-strength score

Acceptance:

- atom payloads carry trace/fold evidence in zero-shot summaries
- tests cover serialization and summary emission

### Step 3: trace-aware strip grouping

Modify `_same_support_strip(...)` and strip grouping so that the same-strip decision requires one of:

- meaningful along-axis trace overlap
- or a very small along-axis trace gap with strong fold support

Acceptance:

- improves `ysherman` or `49204020_04`
- does not widen their validated bands
- does not regress dev / holdout protected metrics

### Step 4: overcomplete candidate pruning

Insert a deterministic graph-pruning step between evidence atoms and final support strips:

- penalize loose ends
- penalize isolated short traces
- prefer connected repeated support
- keep ambiguous alternatives only when they preserve range containment

Acceptance:

- known-count audit improves exactness or range tightness on hard full scans
- repeated-near-90 benchmark stays green
- small-tight-bend slice does not regress

## Benchmarks That Must Gate This Work

This tranche should be rejected unless it stays green on:

- `data/bend_corpus/evaluation_slices/zero_shot_development.json`
- `data/bend_corpus/evaluation_slices/zero_shot_holdout.json`
- `data/bend_corpus/evaluation_slices/zero_shot_known_count_plys.json`
- `data/bend_corpus/evaluation_slices/zero_shot_small_tight_bends.json`
- `data/bend_corpus/evaluation_slices/zero_shot_repeated_near_90.json`

Primary watch items:

- `ysherman` should not broaden beyond the validated `10-12`
- `49204020_04` should not broaden beyond the validated `11-12`
- `47959001` should not lose its exact `6`
- `10839000` should not lose its exact `10`

## Why This Is The Right Next Step

The current engine has already extracted most of the safe value available from selector-side determinization.

The remaining hard failures are telling us:

- local candidates disappear or fragment upstream
- voxel perturbations destabilize repeated families
- midpoint-only grouping is too weak for tight neighboring bends

Hackel and AGPN both point in the same direction:

- strengthen local edge/fold evidence
- preserve richer candidate structure longer
- use continuity and tracing before final object collapse

That matches the current repo bottleneck.

## When Professor Input Becomes Necessary

This literature-guided tranche should still be done without professor intervention.

Professor input becomes necessary when we want to move from:

- heuristic strip ownership
- traced support continuity
- deterministic graph pruning

to:

- explicit latent support assignment
- principled bend vs flange ownership
- decomposition-level confidence over competing bend sets

That is the point where this should evolve from a stronger engineering system into the more formal `p(B, z, F | Y)` program.
