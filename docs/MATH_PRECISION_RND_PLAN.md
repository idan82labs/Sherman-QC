# Math-Driven Precision R&D Plan (Accuracy-First, Regression-Gated)

This plan defines mathematically grounded upgrades for CAD/scan inspection where
accuracy is prioritized over throughput. It is tied to current modules so each
change can be tested and promoted only if non-regressive.

## 1) Core mathematical model

### 1.1 Plane and bend geometry

For each flange neighborhood:

- Fit plane by SVD on centered points: smallest singular vector is normal `n`.
- Plane equation: `n^T x + d = 0`, where `d = -n^T c` and `c` is centroid.
- Bend dihedral: `theta = 180 - arccos(|n1^T n2|)` in degrees.
- Bend line direction: `u = normalize(n1 x n2)`.

These are already implemented in:

- `/Users/idant/82Labs/Sherman QC/Full Project/backend/feature_detection/bend_detector.py`

### 1.2 Radius and arc consistency

For local bend neighborhoods projected to a 2D normal section:

- Fit circle `(center, r)` by Taubin algebraic fit.
- Arc length `L = r * theta_rad`.
- Per-bend mm deviations:
  - `Delta_r = r_scan - r_cad`
  - `Delta_L = L_scan - L_cad`
  - line start/end/center offsets in mm.

These are surfaced in match/report data for operator actions.

## 2) Registration accuracy stack (for complex/ambiguous scans)

Accuracy-priority order:

1. Global candidate generation: FPFH + RANSAC correspondence filtering.
2. Local refinement: point-to-plane ICP and/or GICP.
3. Hard-case fallback: certifiable/robust methods (TEASER++ class).
4. Optional non-rigid diagnostic (CPD) for detecting warped material behavior.

Rationale:

- ICP is local and initialization sensitive.
- GICP improves for sheet-metal planar neighborhoods.
- Certifiable/robust fallback reduces catastrophic local minima on low overlap.

## 3) Vector-space consistency for bend identity

Use subspace consistency in addition to scalar angle thresholds:

- Compare flange normal pair `(n1, n2)` signatures between CAD and scan.
- Use principal-angle style checks between local tangent subspaces.
- Reject candidates where orientation-consistent bend line direction cannot be
  established under allowed rigid transform hypotheses.

This addresses false positives on parts with repeated similar bend angles.

## 4) Rounded part pathway (distinct from folded parts)

Rounded/rolled parts should run a dedicated primitive branch:

1. Segment candidate curved neighborhoods.
2. Fit cylinder/cone primitives (RANSAC + robust LS refinement).
3. Report per-feature expected vs actual in mm:
   - radius
   - arc length
   - chord length
   - local normal deviation.

Do not score rounded features with folded-bend heuristics only.

## 5) Statistical promotion rule ("best survives")

Each configuration is treated as a statistical estimator over scan outcomes.

Per-scan utility:

`U = 0.60*completion + 0.25*in_spec - 0.08*overdetect - 0.05*fail_ratio - 0.02*warn_ratio`

Where each term is clipped to valid bounds.

Per-candidate evaluation:

- Compute mean utility and bootstrap CI.
- Use conservative bound `LCB95` (2.5th percentile) in score.
- Gate challengers against incumbent on non-regression constraints:
  - utility lower bound
  - over-detect penalty
  - full completion penalty
  - partial monotonicity penalty
  - partial over-complete penalty
  - completion recall.

Current implementation:

- `/Users/idant/82Labs/Sherman QC/Full Project/scripts/bend_improvement_loop.py`

Important reproducibility rule now applied:

- Score bootstrap seeds use stable SHA-256-derived offsets (not Python's
  process-randomized `hash`) so promotion decisions are deterministic for the
  same data and base seed.

## 6) Regression protocol (mandatory)

Any math change is accepted only if:

1. Unit tests pass for scoring/gating math invariants.
2. Bend pipeline tests pass for status/progression semantics.
3. Part-level replay tests pass on known difficult parts.
4. Challenger passes non-regression gate vs incumbent.

Current regression suites:

- `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_bend_improvement_loop_math.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/tests/test_bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/tests/test_bend_inspection_pipeline.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_feature_measurement.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_part_47266000.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_part_11979003.py`

## 6.1) New finding from staged-part regression (March 10, 2026)

Hard staged tray parts exposed a specific failure mode in local-refinement merging:

- a high-quality global alignment can coexist with locally unobserved bend neighborhoods
- treating local `UNOBSERVED` as authoritative absence erases plausible global detections
- this is mathematically wrong because alignment quality is global, while bend observability is local

Operational rule added to the regression methodology:

- only locally observed absence (`OBSERVED_NOT_FORMED` or `PARTIALLY_OBSERVED`) may override a weak global bend match
- pure `UNOBSERVED` local results must be interpreted as missing evidence, not negative evidence

Research implication:

- partial-process bend inference should distinguish "absence of evidence" from "evidence of absence" at the bend-neighborhood level
- this distinction is likely publishable if formalized as an observability-aware matching criterion and benchmarked against naive merge rules

Important regression note:

- a follow-up hypothesis was tested: demote very-low-confidence one-sided local probe-region bends from `OBSERVED_FORMED` to partial observation
- that rule was more conservative, but it regressed the hard staged benchmark slice
- conclusion: local evidence tightening needs a better identity model, not only a stricter observability threshold
- promotion rule remains: do not keep a mathematically neat change if it worsens real staged progression accuracy

Second regression note:

- a geometry-weighted local assignment score was tested for dense local refinement, using bend-line direction and flange-normal consistency in addition to center/angle proximity
- a second follow-up replaced greedy local assignment with a global optimal-assignment solve
- both ideas were mathematically well motivated, but neither survived the regression gate:
  - the global assignment variant regressed `49024000_A.ply`
  - the score-weighting variant did not close the tray-family gap and introduced run-to-run sensitivity on at least one full scan family
- conclusion: repeated-bend tray parts likely need a richer identity feature set or an explicitly conditioned local state model, not just a different assignment objective

## 7) Next high-value math upgrades

1. Add trimmed/overlap-aware ICP objective for partial scans.
2. Add explicit uncertainty propagation per bend (`sigma_angle`, `sigma_radius`).
3. Calibrate confidence with reliability diagrams (probability calibration).
4. Add per-part Bayesian model ranking (posterior over config quality).
5. Add synthetic perturbation testbench (noise, occlusion, partiality sweeps).

## References (primary / official docs)

- Besl & McKay, ICP (PAMI 1992): <https://doi.org/10.1109/34.121791>
- Chen & Medioni, point-to-plane registration (1992): <https://www.sciencedirect.com/science/article/pii/026288569290066C>
- Segal et al., Generalized-ICP (RSS 2009): <https://www.roboticsproceedings.org/rss05/p21.html>
- Fischler & Bolles, RANSAC (CACM 1981): <https://doi.org/10.1145/358669.358692>
- Rusu et al., FPFH (ICRA 2009): <https://doi.org/10.1109/ROBOT.2009.5152473>
- Yang et al., TEASER++: <https://arxiv.org/abs/2001.07715>
- Yang et al., Go-ICP: <https://arxiv.org/abs/1605.03344>
- Myronenko & Song, CPD: <https://arxiv.org/abs/0905.2635>
- Open3D robust kernels: <https://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html>
- Open3D global registration: <https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html>
- NIST metrology/standards context: <https://www.nist.gov/publications/metrology-standards-additive-manufacturing>
