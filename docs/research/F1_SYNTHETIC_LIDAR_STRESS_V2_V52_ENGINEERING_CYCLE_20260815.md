# F1 synthetic LiDAR stress V2 and V52 engineering cycle

Date: 2026-08-15

## Outcome

A 30-scan, 15-family synthetic corpus was generated at the measured spatial density of the available LiDAR meshes and used as a sealed, non-confirmatory engineering stress lane. The frozen V51 engine scored F1 0.382. Candidate-ceiling analysis showed that 91 of 100 truth bends were already present in the source-only candidate pool, isolating selection and ownership as the dominant defect. A bounded V52 cylindrical-consensus recovery raised F1 to 0.581 with 22 additional true positives and no additional false positives.

V52 is retained as an engineering prototype. This result does not establish generalization because every case is synthetic.

## Corpus construction and calibration

- Corpus: `output/research/f1-synthetic-lidar-stress-v2-20260814`
- Cases: 30 scans, two captures for each of 15 part families.
- Explicit evidence token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`.
- `confirmatory_eligible` is false in the manifest and every downstream score.
- Real reference median nearest-neighbor range: 0.153753-1.039131 mm.
- Real reference median-of-medians: 0.463105 mm.
- Synthetic median nearest-neighbor range: 0.363922-0.674757 mm.
- Synthetic median-of-medians: 0.476373 mm.
- Every synthetic mesh remains below 200,000 vertices.
- Simulated effects include LiDAR-scale surface noise, scan pose, ragged holes, occlusions, and controlled capture-B vertex offsets.

The generator is `scripts/generate_f1_lidar_like_synthetic_corpus.py`. Density is audited by `scripts/audit_f1_synthetic_lidar_density.py`. Truth renders are produced by `scripts/render_f1_synthetic_corpus_truth.py`.

## Visual corpus audit

All five truth contact sheets were reviewed, covering S001-S030. Native-resolution truth renders for S019, S020, and S030 were compared with a native real-scan reference. Bend topology, scan holes, pose variation, and target axes were coherent. The important fidelity limitation is that the synthetic surfaces are more uniformly triangulated than the most pitted and ragged real reconstructions.

The preserved truth audit is `output/research/f1-synthetic-lidar-stress-v2-20260814/truth_visual_manual_audit.json`.

## Frozen V51 result

The exact frozen V51 chain was run in a separate synthetic sandbox. Truth access was denied during inference. Thirty primary and thirty replay executions were semantically byte-identical.

| Metric | V51 |
| --- | ---: |
| Exact cases | 3/30 |
| TP / FP / FN | 30 / 27 / 70 |
| Precision | 0.5263 |
| Recall | 0.3000 |
| F1 | 0.3822 |
| Minimum family recall | 0.0000 |

One real engine defect was exposed: the frozen V16 selector crashed on a valid empty candidate pool with a zero-row feature matrix. A narrow adapter now terminalizes that state as a deterministic zero-prediction selection. The failed first attempt remains preserved; S028 primary/replay smoke tests and the full second attempt were deterministic.

The V51 score is `output/research/f1-synthetic-lidar-stress-v2-20260814/v51-full-primary-replay-attempt2/synthetic_score.json`.

## Candidate ceiling

Source-only candidate-ceiling analysis found:

- 91/100 truth bends had a match somewhere in the full candidate pool.
- V51 selected only 30/100.
- Twenty-two of 30 cases had complete candidate generation.
- The dominant loss was selection and ownership, not candidate birth.
- The largest selection losses occurred in SF12, SF10, SF09, SF15, and SF08.
- True candidates were most often sourced from cylindrical-standard, mesh-adjacency, plane-lattice, and cylindrical-wide routes.

The analysis is `output/research/f1-synthetic-lidar-stress-v2-20260814/v51-full-primary-replay-attempt2/candidate_ceiling_v2.json`.

## V52 bounded recovery

V52 retains V51 and adds only an unrepresented candidate satisfying all of the following:

- cylindrical-standard and cylindrical-wide source agreement;
- at least three source routes in the candidate cluster;
- span at least 50% of the mesh bounding-box scale;
- deterministic ranking and fixed 7-degree, 6-mm, 0.5-overlap representation suppression.

The implementation is `backend/f1_cylindrical_consensus_recovery_v52.py`. It was run under the synthetic truth-denying sandbox by `scripts/run_f1_v52_synthetic_from_v51.py`.

| Metric | V51 | V52 | Delta |
| --- | ---: | ---: | ---: |
| Exact cases | 3/30 | 6/30 | +3 |
| True positives | 30 | 52 | +22 |
| False positives | 27 | 27 | 0 |
| False negatives | 70 | 48 | -22 |
| Precision | 0.5263 | 0.6582 | +0.1319 |
| Recall | 0.3000 | 0.5200 | +0.2200 |
| F1 | 0.3822 | 0.5810 | +0.1988 |
| Minimum family recall | 0.0000 | 0.0000 | 0 |

All 30 V52 primary/replay outputs were semantically byte-identical. V52 added 22 predictions; aggregate TP increased by exactly 22 and FP did not change, so every added prediction matched a truth bend under the locked scorer.

## V52 visual audit

All five scored contact sheets were reviewed. Native-resolution overlays were reviewed for S005, S007, S011, S016, S024, and S029.

- S005 and S007 became exact: the added lines are full-length and coincide with visible fold axes in every view.
- S011 recovered both missing folds; one pre-existing short false fragment remains.
- S016 recovered both missing folds, but three pre-existing diagonal cross-panel false axes remain.
- S024 recovered four of six folds; two misses and one short false fragment remain.
- S029 recovered one of five folds; four misses and one short diagonal fragment remain.

The visual evidence agrees with the score: V52 recovers real fold-aligned axes and does not create extra false positives. It does not solve complete misses, non-cylindrical candidate gaps, or ownership/adjacency errors.

The preserved V52 audit is `output/research/f1-synthetic-lidar-stress-v2-20260814/v52-cylindrical-consensus-v1-attempt2/manual_visual_audit.json`.

## Scientific ruling and next optimization step

Fund V52 as an engineering prototype and keep its rule frozen while working on the remaining error classes. The next bounded cycle should target one defect class at a time:

1. non-cylindrical candidate incompleteness in SF02, SF10, and SF14;
2. ownership/adjacency rejection of short diagonal cross-panel axes;
3. capture-pair stability for families with large nominal/degraded swings.

No synthetic score may be reported as held-out generalization. A genuinely independent, physically scanned and manually adjudicated corpus is still required for confirmatory F1 and paper claims.

## Verification

- `./.venv/bin/pytest -q tests/test_generate_f1_lidar_like_synthetic_corpus.py tests/test_f1_cylindrical_consensus_recovery_v52.py`
- Result: 4 passed.
- All V51 and V52 primary/replay semantic outputs were byte-identical.
- Detector sandbox truth access was denied.
