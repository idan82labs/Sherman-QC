# 2026-08-30 daily-scan expanded development baseline

## Outcome

Twenty-eight unique raw PLY scans delivered on 2026-08-30 were fetched into a hash-locked custody directory and processed by the retained V51 scan-only pipeline before labels were exposed. Primary/replay inference completed for all 28 with byte-identical semantic outputs.

The scans were rendered into prediction-hidden 4320×3240 six-view sheets. Three independent AI visual reviewers proposed 69, 61, and 63 bend lines. A fourth agent adjudicated the disagreements to 65 finite raw-frame bend lines. An independent fifth-agent audit rehashed every scan, render, review, adjudication, and label artifact and found no integrity blocker.

The 28 cases were appended to a new versioned development catalog, preserving the 37-case v1 catalog byte-for-byte. The expanded catalog contains 65 cases, 55 part IDs, and 247 bend lines.

## Accuracy

| Cohort | Cases | Gold | Predictions | TP | FP | FN | Precision | Recall | F1 | Exact localized sets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy 37, matched V51 rerun | 37 | 182 | 209 | 124 | 85 | 58 | 0.593301 | 0.681319 | **0.634271** | 3/37 (0.081081) |
| Today 28 | 28 | 65 | 77 | 14 | 63 | 51 | 0.181818 | 0.215385 | **0.197183** | 2/28 (0.071429) |
| Expanded 65 | 65 | 247 | 286 | 138 | 148 | 109 | 0.482517 | 0.558704 | **0.517824** | 5/65 (0.076923) |

Scoring uses the catalog's canonical one-to-one line matcher: 5° angle, 4 mm transverse offset, 0.60 gold overlap, and 0.60 prediction overlap.

Two legacy STL scan-geometry cases exceeded the frozen candidate-pool point budget in both primary and replay. They remain in the denominator as deterministic empty-prediction fail-closed abstentions; no case was dropped.

## Authority and interpretation

This is a development baseline, not authoritative ground truth or generalization evidence. Today’s labels are `codex_manual_multiview_scan_only`, with `authoritative_ground_truth=false` and `promotion_eligible=false`. Of 65 adjudicated lines, 19 had three first-pass reviewers, eight had two, and 38 had one first-pass proposal plus fourth-agent adjudication. Human/custodian verification is still required before promotion-tier use.

The low today-only F1 is the central result: the retained V51 composition transfers poorly to this daily cohort. The expanded F1 of 0.517824 must not be presented as an improvement over the matched legacy F1 of 0.634271; adding the new cohort lowers pooled performance.

## Frozen artifacts

- Daily opaque pipeline manifest: `data/bend_corpus/incoming/2026-08-30_shaked_daily_scans/pipeline_manifest.json`, SHA-256 `11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c`
- Prediction-hidden render manifest: `output/research/autoresearch-loop/f1-today-20260830-prediction-hidden-render-pack/render_pack_manifest.json`, SHA-256 `9af6e6e2d7493b41f57fcf979d562425b065f94697550537dfff2cc645231688`
- Adjudicated label manifest: `output/research/f1-today-20260830-ai-labels/consensus-adjudicated/complete_label_manifest.json`, SHA-256 `c3371c0bbcf66e7e08455417b1045ccf0056fae5dfdea184c4fd5a0b0496096c`
- Today V51 execution manifest: `output/research/f1-v51-today-scan-only-development-20260830-r2/development_execution_manifest.json`, SHA-256 `1f87f8bc6c0b6b2b3ca16fb5684f9506c88d86e931821775eeba4aeb6bde57bb`
- Legacy V51 execution manifest: `output/research/f1-v51-legacy37-scan-only-development-20260830-r3/development_execution_manifest.json`, SHA-256 `608b6ccadd65eb1f3b95c4a7d99060eb4203ce6fd86041de38610e81891d6f1a`
- Expanded development catalog: `data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json`, SHA-256 `7cbfd0a0da62ab6181d89a796c5ff217c82660dc718696606d2eed4546a4dee0`
- Prediction map: `output/research/f1-v51-expanded-development-baseline-20260830/prediction_map.json`, SHA-256 `78e282627f570f17639673ddf8995ad40bec51e980b5253c54601def232aa9c6`
- Baseline summary: `output/research/f1-v51-expanded-development-baseline-20260830/baseline_accuracy_summary.json`, SHA-256 `3cc7adb98eabf34f178dc65df30d08502afbcc976e997a90f5413d5f2a34370b`
- Immutable legacy catalog after execution: SHA-256 `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`

Focused verification: 43/43 tests passed across the five new workflow modules plus four legacy-catalog integrity tests.
