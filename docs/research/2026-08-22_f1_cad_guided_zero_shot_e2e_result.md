# CAD-guided zero-shot E2E result — 2026-08-22

## Outcome

CAD is useful as infrastructure, but not as a replacement for scan evidence.

On the exact 11-scan physical slice shared with the frozen A0 scan-only baseline, a fixed label-free union of scan-only predictions and scan-supported CAD predictions increased line-localization F1 from **0.3654 to 0.4580** (**+0.0926**). Recall increased from **0.2676 to 0.4225** while precision decreased from **0.5758 to 0.5000**. Minimum family recall increased from **0.1111 to 0.2000** and exact-count accuracy increased from **0/11 to 1/11**.

This is a positive development result, not yet a generalization or promotion result. The 11 matched scans were already consumed during development, their truth is not a new independent annotation set, and adding CAD changes the task from scan-only zero-shot to CAD-guided zero-shot.

## E2E coverage

- 13 physical scans, 8 CAD-backed families in the full CAD lane.
- 13/13 jobs completed successfully under the blind subprocess contract.
- Expected counts, labels, gold lines, and part identity were not exposed to inference.
- Full-pipeline replay completed 13/13 jobs.
- Detected line endpoints, the scan-frame-gated line set, and all score tables reproduced exactly.
- Primary-run p95 latency was 37.63 seconds per case.

Two full-lane cases had no exact A0 hash match, so the controlled scan-only comparison uses the remaining 11 scans across 6 families.

## Controlled results

| Mode | Precision | Recall | F1 | Exact count | Minimum family recall |
|---|---:|---:|---:|---:|---:|
| Scan-only A0 | 0.5758 | 0.2676 | 0.3654 | 0/11 | 0.1111 |
| Raw CAD-only | 0.2344 | 0.2113 | 0.2222 | 3/11 | 0.0000 |
| Scan-supported CAD-only | 0.4545 | 0.2817 | 0.3478 | 1/11 | 0.0000 |
| Scan/CAD intersection | 0.5882 | 0.1408 | 0.2273 | 0/11 | 0.0000 |
| **Scan + scan-supported CAD union** | **0.5000** | **0.4225** | **0.4580** | **1/11** | **0.2000** |

The fusion rule was fixed before scoring: retain all A0 scan lines and add a CAD proposal only when it survives the pre-existing scan-frame support gate and is not a geometric duplicate of an A0 line. Duplicate thresholds were fixed at 5 degrees direction difference, 4 mm transverse offset, and 60% overlap of the shorter span. No parameter was fitted to these scores.

## Visual ruling

All 13 scan-supported CAD artifacts were inspected in six native views through five contact sheets.

The support gate materially improves the overlays: most surviving lines lie on visible edges or flange transitions, and the conspicuous floating axes from raw CAD inference are removed. The remaining weaknesses are visible and important:

- `10839000_01_full_scan` is over-segmented around its rectangular frame.
- `40500000_0_SCAN2` contains an implausible interior-spanning proposal.
- Partial `38172000` views remain under-covered.
- Short-feature endpoints are still inconsistent.

Visual verdict: **pass with limitations for another bounded development test; fail for production release**.

## Interpretation

The CAD acts as a structural prior: it supplies plausible bend locations that the scan-only detector misses. The scan then acts as the reality check, rejecting CAD lines unsupported by measured points. This combination recovered 11 additional true positives at the cost of 16 additional false positives relative to A0, producing the net F1 gain.

CAD alone performed worse because registration and visibility are not yet reliable enough. A nominal CAD bend can be absent from a partial scan, shifted by registration error, or mapped onto an outer boundary rather than the physical bend axis. Therefore the next engineering target is **CAD-to-scan registration and visible-axis localization**, not broader CAD enumeration.

## Generalization Progress Indicator

**GPI-C — positive development indicator, no clean generalization evidence.**

The direction is encouraging and deterministic, but the evidence is development-consumed. The promotion test must use the same frozen fusion rule on genuinely new physical scans with CAD and independently trusted bend-line annotations. No tuning may occur after those scans are opened.

## Evidence

- Slice: `data/bend_corpus/evaluation_slices/cad_guided_gold13_physical_20260822.json`
- Primary summary: `output/research/f1-cad-guided-gold13-e2e-20260822/summary.json`
- Scan-frame gate: `output/research/f1-cad-guided-gold13-e2e-20260822/SCAN_FRAME_GATE_AUDIT.json`
- Controlled fusion scores: `output/research/f1-cad-guided-gold13-e2e-20260822/FUSION_GOLD11_DIAGNOSTIC.json`
- Replay scores: `output/research/f1-cad-guided-gold13-e2e-replay-20260822/FUSION_GOLD11_DIAGNOSTIC.json`
- Native visual artifacts: `output/research/f1-cad-guided-gold13-e2e-20260822/native-visual-audits-scan-frame-gated/`
- Visual ruling: `docs/research/artifacts/2026-08-22_cad_guided_e2e_visual_audit.json`
- GPI: `docs/research/artifacts/2026-08-22_cad_guided_e2e_generalization_progress_indicator.json`
