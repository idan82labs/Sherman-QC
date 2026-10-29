# Benchmark Stability Progress

## Goal
Stabilize corpus evaluation so benchmark results are reproducible, traceable, and comparable across policy/algorithm passes.

## Current known issues
1. Per-scan output filename collisions due to `Path(scan_name).stem`
2. Summary regeneration cannot fully reconstruct original run state from saved JSON outputs alone
3. Heavy scans still hit the `420s` timeout cap inconsistently
4. Core MAE benchmarking and advisory heatmap-consistency benchmarking still share the same case runner by default

## 2026-03-27 — Benchmark-stability workstream opened
### Context
- Heatmap-consistency backtests reached a point where benchmark integrity matters more than additional threshold tuning.
- Current MAE and validator comparisons are directionally useful, but not yet strong enough for a final benchmark baseline because runtime/output stability is incomplete.

### Work completed
- Identified the output filename collision mechanism in:
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py`
- Confirmed that file-only regeneration of corpus summaries is not authoritative under the current evaluator design.
- Restored the v4 summary from the original run aggregate after detecting undercount in post-hoc regeneration.
- Added a dedicated critique trail:
  - `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/BENCHMARK_STABILITY_CRITIQUE_LOG.md`

## 2026-03-28 — Manifest-driven evaluator baseline
### Code changes
- Added collision-safe per-scan output naming in:
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py`
- Added persisted run manifest:
  - `manifest.json` in each evaluation output directory
- Switched summary generation to load results from the manifest instead of crawling the directory blindly

### Validation
- Tests:
  - `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_evaluate_bend_corpus.py`
  - result: `13 passed`
- Smoke benchmark run:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000`

### Smoke-run proof points
- Distinct colliding scan names now persist separately:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000/44211000/44211000_A.ply--a1522cf5.json`
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000/44211000/44211000_A.stl--e7e42a92.json`
- Manifest persisted:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000/manifest.json`
- Summary persisted from manifest-driven accounting:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000/summary.json`

### Current decision
- Treat manifest-driven summary generation as the new required baseline for future corpus runs.
- Do not rely on `Path(scan_name).stem`-based naming anymore.

## 2026-03-28 — Broad stabilized corpus baseline
### Run
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328`

### Accounting validation
- manifest entries: `34`
- manifest successes: `33`
- manifest failures: `1`
- summary scans processed: `34`
- summary successful scans: `33`
- summary failed scans: `1`
- summary result entries: `34`

### Collision-proof output validation
- verified separate persisted outputs for formerly colliding families:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328/44211000/44211000_A.ply--a1522cf5.json`
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328/44211000/44211000_A.stl--e7e42a92.json`
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328/49001000/49001000_A-PRELIM.obj--180d1981.json`
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328/49001000/49001000_A-PRELIM.ply--bbefc5e1.json`

### Baseline metrics
- heuristic MAE vs expected total completed bends: `1.969`
- heuristic MAE vs expected completed bends: `0.600`
- structured MAE vs expected total completed bends: `1.781`
- structured MAE vs expected completed bends: `0.350`
- heatmap supported rate: `0.300`
- heatmap contradicted rate: `0.090`

### Remaining instability
- one heavy scan still timed out:
  - part `47266000`
  - scan `47266000_F.stl`

### Current decision
- This is now the first broad corpus benchmark baseline that is structurally trustworthy from an accounting perspective.
- Remaining benchmark instability is operational/runtime-bound, not result-accounting-bound.

### Current decision
- Treat benchmark stability as its own engineering track.
- Do not rely on directory crawl alone for benchmark summaries.
- Do not assume per-run comparisons are perfectly controlled until output naming and manifest accounting are fixed.

### Next steps
1. Add a stronger reproducibility test for manifest-driven summary reconstruction.
2. Decide whether to persist per-scan timing fields in the manifest for timeout diagnostics.
3. Tackle the heavy-case runtime instability around `47266000_*`.
4. Add tests for:
   - output filename uniqueness
   - manifest accounting
   - reproducible summary generation
5. Use this stabilized run as the new reference baseline for future MAE comparisons.

## 2026-03-28 — Timeout-path isolation and core benchmark split
### Diagnosis
- The failing `47266000` pair was checked directly:
  - `/Users/idant/Downloads/47266000_F.stl`
  - `/Users/idant/Downloads/47266000_F (1).stl`
- Both files are byte-identical:
  - SHA-256 `9b9fd762f361416ebd33143da5e896bc1bf3c22257c525c04ad71f54a2ed286f`
- Single-case profiling with bend phase logs showed:
  - `run_progressive_bend_inspection` completed in about `21s`
  - the long stall happened after bend report generation, inside the extra heatmap-consistency evidence pass
- Isolated measurement:
  - with heatmap-consistency enabled, `47266000_F.stl` ran for about `215s` and peaked near `18 GB`
  - with `--skip-heatmap-consistency`, the same case completed in about `23s` and peaked near `1.8 GB`

### Code changes
- Added `--skip-heatmap-consistency` to:
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_case.py`
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py`
- Added per-scan timing payload in case outputs:
  - `bend_report_sec`
  - `heatmap_consistency_sec`
  - `total_sec`
  - `heatmap_consistency_skipped`
  - `heatmap_consistency_error`
- Persisted timing into manifest entries for successful scans.

### Validation
- Tests:
  - `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_evaluate_bend_corpus.py`
  - result: `14 passed`
- Isolated heavy-case run:
  - `/tmp/47266000_probe_skip_heatmap.json`
  - completed successfully in `23.16s`

### New core benchmark run
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328`

### Core benchmark outcome
- manifest entries: `34`
- manifest successes: `34`
- manifest failures: `0`
- summary scans processed: `34`
- summary successful scans: `34`
- summary failed scans: `0`
- heuristic MAE vs expected total completed bends: `1.909`
- heuristic MAE vs expected completed bends: `0.571`
- structured MAE vs expected total completed bends: `1.727`
- structured MAE vs expected completed bends: `0.333`

### Current decision
- Treat `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328` as the new **core MAE baseline**.
- Keep heatmap-consistency benchmarking as a separate, explicitly enabled track until its runtime profile is reduced.
- The remaining benchmark-stability issue is no longer accounting; it is the cost of the advisory heatmap-evidence pass.

## 2026-03-28 — Advisory heatmap-evidence cache
### Code changes
- Added cache support in:
  - `/Users/idant/82Labs/Sherman QC/Full Project/domains/bend/services/heatmap_evidence_service.py`
- Added advisory-cache CLI plumbing in:
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_case.py`
  - `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py`
- Added timing fields for cache diagnostics:
  - `heatmap_cache_dir`
  - `heatmap_cache_hit`

### Validation
- Tests:
  - `/Users/idant/82Labs/Sherman QC/Full Project/backend/tests/test_heatmap_evidence_service.py`
  - `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_evaluate_bend_corpus.py`
  - result: `16 passed`

### Heavy-case cache proof
- Warm advisory run:
  - `/tmp/47266000_heatmap_cache_warm.json`
  - `heatmap_consistency_sec = 338.533`
  - `heatmap_cache_hit = false`
- Cached advisory rerun:
  - `/tmp/47266000_heatmap_cache_hit.json`
  - `heatmap_consistency_sec = 0.049`
  - `heatmap_cache_hit = true`
- Cache artifacts:
  - `/tmp/heatmap_evidence_cache_47266000/6333149bdcfd4603/aligned_scan.ply`
  - `/tmp/heatmap_evidence_cache_47266000/6333149bdcfd4603/deviations.npy`
  - `/tmp/heatmap_evidence_cache_47266000/6333149bdcfd4603/deviation_stats.json`
  - `/tmp/heatmap_evidence_cache_47266000/6333149bdcfd4603/bundle_meta.json`

### Current decision
- Keep the core benchmark default on `--skip-heatmap-consistency`.
- Use heatmap-consistency as a separate benchmark track with cache reuse enabled.
- The remaining optimization problem is first-run heatmap evidence cost, not repeat-run cost.
