# Benchmark Stability Critique Log

## Purpose
Keep a traceable record of critique findings, review outcomes, and benchmark-integrity concerns that affected evaluation decisions.

## 2026-03-27 — Initial benchmark-stability critique
### Scope
- Corpus evaluator stability
- Output reproducibility
- Timeout behavior on heavy scans
- Reliability of post-hoc summary regeneration

### Findings
1. Output filename collisions exist in corpus evaluation outputs.
   - Current evaluator writes per-scan JSON using:
     - `Path(scan_name).stem`
   - This can collapse distinct scans onto the same output filename when only the extension differs.
   - Example failure mode:
     - `.ply` and `.obj` variants of the same scan name can overwrite each other.
   - Consequence:
     - post-hoc regeneration from saved JSONs can undercount scans
     - summary files can become non-reproducible from the output directory alone

2. Benchmark summaries are not fully reconstructible from persisted per-scan outputs.
   - Failed-scan bookkeeping and staged metrics depend on runtime context from the original evaluator run.
   - Once a summary is overwritten from file-only reconstruction, the result can be incomplete or misleading.
   - Consequence:
     - summary repair must preserve the original run aggregate unless the evaluator is redesigned to persist a full manifest

3. Heavy-scan timeout behavior is still unstable.
   - Repeated corpus sweeps show persistent `420s` timeouts on a small set of heavy scans.
   - Known problematic family in recent runs:
     - `47266000_*`
   - Consequence:
     - policy comparisons remain directionally useful
     - but benchmark runs are not yet fully stable enough for high-confidence apples-to-apples comparisons

### Decisions taken
- Do not treat file-based summary regeneration as authoritative.
- Preserve the original run aggregate when restoring `summary.json`.
- Keep using fresh evaluation output directories per policy pass.
- Treat timeout instability as a separate benchmark-engineering problem, not a validator-policy problem.

### Action items
1. Fix per-scan output naming so filenames are collision-resistant and deterministic.
2. Persist a run manifest with:
   - expected scan list
   - started scans
   - successful scans
   - failed scans
   - output file paths
3. Make summary generation derive from the manifest instead of directory crawling alone.
4. Add targeted tests for output-path uniqueness and manifest-based summary accounting.

## 2026-03-28 — Benchmark-stability fixes applied
### Completed fixes
1. Collision-safe output naming implemented.
   - Output filenames now use the full scan filename plus a stable hash.
   - This removes `.stem` collisions across extensions.

2. Run manifest implemented.
   - Each evaluation output directory now includes:
     - `manifest.json`
   - The manifest records:
     - part key
     - scan name
     - scan path
     - scan state
     - expected bend counts
     - output JSON path
     - status
     - sequence metadata

3. Summary generation made manifest-driven.
   - Summary generation now reloads results from the manifest rather than inferring scan coverage from directory contents.

### Validation evidence
- Targeted tests passed:
  - `/Users/idant/82Labs/Sherman QC/Full Project/tests/test_evaluate_bend_corpus.py`
  - result: `13 passed`
- Smoke run proving former collision case is fixed:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_smoke_20260328_44211000`

### Remaining benchmark-stability issues
1. Heavy-scan timeout instability remains.
2. Historical benchmark folders created before the manifest fix are still partially non-reconstructible.
3. A broader controlled corpus rerun is still needed to confirm the new evaluator path end-to-end.

## 2026-03-28 — Broad stabilized benchmark review
### Outcome
- The manifest-driven evaluator path held up on a broad corpus rerun.
- Scan accounting is now coherent:
  - manifest entries = `34`
  - summary scans processed = `34`
  - summary results = `34`
- Collision-prone scan families now persist distinct outputs instead of overwriting each other.

### Remaining critique
1. The benchmark is now structurally trustworthy, but not yet fully runtime-stable.
   - one heavy scan still timed out:
     - `47266000_F.stl`

2. Historical benchmark directories created before the manifest fix should not be treated as fully reconstructible.

### Decision
- Use `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_full_20260328` as the new accounting-stable baseline.
- Treat timeout reduction on heavy scans as the next benchmark-stability task.

## 2026-03-28 — Heavy-scan timeout root cause review
### Scope
- isolate the remaining runtime instability on `47266000_*`
- determine whether the timeout lives in bend metrology or in auxiliary post-processing

### Findings
1. The `47266000` heavy-scan failures were not scan-content-specific.
   - `/Users/idant/Downloads/47266000_F.stl`
   - `/Users/idant/Downloads/47266000_F (1).stl`
   - are byte-identical and have the same SHA-256
   - therefore the instability was not caused by geometric differences between the two scan files

2. The core bend pipeline is not the timeout source.
   - Phase-logged single-case run showed:
     - `run_progressive_bend_inspection` finished in about `21s`
   - Therefore the benchmark timeout was happening after the main bend report was already complete

3. The added heatmap-consistency evidence pass is the heavy timeout source.
   - With the advisory heatmap-evidence step enabled:
     - `47266000_F.stl` ran for about `215s`
     - peak memory reached about `18 GB`
   - With the same case run under `--skip-heatmap-consistency`:
     - it completed in about `23s`
     - peak memory was about `1.8 GB`
   - This is a structural diagnostic result, not a heuristic guess

### Decisions taken
- Split **core benchmarking** from **advisory heatmap-consistency benchmarking**
- Add explicit skip control:
  - `--skip-heatmap-consistency`
- Persist per-scan timing so timeout claims are traceable in the manifest
- Use the heatmap validator as a separate benchmark track until its evidence generation cost is reduced

### Resulting baseline decision
- Treat `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328` as the first broad corpus run that is both:
  - accounting-stable
  - runtime-stable for core MAE benchmarking

### Remaining critique
1. Heatmap-consistency benchmarking is still too expensive to share the default core benchmark path.
2. A later pass should reduce the cost of:
   - `build_bend_heatmap_evidence`
   - especially on dense full scans
3. Until then, MAE benchmarking and advisory heatmap benchmarking should be treated as intentionally separate tracks.

## 2026-03-28 — Advisory cache review
### Scope
- reduce repeat-run cost of the advisory heatmap-consistency benchmark path
- preserve exact evidence semantics once a CAD/scan pair has already been processed

### Findings
1. Cache reuse is a valid benchmark-safe optimization for the advisory track.
   - The expensive work is deterministic for a fixed:
     - CAD path
     - scan path
     - file mtimes/sizes
     - tolerance angle
     - deterministic seed
   - Therefore cache-keyed reuse does not change semantics for repeated evaluations of the same pair.

2. The cache materially fixes repeat-run practicality.
   - First advisory run on `47266000_F.stl`:
     - `heatmap_consistency_sec = 338.533`
     - `heatmap_cache_hit = false`
   - Second advisory run on the same pair:
     - `heatmap_consistency_sec = 0.049`
     - `heatmap_cache_hit = true`
   - This is sufficient to make repeated validator/backtest iterations realistic without destabilizing the core benchmark.

### Decision
- Keep cache support in the advisory path.
- Do not move the advisory path back into the default core benchmark loop yet.
- Treat first-run heatmap-evidence cost as the remaining optimization target.

## 2026-03-27 — Heatmap validator review findings captured for traceability
### Review source
- Critique sub-agent review on heatmap-consistency support-recovery work

### Findings that led to fixes
1. Support-score normalization drift
   - `FAIL` / `WARNING` support gate was relaxed
   - score normalization still used the older stricter thresholds
   - consequence: a bend could show `SUPPORTED` with an artificially low score
   - status: fixed

2. Active-fail aggregate drift
   - evaluator tracked `FAIL` support but omitted `WARNING` support from headline aggregates
   - status: fixed

3. Dead `NOT_DETECTED contradicted` reporting branches
   - policy explicitly disallowed contradiction on `NOT_DETECTED`
   - old aggregate fields still implied it was an active branch
   - status: removed from current reporting path

### Remaining non-blocking note
- The final review did not return further blocking findings after the fixes.
- One additional direct service-level test was added for `WARNING` to close the remaining coverage gap.

## 2026-03-29 — Link to professor mathematical critique
The professor's follow-up critique is relevant to benchmark interpretation.

Key point:
- stabilized benchmark numbers are now trustworthy enough to expose the next blocker
- the next blocker is not benchmark accounting; it is lack of formal mathematical specification for:
  - target discrimination
  - datum-island registration
  - observability
  - local bend metrology

Implication:
- benchmark stability work should now support a mathematically specified algorithm program, not open-ended heuristic tuning
- core MAE baseline remains useful, but future deltas should be interpreted against the coming mathematical-specification documents
