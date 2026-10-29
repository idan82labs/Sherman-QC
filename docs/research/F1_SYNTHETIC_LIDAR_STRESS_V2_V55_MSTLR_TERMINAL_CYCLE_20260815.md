# F1 synthetic LiDAR stress V2 — V55 MSTLR terminal cycle

Date: 2026-08-15  
Evidence class: synthetic engineering only; non-confirmatory  
Corpus token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`

## Outcome

`V55_MESH_SECTION_TURN_LOCUS_RESELECTION_V1` is terminally retired. Its single authorized execution stopped before completing S001 with the mandatory scan-level reason `ABSTAIN_MALFORMED_FROZEN_LINE`. It produced no complete case, no replay, no score, and no changed visual artifact. Ground truth was never opened.

The failure is a frozen candidate-population schema mismatch, not a measured detector result. V55 treated every row in `pool["candidates"]` as a fully materialized line and required either `line_start` plus `line_end` or two `span_endpoints`. S001 pool ordinal 1, `bend_hypothesis::B1`, is the first row without either representation. S001 ordinal 2, `control_support_strip::S1`, has the same mismatch. Both instead contain `anchor`, `axis_direction`, and `span_mm`, from which a line could be constructed only under a separately preregistered mapping. V55 did not authorize such a mapping, and its scan-level fail-closed abstention was therefore correct.

V55 may not be patched, rerun, or rescued. V52 remains the retained research baseline.

## Professor authorization

The complete post-V54 ruling is preserved verbatim at:

`docs/research/PROFESSOR_RESPONSE_POST_V54_NEXT_ACTION_20260815.md`

SHA-256: `449d092fd3fe853bbc4b2b7f164f265639629be7cb2e0a7642b296b91ab4524e`

The ruling was `RULING: EXECUTE_ONE` for V55 MSTLR V1. It authorized one deterministic source-only primary/replay execution, direct mesh-section turn-locus centering, bounded pruning and additions, immediate terminal failure on any scan-level malformed frozen line, no V53/V54 import, no geometry refit or new candidate birth, no rescue or rerun, and native-resolution review of all changed cases only if cases completed.

## Frozen baseline

V52 remains retained and frozen:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v52-cylindrical-consensus-v1-attempt2`

- 30 scans / 15 families
- TP 52, FP 27, FN 48
- precision 0.6582278481
- recall 0.52
- F1 0.5810055866
- exact count 6/30
- deterministic replay 30/30

V53, V54, and V55 are terminal. None may be rescued. V52 remains research-only and is not promoted to runtime/main or production.

## Implementation and source-free validation

Implemented files and frozen SHA-256 values:

- `backend/f1_mesh_section_turn_locus_reselection_v55.py`: `23bb0f3052b210779b4cb59412bc3517329d8ccf4543600cdb93efe1b68f304e`
- `scripts/run_f1_v55_mstlr_worker.py`: `b8208a9728aff9a37991af6f6c7419fa060bfdb5f1d7555abf3a5ddd6428aae0`
- `scripts/run_f1_v55_synthetic_from_v52.py`: `1e3c1e997eb0517471c0bc6745b58ff4838892ff325e2fe797f0ba1786131ea2`
- `scripts/render_f1_v55_mstlr_source_only_audit.py`: `aaed1daf7b00655132bdc047535d571246bbfe3d33074bb5286e388cf8063288`
- `scripts/score_f1_v55_synthetic_stress.py`: `6c93045bdf117fd3ac769a48789cf90645a4789d5c9b0024ab37e344b142d80e`
- `tests/test_f1_mesh_section_turn_locus_reselection_v55.py`: `8a64014ee99e018155ac483f68ee19f0b5d1720776c37cccbc036c29149ae779`

Focused implementation tests passed before the sealed run:

```text
5 passed in 2.02s
```

The ten professor-specified source-free fixtures all passed. The preserved preflight artifact is:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v55-mesh-section-turn-locus-reselection-v1/source_free_preflight.json`

Its SHA-256 is `cee6f22bdd9650e069f05bc400081dc98be9ec1425a633d19222323d9ecfe30c`.

The pre-execution lock is:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v55-mesh-section-turn-locus-reselection-v1/preexecution_lock.json`

Its SHA-256 is `b818de951023e4060cbe7b5e38110047b4f961504f07206d853fa30568e3e95b`. It records precommit token `V55_MSTLR_MECHANISM_LOCKED`, all frozen primary/replay input hashes, compact aggregate-only output, a 285-second per-scan ceiling, a 15.5 GiB RSS limit, and `truth_opened: false`.

Primary and replay V52/pool artifacts were byte-identical for all 30 cases. The largest frozen pool had 755 rows, below the professor's 2,048-row cap. These checks established byte integrity and boundedness; they did not establish that every heterogeneous pool row had endpoint geometry.

## Single sealed execution

The command was consumed exactly once:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python scripts/run_f1_v55_synthetic_from_v52.py \
  --v52-execution-root output/research/f1-synthetic-lidar-stress-v2-20260814/v52-cylindrical-consensus-v1-attempt2 \
  --pool-execution-root output/research/f1-synthetic-lidar-stress-v2-20260814/v51-full-primary-replay-attempt2 \
  --scan-root output/research/f1-synthetic-lidar-stress-v2-20260814/scans \
  --output-root output/research/f1-synthetic-lidar-stress-v2-20260814/v55-mesh-section-turn-locus-reselection-v1 \
  --confirmation-token SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY
```

Observed facts:

- failed pass: primary
- completed records: none
- first case entered: S001
- worker reason: `ABSTAIN_MALFORMED_FROZEN_LINE`
- terminal reason: `PRIMARY_WORKER_FAILURE`
- terminal token: `V55_MSTLR_TERMINAL_RETIRE`
- worker return code: 2
- peak process RSS: 338,378,752 bytes, below 16 GiB
- truth opened: false
- replay performed: false
- scoring performed: false
- rerun or replacement input: none

Preserved artifacts:

- `primary/worker_failure.json`: SHA-256 `814eb7303ef8a3a46a27305ca11103b651ba015a2f6e920118726baed8682a00`
- `terminal_failure.json`: SHA-256 `0140f5a351020292e629748108cf6494257ab8be3055ac7f531045d07b2c1bea`

## Exact schema evidence

The frozen S001 pool contains 17 rows. Ordinal 0 has valid `span_endpoints`. Ordinal 1 is the first invalid V55 line row:

```text
ordinal: 1
candidate_id: bend_hypothesis::B1
candidate_source: bend_hypothesis
present alternative geometry fields: anchor, axis_direction, span_mm
missing V55 geometry fields: line_start/line_end and span_endpoints
```

Ordinal 2 is independently the same population mismatch:

```text
ordinal: 2
candidate_id: control_support_strip::S1
candidate_source: control_support_strip
present alternative geometry fields: anchor, axis_direction, span_mm
missing V55 geometry fields: line_start/line_end and span_endpoints
```

The professor-mandated malformed-line rule was scan-level, so candidate skipping was not authorized. A conversion from anchor/direction/span is plausible but would be new inference behavior and cannot be introduced post hoc into consumed V55.

## Visual audit disposition

No V55 case completed and no V55 prediction artifact was written. Therefore there was no changed native-resolution render to inspect. This is not a waived visual gate: the gate was inapplicable because the terminal schema failure preceded any output. All prior V52 and V54 visual records remain unchanged.

## Scientific disposition

V55 did not falsify mesh-section turn-locus centering. It failed before that mechanism was evaluated. It did falsify the implementation assumption that the frozen union pool is a homogeneous endpoint-line table.

The next cycle must be separately named and preregister its candidate population and any geometry-materialization rule before execution. It may choose one of three scientifically coherent routes:

1. define the eligible population as only rows already carrying explicit endpoint geometry;
2. define a deterministic, source-free canonical line materialization for every supported frozen schema, including anchor/direction/span rows; or
3. open a minimal capability that regenerates a homogeneous source-derived candidate set.

Choosing among these routes requires a fresh professor ruling. Any new cycle must still be one-shot, source-only unless explicitly opened, deterministic, bounded, compact, visually audited, and forbidden from consulting locked truth before the output freezes.
