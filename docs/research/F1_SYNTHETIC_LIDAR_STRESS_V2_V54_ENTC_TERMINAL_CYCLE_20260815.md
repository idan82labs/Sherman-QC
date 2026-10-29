# F1 synthetic LiDAR stress V2 — V54 ENTC terminal cycle

Date: 2026-08-15  
Evidence class: synthetic engineering only; non-confirmatory  
Corpus token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`

## Outcome

`V54_EXCLUSIVE_NORMAL_TURN_COVERAGE_V1` is terminally retired. The one authorized execution stopped at S013 with the mandatory scan-level reason `ABSTAIN_V52_PROVENANCE_MAPPING`. It produced no frozen 30-scan output, no replay, and no score. Ground truth was never opened.

The mandatory native-resolution review of every changed case completed before the abstention also rejected both partial V54 additions:

- S010: the proposed axis crosses the interior of a broad flat panel; the visible source curvature is transversely displaced to panel corners.
- S012: the proposed axis crosses the interior of a planar wall; the visible rounded corner is consistently about 8–9 mm away in every axis-normal station.

This is a mechanism-level negative result: longitudinal normal-turn coverage plus exclusive ownership does not prove that the turn evidence is centered on the reported axis.

## Professor authorization

The complete post-V53 ruling is preserved verbatim at:

`docs/research/PROFESSOR_RESPONSE_POST_V53_NEXT_ACTION_20260815.md`

SHA-256: `1b17a87b035caaef94f734edcf9b3039e85ca65591ffd0d6c9137486c36a0700`

The ruling was `EXECUTE_ONE` for V54 ENTC V1. It required an exact frozen V52 prefix, append-only ENTC candidates, deterministic source-only 24-bin exclusive ownership, immediate terminal failure on any scan-level abstention, one execution only, no rescue or rerun, and native-resolution visual review of every changed case.

## Frozen baseline

V52 remains retained and frozen:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v52-cylindrical-consensus-v1-attempt2`

- 30 scans / 15 families
- TP 52, FP 27, FN 48
- Precision 0.6582278481
- Recall 0.52
- F1 0.5810055866
- Exact count 6/30
- Deterministic replay 30/30

V53 remains terminal and contributed no additions. V54 imported no V53 behavior.

## Implementation and preflight

Implemented files:

- `backend/f1_exclusive_normal_turn_coverage_v54.py`
- `scripts/run_f1_v54_synthetic_from_v52.py`
- `scripts/score_f1_v54_synthetic_stress.py`
- `scripts/render_f1_v54_entc_source_only_audit.py`
- `scripts/finalize_f1_v54_entc_visual_gate.py`
- `tests/test_f1_exclusive_normal_turn_coverage_v54.py`

Pre-execution verification:

```text
9 passed in 0.79s
```

The pre-execution lock is preserved at:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v54-exclusive-normal-turn-coverage-v1/preexecution_lock.json`

It records token `V54_ENTC_MECHANISM_LOCKED`, frozen code and dependency hashes, the 30-primary/30-replay order, a 300-second per-scan ceiling, a 16 GiB RSS ceiling, compact aggregate-only output, and `truth_opened: false`.

## Single execution

Terminal record:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v54-exclusive-normal-turn-coverage-v1/terminal_failure.json`

Observed execution facts:

- S001–S012 primary passes completed.
- V54 proposed one addition on S010 and one on S012; all other completed scans were unchanged.
- S013 abstained with `ABSTAIN_V52_PROVENANCE_MAPPING`.
- Status: `TERMINAL_RETIRED_SCAN_ABSTENTION`.
- Terminal token: `V54_ENTC_TERMINAL_RETIRE`.
- Peak process RSS: 892,993,536 bytes (about 0.832 GiB), below the 16 GiB ceiling.
- Truth opened: false.
- Replay performed: false, as required after an immediate terminal abstention.
- Scoring performed: false, because a complete frozen output did not exist.
- Rerun/rescue performed: false.

A read-only static provenance inspection found that S013's frozen V52 prediction sources and B2 support records exist, including all 79 referenced B2 support atom IDs. The failure is therefore in exact source-bank re-resolution rather than missing frozen records. That observation does not authorize a rerun or code correction.

## Native-resolution visual audit

Manual audit record:

`output/research/f1-synthetic-lidar-stress-v2-20260814/v54-exclusive-normal-turn-coverage-v1/partial_visual_manual_audit.json`

For both S010 and S012, the following were reviewed directly:

- raw-source overview;
- six-view immutable-V52 plus V54 composite;
- source normal-turn intensity and 24-bin dashboard;
- nine-station axis-normal source-turn stack.

The render manifests confirm native-resolution source use, zero source-identifier serialization, an unchanged V52 pixel layer outside the ENTC overlay, and exactly one V54 addition per changed case.

### S010

The red V54 line appears long and stable in the six-view overlay, but the axis-normal stack reveals that its cyan axis marker sits in the flat center of the middle panel. The source profile bends near roughly -18 mm and +5 to +9 mm instead. V54 aggregated real turn evidence belonging to offset corners and attributed it to a line through the panel interior. Visual ruling: reject.

### S012

The red V54 line runs along the middle of the tall wall. In all nine axis-normal stations, the cyan proposed-axis marker is on the straight portion at x≈0 while the visible rounded bend is near x≈+8 to +9 mm. Visual ruling: reject.

## Scientific disposition

V54 ENTC V1 is terminal and must not be tuned, rerun, or rescued. Its two independent failure modes are:

1. exact frozen-provenance mapping was not total across the corpus; and
2. its aggregate support statistic was not transversely local to the proposed bend axis.

V52 remains the retained research baseline. No production/runtime/main promotion is authorized. The next cycle must be a fresh, separately named, bounded falsifiable architecture or a formal capability-opening STOP; it may not be a disguised V54 threshold adjustment.
