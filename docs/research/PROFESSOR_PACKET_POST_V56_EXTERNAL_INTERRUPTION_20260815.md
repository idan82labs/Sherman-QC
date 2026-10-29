# Sherman QC F1 — post-V56 external-interruption ruling request

Date: 2026-08-15 (Asia/Jerusalem)

## Requested ruling

Please act as the research professor for the strict-blind Sherman QC F1 program. Select exactly one smallest falsifiable next action, or issue a formal capability-opening/research STOP. Do not authorize post-hoc label tuning, filename or part-identity use, a V56 rescue/rerun, or scoring of incomplete V56 evidence.

This packet reports an operator error plainly: the one-shot V56 process was interrupted during replay S009 because Codex inferred a latency breach from a stale completion count. The filesystem later proved that inference was wrong. V56 is therefore terminal due external interruption, not due measured latency and not due an empirical accuracy result.

## Frozen research objective and constraints

- Strict blind scan-only inference; no gold, labels, expected counts, CAD, filename/identity, or benchmark feedback in inference.
- Deterministic CPU path; approximately 200,000-point bounded upstream detector input.
- Compact cache only; no source-point serialization.
- Peak RSS below 16 GiB; per-scan wall time no more than 300 seconds; deterministic replay required.
- Locked synthetic engineering benchmark: 30 scans, 15 families. Its truth remains closed for V56 because the execution did not complete.
- V52 remains the retained research baseline: TP 52, FP 27, FN 48, precision 0.6582, recall 0.5200, F1 0.5810, exact count 6/30.
- V53, V54, V55, and now V56 are terminal. V56 may not be resumed, replaced, rerun, or rescued.
- Research-only; runtime/main and production remain closed.

## Professor-authorized V56 mechanism

Your prior ruling selected `V56_CANONICAL_TURN_MEDIAN_LINE_RECONSTRUCTION_V1` (`CTMLR_V1`) with precommit token `V56_CTMLR_MECHANISM_LOCKED` and terminal token `V56_CTMLR_TERMINAL_RETIRE`.

The implementation was a complete V52/pool replacement layer:

- total E/S/A seed materialization with redundant-field consistency checks;
- seven stations at fractions 0.10 through 0.90;
- exact triangle/plane sections near each seed;
- multiscale local TLS turn evidence;
- signed turn-mass median and s25/s50/s75;
- five-of-seven, 0.60 coverage, run-three line reconstruction;
- direct transverse centering certificate;
- byte-preserving V52 HOLD only when directly certified;
- V52 reconstruct/prune and pool reconstruct otherwise;
- deterministic output duplicate collapse;
- no V55 import, invocation, or artifact reuse.

The code and downstream gates were implemented before execution:

- `backend/f1_canonical_turn_median_line_reconstruction_v56.py`
- `scripts/run_f1_v56_source_free_preflight.py`
- `scripts/run_f1_v56_synthetic_from_v52.py`
- `scripts/score_f1_v56_synthetic_stress.py`
- `scripts/render_f1_v56_ctmlr_source_only_audit.py`
- `scripts/finalize_f1_v56_ctmlr_visual_gate.py`

The visualizer was prepared to render every changed case, the six V52 exact cases, S011/S016/S024/S029, and every reconstructed/pruned line with a 4096×2304 three-view full-mesh overlay plus seven station sections. It was never run because V56 did not reach a score-ready state.

## Source-free and schema evidence

The source-free suite passed 14/14 fixtures before sealed execution:

1. E/S/A schema equality, reversal/axis-negation invariance, redundant-field consistency, and fail-closed malformed/nonfinite rows.
2. Exact duplicate collapse.
3. Centered sharp 90-degree fold accepted.
4. Centered rounded 60-degree fold accepted.
5. Single plane rejected.
6. 0.40R offset seed recentered and accepted.
7. Equidistant two-fold ambiguity rejected.
8. Two seeds/two folds retained as two.
9. Distributed cylindrical curvature rejected.
10. Open boundary without fold rejected.
11. Diagonal cross-panel seed rejected.
12. Parallel folds separated by approximately 0.08B retained as two.
13. Directly certified V52 line byte-preserved.
14. Canonical primary/replay identity and nonfinite-output rejection.

Source-free fixture result:

- status: PASS
- fixture count: 14
- backend SHA-256: `691fce71662c3f7a1e750c4d732718b537f17021bc396581a3c5d798775dd576`
- source-free runner SHA-256: `6934239964c47af2aabbc8dbbfd99f794606d7f71ce3ca72b95624d4c266dc6f`
- sealed preflight SHA-256: `3c70475a7ef71115a98b8de401f4a62cb58f28ff24e13d1a08d17a8ad5786fd2`

The total pre-execution schema audit covered all rows in all 60 planned primary/replay records. It passed with 100% materialization, no malformed or nonfinite rows, no cap failure, and primary/replay schema identity on all 30 cases. Observed complete schema codes included A, E, S, SA, and ESA. Schema audit SHA-256: `8689a6e9af6c299ce5134e375fcbb0a6c445075842c41c6873d2a07ae5caf220`.

Preexecution lock SHA-256: `c6a867d96e04707fcea84d23cbbd652e1e8074ba0d5380d8a441408e0c0a91df`.

## What actually happened during the one-shot execution

The frozen runner completed:

- all 30 primary cases;
- replay S001 through S008;
- 38/60 records total;
- all eight completed replay cases were semantically and prediction-byte identical to their primary cases;
- maximum completed per-case CTMLR wall time: 256.3601555 seconds (S029 primary), below the 300-second limit;
- maximum completed process RSS: 774,455,296 bytes (about 0.72 GiB), below the 16 GiB limit.

During replay S009, Codex sent KeyboardInterrupt. The stack leaf was inside `build_station_contours` while evaluating a face-center norm. There is no evidence that replay S009 exceeded 300 seconds. Primary S009 had completed in 31.3245731 seconds. The operator's earlier statement that an active case had exceeded 300 seconds was therefore unproven and wrong.

The process did not create a complete execution manifest. Truth was never opened; no V56 score was consumed; no post-score visual gate was consumed. The partial primary predictions must not be treated as a benchmark result.

The terminal artifact records:

- status: `TERMINAL_RETIRED_EXTERNAL_INTERRUPTION`
- reason: `OPERATOR_SENT_KEYBOARD_INTERRUPT_DURING_REPLAY_S009_BEFORE_ANY_PROVEN_LIMIT_BREACH`
- `operator_assessment_error: true`
- `hard_latency_limit_breach_proven: false`
- `resume_or_rerun_authorized: false`
- terminal artifact SHA-256: `6e35b7ec3ef4dd8d115d14ae8b232b00ae8edaf010087386eca7d9c849d15a79`

## Interpretation boundaries

What is supported:

- CTMLR's source-free mechanism and total E/S/A parser passed their precommitted tests.
- All 30 primary executions completed without a measured per-case wall-time or RSS gate violation.
- The first eight replay cases were deterministic.
- The one-shot research protocol was broken by operator interruption before replay completion.

What is not supported:

- no V56 accuracy, exact-count, family-recall, or visual-alignment conclusion;
- no claim that all 30 replays would have been deterministic;
- no claim that V56 passes or fails the retention gates;
- no claim that replay S009 was slow or hung;
- no justification for opening truth, scoring the partial primary set, resuming at S009, or rerunning V56.

## Requested professor decision

Choose exactly one:

1. A fresh, named, bounded source-only architecture with a new mechanism identity and a new one-shot protocol; or
2. The smallest scientifically necessary capability opening, precisely scoped; or
3. A formal research STOP.

Please also rule explicitly on these protocol questions:

- May the unscored V56 primary outputs be used only for truth-free visual/mechanistic diagnosis when designing a genuinely new architecture, or must every V56 output remain closed?
- Should the next one-shot runner enforce the 300-second ceiling in a separate child process per scan so the supervisor can terminate and record a timeout without human inference?
- Is an infrastructure-only replication of CTMLR under a new version forbidden as a V56 rescue? We currently assume yes unless you explicitly define a scientifically valid non-rescue distinction.

Ground the recommendation in relevant robust-geometry, change-point/turn-localization, deterministic evaluation, and experiment-integrity research where it materially constrains the choice. Keep the ruling action-oriented and falsifiable. Specify:

- exact architecture name and one-sentence mechanism;
- allowed inputs and forbidden dependencies;
- source-free positive/negative fixtures;
- total schema and operational preflight;
- fixed constants/formulas;
- execution count/order and hard process supervision;
- compact output/cache schema;
- numerical, family-breadth, determinism, latency, RSS, and visual retention gates;
- terminal failure conditions and no-rescue boundary.
