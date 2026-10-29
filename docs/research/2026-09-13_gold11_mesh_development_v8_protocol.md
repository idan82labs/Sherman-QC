# Gold11 full11 V8 — prospective consumed-development scoring protocol

Date: 2026-09-13. This protocol is not PRE, source freeze or launch authority.
All prior terminal attempts, R0, raw data, references and historical results
stay unchanged. This is a new full11 run, never a resume/retry of an old attempt.

## Engineering reason and question

V7 completed on all 22 slots with zero emitted supports: 10 extrema-coordinate
capacity refusals, 4 completed prefilters without a passing start, 2 proposal
membership refusals, 2 competitive-interval-width refusals, 2 near-tie storage
refusals, 1 completed residual rejection and 1 CPU termination. V8 changes the
mechanisms behind the first, fifth and sixth groups and one residual gate:

- kd-partition raw extrema V3 inside context V4 (whole-buffer queries evaluate
  a fraction of the rows that the single-axis slab index evaluated);
- triangle surface V4 with a sound nearest-vertex pass-1 threshold, 256 tie
  slots and wide-interval competitors converted to per-row uncertainty;
- registration V4 consumer and initializer V14 / mesh offline V15 wiring;
- `max_inlier_plane_p95_mm` 0.5 -> 1.0 mm for pose admission (provisional
  development gate; V7 CG4BDD control had 98.8%/99.1% coverage at 0.666 mm);
- a deterministic measurement-context schedule (pose-class dedup, evidence-
  ranked cap of at most 8 contexts, further bounded by the frozen consumer's
  window/crop preflight budgets) so admitted near-duplicate contexts cannot
  refuse a whole case at those budgets.

The question is the complete same-input contact-localization score of both
fresh V15 arms, retaining every refusal, timeout and empty output, and the
paired before/after against V7's retained all-22 ledger. Positive geometry on
every case is not a prerequisite; several cases are expected to keep refusing
(see the delta design's deferred blockers).

## Immutable authority and complete denominator

Use only the eleven consumed Gold11 scans, six distinct CADs and unchanged 71
reference lines bound by the September 9 input review. The sole case authority
is the literal CASES in `scripts/review_gold11_inputs_v1.py`, SHA256
41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955.
Reference-container SHA256 remains
4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36;
selected-record snapshot SHA256 remains
d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc.
No mutable manifest may add or substitute a case. No reference placement,
label, endpoint, frame declaration or count is edited.

Keep the exact review-source case order, control then challenger per case:
22 serial inference slots. Original full scan/CAD bytes are bounded and
hash-verified for each slot before dispatch. Gold is accessed only by the
frozen post-terminal scoring path after all 22 owned results/refusals are
reconstructed; it is never available to inference, pose, datum, candidate/
carrier selection, plane construction or measurement.

## Candidate, comparator and interpretation

Interpretation boundary (Codex scope note, 2026-09-12): V8 results are COMBINED
consumed-development changes (new index/surface/registration mechanisms, new
declared capacities and the provisional 1.0 mm pose-admission gate), not an
isolated mechanism speedup, a calibrated accuracy improvement over V7 or a
production result. The arm comparison inside V8 isolates only the scan-plane
lane and scheduler (V7 planes/schedule V1 versus composite planes/schedule V2)
on the same new engine; it is not an old-engine versus new-engine comparison.
Any pose admitted under the 1.0 mm gate is labelled a provisional development
admission with unresolved false-pose uncertainty, never conformance.

Runner: `scripts/run_gold11_mesh_development_v8.py` (basis: the V7 runner,
169180 bytes, SHA256 e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52).
Both fresh arms use native V1, mesh offline V15, initializer V14, context V4,
surface V4, registration V4, stage policies V2 and schedule projection V3.
Control uses scan-plane V7 and schedule V1; challenger uses composite_v1 scan
planes and schedule V2. The complete CONFIG (including the four declared deltas
recorded in `CONFIG_DELTA_FROM_V7`), certificate caps, arm requests, sampler,
scorer and failure handling are bound to exact source/runtime hashes by the
freeze. Do not alter thresholds or scheduling in response to Gold, omit a failed
family, infer an observed count from CAD, or substitute a nominal CAD answer.

Common millimetres, common raw coordinates and whole-part best-fit remain
explicit exploratory assumptions, not verified calibration, stable-datum
metrology or physical uncertainty. The 1.0 mm admission gate is a declared
provisional development assumption, not a tolerance or conformance claim.

## Scoring, negative evidence and claims

Use the unchanged exact contact scorer and legacy one-to-one minimum-cost
matcher: axis angle <= 5 degrees, transverse offset <= 4 mm, Gold-span overlap
>= .60 and prediction-span overlap >= .60. Preserve null/undefined metrics; do
not coerce them to zero or omit their cases. Report each arm's TP/FP/FN,
precision/recall/F1, exact emitted-support count and localized target-set
accuracy, minimum-family recall, full per-case and per-family dispositions,
paired differences, resources and returned stage evidence. Refused/malformed/
empty cases contribute zero predictions and retain all their reference false
negatives. The emitted-support/count proxy is NOT a physical observed-bend
count; physical angle/radius/deviation truth remains missing.

Before/after: the defensible comparison is this fresh paired run against V7's
retained all-22 ledger (both arms TP0/FP0/FN71, F1 undefined) on identical
inputs, references and scorer, plus per-case retained context/registration
counters. Historical F1 .4580152672 is contextual only; do not claim this run
beats it. A nonzero F1 is development progress only when the full custody
audit passes; negative/null results remain intact.

## Expected retained negative outcomes

Besides the deferred blockers listed in the delta design, a case may refuse at
the frozen measurement consumer's preflight budgets because more contexts are
admitted under the 1.0 mm gate (`CONTEXT_WINDOW_PREFLIGHT_BUDGET`,
`FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`, aggregate local-row budgets). These are
retained zero-prediction outcomes of this configuration, never grounds for a
cap change after launch. The enriched report carries
`pose_admission_declaration`, `result_interpretation` and
`arm_comparison_scope` labels binding the combined-change interpretation.

## Fixed prospective resource budget

Unchanged from V7: one heavy geometry job, serial leaves, numerical threads
one, repository 16 GiB ceiling, stricter 12 GiB campaign outer limit.

| Resource | Frozen value |
| --- | ---: |
| Inference per-case wall / CPU | 3540 / 1860 seconds |
| Worker sampled tree / coordinator HWM / outer tree | 8 / 4 / 12 GiB |
| Scan / CAD input buffer | 64 / 8 MiB |
| Leaf output log / complete-or-partial result | 1 / 128 MiB |
| Retained per-case projection and leaf receipt | 8 MiB each |
| Scorer wall / CPU / sampled tree | 180 / 120 seconds / 2 GiB |
| Complete scorer request | 192 MiB |
| Scorer result / enriched report | 32 MiB each |
| Coordinator log / parent bounded read | 2 MiB / 180 seconds |
| Campaign wall / terminal reserve | 104400 / 2400 seconds |

22*3540 + 180 + 2400 = 80460 seconds within 104400. Cases that now pass
initialization may run to their per-case limits; a CPU or wall termination is a
retained negative outcome, never a cap extension.

## Exact release and one-use execution boundary

Required in order: independent source review of the V8 engine wiring
(initializer V14, mesh offline V15, projection V3, runner V8) with the earlier
extrema/surface reviews and the recorded follow-through delta; passing focused
tests recorded in the progress ledger; exclusive `--freeze`; independent exact
physical PRE binding the frozen source/runtime bundle, raw17 declaration and
full11 order, unchanged 71 reference hashes, both arm requests, scorer, limits,
failure policy and control sizes; verified external archive; fresh no-overlap
process check. Only then, once:

```sh
.venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v8.py --execute
```

Actual scorer output, enriched report, all 22 terminal records and the complete
final tree must reconcile before any score is accepted. Independent POST then
checks every case/arm, raw/reference/source/runtime bindings, result/refusal/
projection evidence, scorer/report equality, resource receipts and final
inventory. On failure preserve it, diagnose the mechanism and freeze a
justified successor; never overwrite or reopen this attempt.
