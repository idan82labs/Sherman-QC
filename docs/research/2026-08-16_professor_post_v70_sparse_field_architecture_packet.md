# Sherman QC F1 — post-V70 sparse-field architecture selection

You are the professor responsible for selecting exactly one bounded next action for the active Sherman QC F1 generalization objective.

Return exactly one of:

1. `EXECUTE_ONE` followed by one fully specified source-only successor architecture; or
2. `FORMAL_CAPABILITY_OPENING_STOP` followed by the smallest scientifically necessary capability opening.

Do not propose a V70 threshold edit, implementation optimization, rerun, replacement input, or rescue. V70 is consumed and terminal.

## Objective and locked promotion target

Generalize bend-line detection across at least 30 scans and 15 part families with:

- exact-count accuracy >= 0.90
- precision >= 0.90
- recall >= 0.90
- F1 >= 0.90
- render-admissible marker accuracy >= 0.99
- deterministic replay = 1.00
- peak RSS < 16 GiB
- P95 latency <= 300 seconds
- no major family below 0.80 recall
- zero truth, identity, filename, expected-count, CAD, score, provenance, or post-hoc tuning leakage

## What V70 established

V70 BRMVT replaced V69 cross-sections with a global source-derived field:

- covariance surface variation at K={24,64,160,256,512,1024};
- local N1280 percentile and relative top-hat contrast;
- two-scale persistence;
- angular/mesh boundary penalty;
- strong/weak hysteresis graph;
- candidate-local partial longitudinal track;
- geometry-only duplicate ownership;
- one unchanged ordinary NMS.

Its complete locked source-free suite passed 24/24, including shallow, rounded, faceted, hem, partial, dropout, opposing, nearby-fold, and sinusoidally displaced positives; flat, free-edge, hole-edge, jittered-edge, cylinder, seam, isolated-short-feature, duplicate, and crossing-feature negatives/ownership; comparator and invariance gates.

## Terminal exposed-real result

The exact one-shot U01–U04 primary/replay preflight used 48 already-exposed V16 thresholded pre-NMS candidates. It opened no truth or identity.

All eight executions atomically aborted in the global field stage:

| Opaque scan | Vertices | Triangles | Candidates | Primary | Replay | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| S8BB74BDA675656176092 | 150,194 | 300,000 | 13 | 301.996s | 300.374s | 1.85 GiB |
| S9567D0D3F33D80F7C915 | 161,295 | 322,216 | 20 | 302.439s | 302.441s | 1.94 GiB |
| SDB1839480738AC85934F | 152,820 | 299,999 | 3 | 302.151s | 301.825s | 3.22 GiB |
| S241AA95BFDFC185004F0 | 153,535 | 299,999 | 12 | 300.516s | 301.906s | 3.49 GiB |

Every primary/replay abort artifact was byte-identical. Memory passed. Latency and no-abort gates failed. No candidate reached real geometry evaluation; no modified output committed. The three-candidate case failed identically, isolating the cost to global field construction rather than candidate tracking.

Evidence:

- terminal report: `docs/research/2026-08-16_f1_v70_brmvt_terminal_runtime_retirement.md`
- summary: `output/research/f1-v70-brmvt-u01-u04-actionability-v1-20260816/summary.json`
- summary SHA-256: `2da80bbc07f9ba1233f41bcaf4277fd6e0a46f049c171caa2f6ddb19b69510bc`
- lock SHA-256: `f22e1a3c77d58b8683ea5411c422abd07103082f39d2e9d39835b4e24229e7b4`

## Required ruling

Select the single smallest action that can materially test the retained geometric idea without re-running V70. Prefer an architecture change such as sparse/candidate-local multiscale evaluation only if you can specify deterministic coverage, boundary behavior, resource caps, mandatory morphology fixtures, exact actionability gates, and native-resolution raw-first visual protocol.

The reserved new physical scan remains sealed. It may be opened only after the selected successor passes unchanged source-free, mechanical exposed-real, and raw-first native-resolution visual gates.

Quarantined synthetic corpus morphology may be used only for mathematical invariance fixtures, not tuning, transfer evidence, family counts, or promotion.

If source-only geometry cannot proceed scientifically, issue `FORMAL_CAPABILITY_OPENING_STOP` and identify the smallest necessary opening. Do not broaden access casually.
