# F1 generalization: 70-version multi-agent bigger-picture audit

Date: 2026-08-20

Scope: independent read-only audits of the architecture trajectory, scientific evidence, and systems/promotion readiness across the V16-V73 research history. The audit was requested before making another architecture investment.

## Executive ruling

The program has made real engineering and scientific-learning progress, but it has not yet demonstrated physical zero-shot generalization.

The strongest gains are deterministic execution, source-only isolation, auditability, failure containment, and a clearer map of mechanisms that do not transfer. Those are valuable. They are not equivalent to an F1 improvement on unseen physical families.

The current research strategy is architecture-thrashing: dozens of non-comparable mechanisms were promoted or retired against different synthetic, pseudo-truth, visual, and physical fragments. The next dollar should fund a valid measuring instrument and independent physical evidence, not an immediate V74.

## Evidence that currently supports the claim

### Physical evidence

- Gold37 A0: 37 scans across 27 part IDs, but only 13 scans across 10 part IDs are promotion-eligible under the truth-tier rules.
- Full Gold37 result: 1/37 exact localized target sets, precision 0.560, recall 0.231, F1 0.327, and minimum-family recall 0.
- Promotion-eligible subset: precision 0.469 and recall 0.211.
- Development-selected strict LOFO reached approximately F1 0.510 with 3/35 exact localized target sets. Because the same folds influenced development, this is not untouched confirmatory evidence.
- Real-scan affine resampling was a strong negative robustness test: 0/8 exact relative to pseudo-truth, with large count changes on same-parent transforms.

### Non-physical evidence

- V16 is the strongest comparable held-out fixture result: precision 0.6875, recall 0.7500, F1 0.7174, 4/35 exact, and minimum-family F1 0.
- V52's 30-scan/15-family synthetic corpus reached F1 0.581, 6/30 exact, and minimum-family recall 0. It is a synthetic construction result, not physical generalization.
- V61 transferred between synthetic generators only weakly: hard-V3 recall 0.258.
- P01-P07 and V46-V51 provided useful mechanism and delivery-count observations, but did not have trusted line-level ground truth.
- V69-V73 mostly exposed actionability, runtime, cap, and invariance failures before physical opening. V73 ended at 31/32 source-free cases because G24 geometry identity was not invariant.

### Capacity evidence that must not be called generalization

The Shaked 12-scan result (9/12 exact, F1 0.967) demonstrates capacity on an inspected set. It is not a clean zero-shot estimate: 49/59 labels were snapped to existing candidates.

## What the versions genuinely taught us

1. Manufacturing provenance is not reliably recoverable from scan geometry; the target must be defined in scan-observable terms.
2. Candidate birth is not the only bottleneck. Ownership, reconciliation, deletion, and absence-as-evidence repeatedly destroy recall after plausible candidates already exist.
3. Deterministic duplicate reconciliation and route-aware selection are useful and reproducible.
4. Cylindrical agreement and organized raw-frame views can work in controlled synthetic settings, but fixture success does not predict real morphology.
5. Missing scan support is unsafe as positive evidence of “not a bend.” Occlusion and sampling create the same signal.
6. Synthetic morphology that looks unlike the physical corpus creates false confidence and false positives; the quarantined synthetic families must remain non-authoritative.
7. The governance discipline now prevents silent rescue and leakage, but one-shot terminal rules have sometimes converted ordinary engineering defects into new architecture numbers instead of comparable iterations.

## Measurement defects that currently block an honest claim

1. `exact_count_accuracy` is misnamed in existing evaluation code: it requires every predicted line to match a gold line, so it is exact localized target-set accuracy rather than count equality.
2. The “99% marker” metric is currently a per-case predicted-count-equals-gold-count proxy, not a validated marker admissibility or visibility judgment.
3. The visual gate checks that audits exist; it can pass even when the visual ruling is failure. It does not aggregate a coherent pass/fail decision.
4. Family identity is often proxied by part ID. A defensible generalization claim needs an external morphology/family ontology fixed before evaluation.
5. Truth needs observable/ambiguous/unobservable states, two independent annotators, disagreement measurement, adjudication, and provenance.
6. Repeated adaptive selection on the same small folds has consumed their confirmatory value.

## Systems and promotion status

Promotion status is RED.

- There is no single authoritative gate: research language uses 30 scans/15 families while enterprise documents use 150/75.
- Current CI can invoke a checker with `--allow-blocked`, so it is not a fail-closed promotion barrier.
- The research stack is largely untracked in a very large dirty workspace, which prevents a reproducible release identity.
- Runtime/main does not currently import the research F1 modules, which is good isolation, but the product pipeline still has truth-hint and filename-sensitive paths that must not enter a blind adapter.
- Daily-job reliability is not production-grade: work is in-memory, without a durable queue, durable job state, process-tree timeout/kill, retry policy, or operational telemetry.
- Container/Helm resources do not encode the research memory and execution contract.

## Statistical reality of the stated gates

A 30-scan/15-family corpus is a useful research gate, not strong evidence for a 99% production claim.

- 27/30 exact has an approximate one-sided lower confidence bound near 0.76.
- Even 30/30 perfect has a lower bound near 0.905.
- Roughly 299 perfect independent marker outcomes are needed for a 99% lower bound near 99%; approximately 598 jobs for a 99.5% lower bound under the same idealized assumptions.

These calculations do not replace family stratification or independence; they show why 30/15 should be described as a research milestone.

## Fund / pause / stop

### Fund now

1. Repair and rename the metrics so exact count, exact localized target set, precision/recall/F1, marker admissibility, and visual rulings are separate.
2. Define and freeze an external family ontology and a canonical candidate/stage intermediate representation.
3. Acquire at least 30 untouched physical scans across at least 15 externally defined families, preferably two independent physical instances per family.
4. Label them with two independent annotators plus adjudication, including ambiguity and visibility.
5. Freeze one runner and one model snapshot, then execute once and visually inspect every case at native resolution.
6. Preserve a clean, immutable research commit, dependency lock, artifact registry, and hashes before the run.

### Pause

- New architecture numbering after V73.
- Further rescue on synthetic or previously consumed physical folds.
- Professor-selection loops that do not add new physical information.
- Runtime/main integration or production claims.

### Stop treating as success evidence

- Synthetic transfer alone.
- Source-free fixture locks alone.
- Count-delivery agreement without trusted line truth.
- Visual-audit completeness without a passing adjudication.
- Results selected after seeing the same evaluation families.

## Ordered next phase

1. Measurement repair: correct scorer semantics, marker semantics, and visual-gate aggregation; add regression tests.
2. Corpus custody: inventory new physical scans, seal a 30/15 split by external family, and prohibit filenames/expected counts from the model path.
3. Ground truth: independent manual bend selections, ambiguity/visibility tags, adjudication, and native-resolution overlays.
4. Frozen baseline: execute the retained baseline once on the untouched corpus.
5. Only after the error map exists, choose one intervention aimed at the dominant physical failure mode and compare it against the frozen baseline.
6. Production work begins only after the research gate passes and a separate soak/reliability program succeeds.

## Native-resolution cross-check performed during this audit

The existing Aug-15 U01-U04 affine-resampling contact sheet and the original-resolution R001, R004, and R008 prediction-only overlays were inspected again during this audit.

- R001/U01 shows that plausible candidate birth is already abundant: several almost parallel axes stack onto the same long fold, while short fragments cluster around the central junction. The visible failure is ownership/consolidation and junction handling, not absence of raw candidates.
- R004/U02 contains long lines crossing the interior of a visually flat face, including diagonal and horizontal overreach. This is a direct flat-surface false-positive mode and a transform-instability symptom.
- R008/U04 is visually clean for the two emitted opposing perimeter folds, but the pseudo-truth comparison records two additional missed folds. Visual cleanliness alone therefore cannot establish recall.

Across all eight resampled cases, the prior audit found only R008 visually passing, while count/localization changed sharply between same-parent variants. This supports a narrow next hypothesis: stabilize canonical representation, reject unsupported flat-face crossings, and reconcile duplicate/fragment candidates without deleting unique folds. It does not support funding another independent candidate-birth mechanism.

## Bottom line

We have learned enough to stop guessing architectures. The limiting resource is now valid physical evidence and trustworthy scoring. Fixing that is not bureaucracy; it is the action that turns the next engine run into a meaningful answer about generalization.
