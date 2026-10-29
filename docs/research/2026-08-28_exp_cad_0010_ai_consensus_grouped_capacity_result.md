# EXP_CAD_0010 — AI-consensus ontology and grouped-capacity fusion

## Ruling

`COMPLETE_REJECTED_TERMINAL / GPI-C_NEGATIVE_DEVELOPMENT_VISUAL_REJECTED`.

Three independent GPT-5.6 Sol/max reviewers produced 63 event decisions, a separate adjudicator resolved disagreements, and all three reviewers accepted the same hash-bound V2 mapping with zero remaining challenges. The mapping was then integrated into a preregistered source-only fusion candidate and benchmarked.

The candidate did not improve accuracy:

| Gold11 | Retained CAD | EXP_CAD_0009 | EXP_CAD_0010 | Delta vs retained |
|---|---:|---:|---:|---:|
| Precision | 0.5000 | 0.5179 | 0.5091 | +0.0091 |
| Recall | 0.4225 | 0.4085 | 0.3944 | -0.0282 |
| F1 | 0.4580 | 0.4567 | 0.4444 | -0.0136 |
| Exact count | 1/11 | 4/11 | 3/11 | +2 cases |
| Minimum-family recall | 0.2000 | 0.2222 | 0.2222 | +0.0222 |
| TP / FP / FN | 30 / 30 / 41 | 29 / 27 / 42 | 28 / 27 / 43 | -2 / -3 / +2 |

The retained CAD incumbent remains unchanged.

## Exact causal finding

Only one case changed geometrically from EXP9: `CG191DEC7F81`.

The V2 ontology grouped two disconnected but collinear tab roots into one physical event. EXP10 therefore removed `C002:CAD_EVENT_0003`. That removed one true positive and zero false positives. The complete aggregate regression is explained by this single contraction.

The corrected production rule is:

> Inspection-event identity is defined by connected physical component ownership plus forming axis. Collinearity alone must never merge disconnected tabs, because each tab can carry an independent angle, springback, or deformation defect.

This is more specific than the prior “one axis = one event” formulation.

## What passed

- 63/63 AI ontology decisions were hash-bound.
- Three independent reviewers accepted the same 53-group V2 mapping before execution.
- The truth-denying binding manifest exactly matched all 13 parent event sets.
- Primary and replay semantic outputs were byte-identical for 13/13 cases.
- Forty-five finite physical groups were capacity-correct.
- EXP9's conflict-null behavior was eliminated.
- C005 emitted zero CAD lines, as frozen.
- Source execution used no gold or expected counts.
- Runtime was approximately 4.6 seconds.

## What failed

- F1, effect-size, and recall gates failed.
- Two independent post-execution visual reviewers agreed exactly: 4/13 pass, 9/13 fail.
- Major recall gaps and scan-only free-boundary false positives remained.
- The AI-authored mapping was derived from these same consumed cases, so it cannot provide unseen-family generalization evidence.

## Generalization progress indicator

`GPI-C_NEGATIVE_DEVELOPMENT_VISUAL_REJECTED`.

This was a valid consumed-development diagnostic. It improved infrastructure and identified a sharper event-identity rule, but detector accuracy moved backward. It is not GPI-A/B evidence and authorizes no production promotion.

## Evidence

- Consensus lock: `output/research/cad-line-ontology-authority-pack-v1/AI_CONSENSUS_FINAL_LOCK_V2.json`
- Binding manifest: `output/research/autoresearch-loop/EXP_CAD_0010_CASE_BINDING_MANIFEST.json`
- Preregistration: `output/research/autoresearch-loop/EXP_CAD_0010_PREREGISTRATION.json`
- Result: `output/research/autoresearch-loop/EXP_CAD_0010_RESULT.json`
- Score: `output/research/autoresearch-loop/exp-cad-0010-ontology-grouped-capacity-v1/SCORE.json`
- Visual adjudication: `output/research/autoresearch-loop/exp-cad-0010-ontology-grouped-capacity-v1/VISUAL_ADJUDICATION.json`
- Independent protocol/metric audit: `output/research/autoresearch-loop/exp-cad-0010-ontology-grouped-capacity-v1/EXP10_AI_PROTOCOL_METRIC_AUDIT.json`

## Next action

Do not split C002 and rerun EXP10; that would be a post-score rescue on a consumed benchmark. Retire this version.

Any later CAD event architecture must encode connected-component ownership before event grouping and must be evaluated prospectively on different physical parts. The capacity-preserving unmatched-group behavior and truth-denying binding infrastructure are retained as engineering mechanisms, not as a promoted detector.
