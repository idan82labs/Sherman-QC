# CAD-assisted production readiness V9 — 2026-08-22

## Outcome

The CAD-assisted stack is ready for **bounded operator-assist evaluation**, not autonomous production disposition.

V9 fixes a real STEP-axis defect, preserves the existing automatic CAD/scan overlay as a traceable transform, extracts native B-Rep bend surfaces, and verifies each visible target using both scan proximity and a two-surface normal-turn test. The result is deterministic and visually useful, but accuracy remains below the bar required to accept or reject manufactured parts without a human.

## Material engineering progress

### Correct STEP parameterization

Earlier topology axes evaluated signed STEP V parameters after canonicalizing the cylinder direction. Negative native axes were therefore mirrored. V8 now evaluates V in the native cylinder frame and canonicalizes only endpoint ordering.

This mathematical correction changed the matched native-topology lane from:

- targeted-only F1: 0.2321 → 0.3546
- topology fusion F1: 0.3969 → 0.4238

### Physical two-surface verification

V9 adds a label-free local scan test around each registered STEP axis:

- 10 mm physical corridor;
- deterministic local normal estimation;
- at least 40 local points;
- at least 15 degrees between unoriented surface-normal modes;
- at least 15% support in the smaller surface mode.

No gold lines, expected counts, part identity, or benchmark feedback enter this verifier.

Of 107 native STEP axes, 81 pass scan proximity, 68 pass the normal-turn test, 67 remain after geometric duplicate suppression, and one scan produces a controlled overlay abstention.

## Accuracy on the matched 11-scan physical slice

| Operating lane | Precision | Recall | F1 | TP | FP | FN | Exact count | Minimum family recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scan-only A0 | 0.5758 | 0.2676 | 0.3654 | 19 | 14 | 52 | 0/11 | 0.1111 |
| Native STEP + normal verifier only | 0.3898 | 0.3239 | 0.3538 | 23 | 36 | 48 | 0/11 | 0.0000 |
| Scan + normal-verified STEP | 0.4429 | 0.4366 | 0.4397 | 31 | 39 | 40 | 2/11 | 0.2000 |
| **Retained incumbent: scan + scan-supported CAD** | **0.5000** | **0.4225** | **0.4580** | **30** | **30** | **41** | **1/11** | **0.2000** |
| High-recall three-source review lane | 0.4138 | **0.5070** | 0.4557 | **36** | 51 | **35** | **2/11** | 0.2000 |

The three-source lane recovers six more true bends than the retained incumbent, but adds 21 false positives. Its F1 is effectively tied but slightly lower. It is therefore useful only as a human review queue where missing a bend is more costly than reviewing an extra candidate.

Full 13-scan native STEP + normal-verifier performance is precision 0.3582, recall 0.3117, and F1 0.3333.

## Production operating boundary

### Permitted now: operator-assist pilot

- Display the retained incumbent overlay as the primary candidate set.
- Display topology-only additions in a separate “CAD-supported review” tier.
- Surface the high-recall three-source lane only when an operator explicitly requests all possible bends.
- Show registration residuals and a clear abstention/rescan state.
- Record operator accept/reject decisions as a new independent annotation stream.
- Do not tune on those decisions until the pilot batch is frozen.

### Not permitted now

- Automatic part PASS or FAIL based on bend count.
- Automatic rejection based on an unmatched topology axis.
- Treating “no predicted bends” as a valid inspection when registration abstained.
- Hiding topology duplicates or ontology ambiguity from the operator.
- Updating production thresholds from the current development-consumed slice.

## Visual assessment

All 13 cases were reviewed through five six-view native-resolution contact sheets.

The V8 axis correction visibly fixes multiple mirrored targets. The V9 normal gate removes several weak proximity-only axes while retaining clear tray and bracket bends. Remaining ambiguity concentrates in inner lips, beads, rolled sections, and compound rounded transitions. These are real CAD features, but they do not yet map one-to-one to the line ontology used by operators and F1 scoring.

Visual gate:

- operator-assist: **PASS WITH LIMITATIONS**;
- autonomous disposition: **FAIL**.

## Promotion gate for a production-floor release

The next clean evaluation must freeze all current rules and use the new physical scans with independently annotated manufacturing bend identities. A production release should require, at minimum:

- precision ≥ 0.90;
- recall ≥ 0.90;
- minimum family recall ≥ 0.80;
- exact-count accuracy ≥ 0.90;
- 100% deterministic replay;
- explicit registration-abstention handling;
- native visual acceptance on every changed case;
- measured operator review time and false-reject rate.

The current evidence is far below those accuracy gates. Deployment should therefore be an instrumented human-in-the-loop pilot, not an autonomous QC station.

## Generalization Progress Indicator

**GPI-C — engineering progress without an incumbent F1 improvement.**

V9 materially improves native STEP correctness and recall-oriented operator support, but the best balanced F1 remains the retained 0.458 incumbent. The evaluation slice is also development-consumed rather than an independent holdout.

## Evidence

- V8 result: `output/research/f1-step-topology-targeted-v8-20260822/`
- V9 result: `output/research/f1-step-topology-normal-verified-v9-20260822/`
- V9 replay: `output/research/f1-step-topology-normal-verified-v9-replay-20260822/`
- Multisource comparison: `output/research/f1-step-topology-normal-verified-v9-20260822/MULTISOURCE_GOLD11_DIAGNOSTIC.json`
- Native visual audit: `output/research/f1-step-topology-normal-verified-v9-20260822/native-visual-audits/`
- Visual ruling: `docs/research/artifacts/2026-08-22_cad_assisted_production_v9_visual_audit.json`
- GPI: `docs/research/artifacts/2026-08-22_cad_assisted_production_v9_gpi.json`
