# Sherman QC CAD bend-line ontology authority pack V1

Date: 2026-08-25

Review-pack SHA-256:

`c52c7399cd7b3263e8f2dd3b9191c0dc356b7175a4ea9e5cb1cf5069a89e2c3f`

## Why this review is required

EXP_CAD_0007 showed that a mathematically valid intersection of the two
observed scan flanks remains visibly on the formed region, yet can score worse
than the CAD neutral half-turn line. The matched-11 F1 fell from 0.3721 to
0.2791 while all 50 relocated lines passed geometric visual review. Therefore
the current bottleneck is not merely localization accuracy: the inspection
line itself lacks a signed manufacturing definition.

Further interpolation or threshold fitting against consumed labels would tune
to an unresolved convention and is prohibited.

## Review material

The pack is at:

`output/research/cad-line-ontology-authority-pack-v1/`

It contains 11 consumed physical cases and 63 CAD events. Each six-view image
shows three source-only alternatives:

- yellow dotted: paired-cylinder curvature-center axis;
- magenta dashed: neutral-radius half-angular-turn line;
- cyan solid: intersection of two locally fitted physical scan flanks.

The pack does not show gold, expected counts, scores, or model correctness.
Case sidecars map the visible indices to opaque CAD event IDs.

## Required independent decisions

Two distinct people—the manufacturing authority and countersigning
authority—must independently complete their respective JSON templates. For
every event they decide:

1. `QC_BEND`, `NOT_QC_BEND`, or `UNRESOLVED`;
2. for a QC bend: `CURVATURE_CENTER`, `NEUTRAL_HALF_TURN`,
   `OBSERVED_SURFACE_INTERSECTION`, or `OTHER_AUTHORITY_DRAWN`;
3. a concise rationale;
4. an explicit replacement line when selecting `OTHER_AUTHORITY_DRAWN`.

The complete 63-event denominator is mandatory. The authorities may not share
answers before submitting their first decisions. Disagreements are reported
for adjudication rather than silently resolved.

## Validation

For a guided local review instead of editing JSON directly, run:

```bash
.venv/bin/python3.11 scripts/serve_f1_cad_line_ontology_review_v1.py
```

The page stores the two reviewer roles independently in browser-local storage,
requires all 63 statuses, conventions, and rationales, and exports a JSON file
in the exact validator schema. Each reviewer must export their first complete
file before discussing decisions with the other reviewer.

```bash
.venv/bin/python3.11 \
  scripts/validate_f1_cad_line_ontology_authority_decisions_v1.py \
  --manifest output/research/cad-line-ontology-authority-pack-v1/MANIFEST.json \
  --manufacturing output/research/cad-line-ontology-authority-pack-v1/MANUFACTURING_AUTHORITY_DECISION_TEMPLATE.json \
  --countersigning output/research/cad-line-ontology-authority-pack-v1/COUNTERSIGNING_AUTHORITY_DECISION_TEMPLATE.json \
  --output output/research/cad-line-ontology-authority-pack-v1/VALIDATION.json
```

Valid matching decisions return `READY_TO_AMEND_CIBEC_CHARTER`. Independent
disagreements return `AWAITING_ADJUDICATION`. Missing identities, choices,
replacement geometry, altered denominators, or the wrong review hash fail
closed.

## Visual QA

The pack was inspected at native resolution on both a compact three-bend
rounded part (`C005`) and the densest twelve-event part (`C013`). Color and
line-style separation remain legible in orthographic and oblique views. C013
is necessarily dense, but numbered events remain traceable across views.

## Current state

The validator is working and currently returns `AWAITING_AUTHORITY_REVIEW`.
No authority identity or decision has been invented by Codex. Once two real
decisions agree, their output can be used to amend and sign the existing
CIBEC_V1 charter before another localization experiment is funded.
