# CAD-first bend localization V1 — result

## Outcome

A reusable CAD-first pipeline now exists: compile the STEP model before scanning, ingest a later scan plus the existing CAD/scan registration, and inspect only scan-supported neighborhoods of CAD bend hypotheses. It is deterministic and production-shaped, but V1 does **not** improve measured bend accuracy and is not an autonomous-QC candidate.

Generalization Progress Indicator: **GPI-C / no new generalization evidence**. The score uses an already-consumed physical slice and is therefore diagnostic only.

## Pipeline

1. `compile_cad_bend_template_v1.py` converts a STEP file to a compact, path-free bend-event template and binds the STEP and extractor hashes.
2. `localize_cad_bends_on_scan_v1.py` accepts only trusted `PLANE_DATUM_REFINEMENT` registration, maps CAD axes into raw-scan coordinates, requires scan support and a two-surface normal turn, deduplicates exact local hypotheses, and otherwise abstains.
3. `run_cad_first_localizer_batch_v1.py` executes the path sequentially over a batch and can render six-view native-resolution audits.
4. `score_cad_first_localizer_v1.py` reports a consumed-physical diagnostic without making a promotion claim.

No gold labels, expected counts, filenames, part identities, or global scan proposals are used during inference.

## Measured result

| Slice | Precision | Recall | F1 | Exact count | Exact localized set | Min-family recall |
|---|---:|---:|---:|---:|---:|---:|
| CAD-first V1, full 13 | 0.3594 | 0.2987 | 0.3262 | 0.0769 | 0.0000 | 0.0000 |
| CAD-first V1, matched 11 | 0.3898 | 0.3239 | 0.3538 | 0.0000 | 0.0000 | 0.0000 |
| Retained CAD incumbent, matched 11 | 0.5000 | 0.4225 | 0.4580 | 0.0909 | 0.0000 | 0.2000 |
| Delta on matched 11 | -0.1102 | -0.0986 | **-0.1042** | -0.0909 | 0.0000 | -0.2000 |

Runtime excluding rendering: median 0.655 seconds and P95 0.879 seconds per case on the local workstation. Eleven cases completed and two correctly abstained because their registration did not meet the frozen trust contract.

## Visual ruling

All 11 completed cases were inspected in six views at native output resolution. The overlay is usually geometrically well registered: prominent CAD axes land on actual formed metal. The dominant failure is semantic accounting, not gross CAD/scan overlay failure.

Observed failure modes:

- parallel inner/outer or sibling CAD surface axes can represent one physical inspection bend;
- rounded slots, lips, and local tab geometry can pass the normal-turn check even though they are not inspection bend events;
- the same CAD template localizes a different number of siblings under different scan visibility, causing count instability;
- two cases correctly abstain because trusted registration evidence is absent.

Accordingly, V1 is rejected as an accuracy winner and for autonomous QC. The infrastructure is retained because it provides deterministic, inspectable CAD-first localization and is conditionally useful as an operator-review overlay.

## Reproducibility

A clean 13-case replay produced byte-identical semantic results and byte-identical CAD templates for every case. Four focused contract tests pass, covering deterministic compilation, controlled abstention, trusted scan verification, and malformed-template rejection.

## Next intervention

The next scientifically justified accuracy intervention is not another scan threshold. It is a source-free CAD-event contraction layer that maps multiple CAD surface primitives to one manufacturing inspection event using signed CIBEC ownership rules. This specifically attacks the visual and metric failure seen here without consulting gold or expected counts. Until that event ontology is authorized and tested on fresh physical cases, V1 remains an operator-assist research path only.

## Commands

```bash
.venv/bin/python3.11 scripts/compile_cad_bend_template_v1.py \
  --step part.step --output part.cad_bend_template.json

.venv/bin/python3.11 scripts/localize_cad_bends_on_scan_v1.py \
  --template part.cad_bend_template.json \
  --scan part_scan.ply \
  --registration registration.json \
  --output result.json \
  --visual-output native_visual_audit.png
```

Artifacts:

- `output/research/f1-cad-first-localizer-v1-20260825/SCORE.json`
- `output/research/f1-cad-first-localizer-v1-20260825/VISUAL_ADJUDICATION.json`
- `output/research/f1-cad-first-localizer-v1-20260825/summary.json`
- `output/research/f1-cad-first-localizer-v1-replay2-20260825/summary.json`
