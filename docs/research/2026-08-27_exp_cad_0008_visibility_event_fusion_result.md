# EXP_CAD_0008 — visibility/event late-fusion result

## Ruling

`COMPLETE_REJECTED / GPI-C_MIXED_POSITIVE_METRIC_VISUAL_REJECTED`.

The intervention moved the matched development metric in the right direction but did not meet its frozen retain gate:

| Gold11 | Retained CAD | EXP_CAD_0008 | Delta |
|---|---:|---:|---:|
| Precision | 0.5000 | 0.5263 | +0.0263 |
| Recall | 0.4225 | 0.4225 | 0.0000 |
| F1 | 0.4580 | 0.46875 | +0.01073 |
| Exact count | 1/11 | 4/11 | +3 cases |
| Minimum-family recall | 0.2000 | 0.2222 | +0.0222 |
| TP / FP / FN | 30 / 30 / 41 | 30 / 27 / 41 | 0 / -3 / 0 |

The source-only hypothesis was partially confirmed: visibility continuity and scan-first ownership remove false positives without losing true positives. It was not strong enough to fund because the preregistered F1 target was 0.5080, recall remained below 0.50, and the native visual gate failed.

## Visual adjudication

All 13 exact candidate payloads were rendered and manually reviewed at native resolution. Six passed and seven failed. Dominant failures:

- scan-first A0 lines can remain overextended even when CAD additions are well cropped;
- conservative visibility filtering leaves obvious recall gaps and a zero-output case;
- repeated/parallel edge ownership remains ambiguous;
- slot/lip/short-return event semantics remain unresolved.

The candidate is not retained and no rescue threshold run is authorized.

## Evidence

- Preregistration: `output/research/autoresearch-loop/EXP_CAD_0008_PREREGISTRATION.json`
- Result: `output/research/autoresearch-loop/EXP_CAD_0008_RESULT.json`
- Score: `output/research/autoresearch-loop/exp-cad-0008-visibility-event-fusion-v1/SCORE.json`
- Candidate-bound visual manifest: `output/research/autoresearch-loop/exp-cad-0008-visibility-event-fusion-v1/VISUAL_MANIFEST.json`
- Visual adjudication: `output/research/autoresearch-loop/exp-cad-0008-visibility-event-fusion-v1/VISUAL_ADJUDICATION.json`

## Next architecture

The next experiment must be materially different rather than another scalar filter:

1. finite endpoint localization must apply to both scan and CAD branches;
2. repeated bends require graph/context identity using cutouts, endpoints, flange adjacency, and relative spacing;
3. each CAD event receives one scan candidate or explicit null under hard exclusivity;
4. visibility uncertainty controls formed/not-formed decisions but does not serve as nominal proof.
