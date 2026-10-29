# F1 synthetic V4 visual domain-gap audit

Date: 2026-08-15

## Ruling

The prospective-complex V4 corpus is **not visually representative enough to support a real-corpus generalization claim**. It remains useful only as an adversarial test of long rounded profiles and parallel-transition over-segmentation.

The V62 score improvement on V4 must therefore not promote V62 to runtime, production, or real-scan authority. V62 remains a research mechanism candidate; V61 remains the retained real-scan-facing candidate until a genuinely new finite part receives an independent manual bend slate and a blind comparison.

## Visual evidence reviewed

Synthetic V4:

- all 30 fused source overlays;
- all 30 eight-view calibrated depth sheets;
- all six V61→V62 changed-case source-only overlays;
- all six scored changed-case overlays.

Representative real corpus scans:

- `part_389_case_A.ply` six-view geometry audit;
- `part_472_case_A.ply` six-view geometry audit;
- `48811000_A.ply` six-view geometry audit;
- real-corpus native audits C006, C019, and C030.

## Concrete mismatch

| Dimension | Real corpus | Synthetic V4 |
|---|---|---|
| Global form | Finite brackets, trays, covers, tabs, and assemblies with local changes | Predominantly constant cross-section extrusions |
| Bend extent | Many local or interrupted bend spans | Nearly every transition runs most of the part length |
| Topology | Irregular cutouts, perforations, attachment tabs, unequal ends, thickness returns | Generic repeated oval holes and simple end truncation |
| Curvature | Mixed local radii coupled to finite features | Repeated smooth profile arcs generating several parallel tangent-like lines |
| Annotation density | Usually a small set of manufacturing fold centers | 318 intended bends over 30 cases (10.6 per case), far denser than the operational corpus |
| Failure pressure | Free edges, local attachments, holes, occlusion, short bends, finite ownership | Parallel ridge/tangent discrimination on long profiles |

## Consequence observed

V61 produced 23 false positives and V62 produced 26 on V4. The V62 false positives in O013 and O020 were visually coherent long transitions rather than random noise. They were encouraged by the synthetic construction: broad rounded flanks supplied persistent multi-view line evidence even though the evaluator did not define those tangent transitions as bend centers.

The low V4 recall is also not a clean model-generalization signal. The corpus labels many more full-span profile transitions than the real operational annotation policy, producing 244 V61 false negatives and 240 V62 false negatives.

## Replacement-data requirements

The next evaluation corpus must be real-scan-first:

1. use a genuinely new finite physical part or an untouched real scan, not an invented extrusion;
2. create the manual bend selection visually before running the engine;
3. preserve the real distribution of short/local bend spans, holes, tabs, returns, and scan boundaries;
4. if augmentation is needed, vary acquisition density, view dropout, noise, and occlusion while preserving the real part geometry and fixed manual bends;
5. visually audit raw scan, manual slate, V61 overlay, and candidate overlay at native resolution;
6. do not use V4 to tune a threshold and then present the same corpus as prospective evidence.

## Intake status

No PLY/PCD/LAS/E57 scan newer than the August 13 U01–U04 intake is currently present in the project, Desktop, Downloads, or Documents. The newly received physical part therefore still needs a scan export before it can become the clean blind case.
