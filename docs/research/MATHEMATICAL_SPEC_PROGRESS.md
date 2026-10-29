# Mathematical Specification Progress

## Purpose
Track the conversion of professor guidance into formal, implementable mathematical specs.

## 2026-03-29 — Workstream opened
### Context
- Benchmark stability is now good enough that the next blocker is not accounting noise.
- The professor's first response gave the right architecture but insufficient mathematics.
- The professor's second response supplied the missing mathematical layer.

### Governing source docs
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/PROFESSOR_RESPONSE_MATHEMATICAL_SPEC_20260329.md`
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/PROFESSOR_RESPONSE_MATHEMATICAL_GAP_20260329.md`
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/PROFESSOR_RESPONSE_MATHEMATICAL_WORKLIST_20260329.md`

### Specs drafted
1. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
2. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_LOCAL_BEND_METROLOGY.md`
3. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_OBSERVABILITY_MODEL.md`
4. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_ROLLED_SECTION_VALIDATION.md`
5. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_FEATURE_CLASSIFICATION.md`

### Specs drafted
1. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
2. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_LOCAL_BEND_METROLOGY.md`
3. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_OBSERVABILITY_MODEL.md`
4. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_ROLLED_SECTION_VALIDATION.md`
5. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_FEATURE_CLASSIFICATION.md`
6. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_EXPECTED_ANGLE_TRUTH.md`
7. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_HEATMAP_BEND_CORRELATION.md`
8. `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_LABELS_AND_BENCHMARKS.md`

### Current decision
- The mathematical base is now complete enough to begin code changes without inventing new semantics mid-implementation.
- Do not resume broad algorithm tuning without referencing these documents.
- Keep `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json` as the baseline during this phase.
- Highest-ROI implementation order remains:
  1. datum-island alignment
  2. ternary observability
  3. section-first bend metrology
  4. expected-angle truth and rolled-feature routing
