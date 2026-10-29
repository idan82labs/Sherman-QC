# Sherman QC F1 — post-V68 real-crease candidate-validity architecture request

## Requested ruling

Select exactly one smallest bounded, source-only intervention that can reject
V52/V16 candidate lines which do not coincide with a persistent visible
physical crease on real LiDAR scans, while preserving rounded, partial,
finite-endpoint, return/hem, opposing, and nearby distinct bends.

Return exactly one of:

1. `EXECUTE_ONE` followed by a complete deterministic specification and a
   one-shot actionability/visual protocol; or
2. `FORMAL_CAPABILITY_STOP` with the smallest scientifically necessary sensor,
   representation, or truth opening.

Do not offer a menu. Do not rescue V67/V68 or tune their thresholds.

Ground the chosen mechanism in relevant primary research where possible—for
example robust crease/ridge extraction on noisy incomplete meshes or point
clouds, scale-space persistence, integral invariants, normal cycles, or
topology-aware curve support. Clearly distinguish literature-backed facts from
your inference for this system.

## Active objective and promotion thresholds

The research target is a trustworthy zero-shot bend inspector for known and
near-neighbor sheet-metal families. Final promotion requires a locked benchmark
of at least 30 scans across at least 15 families with:

- at least 90% exact bend-count accuracy;
- at least 90% line precision and 90% line recall;
- at least 99% render-admissible marker accuracy;
- 100% deterministic replay;
- peak RSS below 16 GiB;
- P95 latency no more than 300 seconds;
- at least 99.5% job success;
- no major family below 80% line recall;
- no truth, filename, identity, count, CAD, or post-hoc tuning leakage.

F1 remains research-only.

## Current retained baseline and real-domain deficit

- V52 remains the retained source-only baseline.
- The consumed R001–R008 real affine/resampling lane scored TP 18, FP 22,
  FN 14: precision 0.45, recall 0.5625, F1 0.50, exact cases 0/8.
- The older historical Gold37 V51 result (not reusable as a fresh one-shot)
  was TP 42 / 75 predictions against 182 truth lines: precision 0.56,
  recall 0.230769, F1 0.326848, minimum family recall 0.
- The dominant problem is not merely close duplicate lines. Real outputs also
  include distinct but visually unsupported panel/perimeter/endpoint responses,
  while valid small, rounded, partial, and finite folds remain at recall risk.

## Synthetic morphology rejection

The user correctly identified that the generated synthetic parts did not look
like the corpus and created many false positives. Native visual review confirmed:

- synthetic parts were complete, uniformly sampled, rectilinear and CAD-like;
- they overrepresented zero-radius junctions, broad clean planes, rectangular
  windows, perimeter edges and artificial shoulders;
- real LiDAR scans contain partial visibility, ragged boundaries, nonuniform
  density, dropout, thin tabs, finite creases, rounded facets and self-occlusion.

All synthetic morphology lanes are quarantined from architecture selection,
threshold selection, generalization claims and pseudo-truth scoring. The next
intervention must use only real-scan geometry or deterministic real-scan
resampling for development/visual falsification.

Evidence SHA-256:

- visual critique: `e62c1763eeea778f5f8c8d307e09468816d11bba9edccb3cde070b3afe41bb19`

## V67 terminal result

V67 `SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1` replaced ordinary
final NMS with a source-only normal-cycle bridge-mode certificate.

- 34 source-free tests passed after a proof-backed fixture-only erratum.
- The erratum changed no detector constant: two requested positive pair
  boundary fixtures were impossible under their simultaneous endpoint locks.
- On already exposed U01–U04 real meshes, five close pairs were profile
  evaluable, all failed `NO_BRIDGE`, and zero suppressions were proposed.
- No candidate output was committed and pseudo-truth remained sealed.

Evidence SHA-256:

- implementation: `9789c0ee14828534b9f6776f1f79ac56005fa7c399f7418d00fbdd873b0db584`
- fixtures: `41c9f777b66bfb899704439f0830dd2d0879fe03981d2caeee8dd209a9c199e1`
- geometric erratum: `9b7abbc7dd4de3cd6b224cb5ff6c71c1e69a18d00c9063237d55247bcea56bad`
- real actionability summary: `1f13700927c3813d3face2b74c1b0551febf4e360c644390d8317f51569a641b`

## V68 causal follow-up and terminal result

V68 tested whether V67 failed only because it was inserted too late.

- It moved the unchanged NCMM decision to the V16 thresholded stream before
  ordinary NMS, where 48 candidate lines still coexisted.
- A deterministic atom-AABB BVH made the exact same corridor clips scalable.
- 37 combined V67/V68 tests passed, including replay, similarity, winding and
  exact PL refinement.
- Across four real meshes: 337 pairs, 44 geometry-eligible pairs, 3,624,918
  exact atom visits, no aborts, 27.4–45.7 s per primary.
- Outcomes: 32 `NO_BRIDGE`, 5 `MULTI_MODE`, 4 `OPPOSING_TURN`,
  3 `MODE_OUTSIDE_BRIDGE`, zero proposed suppressions, zero changed cases.

This proves insertion timing and full-atom rescanning were real engineering
issues, but neither explains the mechanism's zero actionability. V68 is
terminal and may not be rescued.

Evidence SHA-256:

- implementation: `1feffce435beef80e94141edb3f04a99f6a1f9883d1553d4d70710e121db3f54`
- tests: `9060b63db782f83cb1a190a968a01a95a968584e3ffa8ed4151788574246bfa7`
- real actionability summary: `1da2c708ee221eac8e1221d51a1cdeedaa4e2ab3aed1b1e1efdf058fa35b31be`

## Available source-only representation

The implementation may consume only bounded source geometry and existing
candidate geometry/opaque hashes:

- approximately 200,000 bounded source vertices/points;
- bounded triangle mesh and adjacency when available;
- candidate finite segment endpoints/directions;
- deterministic local scale from source geometry;
- compact derived summaries with no source-point serialization.

It may not consume candidate score, route provenance, filename, identity,
expected count, labels, CAD, benchmark feedback, or manual truth in inference.

Real meshes can contain roughly 300,000 triangles and about 450,000 nonzero
polyhedral hinge atoms after decimation. Any per-candidate or per-pair full atom
rescan is therefore disfavored; indexed/local computation is available.

## What the next mechanism must distinguish visually

Positive candidate validity should mean that a finite line is supported by a
spatially persistent physical crease/ridge in the actual noisy scan, not merely
that it lies near a boundary, panel edge, hole perimeter, finite endpoint,
triangulation seam, or broad normal transition.

It must fail closed on:

- open boundaries and nonmanifold-only support;
- hole/window perimeters;
- isolated endpoint corners and short junctions;
- broad panels without a persistent crease;
- scan dropout/drop edges;
- triangulation-noise curvature;
- duplicate geometry already removed upstream.

It must preserve:

- small but longitudinally persistent creases;
- rounded/faceted bends whose curvature is distributed;
- finite and partially observed folds;
- two nearby physical folds;
- U-channel opposing folds;
- returns and hems/double skins;
- low-density but visually coherent real creases.

## Required response specification for `EXECUTE_ONE`

Provide all of the following precisely enough to implement without another
design round:

1. architecture/version name and abbreviation;
2. exact insertion point and whether it is veto-only, ranking, replacement or
   deletion-only;
3. allowed inputs and an explicit forbidden-input audit;
4. deterministic source preprocessing and similarity/rigid invariance;
5. exact candidate eligibility and every numerical constant;
6. exact persistent-crease support measure, boundary/dropout vetoes, and
   rounded-bend treatment;
7. deterministic decision/tie rules and atomic fail-closed behavior;
8. CPU/RSS/time/visit/cache caps compatible with <16 GiB and <=300 s P95;
9. compact canonical audit schema without source coordinates;
10. a complete source-free fixture suite with mathematically consistent
    positive, negative, equality/below-boundary, replay, order, winding,
    0.8x/1.25x similarity and exact PL-refinement cases;
11. a real-scan-only actionability preflight on already exposed scans that may
    falsify the mechanism without constituting generalization evidence;
12. a genuinely fresh physical-scan one-shot protocol only if the development
    preflight passes unchanged;
13. native 4096px visual artifacts and explicit manual visual pass/fail rules
    before any truth is opened;
14. final locked retain/retire gates for precision, recall, F1, exact count,
    determinism, latency, RSS and family safety.

Before issuing `EXECUTE_ONE`, prove that every mandatory positive fixture is
compatible with every simultaneous eligibility predicate, and include an
actionability gate that prevents spending a fresh scan on a zero-domain method.

## Scientific discipline

- Do not use the quarantined synthetic parts to justify the intervention.
- Do not propose tuning on R001–R008, U01–U04, Gold37 answers, or the new part.
- Already exposed real scans may be used only for source-only actionability and
  pretruth visual falsification.
- The newly scanned physical part remains reserved for a later genuinely fresh
  one-shot after implementation, fixtures, actionability and visuals pass.
- A failed mechanism is terminal; no threshold rescue or replacement input.
