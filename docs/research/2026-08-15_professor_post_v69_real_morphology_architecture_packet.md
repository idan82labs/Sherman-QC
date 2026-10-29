# Sherman QC F1 — Post-V69 Real-Morphology Architecture Request

Date: 2026-08-15

## Requested ruling

Select exactly one smallest bounded, source-only next intervention that can materially improve real-scan bend-line precision and recall after V69's terminal mass-veto failure.

Return exactly one of:

1. `EXECUTE_ONE` followed by a complete deterministic implementation and one-shot falsification protocol; or
2. `FORMAL_CAPABILITY_OPENING` identifying the smallest scientifically necessary new representation, sensor signal, weak supervision, or truth opening.

Do not offer a menu. Do not tune or rescue V67, V68, or V69. Do not use the quarantined synthetic morphology as positive evidence. Ground the selected mechanism in primary research where possible and distinguish literature-backed claims from your system-specific inference.

## Active objective and promotion thresholds

The target is a trustworthy zero-shot bend inspector for known and near-neighbor sheet-metal families. Promotion requires a locked benchmark of at least 30 scans across at least 15 families with:

- at least 90% exact bend-count accuracy;
- at least 90% line precision and 90% line recall;
- at least 99% render-admissible marker accuracy;
- 100% deterministic replay;
- peak RSS below 16 GiB;
- P95 latency no more than 300 seconds;
- at least 99.5% job success;
- no major family below 80% line recall;
- no truth, filename, identity, expected-count, CAD, or post-hoc tuning leakage.

F1 remains research-only.

## Retained baseline and real-domain deficit

- V52 remains the retained source-only baseline.
- The consumed R001–R008 real affine/resampling lane scored TP 18, FP 22, FN 14: precision 0.45, recall 0.5625, F1 0.50, exact cases 0/8.
- The older historical Gold37 V51 result was TP 42 / 75 predictions against 182 truth lines: precision 0.56, recall 0.230769, F1 0.326848, minimum family recall 0.
- The real problem remains simultaneous: unsupported panel/perimeter/endpoint responses create false positives, while small, rounded, partial, finite, and local bends are missed.

## User visual diagnosis and synthetic quarantine

The user observed that the procedural parts do not resemble the corpus and create many false positives. A second native-resolution side-by-side audit confirmed the diagnosis.

The six synthetic cases were not independent manufacturing families. They repeated one visual grammar: broad clean plate, long full-span flange/lip, rectangular aperture, abrupt tab or shoulder. Their longest and most salient line structures were frequently artificial construction cues, stronger than the designated bends.

The physical references instead show:

- finite material thickness and bend radii;
- continuous formed material flow rather than zero-radius plane joins;
- irregular, rounded, chamfered, stamped, and compound contours;
- local and interrupted bend spans;
- asymmetric end reliefs, ears, returns, and coupled transitions;
- structured viewpoint-dependent dropout, erosion, occlusion, and scan texture.

Matching point spacing or TSDF density did not correct the underlying geometry. Adding noise after clean CAD construction preserved the wrong cues.

Binding disposition:

- all affected synthetic lanes are quarantined from architecture/threshold selection, tuning, promotion, family counts, and generalization claims;
- they may be used only as explicitly labeled adversarial morphology sketches or mathematical/invariance fixtures;
- any future synthetic part must pass native-resolution visual similarity against a nearest real morphology **before** engine inference;
- scan-derived augmentation is preferred over procedural clean-CAD generation.

Evidence:

- visual critique: `docs/research/2026-08-15_f1_synthetic_corpus_visual_critique_and_quarantine.md`

## V67 and V68 terminal results

V67 normal-cycle bridge-mode pair suppression was deterministic but proposed zero suppressions on real candidates. V68 moved the same mechanism to the 48-row pre-NMS stream and added a scalable atom BVH; across 337 pairs and 44 geometry-eligible pairs it again proposed zero suppressions. Both are terminal.

Key V68 outcome counts: 32 `NO_BRIDGE`, 5 `MULTI_MODE`, 4 `OPPOSING_TURN`, 3 `MODE_OUTSIDE_BRIDGE`, zero changed cases.

## V69 frozen architecture

V69 `CANDIDATE_ALIGNED_MULTISCALE_NORMAL_TURN_PERSISTENCE_V1` (`CAMNTP_V1`) was a candidate-local deletion-only veto placed on the frozen V16 thresholded pre-NMS stream.

It used exact-topology triangle-plane cross-sections at 17 longitudinal stations and 15 relative scales, projected-normal shelf coherence, accumulated signed/absolute turn, centered-turn concentration, boundary/nonmanifold clearance, a three-scale plateau, and a 7-linked-station longitudinal run rule. Candidate score, route provenance, identity, filename, truth, CAD, expected count, and other candidates were forbidden.

Implementation/test result:

- 38 source-free tests passed;
- all mandatory positive, negative, comparator-boundary, invariance, exact-refinement, replay, permutation, and resource fixtures passed;
- the 350,000-triangle / 128-candidate stress fixture passed;
- exact replay was byte-identical;
- peak RSS remained below 3 GiB in the real preflight;
- each primary completed in 10.8–21.0 seconds.

## Corrected V69 real actionability result

The first 21-row attempt was invalid because it consumed a post-composition V52 stream rather than the mandated 48-row V16 thresholded pre-NMS stream. It is preserved as invalid evidence and was not used.

The corrected single primary-plus-replay attempt used exactly the required 48 real-corpus candidates:

| Scan | Candidates | Vetoed | Retained | Replay byte-identical |
|---|---:|---:|---:|---|
| U01 | 13 | 13 | 0 | yes |
| U02 | 20 | 19 | 1 | yes |
| U03 | 3 | 3 | 0 | yes |
| U04 | 12 | 12 | 0 | yes |
| **Total** | **48** | **47** | **1** | **yes** |

V69 deleted 97.9% of real candidates, far above the frozen maximum 50% deletion rate. It therefore failed mechanically before visual adjudication or fresh-scan opening and is terminally retired.

Diagnostic structure:

- terminal reasons: 47 `LONGITUDINAL_COUNT`, 1 `KEEP`;
- station-pass counts: 41 candidates had 0 passing stations; the remaining seven had 3, 4, 4, 4, 4, 5, and 10;
- longest linked runs: 41 had 0; the others had 1, 2, 3, 3, 4, 5, and 10;
- the only retained candidate had 10 passing stations and a linked run of 10.

This is an architecture-level representation/actionability failure, not a marginal threshold error. V69 may not be tuned, rescued, or rerun on replacement inputs.

Evidence SHA-256:

- corrected summary: `59215cd9fa02a095ed6b6c6645aa0961ac31b62140867a7f3ccd520ae40c1883`
- corrected lock: `6c03416072a23ea04faac2b83636d110236bbaf6f0be581b654f0cf7706daf3b`
- invalid-attempt summary: `4cc7c558ac24f2bdedd23b5b88c82b7bfdf902e3cd92fd599968504839dc431b`
- invalid-attempt lock: `223dc1c55cf1bedb420c7f4d05aea83dc7538d48cb2da32f2b0fbcefbeca6e28`

## What the next mechanism must achieve

The next intervention must be designed around actual real-scan morphology and must not assume that a valid physical crease yields a complete exact cross-section with bilateral coherent shelves across many stations.

It must distinguish and suppress:

- silhouette/free edges and hole/window perimeters;
- open boundaries, dropout edges, and eroded scan limits;
- broad panel transitions and triangulation seams;
- isolated endpoints and short junctions;
- repeated hypotheses on one physical bend.

It must preserve or recover:

- finite-radius and faceted bends;
- small or shallow creases;
- partial, locally observed, and finite-endpoint bends;
- returns, hems, opposing folds, and nearby distinct bends;
- low-density but visually coherent crease evidence;
- compound stamped/formed regions whose support is not a long ideal straight section.

## Available source-only representation

The intervention may consume only bounded source geometry and candidate geometry/opaque hashes:

- approximately 200,000 bounded source vertices/points;
- at most roughly 350,000 triangles and adjacency when available;
- finite candidate endpoints and directions;
- deterministic local scales derived from source geometry;
- compact derived summaries without source-coordinate serialization.

It may not consume candidate score, rank, route provenance, filename, identity, family, expected count, labels, CAD, benchmark feedback, or manual truth in inference.

Indexed/local computation is available. Per-candidate global rescans are disfavored. Resource target remains below 16 GiB and 300 seconds P95.

## Required response for `EXECUTE_ONE`

Provide all of the following precisely enough to implement without another design round:

1. architecture/version name and abbreviation;
2. exact insertion point and whether it is veto-only, ranking, replacement, generation, or composition;
3. allowed inputs and explicit forbidden-input audit;
4. deterministic preprocessing and invariance requirements;
5. exact numerical constants and tie/equality rules;
6. exact geometric/statistical support measure and failure behavior;
7. handling of boundaries, dropout, rounded bends, partial support, endpoints, and nearby folds;
8. atomic abort/fail-closed rules and resource/visit/cache caps;
9. compact canonical audit schema with no source coordinates;
10. a mathematically consistent source-free fixture suite including positive, negative, comparator-boundary, replay, order, winding, reflection, endpoint reversal, 0.8x/1.25x similarity, exact refinement, and stress cases;
11. a real-scan-only actionability preflight on already exposed scans that cannot tune the mechanism;
12. a native-resolution visual protocol in which raw geometry is judged before intervention reasons or outputs are shown;
13. a fresh physical-scan one-shot only if the actionability and visual gates pass unchanged;
14. locked retain/retire gates for precision, recall, F1, exact count, family safety, determinism, latency, RSS, and job success.

Before selecting `EXECUTE_ONE`, prove that every mandatory positive fixture is compatible with every simultaneous predicate. Include a mechanical actionability gate that prevents spending the fresh scan on another near-zero or mass-deletion method.

## Required response for `FORMAL_CAPABILITY_OPENING`

If bounded source geometry is no longer sufficient, identify exactly one smallest scientifically necessary opening. Specify:

- what new signal or supervision is opened;
- why prior source-only evidence proves it is necessary;
- how leakage and overfitting are prevented;
- the smallest dataset/acquisition requirement;
- a one-shot protocol and success/failure gates;
- what remains closed.

## Scientific discipline

- Do not use quarantined synthetic morphology as transfer evidence.
- Do not propose tuning on R001–R008, U01–U04, Gold37 answers, or the reserved new part.
- Already exposed real scans may be used only for source-only actionability and pretruth visual falsification.
- The new physical part remains sealed for a later genuinely fresh one-shot.
- A failed mechanism is terminal; no threshold rescue or replacement input.
- Prefer a representation capable of modeling real acquisition incompleteness and finite-radius manufacturing morphology over another stack of brittle simultaneous gates.
