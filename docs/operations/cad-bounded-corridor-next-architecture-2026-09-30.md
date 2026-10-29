# CAD bounded-corridor coverage: next architectural gate

Date: 2026-09-30. Source-only diagnosis; no native execution or raw CAD access.

## Outcome and scope

Root released the terminal derived EXTRACTION/RESULT/REJECTIONS/DIAGNOSTICS records of `artifacts/vendor-cad-corpus-bounded-corridor-v5-2026-09-30-attempt1`. Root observed all 20 cases exit zero with process-tree cleanup, 31.49 seconds total, 98 extracted axes, 19 folded displays, and six flat previews. Flat availability is restored to the baseline six; neither availability nor candidate count establishes extraction precision/recall.

This report examines the 24 remaining ordinary baseline pair groups, identified by native face identities, not by treating baseline detections as truth. Frozen extractor V6 and all frozen artifacts remain unchanged.

## Exact refusal evidence

All 48 associated skins have one outer wire and stop on a visited B-spline pcurve with `PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK`. The failing coefficient receipt reports positive weights with range `[1,1]`: these failures are not caused by a rational denominator sign. All declare source tolerance `1e-7` mm and corridor fraction `0.25`, hence permitted coefficient corridor `2.5e-8` mm.

Coordinates are physical cylinder coordinates `(R*u,v)`, both measured in mm. The following are control-coefficient range widths, **not measured curve excursions or material bounds**. The best possible axis-aligned control corridor is half the smaller coordinate width.

| Case | Pair groups / skins | R*u coefficient width (mm) | v coefficient width (mm) | Best coefficient half-width (mm) |
| --- | --- | --- | --- | --- |
| 04 | 3 / 6 | 0.577927–8.377580 | 0.229308–2.929897 | 0.114654–1.464949 |
| 11 | 2 / 4 | 1.613237–7.853982 | 0.030717–0.697721 | 0.015359–0.348860 |
| 12 | 1 / 2 | 0.862697–1.294046 | 0.127017 | 0.063508 |
| 13 | 4 / 8 | 0.841610–6.451298 | 0.034157–1.268008 | 0.017078–0.634004 |
| 16 | 2 / 4 | 0.571429–1.428571 | 1.414214 | 0.285714–0.707107 |
| 17 | 5 / 10 | 0.316258–7.853982 | 0.038816–1.036561 | 0.019408–0.518280 |
| 19 | 7 / 14 | 1.256637–2.827433 | 0.446447 | 0.223223 |

The smallest half-width is `0.015358726313081661` mm: approximately 614,349 times the permitted corridor. The largest is `1.4649487128999965` mm, approximately 58.6 million times that budget. A tolerance adjustment would change the mechanism from a CAD-tolerance straight-locus proof to macroscopic boundary approximation; it is not justified.

Native face-pair trace:

- 04: 2/3, 5/6, 55/56.
- 11: 6/56, 54/55.
- 12: 13/14.
- 13: 1/86, 4/5, 38/63, 54/57.
- 16: 1/2, 4/174.
- 17: 3/21, 5/22, 23/106, 51/80, 53/59.
- 19: 1/240, 8/9, 42/44, 81/179, 82/178, 86/369, 206/207.

## What the records do and do not prove

The coefficient-width evidence proves that the current **whole coefficient set** cannot satisfy either axis corridor, independently of its monotonicity. This is a mathematically necessary refusal of the current certificate, not proof that a bend is absent.

The source calls `certify_axis_corridor` before verifying the complete clamped trim interval and knot continuity. On the failing pcurve, `basis_domain_proven` and `complete_trim_interval` are therefore absent. The records cannot distinguish a genuinely curved boundary from an unnecessarily broad coefficient enclosure of a smaller actual trim span. Positive convex-hull coefficients provide an upper enclosure, not a lower bound on actual curve excursion.

Likewise, no pole ordering or derivative-sign receipt is published for these failures. The combined reason must not be rewritten as confirmed backtracking. Previously visited analytic lines can have proved intervals, but that does not prove the failing spline interval. `pcurve_types` is a visited prefix because of early return, not an inventory of all boundary representations. One wire alone does not establish a simple non-self-intersecting material loop.

Consequently the defensible classification is: 48 coefficient-corridor refusals; zero independently established backtracking diagnoses; zero independently established nonrectangular-material diagnoses; failing-spline full-trim proof unresolved for all 48. These latter zeroes express missing proof, not evidence of absence.

## Exact next mechanism

Implement a separately versioned **trimmed boundary-domain certificate**, not another epsilon change:

1. Capture every native edge use and its actual pcurve parameter interval, degree, knot multiplicities, periodicity, weights, and oriented wire incidence before classification. Publish separate interval, continuity, corridor, monotonicity, and topology outcomes; retain all refused edges rather than only the first.
2. Restrict polynomial/rational splines to the actual trim interval using geometry-preserving knot insertion and Bezier-span decomposition. Prove interval containment and positive denominator on each span. This is exact representation conversion, not sampling or clipping the material boundary. Bound span counts and refuse unresolved periodic/seam cases.
3. Reapply the unchanged physical tolerance budget to the **restricted** coefficients. A proved straight rectangular boundary can then use the existing side/join/adaptor additive bound. Publish original-versus-restricted coefficient ranges so a representation gain is distinguishable from a tolerance relaxation.
4. If restricted coefficients still fail the rectangle corridor, preserve that negative result. Generalize finite cylinder material support explicitly: certify the complete oriented UV trim loop, seam lift, no self intersections, and any hole loops; establish the actual domain rather than its box. Use bounded polynomial/rational intersection and derivative-root isolation or interval subdivision with certified bounds, not sampled polygons as proof.
5. Pair skins using their transported common angular frame and corresponding physical material domains. Curved axial end boundaries require domain correspondence and finite supported tangent-rail chains; a midpoint line is metadata, not a fabricated full-length rectangular material strip. Extraction and unfolding must have distinct capability gates: a finite bend can be recognized while general trimmed-strip unfolding remains unsupported.

The initial bounded successor should implement steps 1–3 and diagnostics only. Step 4 is a materially new domain representation and needs independent mathematical review and synthetic adversarial qualification before a newly frozen corpus evaluation. Case IDs and observed coefficient magnitudes must not become inference features or threshold inputs.

Independent synthetic tests should include identical trimmed loci represented by whole splines with remote unused poles; exact knot-insertion invariance; curved trimmed spans whose control bounds tighten but still exceed budget; positive rational weights; interior repeated knots; reversal/backtracking; seam-crossing loops; holes; intersecting/retraced wires; and common-frame domain correspondence under SO(3), translation, and reversed axes. No test may promote a sampled fit into a certificate.

## Accuracy reference gate and checkpoint

The 24 historical pair groups are coverage probes, not gold labels. A greater-than-90% precision/recall claim requires a frozen independent per-bend technical CAD reference, authored without viewing extractor results: native solid/face identities, finite material domain and exclusions, carrier, radii, tangent flanges, angle convention, operation-versus-feature identity, and explicit unknowns. Independent review should adjudicate disagreements before the extractor is evaluated. Match complete features one-to-one under predeclared geometric/domain tolerances; unmatched predictions are false positives only where reference coverage is complete, unmatched known references false negatives, and unknown/uncovered regions remain separately reported. Report micro and per-part/family precision, recall, count error, availability, and abstention coverage. Consumed mapped-corpus results are engineering evidence, not unseen-family generalization.

Standing user authority covers necessary CAD-only source successors and evaluations on the mapped existing corpus under new freezes and independent gates. It does not authorize new datasets, outside human contacts, or human manufacturing approval. Technical CAD references may be authored and reviewed here under a frozen blind procedure; they must not impersonate qualified human/manufacturing truth. Ambiguous physical intent remains unknown.

Checkpoint: frozen V6 unchanged; this report uses only released derived records and source. No native jobs were run. Authorized next action is a source-only trimmed-interval certificate successor with independently authored tests, followed by root-coordinated qualification and a fresh exact freeze/PRE gate. Stop if finite trimmed-domain proof cannot be established within declared budgets; retain the refusal.

Source successor checkpoint: V7 now implements copy-only `Segment(first,last,0.0)` restriction for nonperiodic B-splines, strictly refuses intervals outside either the original actual parameter domain or underlying knot domain, and demands exact returned parameter/knot boundaries plus clamped endpoint multiplicities. Its receipt retains both original domains and underlying coefficient ranges beside the restricted proof. Thirteen own pure tests passed; source compilation passed. Four authored native fixtures are ready but were not executed here, including a two-span curved carrier with a straight actual edge interval and explicit V6-refusal/V7-proof expectations on the same synthetic face. Independent mathematical review and root's bounded native qualification remain required before freeze; this source result does not establish any corpus recovery.
