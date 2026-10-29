# CAD extraction coverage: next mathematical gates

2026-09-30. Derived-only review of released `RESULT`, `EXTRACTION`, `REJECTIONS`
and `GEOMETRY_DIAGNOSTICS` in the explicit-frame V3 attempt2, compared with
baseline V1 extraction records. No raw CAD, private `source.step`, scans, truth
or native jobs were read/run. Frozen sources and terminal attempts are unchanged.

## Outcome

The corrected explicit-frame extractor emits 97 candidates versus baseline 127.
The 30-group difference is not measured recall loss: neither candidate set is
independent truth. Nonetheless it is a concrete coverage regression requiring
explanation, not an automatic precision improvement.

Matching corresponding native face sets accounts for every difference:
28 baseline pair groups lose at least one skin at the rectangular-domain gate;
two more pass that gate but lack both required tangent flange rails. All lost
cylindrical faces have one recorded valid solid owner `[1]`, so the generic
`uncertified_trim_domain_or_solid_owner` refusal is a trim-domain failure here,
not an ownership failure. These comparisons use one-based extraction face IDs;
geometry diagnostics use zero-based native face IDs (subtract one).

| Case | Baseline → corrected | Lost native pairs | Immediate source gate |
| --- | --- | --- | --- |
| 04 | 3 → 0 | 2/3, 5/6, 55/56 | All trim-uncertified; 55/56 have 5 edge uses |
| 10 | 3 → 1 | 1/15; 26/27 | Rolled pair trim-uncertified; 27° pair has only 1 rail per skin |
| 11 | 3 → 1 | 6/56, 54/55 | All trim-uncertified despite 4 edge uses and 2 rail witnesses |
| 12 | 4 → 3 | 13/14 | Trim-uncertified; 5 edge uses per skin |
| 13 | 5 → 0 | 1/86, 4/5, 38/63, 54/57, 82/83 | Nine skins trim-uncertified; 82 is certified but loses partner 83 |
| 14 | 3 → 1 | 1/25; 21/22 | Rolled pair trim-uncertified; 27° pair has only 1 rail per skin |
| 15 | 1 → 0 | 1/23 | 178° rolled pair, 31 edge uses per skin, zero rail witnesses |
| 16 | 11 → 9 | 1/2, 4/174 | Trim-uncertified; 5 edge uses per skin |
| 17 | 5 → 0 | 3/21, 5/22, 23/106, 51/80, 53/59 | All trim-uncertified; 4 or 6 edge uses |
| 19 | 14 → 7 | 1/240, 8/9, 42/44, 81/179, 82/178, 86/369, 206/207 | All trim-uncertified despite 4 edge uses |

## Necessary refusal versus implementation restriction

`_rectangular_domain` in `scripts/extract_step_topology_bend_axes_v4.py` requires
one wire, exactly four native edge uses, each pcurve reported as an analytic line,
axis-aligned endpoints at the UV bounding corners after eight-decimal rounding,
and four distinct rectangle sides. That is a sufficient restricted certificate,
not the mathematical definition of a bend domain. Five/six-edge boundaries can
be a rectangle with split edges or a genuinely nonrectangular cut; a spline
pcurve can describe an exact straight boundary. Neither should be guessed from
the face bounding box. Four-edge failures can arise from pcurve representation,
oblique/curved trims or numerical endpoint/corner mismatch.

The released inspector records edge incidence but not wire counts, pcurve types,
coefficients or rectangle-subgate residuals. Thus exact subgate attribution for
four-edge refusals is **unknown**; claiming all are numerical rounding errors,
all are holes, or all are harmless split edges would exceed the evidence.

Twenty-five of the 28 trim-rejected baseline groups are ordinary partial-cylinder
carriers with two analytic tangent-edge witnesses per skin. They are the priority
for a representation-independent finite-domain certificate. Rail count alone is
not proof of full paired axial coverage, constant material thickness or physical
operation identity. Refusal is currently scientifically necessary because those
domains are not certified; requiring a four-edge rectangle forever is overly
restrictive engineering, not a universal geometric law.

The other three trim groups are distinct rolled/high-turn cases. Case 10 pair 1/15
has sweeps 154.246086409/154.236033636 degrees, radii 38.5/37 and 13 edge uses per skin.
Case 14 pair 1/25 has 154.075834451/154.060682850 degrees, radii 31.5/30 and 13 edge uses.
Both have zero accepted plane-flange rails. Unequal face bounds are genuine
metadata/domain differences, not merely a two-versus-one sweep schema problem.
The separate27° pairs (10:26/27,14:21/22) each have one plane-flange rail and an
adjacent cylinder; a bend-to-bend or rolled-transition model may be needed.
Dropping the two-rail rule would permit holes/bosses and is not a defensible fix.

Case 15 pair 1/23 is 178 degrees (radii 75/74), with 31 edge uses and no accepted rails.
It also exceeds the current downstream 175 degree capability. Fixing trims alone
cannot make that selected-skin path supported. Do not widen the cap or invent
supplementary-angle semantics from this case; freeze a separate high-turn model
and independently authored counterexamples first.

## Exact next source mechanism

1. Replace the Boolean rectangle diagnostic with a typed domain certificate and
   subgate ledger: native wire/edge IDs, outer/inner orientation, pcurve carrier
   type, interval/endpoint residuals, source tolerances, analytic frame and status.
   Separate owner failure from trim failure. This supplies causal evidence without
   promoting an uncertified face.
2. Certify straight pcurve loci independently of representation (positive rational
   spline coefficients on a common line), merge only native-connected collinear
   boundary segments, and prove simple closed rectangular loops. Preserve holes
   and branching as explicit unsupported domains. This is the smallest extension
   for split/representation-only rectangles, not blanket `edge_count>=4` acceptance.
3. Add actual intrinsic cylinder trim domains: preserve all original wire loops,
   unwrap periodic coordinates in the explicit common angular frame, certify
   boundary arcs and material winding, and compute angular/axial material cells.
   Pair inner/outer material support through native solid provenance; retain hole
   exclusions and multiple axial components. Do not substitute UV bounding boxes.
4. Generalize rails as native-connected tangent **chains** with independently
   certified geometric carrier and full finite support. Keep two ordinary flange
   boundaries for the ordinary-bend classifier. Create a separately typed curved
   transition/rolled-feature classifier when a boundary adjoins another cylinder;
   geometry-feature identity must remain separate from manufacturing operations.
5. Keep unsupported cylinder split ownership and high-turn development explicit
   until stitch/native oracle consumers can validate their complete finite domains.
   Extraction coverage and unfolding availability are different evaluations.

Independent authored tests should include collinear split pcurves, exact straight
rational spline pcurves, oblique/scalloped cylinder ends, holes crossing a bend,
multiple wires, same-axis disconnected sectors, folded/boss/hole counterexamples,
adjacent bends, unequal inner/outer trims, 178 degree turns and representation/rigid
transform/seam metamorphisms. Each has formula-defined material and feature truth.

## Independent 90 percent reference protocol

Before candidate outputs are shown, obtain an independently authorized CAD-only
reference cohort and bind exact bytes, verified units, part/family status and
annotation convention. Existing 20 parts are consumed development; no unseen-family
claim may be inferred from a successful evaluation on them.

Two qualified annotators work independently using authoritative CAD feature
history where available and manual native-CAD geometry review otherwise. They
must not see extractor outputs, candidate counts or generated bend markers.
Record every reference bend's identity, finite support/holes, curve axis, inner
and outer radius meaning, signed-turn/sweep convention, adjoining material and
operation-versus-geometric-feature interpretation. Holes, fillets, bosses, rolled
transitions and ambiguous operation identities require explicit dispositions.
Adjudicate disagreements and freeze the complete reference before predictions.
Unresolved truth is reported as reference coverage, not silently removed to make
scores pass. The standing scope permits independent technical CAD references to
be authored and reviewed here for the mapped existing corpus under a frozen blind
procedure and independent gates. Such engineering references must not impersonate
qualified human review or manufacturing approval; ambiguous geometry remains
unknown. New datasets, outside human contacts and human manufacturing approval
remain outside this scope. Engineering reference geometry is not manufacturing
truth or physical metrology.

Freeze one-to-one geometric matching tolerances and matching code independently:
world axis/finite support, angular domain/turn, radius and units. One prediction
matches at most one truth bend; duplicates count as false positives, merged
features leave unmatched truth bends as false negatives, and refused cases retain
their reference bends in end-to-end recall. Count equality cannot cancel a false
positive against a false negative. Report micro precision/recall/F1, part-macro
scores, exact-count rate, family strata, zero-output/refusal rate and reference
coverage. Define empty-denominator conventions in advance.

Acceptance should require precision **and** recall above 0.90 under a frozen lower
confidence-bound rule, not merely F1 or candidate capture. Resample independent
part/family units, not correlated bends, and report finite-corpus point estimates
separately from generalization claims. Small samples can fail the confidence gate
even with point estimates above 90 percent. Manufacturing and scan accuracy remain
separate truth/evaluation problems.

## Checkpoint

Only this diagnosis report was authored. No source change, physical access,
native run or terminal rerun occurred. The user's standing scope already permits
necessary CAD-only source successors and evaluations on the mapped existing
corpus. Next: implement and source-review the boundary-certificate successor and
freeze a blind technical reference procedure; any new physical evaluation still
requires its own freeze and independent gates. No additional user authorization
is needed for ordinary in-scope source-only work. New datasets, outside human
contacts and human manufacturing approval remain outside scope. Never reopen
the consumed baseline or corrected explicit-frame attempts.
