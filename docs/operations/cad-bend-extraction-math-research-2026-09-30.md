# CAD bend extraction: representation-independent finite features

2026-09-30. Source-only mathematical research. No CAD, scan, truth, manifests,
terminal attempt artifacts or native jobs were accessed. This report proposes
engineering work; it is not measured extraction accuracy or authorization to run.

## Finding and proposed successor

The next mechanism should separate three questions: is a surface geometrically
cylindrical; what finite material domain belongs to it; and does that domain form
one sheet-metal bend? A STEP surface type answers only the first, incompletely.
Recognize cylinders independently of representation, recover actual trimmed
domains, then construct inner/outer pairs within a topology-aware feature graph.
Do not classify every developable or ruled spline as a circular bend.

Source inspected: `scripts/extract_step_topology_bend_axes_v2.py` and
`tests/test_extract_step_topology_bend_axes_v2.py`, plus repository instructions,
current status, repository map and Astra workflow. Existing V2 accepts only
`GeomAbs_Cylinder`, eligibility uses rectangular adaptor parameter bounds, and
grouping tests axis angle/offset and projected axial overlap against the first
group row. Acceptance then needs at least two radii in a fixed separation range.
There is no angular-domain overlap, solid ownership, adjacency or material-side
test. Consequently these are constructible failure modes, not corpus findings:

- Exact circular cylinders represented by rational splines are omitted.
- Distinct bends in opposite angular sectors on the same axis can merge.
- Split hole-wall faces can each pass the partial-cylinder sweep cutoff.
- A face's bounding UV rectangle can bridge missing material or holes.
- First-row grouping is order-dependent for near-threshold nontransitive pairs.
- Longest-face extent and unweighted mean of face midpoints need not represent
  paired material extent; splitting one skin changes the mean.
- The `NEUTRAL_RADIUS_HALF_ANGULAR_TURN` label is not a demonstrated neutral axis:
  averaging inner/outer radii is a mid-thickness geometric marker, not K-factor
  physics, and unequal face splits additionally bias it.

The existing tests are useful pure-row boundary and coordinate checks; they do
not establish these feature-recognition properties or corpus precision/recall.

## Representation-independent geometry

For points p and unit surface normals n, let a be a unit candidate cylinder axis,
P = I - aaᵀ, and choose axis origin c with a·c = 0. A cylinder obeys

    n·a = 0,
    ||P(p-c)|| = r,
    n parallel to P(p-c),
    p(θ,z) = c + za + r(cos θ e1 + sin θ e2).

Propose a from the minimum-eigenvalue eigenvector of Σ w nnᵀ. Estimate c,r by
radial residual minimization; use normals as additional constraints rather than
assuming curvature alone is sufficient. Narrow angular sectors are poorly
conditioned: require a published conditioning/uncertainty bound and abstain when
it fails. Cone, extrusion of a noncircular curve and arbitrary developable sheet
all have special low curvature but are not constant-radius cylinders.

OCCT supplies `GeomConvert_SurfToAnaSurf`, including target surface selection,
bounded conversion and reported `Gap`; its documentation describes approximate
analytic conversion of spline/Bezier surfaces. Use it as a candidate proposer,
retaining original native geometry, not as independent proof or topology repair.
[Official conversion reference](https://dev.opencascade.org/doc/occt-7.8.0/refman/html/classGeomConvert__SurfToAnaSurf.html).

Validation must use disjoint points/cells from proposal fitting and include every
trim boundary and interior knot span: radial error, normal-axis orthogonality,
radial-normal agreement, common generator direction, and radius constancy.
Adaptive sampling with derivative-based cell bounds can provide a certified
maximum; sampling alone must be labelled sampled agreement, not a continuous
proof. Preserve tolerance and conditioning receipts. Exact rational-cylinder
fixtures and near-cylinder perturbations should distinguish exact recognition,
tolerance-qualified approximation and refusal. OCCT explicitly supports analytic
surface-to-B-spline conversion, making representation metamorphism constructible.
[Official GeomConvert reference](https://occt3d.com/dev/doc/refman/html/class_geom_convert.html).

## Finite material and feature identity

Map original trimmed material to intrinsic cylinder coordinates (θ,z), unwrap θ
continuously per patch, and carry original outer/inner wire loops. Do not replace
the domain by min/max UV bounds. UV values on a spline need not be radians or
physical axial millimetres. Classify source UV points in the original face, then
map their world points to cylinder coordinates; periodic seams need explicit
unwrapping. [Official face-classifier reference](https://occt3d.com/dev/doc/refman/html/class_b_rep_class___face_classifier.html).

Use same-solid identity and edge-to-face incidence, merging representation splits
only when geometry, trim connectivity and material-side continuity agree. OCCT
provides unique-ancestor incidence mapping; preserve orientation and location.
[Official topology reference](https://occt3d.com/dev/doc/refman/html/class_top_exp.html).

Inner/outer pairing should require compatible angular material support, positive
paired axial extent, consistent radius difference, and an enclosing solid's
material between the two skins. Tangent plate-boundary chains distinguish forming
bends from rounded thickness edges, holes, bosses and isolated fillets. A complete
2π wall assembled from partial faces is still a hole wall; check the merged
angular domain before a sweep eligibility rule. Opposite-sector same-axis bends
remain distinct. Holes intersecting a bend create exclusions in the bend domain,
not new bends. Separate islands separated by genuine material gaps require a
frozen feature-identity convention; do not let transitive tolerance chains decide.

Emit support-domain intervals/components, native face/edge provenance, pairing
residuals and typed ambiguity. A visualization may render several finite marker
segments for one operation-level bend ID. Manufacturing operation identity is
not always uniquely inferable from final geometry; ambiguous cases need truth
policy or abstention, never a fabricated unique count.

## Angle convention

For consistently oriented plate tangent vectors t0,t1 perpendicular to canonical
axis a, signed turn is atan2(a·(t0×t1), t0·t1). Axis reversal changes the sign,
not geometry; reversing face winding must not change the physical magnitude.
Publish geometric sweep separately from flange interior angle and signed forming
turn. For an ordinary single convex bend under the chosen convention, the
interior flange angle is π minus unsigned turn; reflex/hem cases need explicit
branch and material-side handling. A normal dot product alone loses the reflex
branch. CAD sweep/radius are nominal geometry, not scan measurements.

## Independently constructible acceptance fixtures

Use authored geometric formulas and explicit feature IDs as truth, not extractor
outputs. Before native synthetic execution, inspect fixture imports and use an
isolated bounded worker; no real corpus input is necessary.

1. Identical cylinder as analytic, exact rational spline, transformed, reversed
   axis, reversed winding, periodic seam shifted, and split into arbitrary faces.
   Count, domain and world marker must be invariant within declared tolerance.
2. Two coaxial bends at disjoint angular sectors; adjacent but distinct bends;
   near-threshold axis chains in all face permutations. No false merge/order bias.
3. Full hole split into 2/4/8 arcs; counterbore/boss; thickness edge fillet. Zero
   forming bends unless an independently authored actual bend is present.
4. Bend with circular/oblique holes and scalloped trims, split inner skin and
   unsplit outer skin, unequal axial extents, disconnected islands. Recover the
   authored support domain; no material bridge across holes/gaps.
5. Cone, torus patch, elliptic extrusion, radius-varying spline, local bump smaller
   than sparse sampling spacing. Reject or honestly report approximation/unknown.
6. Acute, obtuse, flat-near-limit, reflex and hem turns; gauge-unit transformations.
   No degree/radian, supplementary-angle or assumed-mm conflation.

## Honest >90% extraction evaluation

Freeze independent human/native-authoring truth before viewing candidate outputs,
including bend identity, support extent, angle convention, radius meaning and
ambiguous/ineligible dispositions. Filenames, incumbent counts, candidate pairs
and candidate unfolding success cannot supply truth. Independently review CAD
geometry or obtain authoritative feature history; disagreements are adjudicated
before scoring. Where geometry cannot resolve operation identity, publish that
coverage limitation instead of silently removing difficult parts.

Freeze a one-to-one bipartite matcher based on finite support, world-axis position,
angle and radius with unit-aware tolerances. One prediction matches at most one
truth bend; duplicate predictions are false positives and merged predictions
leave unmatched truth bends as false negatives. Report

    precision = TP/(TP+FP), recall = TP/(TP+FN),
    F1 = 2TP/(2TP+FP+FN), exact_count_rate = parts with Npred=Ntruth / all parts.

Count equality alone is insufficient: one missed and one false bend can cancel.
Refused/failed parts stay in the denominator; their truth bends are unmatched for
end-to-end recall, while extraction-only successful-case metrics are secondary.
Report micro and part-macro metrics, exact count, catastrophic error distribution,
failure/unknown coverage and family-stratified scores. Zero truth/prediction
denominators need a frozen convention, not automatic 100% precision.

Define >90% before execution: recommend precision and recall each above 0.90,
with a stated lower confidence bound and all-part exact-count rate separately.
Resample independent part/family units for intervals, not correlated bend rows.
If the entire authorized existing corpus is scored, this describes that finite
corpus only; consumed cases are development evidence, not unseen-family accuracy.
After extraction passes, freeze a distinct unfolding evaluator (native material,
metric/area, seam closure, overlap, holes and orientation). Manufacturing blank
accuracy additionally requires verified units, thickness and neutral-axis model.

## Checkpoint

Only this report was authored. No runtime change and no tests/native jobs were
executed. Next authorized engineering action: a separately versioned synthetic
prototype with independent feature/domain truth and a source-only audit. Physical
inventory, freeze, PRE and the new corpus execution remain with the directing
agent; previous terminal attempts remain untouched.

### Source-only implementation follow-up

The directing agent subsequently authorized separately versioned
`scripts/extract_step_topology_bend_axes_v3.py` and its pure-row tests. V2 remains
unchanged. V3 uses solid ownership, world-frame periodic angular support, stable
native face identities, complete-link carrier checks and native shared-edge
connectivity. Full assembled cylinders and ambiguous radius partners refuse.
Both skins require two full-span tangent flange rails through native line edges
adjoining analytic planes. Markers explicitly say geometric mid-thickness.

Native domain acceptance is narrow: one wire, four linear axis-aligned pcurves
with the four bounding corners and distinct rectangle sides. Nonrectangular
trims, holes and varying axial split domains refuse; B-spline cylinder
certification remains unsupported. This is tolerance-qualified rectangle
recognition, not a general continuous trim or material-thickness proof.
Geometric features remain separate from manufacturing operation groups.

Verification: 9 isolated pure-row tests passed, most recent execution 0.40 seconds.
OCP constructor/static-method availability was introspected without constructing
native geometry. No authored-native or physical corpus execution occurred.

Root audit follow-up: fixed an in-place native-direction canonicalization bug by
copying before canonical ordering; a negative-axis signed-parameter regression
checks independently authored world endpoints. Replaced V2 loose coaxial gates
with explicit 1e-7 mm carrier-position and 1e-9 direction-cross bounds, publishing
max offset/angle residuals and raw native face indices. Native admission requires
one valid solid and bounded topology traversals before global inventories (10,000
faces / 50,000 edges). Pure-row validation refuses invalid/nonfinite geometry and
duplicate native identities. Latest isolated check: 19 passed in 0.41 seconds.
An initial new regression assertion used unsupported nested-list `pytest.approx`;
correcting its expected value to an array resolved that test-harness error.
Native parse/negative-axis authored fixtures still require scheduled verification.

Authored-native fixture source is now
`tests/test_extract_step_topology_bend_axes_v3_native.py`: formula-defined L bend
(inner radius 2, outer 3, 90 degrees, two 10 mm tangent flanges, 40 mm extrusion),
reversed inner-circle native axis, proper transformed/reversed world axis, and a
cylindrical tube counterexample. Truth marker radius is independently 2.5.
Its explicit environment gate prevents accidental native execution. Only syntax
was checked; root owns serial 60-second/4-GiB supervised execution scheduling.

Independent source audit of `backend/vendor_native_folded_mesh_v1.py` identified
non-FORWARD/REVERSED winding acceptance and nonpositive circle-sweep preflight;
root corrected both. Re-audit is green for bounded display-only, not continuous
material metrology or an unfolding proof. Caller time/RSS supervision remains
essential because post-mesh triangle caps cannot prevent native allocation.
