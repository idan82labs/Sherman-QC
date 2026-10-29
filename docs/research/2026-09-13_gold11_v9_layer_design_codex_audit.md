# V9-B material-side layer selection: independent design and sign audit

Date: 2026-09-13. Reviewer: Codex/Astra `v8_flange_policy_math_audit`.
Scope: read-only source/design audit of
`docs/research/2026-09-13_fable51_v9_diagnosis.md` and the connected frozen
kernel/native semantics. No imports, tests, geometry jobs, physical input,
Gold, attempt artifacts or caches were accessed. Only this new report was
written; the original flange audit remains unchanged. Fable retains sole
implementation ownership.

The diagnosis was read fully. At inspection its SHA256 was
`db9e154bcfff05af4679d7c2a458f182cf50529604b19843383a6b2f4822aa1a`.
The existing measurement kernel SHA256 was
`d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb`;
native V1 SHA256 was
`3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b`.
The initial exact-name source inventory found no `cad_guided_measurement_v2.py`,
`cad_guided_context_measurement_v2.py`, or corresponding V2/layer test file at
that check. A V2 kernel appeared during this review and was then read fully;
the exact-hash source addendum at the end updates the design findings. These
are not physical POST findings.

## Decision

**Reasonable research hypothesis, not greenlit for promotion as written.**
Separating scan-observed layers may legitimately close a single-inner-face
kernel/input mismatch. The proposed sign conventions, singleton fallback and
arc exclusion require correction or explicit restrictions first. The existing
derived aggregate supports investigating mixed surfaces; it does not prove
that both material skins enter every failed flange window, eliminate narrow
support as a cause, or attribute global residuals to specific physics.

Do not convert this into another broad readiness phase. The immediate bounded
deliverable should be explicit signed geometry, decisive synthetic
counterexamples, per-window layer diagnostics and an abstention contract for
unidentifiable material side. No score or nominal measurement should change
to compensate for ambiguous input.

## Priority findings

### P1 — The proposed signed rules are reversed if applied to kernel normals

The kernel fits each flange with reference `-radial`
(`backend/cad_guided_measurement.py:307`). The fitted normal is oriented to
that reference (`:126`). Its own comment identifies these as **inner outward
normals** (`:345`). Consequently an outer flange displaced by sheet thickness
t along the positive radius has signed distance **-t**, not +t, under the
fitted normal.

The V9 diagnosis at `:160` proposes keeping the smaller signed-distance layer
and says outer skin is +t along the “inner normal.” Under the actual fitted
normal this chooses the outer layer. The rule is consistent only if the signed
projection is explicitly defined using the opposite **positive radial into
material**, not the fitted outward normal. Do not use the same generic
`normal` name for these opposite directions.

Likewise, at `:166` it proposes retaining the non-material side of both fitted
planes using signed distance <= plane exclusion. Under the actual outward
normals, inner-arc interior lies on the **positive** side. With the old
absolute-distance exclusion preserved, the correct test is signed distance
> plane exclusion. Using <= while retaining the old absolute test selects the
opposite side and excludes the intended inner arc. The exact derivation is
below. Both positive and negative sweeps must be tested; changing sweep sign
does not reverse the material-side convention.

### P1 — A singleton observed layer does not identify the inner material skin

`SINGLE_LAYER_ASSUMED_INNER` (`v9_diagnosis.md:165`) is not sufficient evidence
for an eligible inner-surface carrier on an input whose acquisition side is
unverified. An only-outer scan can give two clean tangent flanges and a clean
circular arc; all unchanged residual, conditioning, tangency and support gates
can pass while the reported “inner radius” is actually the outer radius.

There is a basic observational ambiguity. A sole circular surface at radius
rho with its two tangent planes may be the outer skin of a sheet with inner
radius rho-t, or a genuine inner surface with radius rho. Geometry from that
one surface alone cannot identify the alternative. Choosing whichever matches
nominal CAD more closely would use the nominal answer as an identity prior
and can erase legitimate deviations. An outer-only observation with radius
R+t must not become a false positive inner measurement merely because the
single-layer path never found a gap.

Without a verified single-inner acquisition declaration or a separately
qualified local side-identification mechanism, retain material side as
ambiguous and withhold the inner-specific carrier/conformance/count. Optional
auxiliary unlabeled surface point estimates must remain ineligible for that
inner-specific score; they are not a substituted observed bend.

### P1 — Signed tangent halfspaces do not generally isolate the inner arc

The quoted condition `phi < 2 acos(R/(R+t))` at `v9_diagnosis.md:168` is an
idealized sufficient zero-clearance condition, not a generally large safe
angle. At R=0.8 and t=1.5 it is about 139 degrees; it changes strongly with
R/t. For R=5, t=1.5 and a 120-degree bend, the midpoint of the outer arc is
1.75 mm on the non-material side of **both inner tangent planes**. It survives
the correctly signed 0.075 mm exclusion and the nominal R+3 mm radial cap.
Thus correct flange separation alone does not make the remaining arc inner.

For sufficiently large sweeps, a sampled only-outer central arc may satisfy
the nominal angular guard, the free-circle fit and a wrong outer-flange pair.
For mixed arcs, it may simply expose a new refusal; neither outcome is proof
of valid material-side selection. A larger-sweep “flag” must have an explicit
eligibility consequence or a separately validated scan-derived isolation
mechanism. A warning while emitting an inner carrier is not a safeguard.

### P2 — The causal conclusions exceed the released evidence

The full 10x12 mm synthetic strip demonstrates that one broad synthetic patch
has sufficient spread. It does not establish that the actual selected raw
flange subsets cover both dimensions at those widths. The earlier native
adapter audit showed that certified axial spans can be far shorter and actual
coverage can be clustered. Therefore “H1 rejected for full windows” at
`v9_diagnosis.md:125` is only true of the constructed fixture, not of the
39 physical weak-gate failures.

The scan-plane pair summary at `:134` uses parallelism, centroid proximity
and offset gap. It supports a multiple-close-parallel-surface hypothesis.
It does not establish exact footprint overlap, material ownership, pair-to-
window membership, or both skins in **every** flange crop. A large pair count
does not even establish that nearly every patch has a partner: 105,805 pairs
could fit entirely within a subset of 461 patches, since 461*460/2=106,030,
while the table lists 6,686 patches. This is a combinatorial counterexample,
not a claim about the actual degree distribution. A per-patch/per-window
coverage table is required to support the stronger statement.

Qualifying plane proposals may overlap in raw rows; pair counts are not
counts of independent material observations. Small local RMS on roughly
106-row proposal patches does not bound the full flange's all-row maximum
residual, and a selected plane-proposal population is not calibrated scan
noise truth. These facts also do not establish that whole-part p95 is
“dominated” by deviation and both skins (`:151`). If both CAD skins and both
scan skins coincide, both can have zero nearest-CAD distance. Attribution
requires a decomposition, not merely the presence of two surfaces.

### P2 — Largest-gap layer selection is not pose-independent material identity

A common translation shifts every signed distance equally, so it preserves
rank for a fixed set of rows. A rotation error can broaden each plane's
nominal-normal projection, and cropping changes the row population. Two
parallel surfaces need not have disjoint projected intervals when their
finite sampled footprints differ. Largest-gap splitting can then cut one
surface into two groups or combine several surfaces in one “layer.” Twelve
rows per side is only a population floor, not two-dimensional support,
parallelism, identity, or independence.

The claim “independent of pose error” (`:163`) needs the narrower conditions
derived below. A global 1 mm proximity p95 is not a local pose-error bound and
does not establish any of those conditions. Multiple gaps, sparse minority
layers, unrelated neighboring planes, local curvature, and changing sample
density must be retained as ambiguity/refusal cases rather than resolved by
a favorable rank convention.

## Signed geometry, with both sweep signs

Let a be the declared unit axis and r0 the start radial, with a dot r0=0.
Set k0=a cross r0, s=sign(phi), beta=abs(phi), and

    r(theta) = cos(theta) r0 + s sin(theta) k0,  0 <= theta <= beta,
    r1 = r(beta).

This is the kernel's radial construction and oriented angular coordinate
(`cad_guided_measurement.py:285`, `:319`). Its two outgoing flange rays are
`-s k0` and `s (a cross r1)` (`:291`). In the idealized circular-extrusion model,
inner contact planes pass through C+R r_i and their material-outward normals
are N_i=-r_i. Native extraction separately checks that same signed tangency
(`backend/cad_guided_native_features_v1.py:590`).

Define two explicitly different signed distances:

    S_i(p) = r_i dot (p - (C+R r_i))  [positive radial / into material],
    D_i(p) = N_i dot (p - (C+R r_i)) = -S_i(p) [inner outward].

For paired ideal flange points p_outer=p_inner+t r_i:

    S_i(p_inner)=0,  S_i(p_outer)=+t;
    D_i(p_inner)=0,  D_i(p_outer)=-t.

So “smallest layer” means inner only for S; it means outer for D. At exact
geometry the axis has S=-R and D=+R, agreeing that “axis side” is negative S
but positive D. An estimate of the axis itself must not be replaced by the
intersection of flange planes: the original kernel fits a circle and checks
its tangency, preserving that separation (`cad_guided_measurement.py:333`).

For a circular arc of radius rho, the signed distances are:

    D_0(theta) = R - rho cos(theta),
    D_1(theta) = R - rho cos(beta-theta).

For the inner arc rho=R these are nonnegative on the complete sector. Hence
retaining the non-material side **and** the unchanged exclusion |D_i|>epsilon
is exactly D_i>epsilon for both i. In positive-radial coordinates the same
rule is S_i<-epsilon.

If an implementation instead combines S_i<=epsilon with the old |S_i|>
epsilon, the conjunction does reduce to S_i<-epsilon. If it replaces the old
absolute test with S_i<=epsilon, it does **not**: it admits the near-plane band
and a small material-side band. This distinction changes support rows and
the derived outer-arc bound. The new source must state which operation it
performs, with strict/equality boundary tests.

The orientation algebra does not change for negative sweeps. N0 cross N1 =
s sin(beta) a. The existing axis sign alignment flips the preliminary axis
when needed, and the signed `atan2` then returns s beta
(`cad_guided_measurement.py:309`). Native cylinder convex/concave identity is
instead determined by oriented surface normal dot radial polarity
(`cad_guided_native_features_v1.py:479`, `:507`), not sweep sign. A negative
sweep is not permission to swap inner/outer layers, and a positive sweep is
not proof of convex material orientation.

## Exact outer-arc exclusion conditions under the declared ideal model

Let epsilon be the positive plane exclusion (0.075 mm in V8), t>0, and assume
exact common center/axis, parallel-offset tangent flanges, circular skins with
radii R and R+t, and exact inner-plane selection. None of these equalities
follows solely from the derived scan-plane table.

With the correct D_i>epsilon tests, an outer-arc sample survives precisely
when:

    cos(theta) < (R-epsilon)/(R+t),
    cos(beta-theta) < (R-epsilon)/(R+t).

For c=(R-epsilon)/(R+t) in [-1,1], write alpha=acos(c). On the modeled
0..beta sector with beta<pi, a surviving interior exists iff beta>2 alpha;
the surviving interval is (alpha, beta-alpha). Equality has no survivor
because the kernel exclusion is strict. If the original angular guard is g,
the intersection with [g,beta-g] is nonempty iff beta>2 max(alpha,g), for
this ideal common-frame case. Actual finite rows may of course miss this
continuous surviving interval; interval existence is not observed support.

At epsilon=0 this becomes the quoted 2 acos(R/(R+t)). For epsilon>0 the
threshold increases, so beta below the zero-epsilon threshold is still a
sufficient exclusion condition in the exact model. The positive-clearance
formula, finite guard, boundary equality and acos-domain cases must all be
defined before using this as a computation policy. Do not silently clip an
invalid ratio into [-1,1] and claim a geometric certificate.

Concrete source-independent counterexample: R=5 mm, t=1.5 mm, beta=120
degrees, theta=60 degrees. Then rho=6.5 mm and both D_i=5-6.5/2=1.75 mm.
The point passes the correctly signed epsilon=0.075 mm test and rho<=R+3.
By contrast, R=0.8 and t=1.5 produces a very different critical angle. One
example at approximately 139 degrees cannot cover arbitrary radii/thicknesses.

For the inner arc itself, D_i>epsilon removes a contact-neighborhood angle
acos(1-epsilon/R) at either end when the expression is in range. If beta is
too short or R too small relative to epsilon, **even the inner arc** has no
remaining eligible sector. That failure must remain an insufficient-support
outcome, not be cured by nominal arc insertion or dropping the exclusion.

For noncoaxial skins, local deformation, uncertain fitted planes, unequal
thickness, or pose-dependent nominal angular clipping, these ideal formulas
are not certification bounds. To obtain a bound, explicitly propagate the
allowed normal/offset/model errors or abstain. Merely inserting nominal R and
nominal thickness into a sufficient formula while treating them as measured
truth would violate the intended measurement separation. If the formula is
used only as a nominal search heuristic, label it as such and independently
validate the selected scan geometry/material association.

## Projection ordering under pose error

Let v be a true positive radial normal of two parallel layers separated by t,
and let u be the nominal radial used for sorting. For a matched location q:

    u dot ((q+t v)-q) = t (u dot v) = t cos(gamma).

Thus corresponding outer points sort after inner points if cos(gamma)>0.
A common translation does not change this difference. But sorted sample
populations consist of **different** q values, not paired coincident footprints.
If a common footprint spans W in the relevant tangential tilt direction,
its u-projection can have width W |sin(gamma)|. With independent bounded
signed error epsilon_s on each layer, a sufficient separation margin is:

    gap >= t cos(gamma) - W |sin(gamma)| - 2 epsilon_s.

For unequal/nonoverlapping footprints one must bound their full relative
tangential range instead; the same W cannot be assumed. A gap floor can be
justified only under a stated bound of this sort or a scan-derived segmentation
that explicitly models plane slope. With t=1.5 mm, W=12 mm and gamma=15
degrees, the tangential term alone exceeds the positive layer separation;
overlapping projected layers are entirely possible even with exact parallel
planes. This example uses an angle permitted by the current orientation
scope, not a claim that the physical poses have that error.

This is why sorting nominal signed distances is not independent of pose error.
It also explains a false split: one tilted plane sampled in two separated
tangential bands can have a large gap in nominal signed distance, while both
groups lie on the same measured plane. Plane compatibility and relative
offset in a **scan-derived** common normal are needed to distinguish this
from actual layered material.

## Required ambiguity and abstention contract

For a versioned development candidate, spell out these outcomes before any
test/rerun:

- **No qualified local layers:** unavailable, retaining all attempted rows and
  gate reasons. No nominal fallback.
- **One qualified layer with unverified acquisition side:** material side
  unknown; do not label it inner solely because it is the only one observed.
- **Exactly two separated, scan-fitted compatible layers:** potential layer
  pair, not automatic physical skin identity. Retain the support IDs, actual
  footprint relation, measured normals/offsets, ordering direction, gap and
  rejected alternatives. Require the separately declared correspondence and
  adjacency/model checks before assigning an inner role.
- **Multiple plausible layerings, three or more layers, competing adjacent
  faces, interleaving projections, or too few minority rows:** ambiguous or
  unsupported. A single largest gap must not silently resolve these by order.
- **Inner flanges identified but arc side not isolated:** withhold inner arc
  carrier. A correct plane pair alone does not identify every curved sample.
- **Two distinct inner/outer explanations both fit all available observations:**
  retain both alternatives and abstain from an inner-specific answer. The CAD
  nominal radius or the eventual Gold line must not break that tie.

Changing support selection is a measurement mechanism change even if every
downstream scalar gate is unchanged. It can remove observations carrying real
deviation and increase apparent precision, so all discarded layers/rows and
their reasons must be retained, with full candidate-level denominators. A
numerical layer count is not an observed bend count.

## Minimum independently constructed discrimination matrix

These are proposed tests; this reviewer executed none. Use analytical or
independently generated geometry, not the implementation's layer helper to
produce expected labels. Freeze exact parameters, budgets and failure rules.

1. **Sign identities:** positive and negative 20/90/160-degree sweeps, oblique
   rigid poses, known inner/outer flange points and arc points. Assert S and D
   have opposite signs; check which side is retained before any circle fit.
2. **Clearance boundaries:** outer samples just below, at and above the derived
   2 acos((R-epsilon)/(R+t)) threshold, with the finite angular guard and
   low/high R/t examples. Include the explicit R5/t1.5/120-degree counterexample.
3. **Single-skin ambiguity:** inner-only, outer-only, and missing-inner/noisy-
   outer cases with otherwise perfect tangent circles. Include an actual
   inner-radius deviation equal to nominal thickness. Require unknown side
   unless independent side evidence exists; prohibit nominal-nearest selection.
4. **Sparse/multiple mixtures:** minority layer counts 0, 1, 11, 12 and well
   above 12; 3+ planes; unequal densities; duplicate positions; gaps below,
   equal to and above the chosen threshold. Retain the complete partition
   ledger, not only the selected population.
5. **Tilt and unequal footprints:** perfect two-plane shells whose nominal
   projections overlap, and one tilted plane with an artificial sampling gap.
   Vary translation separately from rotation and cropping. Verify that a
   largest gap alone never becomes material identity.
6. **Neighboring geometry:** a close unrelated parallel flange, an adjacent
   bend, folded return, hole/trim-separated patch, crossing surfaces, curved
   contaminant, and nonuniform thickness. The chosen layer must not acquire
   an unrelated component merely because its signed mean is smaller.
7. **Honest deviations:** coherent radius change, angle change, center shift
   and flange warp. Confirm measured geometry follows retained scan evidence,
   not nominal CAD, and that real deformation is not relabeled an outer layer
   then discarded to improve the result.
8. **End-to-end binding:** selected flange/arc raw IDs through context fitting,
   canonical refit, support cover and inner-specific carrier eligibility.
   Check uncertainty/side ambiguity is not lost before the unchanged scorer.

Add per-side telemetry before early refusal: selected/raw candidate counts,
actual axial and tangential extents, signed projection intervals, every gap,
candidate partitions, plane spectra and fit residuals, and exact failure
booleans. This both distinguishes the physical hypotheses and makes future
negative runs diagnostic. Any new consumed-input diagnostic remains a new
bounded frozen protocol; do not reconstruct favorable measurements into V8.

## Evidence wording and next step

Correct the diagnosis to distinguish “a two-surface mixture reproduces the
same refusal reason under an invented broad-strip model” from “this is the
physical cause of every failed window.” Release a per-window membership and
spectrum ledger if an authorized diagnostic establishes the latter; the
current global pair summary is insufficient. Keep narrow support and local
coverage as active alternatives until the relevant actual spans are known.

The versioned layer mechanism is worth investigating under that corrected
scope. Promotion requires sign-correct source, explicit side-identifiability
and abstention semantics, adversarial discrimination, and connected evidence
binding. It does not require every unsupported case to become positive, but
must not convert an ambiguous outer-only or adjacent-surface fit into a
claimed inner measurement. No historical-score comparison or production
release follows from this design review.

## Addendum: V2 kernel appeared during the review

`backend/cad_guided_measurement_v2.py` was read fully, including a separate
read of the initially truncated middle. Its SHA256 was stable across the
read and recheck:
`59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d`.
This is an early source snapshot, not a frozen candidate or test result.
No new context consumer or corresponding V2 test was present at the inventory
check. Do not apply these line references to an unreviewed later revision.

### Closed at this source snapshot: the immediate sign reversal

The new kernel explicitly distinguishes positive material-side radial from
the fitted outward normal (`:9`). It uses the crop's TLS normal oriented
toward +radial to order flange layers (`:254`), then retains arc rows with
signed distance against the fitted -radial normal **strictly greater than**
the plane exclusion (`:518`). Those signs agree with the derivation above.
The original design prose must still be corrected, but the immediate sign
reversal is **not** an observed defect in this source snapshot.

Using a scan-derived TLS direction also avoids the direct nominal-projection
tilt problem for a well-supported ideal parallel mixture whose least-variance
direction is the true layer normal. It is not a general solution: a narrow
in-plane extent can become the least-variance direction, nonparallel mixtures
can bias the TLS direction, and this preliminary direction is not checked for
normal identifiability or alignment before splitting (`:254`). The after-split
`_plane` gates remain important but do not establish discarded-layer identity.

### Open P1: side ambiguity remains metadata-only in the kernel

A missing split returns all rows and the label
`SINGLE_LAYER_IDENTITY_UNRESOLVED` (`:263`). Two layers are explicitly
unverified (`:269`). `_measure_arc` computes the corresponding material-side
status (`:554`) but does not abstain: it continues the free-circle fit and
returns measured radius/sweep/support (`:567`, `:599`). The public entrypoint
can mark this `MEASURED_CONDITIONAL` (`:690`). Thus the source does not claim
singleton identity in its string, but still produces inner-model geometry with
unresolved identity. A downstream versioned consumer could enforce the
required exclusion, but that consumer was not available for review here.
Until connected eligibility is demonstrated, the singleton/outer-only
counterexample is unresolved. A metadata warning is not evidence of refusal.

### Open P1: “COMPLETE_FOR_NOMINAL_SWEEP” overstates an unverified proxy

At `:525`, the code takes the minimum of any observed flange split-mean gaps;
even one observed gap suffices. It inserts that value and the **nominal**
radius into the ideal formula, clips the acos argument, and publishes
`COMPLETE_FOR_NOMINAL_SWEEP` at `:535` when nominal sweep lies below it.
There is no requirement that both gaps agree, that the split groups are actual
parallel skins, that actual radius equals nominal, or that fitted plane and
model errors have a bound. The status is not used to stop the subsequent fit
when it is unknown/incomplete either (`:538`).

A concrete idealized parameter counterexample to the “complete” label uses
nominal R=4 mm, actual inner R=5 mm, t=1.5 mm, and beta=85 degrees. The nominal
formula gives a threshold of about 88.9 degrees and hence reports complete.
But the true outer midpoint has
D_i=5-6.5*cos(42.5 degrees), approximately 0.208 mm >0.075 mm, and its radius
6.5 mm is below the nominal radial cap 4+3=7 mm. The flange nominal offset
is only 1 mm, within the search margin. So allowed legitimate radius deviation
can invalidate the nominal exclusion proxy even in exact coaxial geometry.
This is an algebraic counterexample, not an executed fixture or physical claim.

At minimum rename this field as an idealized nominal-model heuristic with
explicit unverified assumptions, never a complete observed exclusion
certificate. Actual inner-carrier eligibility still needs side isolation or
abstention; changing the label alone does not supply that evidence.

### Open P2: disabled layer mode does not reproduce V1 crops

`LayerSelectionPolicyV1` says `enabled=False` reproduces V1 crops (`:54`),
and `_measure` defaults to that disabled mode (`:445`). Flange splitting is
indeed bypassed (`:251`), but the new one-sided arc filter is unconditional
(`:518`). V1 used absolute signed distance; V2 disabled mode can therefore
remove material-side rows that V1 retained. Either make an explicit combined
mechanism switch or correct the control-mode identity and freeze that delta.
Do not treat this disabled path as a byte/geometric V1 control without the
corresponding test and accurate declaration.

### Open P2: partition and diagnostics limitations

`_split_layers` tests only the globally largest gap and splits into at most
two groups (`:225`). A failed minimum-row condition at that gap produces a
singleton label even if a smaller qualifying gap exists. Three or more
physical layers are never enumerated; `max_layers_reported` is admitted in
the policy (`:58`) but unused by the splitter. These are declared-algorithm
limitations that require explicit ambiguous/unsupported outcomes, not a claim
that exactly one or two material skins were detected.

The new before/after spectra and both weak-gate booleans are useful diagnostic
progress (`:171`, `:478`). However, `ray_extent_mm` is the extent along the
second principal vector `vh[1]`, not the feature's actual flange ray (`:210`);
rename it or report the real ray extent separately. Otherwise future narrow-
support diagnoses may confuse axial and radial/ray extents. Plane-fit
exceptions carry diagnostics (`:493`), but axis/sweep exceptions before the
arc helper do not (`:500`), and the public exception handler only records the
reason string (`:692`). The connected consumer must deliberately preserve the
new diagnostics across all relevant failure exits.

The source addendum closes the immediate sign concern, not the material-side
identifiability or promotion concerns. No code was edited and no test was run.
