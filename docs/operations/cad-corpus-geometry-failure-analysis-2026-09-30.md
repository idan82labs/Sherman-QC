# CAD corpus geometry refusals: derived mathematical analysis — 2026-09-30

## Decision

Do not clamp spline trims or treat a degree-one direction as a whole-part recovery. The released diagnostics show two simultaneous problems: tiny trim/knot discrepancies, often in the **higher-degree transverse direction**, and other spline faces that refuse even with contained trims. A successor needs bounded native trim-domain and coefficient evidence, followed by complete material ownership. No >90% claim follows from this diagnosis.

The canonical next design is an evidence-first native inventory: preserve located/oriented face/edge/wire/pcurve identity; classify carrier loci using coefficient bounds on the **actual** trim domain; pair forming skins and certify boundary components; then develop selected skin geometry with the independent metric/coverage/overlap oracle. Carrier proof, boundary ownership and manufacturing blank generation remain distinct promises.

## Custody and source of evidence

The parent reported the V2 campaign terminal with observed outer exit 0, all 20 case exits retained and owned-process cleanup. Only subsequently released derived files were inspected:

`artifacts/vendor-cad-corpus-challenger-v2-2026-09-30-attempt1/case-{02,04,08,11,12,13,16,17,19}/GEOMETRY_DIAGNOSTICS.json`.

No `source.step`, physical CAD, scan, held-out, email, Gold, pool/manifest or authorization artifact was read. No native geometry was executed or campaign rerun by this analyst. These diagnostics describe source geometry and proof refusals, not measured bend truth or manufacturing acceptance. Face indices below are zero-based native-explorer indices. Overshoots are **parameter units**, not millimetres.

The frozen reader, normalization, inspector and oracle source remain unchanged by this report. Their failed/successful synthetic attempts and V2 corpus attempt remain terminal.

## Four interval-refusal cases

The table summarizes every refusing spline face in the requested four cases. Counts are face-level diagnostics; they are not independent parts, bends or inference errors.

| Case | Spline degrees U,V | Interval-refusing faces | In-bounds unproven faces | Largest U overshoot | Largest V overshoot |
| --- | --- | --- | --- | --- | --- |
| 04 | 1,3 | 0,51,52,72 | 3 | 3.2010330956795557e-15 | 4.033252215950479e-12 |
| 11 | 1,3 | 6,43,70 | 69 | 2.0636700168289385e-13 | 6.167664868768971e-13 |
| 13 | 1,3 | 1,2,5,38,57,61,64,65,90 | 54,55,58 | 1.7053025658242404e-13 | 5.107206343804357e-9 |
| 17 | 5,1 | 3,51,56,65,87,89,94,95,110 | 57 | 5.392988369214082e-13 | 1.6653345369377348e-13 |

For 04/11/13, the two-pole degree-one **U** direction is a possible affine generator direction, but many overshoots occur in cubic **V**, the possible transverse curve direction. For 17, the two-pole degree-one **V** direction is a possible generator, but most overshoots occur in degree-five **U**. "Possible" is intentional: degrees, pole counts and equal weights do not prove common translation or an extrusion locus. The diagnostics do not retain the actual pole vectors needed to make that decision.

Representative actual intervals make this distinction concrete:

- 04 face 0: U knot span `[0,1]`, trim `[-3.2010330956795557e-15,1]`; V knot span `[0,1]`, trim `[-4.033252215950479e-12,1]`. Both directions exceed the knot domain; only U is degree one. Face 51 has contained U but V upper trim `1.0000000000011573`.
- 11 face 43: contained U `[0,1]`; cubic V upper trim `1.0000000000001108`. Face 70 has contained U but cubic V lower trim `-6.167664868768971e-13`.
- 13 face 61: contained U `[0,1]`; cubic V lower trim `-5.107206343804357e-9`. Face 2 has only degree-one U overshoot `1.5942802633617248e-13`, while V remains `[0,1]`; face 65 similarly has only U overshoot `1.0502709812953981e-13`. An affine-generator continuation hypothesis could address these individual faces only after coefficient proof.
- 17 face 3: degree-five U trim equals its knot span `[0,9.28335128002]`; degree-one V upper trim is `2.0000000000000293`, compared with upper knot `2`. Face 51 has analogous V-only overshoot `1.6653345369377348e-13`. Conversely, face 56 has degree-five U lower trim `-5.078553522055199e-13` against lower knot `0`; V is contained. Face 65 has the analogous U overshoot `5.392988369214082e-13`.

Every one of these four parts also has a contained-domain `UNPROVEN_NONPLANAR_BSPLINE_SURFACE` face. Consequently, even perfect handling of all interval discrepancies would not certify a complete unfold for any of them. It could simply expose the next legitimate refusal.

## Other unproven cases

All refusing spline faces in cases 12/16/19 have degree pair `[1,3]`, two U poles, positive finite weights and reported weights identical across both directions.

| Case | In-bounds unproven faces | Additional interval-refusing faces | Interpretation |
| --- | --- | --- | --- |
| 12 | 14 | none | Refusal cannot be explained by trim overshoot. Four adjacent faces include three cylinders and one plane; adjacency alone supplies no wall/fiber proof. |
| 16 | 6 | 174 | Face 6 has contained U/V; face 174 has U lower overshoot 5.090007594231338e-15 and V upper overshoot 4.5075054799781356e-13. Both issues remain. |
| 19 | 42,240,309 | 84,204,311,360,369 | Three contained-domain failures plus five interval failures; maximum U overshoot 1.5987211554602254e-14, V overshoot 2.456883445294279e-14. |

Equal weights are a denominator/separation precondition, not a geometric proof. Unknown causes include nonconstant generator vectors, an oblique/nonperpendicular base, inconsistent physical endpoint levels, or other coefficient geometry. The current summaries cannot distinguish them. Naming such a face "cut wall" from its neighbors would be an unsupported omission of material.

## Why clamping is wrong, and what exact continuation would require

[OCCT BRepAdaptor_Surface](https://dev.opencascade.org/doc/refman/html/class_b_rep_adaptor___surface.html) documents face-restricted UV parameter ranges. A restriction bound need not be identical to the underlying spline's knot endpoint. [Geom_BSplineSurface::LocalValue](https://occt3d.com/dev/doc/refman/html/class_geom___b_spline_surface.html) documents evaluating the selected local patch outside its knot interval. These facts explain why the mismatch is plausible; they do not establish that every out-of-domain face admits the existing positive-basis certificate or that ordinary native evaluation follows the proposed continuation.

For a degree-one, two-pole generator with clamped knots `k0,k1`, equal positive weights in that direction, and a coefficient-certified common displacement `d`, the locus has affine continuation:

\[
X(u,v)=C(u)+\frac{v-k_0}{k_1-k_0}d.
\]

The physical trimmed displacement must use the **actual** `v0,v1`, including any overshoot. Clamping changes it to another locus/domain. A near-common displacement with residual bound `E` instead needs the extrapolation amplification bound

\[
E\bigl(|1-s|+|s|\bigr),\qquad s=(v-k_0)/(k_1-k_0),
\]

over the entire actual interval, plus base-plane and native trim/ownership bounds. Merely checking overshoot against an arbitrary parameter epsilon is not a geometric proof. Length, parameter scaling and weights determine the physical effect.

For higher-degree transverse continuation, positive convex-hull bounds no longer apply unchanged outside the knot domain. For a degree-p Bézier span, the absolute Bernstein sum outside `[0,1]` is

\[
\sum_i |B_i^p(s)|=(|s|+|1-s|)^p.
\]

This gives an amplification factor for polynomial coefficient residuals. Rational geometry additionally needs a strictly positive denominator bound over the extended interval; original positive weights do not automatically prove it when basis coefficients change sign. A bounded successor can convert only the first/last active native B-spline spans to homogeneous Bézier form, reparameterize them over the proposed extended interval, and certify denominator, residual and derivative/regularity bounds by Bernstein interval arithmetic and finite subdivision. It must preserve exact span/knot/weight evidence and refuse if a bound cannot be proved within the frozen budget. Tensor products require the corresponding two-direction absolute-basis amplification.

None of the necessary pole vectors, homogeneous coefficients, knot multiplicities or derivative bounds are retained in the current diagnostic summary. This design therefore needs new source, independently authored tests and a **new** frozen evidence campaign; these terminal files cannot be used to synthesize a missing proof.

The V1-first path also matters: the incumbent checks candidate parameter intervals before some direction-specific coefficient decisions. An invalid cubic transverse interval can stop evaluation before the affine direction is considered. A successor should separate "carrier coefficient candidate" from "actual-domain validity" and publish each refusal; it must not catch every failure and pretend that the face was proved.

## Cases 02 and 08: stepped collar/boss-like boundary evidence

Both cases have no surface-carrier refusals in the inspector. Their `UNKNOWN_HOLE_WALL_BOUNDARY` failures concern ownership/model scope, not unsupported surface representation.

Case 02 has principal plane faces 26 and 30 at world X approximately `0` and `2`, corresponding to the bend radius difference `2`. Two full cylinders, faces 22 and 27, have radius `3.95`, axis +X and axial extent X `[-2,0]`; they join principal plane 26 to planar faces 23/28 at X `-2`. Smaller coaxial cylinders 31/29 have radius `2.7`, axis -X and extent X `[-2,2]`, joining those X=-2 planes to principal plane 30. This is extra boundary material beyond the nominal X `[0,2]` sheet slab, not a single uniform through-thickness wall. The inference is collar/boss-like protrusion, not a proved manufacturing operation.

Case 08 has principal planes 23 and 25 at Z approximately `-1.5` and `0`, matching the bend radius difference `1.5`. Outer full-cylinder faces 17/18 and partial cylinder 20 at a third location have radius `3.95` and lie over Z `[-3.5,-1.5]`; planar face 19 connects that third partial cylinder to faces 23/32. Their associated inner cylinder faces 38/37/36 have radius `1.65` over the same two-unit interval. Hole cylinders 27/34/33 have radius `2.7` through Z `[-1.5,0]`. Planes 29/31/32 occur at Z=-3.5; planes 28/30/35 occur at Z=-1.5 and connect the differently sized inner hole sections. The native adjacency supports a multi-face stepped protrusion/shoulder hypothesis, not the incumbent one-wall translated-rail rule.

These descriptions use retained analytic carriers, parameter ranges and incidence. They do **not** verify complete annular trims, solid fiber uniqueness, feature provenance, whether material was attached/machined/formed, or the pre-forming hole size. A future selected-skin preview could retain its chosen hole footprints plus explicit feature overlays only after complete source-face ownership and native boundary proofs. Manufacturing blank dimensions cannot be inferred uniquely from the final shape alone.

## Authorized next source design and stopping conditions

1. Add bounded coefficient diagnostics for direction candidates: common-displacement residual, normal-fiber direction variation, base-plane residual, actual trimmed endpoint variation, weight coupling, and extension denominator/derivative bounds. Retain native pcurve/wire trim topology, not UV boxes alone.
2. Author independent pure Bézier/homogeneous interval witnesses and in-memory native fixtures: affine generator overshoot without clamping; cubic transverse extension with a certified positive denominator; denominator-near-zero refusal; tiny parameter overshoot with large physical derivative; common full-knot endpoints but unequal actual trimmed endpoints; repeated knots and reversed parameter direction.
3. Keep an affine-only carrier continuation as a narrow face-level capability. Do not present it as a complete four-case solution. Unknown spline roles and stepped collars remain explicit material-owning refusals until a boundary-component successor proves them.
4. Independently review the source and freeze a separately versioned bounded campaign before collecting any new native coefficients. Do not modify or re-execute V2. Separate extraction counts, folded display availability, selected-skin development availability, continuous geometric accuracy and manufacturing truth.

This report changes documentation only. No new native execution or real-input authority is created.
