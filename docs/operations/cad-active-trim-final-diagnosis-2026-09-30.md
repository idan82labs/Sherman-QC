# CAD active-trim final diagnosis

2026-09-30. Bounded derived-only review; no raw CAD, native execution, or source changes.

## Result

Root observed terminal outer exit zero for `artifacts/vendor-cad-corpus-exact-trim-v6-2026-09-30-attempt1`; its independent POST was released with SHA256 `0bef61bca4641323d0ba07b81d49fc9761ceedaee9489c71feb6f183c71598f4`. Root reports 98 axes, 19 folded displays, six flat previews: unchanged from the bounded-corridor predecessor. This review independently inspected the released EXTRACTION certificates for the 24 remaining ordinary historical pair groups (48 skins), not the physical inputs or POST contents.

Exact active-trim restriction **succeeded on all 48 failing spline edges**. Every `active_interval_proof` has `status=PROVED`, `restriction=COPY_SEGMENT_EXACT_INTERVAL`, `segment_tolerance=0`, `source_mutated=false`, and `interval_clamped=false`. All 48 actual trim intervals equal the original actual parameter domain and underlying knot interval. Forty-seven are `[0,1]`; case17 face21 is `[0,0.450612985013]`. The restricted coefficient ranges equal the recorded underlying coefficient ranges on all 48. Consequently this mechanism fixes a legitimate representation limitation but offers no range tightening for these already-full-domain edges.

All 48 remain `PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK`; all report positive weights `[1,1]`, source tolerance `1e-7` mm, and corridor fraction `0.25`. None reaches the final side/join/adaptor `error_budget`: refusal is at the coefficient corridor subgate, not an exceeded composed residual after rectangle proof.

## Quantified remaining certificate limitation

The coordinates are physical cylinder coordinates `(R*u,v)` in mm. For each failing coefficient set, the best axis-aligned coefficient corridor is half the smaller coordinate range width. These are coefficient enclosures, **not actual curve excursion, material-boundary measurements, or independent lower bounds on geometric deviation**.

| Case | Historical pairs / skins | Best coefficient half-width range, mm |
| --- | --- | --- |
| 04 | 3 / 6 | 0.114654031–1.464948713 |
| 11 | 2 / 4 | 0.015358726–0.348860283 |
| 12 | 1 / 2 | 0.063508327 |
| 13 | 4 / 8 | 0.017078299–0.634004103 |
| 16 | 2 / 4 | 0.285714286–0.707106781 |
| 17 | 5 / 10 | 0.019408122–0.518280426 |
| 19 | 7 / 14 | 0.223223305 |

The allowed corridor is `2.5e-8` mm; even the smallest coefficient half-width is about 614,349 times larger. The current whole-active-domain certificate necessarily refuses these coefficients. Increasing epsilon would replace a tolerance-scale straight-locus proof with a macroscopic approximation and is not warranted.

Full active-interval proof is now present, so the earlier uncertainty about unused portions of the carrier is resolved for these visited failing edges. Remaining uncertainties are distinct: the receipts do not publish ordered coefficients or derivative signs, so the combined refusal cannot be classified as proved backtracking; coefficient hull widths alone cannot prove actual nonrectangular material. Early return also leaves later edge uses unclassified. One wire does not independently prove a simple closed non-self-intersecting material boundary.

Native face groups: 04: 2/3,5/6,55/56; 11: 6/56,54/55; 12: 13/14; 13: 1/86,4/5,38/63,54/57; 16: 1/2,4/174; 17: 3/21,5/22,23/106,51/80,53/59; 19: 1/240,8/9,42/44,81/179,82/178,86/369,206/207. These historical groups are coverage probes, not gold truth.

## Next architectural seam

Stop further global corridor/tolerance changes. Introduce a bounded **complete trimmed-cylinder domain representation**, separately from the existing rectangle-only unfolding capability:

1. Collect all oriented native edge uses and full active-span coefficient/interval proofs before returning a face classification. Split combined diagnostics into corridor excess, monotonicity/backtracking, interval failure, topology failure, and unsupported representation.
2. Use exact knot insertion/Bezier-span subdivision with certified coefficient bounds to test the actual locus against proposed straight supports. Bounded subdivision may tighten an enclosure; it must not sample a curve into proof. If the certificate budget is exhausted, retain unknown. Publish all derivative/interval bounds needed to distinguish a proved curved boundary from an unresolved enclosure.
3. For boundaries that are certified nonrectangular, retain their actual continuous UV loops, seam lift and hole exclusions; prove incidence, orientation, nonintersection, and material-domain coverage. Transport those domains to the already-recorded common carrier frame. Do not substitute UV boxes or invented full-span strips.
4. Recognize finite bends from corresponding inner/outer material domains and opposed tangent flange-rail chains even when axial end boundaries are curved. Preserve geometric-mid-thickness marker semantics. Keep operation grouping distinct from finite feature identity.
5. Permit extraction to report a qualified nonrectangular finite bend while rectangle-only unfolding explicitly abstains. General trimmed-strip unfolding is a separate mathematical and native-oracle gate, not a schema relabel.

Qualification must include independently authored curved-end cylinder strips, certified straight loci with loose control enclosures, trim/seam/hole/retrace negatives, split rail chains, reversed axes and SO(3) poses. Freeze geometric proof, complexity caps and error budget before any root-coordinated corpus evaluation. No filename or observed case magnitude becomes a threshold feature.

## Independent accuracy gate and checkpoint

No greater-than-90% precision/recall claim follows from 98 axes or six previews. Freeze blind per-bend technical CAD references independently of extractor output: native face/solid identities, finite material domains/exclusions, paired radii, tangent flanges, angle convention, feature-versus-operation identity, and explicit unknown regions. Independent review resolves disagreements before one-to-one geometric/domain matching under predeclared tolerances. Report false positives/negatives only where reference coverage is complete, plus abstentions, unknown coverage, per-part/family counts and precision/recall. Technical references are engineering evidence, not qualified human manufacturing truth or unseen-family validation.

Standing authority covers CAD-only source successors and mapped-corpus evaluations under fresh freezes and independent gates. New datasets, outside contacts and human manufacturing approval remain outside scope. Frozen sources are unchanged. The next action is source-only complete-boundary diagnostics/proof design; unresolved material domains remain unknown, not absent bends.

### Derived evidence hashes

Reviewed `case-XX/EXTRACTION.json` SHA256 values within the released attempt:

- 04: `a2207e899d519d34acf1e65c168165a4042a93f61f7581f55c60864a8209e338`
- 11: `736a11f7a6abb36d9c684d3178f2227ddb72c1548c7edf8935028f92048352af`
- 12: `411062549e40c354fd19a2f89d2b595132a9565f3d47e2d195fc8ccfaa05a4e7`
- 13: `2370eb2ad11447145577afb97b699f2ac7ef8b9803bb4476f67ae48124bf7033`
- 16: `c1c9d4dd3ac256076967607e9dcd7d430fe1029f949850f4516669f8ea6ce3b3`
- 17: `fd4b83b739ff40bdcf7ec0086412c6188ffb6ab12d98daa5fed7aa2f8ce7224b`
- 19: `13d16d3689fe0930c01a88c35f7d1dbaeda09d7e7c9e181a47f7cae26d347aef`
