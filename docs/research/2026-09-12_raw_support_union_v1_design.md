# Raw-support union fallback: prospective engineering design

Date: 2026-09-12. Status: source-only development, not qualified or integrated.
The Gold11 V5 paired run is already active and remains unchanged. This design
does not amend that candidate, protocol, scoring, resources or terminal evidence.

## Mechanism and acceptance, not a new scoring rule

The retained connected synthetic result was 2/4 successful measurements. The
lower-scan cases lacked an existing member whose raw-role support covered the
whole ownership component. Their union contains additional valid candidate
support; fitting that union has not been shown to succeed. The existing-cover
gate is a set-level limitation, separate from circle/plane or finite-support
accuracy. Physical Gold11 outcomes must be reported independently.

The proposed successor keeps every maximal ownership component and every
structurally valid support record, including records whose earlier fit or finite
coverage failed. It never discards an inconvenient member or retries a favorable
subset. It adds a fresh whole-union model only when the structural reason is
`NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER`. An existing cover that fails its
geometry or finite-support check cannot enter this fallback.

Success requires an integrated, independently tested measurement route that
improves connected output while preserving the existing lane and required
refusals. A planner's COMPLETE or UNION_PROPOSED status is not that success.
No altered Gold line, nominal CAD answer, count proxy or case-identity rule is
part of the mechanism.

## Mathematical conditions

Ownership components retain the existing arc/arc and arc/flange exact-position
edges. Shared flanges alone do not merge components. Arc/flange position
conflicts veto the entire component. Raw row IDs, not welded coordinates, carry
support; distinct IDs with identical positions cannot add independent support.

For member i, let x_i be its binary flange swap and s its local flange label.
If one raw ID belongs to local flange s in i and t in j, consistency requires
s XOR x_i = t XOR x_j, equivalently x_i XOR x_j = s XOR t. All constraints must
be satisfied. A contradictory cycle or disconnected relative parity refuses
the union. A checked star per shared ID represents its full equality constraint
without constructing every pair. Remaining global-flip freedom is resolved by
lexicographic raw-ID role ordering, not fit quality or CAD geometry.

After alignment require nonempty n-ary intersections of the arc and each of
the two flanges across every member. Pairwise intersections and overlap chains
are insufficient. These conservative common anchors establish only sample
association, not one physical bend, a unique pose, or adequate conditioning.
They may refuse legitimate distributed observations with no global anchor.

Fit all rows in the complete aligned union once, using the unchanged free-circle
and scan-plane mathematics. A common model's maximum residual bound applies to
its subsets under that same model. It does not prove that each subset's own
estimator is conditioned, that every member has finite coverage, or that an
earlier failed fit passed. Keep those original outcomes intact.

The canonical raw-axis sign and canonical flange order are serialization
conventions. Swapping the flanges reverses the signed sweep while leaving sweep
magnitude, fitted radius and mid-angle carrier geometry invariant under the
declared model. Neither sign is a verified manufacturing bend sense. Radius
and sweep-magnitude deviations may reference CAD nominal values, but no nominal
center, radius, axis or sweep supplies the measured answer.

## Source seams and required integration

1. New `backend/cad_guided_raw_support_union_v1.py`: bounded structural planning,
   distinct EXISTING_COVER / UNION_PROPOSED / UNAVAILABLE groups, exact raw-role
   unions, member permutations, common anchors and explicit work reservations.
   Every result declares measurement_admissible false.
2. A future separately versioned measurement wrapper consumes the complete
   original context evidence. It preserves existing-cover results unchanged and
   attempts only the predetermined structural union lane. The wrapper must
   revalidate source publication and byte/raw-row lineage, and veto a union if
   any row is explicitly excluded by any contributing association's published
   surface-role filter. It must not silently remove that row or claim a single
   existing context covered the assembled union.
3. The wrapper performs one full union fit and all finite axial, local-slab and
   angular-coverage checks. Publication independently reconstructs membership,
   parity, anchors, filters, raw rows and the exact group-only prediction
   projection. No fabricated cover_record_id or full-cover context witness is
   permitted. The new claim is a common model on the complete component's raw
   role union, not EXISTS_COMPLETED_EXISTING_COVER_CONTEXT.
4. Only after independent source/math audit and a frozen bounded test protocol:
   source controls, adversarial numerical cases, then a connected byte workflow.
   Qualify a separately frozen physical successor only if the retained V5
   outcome and tested mechanism justify it. Never reopen V5 or infer uplift from
   source/test counts.

## Prospective bounds and independent tests

The proposed planner configuration has hard/default limits of 512 records,
1,000,000 raw points, 4,000,000 role memberships and each ownership/parity-link
ledger, 32,000,000 set-row work, 262,144 member comparisons, 128,000,000 ordering
work, 64 MiB input-binding bytes, 8 MiB plan and a conservative 4 GiB accounted
working reservation. These are proposed implementation guards, not measured
RSS or a release to execute. Every potentially large allocation/operation must
be admitted first. Fatal late errors clear all proposed groups, retaining work
and refusal evidence. Wrapper fit/filter/publication budgets still need their
own reviewed accounting before execution.

Before any execution or freeze, independent source-only capacity review replaced
the initial 256 MiB proposal: at 1M float64 points and 512 records it would admit
only 162135 role memberships, unintentionally narrowing the old 4M contract.
The complete reservation formula at 4M memberships is 4198408704 B for float64
or 4270408704 B for sixteen-byte source scalars, both below 4 GiB. This establishes
reservation compatibility only; set/order/binding/report limits may still refuse
such an input. The 1024-byte-per-membership factor is an accounting assumption,
not allocator proof. Add allocation-free arithmetic and exact/one-byte-below
reservation controls. Actual wrapper/engine/inputs/interpreter/fit memory must
be funded separately inside its future frozen process-tree budget. V5's active
budgets are unchanged.

Independent source-only tests are being authored for global role/order
invariance, parity chains/cycles/contradictions, disconnected parity, pairwise
but absent common anchors, role conflicts, policy mismatch, original failed
fits, complete non-nested unions, duplicate positions, flange-only sharing,
existing-cover preservation, exact/one-over budgets and late whole-output
clearing. Later wrapper/numerical tests must include explicit filter exclusions,
clean/noisy complete and partial scans, real radius/angle deviations, incompatible
axes/radii, axial gaps, angular holes and nearby competing sheets. Passing the
declared common-sample model cannot certify physical surface identity.

Independent design review accepted the parity equation and the conditional
architecture with these caveats. No new module/test has executed, no union fit
has been measured, and no production or floor release is claimed here.

## Independent numerical-seam review, source only

The separate reviewer traced `_canonical_fit`, `_plane`, `_circle` and the finite
constructor. It confirmed no CAD nominal answer enters the fixed-role refit;
role selection itself remains CAD-conditioned. Required later tests include:

- Change valid CAD nominal radius, sweep, origin, axis, span and radial reference
  while holding raw roles and numeric policy fixed: measured geometry must not
  change. Explicitly require free_circle before calling the helper; the helper
  itself always computes that model rather than validating the selected mode.
- Retain plane rank/spread/noise-relative singular-value and maximum-residual
  checks. Test warped or narrow-strip flanges and nearly parallel/opposed planes.
  Axis sensitivity grows as 1/abs(sin(theta)); the existing 20–160-degree sector
  restriction is not a propagated uncertainty interval. The flange orientation
  reference comes from its own raw SVD, not independent material-normal truth.
- Point-weighted least squares is not density invariant. Test dense biased arc
  samples with sparse contradictions, incompatible radii/axes, conical/twisted
  unions and shallow arcs compatible with a flat strip. Preserve all-row maximum
  residuals, flat-competitor/Jacobian checks, 80 fit evaluations and the 2048
  convex-hull-vertex limit. No average residual or row trimming may conceal a
  contradictory member.
- Test axial/angular holes, complementary-half flange support and a central
  partial arc between wider flange-implied contacts. Present finite support
  does not require arc samples near both extrapolated tangent contacts, nor
  independently conditioned flange fits in every local slab. The carrier remains
  conditional and may extrapolate angular contact; it is not physical trim truth.
- Under rigid transformations compare axes up to sign and endpoints as an
  unordered pair, including canonical-axis component ties. Require geometric
  rather than signed-serialization invariance under global flange swaps.

These are bounded successor requirements, not retroactive changes to V5, a new
benchmark prerequisite, or claims that the numerical union tests have passed.
