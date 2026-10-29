# Two-plane plus finite-context pose proposals

Date: 2026-09-10. **DESIGN ONLY — no implementation, run release or advancement.**
This additive proposal family may address the exact-configuration initialization
stop. It does not change epsilon, normal admission, registration, measurement or
scoring, and is not a prerequisite for the independently frozen all11 rerun.
Unsupported cases remain in that run's denominator. No physical input, Gold,
retained prediction geometry or authorization artifact was accessed for this design.

## Evidence and the smallest useful change

The [released exact-configuration diagnostic](2026-09-10_mesh_physical_config_constructed_v9_result.md)
records 0/2 nominal outputs: both 27-plane banks reject all 48 scheduled triplets
before any pose or finite query (lines 9–38). Epsilon .05 mm makes the narrow
cap's conditional cone approximately 17.83 degrees, above the unchanged 10-degree
policy. Only two clean normal families survive. This proves a triplet-proposal
gap on that construction, not global unobservability or a physical-scanner bound.

Propose **two qualified planes, with several axial placements suggested by finite
scan/CAD context**. Do not duplicate a plane to manufacture a third direction,
promote a rejected cap normal, thicken the .8 mm cap, or lower epsilon/cone gates.
Importantly, the original source has flange/arc samples at z=1..29 but cap samples
at z=0 and 30 (`tests/test_cad_guided_mesh_connected_native_v1.py:130`). The full
owned scan-row axial extent therefore contains context absent from qualified
plane-core extrema. Those positions can suggest a start even when their cap
normal is unsuitable for initialization. This is source-derived motivation,
not a promise that refinement or measurement will succeed.

## What two planes determine

Use anchored equations `n_i · (x-a_i) = d_i`, unit normals and declared cones.
For each correspondence/order and target-sign branch, write target normals
`m_i = sigma_i n_ci`, with each `sigma_i` in {-1,+1}. Two nonparallel, correctly
corresponded, oriented normals fix a proper rotation **only in the exact
equal-included-angle case**. Construct right-handed frames using

```
e1 = n1
e2 = (n2 - (n1·n2)n1) / ||n2 - (n1·n2)n1||
e3 = e1 × e2
R  = [e1c e2c e3c] [e1s e2s e3s]^T
```

Enumerate both correspondence orders and all four target signs; never permit a
reflection. Under unequal noisy normals this central frame construction is only
a proposal. Validate properness and both normal residuals; retain the declared
cones and do not claim a pose-uncertainty enclosure. Keep the existing angular
compatibility/extra-residual policy. An additive rank-two admission must also
check nominal conditioning and a positive independence margin for the **two-row**
normal system under both cones, not call the three-row solver on padded input.
For example, the V6 perturbation bound
`eta = sqrt(sum((2 sin(alpha_i/2))^2))` can be applied to the two-row matrix:
require `s_min > eta` as well as the unchanged .02 nominal ratio. This is the
same engineering arithmetic limitation as V6, not a new interval certificate.

Choose local origins `o_s,o_c` and map `y = R(x-o_s) + o_c + t`.
For a point on each source witness plane, `x_i = a_si + d_si n_si`, form

```
M_i = m_i^T
b_i = sigma_i [n_ci·(a_ci-o_c) + d_ci] - m_i·R(x_i-o_s)
t0  = M^T (M M^T)^(-1) b
u   = (m1 × m2) / ||m1 × m2||
t   = t0 + lambda u
```

Use a checked rank-two solve, not an explicit inverse in implementation. In exact
compatible geometry, these are the two plane-offset constraints; in noisy
geometry they align the witness anchors only. Rotation/sign/correspondence
uncertainty remains. Translation along the intersection direction `u` is wholly
unresolved by the planes: two planes do **not** determine six degrees of freedom.
Changing either coordinate origin must leave physical pose action unchanged.

## Finite context: votes, not correspondence truth

The first helper should use extrema only, with no learned features or new native
dependency. Project **all** bound rows of each assigned scan support and the
whole owned scan; project all vertices of the complete assigned CAD faces and
the whole supplied CAD triangle buffer. Preserve holes/disconnected islands in
the backing geometry. Use the SAME anchored proposal frame for both intervals:

```
A = range over source x of u·[R(x-o_s) + t0]
B = range over target y of u·(y-o_c)
```

These are projected coordinates BEFORE adding `lambda u`. Do not substitute
raw-coordinate extrema: for identical z=[0,30] buffers with R=I, o_s=7z and
o_c=-3z, the correct vote is lambda=10, not0. Under origin changes h_s,h_c, let
q=R h_s-h_c. Then t0'=t0+(I-uu^T)q and lambda'=lambda+u·q, so physical pose action
is unchanged. Test this explicitly, not only rotation/translation of the part.

Cache projection work by bound buffer/support digest, coordinate frame and exact
direction (R^T u for source projections), accounting for the declared origins
and constant offsets. Direction alone is not a cache identity across different
faces or frames. Reuse is independent of lambda; retain
exact source-row/triangle identities and extrema witnesses. No ROI crop, family
label, Gold endpoint, expected bend count/radius, CAD measurement result or
post-inference score participates. CAD surface positions are pose-model context,
not measured bend answers.

For projected source interval `A=[a-,a+]` and target `B=[b-,b+]`, endpoint votes
are `lambda=b- - a-` and `lambda=b+ - a+`. Ideal exact containment would allow
the interval `[b- - a-, b+ - a+]`. In real partial scans this is only a **heuristic
context interval**, not an enclosure of the true pose: outliers, normal error,
holes and nonuniform visibility invalidate that interpretation. Never use its
emptiness or midpoint to certify exclusion, support or uniqueness. Extrema
arithmetic is proposal-only; any future certified pruning needs its own outward
error analysis and independent qualification.

Use eight fixed candidate slots per signed pair assignment: two whole-buffer
endpoint votes, four assigned-face endpoint votes (two per face), the midpoint
of a nonempty whole-buffer context interval, and the midpoint of the nonempty
intersection of the two assigned-face context intervals. Missing midpoint slots
remain explicitly absent. Deduplicate exactly equal poses only while retaining
every contributing slot/witness; do not merge nearby axial placements by an
undeclared tolerance. Midpoints are extra seeds, never a claim that centered
occlusion is the true placement.

Every resulting candidate retains the intervals and source of its lambda.
Partial interior scans can slide; repeated tabs can support several separated
placements; two endpoint votes can conflict. Retain those ambiguities and all
scheduled outcomes. Eight samples do not resolve a continuum or exhaust a normal
cone. Report `AXIAL_CONTEXT_NOT_CERTIFIED`, selected-policy completion separately
from geometric-search completeness, and global uniqueness `NOT_ESTABLISHED`.
An ungenerated or unscheduled alternate pose is not disproved. More positional
context (finite edge/notch correspondences) is a later, separately qualified
extension if this deliberately small helper cannot generate useful starts.

## Integration boundary and unchanged refusal

Relevant source seams, read without imports:

| Existing seam | Required additive boundary |
| --- | --- |
| `backend/cad_guided_plane_hypotheses_v4.py:138,239` | Reuse proper-pose/sign principles, not its three-plane intersection or schema. |
| `backend/cad_guided_plane_search_v6.py:296,322,335,574` | Reuse owned bank and angle scheduling principles; retain original triplet search unchanged. |
| `backend/cad_guided_plane_initialization_v8.py:90,117,147` | Reuse copied buffers, complete extraction, admitted plane records and exact raw-support binding. |
| `backend/cad_guided_plane_initialization_v10.py:227,267,477` | Current adapter requires exactly three distinct assignments. A new two-assignment adapter is necessary; no fake third match or mutable monkeypatch. |
| `backend/cad_guided_global_registration_v3.py:415,447,524` | Preserve finite support, all three rank checks, residuals, coverage, competing-pose selection and refusal. |

The new adapter must eagerly bind complete CAD faces and both complete scan
supports before any screening. All generated/scheduled sign/axial candidates
receive the same rejection-only box checks and unchanged finite-distance policy;
boxes never admit support. Do not pass a support chunk as the full denominator.
Keep full raw input for downstream scan measurement and unchanged declared
registration sampling. Neither a pose vote nor CAD context may fill missing scan
geometry, set an observed count, or bypass `free_circle` fitting/publication.

There is a real possible **second blocker**. V3 computes rank from definite,
unambiguous, nonuncertain face-interior rows, including optimizer and weighted
subsets (`cad_guided_global_registration_v3.py:422–447`;
`cad_guided_global_registration_v2.py:183`). For point-plane Jacobian
`J_i=[p_i×n_i, n_i]`, if every admitted `n_i·u=0`, the twist `[0,u]` is a null
direction. Adding more points on those same faces does not fix axial rank.
This derivation is consistent with the point-plane sliding/stability analysis in
[Gelfand et al., Geometrically Stable Sampling for the ICP Algorithm, section 2](https://pixl.cs.princeton.edu/pubs/Gelfand_2003_GSS/stabicp.pdf).
That paper supports the constraint diagnosis, not this proposed implementation
or a guarantee at .05 mm.

Weak-cap rows may supply usable later finite-query face normals, but this is
unknown until the unchanged consumer evaluates them. Endpoint correspondences
could constrain position while the existing face-normal rank still refuses;
do not redefine that refusal as success. A finite-edge/positional observability
model, if needed, is a separate mechanism and proof obligation outside this
proposal. Proposal generation alone cannot close the 0/2 output gate.

## Bounded enumeration and evidence

Prospective helper design caps, **not an execution protocol or increased runtime
allowance**: at most the existing 128 bank planes and 2048 CAD planes; 64 distinct
scan pairs; four ordered CAD matches per scan pair (order included in this quota);
four signs; eight axial slots: at most 8192 candidate slots. Use deterministic
rank-bin/fair scheduling over compatible pairs, not a first-prefix quota. Bound
CAD pair records at 2,096,128 and join tests at 32,000,000; never materialize the
scan-by-CAD Cartesian product. Retain scheduled and omitted identities/counts.

Count at most 64,000,000 projected scan-row visits and 128,000,000 projected CAD
vertex visits, with chunks <=65,536, bounded numeric working buffers <=256 MiB
and proposal/report bytes <=32 MiB. All finite-buffer validation and projection
work must be charged, including cache misses and failed branches. These caps may
refuse large/repetitive inputs; no capacity or timing claim follows. A proposed
first pure invented gate is <=60 s wall /45 s CPU /1 GiB process-tree RSS /1 MiB
stdout, subject to a separate reviewed freeze before execution.

The new helper's numeric/custody/byte/work overflow invalidates its whole proposal
set, with a retained attempted ledger, not a usable favorable prefix. Ordinary
declared scheduling omissions are distinct from fatal work incompletion. Do not
steal the old triplet lane's quota or add another 8192 poses invisibly: initially
qualify the helper separately; later union preflight must charge both lanes under
the unchanged integration caps or explicitly refuse the union. This document
does not authorize another native/physical budget or change the active V3 run.

## Independently invented adversarial gate and advancement

Construct oracles independently of proposal code. Exact rational/permutation
poses should use direct Fraction plane/nullspace checks plus <=1e-8 mm pose-action
error on modest coordinates; oblique floating fixtures need separately frozen
roundoff/error ceilings. Declare noise before evaluation, not from outputs.

| Fixture | Required outcome |
| --- | --- |
| Two orthogonal planes, asymmetric complete axial context, independent proper pose; shuffled IDs/orders/signs/origins | True axial seed present, correct plane action, det +1; no accepted-registration claim from the helper. |
| Same two planes with an interior scan window inside a longer extrusion | Preserve the nonzero-width axial ambiguity; no midpoint/endpoint uniqueness; unchanged downstream null-rank refusal. |
| Repeated/disconnected axial islands, plus a hole inside a loose finite box | Multiple placements retained; box containment cannot admit support across the hole; full-triangle consumer rejects unsupported rows. |
| Partial endpoint occlusion, asymmetric cropping or one remote outlier | Extent votes may miss truth; report limited proposal coverage, not wrong-part/missing-bend or accepted count. |
| Parallel/near-parallel cones, sign/permutation branches and mirror-only context | Refuse unresolved independence; never return det -1; a proper plane-pair rotation alone cannot certify chiral whole geometry. |
| Cap-boundary/late NaN/stale row union/mismatched CAD-face digest/byte overflow | Whole-family failure, exact attempted/omitted counters, no surviving positive prefix or mutated input. |

**Advance this mechanism to connected testing only** after independent source
review and all frozen invented tests pass: every declared exact positive contains
its oracle seed; every ambiguity/invalidity control preserves its required
unknown/refusal; all provenance/counter/immutability checks pass. This establishes
proposal support only, not physical F1 or output coverage.

**Close the original output blocker only** in a separately frozen connected
successor with both original nominal constructions unchanged: 2/2 publish the
required single scan-derived conditional instance and satisfy every original
registration, completeness, support and free-circle oracle. Preserve clean limits
radius 2e-5 mm, sweep 2e-5 degrees, endpoints .02 mm, pose action .20 mm; noisy
limits .01 mm/.1 degrees/.05 mm/.20 mm
(`tests/test_cad_guided_mesh_physical_config_constructed_v9.py:44–61,188`).
Required missing-arc safety controls remain separate; nominal abstention is
failed output coverage, never a passing safety sentinel. Do not replace the thin
fixture with broad/thick easier geometry. Its 2752 rows still do not exercise
8192 effective refinement samples. Retain every stage ledger and failure; no
physical, unseen-family, calibrated-measurement or promotion claim follows.

Independent `/root/cad_measurement_integration_review` checked the proper-rotation
and rank-two/sign formulas, the conditional Frobenius/Weyl cone margin on BOTH
source and CAD systems, and the unchanged output oracles. Its two pre-implementation
findings—explicit anchored projection coordinates and full cache identity—are
incorporated above. No statistical independence of cone errors is assumed.

Checkpoint: design artifact only; source/released narrative and one primary
paper inspected, no runtime edits, imports, experiments or protected input reads.
