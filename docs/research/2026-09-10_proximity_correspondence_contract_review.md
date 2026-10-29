# Proposal proximity is not correspondence: connected contract review

Date: 2026-09-10. Source-only mathematical/design review. No runtime change,
target import, test, geometry query, physical access or benchmark execution.
This is a conditional recommendation for the requested strategic consultation,
not a frozen execution protocol or an approved engine successor. Independent
source/math review by `cad_support_lineage` is GREEN for consultation only;
its finite-set, completion-status and source-citation clarifications are included.
No implementation or execution is released by that review.

## Decision and newly localized integration risk

Separate finite proximity from correspondence in a NEW versioned proposal
contract. Do not remove ambiguity flags globally or relabel a proposal as a
validated pose. A prefilter-only edit could rescue the constructed initializer
while leaving the connected engine unable to measure anything:

| Current seam | Source | Consequence for a successor |
| --- | --- | --- |
| Initializer support | `cad_guided_plane_initialization_v11.py:542` | Intersects finite distance and ambiguity mask before .95 coverage. |
| Refinement selection | `cad_guided_global_registration_v3.py:257` and `:355` | The same exclusions still determine fixed-target optimizer rows. |
| Final support and rank | `cad_guided_global_registration_v3.py:419` and `:430` | Support again excludes uncertainty; rank further excludes boundaries and uncertain regions. |
| Workflow evidence binding | `cad_guided_mesh_offline_v11.py:44` | Requires initializer and final support to carry the same old policy identifier. A split cannot honestly retain that identifier. |

The [audited noisy diagnostic](2026-09-10_two_plane_v11_stage_diagnostic_result.md)
proves only the first seam rejects an available correct pose. It does not prove
the later gates fail for that same case; they were not run in the diagnostic.
Source establishes the repeated dependency, not a new connected outcome.

## 1. Minimal mathematical contract

Let F be the nonempty closed union of the admitted finite triangles for one
assigned CAD face, with its exact represented geometry and provenance. Let p_i
be one stored transformed scan coordinate and let a completed query supply
0 <= L_i <= d(p_i,F) <= U_i under its declared arithmetic model. Let g >= 0 be
the declared algorithmic distance gate, not a manufacturing tolerance.

Define three disjoint row states:

- DEFINITE_PROXIMITY: U_i <= g.
- DEFINITE_OUTSIDE: L_i > g.
- DISTANCE_UNRESOLVED: otherwise.

Proximity means there exists a point in F close enough to p_i. It does not
establish a unique closest point, normal, physical face, bend identity, material
side, pose, datum or manufacturing event. The current distance-only
`_support_masks` already has this trichotomy. The semantic change belongs at its
consumer, not in the triangle certificate or by erasing evidence flags.

For N > 0 original support rows, D definite-proximity rows and O definite-outside
rows, the true represented-coordinate in-gate fraction lies in
[D/N, (N-O)/N]. Compare outward enclosures of those fractions with the unchanged
coverage policy gamma; do not nearest-round a fraction across a threshold.

| Coverage evidence | Proposal interpretation only |
| --- | --- |
| Lower bound >= gamma | DISTANCE_SUPPORTED_CANDIDATE; correspondence may remain ambiguous. |
| Upper bound < gamma | DISTANCE_BOUND_REJECTED under this declared face assignment/model. |
| Neither | DISTANCE_COVERAGE_UNRESOLVED, not proof of exclusion or acceptance. |

All original support rows remain in N, including unresolved rows. Do not shrink
N to successful queries or use a CAD face count as observed bend count. N=0,
malformed evidence, incomplete primitive search, or exhausted budget is
UNAVAILABLE, not a zero-distance row. A policy may decline to refine an
unresolved candidate under a frozen finite budget, but must retain that omission
and cannot claim that every feasible pose was excluded. Candidate quotas must
not silently turn retained alternatives into a uniqueness certificate.

Keep the current COMPLETE-query requirement for the minimal successor. Although
a separately retained feasible witness could mathematically establish an upper
bound without exhaustive nearest-neighbor search, the present primitive clears
all outputs on an unresolved certificate or budget failure. This review does
not authorize salvaging an internal partial array or inventing a witness API.

## 2. Coordinate, uncertainty and geometry scopes must remain distinct

The surface API accepts points, not positional-error balls. V3's module contract
explicitly excludes transform-rounding, physical-uncertainty and native-CAD
certification. Its [L_i,U_i] encloses distance for stored transformed binary64
coordinates and the represented triangle hull. Merely carrying epsilon=.05mm
elsewhere does not make those intervals scanner-uncertainty enclosures.

For a separately justified latent point x_i satisfying ||x_i-p_i|| <= e_i,
distance to a fixed nonempty set is 1-Lipschitz. Proof: for every y in F,
||x_i-y|| <= ||x_i-p_i||+||p_i-y||; take the infimum and exchange x_i,p_i.
Consequently a valid latent-point distance enclosure is
[max(0,L_i-e_i), U_i+e_i], with outward arithmetic required in implementation.
Thus U_i <= g certifies represented-coordinate proximity, whereas U_i+e_i <= g
is a sufficient condition for proximity of every point in the declared ball.
An interval crossing g is unresolved; it does not prove a latent outlier exists.

These are conditional implications, not an instruction to enlarge g, lower
epsilon, or change the failed fixture. To derive e_i from raw-coordinate error,
an exact rigid transform preserves the norm bound, but stored matrix arithmetic
also needs its own error allowance. For a represented linear map R, a bound
||R||_2*epsilon_i plus a justified transform-evaluation error is sufficient for
the corresponding represented-map model. Neither approximate orthogonality nor
an SVD condition number alone supplies a physical calibration or pose-error bound.
Unknown pose error must be carried separately, not set to zero.

Likewise, transfer from represented F to a physical/native surface S requires
additional justified geometry-error assumptions. A two-sided Hausdorff bound h
would imply |d(x,F)-d(x,S)| <= h. No such physical h is established here; do not
manufacture it from tessellation settings or treat CAD nominal geometry as the
actual bent scan. Observed deviation is not automatically scanner noise.

## 3. Analytic counterexamples constrain the falsifiers

The following are exact hand-derived constructions, NOT executed tests or claims
about physical cases. Coordinates are in stipulated mm and g=1/2 throughout.

| Construction | Exact conclusion | Forbidden inference |
| --- | --- | --- |
| F=[-1,1]^2 at z=0, p=(0,0,49/100), e=1/20 | Observed distance49/100 passes g; latent distances range44/100 to54/100. | Observed proximity does not certify the whole error ball. |
| Same rectangle, p=(0,0,1/50) | Distance1/50, independent of the rectangle's diagonal. | Different triangle indices or uncertain optimizer coincidence cannot change the exact union distance. |
| Parallel squares [-1,1]^2 at z=+/-1/10, p=(0,0,0) | Distance1/10 with two distinct nearest points. | Distance-supported must not imply a unique sheet or signed deviation. |
| F=({0} x [-1,1] x [-1,1]) union ([-1,1] x {0} x [-1,1]), p=(1/10,1/10,0) | Distance1/10; nearest points differ and normals are perpendicular. | A retained proximity proposal does not authorize arbitrary fixed-target/normal selection. |
| Square frame: outer[-1,1]^2, removed open inner(-a,a)^2, p=(0,0,0) | Distance a: a=3/4 is outside, a=1/4 is inside the distance gate despite the hole. | A hole cannot be required to reject every point irrespective of its size and g; proximity does not fill or identify the hole. |

Same-union retessellation has invariant exact distance, but independently valid
floating-point enclosures need not have identical endpoints or threshold states
arbitrarily close to g. A legitimate falsifier requires declared separation from
the gate and adequate enclosure width, or accepts explicit numerical unresolved
status at the boundary. Work must also complete under the declared budget:
arbitrary subdivision can change work/cap outcomes without changing the union.
DISTANCE_UNRESOLVED from a complete interval crossing the gate is different from
UNAVAILABLE due to query work/tie limits or a width refusal; retain both actual
dispositions. Do not force exact floating equality or widen guards to make a
near-threshold test pass. Face-ID renaming also requires consistent
remapping of the assigned face subset; changing that subset changes F.

Normal/point ambiguity flags are engineering diagnostics. Surface lines561–562
explicitly say a false point-ambiguity flag is NOT a certificate of coincident
optimizers. A new type must not promote `not ambiguous` to certified uniqueness.
Coincident duplicated faces can have identical distances yet unresolved source
identity/material side. Preserve all native ownership and support lineage.

## 4. What a connected candidate would have to prove

The smallest meaningful advancement has two separate outcome ledgers:

1. Proposal recall/semantics: retain the known noisy correct-pose candidate with
   unchanged inputs/gates and distinguish true finite-distance exclusions from
   ambiguity. Independently constructed seam, hole, fold, sheet, near-gate,
   incomplete-query and row-lineage cases must check the typed evidence.
2. Connected usefulness: independently posed .05mm/free-circle cases must reach
   actual scan-derived measurements or principled quantity-specific abstention,
   with every step's support, selected/omitted hypotheses and resource work
   retained. More starts or a passing initializer alone is insufficient.

The current fixed-materialized-target optimizer may remain a proposal mechanism
only with its explicitly non-certified objective. Merely removing its ambiguity
mask would not certify descent of true finite-surface distance or correct
correspondence. Any set-valued or topology-aware replacement needs its own
review; do not conceal that change inside a proposal-only policy flag.

Two nonparallel-normal interior point-to-plane rows still leave axial translation
unobservable: with a=n1 cross n2, J=[p cross n,n] gives J[0,a]=n dot a=0 whenever
n lies in span(n1,n2). Other scan-supported normals or actual finite boundary
information may help, but extra axial seeds do not prove identifiability.
Distance support and all-false ambiguity flags do not remove this nullspace.
No inverse Hessian, finite seed list or isolated best-fit minimum is a stable
manufacturing datum or calibrated uncertainty certificate.

A successor workflow must separately bind proposal-support, optimization,
registration/observability and measurement policies to their exact versioned
evidence. The current equality of proposal/final policy identifiers is too
coarse for such a split. Existing V11/V3 source, frozen attempts, old score and
all output eligibility gates remain unchanged by this review.

Reviewed source SHA256 (current bytes, not a new execution freeze):

- `backend/cad_guided_triangle_surface_v3.py`:
  `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10`
- `backend/cad_guided_global_registration_v3.py`:
  `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0`
- `backend/cad_guided_plane_initialization_v11.py`:
  `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e`
- `backend/cad_guided_mesh_offline_v11.py`:
  `dbd60ebc8c2c159866c212506e46e35d56af17b196827ba5243c637c6dfe6dfa`

## Scope and next decision

This narrows the requested architecture consultation to a falsifiable question:
can staged evidence admission recover connected output without converting
proximity into measurement truth? The alternative robust/global proposal lane
remains a comparison candidate, not a presumed replacement. Its evaluation must
explain missing true proposals separately from falsely rejected ones.

The browser consultation has NOT been sent. Verified GPT-6 Astra Ultra awaits
the user's choice as a substitute for the requested Pro label. No engine
promotion, Gold11 rerun, success-criterion reduction or floor release follows
from this document. Cached F1 remains .4580152672; physical V3 remains zero-output
and separately formal-custody-blocked.
