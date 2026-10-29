EXECUTE_ONE

Architecture: V66_TRANSVERSE_FACE_TURN_OWNERSHIP_CERTIFICATE_V1 (TFTOC_V1)
Mechanism lock: V66_TFTOC_V1_MECHANISM_LOCKED
Source-free suite: V66_TFTOC_SOURCE_FREE_FIXTURES_V1
Fixture lock: V66_TFTOC_FIXTURES_LOCKED

TFTOC_V1 is a pre-NMS, suppression-only ownership certificate. It replaces V65’s crease-averaged vertex projector with exact transverse sections of the piecewise-linear source mesh and face-local normal turns. It creates no candidates, changes no V52 score, and cannot promote a rejected route.

1. Why this mechanism does not inherit the V65 contradiction

For a section plane perpendicular to an ideal extruded fold axis u, each section segment lies in exactly one incident face. Let t
1
	​

,t
2
	​

 be the two section tangents and n
1
	​

,n
2
	​

 the separately retained face normals. For an ideal dihedral of angle θ,

∣n
r
	​

⋅(u×t
r
	​

)∣=1,α=arccos(n
1
	​

⋅n
2
	​

)=θ.

Therefore, at the mandatory 70
∘
 sharp fold,

c
1
	​

=c
2
	​

=1≥0.5,α=70
∘
,

with no averaged crease normal and no projector-eigenvalue gate. The V65 value

λ
max
	​

=
2
1+∣cos70
∘
∣
	​

≈0.671

is never computed and cannot invalidate TFTOC.

The causal hypothesis is narrow: close parallel V52 routes generated around one physical fold should repeatedly attach to the same turn event in shared transverse sections; routes belonging to nearby real folds should attach to different events and produce explicit negative evidence.

2. Exact insertion point and mutation boundary

The frozen order is:

V52 route generation→all frozen V52 route-local gates→frozen V52 composition→TFTOC_V1→one unchanged ordinary final NMS
	​


TFTOC consumes exactly the candidate list that V52 would otherwise submit to final NMS.

It may only remove candidates belonging to certified all-pairs-positive ownership cliques. It must not:

alter candidate endpoints, axes, scores, route-local fields, or V52 rank keys;

create or reposition a candidate;

reconsider a route rejected by V52;

call ordinary NMS internally;

invoke ordinary final NMS more than once.

If TFTOC performs an atomic no-op, the original V52 pre-NMS list is passed unchanged to the ordinary final NMS.

3. Inputs, source population, deterministic scale, and normalization
3.1 Permitted inputs

For each source scan:

M=(V,F,E)

is the source triangle mesh, including vertex coordinates, triangle incidence, and undirected edge adjacency.

For each V52 pre-NMS candidate c
i
	​

, TFTOC receives only:

c
i
	​

=(a
i
	​

,b
i
	​

,H
i
	​

,K
i
	​

),

where:

a
i
	​

,b
i
	​

∈R
3
 are segment endpoints;

H
i
	​

 is the opaque route hash;

K
i
	​

 is the existing immutable V52 better-first total-order key.

No filename, part identity, family identity, expected count, score truth, CAD, label, drawing, good/bad status, or historical case field is available.

A point-only source with no triangle adjacency causes a global atomic no-op with reason NO_TRIANGLE_ADJACENCY; it does not cause candidate rejection.

3.2 Pair-local source scale

For a candidate pair (i,j), define

L
i
	​

=∥b
i
	​

−a
i
	​

∥,L
j
	​

=∥b
j
	​

−a
j
	​

∥,L
min
	​

=min(L
i
	​

,L
j
	​

).

Construct a coarse capsule equal to the union of the two candidate segments dilated by 0.15L
min
	​

, with the common axial interval extended by 0.05L
min
	​

 at both ends.

Collect every unique undirected source-mesh edge whose midpoint lies in that capsule and whose length ℓ satisfies

2
−20
L
min
	​

≤ℓ≤0.05L
min
	​

.

Sort the edges by the tuple

(ℓ,canonical endpoint-coordinate hash)

and define h
ij
	​

 as the lower median edge length:

h
ij
	​

=ℓ
⌊(m−1)/2⌋
	​

.

The pair is scale-valid only when

m≥256,40≤
h
ij
	​

L
min
	​

	​

≤400.

All subsequent pair-local coordinates are normalized by

x
′
=
h
ij
	​

x−r
ij
	​

	​

,r
ij
	​

=
4
a
i
	​

+b
i
	​

+a
j
	​

+b
j
	​

	​

.

Every spatial constant below is therefore expressed in units of h
ij
	​

. The scale is similarity-equivariant and depends only on source geometry.

4. Exact pair eligibility

Let

u
i
	​

=
L
i
	​

b
i
	​

−a
i
	​

	​

,u
j
	​

=
L
j
	​

b
j
	​

−a
j
	​

	​

.

Sign-align u
j
	​

 to u
i
	​

:

u
~
j
	​

={
u
j
	​

,
−u
j
	​

,
	​

u
i
	​

⋅u
j
	​

≥0,
u
i
	​

⋅u
j
	​

<0,
	​


and define the unoriented pair axis

u
ij
	​

=
∥u
i
	​

+
u
~
j
	​

∥
u
i
	​

+
u
~
j
	​

	​

.

A pair is eligible if and only if all of the following hold:

∣u
i
	​

⋅u
j
	​

∣≥cos3
∘
=0.9986295347545738,
min(L
i
	​

,L
j
	​

)
max(L
i
	​

,L
j
	​

)
	​

≤1.5,
O
ij
	​

≥0.75L
min
	​

,

where O
ij
	​

 is the overlap of the two segment projection intervals on u
ij
	​

, and

z∈{z
min
	​

,z
max
	​

}
max
	​

	​

(I−u
ij
	​

u
ij
T
	​

)(x
i
	​

(z)−x
j
	​

(z))
	​

≤8h
ij
	​

.

Here x
i
	​

(z) and x
j
	​

(z) are the intersections of the two candidate lines with the axial level z, evaluated at the two endpoints of the common overlap. Because the transverse difference is affine in z, checking both endpoints fixes the maximum over the interval.

Ineligible pairs generate neither positive nor negative edges.

5. Face-local transverse-section representation
5.1 Shared station planes

For each eligible pair, place nine shared section planes at

z
k
	​

=z
min
	​

+
10
k
	​

O
ij
	​

,k=1,…,9,
Π
k
	​

={x:u
ij
	​

⋅(x−r
ij
	​

)=z
k
	​

}.

Reversing either candidate orientation reverses station order but leaves the physical set of nine planes unchanged.

Let x
i,k
	​

 and x
j,k
	​

 be the candidate-line intersections with Π
k
	​

, and let

q
k
	​

=
2
x
i,k
	​

+x
j,k
	​

	​

.

Only source triangles whose plane-intersection segment comes within 16h
ij
	​

 of q
k
	​

 are retained.

5.2 Exact piecewise-linear section graph

Calculations use IEEE-754 binary64, round-to-nearest ties-to-even, with fast-math and fused-operation contraction disabled. The normalized plane-zero tolerance is

ε
Π
	​

=2
−30
.

A queried triangle is admissible only if:

it is nondegenerate;

no vertex has normalized signed plane distance with magnitude ≤ε
Π
	​

;

it is not coplanar with Π
k
	​

;

it has exactly two mesh edges crossing the plane.

A queried plane-through-vertex, plane-through-edge, coplanar-face, or nonfinite condition makes that station INDETERMINATE, not positive.

Each crossed undirected mesh edge is a section-graph node. Each intersected triangle contributes one segment joining its two crossed-edge nodes. Components touching either candidate anchor are valid only if every node has degree at most two. A degree greater than two, duplicated section edge, or unresolved nonmanifold branch makes the station indeterminate.

Open chains and simple cycles are permitted. Ordering is determined from geometric node keys, not face or vertex indices.

5.3 Candidate-to-component attachment

For each candidate intersection x
i,k
	​

, compute its closest point on every admissible section component.

Candidate i has a unique component attachment only if

d
i,1
	​

≤1.0h
ij
	​


and either no second component exists or

d
i,2
	​

−d
i,1
	​

≥0.5h
ij
	​

.

Failure is AMBIGUOUS_COMPONENT or NO_COMPONENT_SUPPORT.

5.4 Winding-invariant face normals

For an ordered section segment with unit tangent t
ℓ
	​

 and arbitrary-winding unit face normal 
n
~
ℓ
	​

, define

g
ℓ
	​

=
n
~
ℓ
	​

⋅(u
ij
	​

×t
ℓ
	​

).

The segment is valid only if

∣g
ℓ
	​

∣≥0.5.

Its canonical face-local normal is

n
ℓ
	​

=sgn(g
ℓ
	​

)
n
~
ℓ
	​

.

Reversing face winding changes both 
n
~
ℓ
	​

 and sgn(g
ℓ
	​

), leaving n
ℓ
	​

 unchanged. Reversing the polyline or u
ij
	​

 flips all canonical normals together, leaving all unsigned event evidence unchanged.

5.5 Aligned face-turn increments

At every transition between consecutive section segments, define

α
ℓ
	​

=atan2(∥n
ℓ
	​

×n
ℓ+1
	​

∥,n
ℓ
	​

⋅n
ℓ+1
	​

)∈[0,π].

If

α
ℓ
	​

<0.05
∘
,

set δ
ℓ
	​

=0.

Otherwise define axial alignment

q
ℓ
	​

=
∥n
ℓ
	​

×n
ℓ+1
	​

∥
∣u
ij
	​

⋅(n
ℓ
	​

×n
ℓ+1
	​

)∣
	​

.

The transition is admissible only when

q
ℓ
	​

≥0.5.

An above-floor turn with q
ℓ
	​

<0.5 makes that component indeterminate for the station. For an admissible transition,

δ
ℓ
	​

=sgn(u
ij
	​

⋅(n
ℓ
	​

×n
ℓ+1
	​

))α
ℓ
	​

.

Thus the magnitude comes from the two separate face normals, while the sign is used only to measure coherent versus oscillatory turning. Reversing axis or traversal negates every δ
ℓ
	​

, which does not change the event definitions below.

6. Turn-event extraction and candidate ownership

Let r
ℓ
	​

 be section arclength at turn increment δ
ℓ
	​

.

6.1 Active turn support

For every turn location, use the arclength window

W
ℓ
	​

={m:∣r
m
	​

−r
ℓ
	​

∣≤1.0h
ij
	​

}.

Define

S
ℓ
	​

=
	​

m∈W
ℓ
	​

∑
	​

δ
m
	​

	​

,A
ℓ
	​

=
m∈W
ℓ
	​

∑
	​

∣δ
m
	​

∣.

A turn location is active exactly when

S
ℓ
	​

≥8
∘

and

A
ℓ
	​

S
ℓ
	​

	​

≥0.80.

Active locations separated by at most 1.25h
ij
	​

 in section arclength form one seed interval. Extend each seed interval to include all turn locations within 1.0h
ij
	​

 of either end. Merge two extended intervals only if their gap is at most 0.5h
ij
	​

.

6.2 Ownable turn event

For an extended interval E, define

Θ
E
	​

=
	​

ℓ∈E
∑
	​

δ
ℓ
	​

	​

,TV
E
	​

=
ℓ∈E
∑
	​

∣δ
ℓ
	​

∣,
C
E
	​

=
TV
E
	​

Θ
E
	​

	​

,W
E
	​

=r
max,E
	​

−r
min,E
	​

.

It is an ownable event if and only if:

20
∘
≤Θ
E
	​

≤145
∘
,
C
E
	​

≥0.85,
W
E
	​

≤6h
ij
	​

,

and both adjacent 2h
ij
	​

 tails exist and satisfy

ℓ∈T
E
−
	​

∑
	​

∣δ
ℓ
	​

∣≤6
∘
,
ℓ∈T
E
+
	​

∑
	​

∣δ
ℓ
	​

∣≤6
∘
.

For an open component, either event tail reaching the component boundary before 2h
ij
	​

 invalidates the event. For a cycle, the tails are cyclic and must not encounter another event seed. Every section segment intersecting the event plus its two tails must have length at most

1.25h
ij
	​

.

The turn-weighted center is

r
E
	​

=
∑
ℓ∈E
	​

∣δ
ℓ
	​

∣
∑
ℓ∈E
	​

∣δ
ℓ
	​

∣r
ℓ
	​

	​

.

The 145
∘
 ceiling intentionally makes return/hem turns fail closed rather than treating opposing skins as an ordinary duplicate basin.

6.3 Candidate ownership

Let r
i,k
	​

 be candidate i’s attached arclength on its section component. Its distance to event E is the shortest component arclength distance

d
i,k
	​

(E)=dist
γ
	​

(r
i,k
	​

,r
E
	​

),

using cyclic distance for a closed component.

Candidate i owns its nearest event only when

d
i,k
	​

(E
1
	​

)≤4.5h
ij
	​


and either no second event exists or

d
i,k
	​

(E
2
	​

)−d
i,k
	​

(E
1
	​

)≥1.0h
ij
	​

.

Otherwise its ownership is indeterminate. No nearest-event guessing is permitted across an ambiguity boundary.

7. Positive evidence, negative evidence, and clustering

Each pair station has exactly one outcome:

SAME_EVENT: both candidates uniquely attach to the same section component and own the same event;

SPLIT_EVENT: both uniquely own valid but different events;

SPLIT_COMPONENT: both uniquely own events on different components;

INDETERMINATE: every other outcome, with a frozen reason code.

Let

S_{ij}=\#\text{SAME_EVENT},\qquad D_{ij}=\#(\text{SPLIT_EVENT or SPLIT_COMPONENT}).

The pair is:

POSITIVE⟺S
ij
	​

≥7 ∧ D
ij
	​

=0,
NEGATIVE⟺D
ij
	​

≥1,

and UNKNOWN otherwise.

A single split station therefore blocks ownership even if the other eight stations agree.

Construct the graph containing only positive edges. For each connected component G with at least two candidates, suppression is permitted only when G is a complete graph and every unordered pair in G was eligible and positive.

Thus:

positive transitive chains that are not all-pairs certified perform no mutation;

an unknown or negative cross-pair prevents the entire connected component from mutating;

broad distance-only clustering is impossible.

This all-pairs-positive rule is stricter than ordinary complete-link clustering.

8. Representative ranking and exact mutation

For candidate c in a suppressible clique, define:

v
c
	​

=number of SAME_EVENT station assignments involving c,

and

m
c
	​

=median{
h
pq
	​

d
c,k
	​

(E)
	​

:all SAME_EVENT stations in clique pairs containing c}.

The representative is the unique minimum of the following lexicographic key:

R(c)=(−v
c
	​

,m
c
	​

,K
c
	​

,−L
c
	​

,H
c
	​

),

where K
c
	​

 is compared using the existing V52 better-first ordering and H
c
	​

 is ascending byte order.

TFTOC removes every nonrepresentative in the clique. It does not average or refit the representative. The resulting candidate list then enters the single unchanged ordinary NMS.

For a true fold route and an adjacent flat-face route assigned to the same event, the route closer in section arclength to the turn center wins before V52 score is consulted.

9. Determinism, compact audit, and resource contract
9.1 Deterministic execution

Eligible pairs are sorted by (minH
i
	​

,H
j
	​

,maxH
i
	​

,H
j
	​

).

Mesh edges, faces, graph nodes, components, and events use canonical coordinate-based keys.

Pair jobs may use at most 16 workers, but each job is isolated and final results are reduced in sorted pair order.

No unordered hash-table traversal may affect a decision.

All comparisons are inclusive exactly as written.

Degree constants are mathematically multiplied by π/180.

No random seed or stochastic sampling exists.

Primary and replay must produce byte-identical canonical semantic JSON after excluding only wall-clock and RSS telemetry.

9.2 Compact audit

The semantic artifact contains:

architecture and lock tokens;

constants SHA-256;

canonical order- and winding-invariant mesh topology hash;

V52 candidate-list digest;

eligible-pair keys;

pair scale;

nine station labels and reason codes;

S
ij
	​

, D
ij
	​

, pair class;

compact integer-quantized event angle, span, coherence, and anchor-distance summaries;

clique membership;

representative key;

suppression list;

pre-NMS output digest and final mutation digest.

Station vectors are canonicalized as the lexicographically smaller of forward and reversed station order.

No source vertex array, source face array, section polyline, or source-point cache is serialized.

9.3 Hard caps
Resource	Frozen cap
Source vertices	220,000
Source triangles	440,000
V52 pre-NMS candidates	128
Eligible candidate pairs	512
Pair-station jobs	4,608
Queried faces per station	8,192
Section graph nodes per station	4,096
Section components per station	64
Total face-plane tests	8,000,000
Total emitted section segments	1,000,000
Incremental TFTOC heap	2 GiB
TFTOC wall time	120 s
Complete V52–TFTOC–NMS wall time	300 s

No pair, face, candidate, or graph truncation is allowed. Exceeding any cap discards every proposed suppression and returns the original V52 pre-NMS list as RESOURCE_ATOMIC_NOOP.

In the one-shot research protocol, any such no-op terminally retires V66; it does not authorize a smaller replacement input or rerun.

10. Frozen source-free fixture suite
10.1 Common generator

All fixtures use h=1 and are generated from a two-dimensional cross-section profile extruded along

z∈[0,80h].

Candidate segments have length 80h and are parallel to the extrusion axis.

Profile segments are at most h. Axial mesh layers are at

z
m
	​

=−0.37h+mh,

so the nine section planes at 8h,16h,…,72h do not pass through fixture vertices. Each profile strip is triangulated with alternating diagonals. The generator must independently verify that every eligible fixture pair computes

h
ij
	​

=h.

Every fixture candidate is marked as having passed the frozen route-local gates. Ordinary NMS is forced to retain everything, so fixture mutations are attributable only to TFTOC.

10.2 Mandatory fixtures
Fixture	Geometry and candidates	Frozen expected result
SF01_SHARP_SINGLE_FOLD_DUPLICATES	One 70
∘
 sharp dihedral; routes at profile arclengths −3h,+1h,+4h from the crease	Three eligible pairs; each 9 SAME, 0 SPLIT; one 3-clique; keep +1h; exactly two suppressions
SF02_ROUNDED_SINGLE_FOLD_DUPLICATES	Tangent-continuous 70
∘
 circular bend, radius 3h; routes at −3h,+1h,+4h from arc midpoint	Same pair and mutation result as SF01
SF03_TWO_NEARBY_SAME_SIGN_FOLDS	Two sharp +55
∘
 turns with centers 6h apart; two routes per turn at offsets −1h,+0.25h	Two positive 2-cliques; all four cross-fold pairs SPLIT at all nine stations; exactly two survivors, one per fold
SF04_U_CHANNEL_OPPOSING_FOLDS	Sharp +80
∘
 and −80
∘
 turns 6h apart; duplicate routes as in SF03	Two positive cliques; no cross-fold merge; exactly two survivors
SF05_RETURN_HEM_DOUBLE_SKIN	Continuous 170
∘
 circular return, radius 1.5h; candidates on opposing legs within 4.5h arclength of the return	Event rejected because 170
∘
>145
∘
; zero positive edges and no mutation
SF06_FLAT_PLATE_PERIMETER_DISTRACTOR	Closed 20h×0.8h plate profile; two routes on the long skin within 4.5h of one perimeter corner	Corner events fail the two-sided 2h tail condition because the thickness side is 0.8h; no mutation
SF07_TRUE_FOLD_PLUS_ADJACENT_FLAT_FALSE_LINE	One 70
∘
 fold; true route at +0.25h, flat-face route at +3h	9 SAME; one positive pair; flat route suppressed; true route retained by centeredness
SF08_MISSING_SUPPORT_FAIL_CLOSED	One 70
∘
 fold; routes at −1h,+0.25h; delete every face intersecting the first three station planes	Exactly 6 SAME, 0 SPLIT, 3 INDETERMINATE; pair UNKNOWN; no mutation
SF09_EXACT_SEVEN_STATION_BOUNDARY	Same geometry; delete every face intersecting station planes 1 and 9 only	Exactly 7 SAME, 0 SPLIT, 2 INDETERMINATE; pair POSITIVE; exactly one suppression
SF10_LONG_FLAT_FACE_CROSSING	One valid fold-axis candidate plus a long flat-face line crossing it at 90
∘
	Pair ineligible by the 3
∘
 parallelism gate; no mutation; this confirms that TFTOC does not claim to solve isolated R004-type crossings
SF11_ORDER_AND_WINDING_INVARIANCE	SF01 replayed unchanged, with all face windings reversed, with SHA-256 face permutation, and with SHA-256 vertex permutation plus remapped faces	Replay semantic artifact byte-identical; all reordered variants have identical pair classes, representative, suppression set, and mutation digest

Any fixture mismatch, including one wrong reason count, terminally retires V66 before U01–U04 are run.

10.3 Explicit threshold consistency

For SF01:

∣n
r
	​

⋅(u×t
r
	​

)∣=1,q=1,Θ=70
∘
,C=1,W=0.

Thus every validity gate is satisfied.

For SF02, the circular-turn support length is

3h⋅
180
70π
	​

=3.665191429188092h<6h.

Even at an arc endpoint, a 1h half-window contains at least

70
∘
3.665191429188092h
1h
	​

=19.098593
∘
>8
∘

of coherent turn. The full event has Θ=70
∘
, C=1, and all routes are at most 4h<4.5h from its center.

For SF03 and SF04, the event-center separation is 6h, exceeding the 1.25h seed-link distance and leaving 2h flat tails on both inward sides. The events therefore remain distinct despite every relevant route pair being within the 8h eligibility radius.

For SF05:

Θ=170
∘
>145
∘
,

so the mandatory no-mutation result is analytically forced.

11. Fresh U01–U04 prospective protocol

The real lane identity is:

V66_TFTOC_U01_U04_PROSPECTIVE_ONE_SHOT_V1

R001–R008 remain unopened and inaccessible.

11.1 Precommit

Before any TFTOC execution:

Hash and freeze the V52 executable, TFTOC executable, fixture generator, constants artifact, visual renderer, pseudo-scoring code, and all U01–U04 source bytes.

Replace U01–U04 names with opaque source hashes and process them in ascending hash order.

Run and freeze V52 pre-NMS candidates on each original source without opening pseudo-truth.

Generate all deterministic variants before the prospective lane begins. No variant may be replaced afterward.

11.2 Variants

For each U source, run TFTOC on:

Identity primary and replay. Same mesh and same frozen V52 pre-NMS candidates.

Similarity variant.

Q(x,y,z)=(z,−x,−y),
x
′
=
4
5
	​

Qx+(8D,−4D,2D),

where D is the original mesh bounding-box diagonal. Apply the identical transform to candidate endpoints. Pair classes and suppression mapping must match identity after candidate-hash correspondence.

Order/winding variant. Reorder vertices and faces by canonical SHA-256 keys, remap all incidence, and reverse the winding of every face whose canonical face hash has low bit one. The semantic artifact, excluding raw-byte input hash, must match identity.

Exact PL-refinement variant. Select the first

min(4096,⌊
128
∣F∣
	​

⌋)

faces in canonical hash order. Replace each selected triangle by three coplanar triangles joined to its exact barycenter. Candidate geometry is unchanged. This changes sampling without changing the piecewise-linear surface. Pair classes, cliques, representatives, and suppression mapping must match identity.

V52 is not rerun on the metamorphic variants; the frozen identity candidate list is transformed with the geometry. This isolates TFTOC invariance from unrelated V52 candidate-generation changes.

Any primary/replay mismatch, variant mutation mismatch, cap no-op, crash, or incomplete artifact terminally retires V66. No rescue rerun is permitted.

12. Native-resolution visual audit before pseudo-truth

Pseudo-truth remains sealed until the following visual packet and adjudication file are complete and hashed.

For all four identity cases, render from the complete source mesh with no vertex, face, or candidate decimation at a minimum of 4096×4096 pixels:

raw source mesh in six fixed orthographic views and one fixed oblique view;

frozen V52 pre-NMS candidates with opaque IDs;

TFTOC ownership cliques, representatives, and suppressed routes;

V52 final output and V66 final output side by side;

for every positive pair, all nine exact section polylines, candidate anchors, ownable event interval, turn-weighted center, ownership distances, and station result;

close views looking along the candidate axis, normal to each incident face, and along the face-normal bisector;

a final-difference view showing every line present in one final output but not the other.

Every changed case must receive a sealed visual classification before pseudo-truth opens:

CONFIRMED_SAME_PHYSICAL_FOLD;

DISTINCT_PHYSICAL_BEND_FALSE_SUPPRESSION;

AMBIGUOUS_SUPPRESSION;

PERIMETER_OR_FLAT_ROUTE_NOT_RESOLVED.

A suppression classified as distinct or ambiguous terminally retires V66. Visual findings may not change a constant, input, candidate, threshold, or execution order.

13. Hard retain/retire gates

V66 is retained as a research-only sidecar only if every gate below passes.

Source-free and determinism gates

All SF01–SF11 outcomes match exactly.

Every U identity primary/replay semantic artifact is byte-identical.

Every deterministic variant preserves the required suppression semantics.

No resource atomic no-op occurs.

Peak total RSS remains below 16 GiB and total wall time remains at or below 300 seconds.

Visual safety gates

At least three suppressions are visually confirmed as duplicate ownership.

Those suppressions occur across at least two of U01–U04.

Every retained representative is visually on the same fold and at least as fold-centered as each route it suppresses.

Zero distinct-bend suppressions.

Zero ambiguous suppressions.

No final-output difference attributable to altered ordinary-NMS interaction removes a visually distinct bend.

Pseudo-truth gates

Only after the visual adjudication hash is frozen, open the existing pseudo-truth once and score both frozen V52 and V66 outputs.

For every individual U case r,

TP
r
V66
	​

≥TP
r
V52
	​

,FN
r
V66
	​

≤FN
r
V52
	​

.

Across U01–U04,

FP
V66
≤FP
V52
−max(2,⌈0.20FP
V52
⌉),
F1
V66
≥F1
V52
+0.03,
Recall
V66
≥Recall
V52
,

and the number of exact-count cases must increase by at least one, with no previously exact case becoming inexact.

Failure of any gate terminally retires V66. There is no threshold adjustment, fixture alteration, input replacement, rerun rescue, or V66b.

Passing retains TFTOC only as a frozen research intervention. Gold, runtime/main, and production remain closed.

14. Leakage audit and status of the new physical part

Every inference decision derives only from:

source coordinates;

source triangle adjacency;

face-local geometry;

V52 candidate segments and frozen rank keys;

opaque hashes.

The local scale is computed from source-mesh edge lengths. Fixtures supply no learned parameters, and U01–U04 pseudo-truth remains sealed through architecture execution and visual precommit. The source-free suite is a contradiction, invariance, and safety test—not evidence of real accuracy.

The newly received physical part is not required now. U01–U04 provide the prospective truth-unopened falsification lane. The new part is reserved for one later unseen-part run only after every V66 retain gate passes and the retained executable and constants are re-frozen.

The representation is grounded in primary geometry work showing why normals mixed across sharp-feature neighborhoods can be corrupted, why profile or section evidence and mesh connectivity are relevant, and why piecewise-linear mesh curvature should be treated through local geometric transitions rather than a smooth vertex normal. 
Georgia Tech Faculty
+1
 The all-pairs-positive clique is a deliberately stricter Sherman-specific use of the must-link/cannot-link principle; constrained-clustering work likewise notes the safety role of complete-link behavior for cannot-link constraints. 
creddy.net
