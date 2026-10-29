EXECUTE_ONE

1. Architecture identity and new lock token

Architecture: V65_SHARED_FOLD_BASIN_OWNERSHIP_CERTIFICATE_V1
Short name: SFBOC_V1
New lock token: V65_SFBOC_V1_MECHANISM_LOCKED

Insertion point:

frozen route-local accepted lines→SFBOC_V1→the unchanged ordinary final NMS exactly once→renderer.

SFBOC_V1 is a positive-evidence, pairwise ownership consolidator. It may suppress one of several responses only after source geometry positively demonstrates that they occupy the same persistent transverse normal-turn basin. It never moves, averages, extends, shortens, or synthesizes a line.

This is the smallest justified intervention because the packet identifies close parallel ownership failure as dominant, while also showing that distance-only NMS would endanger genuine nearby bends and that the flat-face crossing defect is geometrically distinct. 

Pasted text

Scope boundary: SFBOC_V1 does not delete singleton flat-face crossings. R004-type unsupported lines therefore remain unchanged. Pairs at or above the frozen V64 boundary are also unchanged: SFBOC_V1 handles only close pairs below 10h, does not alter V64, and does not authorize a V64 run. The 10h partition is inherited from the already frozen V64 boundary rather than fitted to R001–R008. 

Pasted text

2. Exact eligible input population
2.1 Source population and scale

Let X be the finite source point population entering the frozen detector, after its existing deterministic bound to at most 200,000 points or mesh vertices.

Before any calculation:

Remove nonfinite coordinates.

Collapse exact coordinate duplicates.

Sort unique points lexicographically by their IEEE-754 binary64 coordinate triples. This order is the canonical point key and is independent of source vertex ordering.

For every x∈X, define the nearest-distinct-point distance

r(x)=
y∈X
y

=x
	​

min
	​

∥x−y∥
2
	​

.

Define the scan scale

h
0
	​

=median
x∈X
	​

r(x).

If ∣X∣<256, h
0
	​

≤0, or h
0
	​

 is nonfinite, SFBOC_V1 returns the unmodified baseline line population.

All linear constants below are in source-coordinate units; they are millimetres when the source scan is expressed in millimetres. Angles are in degrees.

2.2 Candidate line normalization

The eligible line population is every finite segment that has passed all frozen route-local gates and would otherwise enter the existing final NMS, irrespective of whether it came from the full-resolution lineage or the cylindrical-recovery lineage.

For line i, with endpoints a
i
	​

,b
i
	​

,

L
i
	​

=∥b
i
	​

−a
i
	​

∥
2
	​

,u
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

.

Canonicalize direction by swapping endpoints when the first nonzero component of u
i
	​

, in x,y,z order, is negative.

For pair i,j, align u
j
	​

 to u
i
	​

 and define

u=
	​

u
i
	​

+sign(u
i
⊤
	​

u
j
	​

)u
j
	​

	​

2
	​

u
i
	​

+sign(u
i
⊤
	​

u
j
	​

)u
j
	​

	​

.

Let [s
0
	​

,s
1
	​

] be the intersection of the two segments’ scalar projection intervals on u. Define

L
ov
	​

=s
1
	​

−s
0
	​

,ρ
ij
	​

=
min(L
i
	​

,L
j
	​

)
L
ov
	​

	​

,θ
ij
	​

=arccos(∣u
i
⊤
	​

u
j
	​

∣).

At 
s
ˉ
=(s
0
	​

+s
1
	​

)/2, evaluate each infinite line at the point having projection 
s
ˉ
 on u, obtaining p
i
	​

,p
j
	​

. Define

r
ij
	​

=(I−uu
⊤
)(p
j
	​

−p
i
	​

),d
ij
	​

=∥r
ij
	​

∥
2
	​

,q
ij
	​

=
d
ij
	​

r
ij
	​

	​

.
2.3 Pair-local scale

For a preliminary pair satisfying θ
ij
	​

≤2
∘
 and ρ
ij
	​

≥0.80, define its source corridor C
ij
	​

 as all x∈X that:

lie within 20h
0
	​

 of either segment’s overlap portion; and

have axial projection in [s
0
	​

−4h
0
	​

,s
1
	​

+4h
0
	​

].

If ∣C
ij
	​

∣<256, the pair is ineligible.

Otherwise,

h
ij
	​

=clip(median
x∈C
ij
	​

	​

r(x),0.5h
0
	​

,2h
0
	​

).

Write h=h
ij
	​

 for that pair.

2.4 Exact pair eligibility

A pair enters SFBOC_V1 only when all of the following hold:

θ
ij
	​

≤2.000000
∘
;
ρ
ij
	​

≥0.800000;
L
ov
	​

≥40h;
0.5h≤d
ij
	​

<10h;

and the frozen existing final-NMS predicate evaluates false for that pair.

No pair failing one of these conditions is modified by SFBOC_V1.

Pairs with d
ij
	​

≥10h, pairs already handled by ordinary NMS, nonparallel fragments, and low-overlap junction lines remain exactly on the baseline path.

3. Allowed and forbidden fields
Allowed for decisions	Forbidden for decisions
Bounded source coordinates and, when present, undirected triangle adjacency	CAD, drawings, design planes, manufacturing bend tables
Candidate endpoints and their source-only segment geometry	Human gold, labels, expected counts, transformed pseudo-truth
The unchanged existing-NMS pair predicate	R001–R008 evaluator outcomes or per-case tuning
Pair-local nearest-neighbour spacing	Filename, part identity, parent identity, family identity
Transient source-derived normal projectors	“Good scan” or “bad scan” status
Transient local graph connectivity and material chains	Benchmark feedback or production outcomes
Opaque candidate hashes for audit joins	Route-specific thresholds or route-priority rules
Resource counters and deterministic reason codes	Manual visual judgements during inference

A detector route identifier may be retained as an opaque audit field, but it may not enter eligibility, certificate, clustering, or representative ranking.

No V63 field, V64 score, V64 profile, or consumed-lane pseudo-truth value is permitted.

4. Deterministic source-geometry construction
4.1 Orientation-invariant source normals

Normals are transient and are never serialized.

For a triangle mesh, ignore degenerate faces. At vertex v, form

P
v
	​

=
∑
f∋v
	​

A
f
	​

∑
f∋v
	​

A
f
	​

n
f
	​

n
f
⊤
	​

	​

,

where A
f
	​

 is face area and n
f
	​

 is either face-normal orientation. Because n
f
	​

n
f
⊤
	​

=(−n
f
	​

)(−n
f
	​

)
⊤
, reversed triangle winding has no effect.

For a point-only source, use v and its 23 nearest distinct neighbours. Compute their covariance eigenvalues

λ
0
	​

≤λ
1
	​

≤λ
2
	​

.

The local normal is the eigenvector of λ
0
	​

. It is valid only if

λ
1
	​

>0and
λ
1
	​

λ
0
	​

	​

≤0.25.

The associated projector is P
v
	​

=n
v
	​

n
v
⊤
	​

.

All 3×3 eigensystems use a fixed-order binary64 Jacobi solver, fixed sweep count, ascending eigenvalue order, and lexicographic eigenvector tie-breaking. No library-dependent unordered reduction is allowed.

4.2 Nine fixed axial stations

For each eligible pair, use exactly nine stations:

s
k
	​

=s
0
	​

+
10
k
	​

L
ov
	​

,k=1,…,9.

At station k, evaluate the two lines at axial coordinate s
k
	​

, obtaining p
i,k
	​

,p
j,k
	​

, and define

c
k
	​

=
2
p
i,k
	​

+p
j,k
	​

	​

.

The pair direction u and transverse direction q=q
ij
	​

 remain fixed across all nine stations.

Select station nodes satisfying

∣(x−c
k
	​

)
⊤
u∣≤3h

and

	​

(I−uu
⊤
)(x−c
k
	​

)
	​

2
	​

≤24h.

A station with more than 4,096 nodes is INDETERMINATE; nodes are not thinned to force a result.

4.3 Transient material graph

For meshes, use undirected original mesh edges whose endpoints are selected and whose length is at most 4h.

For point-only data, use mutual eight-nearest-neighbour edges satisfying

∥x
a
	​

−x
b
	​

∥
2
	​

≤3h

and

min(∣(x
b
	​

−x
a
	​

)
⊤
n
a
	​

∣,∣(x
b
	​

−x
a
	​

)
⊤
n
b
	​

∣)≤0.75∥x
b
	​

−x
a
	​

∥
2
	​

.

A station with more than 32,768 admissible edges is INDETERMINATE.

For node x, let

t(x)=(x−c
k
	​

)
⊤
q.

The left and right tail seed sets are

S
L
	​

={x:−
2
d
	​

−10h≤t(x)≤−
2
d
	​

−6h},
S
R
	​

={x:
2
d
	​

+6h≤t(x)≤
2
d
	​

+10h}.

Each seed set must contain at least 12 nodes.

Run deterministic multi-source Dijkstra from S
L
	​

 to S
R
	​

 with edge cost

w(a,b)=∥x
b
	​

−x
a
	​

∥
2
	​

[1+4
∥x
b
	​

−x
a
	​

∥
2
2
	​

((x
b
	​

−x
a
	​

)
⊤
u)
2
	​

].

Equal-distance queue ties are resolved by canonical coordinate key.

The resulting path γ
k
	​

 is admissible only if:

length(γ
k
	​

)≤48h;
e
∑
	​

max(0,−Δt
e
	​

)≤2h;

and the nearest path distance from each of p
i,k
	​

 and p
j,k
	​

 is at most 3h.

Failure to construct this path is INDETERMINATE and cannot suppress a line.

4.4 Ordered normal-turn profile

Resample γ
k
	​

 at arc positions

z
m
	​

=min(mh,S),m=0,…,⌈S/h⌉,

where S is path length.

At each z
m
	​

, average path-node projectors with triangular weights

w
v
	​

=max(0,1−
1.5h
∣z
v
	​

−z
m
	​

∣
	​

).

The principal eigenvector of this average projector is n
m
	​

. Require its largest eigenvalue to be at least 0.85. Any invalid sample makes the station INDETERMINATE.

Canonicalize n
0
	​

 by making its first nonzero component positive. For m>0, negate n
m
	​

 when n
m−1
⊤
	​

n
m
	​

<0.

Define inner normal-turn increments, in degrees,

δ
m
	​

=
π
180
	​

arccos(clip(n
m
⊤
	​

n
m+1
	​

,−1,1)),m=0,…,M−1.

Let

T=
m=0
∑
M−1
	​

δ
m
	​

.

Define the fine turn density

g
0
(1)
	​

=
3
2δ
0
	​

+δ
1
	​

	​

,
g
m
(1)
	​

=
4
δ
m−1
	​

+2δ
m
	​

+δ
m+1
	​

	​

,1≤m≤M−2,
g
M−1
(1)
	​

=
3
δ
M−2
	​

+2δ
M−1
	​

	​

.

Define g
(2)
 by applying the identical three-tap rule once to g
(1)
.

No path vertices, normals, projectors, or profile arrays may survive the station evaluation.

5. Exact positive certificate and fail-closed behavior
5.1 Tail certificate

Let the left-tail samples be those within the first 4h of the path and the right-tail samples those within the final 4h.

For each tail, the largest eigenvalue of the mean projector must be at least 0.95.

The sum of δ
m
	​

 wholly inside either tail must satisfy

T
L
	​

≤4.000000
∘
,T
R
	​

≤4.000000
∘
.

Let n
L
	​

,n
R
	​

 be the two tail normals. Define

Θ=
π
180
	​

arccos(∣n
L
⊤
	​

n
R
	​

∣).

Require

Θ≥12.000000
∘

and

T≤1.25Θ+6.000000
∘
.

The first inequality positively establishes a scan-observable surface turn. The second rejects excessive return, oscillation, or multiple-turn travel.

5.2 Fold-axis alignment

For increments with δ
m
	​

≥0.35
∘
, define

a
m
	​

=
∥n
m
	​

×n
m+1
	​

∥
2
	​

n
m
	​

×n
m+1
	​

	​

.

Define turn-mass axis alignment

A=
∑
δ
m
	​

≥0.35
∘
	​

δ
m
	​

∑
δ
m
	​

≥0.35
∘
	​

δ
m
	​

1[∣a
m
⊤
	​

u∣≥cos20
∘
]
	​

.

Require

A≥0.850000.

Thus the observed normal rotation must predominantly occur about the proposed line direction.

5.3 Persistent single-basin certificate

For each scale r∈{1,2}, set

τ
r
	​

=max(0.35
∘
,0.025T).

Mark index m active when g
m
(r)
	​

≥τ
r
	​

. Apply exactly one one-bin closing operation: a single inactive index with active immediate neighbours is changed to active. Do not iterate this operation.

Let the active connected components be C
1
	​

,…,C
R
	​

. Component turn mass is calculated from the original δ
m
	​

, not from the smoothed density.

At each scale there must be one dominant component C
r
⋆
	​

 satisfying

T
mass(C
r
⋆
	​

)
	​

≥0.750000

and

T
∑
C

=C
r
⋆
	​

	​

mass(C)
	​

≤0.100000.

At both 0.50g
max
(r)
	​

 and 0.75g
max
(r)
	​

, the corresponding superlevel index set, after the same single one-bin closing, must have exactly one connected component.

Let I
1
	​

,I
2
	​

 be the arc intervals of the fine and coarse dominant components. Require

∣I
1
	​

∪I
2
	​

∣
∣I
1
	​

∩I
2
	​

∣
	​

≥0.600000.

This is the required persistence across the two fixed profile scales.

5.4 Both lines must occupy that basin

Map p
i,k
	​

 and p
j,k
	​

 to their nearest arc positions on γ
k
	​

, called z
i
	​

,z
j
	​

. Their Euclidean path distances must each be at most 3h.

Expand both dominant intervals by exactly 5h at each end, clipped to [0,S]. Require both z
i
	​

 and z
j
	​

 to lie inside both expanded intervals.

The unexpanded fine dominant interval must leave at least 4h of material-chain length on each outer side.

When all Sections 5.1–5.4 pass, the station outcome is SAME_BASIN.

5.5 Positive split veto

A station is SPLIT_BASIN when its source support and tail certificate are valid and either:

the fine-scale profile contains two components, each with at least 0.20T turn mass, separated by at least 2h of arc length below τ
1
	​

, with z
i
	​

,z
j
	​

 lying in different components expanded by 2h; or

T≥1.50Θ+10
∘
, two disjoint intervals each contain at least 0.15T, their separation is at least 2h, and z
i
	​

,z
j
	​

 lie in different intervals expanded by 2h.

All other outcomes, including missing tails, ambiguous normals, graph failure, holes, resource caps, weak turn, or inconsistent profiles, are INDETERMINATE.

5.6 Pair certificate

A pair receives a positive ownership edge only when

#{SAME_BASIN}≥7

over the nine exact stations and

#{SPLIT_BASIN}=0.

An indeterminate station supplies no positive or negative evidence.

A failed section, missing source support, sparse sampling, ambiguous normal field, or cap therefore produces no consolidation, never deletion. This directly respects the demonstrated V63 failure: missing profile construction cannot become negative evidence. 

Pasted text

6. Exact mutation rule and representative ranking
6.1 Complete-link consolidation

Construct an undirected graph whose vertices are eligible lines and whose edges are positive pair certificates.

Begin with singleton clusters. Two clusters A,B are mergeable only when:

every cross-pair i∈A,j∈B has a positive edge; and

∣A∪B∣≤8.

For positive pair i,j, define its deterministic quality tuple

Q
ij
	​

=(N
ij
same
	​

,−
e
ij
	​

,−
h
ij
	​

d
ij
	​

	​

),

where 
e
ij
	​

 is the median, over SAME_BASIN stations, of the mean normalized distance of both lines to the station’s turn centroid.

For mergeable clusters,

Q(A,B)=
i∈A,j∈B
min
	​

Q
ij
	​


using lexicographic ordering.

At each step merge the pair of clusters with maximal Q(A,B). Resolve ties by the lexicographically smallest sorted tuple of canonical candidate hashes. Repeat until no mergeable pair remains.

Any positive-edge connected component with more than eight lines is left wholly unchanged; it is not partially consolidated because of the cluster cap.

6.2 Representative ranking

For every SAME_BASIN station in a final cluster, compute the turn centroid on the material chain:

z
⋆
=
∑
m∈C
⋆
	​

g
m
(1)
	​

∑
m∈C
⋆
	​

g
m
(1)
	​

z
m+1/2
	​

	​

,

and its interpolated 3D point y
⋆
.

For candidate i, over all cluster core points y
⋆
, calculate:

n
i
	​

: number whose axial projection lies within segment i, expanded by 2h at each endpoint;

e
i
	​

: median of dist(y
⋆
,ℓ
i
	​

)/h;

e
90,i
	​

: deterministic nearest-rank 90th percentile of the same residuals.

Rank candidates by the ascending tuple

(−n
i
	​

,e
i
	​

,e
90,i
	​

,−L
i
	​

,canonical candidate hash).

The first candidate is retained without any geometric modification. Every other member of that cluster is suppressed before ordinary NMS.

No singleton is deleted. No cluster representative is averaged from its members. No endpoints are refitted. No flat-face candidate is deleted unless it belongs to a positively certified same-basin cluster—which the flat fixtures must prevent.

7. Compact audit schema

All decision calculations use unrounded binary64 values. Audit values are quantized only after decisions into fixed signed integers.

Scan record

architecture identity and lock token;

implementation SHA-256;

source count and unique-source count;

h
0
	​

×10
6
;

incoming candidate count;

preliminary and eligible pair counts;

no-op or cap reason;

primary/replay semantic hashes;

wall time, CPU time, peak RSS;

output and audit SHA-256.

Pair record

two 128-bit candidate-hash prefixes;

θ in microdegrees;

ρ×10
6
;

d/h×10
6
;

nine two-bit station states;

SAME_BASIN, SPLIT_BASIN, and INDETERMINATE counts;

quantized medians of Θ,T,A, dominant-mass ratio, and core residual;

positive-edge Boolean and fixed reason code.

Cluster record

ordered member hashes;

complete-link merge sequence;

representative hash;

representative rank tuple;

suppressed hashes.

The audit must not contain source points, vertices, triangles, graph nodes, graph edges, material chains, point indices, normals, projectors, profile bins, or station core coordinates.

Maximum canonical audit size is 512 KiB per scan. Exceeding that cap atomically discards every SFBOC mutation and returns the baseline output with AUDIT_CAP_NOOP.

8. CPU and resource caps

Source input: at most 200,000 points or vertices.

Incoming candidate cap: 64. Above 64, whole-scan SFBOC no-op.

Eligible pair cap: 128. Above 128, whole-scan SFBOC no-op.

Stations per pair: exactly 9.

Station nodes: at most 4,096.

Station edges: at most 32,768.

Cluster size: at most 8, with oversized connected components left unchanged.

SFBOC wall-time budget: 90 seconds per scan.

Total pipeline wall time: at most 300 seconds per scan; for an eight-case lane the nearest-rank P95 is therefore the maximum case time.

SFBOC private working memory: at most 8 GiB.

Whole-process peak RSS: strictly less than 16 GiB.

Up to eight fixed worker threads may process independent pair IDs. Each pair writes to a private buffer; results are joined only in canonical pair order. No shared floating-point accumulator or unordered parallel reduction is allowed.

Any wall, memory, graph, pair-count, or audit cap causes an atomic baseline no-op. A cap can never cause partial deletion.

9. Preregistered source-free scientific fixtures

All fixtures use h=1 source unit, deterministic triangulation or point sampling, line length 80h unless stated otherwise, and the unchanged ordinary-NMS predicate forced false for the tested close pair.

ID	Frozen fixture	Required result
SF01_SHARP_ONE_FOLD_THREE_ROUTES	One 70
∘
 dihedral fold; three 70h-span candidates at transverse positions −3h,+1h,+4h.	All three form one cluster; exactly two suppressed; at least 8/9 SAME_BASIN on every positive pair.
SF02_RESAMPLED_NOISY_ONE_FOLD	One 50
∘
 fold; alternating local spacing 0.7h–1.3h; deterministic normal noise σ=0.07h; two candidates 8h apart with overlap 0.82.	One positive edge and one suppression; primary/replay identical.
SF03_TWO_NEARBY_SAME_SIGN_FOLDS	Two 35
∘
 folds separated by a 6h planar web; one candidate on each fold.	No edge; at least 7/9 stations SPLIT_BASIN.
SF04_U_CHANNEL_PROTECTION	Two opposing 90
∘
 folds separated by an 8h web.	No edge; return-excess or two-lobe split on at least 7/9 stations.
SF05_RETURN_PROTECTION	A 135
∘
 return with a 5h intermediate face; candidates on entry and return axes, 7h apart.	No edge and no suppression.
SF06_HEM_DOUBLE_SKIN_PROTECTION	A 170
∘
 hem, 2h skin separation and 2h radius; candidates on the outer and returning surfaces.	No edge; point-cloud fallback graph must not shortcut between skins.
SF07_CLOSE_ROUNDED_RIBBON	One 80
∘
 cylindrical ribbon of radius 6h; two route responses arc-separated by 7h.	One positive edge and one suppression; one persistent broad turn component at both profile scales.
SF08_FLAT_PLATE_PERIMETER_DISTRACTOR	Flat plate with an open perimeter; two long parallel candidates 6h apart, one near the boundary.	Zero SAME_BASIN; no mutation.
SF09_FLAT_FACE_CROSSING_PLUS_TRUE_FOLD	One true fold and one parallel false line 7h onto the adjacent broad flat face.	No pair edge; both lines remain, explicitly demonstrating that singleton flat-face rejection is outside V65 scope.
SF10_MISSING_SUPPORT_NO_DELETE	SF01 with complete source removal around stations k=2,5,8.	Exactly six or fewer SAME_BASIN; no cluster and no deletion.
SF11_TWO_INTERRUPTED_STATIONS	SF01 with holes at exactly stations k=4,7, leaving seven intact stations.	Positive edge and consolidation, proving the preregistered 7/9 boundary rather than post-hoc rescue.
SF12_REPLAY_WINDING_AND_ORDER	SF01 executed as primary, replay, reversed triangle winding, reversed face order, and canonical random vertex permutation.	Bit-identical semantic output, station states, clusters, representative, and canonical audit hash in all forms.

All twelve must pass before any real-scan-derived execution. Fixture failure retires V65 without opening the prospective lane.

10. Fresh truth-unopened prospective protocol with native-resolution visual precommit
10.1 Lane

Create eight fresh cases: exactly two new deterministic variants of each exposed parent U01–U04. R001–R008 may not be reused.

Center each parent at its source centroid before the affine transformation.

Variant A uses

M
A
	​

=R
z
	​

(0.80
∘
)R
y
	​

(−0.45
∘
)R
x
	​

(0.35
∘
)
	​

1.003
0
0
	​

0.0015
0.997
0
	​

0
−0.0010
1.001
	​

	​

,

quadric resampling to

N
A
	​

=min(N
parent
	​

,175000),

and zero-mean normal-direction Gaussian noise with

σ
A
	​

=0.035 mm.

Variant B uses

M
B
	​

=R
z
	​

(−1.10
∘
)R
y
	​

(0.55
∘
)R
x
	​

(−0.65
∘
)
	​

0.996
0
0
	​

−0.0015
1.004
0
	​

0.0010
0.0015
0.999
	​

	​

,

quadric resampling to

N
B
	​

=min(N
parent
	​

,140000),

and

σ
B
	​

=0.070 mm.

Noise uses PCG64-DXSM with the first 64 bits of

SHA256(parent source bytes∥“V65-A”)

or V65-B as the seed.

The detector receives only the transformed source buffer under a random opaque case ID. It receives no parent ID, transformation matrix, seed, or variant label.

10.2 Freeze order

Before the first detector invocation, freeze and hash:

V65 implementation and constants;

all twelve fixture outputs;

transformation harness and RNG version;

parent source hashes;

eight transformed source hashes;

frozen V52 baseline bytes;

candidate and audit schemas;

native-resolution visual rubric;

evaluator program and transformed-reference generator, while keeping evaluator data closed.

10.3 Execution

Run, once each:

V52 primary and replay on all eight cases;

V52 plus SFBOC_V1 primary and replay on all eight cases.

No V63 execution, V64 execution, threshold adjustment, case replacement, or second prospective tranche is permitted.

10.4 Prediction-only visual precommit

Before opening transformed references, render for every case:

one full 4096-pixel source overlay;

one baseline/V65 comparison;

one crop for every suppression cluster showing all original members and the retained representative;

a separate overlay containing every baseline line that V65 left unchanged.

The manual audit must answer, for each suppression:

Do every suppressed line and its representative visibly occupy one physical fold transition?

Is there any visible intermediate face, return, channel wall, hem skin, or second structural fold?

Does the retained unchanged line remain localizable and render-admissible?

Did V65 suppress a line crossing a flat face?

Did any graph path apparently bridge a hole, gap, or opposite skin?

Did the transformation pair become more count-stable without visibly losing shorter true bends?

Any ambiguous visual judgement is a safety failure, not a pass.

Freeze the completed visual rulings and their contact-sheet hashes before opening evaluator pseudo-truth.

10.5 One-time evaluator opening

Only after the visual audit is frozen, open the transformed frozen V51 reference lines and score them with the unchanged evaluator exactly once.

These remain development pseudo-references from exposed parents. They are not human gold, independent-part evidence, or promotion evidence.

11. Hard retain/retire gates
Gate A — analytic validity

Required before the real-derived lane:

12/12 fixtures produce their exact expected mutation outcome;

all replay, reversed-winding, and reordered-source hashes match;

no forbidden field is present in an execution trace;

no fixture audit contains serialized source geometry.

Failure token:

V65_SFBOC_V1_TERMINAL_RETIRE_FIXTURE

No real-derived execution follows.

Gate B — determinism and resources

On the eight prospective cases:

8/8 primary/replay semantic output hashes match;

8/8 audit hashes match;

8/8 jobs complete successfully;

no scan enters resource-cap or audit-cap no-op;

every scan is at most 300 seconds;

nearest-rank P95 is at most 300 seconds;

every process peak RSS is strictly below 16 GiB.

Any failure retires V65.

Gate C — visual safety

Required:

every suppressed line is manually classified as the same physical fold as its representative;

zero distinct structural bends are consolidated;

zero return, channel, hem, or opposing-skin pairs are consolidated;

zero flat-face crossing lines are suppressed;

zero retained representatives are visibly displaced from the physical transition;

zero suppression is accepted on ambiguous visual evidence.

One violation is terminal.

Flat-face crossings that remain unchanged are not a V65 failure because singleton flat-support rejection is explicitly outside this architecture.

Gate D — material mechanism effect

Required:

at least four predictions are suppressed;

suppressions occur in at least three of eight variants;

suppressions span at least two distinct parent morphologies;

every one of those suppressions passes Gate C.

Fewer than four correct suppressions is terminal INEFFECTIVE, even if otherwise safe.

Gate E — one-time pseudo-reference accounting

Relative to frozen V52 on the same eight inputs:

TP
V65
	​

=TP
V52
	​

,
FN
V65
	​

=FN
V52
	​

,
FP
V65
	​

≤FP
V52
	​

−4,

and

FP
V65
	​

≤0.80FP
V52
	​

.

Additionally:

pseudo-reference exact-case count may not decrease;

no transformed reference may lose its only matched prediction;

V65 cannot create a new prediction, so any observed added prediction is an implementation failure.

Gate F — resampling stability

For parent p, let

D
p
	​

=∣count(p,A)−count(p,B)∣.

Require

p=1
∑
4
	​

D
p,V65
	​

≤
p=1
∑
4
	​

D
p,V52
	​

.

An increase is terminal. Equality is permitted only when Gates D and E pass.

Final tokens

All gates pass:

V65_SFBOC_V1_RESEARCH_RETAIN

Any prospective gate fails:

V65_SFBOC_V1_TERMINAL_RETIRE

There is no threshold rescue, same-identity revision, selective rerun, or case substitution after invocation.

Passing retains one research-only precision primitive. It does not open runtime/main, production, or a promotion claim.

12. Need for a new physical scan or independent annotation

The newly received physical part does not need to be scanned or independently annotated before this V65 execution.

A fresh transformation-only lane is sufficient to falsify the narrow mechanism:

whether close multi-route responses can be positively shown to share one source-observable fold basin;

whether nearby genuine folds remain separate;

whether the result is stable under new resampling and noise;

whether the certificate is deterministic and bounded.

That lane is explicitly not sufficient to establish unseen-part F1 or generalization, because all eight cases inherit four exposed parent morphologies.

Before any broader retain-to-promotion decision, at least one genuinely fresh physical part must be scanned under a sealed acquisition protocol and independently annotated without access to V65 predictions. The final promotion claim still requires the locked 30-scan, 15-family benchmark and its stated family and operational gates.

13. Research grounding and separate novel assumptions
Established research

Surface-feature research establishes that curvature or normal-field behaviour can produce salient ridge and valley responses. Ohtake, Belyaev, and Seidel extract ridge-valley lines using first- and second-order curvature derivatives estimated through multilevel surface fitting; that supports examining a persistent local turn field, but it does not establish when two detector responses own the same physical fold. 
Heriot-Watt Research Portal

Pauly, Keiser, and Gross use local PCA and multiple neighbourhood scales for line-feature extraction on point-sampled surfaces, reporting improved reliability and noise robustness from multiscale analysis. Their method is explicitly semi-automatic and parameter-dependent, so it supports the use of fixed multiscale persistence tests but does not validate the numerical constants above. 
CGL ETHZ
+1

Pottmann and co-authors emphasize that differential invariants are sensitive to noise and small perturbations, while neighbourhood-integrated quantities can have useful stability properties. That supports aggregating normal projectors and turn mass over material neighbourhoods rather than allowing a single local curvature sample to delete a line. 
Guibas Lab

Robust multi-model research also argues against treating independently generated geometric hypotheses greedily. Isack and Boykov formulate multi-model fitting as a global labeling problem balancing geometric fit, spatial coherence, and model cost, rather than repeatedly accepting the largest local consensus. That supports explicit ownership and complete-link compatibility instead of a generic broad pairwise NMS. 
Springer

Cazals and Pouget establish polynomial-fitting methods for estimating differential properties from sampled curves and surfaces. This supports the general feasibility of source-derived local differential evidence, while not proving SFBOC’s material-chain or ownership semantics. 
ScienceDirect
+1

Novel, unvalidated Sherman-specific assumptions

Two close route responses to one physical fold will occupy one axis-aligned, single-lobed normal-turn basin at at least seven of nine transverse stations.

Two genuine nearby folds resolvable at the source sampling scale will produce either two persistent turn lobes or excess return variation before they can satisfy the same-basin certificate.

The deterministic local material path will follow the observed sheet surface rather than shortcut through a thin opposite skin, hole, or resampling bridge.

The inherited d<10h close-pair partition, the 2
∘
 angle gate, 0.80 overlap, tail and turn constants, 7/9 station rule, and multiscale component ratios will transfer across the real LiDAR-derived morphology without case-specific adjustment.

The existing candidate nearest the consensus turn centroid is the safest unchanged representative.

Correct close-pair consolidation will reduce extra predictions without deleting the sole match to a real structural bend.

These assumptions are not established by the cited literature or by the consumed R001–R008 pseudo-truth. They are the exact hypotheses that the source-free fixtures and single fresh transformation-only execution must falsify.

