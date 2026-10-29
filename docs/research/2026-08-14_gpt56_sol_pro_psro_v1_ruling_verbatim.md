1. RULING

Authorize exactly one V18 intervention: PAIRED_SKIN_RIBBON_OWNERSHIP_V1 (PSRO_V1).

PSRO_V1 is an append-only ownership selector for native full-resolution meshes. It does not repair or inherit V17. In particular, it does not read V17’s axial-run, endpoint-support, p90, span ratio, transition, normal-angle, balance, minimum-side-support, coherence, score, or rescue decisions.

The intervention asks a different question:

Does this rejected scan_mesh_adjacency line belong to a nondegenerate, constant-thickness four-face ribbon representing two flange skins joined by the same fold on both sides of the material?

The existing compact structures are not sufficient by themselves. They encode line support, normals, transitions, parents, and adjacency, but not the directed first-opposite-surface relation required to decide whether a side is a principal sheet skin or a lateral/thickness/boundary face. PSRO_V1 therefore adds exactly one bounded source generator: a fixed-count reciprocal through-thickness ray probe, emitted while the native mesh and existing V11 BVH remain available.

No full midsurface is reconstructed.

2. RESEARCH BASIS

Midsurface abstraction treats a thin solid through relationships between opposite surface regions, and Rezayat explicitly argued that successful abstraction and feature recognition require geometric and topological information together. 
ScienceDirect
 Lee, Nam, and Park subsequently used ray casting to identify face pairs and retained those pairings in a graph used to connect midsurface patches. 
IEEE Xplore
+1
 Gupta and Gurumoorthy represented sheet-metal bends and walls as characteristic constant-thickness arrangements of faces, with compound deformation features represented by a graph rather than by an isolated edge response. 
Manipal Academy

PSRO_V1 adapts those established ideas narrowly:

Established machinery: reciprocal opposite-face pairing, normal opposition, local thickness consistency, and attributed face-adjacency motifs.

Sherman-specific hypothesis: a true missing bend in this lane should instantiate a four-owner ribbon

A−c−B,
A
ˉ
−m−
B
ˉ
,

with reciprocal thickness pairs A↔
A
ˉ
 and B↔
B
ˉ
. A rim, opening, or thickness edge instead leaves an owner unpaired or degenerates into A↔B. A typical seam lacks the conjugate opposite-skin transition m.

This is not a claim that manufacturing history is always recoverable from surface geometry. A perfectly ground, constant-thickness miter weld can be geometrically indistinguishable from a sharp fold; that case remains the explicit expected failure mode below.

3. CAUSAL FEATURE
Paired-skin ribbon ownership

For a rejected candidate c, let A and B be its two incident plane-owner parents from the already-emitted side summaries.

PSRO_V1 seeks:

a unique reciprocal opposite owner 
A
ˉ
 across the material thickness from A;

a unique reciprocal opposite owner 
B
ˉ
 across the material thickness from B;

four distinct owners:

A

=B,
A
ˉ

=
B
ˉ
,
A
ˉ
∈
/
{A,B},
B
ˉ
∈
/
{A,B};

an existing compact transition parent m connecting exactly {
A
ˉ
,
B
ˉ
}, aligned with and axially covering c.

That motif establishes that c joins two principal sheet regions on one skin and that the same material transition exists on the opposite skin.

It distinguishes the target cases as follows:

Silhouette or open mesh boundary: one or both probe rays have no unique reciprocal opposite hit.

Opening/perimeter edge: an incident owner is a lateral wall or the pairing collapses to an adjacent skin owner rather than yielding four distinct owners.

Thickness face: its ray thickness is not thin relative to its support width, is not stable along the line, or has no conjugate transition.

Cycle-closing rim: the candidate joins owners that are themselves an opposite-skin pair, producing the prohibited degeneracy 
A
ˉ
=B or 
B
ˉ
=A.

Already-explained corner: the conjugate parent is already represented by a V11 incumbent, or the candidate loses incumbent-first frozen V11 NMS.

Typical welded/formed seam: it may have two broad faces, but normally lacks a reciprocal, axially corresponding transition between their opposite skins.

The decisive information is therefore material-side ownership across thickness, not additional evidence along the candidate axis.

4. EXACT SPECIFICATION
4.1 Eligible pool and insertion point

PSRO_V1 runs only in the native full-resolution mesh lane.

Its eligible set is:

source family exactly scan_mesh_adjacency;

rejected by frozen V11;

finite candidate line, finite support interval, and valid frozen V11 source construction;

not already represented by a V11 accepted source parent.

No V17 field or decision is consulted.

The fixed-ray record is generated at the end of native scan_mesh_adjacency candidate construction, while the existing native mesh, face-to-mode mapping, transient side-support membership, and V11 triangle BVH are still available.

Selection occurs:

after frozen V11 has finalized its incumbents and reject pool;

before the native-mesh route result is serialized;

before V16 chooses the already-frozen non-native routes.

All V11 incumbents remain byte-identical and retain priority.

4.2 Canonical candidate coordinates

For candidate c:

canonicalize its unit direction d
c
	​

 by making the largest-magnitude component positive, with ties resolved x,y,z;

project the frozen support endpoints onto d
c
	​

, obtaining interval [a
c
	​

,b
c
	​

];

let L
c
	​

=b
c
	​

−a
c
	​

>0;

use the five fixed stations

Q={
16
1
	​

,
4
1
	​

,
2
1
	​

,
4
3
	​

,
16
15
	​

}.

Stations 1/16, 1/2, and 15/16 are mandatory.

The two incident owners A,B are the maximum-support plane-parent IDs in the existing side summaries. Ties use the lower unsigned parent ID. Missing, identical, or non-plane owners reject the candidate.

4.3 Deterministic probe origin

For each owner side k∈{A,B} and station q:

Let t
q
	​

=a
c
	​

+qL
c
	​

.

Consider only transient side-support atoms assigned to owner k with

∣⟨x
i
	​

,d
c
	​

⟩−t
q
	​

∣≤L
c
	​

/32.

Select the atom by the lexicographic key

(∣⟨x
i
	​

,d
c
	​

⟩−t
q
	​

∣,−dist(x
i
	​

,c),atom_id).

Use the corresponding native support face and its frozen oriented normal as the ray origin face.

An empty station is invalid. No replacement from another station is permitted.

The side width W
k
	​

 is the nearest-rank 90th percentile of the positive perpendicular distances from the candidate line among that side’s existing support atoms. Nonfinite or nonpositive W
k
	​

 rejects.

4.4 Reciprocal first-hit pairing

Let s
mesh
	​

 be the frozen native-mesh scale already used by the mesh-adjacency producer. Set

ϵ=max(0.5s
mesh
	​

,8ulp(max(1,∥x∥))).

From each selected face centroid x, cast rays in the fixed order −n,+n, starting at x+ϵr.

For each ray:

ignore the origin triangle and its one-ring;

select the first intersection;

intersection ties within eight ulps of ray parameter t use the smallest triangle ID;

the hit must map to an existing plane-owner parent;

the frozen oriented normals must satisfy

n
origin
	​

⋅n
hit
	​

≤−cos15
∘
.

Exactly one of the two directions must yield a valid opposite hit. Zero or two valid directions reject that station.

For each side, the station mask must:

include 1/16, 1/2, and 15/16;

contain at least four of five stations;

map every valid station to the same opposite owner parent.

At the center station, cast the corresponding inward ray from the hit face back toward the material. Its first valid hit must return to the original owner, and

min(h
forward
	​

,h
return
	​

)
max(h
forward
	​

,h
return
	​

)
	​

≤1.20.

This reciprocal test is mandatory on both sides.

4.5 Thickness and nondegeneracy gates

For each side k, let h
k,j
	​

 be its valid first-hit distances and 
h
~
k
	​

 their deterministic median.

Require:

min
j
	​

h
k,j
	​

max
j
	​

h
k,j
	​

	​

≤1.35,
h
~
k
	​

≤0.25W
k
	​

,

and across the two sides,

min(
h
~
A
	​

,
h
~
B
	​

)
max(
h
~
A
	​

,
h
~
B
	​

)
	​

≤1.25.

Let the opposite owners be 
A
ˉ
,
B
ˉ
. Hard reject unless

∣{A,B,
A
ˉ
,
B
ˉ
}∣=4.

In particular, reject whenever

A
ˉ
=Bor
B
ˉ
=A.

That is the explicit thickness-edge, opening-rim, and paired-boundary degeneracy gate.

4.6 Conjugate transition gate

Search the existing bounded transition-parent identifier graph for parents incident to exactly the unordered owner pair

{
A
ˉ
,
B
ˉ
}.

A parent m is admissible only if it has finite line geometry and satisfies all of:

arccos(∣d
c
	​

⋅d
m
	​

∣)≤5
∘
;
min(∣I
c
	​

∣,∣I
m
	​

∣)
∣I
c
	​

∩I
m
	​

∣
	​

≥0.80,

where I
m
	​

 is projected onto d
c
	​

;

the three mandatory station coordinates of c lie within I
m
	​

 expanded by 2s
mesh
	​

;

dist(c,m)≤2.5max(
h
~
A
	​

,
h
~
B
	​

);
∣θ
c
	​

−θ
m
	​

∣≤10
∘
,

where each θ is the frozen unordered two-owner normal angle.

If multiple parents qualify, choose the lexicographically smallest tuple

(direction error,−overlap,line distance/max(
h
~
A
	​

,
h
~
B
	​

),turn-angle difference,parent ID).

No weighted score is permitted.

Reject c if the selected m is already the source parent of a V11 incumbent.

4.7 Ranking and duplicate behavior

A passing candidate receives the ascending lexicographic rank

(
	​

−min(popcountM
A
	​

,popcountM
B
	​

),
−overlap(c,m),
max(R
A
	​

,R
B
	​

),
R
AB
	​

,
mate direction error,
mate turn-angle difference,
dist(c,m)/max(
h
~
A
	​

,
h
~
B
	​

),
−L
c
	​

,
candidate ID),
	​


where R
k
	​

=maxh
k,j
	​

/minh
k,j
	​

 and
R
AB
	​

=max(
h
~
A
	​

,
h
~
B
	​

)/min(
h
~
A
	​

,
h
~
B
	​

).

The canonical ribbon key is:

form thickness pairs

P
A
	​

=(min(A,
A
ˉ
),max(A,
A
ˉ
)),P
B
	​

=(min(B,
B
ˉ
),max(B,
B
ˉ
));

sort P
A
	​

,P
B
	​

 lexicographically;

append the sorted pair of transition-parent IDs

(min(p
c
	​

,p
m
	​

),max(p
c
	​

,p
m
	​

)).

Candidates with the same ribbon key are symmetric duplicates. Retain only the best-ranked one.

Then:

initialize NMS with all frozen V11 incumbents in their original order;

traverse retained PSRO candidates in rank order;

apply the frozen V11 geometric duplicate/NMS predicate without altered constants;

incumbents always win;

among new candidates, earlier rank wins;

append surviving candidates after the incumbents.

The original candidate line geometry is emitted. No averaging, recentering, extension, endpoint repair, or mate-derived axis synthesis is allowed.

There is no top-k, scan-level rescue count, family cap, or score threshold.

4.8 Schema and resource bound

Each candidate emits one fixed-size paired_skin_owner_probe_v1 record containing:

candidate, near-owner, opposite-owner, and mate-parent IDs;

two five-bit station masks;

ten binary64 thickness values;

two side widths;

reciprocal flags;

mate direction error, overlap, line distance, and turn-angle difference;

canonical ribbon key;

reject-bit mask.

Nonfinite values are forbidden; serialized -0.0 is canonicalized to +0.0.

Maximum serialized size: 224 bytes per candidate plus a 64-byte scan header.

No probe origins, mesh vertices, point samples, triangle lists, hit coordinates, or source clouds are serialized. Audit rays are regenerated from candidate and mesh IDs.

Runtime work is at most:

ten primary BVH rays;

two reciprocal center rays;

one bounded identifier-graph lookup

per eligible candidate.

The existing native-mesh BVH and face-to-mode mapping must be reused; a second mesh copy or second BVH is forbidden.

5. FALSIFICATION GATES

All constants and code are frozen and hashed before the first real scan. Every failure below is terminal for PSRO_V1; there is no threshold alteration, alternative station set, rerun rescue, or V18b.

5.1 Source-free construction gates

Exactly one frozen suite:

finite-thickness folded ribbon with paired inner/outer transitions: exactly one retained ribbon;

top-skin-to-thickness perimeter edge: zero;

opening rim with a lateral wall: zero;

two broad faces joined by a single miter seam without an opposite-skin conjugate transition: zero;

symmetric candidate/mate presentation: one canonical ribbon, independent of producer ordering;

three rigid transforms and one candidate/atom permutation: byte-identical semantic output after inverse transformation;

whole-component face-orientation reversal: identical decision and canonical owner mapping.

Any failure stops before U02.

5.2 Mandatory label-closed U02–U05 gate

Run U02 first.

Hard requirement:

zero PSRO append on U02;

SMA1 and SMA2 both have retain=false;

the V16 axis-result and route-result bytes remain identical;

the new research-only probe cache is stored separately and does not alter retained output bytes.

If either SMA1 or SMA2 survives, issue PAIRED_SKIN_RIBBON_OWNERSHIP_V1_STOP immediately. C017 and development gold remain unopened.

Then U03–U05 must also remain byte-identical to V16.

No particular U02 reject bit is prescribed; prescribing the reason after seeing the execution would be post-hoc tuning. The emitted reject bits and native audit must nevertheless make the failure mechanism inspectable.

5.3 C017 source-only activation gate

With gold still closed, C017 must produce at least one V11-rejected scan_mesh_adjacency candidate that:

passes every PSRO hard gate;

has four distinct reciprocal owner IDs;

has mandatory endpoint and center probe coverage on both sides;

has an admissible conjugate transition;

survives canonical duplicate collapse and incumbent-first V11 NMS.

Zero activation is terminal. Activation count is not used at inference and does not become a scan-level cap.

Before opening gold, hash a native-resolution audit for every C017 activation showing:

candidate c and mate m;

the four owner regions;

all ten primary rays and two reciprocal rays;

endpoint and center stations;

the candidate and mate support intervals;

nearby mesh boundaries and all V11 incumbents.

The audit must not show a ray pairing to an unrelated nearby panel, a lateral wall treated as a principal skin, or a candidate lying visibly on a perimeter/opening chain. Any such contradiction is a source-only STOP.

5.4 One-shot 35-scan development gate

Only after the source gates pass may the frozen development gold be opened once.

Relative to 132 TP / 60 FP / 44 FN, all of the following are required:

ΔTP≥+1;

ΔFP=0;

ΔFN≤−1;

F1 strictly greater than 0.717391;

exact cases at least 4/35;

no scan-level F1 decrease;

no family-level F1 decrease;

all V16 incumbent records remain byte-identical and in the same order;

U02–U05 remain exact.

Thus the weakest acceptable aggregate result is at least 133 TP / 60 FP / 43 FN. Any added false positive is terminal because the intervention’s sole causal purpose is ownership discrimination.

5.5 Fresh held-out gate

Run once on a prospectively locked family-held-out tranche with paired V16 and PSRO outputs.

Require:

at least one held-out PSRO activation before gold;

after gold, ΔTP≥+1;

ΔFP=0;

ΔFN≤−1;

no scan or family F1 decrease;

no exact-case loss;

all baseline incumbent axes retained byte-identically.

Zero source activation, zero added TP after activation, or any added FP is terminal for this architecture identity.

5.6 Determinism, resource, and visual gates

Across development and held-out:

two independent CPU replays must produce byte-identical axis outputs and probe caches;

peak process-tree RSS must remain below 16 GiB;

P95 end-to-end latency must remain at most 300 seconds;

cache size must respect the 224-byte-per-candidate bound;

no source-point or face-list serialization;

no runtime/main or production modification.

Every appended line receives a native-resolution four-view audit, not a sampled audit. Hard visual rejection occurs if a retained line is visibly:

a silhouette, opening rim, or thickness wall;

paired across empty space to a separate nearby panel;

one side of an already-explained corner;

a seam without a continuous opposite-skin fold transition;

or a boundary closure masquerading as an interior ribbon.

No constants may change in response to that audit.

6. EXPECTED FAILURE MODE

The most likely surviving false positive is a constant-thickness, double-skin mitered or welded corner whose outer and inner seams are both present and geometrically aligned.

Such a corner can instantiate the same four-owner motif as a sharp bend:

A↔
A
ˉ
,B↔
B
ˉ
,A−c−B,
A
ˉ
−m−
B
ˉ
.

The mandatory native audit exposes the ordinary version by showing two straight seam chains, miter termination or weld-bead geometry, and no finite-width continuous bend strip between the flange owners. If the seam is perfectly ground and its scanned surface geometry is exactly identical to a sharp folded sheet, manufacturing history is not identifiable under blind scan-only inference. That is the terminal scientific boundary of PSRO_V1, not a reason to add another threshold.

AUTHORIZE_PAIRED_SKIN_RIBBON_OWNERSHIP_V1

Sources
