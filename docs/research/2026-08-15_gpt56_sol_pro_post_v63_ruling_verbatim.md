Worked for 16m 20s

RULING: EXECUTE_ONE

1. Architecture identity

Architecture: V64_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE_V1
Short name: PSROC_V1
Lock token: V64_PSROC_MECHANISM_LOCKED

PSROC_V1 is a narrowly scoped pairwise certificate for identifying two V52 detections as interior responses on one continuously curved surface ribbon. It is not a general line-quality classifier.

2. Intervention type

Deletion-only.

The input line set is the frozen V52-lineage final output before V63. PSROC_V1 may delete both members of a certified pair. It may not:

append detections;

move, shorten, extend, merge, split, or refit a line;

invoke additional NMS;

reconsider any singleton;

consume V63 output, V63 profiles, or V63 predicates.

Every retained V52 line remains byte-for-byte geometrically unchanged.

3. Exact eligible input population

Only unordered pairs of final V52 line instances from the same scan are eligible.

Let D be the source-mesh bounding-box diagonal. Let h be the scan sampling scale defined in Section 5.

Each V52 line retains its frozen deterministic ordinal. The lower ordinal is pair member A; the higher is B. Directions are oriented so that the direction of B has nonnegative dot product with that of A.

A pair enters geometric eligibility only when all of the following hold:

Each segment length is at least 24h.

Their acute direction difference is at most 7.5
∘
.

Their overlap along the normalized mean direction is at least:

max(20h,0.65min(L
A
	​

,L
B
	​

)).

At seven overlap stations, defined at fractions

{1/8,2/8,3/8,4/8,5/8,6/8,7/8},

the transverse separation can be evaluated from both finite segments.

The median station separation w satisfies:

10h≤w≤0.35L
overlap
	​

.

The station-separation median absolute deviation divided by w is at most 0.15.

After orienting all transverse vectors from A toward B, their maximum angular deviation from their normalized componentwise-median direction is at most 12
∘
.

Parallelism, length, curvature, cylindrical appearance, or V52 origin creates eligibility only. None is deletion evidence.

One-to-one pair ownership

For every geometrically eligible partner j of line i, define the fixed ranking tuple:

(
L
overlap,ij
	​

w
ij
	​

	​

,−
min(L
i
	​

,L
j
	​

)
L
overlap,ij
	​

	​

,ordinal(j)).

A line chooses the lexicographically smallest tuple. A pair is evaluated only when the choice is mutual. This produces disjoint candidate pairs and prevents cascade deletion.

If more than 32 mutual pairs exist in one scan, the scan is returned unchanged and the run fails the resource gate.

4. Allowed and forbidden source fields
Allowed

source triangle-mesh vertex XYZ positions;

source triangle indices and mesh adjacency derived from those indices;

frozen V52 segment endpoints, ordinals, and source-scan association;

deterministic quantities computed solely from those fields;

the scan’s source hash and baseline-output hash for provenance.

Forbidden

V5R3B truth or scored overlays;

V63 profile sheets, status flags, missing-support states, coverage values, output-cap values, or predicates;

CAD, labels, bend counts, expected counts, family identity, filenames, part numbers, benchmark scores, or operator annotations;

calibrated view identity, camera identity, source-view count, intensities, colors, acquisition confidence, or dropout labels;

runtime/main or production state;

serialized source vertices, triangles, section paths, or normal arrays in the compact cache.

The executable shall receive V63 paths on an explicit filesystem deny list. PSROC_V1 computes a new pairwise triangle-section graph directly from the source mesh.

5. Deterministic geometry and graph construction
5.1 Source scale

Enumerate unique finite mesh edges in sorted (minv,maxv) order. Discard zero-length edges. Remove the nearest-rank bottom 5% and top 5% by length. Define h as the nearest-rank median of the remainder.

If fewer than 1,024 valid edges remain, or if h<2
−40
D, return the V52 line set unchanged.

5.2 Mesh orientation

Build face adjacency only through edges having exactly two incident faces.

For each face component:

choose the lowest face ordinal as root;

perform breadth-first traversal with neighboring faces visited by ascending face ordinal;

flip each newly visited face when necessary so that the common edge is oppositely directed in the two faces.

Any orientation conflict marks that component unusable for certification. It cannot supply negative evidence.

5.3 Common pair frame

For each mutual pair:

t
^
: normalized mean line direction, oriented so its first nonzero Cartesian component is positive;

at each station, a
s
	​

 and b
s
	​

: points on A and B at the same coordinate along 
t
^
;

c
^
s
	​

: normalized projection of b
s
	​

−a
s
	​

 onto 
t
^
⊥
;

b
^
s
	​

=
t
^
×
c
^
s
	​

;

m
s
	​

=(a
s
	​

+b
s
	​

)/2.

5.4 Exact transverse section

At each of the seven stations, intersect the oriented mesh with the plane through

m
s
	​

+2
−40
D
t
^

whose normal is 
t
^
.

A deterministic BVH is built once per scan:

split on the longest bounding-box axis;

axis ties resolve x, then y, then z;

faces are partitioned at the stable median of centroid coordinate;

centroid ties resolve by face ordinal.

Only triangles intersecting the oriented prism below are queried:

∣(x−m
s
	​

)⋅
t
^
∣≤3h,
∣(x−m
s
	​

)⋅
c
^
s
	​

∣≤w
s
	​

+4h,
∣(x−m
s
	​

)⋅
b
^
s
	​

∣≤w
s
	​

+4h.

A triangle that straddles the plane contributes one section segment. Graph nodes are crossed mesh edges; graph edges are section segments identified by source-face ordinal.

A station is geometrically usable only when:

no relevant triangle is coplanar within 2
−48
D;

the selected section component is an open degree-at-most-two polyline;

it belongs to one consistently oriented manifold component;

the plane intersections of A and B project onto the same section component within 2h;

the selected component minimizes, in order, summed squared projection distance, maximum projection distance, then minimum incident face ordinal;

the arc between projected A and B has length g
s
	​

 satisfying

10h≤g
s
	​

≤1.75w
s
	​

;

the component extends at least 0.25g
s
	​

 beyond A, away from B, and at least 0.25g
s
	​

 beyond B, away from A.

Cycles, branches, ambiguous coplanar intersections, disconnected skins, insufficient flank extension, or absent sections make the station MISSING, never negative.

5.5 Normal transport sequence

Take the full section arc from 0.25g
s
	​

 before A to 0.25g
s
	​

 after B. Its length is 1.5g
s
	​

.

Sample it at 61 equal arc-length positions j=0,…,60. Consequently:

A is at index 10;

B is at index 50.

At each sample, form an oriented normal by averaging the normals of section faces within an along-section radius of 1.5h. Each contribution is weighted by its intersected section length multiplied by the triangular kernel

K(d)=max(0,1−
1.5h
∣d∣
	​

).

Project the resulting normal into 
t
^
⊥
. Every sample must retain projected magnitude of at least 0.75 before normalization. Otherwise the station is MISSING.

Define an unwrapped normal phase ϕ
j
	​

 around 
t
^
, with phase differences wrapped into (−π,π]. All computations use IEEE-754 binary64.

6. Fixed geometric constants and smooth-ribbon certificate

For inner indices j=10,…,50, define

δ
j
	​

=ϕ
j+1
	​

−ϕ
j
	​

,j=10,…,49,
S=
j=10
∑
49
	​

δ
j
	​

,T=∣S∣,σ=sign(S).

A geometrically usable station is POSITIVE only if every condition below passes.

Measure	Frozen requirement
Net normal rotation	18
∘
≤T≤135
∘

Opposite-sign transport	∑max(0,−σδ
j
	​

)≤0.08T
Largest single inner increment	(\max
10–90% rotation support width	at least 0.55 of the A-to-B arc
25–75% rotation support width	at least 0.30 of the A-to-B arc
Largest six-interval positive-rotation mass	at most 0.40 of total positive rotation
Tangent jump at A	J
A
	​

≤max(6
∘
,0.16T)
Tangent jump at B	J
B
	​

≤max(6
∘
,0.16T)

Here positive rotation increments are

p
j
	​

=max(0,σδ
j
	​

).

The cumulative 10%, 25%, 75%, and 90% locations use linear interpolation within equal arc-length intervals.

Candidate-centered discontinuities are:

J
A
	​

=∣median(ϕ
7
	​

,ϕ
8
	​

,ϕ
9
	​

)−median(ϕ
11
	​

,ϕ
12
	​

,ϕ
13
	​

)∣,
J
B
	​

=∣median(ϕ
47
	​

,ϕ
48
	​

,ϕ
49
	​

)−median(ϕ
51
	​

,ϕ
52
	​

,ϕ
53
	​

)∣.

A fully constructed station that fails any listed condition is CONTRADICTORY, not MISSING.

Pair-level certificate

A mutual pair is certified only when:

at least five of seven stations are POSITIVE;

no station is CONTRADICTORY;

all POSITIVE stations have the same rotation sign;

the maximum T
s
	​

 divided by the minimum T
s
	​

 is at most 1.50;

the maximum g
s
	​

 divided by the minimum g
s
	​

 is at most 1.25;

axial normal-transport phase is consistent:

For each POSITIVE station, normalize its cumulative positive rotation between A and B to F
s
	​

(q)∈[0,1]. Evaluate it at

q={0,1/8,2/8,…,1}.

Form the pointwise nearest-rank median profile. Every station must have maximum absolute deviation from that profile of at most 0.18.

All numerical comparisons are inclusive. Pair and station processing order is ascending baseline ordinal. No data-dependent threshold, percentile threshold, threshold search, or adaptive rescue is permitted.

7. Fail-closed behavior

MISSING stations contribute zero positive votes.

Fewer than five POSITIVE stations keeps both lines.

One CONTRADICTORY station keeps both lines.

Any nonfinite value keeps both lines.

Any orientation conflict, section ambiguity, disconnected-skin ambiguity, or inadequate normal projection keeps both lines.

Missing surface, sparse support, failed section construction, or insufficient extension can never itself cause deletion.

A triangle-query cap, pair cap, watchdog, allocation failure, or serialization failure returns the entire scan’s original V52 output unchanged. No partial mutation is retained.

Resource or serialization failure also fails the run-level retain gate.

This explicitly prevents the V63 failure mode: absence of evidence is never converted into evidence against a line.

8. Precise mutation rule

After evaluating all mutual pairs against the unchanged V52 baseline:

delete both members of every certified pair simultaneously;

perform no cascading re-pairing;

perform no second pass;

perform no NMS;

preserve original ordering among surviving lines;

preserve every surviving line’s original coordinates and metadata.

The audit reason is exactly:

DELETED_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE

No one-member deletion is permitted. The certificate asserts that neither candidate is a localized structural transition and that both are responses on one shared smooth ribbon.

9. Primary/replay and resource bounds
Determinism

single-threaded pair and section evaluation within each scan;

scans processed in ascending source-hash order;

fixed binary64 operations and sorted graph traversal;

canonical JSON with sorted keys;

all floating audit metrics serialized using hexadecimal floating-point representation;

NaN and infinity forbidden;

primary and replay must produce identical deletion IDs, statuses, metrics, hashes, and surviving geometry.

Bounds

maximum source vertices: existing approximately 200,000-point bound;

maximum mutual pairs: 32 per scan;

maximum queried triangles: 50,000 per station;

maximum section constructions: 32×7=224 per scan;

additional peak RSS: at most 1.5 GiB;

total process peak RSS: below 16 GiB;

PSROC_V1 wall-time watchdog: 90 seconds per scan;

complete detector P95 must remain at or below 300 seconds;

100% job completion on the prospective lane;

compact cache contains only scale, line IDs, pair metrics, seven station summaries, deletion decisions, timings, and hashes.

No vertex array, triangle array, section polyline, or sampled normal sequence may be serialized.

10. Prospective activation gate

V5R3B is excluded from execution, scoring, threshold selection, visual confirmation, and mechanism validation. Its evidence is the causal motivation only.

The one execution shall use one fresh truth-unopened lane satisfying all of the following before any PSROC_V1 invocation:

At least six scans.

Exact scan hashes and exact frozen V52 primary/replay outputs committed in a lane manifest.

Lane membership fixed before computing PSROC eligibility.

No case replacement if PSROC finds no eligible pair.

Truth inaccessible to the executable and operator until PSROC primary/replay outputs and pre-truth visuals are frozen.

Source-free implementation gate

The fixed mechanism must pass these eight preregistered tests before the prospective lane:

Circular smooth ribbon: 60
∘
 roll, radius 24h, axial length 80h, two interior parallel candidates — delete both.

Variable-curvature smooth ribbon: 75
∘
 monotone roll with curvature density proportional to 1+0.5cos(2πq), deterministic normal-direction displacement bounded by 0.15h — delete both.

Sharp fold plus parallel distractor: 60
∘
 tangent discontinuity with candidates at the fold and 12h away — delete neither.

U-channel: two true 90
∘
 parallel bends separated by 20h — delete neither.

Return/hem: two structural transitions with 150
∘
 total turn — delete neither.

Smooth-roll singleton: only one candidate — no mutation.

Insufficient positive evidence: remove all triangles intersecting three of the seven station slabs, leaving at most four usable sections — no mutation.

Determinism: identical source bytes replay identically; reversing every face winding before the deterministic orientation pass yields the same deletion decision.

These expectations and all constants are frozen now. A scientific fixture failure retires V64; it does not authorize changing a constant.

Lane activation

After primary/replay and pre-truth visual freezing:

at least one pair must be certified, producing at least two proposed deletions;

otherwise record V64_PSROC_NO_ACTIVATION, do not open private truth, and retire V64.

11. Hard retain/retire gate

After a valid activation, open the fresh lane’s truth exactly once and score once with the already frozen matcher.

Retain PSROC_V1 as a research result only when every condition passes:

Primary and replay are exactly identical.

No scan experiences a resource, graph, serialization, or job failure.

No baseline true positive is deleted:

TP
new
	​

=TP
V52
	​


globally and for every scan.

False negatives are unchanged globally and for every scan:

FN
new
	​

=FN
V52
	​

.

Recall is therefore unchanged globally, per scan, and per represented family.

Every deleted line scores as a false positive.

False positives fall by at least:

max(2,⌈0.25FP
V52
	​

⌉).

Precision and F1 both increase strictly.

Exact-count accuracy does not decrease.

The post-truth visual audit passes with no ambiguous or structurally plausible deletion.

Failure of any one condition produces:

V64_PSROC_TERMINAL_RETIRE

There is no partial retention, threshold exception, family exception, or “promising but inconclusive” status.

12. Required native-resolution visual checks

Before truth, generate for every mutual eligible pair:

A 4096-pixel-long-edge raw-mesh overlay showing both V52 lines, overlap interval, seven station planes, and the selected section components.

One seven-row section sheet showing:

raw triangle-plane intersection;

projected locations of A and B;

both 0.25g
s
	​

 flanks;

raw projected normals;

unwrapped normal phase;

cumulative positive rotation;

10%, 25%, 75%, and 90% locations;

candidate jump windows;

exact POSITIVE, CONTRADICTORY, or MISSING reason.

A final overlay distinguishing proposed deletion from unchanged baseline geometry.

The pre-truth visual gate requires every proposed deletion to visibly use one connected surface sheet. A section that crosses a hole, jumps to an opposite skin, changes mesh component, or follows a nearby unrelated wall terminates the run before truth.

After truth:

inspect every deleted pair at native resolution;

inspect every eligible but retained pair’s first CONTRADICTORY station;

verify that no deletion was enabled by MISSING status;

verify that no real wall, channel, return, hem, endpoint junction, or localized crease was removed;

verify surviving V52 geometry by coordinate hash.

Visual inspection may retire V64. It may not rescue a metric failure or modify the decision.

13. Causal target: O007 versus O003/O009

O007 presents the exact population this mechanism can affect: two long, close, overlapping, nearly parallel detections. If they are interior responses on one smooth roll, repeated transverse sections should exhibit:

one connected surface ribbon;

monotone normal transport;

rotation distributed over much of the corridor;

no candidate-centered tangent jump;

stable rotation phase along the pair’s shared axis.

That is positive ownership evidence for deleting both lines, rather than evidence that either line lacks local support.

The legitimate O003 and O009 transitions deleted by V63 are protected first by population restriction: an isolated line is completely ineligible. If a real channel or return line does have a parallel partner, a genuine structural pair should exhibit a concentrated normal change at one or both candidates, or a flat face bracketed by localized turns. Either produces a CONTRADICTORY station and blocks deletion.

O005’s oblique false singleton will remain. That is intentional. V64 is not a broad false-positive cleanup rule; it tests only whether the demonstrated O007-type doublet can be removed without spending any recall.

14. Research grounding

Curvature-extremum methods deliberately extract ridge and valley lines from otherwise smooth triangle meshes, so the existence of salient, parallel line responses on a smooth roll is geometrically plausible without either line being a material crease. 
Heriot-Watt Research Portal

Sharp-feature reconstruction work instead treats true sharp lines as borders between surface patches and uses first-order surface segmentation plus graph connectivity to recover those boundaries. That supports using localized tangent-normal discontinuity as a protection criterion, rather than treating absent or truncated support as evidence against a line. 
Lirias

For developable and nearly developable surfaces, tangent planes form an approximately one-parameter family; along a generator of an ideal developable surface, the tangent plane is constant. Repeatedly observing the same smooth normal-transport phase in transverse sections is therefore a reasonable finite-mesh certificate that two axial detections share one rolled ribbon. 
Geometrie
+1

Point-sampled feature research also shows that local surface variation and persistence across scale or neighborhood are central to robust geometric feature analysis under noise. PSROC_V1 uses source-scale normalization and repeated axial sections for that robustness, while deliberately avoiding a general feature detector or learned classifier. 
CGL ETHZ

The exact pair certificate, constants, and simultaneous two-line deletion rule are Sherman-specific hypotheses. They are not claimed as consequences proved by those papers.

15. Strict no-rerun/no-rescue clause

PSROC_V1 receives:

one frozen implementation;

one fresh frozen lane;

one primary execution;

one deterministic replay;

one pre-truth visual audit;

at most one truth opening;

one score;

one post-truth visual audit.

After prospective invocation, no change is permitted to station count, eligibility, pair matching, section construction, flank length, normal kernel, angular bounds, diffuseness bounds, phase-consistency bounds, mutation rule, resource caps, lane membership, or scorer.

V5R3B may not be used as a V64 confirmation or rescue lane. A failure, nonactivation, TP deletion, insufficient FP reduction, visual ambiguity, nondeterminism, or resource breach terminally retires V64. Any later attempt must receive a new architecture identity, a materially different frozen mechanism, and a fresh truth-unopened lane.
