EXECUTE_ONE

# V67 — `SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1`

Abbreviation: `NCMM_NMS_V1`

V67 replaces the scale-sensitive ordinary final NMS with a conservative, source-supported duplicate matcher. It suppresses a candidate only when a close, overlapping candidate pair is explained by one elongated interior-curvature mode. It never deletes a candidate merely because support is missing, weak, partial, boundary-adjacent, or topologically ambiguous.

This is not a V66 rescue. It uses no transverse section planes, face-local turn events, section topology, face-pair ownership, all-pairs clique, or V66 cache field.

## 1. Frozen insertion and mutation boundary

Run frozen V52 unchanged through its complete candidate-generation and composition path, stopping at the exact call boundary immediately before the ordinary final NMS.

At that boundary:

1. Freeze the complete pre-NMS candidate multiset, candidate segment coordinates, and opaque candidate hashes.
2. Do not call the ordinary final NMS in the V67 route.
3. Invoke `NCMM_NMS_V1` exactly once as the sole final deduplicator.
4. Do not run any later NMS, endpoint merge, line averaging, snapping, continuation, count selection, or score-based filtering.
5. Preserve the original V52 route, including its original NMS, byte-for-byte as the retained comparison baseline.

V67 is removal-only relative to the frozen pre-NMS candidate multiset. It may not add, move, rotate, extend, shorten, average, or split a segment.

## 2. Exact inputs and source population

For one case, the only decision inputs are:

* the exact bounded source coordinates given to V52;
* triangle adjacency when a mesh exists;
* the frozen pre-NMS candidate segments ([a_i,b_i]);
* stable opaque candidate hashes (h_i), carried unchanged through all variants.

No V52 score, lane identity, provenance tag, filename, part identity, expected count, label, CAD field, or previous benchmark outcome is available to V67.

Use the same bounded source mesh that entered V52, with at most 200,000 source vertices. Do not resample or smooth it. Source arrays are held only in memory and are never serialized.

A point-only source has no admissible polyhedral curvature measure. It therefore returns `POINT_ONLY_KEEP_ALL`, with no suppression. This is fail-closed, not a failure to be rescued.

## 3. Interior hinge measure

### 3.1 Mesh topology

Build the undirected triangle-edge incidence graph. An edge contributes positive evidence only when:

* it has exactly two nondegenerate incident faces;
* those faces belong to an orientable manifold component;
* the edge is not a mesh boundary;
* the edge is not nonmanifold.

Orient each manifold component consistently by deterministic graph traversal. The arbitrary global orientation of a component is irrelevant because all final sign tests use absolute coherence.

For an eligible edge (e), with consistently oriented adjacent unit normals (n_1,n_2), define

[
\theta_e
========

\operatorname{atan2}
\left(
\lVert n_1\times n_2\rVert,,
n_1\cdot n_2
\right)
\in[0,\pi].
]

Choose an edge direction (\hat e). The signed turn is

[
\widetilde{\theta}_e
====================

\operatorname{atan2}
\left(
\hat e\cdot(n_1\times n_2),,
n_1\cdot n_2
\right).
]

Reversing the entire component winding can reverse all signed turns, but does not change (\theta_e) or the absolute sign-coherence statistic defined below.

### 3.2 Hinge atoms

Before candidate evaluation, merge collinear consecutive fragments that:

* share the same unordered pair of supporting face planes;
* have the same signed dihedral turn after component orientation;
* meet endpoint-to-endpoint without branching.

The result is a set of maximal hinge atoms. This coalescing is mandatory for compactness and stable audit, although the underlying measure is additive even without it.

Exact PL subdivision of a face creates only zero-turn interior edges. Exact subdivision of an existing hinge partitions its length while preserving its dihedral turn. The accumulated measure below is therefore unchanged.

## 4. Candidate and pair eligibility

Every finite segment of positive length is candidate-eligible. Ineligibility never removes it.

For candidate (i),

[
L_i=\lVert b_i-a_i\rVert,\qquad
u_i=\frac{b_i-a_i}{L_i}.
]

For pair (i,j), reverse (u_j) when necessary so that (u_i\cdot u_j\ge0), and define

[
t_{ij}
======

\frac{u_i+u_j}{\lVert u_i+u_j\rVert}.
]

Project both candidate segments onto (t_{ij}). Let their common axial interval be ([z_0,z_1]), with length

[
L=z_1-z_0.
]

At axial coordinate (z), define the corresponding point on candidate line (i) by

[
p_i(z)
======

a_i
+
u_i,
\frac{z-t_{ij}\cdot a_i}{t_{ij}\cdot u_i}.
]

Let

[
r(z)
====

\left(I-t_{ij}t_{ij}^{\mathsf T}\right)
\left(p_j(z)-p_i(z)\right).
]

Evaluate (r_0=r(z_0)), (r_m=r((z_0+z_1)/2)), and (r_1=r(z_1)), with lengths (d_0,d,d_1).

A noncoincident pair is geometry-eligible exactly when all predicates pass:

[
\arccos(|u_i\cdot u_j|)
\le
\frac{\pi}{24}
\quad\text{(7.5 degrees)};
]

[
\frac{L}{\min(L_i,L_j)}
\ge
\frac58;
]

[
\frac1{4096}
\le
\frac dL
\le
\frac1{16};
]

[
\frac{\min(d_0,d_1,d)}
{\max(d_0,d_1,d)}
\ge
\frac34;
]

[
\frac{r_0\cdot r_1}{d_0d_1}
\ge
\cos!\left(\frac{\pi}{18}\right)
\quad\text{(10 degrees)};
]

and neither denominator is zero.

Set

[
e=\frac{r_m}{d},
\qquad
n=t_{ij}\times e.
]

At both overlap endpoints, candidate (j), expressed in this frame relative to candidate (i), must satisfy

[
\frac{(p_j(z)-p_i(z))\cdot e}{d}
\in
\left[\frac78,\frac98\right],
]

[
\frac{|(p_j(z)-p_i(z))\cdot n|}{d}
\le
\frac18.
]

These last predicates reject crossing and materially skew pairs.

### Exact coincident rule

A pair whose quantized normalized segment Hausdorff distance is zero, whose angular difference is zero, and whose overlap ratio is at least (15/16) is `EXACT_COINCIDENT`. It is a positive duplicate pair without requiring curvature evidence. Two spatially distinct physical bends cannot occupy the same line geometry.

All other pairs failing geometry eligibility remain untouched.

### Why real V52 candidates can enter

There is no exact face-hit, transverse-section, clique, station, or topology-pattern prerequisite. Every finite candidate is admitted individually, and every sufficiently close, overlapping, approximately parallel pair lies in an open pair-eligibility domain. The fresh real-scan preflight below additionally requires actual positive matched pairs before any V67 output is committed. A zero-eligibility result therefore cannot consume the prospective execution.

## 5. Pair-local normal-cycle profile

For a geometry-eligible pair, define, for every point (q) on a hinge atom,

[
z=t_{ij}\cdot q,
\qquad
\tau(q)=\frac{z-z_0}{L},
]

[
\xi(q)
======

\frac{(q-p_i(z))\cdot e}{d},
\qquad
\eta(q)
=======

(q-p_i(z))\cdot n.
]

Use the pair corridor

[
0\le\tau\le1,
\qquad
-\frac12\le\xi\le\frac32,
\qquad
|\eta|\le R,
]

where

[
R=\max!\left(\frac d2,\frac{L}{128}\right).
]

Clip hinge atoms exactly against the corridor. No vertex, midpoint, or station sampling is permitted.

Partition the corridor into:

* 16 equal axial bins in (\tau);
* 32 equal transverse bins in (\xi), spanning ([-1/2,3/2]).

Split an atom exactly wherever it crosses a corridor or bin boundary.

For a clipped atom fragment from (q_0) to (q_1), add unsigned mass

[
w
=

\theta_e
\frac{|t_{ij}\cdot(q_1-q_0)|}{L},
]

and signed mass

[
\widetilde w
============

\widetilde{\theta}*e
\frac{|t*{ij}\cdot(q_1-q_0)|}{L}.
]

The projected-length factor rejects curvature edges transverse to the candidate direction while preserving oblique triangulations.

Let (H_b) be unsigned mass in transverse bin (b), (A_a) unsigned mass in axial bin (a), and (T_c) unsigned mass contributed by manifold component (c). Define

[
T=\sum_{b=0}^{31}H_b=\sum_{a=0}^{15}A_a.
]

No edge-angle threshold is applied before integration.

## 6. Positive evidence, negative evidence, and fail-closed decision

### 6.1 Axial support

Define the axial participation number

[
N_{\mathrm{eff}}
================

\frac{T^2}{\sum_{a=0}^{15} A_a^2}.
]

Uniform support in four axial bins gives (N_{\mathrm{eff}}=4). A localized junction concentrated in one bin gives (N_{\mathrm{eff}}=1).

### 6.2 Transverse mode profile

Smooth the transverse profile once with the fixed kernel

[
G_b
===

\frac{
H_{b-2}+2H_{b-1}+3H_b+2H_{b+1}+H_{b+2}
}{9},
]

with out-of-range terms equal to zero.

A local-maximum plateau is significant when its height is at least

[
\frac{T}{16}.
]

For two consecutive significant peak plateaus (P,Q), let (V_{PQ}) be the minimum (G_b) strictly between them. Connect the peaks into the same persistent mode exactly when

[
V_{PQ}>
\frac12\min(G_P,G_Q).
]

Equality is not merged. Connected components of this peak graph are the persistent modes.

For a persistent mode, define its transverse center as the (G_b)-weighted mean of the bin centers from its leftmost through rightmost constituent peak.

### 6.3 Bridge mass

The bridge is

[
\frac14\le\xi\le\frac34.
]

Because the transverse bin width is (1/16), this is exactly bins 12 through 19. Define

[
B=\sum_{b=12}^{19}H_b.
]

### 6.4 Component and turn-sign coherence

Let (c^\star) be the component of maximum mass. Define

[
D=\frac{T_{c^\star}}{T},
]

and, using only contributions from (c^\star),

[
C
=

\frac{
\left|\sum \widetilde w_{c^\star}\right|
}{
T_{c^\star}
}.
]

A disconnected second skin therefore fails component dominance when it contributes substantial mass. Opposing turns on one connected return or hem fail sign coherence.

### 6.5 Boundary and nonmanifold veto

Inside the bridge subcorridor, accumulate projected axial length from mesh-boundary and nonmanifold edges:

[
Q_{\partial}
============

\frac1L
\sum
|t_{ij}\cdot(q_1-q_0)|.
]

These edges provide no positive turn mass. They are a veto because they can represent a perimeter or an observation boundary hiding a second feature.

### 6.6 Positive duplicate certificate

A noncoincident pair is `ONE_FOLD_MODE` if and only if all of the following hold:

[
T\ge\frac{13}{64}\ \text{radians};
]

[
N_{\mathrm{eff}}\ge4;
]

[
\frac BT\ge\frac12;
]

there is exactly one persistent transverse mode;

[
\frac14
\le
\xi_{\mathrm{mode}}
\le
\frac34;
]

[
D\ge\frac34;
]

[
C\ge\frac34;
]

[
Q_{\partial}<\frac14.
]

All positive inequalities are inclusive except the boundary/nonmanifold veto, which is strict.

The only positive pair states are:

* `EXACT_COINCIDENT`;
* `ONE_FOLD_MODE`.

All other states keep both candidates. Frozen negative reason codes are:

* `PAIR_GEOMETRY`;
* `LOW_TURN`;
* `AXIAL_LOCALIZED`;
* `NO_BRIDGE`;
* `NO_SIGNIFICANT_MODE`;
* `MULTI_MODE`;
* `MODE_OUTSIDE_BRIDGE`;
* `MULTI_COMPONENT`;
* `OPPOSING_TURN`;
* `BOUNDARY_OR_NONMANIFOLD`;
* `ORIENTATION_UNAVAILABLE`;
* `NUMERIC_INDETERMINATE`;
* `RESOURCE_ABORT`.

Absence of evidence is never deletion evidence.

## 7. Disjoint matching and representative ranking

Positive pairs do not form transitive clusters. This prevents a chain (A!\sim!B!\sim!C), with (A\not\sim C), from collapsing two distinct bends.

For each positive pair define the confidence tuple, sorted descending:

[
\left(
\frac BT,,
C,,
D,,
N_{\mathrm{eff}},,
T,,
-\frac dL
\right),
]

followed by the ascending ordered pair of opaque hashes as the final deterministic tie-break.

Traverse positive pairs in that order. Select a pair only when neither candidate has already been selected. The resulting clusters are disjoint clusters of exactly two candidates.

Thus:

* one candidate participates in at most one suppression;
* a three-candidate chain can lose at most one candidate;
* no suppressed candidate can suppress another candidate;
* there is no connected-component or clique collapse.

### Representative

For `ONE_FOLD_MODE`, take the half-height basin of the sole persistent mode: the maximal contiguous bin interval containing its highest plateau and satisfying

[
G_b\ge\frac12G_{\max}.
]

Using only unsigned mass from dominant component (c^\star) inside that basin, compute the curvature-weighted cross-plane centroid (\bar q_\perp). Define the mode line

[
\ell^\star
==========

\bar q_\perp+\mathbb R,t_{ij}.
]

For each candidate, compute the exact RMS perpendicular distance to (\ell^\star) over the common axial interval, normalized by (L):

[
E_i
===

\frac1L
\sqrt{
\frac1L
\int_{z_0}^{z_1}
\operatorname{dist}^2(p_i(z),\ell^\star),dz
}.
]

The representative ranking is:

1. smaller quantized (E_i);
2. longer candidate length (L_i);
3. lower opaque candidate hash.

For `EXACT_COINCIDENT`, use length and then hash.

Suppress exactly the losing candidate. Retain the winner with byte-identical original geometry.

## 8. Numerical and transformation contract

Every decision scalar is converted to

[
\widehat x
==========

2^{-40}
\operatorname{round}_{\mathrm{ties\ to\ even}}
\left(2^{40}x\right)
]

before a predicate or ordering comparison. All sums use an order-independent binary superaccumulator. NaN and infinity are forbidden.

The semantic audit excludes timing and timestamps.

### Similarity proof

Under

[
q'=\lambda Rq+t_0,\qquad \lambda>0,
]

angles are unchanged; (d,L,R), atom lengths, and candidate lengths scale by (\lambda); and therefore

[
d/L,\quad
\tau,\quad
\xi,\quad
\eta/R,\quad
\theta_e\frac{|\Delta z|}{L}
]

are unchanged. Histograms, persistent modes, pair states, matching, representative selection, and survivor hashes are consequently unchanged.

Rigid transforms are the (\lambda=1) case.

Triangle and candidate ordering cannot change the superaccumulated measure or hash-sorted matching.

Global or component winding reversal does not change unsigned mass, component dominance, or absolute sign coherence.

For exact PL refinement:

* new coplanar edges have (\theta=0);
* a split hinge retains its angle;
* fragment projected lengths and exact bin clips sum to the original contribution.

No deduplicator runs after V67, so there is no downstream scale-sensitive operation.

### Mandatory variant signature

For every prospective case, compute the V67 semantic signature on:

1. identity;
2. replay;
3. scale (0.8), proper signed-axis permutation, and translation;
4. scale (1.25), a different proper axis permutation, and translation;
5. reversed triangle and candidate ordering;
6. global winding reversal;
7. deterministic exact midpoint PL refinement of a bounded canonical triangle subset.

Candidate hashes are carried through variants rather than recomputed.

After inverse mapping, the following must be identical:

* pair reason for every candidate-hash pair;
* all quantized pair metrics;
* selected matching;
* winner and loser hashes;
* final survivor hash set.

Any mismatch aborts the entire case before output commit.

## 9. Compact deterministic audit

Write one canonical JSON semantic artifact containing:

* source and candidate-input hashes;
* architecture/version/constants hash;
* mesh, component, hinge-atom, candidate, and pair counts;
* every geometry-eligible pair’s hashes, reason, and quantized metrics;
* 16 axial and 32 transverse Q40 bins for positive pairs and vetoed near-positive pairs;
* persistent-mode intervals and centers;
* boundary/nonmanifold ratio;
* selected disjoint matching;
* representative-distance values;
* winner, loser, and survivor hashes;
* primary/replay/variant semantic hashes;
* final status and atomic-commit hash.

Do not serialize source vertices, source points, triangle arrays, normals, atom coordinates, or candidate coordinate arrays beyond the pre-existing candidate artifact.

The semantic JSON is canonical, has `allow_nan=false`, and contains no runtime-dependent fields. Resource telemetry is written separately and is not part of replay identity.

Primary and replay semantic JSON must be byte-identical.

## 10. Resource and atomicity caps

Frozen per-case caps:

| Resource                                    |       Cap |
| ------------------------------------------- | --------: |
| Primary source vertices                     |   200,000 |
| Primary triangles                           |   400,000 |
| Variant triangles                           |   500,000 |
| Incidence-graph edges                       |   600,000 |
| Hinge atoms                                 |   500,000 |
| Candidates                                  |        96 |
| Total candidate pairs                       |     4,560 |
| Geometry-eligible pairs                     |     2,048 |
| Total atom/corridor clip visits             | 8,000,000 |
| Semantic audit                              |     4 MiB |
| Additional V67 peak RSS                     |     2 GiB |
| Primary V67 wall time                       |      30 s |
| All V67 variants and replay per source case |     180 s |
| Complete V52/V67 research route per case    |     300 s |

Execution is single-threaded and deterministic.

All proposed mutations remain in memory until:

* every cap passes;
* all variant signatures pass;
* the audit serializes successfully;
* the output and audit hashes are finalized.

Commit uses temporary files, flush, and atomic rename. On any exception, cap breach, serialization error, or invariance mismatch:

* no V67 candidate output is committed;
* no candidate cache is modified;
* the frozen V52 artifacts remain untouched;
* only a terminal audit status is written;
* the case is not scored.

There is no replacement input, second attempt, or serializer rescue.

## 11. Frozen source-free fixture suite

These fixtures validate implementation and predicate consistency only. They are not generalization evidence and must not be combined with the quarantined synthetic morphology lanes.

### 11.1 Exact fixture generator

Use (L=1), 16 exact axial cells, and a default candidate separation (d=1/32). The default candidates are

[
c_0:(0,0,0)\rightarrow(0,0,1),
]

[
c_1:(d,0,0)\rightarrow(d,0,1).
]

An `XTURN` record

[
(\xi,\beta,[a,b],s,k)
]

creates a narrow extruded manifold ribbon with:

* a longitudinal hinge at transverse coordinate (\xi d);
* signed turn (\beta);
* axial support (z\in[a,b]);
* sign (s\in{-1,+1});
* component identifier (k).

Each hinge is triangulated into the exact 16-cell axial grid. Flat connecting strips are placed outside the pair corridor or have zero turn. For this construction, one record contributes

[
|\beta|(b-a)
]

to (T). Multiple records may represent a faceted rounded bend, two distinct folds, or multiple skins.

### 11.2 Core fixtures

| ID                           | Exact construction                                                                                                  | Required result                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `F01_SHARP_DUP_60`           | One component; (\xi=17/32), (\beta=\pi/3), support ([0,1]).                                                         | One suppression. (T=\pi/3), (N_{\rm eff}=16), (B/T=C=D=1), one mode. |
| `F02_TRIPLE_DUP_NO_CHAIN`    | Candidates at (-d,+d,+2d); one 60-degree hinge at zero. Hash ordering and confidence make the ((-d,+d)) pair first. | Exactly one candidate suppressed; never two.                         |
| `F03_NEARBY_DISTINCT`        | Same component; two same-sign full-span hinges at transverse bin centers (11/32) and (23/32), each (\beta=1/2).     | Keep both: two persistent modes despite (B/T=1).                     |
| `F04_U_CHANNEL_OPPOSING`     | Two full-span hinges at (\xi=1/8,7/8), turns (+\pi/2,-\pi/2).                                                       | Keep both: multi-mode and opposing structure.                        |
| `F05_RETURN_HEM_DOUBLE_SKIN` | Two components, adjacent bridge-bin hinges, equal mass (1/2), opposite signs.                                       | Keep both: (D=1/2); sign cancellation is also visible.               |
| `F06_ROUNDED_DUP`            | Five consecutive bridge-bin hinges, bins 14–18, same sign, each (\beta=3/25).                                       | One suppression. (T=3/5), one persistent rounded mode.               |
| `F07_SMALL_CREASE`           | One bridge hinge, (\beta=1/4), full span.                                                                           | One suppression; (1/4>13/64).                                        |
| `F08_FLAT_FACE`              | One connected planar mesh; two close candidates.                                                                    | Keep both: (T=0).                                                    |
| `F09_PERIMETER_DISTRACTOR`   | Flat open strip; candidates parallel to a full-span boundary in the bridge.                                         | Keep both: no positive interior turn and (Q_\partial\ge1/4).         |
| `F10_LOCAL_JUNCTION`         | One bridge hinge with (\beta=6/5), but support only one axial bin.                                                  | Keep both: (N_{\rm eff}=1), (T=3/40).                                |
| `F11_PARTIAL_SUPPORT_4`      | One bridge hinge, (\beta=1), support exactly four axial bins.                                                       | One suppression. (T=1/4), (N_{\rm eff}=4) exactly.                   |
| `F12_PARTIAL_SUPPORT_3`      | Same, support exactly three axial bins.                                                                             | Keep both. (T=3/16<13/64), (N_{\rm eff}=3).                          |

### 11.3 Exact confidence-boundary fixtures

| ID                    | Exact construction                                                                                                                    | Required result                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `F13_TURN_EQ`         | One full bridge hinge with (\beta=13/64).                                                                                             | Suppress; total-turn equality passes.                               |
| `F14_TURN_BELOW`      | (\beta=13/64-2^{-20}).                                                                                                                | Keep.                                                               |
| `F15_BRIDGE_EQ`       | Adjacent transverse bins 11 and 12, each with mass (13/128), same sign and full span. Bin 12 is inside the bridge and bin 11 outside. | Suppress. (T=13/64), (B/T=1/2), one peak plateau centered at (1/4). |
| `F16_BRIDGE_BELOW`    | Move (2^{-20}) mass from bin 12 to bin 11 while preserving total mass.                                                                | Keep; (B/T<1/2).                                                    |
| `F17_SIGN_EQ`         | One component, adjacent bridge hinges with unsigned masses (7/32) and (1/32), opposite signs.                                         | Suppress; (C=(7/32-1/32)/(8/32)=3/4).                               |
| `F18_SIGN_BELOW`      | Increase the opposite-sign mass by (2^{-20}), decrease the first by the same amount.                                                  | Keep; (C<3/4).                                                      |
| `F19_COMPONENT_EQ`    | Co-located same-sign bridge mass (3/16) on component A and (1/16) on component B.                                                     | Suppress; (D=3/4).                                                  |
| `F20_COMPONENT_BELOW` | Transfer (2^{-20}) mass from A to B.                                                                                                  | Keep; (D<3/4).                                                      |
| `F21_BOUNDARY_EQ`     | Bridge boundary/nonmanifold projected ratio exactly (1/4), otherwise positive.                                                        | Keep; positive requires (Q_\partial<1/4).                           |

### 11.4 Exact pair-domain fixtures

For each predicate below, use the `F01` source and keep every other pair predicate strictly interior:

* `F22A_ANGLE_EQ`: angle (=\pi/24), suppress.
* `F22B_ANGLE_OVER`: angle (=\pi/24+2^{-20}), keep.
* `F22C_OVERLAP_EQ`: overlap (=5/8), suppress.
* `F22D_OVERLAP_BELOW`: overlap (=5/8-2^{-20}), keep.
* `F22E_SEPARATION_EQ`: (d/L=1/16), suppress.
* `F22F_SEPARATION_OVER`: (d/L=1/16+2^{-20}), keep.
* `F22G_VARIATION_EQ`: (d_{\min}/d_{\max}=3/4), suppress.
* `F22H_VARIATION_BELOW`: ratio (=3/4-2^{-20}), keep.
* `F22I_DIRECTION_EQ`: endpoint separation-direction change (=\pi/18), suppress.
* `F22J_DIRECTION_OVER`: change (=\pi/18+2^{-20}), keep.

### 11.5 Safety and metamorphic fixtures

* `F23_EXACT_COINCIDENT`: identical segments; retain the lower-hash candidate.
* `F24_POINT_ONLY`: unchanged candidate set.
* `F25_NONMANIFOLD_ONLY`: apparent turn carried only by nonmanifold edges; unchanged.
* `F26_CANDIDATE_CAP`: 97 candidates; atomic no-op.
* `F27_PAIR_CAP`: more than 2,048 geometry-eligible pairs; atomic no-op.
* `F28_VISIT_CAP`: force the 8,000,001st clip visit; atomic no-op.
* `F29_SERIALIZATION_ABORT`: inject a noncanonical audit scalar before commit; no output artifact.
* `F30_PRIMARY_REPLAY`: byte-identical semantic artifact.
* `F31_ORDER_WINDING`: all core fixture outcomes unchanged.
* `F32_SIMILARITY_08`: all core fixture outcomes unchanged.
* `F33_SIMILARITY_125`: all core fixture outcomes unchanged.
* `F34_FULL_PL_REFINEMENT`: refine every fixture triangle 1-to-4; all metrics and outcomes unchanged.

All fixtures must pass before any real source is read. One fixture failure blocks execution; no constant may be changed.

The mandatory positive fixtures are mathematically compatible with the thresholds:

[
T_{\text{F01}}=\pi/3>\frac{13}{64},
]

[
T_{\text{F06}}=\frac35>\frac{13}{64},
]

[
T_{\text{F07}}=\frac14>\frac{13}{64},
]

[
T_{\text{F11}}=1\cdot\frac4{16}=\frac14,
\qquad
N_{\mathrm{eff}}=4,
]

[
T_{\text{F13}}=\frac{13}{64},
]

[
(B/T)*{\text{F15}}=\frac12,
\qquad
C*{\text{F17}}=\frac34,
\qquad
D_{\text{F19}}=\frac34.
]

No positive fixture depends on a value lying on the wrong side of a frozen predicate.

## 12. Prospective real-scan protocol

Use the next four predesignated unopened corpus meshes in ascending opaque source-hash order, excluding:

* R001–R008;
* every V66 one-shot case, including U01–U04;
* the newly received physical part;
* every quarantined synthetic part.

Do not expose filenames or identities to the implementation or reviewer.

For each fresh case:

1. Run frozen V52 composition and freeze the complete pre-NMS candidate multiset.
2. Run the frozen original V52 final NMS once on the primary only, solely to produce the comparison baseline.
3. In one V67 process, compute primary, replay, and every locked variant.
4. Hold all V67 mutations uncommitted.
5. Produce the read-only actionability census.
6. Commit only when the tranche-level preflight gate passes.
7. Freeze all output and audit hashes.
8. Perform no rescue, replacement input, or threshold change.

### Mandatory actionability preflight

Before any V67 candidate output is committed, the four-case tranche must have:

* at least two geometry-eligible real candidate pairs;
* at least two profile-evaluable pairs;
* at least two disjoint proposed suppressions in total;
* at least one proposed changed case;
* identical proposed suppressions across all locked variants;
* no resource, topology, numeric, or serialization abort;
* V67 output count no greater than the frozen primary V52-final count in every case;
* total V67 output count at least two lower than the V52-final total.

The preflight may report only aggregate counts, reason histograms, and hashes. It may not reveal identity, filenames, pseudo-truth, or expected counts.

Failure of this gate terminally retires V67 before candidate-output commit. The constants remain frozen.

## 13. Native-resolution visual audit

Before pseudo-truth is opened, inspect every case whose V67 primary candidate set differs from the frozen V52-final primary set.

Render from the full bounded source geometry with no visual decimation, at the established native 4096-pixel output or higher. For every changed case preserve:

* raw source only;
* V52-final overlay;
* V67 overlay;
* removed candidates only;
* winner/loser pair links;
* the dominant-mode corridor and curvature-mode line;
* mapped 0.8x and exact-PL-refinement overlays.

Every suppression must visibly satisfy all of the following:

* winner and loser are responses to the same visible physical fold;
* the loser does not correspond to a separately visible nearby turn;
* the pair is not an opposing U-channel fold;
* the pair is not two sides of a return or hem;
* the loser is not the sole marker on a partially observed crease;
* the selected winner is at least as well centered on the visible fold as the loser;
* no unmatched pre-NMS candidate has been resurrected as new clutter.

Any uncertain suppression is a visual failure. Pseudo-truth remains sealed and V67 is retired.

## 14. Hard retain and retire gates

### Pre-truth retain gate

All must pass:

* 100% source-free fixtures;
* 100% primary/replay semantic identity;
* 100% rigid, similarity, order, winding, and PL-refinement semantic invariance;
* at least two real suppressions;
* at least one changed real case;
* no case-level output-count increase versus V52 final;
* every changed case reviewed at native resolution;
* every removed candidate visually certified as a duplicate;
* zero visible suppression of a distinct bend, second skin, return, hem, opposing fold, or isolated partial crease;
* all resource caps.

### Pseudo-truth opening

Only after the complete visual audit is signed and frozen, open the fresh-lane pseudo-truth once and score V52 and V67 on the same primaries.

### Final retain gate

Retain V67 only if all hold:

[
TP_{67}\ge TP_{52},
]

[
FN_{67}\le FN_{52},
]

[
FP_{67}
\le
FP_{52}
-------

\max!\left(2,\left\lceil0.15,FP_{52}\right\rceil\right),
]

with:

* strictly higher precision;
* strictly higher F1;
* no changed case with increased FN;
* exact-count cases not decreased;
* 100% render-admissible retained representatives;
* deterministic and resource gates still passing.

Any false suppression, absent real visual delta, recall loss, FP failure, invariance mismatch, cap breach, or serialization failure terminally retires V67. No rescue or rerun is authorized.

## 15. Newly received physical part

Reserve the newly received physical part as later unseen physical evidence.

No capability opening is scientifically necessary now: the permitted mesh geometry already contains a similarity-covariant interior hinge measure, and the actionability preflight prevents another zero-domain one-shot. The new part must not influence V67 constants, fixtures, eligibility, visual criteria, or the current retain/retire decision.

It may be opened only after V67 survives the fresh four-case real lane unchanged.

## 16. Research grounding and hypothesis boundary

Normal-cycle work provides a principled framework for curvature measures on polyhedral surfaces. Discrete mean curvature is edge-supported; a scalar integrated contribution of the form (\theta_e|e|) is standard, and subdividing an existing polyhedral edge preserves the corresponding integrated edge contribution. Those facts motivate the hinge measure and exact-PL-refinement contract here. ([math.univ-lyon1.fr][1])

Integral geometric quantities are also used specifically to avoid the instability of purely pointwise differential estimates under noise and small perturbations. That supports aggregating turn over an axial corridor rather than relying on isolated face events or exact cross-sections. ([ScienceDirect][2])

The one-mode bridge predicate, its constants, the sign/component vetoes, and the disjoint matching policy are Sherman-specific falsifiable hypotheses. They are not claims established by those papers.

[1]: https://math.univ-lyon1.fr/m/~morvan/SCANPUBLICATIONS/cohensteinermorvansocg2003.pdf "https://math.univ-lyon1.fr/m/~morvan/SCANPUBLICATIONS/cohensteinermorvansocg2003.pdf"
[2]: https://www.sciencedirect.com/science/article/pii/S0167839608000095?utm_source=chatgpt.com "Integral invariants for robust geometry processing"
