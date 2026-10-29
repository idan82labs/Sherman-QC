EXECUTE_ONE

# V70_BOUNDARY_REJECTING_MULTISCALE_VARIATION_TRACK_COMPOSITION_V1

**Abbreviation:** `BRMVT_V1`

V69 is retired without rescue. Its corrected run rejected 47 of 48 real candidates because almost every candidate failed the longitudinal station requirement; the packet correctly classifies that result as a representation/actionability failure rather than a threshold miss. 

`BRMVT_V1` replaces exact candidate-aligned cross-section reconstruction with a **density-adaptive, multiscale ridge field on the observed source samples**, followed by a boundary-rejecting hysteresis graph and a partial longitudinal support track. It starts from the richer frozen V16 thresholded pre-NMS pool, not V52’s composed output. It can therefore both reject unsupported V16 hypotheses and preserve V16 hypotheses that V52 would have removed, making simultaneous precision and recall improvement possible without generating unconstrained new lines.

---

## 1. Insertion point and operation type

**Insertion point**

```text
frozen V16 thresholded pre-NMS candidates
        ↓
BRMVT_V1 geometry-only support and duplicate ownership
        ↓
the existing single ordinary NMS, exactly once and unchanged
        ↓
render/output
```

`BRMVT_V1` is a **replacement composition**, not a post-V52 veto:

* V52 remains the retained comparison baseline.
* V52 is not called inside the V70 branch.
* V67, V68, and V69 code, outputs, reasons, and thresholds are unavailable.
* BRMVT may keep any geometrically supported member of the V16 thresholded pre-NMS set, including one absent from the V52 result.
* It does not alter candidate endpoints or directions.
* It does not generate a candidate where V16 supplied none.
* Its internal duplicate ownership occurs before the one frozen ordinary NMS.

That bounded scope attacks the real simultaneous FP/FN deficit without pretending to solve bends completely absent from V16. The real lane currently has precision 0.45 and recall 0.5625, with unsupported responses and missed local bends occurring together. 

---

## 2. Allowed and forbidden inputs

### Allowed

For one scan:

* at most 200,000 finite source points or mesh vertices (P={p_i}\subset\mathbb R^3);
* at most 350,000 triangles and undirected adjacency, when available;
* at most 128 finite V16 candidate segments (c=(a_c,b_c));
* one opaque candidate hash per segment;
* source-geometry opaque hash;
* deterministic quantities derived solely from the above.

These are within the representation explicitly opened by the packet. 

### Forbidden

Neither BRMVT nor its cache or audit may read or infer from:

* candidate score or candidate rank;
* route or detector provenance;
* the V52 keep/drop result;
* filename, part identity, family identity, or acquisition identity;
* expected bend count;
* CAD;
* labels, gold, benchmark feedback, visual judgments, or manual truth;
* candidate ordering supplied by an upstream score;
* another candidate’s original score or route.

The term “percentile” below refers only to a **local percentile of a source-derived surface-variation scalar**, never to candidate rank.

Opaque candidate hashes may be used only for canonical ordering, final equality ties, and audit linkage.

---

## 3. Deterministic preprocessing

### 3.1 Canonical source set

1. Reject the scan atomically if any point, triangle index, or candidate endpoint is nonfinite.
2. Canonicalize every `-0.0` coordinate to `+0.0`.
3. Merge exactly coincident source points.
4. Reject atomically if the resulting point count is outside

[
2048\le |P|\le 200000.
]

5. Reject atomically if any candidate has

[
L_c=|b_c-a_c|=0.
]

6. Triangle winding is ignored. Only undirected vertex incidence and undirected edge incident-triangle counts may be used.

### 3.2 Reproducible neighborhoods

For every source point (p_i), conceptually order all other points by squared Euclidean distance. A KD-tree or equivalent exact spatial index shall implement the ordering.

Required neighborhood cardinalities are

[
K={6,12,24,64,160,256,512,1024,1280}.
]

At a cardinality boundary, include the complete exact-distance tie shell. No arbitrary member of a tie shell may be selected.

Per-point tie-shell expansion is capped at 64 additional points:

[
|N_k(i)|\le k+64.
]

Any larger tie shell causes atomic `NEIGHBOR_TIE_CAP_ABORT`.

All covariance sums use a reproducible order-independent floating-point accumulator. An ordinary parallel reduction whose result changes with thread scheduling is forbidden.

### 3.3 Sampling scale

Define

[
h_i=\operatorname{median}{|p_j-p_i|:p_j\in N_6(i)},
]

and

[
h_0=\operatorname{median}_i h_i.
]

Abort if (h_0\le0).

This scale is used only for graph connectivity, candidate tubes, and support lengths. Feature classification itself uses several neighborhood populations, so it is not tied to one sampling radius.

---

## 4. Multiscale geometric ridge field

### 4.1 Surface variation

For each point (p_i) and each

[
k\in K_v={24,64,160,256,512,1024},
]

let

[
\bar p_{i,k}
=\frac{1}{|N_k(i)|}\sum_{p_j\in N_k(i)}p_j
]

and

[
C_{i,k}
=\frac{1}{|N_k(i)|}
\sum_{p_j\in N_k(i)}
(p_j-\bar p_{i,k})(p_j-\bar p_{i,k})^\top .
]

Let its eigenvalues be ordered

[
0\le\lambda_{0,i,k}\le\lambda_{1,i,k}\le\lambda_{2,i,k}.
]

Define source surface variation

[
\sigma_{i,k}
============

\frac{\lambda_{0,i,k}}
{\lambda_{0,i,k}+\lambda_{1,i,k}+\lambda_{2,i,k}}.
]

If the trace is exactly zero, set (\sigma_{i,k}=0).

A computed eigenvalue in

[
[-2^{-40}\operatorname{tr}(C_{i,k}),0)
]

is clamped to zero. A more negative eigenvalue causes atomic `EIGENVALUE_ABORT`.

### 4.2 Local ridge percentile

For the comparison neighborhood (N_{1280}(i)), define the mid-percentile

[
r_{i,k}
=======

\frac{
#{j:\sigma_{j,k}<\sigma_{i,k}}
+
\frac12#{j:\sigma_{j,k}=\sigma_{i,k}}
}{
|N_{1280}(i)|
}.
]

Define

[
u_{i,k}
=======

\operatorname{clip}*{[0,1]}
\left(
\frac{r*{i,k}-0.70}{0.25}
\right).
]

Thus:

* (r_{i,k}=0.70) gives (u_{i,k}=0);
* (r_{i,k}=0.95) gives (u_{i,k}=1).

### 4.3 Relative top-hat contrast

Let

[
m_{i,k}
=======

\operatorname{median}
{\sigma_{j,k}:p_j\in N_{1280}(i)}.
]

Define

[
a_{i,k}
=======

\max\left(
0,,
\frac{\sigma_{i,k}-m_{i,k}}
{\max(m_{i,k},2^{-20})}
\right)
]

and

[
v_{i,k}
=======

\operatorname{clip}*{[0,1]}
\left(
\frac{a*{i,k}-0.01}{0.09}
\right).
]

Thus:

* (a_{i,k}=0.01) gives (v_{i,k}=0);
* (a_{i,k}=0.10) gives (v_{i,k}=1).

This second term is essential: a broad cylinder or broad panel transition can have nonzero variation, but if that variation is spatially uniform, its local top-hat contrast is approximately zero.

### 4.4 Scale persistence

Order the six (u_{i,k}) values as

[
u_{i,(1)}\ge u_{i,(2)}\ge\cdots .
]

Define

[
P_i=\sqrt{u_{i,(1)}u_{i,(2)}}
]

and

[
Q_i=\sqrt{\max_{k\in K_v}v_{i,k}}.
]

The unpenalized ridge saliency is

[
s_i=P_iQ_i.
]

Consequences:

* a response must be locally high at **at least two scales**;
* at least one scale must also have nontrivial relative geometric contrast;
* the two scales need not be adjacent;
* no scale plateau, fixed bend radius, exact shelf pair, or exact section topology is required.

---

## 5. Boundary and dropout treatment

### 5.1 Angular boundary confidence

Use (C_{i,24}).

If (\lambda_{1,i,24}=0), set (B_i^{\rm angular}=1).

If

[
\frac{\lambda_{0,i,24}}{\lambda_{1,i,24}}>0.35,
]

the local tangent plane is not sufficiently surface-like for a boundary decision; set

[
B_i^{\rm angular}=0.
]

Otherwise:

1. take the span of the two eigenvectors belonging to (\lambda_1,\lambda_2);
2. project vectors (p_j-p_i), (p_j\in N_{24}(i)), into that plane;
3. sort their polar angles;
4. let (g_i) be the largest circular gap.

Define

[
B_i^{\rm angular}
=================

\operatorname{clip}_{[0,1]}
\left(
\frac{g_i-\frac{5\pi}{9}}
{\frac{4\pi}{9}}
\right).
]

Therefore:

* (g_i=100^\circ) gives (B_i^{\rm angular}=0);
* (g_i=180^\circ) gives (B_i^{\rm angular}=1).

### 5.2 Mesh-boundary confidence

When triangles are available, set

[
B_i^{\rm mesh}=1
]

if (p_i) is within two undirected adjacency hops of any edge whose incident-triangle count is not exactly two. Otherwise set it to zero.

When triangles are unavailable,

[
B_i^{\rm mesh}=0.
]

The combined boundary confidence is

[
B_i=\max(B_i^{\rm angular},B_i^{\rm mesh}).
]

### 5.3 Boundary-rejected feature weight

Define

[
w_i=s_i(1-0.70B_i).
]

This is deliberately a **soft** penalty:

* a complete interior crease remains strong;
* a locally one-sided dropout segment can remain as weak bridge evidence;
* an entirely one-sided perimeter or eroded edge cannot seed a feature track merely because of localized scan noise.

A candidate for which the whole observed support is indistinguishable from a one-sided boundary is rejected. That is the only source-only abstention implied by this architecture: without any locally observed turn or interior ridge segment, retaining it would recreate the edge/crease ambiguity that the intervention is meant to remove.

---

## 6. Hysteresis feature graph

### 6.1 Graph

Construct an undirected graph (G) using:

* the symmetric union of 12-nearest-neighbor relations; and
* valid mesh edges, when present.

Keep an edge ((i,j)) only when

[
0<|p_i-p_j|
\le
3\max(h_i,h_j).
]

### 6.2 Strong and weak nodes

A point is:

[
\text{strong}\iff w_i\ge0.55,
]

[
\text{weak}\iff w_i\ge0.20.
]

Equality is included.

Run deterministic multi-source Dijkstra on the subgraph induced by weak nodes, with all strong nodes as seeds and normalized edge cost

[
c_{ij}
======

\frac{|p_i-p_j|}
{\max(h_i,h_j)}.
]

A weak point becomes a retained feature node iff its shortest normalized distance to a strong node is at most

[
8.0.
]

Equality at 8.0 is retained.

No strong node means no retained feature component. This prevents a long noisy boundary made entirely of boundary-penalized weak points from becoming a candidate-support curve, while permitting finite dropout gaps around a genuinely strong crease.

---

## 7. Candidate support track

### 7.1 Candidate-local scale

For candidate (c), define

[
d_c=\frac{b_c-a_c}{L_c}.
]

At each of 17 positions

[
x_j=a_c+\frac{j+\frac12}{17}(b_c-a_c),
\qquad j=0,\ldots,16,
]

query the eight nearest source points. Take the median (h_i) at each station and then the median of the 17 station medians:

[
\hat h_c
========

\operatorname{median}*{j}
\operatorname{median}*{p_i\in N_8(x_j)}h_i.
]

Clamp:

[
h_c
===

\operatorname{clip}_{[0.5h_0,,2h_0]}(\hat h_c).
]

### 7.2 Candidate tube and bins

Collect retained feature nodes satisfying

[
-3h_c
\le
(p_i-a_c)\cdot d_c
\le
L_c+3h_c
]

and

[
\left|
(p_i-a_c)
---------

[(p_i-a_c)\cdot d_c]d_c
\right|
\le
6h_c.
]

Define the number of longitudinal bins

[
n_c
===

\min\left(
64,,
\max\left(
4,,
\left\lceil\frac{L_c}{2h_c}\right\rceil
\right)
\right)
]

and bin width

[
\Delta_c=\frac{L_c}{n_c}.
]

### 7.3 Cross-sectional atoms

Within each longitudinal bin:

1. project feature nodes into the plane perpendicular to (d_c);
2. connect projected nodes whose separation is at most (2.5h_c);
3. each connected component becomes one atom.

For atom (z):

* let (x_z) be its (w_i)-weighted centroid;
* let (\rho_z) be its perpendicular distance from the candidate line;
* let (\alpha_z) be the median of the largest (\min(5,m_z)) constituent (w_i) values.

Define candidate-anchored atom strength

[
\beta_z
=======

\frac{\alpha_z}
{1+\left(\frac{\rho_z}{4h_c}\right)^2}.
]

This permits a moderately displaced V16 candidate to find the visible ridge but prevents it from preferentially claiming a stronger neighboring fold.

### 7.4 Longitudinal track DAG

Create directed edges from an atom in bin (b) to an atom in bin (b+g), where

[
g\in{1,2,3},
]

iff their projected centroids differ by at most

[
3h_c.
]

Thus the track may cross at most two empty bins, but it may start and stop anywhere. No evidence is required at candidate endpoints.

For a path (\pi), define occupied length

[
O_\pi
=====

|\pi|\Delta_c,
]

weighted support mass

[
M_\pi
=====

\Delta_c\sum_{z\in\pi}\beta_z,
]

and mean support strength

[
\mu_\pi
=======

\frac{M_\pi}{O_\pi}.
]

The required physical support is

[
T_c
===

\min\left(
L_c,,
\max(8h_c,0.30L_c)
\right).
]

Define path admissibility

[
A_\pi
=====

\min\left(
\frac{O_\pi}{T_c},
\frac{\mu_\pi}{0.35}
\right).
]

Use dynamic programming indexed by ending atom and occupied-bin count. It must examine every attainable occupied-bin count; merely maximizing total mass is forbidden because a long weak boundary must not defeat a shorter strong path.

The candidate statistic is

[
A_c=\max_\pi A_\pi.
]

The candidate is supported iff

[
A_c\ge1.
]

Equality is `KEEP`.

Path ties are resolved by:

1. larger (\mu_\pi);
2. larger (O_\pi);
3. smaller mean (\rho_z/h_c);
4. smaller canonical forward/reverse path hash.

### 7.5 Why this does not reproduce V69

BRMVT does **not** require:

* an exact triangle-plane section;
* two fitted shelves at every successful station;
* signed normal orientation;
* a fixed normal-turn profile;
* a plateau at three predetermined radii;
* seven successful linked stations;
* success along most of the candidate.

It asks one question: whether a locally high, multiscale, non-boundary geometric ridge supports a connected portion of at least (30%) of the line or (8h_c), whichever is larger.

---

## 8. Duplicate ownership

For every supported candidate, define its support-node set (S_c) as retained feature nodes that are:

* in the selected path bin or an immediately adjacent bin; and
* within (2h_c) of a selected path atom in the candidate-orthogonal plane.

For supported candidates (c_i,c_j), let

[
\theta_{ij}
===========

\arccos|d_i\cdot d_j|.
]

Let (o_{ij}) be the overlap of their selected-path longitudinal intervals divided by the shorter interval length.

Define weighted Jaccard overlap

[
J_{ij}
======

\frac{
\sum_{p\in S_i\cap S_j}w_p
}{
\sum_{p\in S_i\cup S_j}w_p
}.
]

Let (D_{ij}) be the maximum of the two directed median nearest-centroid distances between the selected atom tracks, and let

[
h_{ij}=\sqrt{h_i h_j}.
]

The candidates are duplicates iff all of the following hold:

[
\theta_{ij}\le\frac{\pi}{18},
]

[
o_{ij}\ge0.60,
]

and

[
J_{ij}\ge0.45
\quad\text{or}\quad
D_{ij}\le1.25h_{ij}.
]

All comparator equalities count as duplicate.

For each connected component of the duplicate relation, keep exactly one owner by the following lexicographic key:

1. larger (A_c);
2. larger selected (\mu_\pi);
3. larger (O_\pi);
4. smaller mean (\rho_z/h_c);
5. smaller opaque candidate hash.

The original owner endpoints and direction are emitted unchanged. Internal ridge atoms are never emitted as replacement bend coordinates.

Nearby distinct folds are preserved because their feature-node sets or tracks must remain separated beyond the very narrow (1.25h) fallback, even where their candidate tubes overlap.

---

## 9. Morphology and acquisition behavior

| Condition                   | Frozen BRMVT behavior                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Finite-radius bend          | Covariance variation may peak at coarse scales; the two-best-scale statistic does not require a sharp tangent discontinuity.                |
| Faceted bend                | One or more facet junctions produce the same local variation ridge.                                                                         |
| Shallow crease              | Absolute curvature may be small, but a local percentile and relative top-hat can remain high against adjacent panel samples.                |
| Partial bend                | Only (T_c=\max(8h_c,0.30L_c)), capped by (L_c), must be supported.                                                                          |
| Finite endpoint             | No endpoint predicate exists; the path may start and stop internally.                                                                       |
| Local dropout               | Two empty longitudinal bins may be crossed; boundary-penalized weak nodes may bridge up to normalized graph distance 8.                     |
| Free edge or hole perimeter | Normally has low surface variation; any scan-noise response is boundary-penalized and cannot seed a track without interior strong evidence. |
| Broad panel transition      | Nonzero variation without local top-hat contrast yields (Q_i\approx0).                                                                      |
| Triangulation seam          | Coplanar geometry produces no source-variation ridge; triangle winding is ignored.                                                          |
| Return or hem               | No oriented-normal comparison is used; a rounded or faceted U-turn may produce multiscale variation irrespective of winding or normal sign. |
| Opposing folds              | Curvature sign is unused. Distinct source tracks remain separate.                                                                           |
| Nearby folds                | Candidate anchoring and source-node ownership prevent a candidate from automatically claiming the stronger adjacent track.                  |
| Compound formed region      | A track may move transversely by up to (3h_c) between linked bins and need only be locally line-coherent.                                   |
| Isolated corner or endpoint | It fails the (8h_c) minimum support unless it genuinely continues as a line feature.                                                        |

---

## 10. Atomic aborts and resource caps

Any abort produces **no candidate output**, only one scan-level abort artifact.

### Hard caps

* source points: 200,000;
* triangles: 350,000;
* candidates: 128;
* neighbor references including tie shells: 268,800,000;
* undirected source-graph edges: 8,000,000;
* candidate feature nodes: 8,192 per candidate;
* atoms: 16 per longitudinal bin;
* total atoms: 1,024 per candidate;
* total candidate-node visits: 1,048,576;
* total candidate-DAG edge examinations: 25,000,000;
* predicted scratch allocation before execution: below 8 GiB.

### Abort conditions

* any nonfinite datum;
* invalid triangle index;
* zero-length candidate;
* neighborhood tie-shell cap exceeded;
* covariance or eigenvalue failure;
* reproducible-summation failure;
* graph or candidate cap exceeded;
* more than 16 atoms in any bin;
* internal canonical-hash collision not resolved by full-byte comparison;
* partial output already written before completion;
* peak RSS reaching 16 GiB;
* wall time exceeding 300 seconds.

The engine must write to a temporary artifact and atomically rename it only after the full scan, all candidates, canonical serialization, and integrity hashes have completed.

No cap breach may be converted into candidate-level `DROP`.

---

## 11. Canonical compact audit

The deterministic artifact is canonical UTF-8 JSON, sorted keys, no insignificant whitespace, and floats encoded as C99 hexadecimal strings.

Wall time and observed RSS go into a separate telemetry sidecar and are excluded from the byte-identical replay comparison.

```text
schema
constants_sha256
source_geometry_sha256
input_counts
global:
  point_count
  triangle_count
  candidate_count
  h0_hex
  neighbor_reference_count
  source_graph_edge_count
candidates[] sorted by opaque hash:
  candidate_hash
  length_over_hc_hex
  hc_over_h0_hex
  tube_feature_node_count
  bin_count
  atom_count
  strong_node_count
  weak_bridge_node_count
  boundary_confidence_q50_hex
  boundary_confidence_q90_hex
  selected_occupied_length_over_L_hex
  selected_mean_beta_hex
  admissibility_A_hex
  support_path_sha256
  decision
  duplicate_owner_hash_or_null
  terminal_reason
output_candidate_hashes[]
canonical_output_sha256
```

Permitted terminal reasons are exactly:

```text
KEEP
NO_FEATURE_NODE
NO_ADMISSIBLE_TRACK
DUPLICATE_SUPPRESS
SCAN_ABORT
```

The audit serializes no:

* source coordinate;
* source-point array;
* neighbor list;
* feature-node index list;
* candidate score;
* candidate route;
* filename or identity;
* visual judgment.

`support_path_sha256` hashes an ephemeral canonical path representation and cannot be reversed from the audit.

---

## 12. Source-free fixture suite

All geometry fixtures use (h=1), longitudinal coordinates (x=0,\ldots,64), transverse sampling at unit arc length, and candidate ([(0,0,0),(64,0,0)]), unless specified otherwise.

For a cross-sectional curve (\gamma(s)), the sampled surface is the extrusion

[
P_\gamma
========

{(x,\gamma(s)):x\in{0,\ldots,64},,s\in{-36,\ldots,36}}.
]

These are mathematical and invariance fixtures only. They remain quarantined from transfer evidence, threshold claims, family counts, and promotion, as required by the packet. 

### Mandatory positive geometry fixtures

| ID  | Geometry                                                                                  | Required result                      |
| --- | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| P01 | Exact (90^\circ) two-plane crease                                                         | one `KEEP`                           |
| P02 | Exact (12^\circ) shallow crease                                                           | one `KEEP`                           |
| P03 | (90^\circ), radius (12h), tangent flats                                                   | one `KEEP`                           |
| P04 | Three-facet (60^\circ) formed bend                                                        | one `KEEP`                           |
| P05 | (170^\circ), radius (4h) hem                                                              | one `KEEP`                           |
| P06 | P03 visible only over central (24h) of the (64h) candidate                                | one `KEEP`                           |
| P07 | P03 with two complete (4h) longitudinal dropouts and one one-sided (8h) interval          | one `KEEP`                           |
| P08 | Two opposing (60^\circ) folds, two candidates                                             | two `KEEP`                           |
| P09 | Two distinct parallel folds separated by (3.5h)                                           | two `KEEP`, no duplicate suppression |
| P10 | Crease track with transverse sinusoidal displacement of amplitude (2h) over the candidate | one `KEEP`                           |

### Mandatory negative and ownership fixtures

| ID  | Geometry                                                                 | Required result       |
| --- | ------------------------------------------------------------------------ | --------------------- |
| N01 | Flat interior line                                                       | `NO_FEATURE_NODE`     |
| N02 | Flat free edge                                                           | drop                  |
| N03 | Circular hole perimeter in a flat panel                                  | drop                  |
| N04 | Eroded/dropout edge with deterministic normal jitter bounded by (0.05h)  | drop                  |
| N05 | Uniform cylindrical panel, radius (80h), generator candidate             | drop                  |
| N06 | Coplanar duplicated/alternating triangulation seam                       | drop                  |
| N07 | Strong isolated corner feature of longitudinal extent (6h)               | `NO_ADMISSIBLE_TRACK` |
| N08 | Same physical fold with candidates offset by (0.75h)                     | exactly one owner     |
| N09 | Complete cylinder with an axial candidate                                | drop                  |
| N10 | Two crossing short features, neither with (8h) candidate-aligned support | both drop             |

### Comparator fixtures

1. (r=0.70\Rightarrow u=0); `nextafter(0.70,+∞)` is positive.
2. (r=0.95\Rightarrow u=1).
3. (a=0.01\Rightarrow v=0); `nextafter(0.01,+∞)` is positive.
4. (a=0.10\Rightarrow v=1).
5. (g=100^\circ\Rightarrow B=0).
6. (g=180^\circ\Rightarrow B=1).
7. (w=0.20) is weak; the preceding float is not.
8. (w=0.55) is strong; the preceding float is not.
9. hysteresis distance (8.0) is retained; the succeeding float is not.
10. (A_c=1) is `KEEP`; the preceding float is `DROP`.
11. duplicate angle (10^\circ), overlap 0.60, and Jaccard 0.45 are inclusive.

### Determinism and invariance fixtures

* 32 independent input-point permutations: identical deterministic artifact.
* candidate-array reversal: identical deterministic artifact.
* all candidate endpoints reversed: identical decisions and owner hashes.
* every triangle winding reversed: identical artifact.
* rigid rotations and translations: identical decisions after inverse transformation.
* one arbitrary-plane reflection: identical decisions.
* uniform similarity transforms of (0.8\times) and (1.25\times): identical decisions and dimensionless audit quantities.
* exact 1-to-4 triangle refinement plus exact midpoint insertion on P01 and P03: identical decisions and owner.
* primary/replay: byte-identical deterministic artifact.
* stress: 200,000 points, 350,000 triangles, 128 candidates, all caps respected, RSS below 8 GiB, wall time at most 240 seconds on the locked experiment host.

### Positive-predicate compatibility proof

The positive fixtures are constructed with the following frozen lower bounds on their selected tracks:

| Fixtures |        Strong-track length |           Minimum selected (\beta) | Consequence                     |
| -------- | -------------------------: | ---------------------------------: | ------------------------------- |
| P01–P05  |           at least (0.70L) |                      at least 0.55 | (O/T\ge2.33), (\mu/0.35\ge1.57) |
| P06      |               (24h=0.375L) |                      at least 0.55 | (O/T=1.25), (\mu/0.35\ge1.57)   |
| P07      |  at least (22h) after gaps |                 at least 0.44 mean | (O/T>1.14), (\mu/0.35>1.25)     |
| P08–P09  | at least (0.60L) per track |                      at least 0.50 | each (A>1)                      |
| P10      |           at least (0.60L) | at least 0.44 after radial penalty | (A>1.25)                        |

For example, any strong node with (\rho\le2h_c) has

[
\beta
\ge
\frac{0.55}{1+(2/4)^2}
======================

0.44

>

0.35.
]

P06 supplies

[
O=24h,\qquad
T=0.30(64h)=19.2h,
]

so

[
\frac{O}{T}=1.25>1.
]

P07’s empty intervals are each at most two bins, so the (g\le3) DAG edge rule crosses them without counting them as occupied support.

P09’s two source support sets are disjoint and their track separation is (3.5h>1.25h), so neither branch of the duplicate predicate can fire.

Thus no mandatory positive requires mutually incompatible values of boundary confidence, persistence, path length, path strength, gap allowance, or duplicate separation. The fixtures must pass before any real scan is opened; failure is a validity defect and terminates V70 before real execution.

---

## 13. Exposed-real actionability preflight

Use exactly U01–U04 and their already exposed 48 V16 thresholded pre-NMS candidates. No truth, V52 decisions, candidate scores, or prior visual classifications may enter BRMVT.

Run exactly:

1. one primary execution;
2. one replay from clean process state;
3. no parameter edit, rescue, replacement input, or third run.

### Mechanical pass gates

All must pass:

1. all source-free fixtures passed beforehand;
2. no scan abort;
3. primary/replay deterministic artifacts byte-identical;
4. final post-NMS retained total is in the inclusive range

[
12\le R\le38;
]

5. at least two of four scans have at least one changed final candidate;
6. every scan with (n\ge3) inputs retains at least

[
\left\lceil0.20n\right\rceil
]

final candidates;
7. at least six nonduplicate kept candidates have

[
A_c\ge1.10;
]

8. at least six support-dropped candidates have

[
A_c\le0.90;
]

9. each primary and replay completes within 240 seconds;
10. each run remains below 8 GiB peak RSS.

The (12)–(38) range is an actionability guard, not an accuracy target: V69’s one retained candidate fails the lower bound, while a near-no-op 48-retained method fails the upper bound. These counts cannot be used to alter an inference constant.

Failure of any gate terminally retires V70 without visual output adjudication or fresh-scan use.

---

## 14. Native-resolution raw-first visual protocol

Only after the mechanical preflight passes:

### Phase A — blind raw-geometry judgment

For each of the 48 input candidates, create a locked native-resolution visual packet containing:

* full-part context;
* candidate crop;
* orthographic view along the candidate direction;
* two oblique views at (+35^\circ) and (-35^\circ);
* neutral, identical candidate rendering;
* all available source geometry at native resolution.

Do **not** show:

* BRMVT keep/drop status;
* colors associated with decisions;
* support nodes or paths;
* reason codes;
* V52 or V69 output;
* candidate score;
* duplicate owner.

The reviewer records exactly one frozen category:

```text
VISIBLE_PERSISTENT_CREASE
NO_VISIBLE_PHYSICAL_CREASE
AMBIGUOUS
```

`VISIBLE_PERSISTENT_CREASE` requires a visible local material turn or curvature ridge over at least one quarter of the candidate’s displayed length in at least two views. A silhouette, hole edge, flat panel, triangulation seam, or isolated endpoint is `NO_VISIBLE_PHYSICAL_CREASE`.

Hash the completed blind manifest before proceeding.

### Phase B — reveal and inspect

Reveal BRMVT outputs, support tracks, boundary confidence, and terminal reasons.

The visual gate passes only if all hold:

1. at least eight support-dropped candidates were preclassified `NO_VISIBLE_PHYSICAL_CREASE`;
2. no more than two support-dropped candidates were preclassified `VISIBLE_PERSISTENT_CREASE`;
3. at least 85% of preclassified `VISIBLE_PERSISTENT_CREASE` candidates survive final composition;
4. no more than 35% of accepted, nonambiguous candidates were preclassified `NO_VISIBLE_PHYSICAL_CREASE`;
5. no pair preclassified as two nearby distinct physical creases is collapsed into one output;
6. every changed candidate is inspected at native resolution, not only a montage or downsample.

The blind categories are never fed back into BRMVT, its constants, or a rerun. Failure terminally retires V70.

---

## 15. Fresh physical-scan one-shot

Only if both the mechanical and visual exposed-real gates pass unchanged:

1. open exactly one reserved physical scan from the sealed new part;
2. run one primary and one replay;
3. freeze canonical outputs, audit, source hash, render artifacts, and blind raw-visual manifest;
4. perform the same raw-first visual protocol;
5. open the scan’s truth exactly once only after all artifacts are frozen;
6. score once;
7. perform no rescue, threshold edit, candidate replacement, or rerun.

### Fresh-scan retain gates

All are mandatory:

* primary/replay byte-identical;
* both jobs complete successfully;
* peak RSS below 16 GiB;
* each execution no more than 300 seconds;
* exact predicted bend count;
* line precision at least 0.90;
* line recall at least 0.90;
* line F1 at least 0.90;
* render-admissible marker accuracy at least 0.99;
* recall for the fresh family at least 0.80;
* no visually confirmed nearby-distinct-fold collapse;
* no candidate attributable solely to a silhouette, open boundary, hole perimeter, or isolated endpoint.

Any failed gate terminally retires `BRMVT_V1`.

Passing this one scan yields only `RETAIN_RESEARCH_ONLY`; it is not promotion evidence by itself.

---

## 16. Final locked promotion gates

After a fresh pass, promotion still requires the unchanged benchmark of at least 30 scans and 15 families:

* exact-count accuracy at least 90%;
* precision at least 90%;
* recall at least 90%;
* F1 at least 90%;
* render-admissible marker accuracy at least 99%;
* deterministic replay 100%;
* peak RSS below 16 GiB;
* P95 latency at most 300 seconds;
* job success at least 99.5%;
* no major family below 80% recall;
* zero truth, identity, filename, expected-count, CAD, score, provenance, or post-hoc tuning leakage.

---

## 17. Scientific grounding and inference boundary

**Literature-backed:** Pauly, Keiser, and Gross define local covariance surface variation as the smallest covariance eigenvalue divided by the eigenvalue sum, treat neighborhood size as a discrete scale, use persistence across scales, and reconstruct line-type features through hysteresis and a graph. Their paper also describes an angular-gap test in a projected tangent neighborhood for detecting surface boundaries. ([CGL ETHZ][1])

**Literature-backed:** Pottmann and coauthors motivate integral or neighborhood-averaged geometric descriptors by their robustness to noise and useful multiscale behavior, and explicitly associate persistence over several kernel radii with higher-confidence feature detection. ([UT Austin Computer Science][2])

**Sherman-specific inference:** The local mid-percentile, relative top-hat, (70%) boundary penalty, strong/weak constants, normalized hysteresis radius, candidate DAG, 30%/8-spacing support rule, duplicate relation, actionability count range, and all visual/retain gates are frozen V70 hypotheses. The cited literature does not establish those constants, does not validate them for this corpus, and does not show that BRMVT will attain the requested benchmark. The one-shot protocol above is the falsification of that hypothesis.

[1]: https://cgl.ethz.ch/Downloads/Publications/Papers/2003/Pau03b/Pau03b.pdf "https://cgl.ethz.ch/Downloads/Publications/Papers/2003/Pau03b/Pau03b.pdf"
[2]: https://www.cs.utexas.edu/~huangqx/integral_cagd15.pdf "https://www.cs.utexas.edu/~huangqx/integral_cagd15.pdf"
