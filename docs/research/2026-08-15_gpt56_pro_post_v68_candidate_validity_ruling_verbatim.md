EXECUTE_ONE

## 1. Frozen architecture

**Version:** `V69_CANDIDATE_ALIGNED_MULTISCALE_NORMAL_TURN_PERSISTENCE_V1`
**Abbreviation:** `CAMNTP_V1`

`CAMNTP_V1` is a **candidate-local, deletion-only validity veto**. It intersects the actual triangle mesh with planes normal to each candidate line, measures accumulated surface-normal turn across each resulting sheet cross-section at fixed relative scales, and retains the candidate only when the turn:

1. is centered near the candidate;
2. has material continuation on both sides;
3. is not supported by an open/nonmanifold boundary;
4. forms a stable plateau across three consecutive scales; and
5. persists through a sufficient longitudinal run.

It does not compare candidate pairs, estimate duplicate ownership, modify candidate endpoints, rank candidates, create new candidates, or reuse the V67/V68 bridge-mode mechanism.

**Insertion point:** immediately after the frozen V16/V52 thresholded finite-segment stream is canonicalized, and immediately before the one existing ordinary NMS call.

For each thresholded candidate:

* `KEEP` means pass it unchanged to ordinary NMS.
* `VETO` means delete it before ordinary NMS.
* Ordinary NMS is still called exactly once.
* Candidate score and route provenance are neither read nor copied into the V69 process.

A scan without usable triangle adjacency receives `ABSTAIN_NO_MESH`; no V69-modified output may be committed and that scan cannot satisfy the actionability or fresh-scan gate.

---

## 2. Allowed and forbidden inputs

### Allowed

For one scan:

* at most 200,000 bounded source points or vertices;
* at most 350,000 triangles;
* triangle adjacency and exact source coordinates;
* finite candidate endpoints (a,b\in\mathbb R^3);
* candidate opaque hash;
* source/mesh opaque hashes;
* deterministic in-memory spatial indices;
* compact derived scalar summaries.

The point set may be used upstream to construct the already-available mesh. `CAMNTP_V1` itself uses the triangle mesh and adjacency, not point-density statistics.

### Forbidden

The V69 process must not read, infer, branch on, serialize, or use:

* candidate score, rank, confidence, threshold margin, or route provenance;
* V16/V52 internal feature vectors;
* filename, part identity, family identity, acquisition batch, expected count, or scan ordinal;
* CAD, nominal geometry, labels, truth, pseudo-truth, benchmark feedback, or prior visual verdicts;
* V67/V68 pair outcomes;
* any candidate other than the one currently being evaluated, except that the unchanged downstream NMS receives the surviving stream;
* source coordinates in the persistent audit/cache.

Candidate order is not an input signal. All candidates are evaluated independently and the surviving stream is restored to the frozen upstream canonical order before NMS.

---

## 3. Deterministic preprocessing and invariance

### 3.1 Mesh canonicalization

1. Reject the scan atomically if any source coordinate is nonfinite.
2. Normalize signed zero to positive zero.
3. Weld vertices only when all three coordinates are bitwise equal after signed-zero normalization. No tolerance welding is allowed.
4. Remove exact duplicate geometric triangles after sorting their three welded coordinate tuples.
5. Discard exactly zero-area triangles.
6. Build the undirected edge-incidence map:

   * one incident triangle: open-boundary edge;
   * two incident triangles: ordinary manifold edge;
   * zero or more than two incident triangles: invalid/nonmanifold edge.
7. Build two-manifold face components using only two-incident edges.
8. Orient each component consistently by deterministic breadth-first propagation. A component with an orientation contradiction is invalid and cannot provide support.
9. A global reversal of an otherwise valid component is immaterial to all predicates below.
10. Build:

    * a triangle BVH;
    * a boundary/nonmanifold-edge BVH for each component.

BVHs are accelerators only. Exact result sets are sorted and deduplicated before measurement; traversal order cannot change a decision.

### 3.2 Candidate canonical quantities

For candidate endpoints (a,b):

[
L=\lVert b-a\rVert,\qquad u=\frac{b-a}{L}.
]

A nonfinite or zero-length candidate causes a scan-level `ABORT_MALFORMED_CANDIDATE_STREAM`, not a candidate-local deletion.

Use exactly 17 stations. For (i\in{0,\ldots,16}),

[
t_i=\frac{i+1}{18},
]

[
p_i=a+t_i(b-a)+(i-8)2^{-42}L,u,
]

and the cross-section plane is

[
P_i={x:u\cdot(x-p_i)=0}.
]

The tiny symmetric displacement avoids repeated vertex-plane coincidences while preserving endpoint-reversal symmetry: reversing (a,b) maps station (i) to station (16-i) and produces the same physical plane.

The fixed scale ladder is

[
r_j=L,2^{j/2-9},\qquad j=0,\ldots,14.
]

Thus (r_j/L) is:

| (j) |        (r_j/L) |
| --: | -------------: |
|   0 |    0.001953125 |
|   1 | 0.002762135864 |
|   2 |     0.00390625 |
|   3 | 0.005524271728 |
|   4 |      0.0078125 |
|   5 | 0.011048543456 |
|   6 |       0.015625 |
|   7 | 0.022097086912 |
|   8 |        0.03125 |
|   9 | 0.044194173824 |
|  10 |         0.0625 |
|  11 | 0.088388347648 |
|  12 |          0.125 |
|  13 | 0.176776695297 |
|  14 |           0.25 |

All geometric thresholds are ratios of (L) or (r_j), or are angles or dimensionless ratios. Therefore the geometric decision is translation-, rotation-, reflection-, endpoint-reversal-, and positive-similarity-invariant. Resource-cap aborts are operational outcomes, not geometric candidate decisions.

### 3.3 Exact PL cross-sections

At station (i), intersect (P_i) with triangles whose exact plane intersection enters the closed ball

[
B(p_i,3L/8).
]

Use filtered exact plane-side predicates followed by exact-topology segment assembly.

A usable intersection component must be a nonbranching open or closed PL curve. Components that touch a coplanar triangle patch, a nonmanifold junction, or a degree-(>2) cross-section vertex are discarded for that station.

For each component (\Gamma):

1. Find its closest point (q) to (p_i).
2. Require the closest point to be unique to within (2^{-40}L).
3. Require initially

[
\lVert q-p_i\rVert\le L/64.
]

4. Parameterize (\Gamma) by signed arc length (s), with (s=0) at (q). Either orientation may be used; every predicate is invariant to reversing (s).

Each arc interval inherits the consistently oriented normal (n(s)) of its containing triangle. Reverse winding sends every (n) to (-n) and leaves every decision below unchanged.

---

## 4. Exact persistent-crease certificate

### 4.1 Projected normal field

For each cross-section interval, project the face normal into the plane perpendicular to (u):

[
\tilde m(s)=n(s)-(n(s)\cdot u)u.
]

It is valid only when

[
\lVert\tilde m(s)\rVert\ge \cos 35^\circ.
]

For valid intervals,

[
m(s)=\frac{\tilde m(s)}{\lVert\tilde m(s)\rVert}.
]

This permits the candidate direction to deviate by as much as (35^\circ) from the local sheet-tangent direction, but prevents an unrelated transverse edge from manufacturing a turn certificate.

For interval (I), define valid-length coverage

[
\eta(I)=\frac{\operatorname{length}{s\in I:m(s)\text{ valid}}}
{\operatorname{length}(I)}.
]

For a scale (r), define:

[
J_r=[-1.5r,1.5r],
]

[
T_r=[-1.25r,1.25r],
]

[
C_r=[-0.5r,0.5r],
]

[
S^-_r=[-1.25r,-0.75r],
\qquad
S^+_r=[0.75r,1.25r].
]

The component must contain the full, nonoverlapping arc (J_r). For a closed component, the two sides of (J_r) may not reuse any PL segment.

Require:

[
\eta(J_r)\ge0.95,\qquad
\eta(S^-_r)\ge0.95,\qquad
\eta(S^+_r)\ge0.95.
]

### 4.2 Bilateral side plateaus

For either shelf (S), define the length-weighted projected-normal resultant

[
M(S)=\int_S m(s),ds,
]

[
R(S)=\frac{\lVert M(S)\rVert}
{\operatorname{valid\ length}(S)},
\qquad
\bar m(S)=\frac{M(S)}{\lVert M(S)\rVert}.
]

Require:

[
R(S^-_r)\ge0.965,
\qquad
R(S^+_r)\ge0.965.
]

Let

[
m^-_r=\bar m(S^-_r),\qquad m^+_r=\bar m(S^+_r).
]

The bilateral turn angle is

[
\theta_r=
\arccos!\left(
\operatorname{clamp}(m^-_r\cdot m^+_r,-1,1)
\right).
]

Require

[
\theta_r\ge8^\circ.
]

No upper-angle veto is imposed; a near-(180^\circ) hem remains eligible.

### 4.3 Accumulated normal turn

At each normal discontinuity along (T_r), let (m_k,m_{k+1}) be the projected normals immediately before and after the discontinuity. Define the signed principal turn around the candidate axis:

[
\delta_k=
\operatorname{atan2}
\left(
u\cdot(m_k\times m_{k+1}),
m_k\cdot m_{k+1}
\right).
]

Let (s_k) be its arc-length position. Define:

[
V_r=\sum_{s_k\in T_r}|\delta_k|,
]

[
A_r=\left|\sum_{s_k\in T_r}\delta_k\right|,
]

[
V^C_r=\sum_{s_k\in C_r}|\delta_k|.
]

Require all of:

[
0<V_r\le1.25\pi,
]

[
\frac{A_r}{V_r}\ge0.70,
]

[
\frac{V^C_r}{V_r}\ge0.70,
]

[
|A_r-\theta_r|
\le
\max(4^\circ,0.20\theta_r).
]

The first ratio rejects oscillatory triangulation noise and opposing local turns. The second rejects curvature distributed broadly across the complete cross-section rather than concentrated near the candidate. The endpoint-match condition rejects multi-turn profiles whose net or accumulated turn does not agree with the two side plateaus.

Define the absolute-turn barycenter

[
\beta_r=
\frac{\sum_{s_k\in T_r}|\delta_k|,s_k}{V_r}.
]

Require

[
|\beta_r|\le0.20r.
]

This makes the measured crease coincide with the candidate rather than merely occur somewhere in its neighborhood.

### 4.4 Candidate-centering and boundary/dropout vetoes

For the component closest point (q), require at scale (r):

[
\lVert q-p_i\rVert
\le
\min(L/128,r/4).
]

Let (B_\Gamma) be all open-boundary, nonmanifold, invalid-face, and orientation-contradiction edges belonging to the selected manifold component. Require

[
\operatorname{dist}(q,B_\Gamma)\ge1.25r.
]

Also require that the entire arc (J_r) remain on the same orientable two-manifold component and cross no edge in (B_\Gamma).

Consequences:

* an open outer perimeter has no bilateral arc or has zero boundary distance;
* an open hole/window rim has zero boundary distance;
* a dropout slit or partial meshing tear is treated as a boundary;
* a nonmanifold-only seam cannot support a scale;
* a legitimate return or hem can pass at a scale smaller than its observed flange continuation;
* endpoint boundaries can invalidate end stations without invalidating a sufficiently long interior run.

A watertight aperture rim must additionally survive the accumulated-turn and side-continuation tests; a thin wall followed by a second turn enters (T_r), destroys endpoint agreement, or prevents a three-scale plateau.

### 4.5 Scale pass

A component passes scale (r_j) only when every predicate in Sections 4.1–4.4 passes.

No failed predicate may be ignored or imputed.

### 4.6 Three-scale persistence plateau

A component is station-valid only if there exists (j\in{0,\ldots,12}) such that scales

[
r_j,\quad r_{j+1},\quad r_{j+2}
]

all pass.

For those three scales, additionally require:

[
\max_k\theta_{r_k}-\min_k\theta_{r_k}
\le
\max!\left(
4^\circ,
0.20,\operatorname{median}*k\theta*{r_k}
\right).
]

For two scales (r_a,r_b), define the unordered shelf-pair drift

[
D_{ab}=
\min\left{
\begin{aligned}
&\max\bigl(
\angle(m^-*{r_a},m^-*{r_b}),
\angle(m^+*{r_a},m^+*{r_b})
\bigr),\
&\max\bigl(
\angle(m^-*{r_a},m^+*{r_b}),
\angle(m^+*{r_a},m^-*{r_b})
\bigr)
\end{aligned}
\right}.
]

Require

[
\max_{a,b}D_{ab}\le12^\circ.
]

A sharp or finite-radius bend reaches a turn plateau once the scale encloses the bend and reaches its side facets. A broad cylinder continues changing its shelf normals and turn angle as scale grows.

If multiple plateaus exist on one component, select the one with smallest starting index (j). If several components are station-valid, select the component with smallest (\lVert q-p_i\rVert/L). Distances within (2^{-40}L) are an ambiguity, and the station fails with `AMBIGUOUS_COMPONENT`; input order may not break the tie.

Record the selected representative scale and angle:

[
r_i^*=r_{j+1},
\qquad
\theta_i^*=\operatorname{median}
(\theta_{r_j},\theta_{r_{j+1}},\theta_{r_{j+2}}).
]

### 4.7 Longitudinal curve persistence

A station is `PASS` when it has one unique selected component and plateau.

For adjacent passing stations (i,i+1), define

[
e_i=q_i-p_i.
]

They are linked only when:

1. their selected arcs belong to the same orientable manifold component;
2.

[
\lVert e_{i+1}-e_i\rVert
\le
0.50\min(r_i^*,r_{i+1}^*)+L/1024;
]

3.

[
\max\left(
\frac{r_i^*}{r_{i+1}^*},
\frac{r_{i+1}^*}{r_i^*}
\right)\le4.
]

A linked run consists of consecutive passing stations satisfying these adjacency tests.

The candidate is `KEEP` exactly when:

[
N_{\mathrm{pass}}\ge9
]

and

[
N_{\mathrm{longest\ linked\ run}}\ge7.
]

Otherwise it is `VETO`.

Equality passes for every inclusive comparator.

Nine of seventeen stations allow a finite or partially observed fold to survive despite failed endpoint or dropout stations. A seven-station linked run spans one-third of the candidate length between its first and last station, preventing scattered corners from aggregating into a false certificate.

---

## 5. Deterministic failure behavior

### Candidate-local geometric failure

Any missing geometric predicate causes that scale or station to fail. Failure to reach the (9/7) longitudinal requirements causes `VETO`.

### Scan-level atomic abort

The following cause scan-level abort:

* unsupported or absent triangle adjacency;
* malformed/nonfinite candidate stream;
* memory allocation failure;
* arithmetic overflow or nonfinite derived scalar;
* any resource cap exceeded;
* canonical serializer failure;
* internal topology inconsistency not represented by the specified nonmanifold logic.

On abort:

1. no partial V69 candidate stream is committed;
2. no candidate is silently kept or deleted by V69;
3. the unmodified frozen baseline result remains the only result;
4. the abort record is committed;
5. during fixtures, actionability, or the fresh one-shot, the version is failed rather than rerun.

No threshold rescue, changed representation, or second fresh invocation is permitted.

---

## 6. CPU, visit, memory, and cache caps

All comparisons are inclusive; cap (+1) aborts.

### Input caps

* triangles: (\le350{,}000);
* thresholded candidates: (\le128);
* stations: exactly 17 per candidate;
* scales: exactly 15 per station.

### Local caps

Per station:

* exact intersecting triangles: (\le20{,}000);
* emitted PL intersection segments: (\le20{,}000);
* cross-section components: (\le512).

Per candidate:

* emitted segment records over all stations: (\le120{,}000);
* projected-normal discontinuity records: (\le240{,}000).

Per scan:

* exact triangle-plane tests after BVH filtering: (\le8{,}000{,}000);
* emitted PL segments: (\le2{,}000{,}000);
* projected-normal discontinuity records: (\le4{,}000{,}000).

### Operational caps

* V69 added wall time: (\le120) seconds per scan;
* full baseline-plus-V69 wall time: (\le300) seconds per scan;
* V69 additional peak RSS: (<4) GiB;
* total process peak RSS: (<16) GiB;
* canonical deterministic audit: (\le256) KiB per scan.

The cross-section index is built once. No per-candidate full-mesh normal-atom rescan is permitted.

---

## 7. Canonical audit

Produce two records.

### 7.1 Deterministic decision record

Canonical UTF-8 JSON with sorted keys, integers/strings/booleans only, no floating-point JSON values, no NaN, and no source coordinates.

All numerical summaries are encoded as:

* angles: integer microdegrees;
* ratios: integer parts per million;
* normalized distances: integer parts per billion;
* masks: unsigned integers;
* counts: unsigned integers.

Required top-level fields:

```text
schema
architecture
constant_sha256
implementation_sha256
fixture_sha256
source_mesh_opaque_hash
candidate_stream_opaque_hash
preprocess_status
triangle_count
manifold_component_count
boundary_edge_count
nonmanifold_edge_count
candidate_count
scan_decision_status
candidate_records
decision_sha256
```

Each candidate record, sorted by opaque hash:

```text
candidate_opaque_hash
decision: KEEP | VETO
station_pass_mask_17
linked_run_mask_17
station_pass_count
longest_linked_run
selected_plateau_start_histogram_13
median_turn_microdegrees
median_scale_over_L_ppb
minimum_coherence_ppm
minimum_concentration_ppm
minimum_boundary_ratio_ppm
maximum_center_offset_ratio_ppm
station_failure_histogram
terminal_reason
```

Forbidden audit fields include candidate score, source coordinates, normal vectors, triangle/vertex arrays, filename, identity, and truth.

All NumPy or implementation-language scalar values must be recursively converted to native booleans/integers before canonical serialization.

### 7.2 Operational envelope

A separate non-replay-hashed record contains:

* wall time;
* CPU time;
* peak RSS;
* BVH node visits;
* exact tests;
* segment counts;
* cap headroom;
* job status.

Replay equality applies to the deterministic decision record and surviving candidate stream, not timing measurements.

---

## 8. Frozen source-free fixture suite

These fixtures are analytic contract tests only. They are not corpus surrogates, may not influence threshold selection, and may not support a generalization claim.

### 8.1 Base analytic generator

Unless stated otherwise:

* candidate:

[
c(x)=(x,0,0),\quad x\in[-0.5,0.5],
]

so (L=1);

* surface is an extrusion along (x\in[-1,1]) of a PL cross-section (\gamma(s)), (s\in[-0.5,0.5]);
* longitudinal grid has 37 unequal-offset slabs so no candidate station lies on a longitudinal mesh edge;
* triangles are well formed and consistently orientable;
* the candidate lies at (s=0);
* all straight side continuations extend at least 0.5 arc-length units unless overridden.

A prescribed normal-turn sequence around the (x)-axis is realizable by integrating its PL tangent directions into (\gamma). Thus the comparator fixtures below correspond to realizable extruded PL surfaces, not abstract impossible metric tuples.

### 8.2 Mandatory positive end-to-end fixtures

| ID        | Geometry                                                                                            | Expected  |                                                               |                                       |          |                                       |
| --------- | --------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------- | ------------------------------------- | -------- | ------------------------------------- |
| P01       | Internal sharp (30^\circ) fold                                                                      | KEEP      |                                                               |                                       |          |                                       |
| P02       | Internal sharp fold of exactly (8^\circ)                                                            | KEEP      |                                                               |                                       |          |                                       |
| P03       | Rounded monotone (60^\circ) fold, turn arc length (0.006), 16 equal facets                          | KEEP      |                                                               |                                       |          |                                       |
| P04       | Faceted monotone (75^\circ) fold, turn width (0.008), nine increments                               | KEEP      |                                                               |                                       |          |                                       |
| P05       | Finite (30^\circ) fold active only for (                                                            | x         | \le0.23), flat for (                                          | x                                     | \ge0.25) | KEEP; exactly 9 central stations pass |
| P06       | Partially observed fold: mesh exists only for (                                                     | x         | \le0.25) around the crease, with the candidate still length 1 | KEEP; exactly 9 central stations pass |          |                                       |
| P07a/P07b | Two distinct (45^\circ) folds separated by 0.06, one candidate on each                              | both KEEP |                                                               |                                       |          |                                       |
| P08a/P08b | U-channel with opposing (90^\circ) folds separated by 0.12                                          | both KEEP |                                                               |                                       |          |                                       |
| P09       | (90^\circ) return with observed return-leg arc length 0.06                                          | KEEP      |                                                               |                                       |          |                                       |
| P10       | Rounded (170^\circ) hem, turn width 0.006, observed return leg 0.04, opposing skin present          | KEEP      |                                                               |                                       |          |                                       |
| P11       | Coarsely triangulated rounded (60^\circ) fold with the same PL surface and low longitudinal density | KEEP      |                                                               |                                       |          |                                       |
| P12       | Sharp fold whose spine has transverse displacement (0.002\sin(2\pi x)) from the straight candidate  | KEEP      |                                                               |                                       |          |                                       |

### 8.3 Positive compatibility proof

For (L=1):

[
r_4=0.0078125,\quad
r_5=0.011048543456,\quad
r_6=0.015625,\quad
r_7=0.022097086912.
]

For P01 and P02, the turn is centered and all side regions are planar; scales (r_4,r_5,r_6) pass.

For P03, half of the rounded turn width is (0.003). At (r_4):

[
0.5r_4=0.00390625>0.003,
]

and

[
0.75r_4=0.005859375>0.003.
]

Therefore the complete turn lies inside (C_{r_4}), both shelves lie on straight side facets, and the same holds at (r_5,r_6).

For P04, half-width is (0.004). At (r_5):

[
0.5r_5=0.00552427>0.004,
]

and

[
0.75r_5=0.00828641>0.004.
]

Thus (r_5,r_6,r_7) provide a valid plateau.

For P05 and P06, stations (i=4,\ldots,12) are the nine central stations. They form one nine-station run. At the two outer passing stations, the longitudinal boundary/transition is at least (0.0277) away, exceeding

[
1.25r_6=0.01953125.
]

For P07, the other physical fold is 0.06 away, while the largest bilateral arc needed by the (r_4,r_5,r_6) plateau is

[
1.5r_6=0.0234375.
]

Each candidate therefore has an isolated small-scale certificate.

For P09 and P10, the observed short side lengths, 0.06 and 0.04, both exceed (1.5r_6); their boundary distances also exceed (1.25r_6).

For P12, the maximum adjacent-station offset change is below 0.0007. Using representative scale (r_6), the linkage allowance exceeds

[
0.5r_6+1/1024
=0.0087890625.
]

Its maximum candidate-to-spine displacement, 0.002, is below both (L/128) and (r_5/4) for the selected (r_5,r_6,r_7) plateau.

Every positive has:

* projected-normal coverage (=1);
* shelf resultants (=1);
* monotone-turn coherence (=1);
* central concentration (=1);
* turn barycenter at the candidate, except P12 where it remains within the frozen centering bound;
* no boundary/nonmanifold support inside the selected plateau;
* at least nine passing stations and a linked run of at least seven.

Thus no mandatory positive relies on mutually incompatible predicates.

### 8.4 Mandatory negative end-to-end fixtures

| ID  | Geometry                                                                 | Required terminal cause                                            |
| --- | ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| N01 | Flat interior panel                                                      | `TURN_TOO_SMALL`                                                   |
| N02 | Candidate coincident with open outer perimeter                           | `BOUNDARY_NEAR` or `NO_BILATERAL_ARC`                              |
| N03 | Candidate on one edge of an open rectangular hole                        | `BOUNDARY_NEAR`                                                    |
| N04 | Candidate on a scan-dropout slit                                         | `BOUNDARY_NEAR` or `NO_BILATERAL_ARC`                              |
| N05 | Three-face nonmanifold seam                                              | `NONMANIFOLD_SUPPORT`                                              |
| N06 | Genuine corner/fold present at only three stations                       | `LONGITUDINAL_COUNT`                                               |
| N07 | Uniform cylindrical panel with radius 0.20 and no side plateaus          | `TURN_DIFFUSE`, `SHELF_DRIFT`, or `NO_SCALE_PLATEAU`               |
| N08 | (30^\circ) transition distributed over arc width 0.20                    | `NO_SCALE_PLATEAU`                                                 |
| N09 | Alternating (+6^\circ/-6^\circ) corrugation                              | `TURN_INCOHERENT`                                                  |
| N10 | Long sharp (7.999999^\circ) seam                                         | `TURN_TOO_SMALL`                                                   |
| N11 | Watertight window rim with wall depth 0.004 followed by an opposing turn | `ENDPOINT_TURN_MISMATCH`, `TURN_INCOHERENT`, or `NO_SCALE_PLATEAU` |
| N12 | Real fold displaced 0.008 from the candidate at every station            | `CANDIDATE_OFF_CENTER`                                             |
| N13 | Candidate direction misaligned (40^\circ) from the fold tangent          | `TANGENT_COVERAGE`                                                 |
| N14 | (+30^\circ) then (-30^\circ) bead within the same central band           | `TURN_INCOHERENT` or `ENDPOINT_TURN_MISMATCH`                      |
| N15 | Eight passing stations, including a seven-station run                    | `LONGITUDINAL_COUNT`                                               |
| N16 | Nine passing stations arranged with maximum run six                      | `LONGITUDINAL_RUN`                                                 |

Every negative must produce `VETO`.

### 8.5 Exact comparator-boundary fixtures

For each pair, the equality fixture must pass the named predicate and the below/above fixture must fail it. Other predicates are supplied as passing constants by the realizable PL-strip generator.

| IDs     | Predicate                                                                                         |       |                                       |
| ------- | ------------------------------------------------------------------------------------------------- | ----- | ------------------------------------- |
| B01/B02 | (\theta=8^\circ) passes; (8^\circ-10^{-6}) degrees fails                                          |       |                                       |
| B03/B04 | valid coverage (0.95) passes; (0.95-10^{-6}) fails                                                |       |                                       |
| B05/B06 | shelf resultant (0.965) passes; (0.965-10^{-6}) fails                                             |       |                                       |
| B07/B08 | coherence (0.70) passes; (0.70-10^{-6}) fails                                                     |       |                                       |
| B09/B10 | central concentration (0.70) passes; (0.70-10^{-6}) fails                                         |       |                                       |
| B11/B12 | (                                                                                                 | \beta | /r=0.20) passes; (0.20+10^{-6}) fails |
| B13/B14 | candidate distance (=r/4<L/128) passes; (r/4+10^{-6}r) fails                                      |       |                                       |
| B15/B16 | boundary distance (=1.25r) passes; (1.25r-10^{-6}r) fails                                         |       |                                       |
| B17/B18 | (V=1.25\pi) passes; (1.25\pi+10^{-6}) fails                                                       |       |                                       |
| B19/B20 | endpoint mismatch exactly (\max(4^\circ,0.20\theta)) passes; (+10^{-6}) degrees fails             |       |                                       |
| B21/B22 | three-scale angle spread exactly at its bound passes; bound (+10^{-6}) degrees fails              |       |                                       |
| B23/B24 | shelf-pair drift (12^\circ) passes; (12^\circ+10^{-6}) fails                                      |       |                                       |
| B25/B26 | exactly three consecutive passing scales passes; only two fails                                   |       |                                       |
| B27/B28 | exactly nine passing stations passes count; eight fails                                           |       |                                       |
| B29/B30 | exactly seven linked stations passes run; six fails                                               |       |                                       |
| B31/B32 | offset-link distance exactly at bound links; bound (+10^{-6}L) does not                           |       |                                       |
| B33     | two valid components whose closest distances differ by at most (2^{-40}L) cause station ambiguity |       |                                       |
| B34/B35 | each resource count exactly at cap completes; cap (+1) causes atomic abort                        |       |                                       |

### 8.6 Metamorphic fixtures

All applicable positive and negative fixtures must additionally pass:

* M01: exact replay, identical deterministic decision SHA-256;
* M02: arbitrary vertex permutation;
* M03: arbitrary face permutation;
* M04: arbitrary candidate-stream permutation;
* M05: candidate endpoint reversal;
* M06: global triangle-winding reversal;
* M07: arbitrary rigid rotation and translation;
* M08: global (0.8\times) similarity;
* M09: global (1.25\times) similarity;
* M10: exact PL refinement by splitting every triangle into four coplanar children;
* M11: diagonal flip on every coplanar quadrilateral shelf;
* M12: insertion of an unconnected distant mesh component outside all query balls.

M02–M12 must preserve decisions, station masks, plateau indices, and canonical dimensionless summaries. Expected visit counts may differ for refinement and are not part of semantic equality.

### 8.7 Stress fixture

One source-free stress fixture shall contain:

* exactly 350,000 triangles;
* exactly 128 candidates;
* mixed positive/negative local patches;
* no query or candidate exceeding its local cardinality cap.

It must complete within:

* 120 seconds V69 wall time;
* 4 GiB additional RSS;
* 256 KiB deterministic audit size.

The corresponding (350{,}001)-triangle and 129-candidate fixtures must abort before candidate evaluation.

Any fixture failure terminates `CAMNTP_V1`. Constants may not be repaired.

---

## 9. Real-scan-only actionability preflight

Use only the already exposed U01–U04 real meshes and their existing 48 thresholded candidate segments.

Do not use R001–R008 truth, Gold37, pseudo-truth, candidate scores, or prior manual labels.

### Execution

1. Freeze implementation, constants, fixtures, and hashes.
2. Run U01–U04 in shadow mode:

   * one primary;
   * one required replay;
   * no modified detector result committed.
3. Produce the deterministic audit and all visual artifacts below.
4. Review visuals before opening any pseudo-truth or aggregate score.

### Actionability gates

All must pass:

1. no scan-level abort;
2. primary/replay deterministic decision hashes identical;
3. at least two candidates are proposed for veto;
4. vetoes occur on at least two of U01–U04;
5. no more than 24 of the 48 candidates are vetoed;
6. every proposed veto is visually classified `NO_PERSISTENT_CREASE`;
7. zero proposed vetoes are classified `CLEAR_PHYSICAL_CREASE` or `AMBIGUOUS`;
8. at least four retained candidates across at least two scans are visually classified `CLEAR_PHYSICAL_CREASE`;
9. full wall time, V69 added time, and RSS remain within their caps.

This is only an actionability and destructive-safety test. It is not generalization evidence and produces no threshold changes.

Zero vetoes, vetoes confined to one scan, a single visually supported or ambiguous deletion, mass deletion above 50%, or any operational failure terminates V69 without spending the fresh scan.

---

## 10. Required native-resolution visuals

Render from the full source mesh without decimation, vertex-normal smoothing, hole filling, denoising, or reconstructed surfaces.

Every artifact is 4096 pixels on its longest side.

For every changed candidate, produce:

1. **Raw full-part view:** candidate context, raw flat-shaded triangles, no V69 annotations.
2. **Local raw quartet:**

   * view along the candidate direction;
   * two opposite oblique views;
   * grazing view across the candidate;
   * raw-only panel first and overlay panel second.
3. **Seventeen-station ribbon:** exact PL cross-section at every station, candidate-plane center, selected component if any, (J_r), shelves, boundary edges, and pass/fail status.
4. **Scale panel:** all 15 scale outcomes at the median passing station, or the station with the largest number of passing scales when the candidate is vetoed.
5. **Before/after full result:** unchanged V52/V16 output versus deletion-only V69 output.

Render all retained candidates during U01–U04 preflight as well, so clear survivors can be audited. Candidate display order is opaque-hash order.

### Manual labels

A reviewer assigns one label from raw geometry before viewing V69 reasons:

**`CLEAR_PHYSICAL_CREASE`**

* a coherent surface turn is visibly centered on the marker;
* it extends through at least seven consecutive station locations;
* material visibly continues on both sides at a usable scale;
* it is not merely an open boundary, hole rim, dropout edge, isolated corner, or broad smooth panel.

**`NO_PERSISTENT_CREASE`**

* no centered coherent turn extends through seven consecutive stations; or
* the response is visibly attributable to an open boundary, hole/window perimeter, scan dropout, isolated endpoint/junction, flat panel, broad nonlocalized curvature, or triangulation corrugation.

**`AMBIGUOUS`**

* neither classification is defensible from the raw native-resolution geometry.

Any changed candidate labeled `CLEAR_PHYSICAL_CREASE` or `AMBIGUOUS` fails the architecture. Visual disagreement cannot be resolved by changing constants or rerunning.

---

## 11. Genuinely fresh physical-scan one-shot

The reserved new physical part is opened only after all fixtures and the U01–U04 actionability gate pass unchanged.

### Protocol

1. Commit implementation, constants, fixture suite, actionability record, and all hashes.
2. Freeze the new scan bytes before inference.
3. Run the frozen V52/V16 upstream path once.
4. Run `CAMNTP_V1` once on that thresholded stream.
5. Perform one required deterministic replay from the same frozen bytes. This replay is a determinism check, not a rescue.
6. Before opening truth:

   * inspect every changed native-resolution result;
   * require at least one veto;
   * require no more than 50% of baseline candidates vetoed;
   * require every veto to be `NO_PERSISTENT_CREASE`;
   * require at least one retained candidate to be `CLEAR_PHYSICAL_CREASE`.
7. Only if that visual gate passes, open truth exactly once and score the frozen baseline and V69 output.
8. No rerun, threshold adjustment, candidate replacement, or representation substitution is allowed.

A zero-change fresh result is terminal for V69 despite the earlier actionability result.

---

## 12. Locked retain, retire, and promotion gates

### 12.1 Fresh-scan research retention

Let subscript (0) denote the frozen V52/V16 output and (69) denote the V69-filtered output.

Retain `CAMNTP_V1` as a research intervention only if every condition holds:

[
TP_{69}=TP_0,
]

[
FN_{69}=FN_0,
]

[
FP_{69}\le FP_0-1,
]

[
\operatorname{precision}_{69}>
\operatorname{precision}_0,
]

[
\operatorname{recall}_{69}=
\operatorname{recall}_0,
]

[
F1_{69}>F1_0,
]

and

[
\left|
N_{\mathrm{pred},69}-N_{\mathrm{truth}}
\right|
\le
\left|
N_{\mathrm{pred},0}-N_{\mathrm{truth}}
\right|.
]

Also require:

* identical primary/replay decision hashes and output segments;
* full wall time (\le300) seconds;
* V69 added wall time (\le120) seconds;
* total peak RSS (<16) GiB;
* V69 additional RSS (<4) GiB;
* zero job errors;
* zero true-positive deletion on the fresh family;
* zero visually supported or ambiguous deletion.

Because V69 is deletion-only, any lost true positive, increased false negative count, worsened exact-count error, or non-action retires the version immediately.

### 12.2 Later locked-benchmark architecture safety

On any later locked benchmark, compare the identical upstream system with and without V69. Retaining V69 requires:

* identical TP and FN counts globally;
* identical TP and FN counts within every family;
* strictly fewer FP globally;
* strictly higher precision and F1;
* unchanged recall;
* exact-count accuracy not lower, with at least one case improved;
* no major family recall reduction;
* 100% deterministic replay;
* operational caps met.

This ablation requirement prevents aggregate precision gains from concealing a family-specific true-bend deletion.

### 12.3 Final F1 promotion

Promotion of the complete system still requires, on at least 30 scans and 15 families:

* exact bend-count accuracy (\ge90%);
* line precision (\ge90%);
* line recall (\ge90%);
* line F1 (\ge90%);
* render-admissible marker accuracy (\ge99%);
* deterministic replay (=100%);
* P95 wall time (\le300) seconds;
* peak RSS (<16) GiB;
* job success (\ge99.5%);
* every major family recall (\ge80%);
* no truth, identity, filename, count, CAD, or post-hoc leakage.

Research retention is not promotion.

---

## 13. Causal and scientific basis

The relevant literature supports three ingredients, but not the Sherman-specific constants or outcome:

* Multi-scale feature analysis has been used to identify line-type structure on unorganized point-sampled surfaces, supporting the decision to require evidence at multiple neighborhood sizes rather than trust one local curvature estimate. ([Wiley Online Library][1])
* Integral-invariant work explicitly studies finite-neighborhood geometric quantities and their stability, supporting integration over shelves and turn bands rather than pointwise curvature extrema. ([ScienceDirect][2])
* Normal-cycle research defines curvature information for polyhedral surfaces from accumulated face-normal changes, supporting the use of summed PL normal turns across a mesh cross-section. ([ACM Digital Library][3])
* Multi-level ridge/valley extraction demonstrates why scale consistency matters for separating salient surface structure from one-resolution derivative noise. ([Heriot-Watt Research Portal][4])

The **Sherman-specific inference** is that a valid bend candidate should exhibit a localized accumulated normal turn that reaches a stable bilateral plateau and persists along the candidate. None of the cited work proves the (8^\circ), (0.70), (9/17), or seven-station constants, nor does it establish performance on these scans.

This mechanism directly attacks the observed real-domain defect:

* flat panels lack the required turn;
* broad curvature lacks concentration or a stable plateau;
* boundaries, holes, and dropout edges fail bilateral/topological support;
* isolated corners fail longitudinal persistence;
* triangulation oscillation fails turn coherence;
* rounded and faceted bends accumulate many same-direction increments into one stable turn;
* nearby bends remain separable at the smallest valid plateau;
* returns and hems can pass using scales smaller than their observed side continuation.

It can only remove predictions, and every removal must be justified by source geometry before truth is opened.

[1]: https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.00675 "https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.00675"
[2]: https://www.sciencedirect.com/science/article/pii/S0167839608000095 "https://www.sciencedirect.com/science/article/pii/S0167839608000095"
[3]: https://dl.acm.org/doi/10.1145/777792.777839 "https://dl.acm.org/doi/10.1145/777792.777839"
[4]: https://researchportal.hw.ac.uk/en/publications/ridge-valley-lines-on-meshes-via-implicit-surface-fitting/ "https://researchportal.hw.ac.uk/en/publications/ridge-valley-lines-on-meshes-via-implicit-surface-fitting/"
