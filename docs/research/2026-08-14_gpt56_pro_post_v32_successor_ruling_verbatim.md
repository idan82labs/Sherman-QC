# 1. RULING

**AUTHORIZE**

* **Architecture token:** `V33_CANDIDATE_ALIGNED_SURFEL_VARIFOLD_V1`
* **Primitive token:** `PROJECTIVE_NORMAL_FAN_MOMENT_V1`
* **Short names:** `V33_CASV_V1` / `PNFM_V1`

V33 is authorized for **immediate implementation and one frozen, sequential source-free execution**. No additional architecture packet, consultation, or approval round is required.

V33 is a rescue sidecar at the frozen V16 pre-NMS boundary. It may add an unchanged V16 source candidate; it may not refit a line, alter a V16-retained candidate, adjust a score, or suppress anything before the single existing ordinary NMS.

---

# 2. Causal hypothesis

A genuine sheet-metal bend aligned with a V16 candidate should induce a candidate-wide **joint surface measure** with four observable properties:

1. Surface normals form a filled, non-oriented normal fan rather than one plane, two isolated planes, or a closed normal cycle.
2. Normal phase changes primarily with **transverse position across the candidate**, not with axial position along it.
3. Both extreme normal phases possess broad material tails.
4. The low-phase tail, bend core, and high-phase tail each occupy substantial continuous axial measure, even when different skins or phase regions are missing at different axial locations.

The key observable is therefore not a routed connection, current, path, edge direction, or station vote. It is the area-weighted joint measure

[
\mu_\ell=\sum_f A_f,\delta_{(u_f,q_f,r_f,[n_f])}
]

conditioned on the unchanged candidate line (\ell), where ([n_f]) is the **projective**, winding-independent face normal.

## Why V32’s wall does not apply

V33 shall build:

* no graph;
* no edge list;
* no adjacency;
* no current or potential;
* no shortest or multipath route;
* no edge-direction classification;
* no axial stations;
* no favorable station offset.

A diagonal triangulation edge literally never enters the V33 primitive. Flipping a diagonal can still change the two finite-resolution face normals of a curved quad, so **finite-mesh remeshing sensitivity is not assumed away**. It is instead placed immediately after the clean base fold as the first representation-specific kill test.

Thus V32’s specific failure mechanism—legitimate cross-fold current being counted as axial leakage on diagonal edges—cannot recur. A different failure, excessive sensitivity of area-weighted face-normal moments to remeshing, remains possible and will terminate V33 at Gate 1.

---

# 3. Research basis

Charon and Trouvé’s varifold formulation represents a surface as a measure over spatial position and **unoriented tangent planes**, with numerical encodings for discrete as well as continuous objects. That is the established basis for using area-weighted projective surfels without requiring globally consistent winding and without losing disconnected fragments through orientation cancellation. ([arXiv][1])

Pottmann and colleagues established the use of integral neighborhood moments and covariance/PCA quantities for robust multiscale curvature and feature extraction. Mérigot, Ovsjanikov, and Guibas similarly developed integral covariance measures for robust recovery of curvature, normal directions, and sharp features from sampled geometry. These works support replacing individual edge events with integrated surface moments. ([Geometrie][2])

What is **not** established by those papers is the Sherman-specific certificate:

* conditioning the measure on a frozen V16 candidate;
* deriving a candidate cross-sectional frame from a projective normal tensor;
* comparing transverse normal-phase coupling against axial coupling;
* requiring filled normal-fan support, broad tails, phase-wise axial coverage, and bounded radial multiplicity;
* or any numerical threshold below.

Those are the falsifiable V33 hypothesis, not published bend-detection theory.

---

# 4. Frozen executable lock

## 4.1 Permitted inputs

For each source job V33 may read only:

* the same observed triangle mesh available to the frozen V16 research lane;
* the frozen V16 source scale (s);
* each frozen V16 rescue-eligible pre-NMS candidate:

  * endpoints (p_0,p_1);
  * original candidate index;
  * original candidate score;
  * existing deterministic candidate hash;
* the unchanged V16-retained candidate list for the final legacy-first union.

V33 may recompute face normals from geometry. Stored or scanner-provided normals are ignored.

### Hard input caps

* vertices: at most `250,000`;
* unique nondegenerate triangles: at most `500,000`;
* V33 rescue candidates: at most `512`;
* candidate length: (8s \le L \le 512s);
* candidate-local support triangles: at most `65,536`;
* clipped polygon pieces: at most `131,072` per candidate;
* interval endpoints: at most `262,144` per candidate.

A cap, nonfinite value, integer overflow, or unsupported mesh is a first-class failure. There is no downsampling or fallback representation inside V33.

## 4.2 Forbidden inputs and operations

V33 must not read or derive:

* gold, labels, CAD, drawings, expected bend counts, filenames, family or part identity;
* benchmark outcomes, production feedback, prior V31/V32 outputs, or manually adjudicated target geometry;
* image or renderer pixels;
* graph adjacency, mesh-edge directions, electrical quantities, paths, station masks, or connected-component routes;
* target-specific scale, threshold, or fixture selection;
* a refitted candidate axis, center, endpoint, or score.

No learned model, optimizer, parameter sweep, alternate branch, random seed, or retry is permitted.

## 4.3 Canonical mesh preparation

All constants below are now frozen.

1. Use the V16 scale (s) without recomputation.
2. Quantize vertex coordinates to

[
\delta_x=s/2^{16}
]

using round-to-nearest-even into signed 64-bit integers.

3. Reconstruct each triangle from the quantized coordinates.
4. Discard a triangle only if its quantized area is below

[
A_{\min}=s^2/2^{20}.
]

5. Canonicalize each triangle by lexicographically sorting its three quantized vertices.
6. Remove exact duplicate and reversed-winding duplicate triangles.
7. Sort triangles lexicographically by the nine resulting integer coordinates.
8. Recompute normals from the canonical geometry. Treat (n) and (-n) as the same projective normal.
9. Quantize:

   * normal components to signed `Q1.30`;
   * areas to (s^2/2^{24});
   * interval endpoints to (s/4096);
   * reported correlations to (10^{-6});
   * reported angles to (10^{-4}) degrees.
10. All accumulation is in canonical triangle order using checked signed 128-bit integer sums where applicable.
11. Symmetric eigendecomposition uses a pinned implementation with 16 fixed Jacobi sweeps and pivot order `(01, 02, 12)`.
12. Angle conversion uses the pinned deterministic math implementation stored under the configuration hash, not the platform’s unconstrained libm.

No mesh adjacency or edge table may be allocated.

## 4.4 Candidate frame

For candidate length (L=|p_1-p_0|), define

[
a=(p_1-p_0)/L,\qquad o=(p_0+p_1)/2,\qquad P=I-aa^\top.
]

The broad frame-support set contains every canonical triangle that:

* overlaps the axial range ([-2s,L+2s]); and
* has exact triangle-to-candidate-segment distance at most (16s).

Using full triangle area weights, compute

[
M=\frac{1}{A_B}\sum_{f\in B} A_f,P n_fn_f^\top P.
]

Let (m) be the principal eigenvector of (M) in (a^\perp), and let

[
b=a\times m.
]

The signs of (a,m,b) are intentionally not semantically fixed. Every downstream predicate is sign-symmetric.

The normalized in-plane eigengap is

[
g=\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}.
]

## 4.5 Exact candidate-local clipping

Clip each support triangle as a polygon against this six-plane prism:

[
-s\le u\le L+s,\qquad |q|\le12s,\qquad |r|\le12s,
]

where

[
u=(x-p_0)\cdot a,\quad q=(x-o)\cdot b,\quad r=(x-o)\cdot m.
]

Use deterministic Sutherland–Hodgman clipping in the fixed plane order:

1. (u\ge-s)
2. (u\le L+s)
3. (q\ge-12s)
4. (q\le12s)
5. (r\ge-12s)
6. (r\le12s)

Quantize every generated clipping vertex to (\delta_x) immediately after each plane.

For coverage measurements, clip again to (0\le u\le L).

## 4.6 Projective normal phase

For every clipped face piece:

[
\tilde n=\frac{Pn}{|Pn|}.
]

Choose its projective representative so that (\tilde n\cdot m\ge0); an exact zero is broken by requiring (\tilde n\cdot b\ge0).

Define

[
z=\tilde n\cdot b,\qquad
\theta=\arcsin(\operatorname{clamp}(z,-1,1)),
]

so (\theta\in[-90^\circ,90^\circ]).

The following are accumulated from exact clipped-polygon area moments:

* (A,\int q,\int q^2,\int u,\int u^2);
* (\int z,\int z^2,\int qz,\int uz);
* a 256-bin area histogram over (\theta);
* tail-specific 256-bin (q) histograms over ([-12s,12s]);
* a `16 × 48` (q/r) occupancy array;
* continuous axial interval sets.

No surfel, face identifier, polygon, histogram, or interval list is serialized.

## 4.7 Derived groups

Let (\theta_{05}) and (\theta_{95}) be area-weighted phase percentiles and

[
\Delta_\theta=\theta_{95}-\theta_{05}.
]

Define:

* low tail (T_-):

[
\theta\le\theta_{05}+0.10\Delta_\theta;
]

* bend core (C):

[
\theta_{05}+0.35\Delta_\theta
\le\theta\le
\theta_{05}+0.65\Delta_\theta;
]

* high tail (T_+):

[
\theta\ge\theta_{95}-0.10\Delta_\theta.
]

The transverse and axial mixed correlations are

[
\rho_{qz}=
\frac{\operatorname{Cov}(q,z)}
{\sqrt{\operatorname{Var}(q)\operatorname{Var}(z)}},
\qquad
\rho_{uz}=
\frac{\operatorname{Cov}(u,z)}
{\sqrt{\operatorname{Var}(u)\operatorname{Var}(z)}}.
]

A zero or nonfinite denominator fails.

## 4.8 Fixed first-failure predicate

Predicates are evaluated in exactly this order. Equality passes.

| Order | Reason token               | Locked pass condition                                                                                                                        |           |          |           |          |
| ----: | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------- | -------- | --------- | -------- |
|     1 | `PNFM_INPUT_CAP`           | All input, scratch, interval, and numeric caps respected                                                                                     |           |          |           |          |
|     2 | `PNFM_AREA_DENSITY`        | (A/(Ls)\ge6.0)                                                                                                                               |           |          |           |          |
|     3 | `PNFM_FRAME_GAP`           | (g\ge0.12)                                                                                                                                   |           |          |           |          |
|     4 | `PNFM_AXIS_NORMALITY`      | (\sum A_f(n_f\cdot a)^2/A\le0.10)                                                                                                            |           |          |           |          |
|     5 | `PNFM_FAN_SPAN`            | (24^\circ\le\Delta_\theta\le135^\circ)                                                                                                       |           |          |           |          |
|     6 | `PNFM_FAN_FILL`            | At least 12 of 16 equal bins over ([\theta_{05},\theta_{95}]) each hold at least (0.0025A), with no run of more than two empty interior bins |           |          |           |          |
|     7 | `PNFM_TRANSVERSE_COUPLING` | (                                                                                                                                            | \rho_{qz} | \ge0.80) |           |          |
|     8 | `PNFM_AXIAL_DOMINANCE`     | (                                                                                                                                            | \rho_{qz} | -        | \rho_{uz} | \ge0.45) |
|     9 | `PNFM_TAIL_MASS`           | (A(T_-)\ge0.12A), (A(T_+)\ge0.12A), and (A(C)\ge0.04A)                                                                                       |           |          |           |          |
|    10 | `PNFM_TAIL_WIDTH`          | Weighted (q_{90}-q_{10}\ge2.5s) independently in both tails                                                                                  |           |          |           |          |
|    11 | `PNFM_TAIL_SEPARATION`     | Absolute separation of the two tail median-(q) values is at least (4.0s)                                                                     |           |          |           |          |
|    12 | `PNFM_AXIAL_COVERAGE`      | Overall union coverage (\ge0.80L); each tail (\ge0.55L); core (\ge0.45L); overall max gap (\le0.20L); each group max gap (\le0.40L)          |           |          |           |          |
|    13 | `PNFM_RADIAL_MULTIPLICITY` | No occupied central (q)-bin contains more than two persistent (r)-runs                                                                       |           |          |           |          |
|    14 | `PNFM_PASS`                | Candidate is emitted unchanged                                                                                                               |           |          |           |          |

For radial multiplicity:

* use 16 fixed (q) bins and 48 fixed (r) bins over ([-12s,12s]);
* inspect only the central 12 (q) bins;
* a (q) bin is occupied when it contains at least (0.01A);
* an (r) bin is active when it contains at least the larger of (0.005A_q) and (0.0002A);
* fill one empty (r) bin only when both immediate neighbors are active;
* discard an (r)-run carrying less than (0.05A_q);
* three remaining runs fail.

This is aggregate layer multiplicity, not skin pairing or mate assignment.

## 4.9 Insertion, ordering, duplicates, and NMS

* Every original V16-retained candidate remains untouched.
* V33 evaluates only the frozen V16 rescue-eligible candidates.
* `PNFM_PASS` emits the original candidate line and original score byte-for-byte.
* It does not suppress duplicate candidates.
* Final input order is:

  1. original V16-retained candidates in frozen order;
  2. V33 additions in original frozen candidate-index order;
  3. candidate hash as the final exact tie break.
* Invoke the existing ordinary NMS **once and only once**, with no changed radius, score, tie rule, or geometry.
* A V16 legacy candidate therefore wins an exact or ordinary-NMS duplicate against a V33 addition.

---

# 5. Output and cache lock

V33 writes one fixed header followed by one fixed record per evaluated rescue candidate.

```text
PNFMFileHeaderV1 — 64 bytes
  magic[8]                 = "PNFMV1\0"
  version_u16              = 1
  record_size_u16          = 192
  record_count_u32
  source_sha256[32]
  config_sha256_prefix[16]

PNFMRecordV1 — 192 bytes
  candidate_sha256[32]
  decision_reason_u16
  flags_u16
  metrics_i32[24]
  passthrough_line_sha256[32]
  reserved_zero[28]
```

The 24 metrics, in order, are:

1. support-face count
2. clipped-piece count
3. area density × (10^6)
4. frame gap × (10^6)
5. axis residual × (10^6)
6. fan span in (10^{-4}) degrees
7. fan-fill count
8. (\rho_{qz}\times10^6)
9. (|\rho_{uz}|\times10^6)
10. transverse-minus-axial margin × (10^6)
11. low-tail mass fraction × (10^6)
12. high-tail mass fraction × (10^6)
13. core mass fraction × (10^6)
14. low-tail width in (s/2^{16})
15. high-tail width in (s/2^{16})
16. tail-median separation in (s/2^{16})
17. overall axial coverage × (10^6)
18. low-tail coverage × (10^6)
19. core coverage × (10^6)
20. high-tail coverage × (10^6)
21. overall maximum gap in (s/2^{16})
22. worst group maximum gap in (s/2^{16})
23. maximum radial-run count
24. predicate bit mask

Cache size is exactly

[
64+192N\ \text{bytes}.
]

No source points, vertices, faces, support identities, histograms, or interval endpoints may be serialized.

---

# 6. Ordered source-free fixtures and stop gates

All fixture geometry, candidate records, expected decisions, renderer settings, and hashes must be frozen before the first PNFM call. The original V32 positive is reused byte-for-byte; it is not regenerated or replaced.

## Gate 1 — cheapest decisive representation test

Run sequentially and stop at the first mismatch:

1. **`SF01_FOLD60_MESH_C2`**
   Exact frozen V32 clean 60-degree fold and candidate.
   Expected: `pre-NMS=1`, `final=1`.

2. **`SF01_RM_ALL_DIAGONALS_FLIPPED`**
   Same vertices and physical surface; every eligible parametric quad receives the opposite diagonal.
   Expected: `1/1`.

3. **`SF01_RM_CHECKERBOARD_DIAGONALS`**
   Same vertices and surface; deterministic alternating diagonals.
   Expected: `1/1`.

4. **`SF01_RM_SKEW_LATTICE`**
   Same analytic C2 fold, line, scale, length, tails, and opposite skin; parameter rows use a frozen (3s/8) axial stagger and deterministic constrained triangulation.
   Expected: `1/1`.

In addition to all four passing the absolute predicate, each remesh variant must remain within the following distance from base SF01:

* fan span: at most (1.5^\circ);
* (|\rho_{qz}|): at most `0.035`;
* normalized area density: at most `3%`;
* each coverage fraction: at most `0.03`;
* each tail width: at most (0.5s);
* radial maximum-run count: identical.

The unchanged candidate line must be byte-identical before NMS.

Any failure produces `V33_REMESH_INVARIANCE_STOP`. No later fixture runs.

## Gate 2 — new-mechanism adversary

**`SF_PNFM_ADV01_AXIAL_TWISTED_FAN`**

Construct a ruled/twisted strip with:

* the same (24s) candidate length;
* the same total area density;
* the same projective normal-phase span;
* the same phase histogram;
* broad lateral material;
* full axial surface support;
* no edge, rim, cap, or third layer;

but arrange the normal rotation primarily along (u), not across (q).

Expected:

* `pre-NMS=0`;
* first applicable rejection is `PNFM_TRANSVERSE_COUPLING` or `PNFM_AXIAL_DOMINANCE`.

Activation produces `V33_JOINT_MEASURE_ADVERSARY_STOP`.

This fixture is unique to V33: it defeats a normal-histogram-only detector while testing whether the **joint** position-normal measure contains the intended causal direction.

## Gate 3 — fragmented positives

1. **`SF_PNFM_FRAG_COMPLEMENTARY_SKINS`**
   Divide the (24s) length into six (4s) slabs. Retain only the outer skin in even slabs and only the inner skin in odd slabs. Every surviving skin still contains the complete transverse fan.
   Expected: `1/1`.

2. **`SF_PNFM_FRAG_PHASE_WINDOWS`**
   Remove two distinct (3s) axial windows independently from the low-tail, core, and high-tail regions so the three phase groups never share identical missing intervals. No group has a gap above (3s).
   Expected: `1/1`.

These fixtures must pass without skin pairing, same-station coincidence, or station-offset selection. A failure produces `V33_FRAGMENT_STOP`.

## Gate 4 — all frozen nuisances

Import every previously frozen source-free nuisance and alias fixture byte-for-byte and in its frozen order, including the available instances of:

* planar sheet and terminal silhouette;
* free or opening rim;
* thickness face;
* cap/wall intersection;
* welded seam;
* sharp formed corner without a resolved bend fan;
* cycle-closing boundary;
* full cylinder or closed curved wall;
* unrelated intersecting panels;
* three-layer stack;
* duplicate and reversed-winding surface data.

Every ordinary nuisance expects zero V33 additions. The first activation produces `V33_NUISANCE_STOP`.

### Observation-equivalence witness

The exterior-identical bend-role/ground-joint-role witness is not assigned contradictory expected decisions. Instead:

* both members must produce byte-identical PNFM metric payloads and the same decision;
* any difference produces `V33_HIDDEN_SEMANTICS_STOP`.

The pair cannot count as evidence of semantic certification or nuisance rejection.

## Gate 5 — duplicate behavior

1. Exact duplicate and reversed-winding faces: decision and all 24 metrics must match the deduplicated source.
2. Two exact duplicate rescue candidates:

   * PNFM may emit both;
   * pre-NMS count must be two;
   * unchanged ordinary NMS must yield one.
3. One V16-retained candidate plus a duplicate V33 rescue candidate:

   * legacy-first ordering must retain the V16 instance.
4. A frozen within-radius near duplicate:

   * PNFM evaluates independently;
   * only ordinary NMS suppresses it.

Any mismatch produces `V33_DUPLICATE_ORDER_STOP`.

---

# 7. Mandatory native-resolution visual evidence

Every fixture receives exactly two lossless `4096 × 4096` review montages before its gate can be credited.

## Pre-invocation montage

Four fixed cameras/panels:

1. shaded native surface;
2. triangle-wireframe view;
3. orthographic candidate-normal cross-section;
4. candidate-axis view showing axial fragmentation.

The pre-image must show:

* the candidate at its frozen location;
* the intended two skins or intentionally one-sided fragments;
* the exact missing-material pattern;
* remesh variants as the same physical fold rather than altered geometry;
* the axial-twist adversary as a real twisted surface;
* no accidental opening rim, cap, wall, seam, or disconnected replacement positive.

All pre-images are manually reviewed and hashed before the first PNFM call. This is fixture validation, not a new approval packet.

## Post-invocation montage

Use the identical cameras and overlays:

* clipped PNFM support;
* low-tail, core, and high-tail areas in separately identifiable render layers;
* occupied radial runs;
* unchanged input candidate;
* emitted candidate, if any;
* first-failure reason and the locked scalar metrics.

A numerical pass does not advance unless the reviewer confirms that the measured fan and tails visibly belong to the intended fold. A numerical rejection does not count as a valid nuisance rejection when the pre/post view reveals fixture corruption or support on the wrong surface.

Any numerical–visual contradiction produces:

`V33_VISUAL_CONTRADICTION_STOP`

There is no fixture repair, alternate camera interpretation, or rerun after that token.

---

# 8. Metamorphic, replay, and stress locks

## Metamorphic lane

Run only after all semantic fixtures pass.

Required transformations:

* all 24 proper signed-axis cube rotations;
* endpoint reversal;
* full vertex permutation;
* full face permutation;
* independent face-winding reversal;
* translations by the frozen vectors
  ((7s,-11s,5s)), ((-13s,3s,17s));
* uniform scales (1/2,2,4), with (s) transformed identically;
* eight frozen non-grid proper rotations generated from the configuration’s fixed quaternion table.

Requirements:

* cube rotations, permutations, winding changes, and endpoint reversal: byte-identical PNFM records after canonical inverse transformation;
* non-grid rotations: identical decision, reason, fill count, radial-run count, and predicate mask; every other scalar within two quantization units; inverse-transformed line within (2\delta_x);
* scaling: identical dimensionless metrics and scale-normalized widths, coverage, and gaps.

First mismatch:

`V33_METAMORPHIC_STOP`

## Replay lane

Run the complete passed source-free fixture sequence in three fresh processes with:

* one worker;
* two workers;
* maximum permitted workers.

Require byte-identical:

* candidate decisions;
* PNFM sidecars;
* final post-NMS candidates;
* reason order;
* artifact manifests.

First mismatch:

`V33_REPLAY_STOP`

## Stress lane

Use the exact frozen:

* `200,000`-source-sample stress fixture;
* `128` rescue candidates.

Execution constraints on the same frozen reference host class used for V31/V32:

* CPU only;
* at most eight candidate workers;
* static worker assignment;
* deterministic ordered reduction;
* wall time at most `30.0 s`;
* peak process-tree RSS below `2 GiB`;
* no cap event;
* no spill file;
* cache size exactly

[
64+128\cdot192=24{,}640\ \text{bytes}.
]

Failure token:

`V33_STRESS_STOP`

---

# 9. Terminal policy

After the first PNFM mechanism invocation, any mismatch, crash, cap, hash drift, visual contradiction, or unexpected reason is terminal for this architecture.

The authorized terminal tokens are:

* `V33_CLEAN_FOLD_STOP`
* `V33_REMESH_INVARIANCE_STOP`
* `V33_JOINT_MEASURE_ADVERSARY_STOP`
* `V33_FRAGMENT_STOP`
* `V33_NUISANCE_STOP`
* `V33_HIDDEN_SEMANTICS_STOP`
* `V33_DUPLICATE_ORDER_STOP`
* `V33_VISUAL_CONTRADICTION_STOP`
* `V33_METAMORPHIC_STOP`
* `V33_REPLAY_STOP`
* `V33_STRESS_STOP`

The sole success token is:

`V33_PNFM_SOURCE_FREE_SURVIVOR`

That token means only that a face-measure, triangulation-tested normal-fan hypothesis survived its bounded analytic tranche. It does **not** open gold, production, runtime/main, the locked benchmark, or new physical scans. Those remain closed for a separately governed prospective test.

**Implementation and immediate execution are authorized now, beginning with the exact frozen `SF01_FOLD60_MESH_C2`.**

[1]: https://arxiv.org/abs/1304.6108 "https://arxiv.org/abs/1304.6108"
[2]: https://www.geometrie.tuwien.ac.at/geom/ig/publications/oldpub/2007/pwylh_principal_07/pwylh_principal_07.html "https://www.geometrie.tuwien.ac.at/geom/ig/publications/oldpub/2007/pwylh_principal_07/pwylh_principal_07.html"

