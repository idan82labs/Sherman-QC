RULING: EXECUTE_ONE

Execute **V55_MESH_SECTION_TURN_LOCUS_RESELECTION_V1**.

An append-only successor is no longer scientifically adequate. V52 already carries 27 false positives, while the frozen pool contains at most 91 of 100 truth bends. Even perfect zero-FP selection of every truth represented in that pool would yield precision of only (91/(91+27)=0.7712); even hypothetical recovery of all 100 truths without removing inherited false positives would reach only (100/127=0.7874), below the 90% target. A bounded pruning/reselection test is therefore necessary, not optional. 

A capability opening is not yet justified. V54’s source-only axis-normal sections already exposed its error directly: the proposed axes lay on straight panel interiors while the actual surface turn was consistently displaced. That does not establish general source-only sufficiency, but it establishes enough observable information for one direct centering test. 

## 1. Intervention identity

**Exact name:** `V55_MESH_SECTION_TURN_LOCUS_RESELECTION_V1`
**Short name:** `MSTLR_V1`
**Precommit token:** `V55_MSTLR_MECHANISM_LOCKED`
**Terminal token:** `V55_MSTLR_TERMINAL_RETIRE`

The terminal token is written on any preflight failure, scan-level abstention, incomplete primary, replay mismatch, resource failure, visual failure, or failure of the frozen retention gate. No corrected V55, V55b, relaxed threshold, replacement input, or second score is permitted.

## 2. Insertion point and mutation semantics

Insert MSTLR after:

1. the frozen source mesh has been loaded;
2. the complete frozen candidate pool has been loaded;
3. the frozen V52 prediction array has been loaded;

and before any runtime, main, production, export, or downstream integration.

MSTLR **replaces final selection only**:

* The retained V52 artifact itself remains byte-immutable.
* A V52 prediction may be omitted from the V55 output only under the high-safety pruning certificate in Section 5.
* Every retained V52 prediction keeps its exact original geometry, bytes, and relative order.
* Every addition must use the exact frozen geometry of one existing pool candidate.
* No line may be translated, rotated, extended, shortened, snapped to the measured turn locus, or otherwise re-estimated.
* No new candidate birth is allowed.
* After retained V52 lines, accepted additions are appended in the frozen order defined below.
* The legacy V52 NMS is not called again. MSTLR performs exactly one explicitly defined conflict pass over new additions.

## 3. Complete deterministic algorithm and constants

### 3.1 Global source scale

For each scan:

* (B) is the Euclidean diagonal of the source-mesh axis-aligned bounding box.
* Form the ascending list of lengths of all finite, nonzero, unique undirected mesh edges.
* (h) is the lower median: element (\lfloor(n-1)/2\rfloor).
* All computation uses IEEE-754 binary64.
* Zero is canonicalized to positive zero before serialization.

The scan is invalid if:

* (B\le 0);
* fewer than 100 eligible mesh edges exist;
* (h) is non-finite or (h/B\notin[10^{-6},0.05]).

### 3.2 Frozen constants

| Quantity                                   |                      Frozen value |
| ------------------------------------------ | --------------------------------: |
| Longitudinal stations                      |                                 9 |
| Station fractions                          |           (0.10,0.20,\ldots,0.90) |
| Minimum line span                          |        (L_{\min}=\max(24h,0.10B)) |
| Section radius                             |   (R=\min(0.12B,\max(12h,0.08L))) |
| Profile half-span                          |                         (W=0.70R) |
| Section-plane epsilon                      |                        (2^{-40}B) |
| Resampled profile points                   |                                65 |
| Minimum distinct section segments per side |                                16 |
| Axis-to-profile attachment tolerance       |           (\tau_a=\max(3h,0.04R)) |
| Flank TLS RMS limit                        |         (\tau_f=\max(1.5h,0.02R)) |
| Center tolerance                           |           (\tau_c=\max(3h,0.08R)) |
| Center 90th-percentile limit               |       (\tau_{c90}=\max(4h,0.12R)) |
| Definite offset threshold                  |           (\tau_o=\max(6h,0.20R)) |
| Maximum accepted offset                    |                           (0.75R) |
| Turn-locus TLS RMS limit                   |        (\tau_\ell=\max(3h,0.05R)) |
| Minimum/maximum flank turn                 |              (10^\circ/170^\circ) |
| Signed-turn coherence                      |                     at least 0.85 |
| Turn-versus-flank mismatch                 |  at most (\max(6^\circ,0.25\phi)) |
| Maximum 80% turn width                     |                           (0.55R) |
| Centered turn-mass fraction                |                     at least 0.45 |
| Offset-axis turn-mass fraction             |                      at most 0.10 |
| Offset-axis local absolute turn            |                 at most (3^\circ) |
| Turn-locus direction error                 |               at most (2.5^\circ) |
| Addition usable/centered stations          |                      at least 7/7 |
| Addition longest centered run              |                        at least 5 |
| Pruning offset stations                    |                        at least 8 |
| Pruning longest offset run                 |                        at least 7 |
| Offset-direction concentration             |                     at least 0.95 |
| Offset (q_{90}/q_{10})                     |                      at most 1.50 |
| Conflict direction angle                   |                 at most (3^\circ) |
| Conflict line separation                   |                 (\max(4h,0.015B)) |
| Conflict span overlap                      | at least 0.50 of the shorter span |
| BVH leaf size                              |                      16 triangles |
| Maximum intersecting triangles per station |                             8,192 |
| Maximum frozen candidates per scan         |                             2,048 |

### 3.3 Canonical line representation

For each finite candidate segment with endpoints (a,b):

1. Let (d=(b-a)/|b-a|).
2. Find the component of (d) having maximum absolute magnitude, breaking ties (x), then (y), then (z).
3. If that component is negative, swap (a,b) and negate (d).
4. Let (L=|b-a|).
5. Station (j), (j=0,\ldots,8), is
   [
   q_j=a+(0.10+0.10j)(b-a).
   ]

Lines shorter than (L_{\min}), or for which (R<10h), receive candidate-level abstention.

### 3.4 Deterministic mesh sections

Build one deterministic triangle AABB tree:

* split on the longest node-box axis, with ties (x,y,z);
* stable-sort triangle centroids;
* break centroid ties by frozen face ordinal;
* leaf size 16.

For each (q_j), intersect the source mesh with the plane

[
\Pi_j={x:(x-q_j)\cdot d=0}
]

inside the radius-(R) disk centered on (q_j).

Triangle-plane intersection follows these frozen rules:

* Signed vertex distances with magnitude at most (2^{-40}B) are snapped to zero.
* A normal crossing point is keyed transiently by its unique undirected mesh-edge ordinal.
* An exactly on-plane vertex is keyed by vertex ordinal.
* Coplanar faces and faces containing an entire on-plane edge are ignored.
* A non-coplanar face emitting exactly two distinct intersection nodes contributes one section segment.
* Any local face producing an unresolved number of distinct nodes makes that station unusable.
* No intersection node, vertex ID, face ID, or segment membership is serialized.

Build section components by shared intersection-node identity. Choose the component whose segment has the smallest Euclidean distance to (q_j); tie by lowest incident face ordinal.

The station is unusable when:

* more than 8,192 triangles intersect the query region;
* the selected component has a node of degree greater than two;
* its nearest distance to (q_j) exceeds (\tau_a);
* it cannot be traversed for at least (W) arclength in each direction;
* either side contains fewer than 16 distinct section segments.

Positive arclength direction is assigned to the branch whose endpoint at distance (W) from the nearest point is lexicographically larger in (x,y,z).

Resample the component at 65 equally spaced arclength locations

[
s_k=-W+\frac{2Wk}{64},\qquad k=0,\ldots,64.
]

### 3.5 Tangent-turn measurement

For each resampled profile:

1. Raw tangent for (k=2,\ldots,62):
   [
   r_k=\operatorname{normalize}(y_{k+2}-y_{k-2}).
   ]
2. Smoothed tangent for (k=4,\ldots,60):
   [
   t_k=\operatorname{normalize}\left(\sum_{u=-2}^{2}r_{k+u}\right).
   ]
3. Signed turn increments for (k=4,\ldots,59):
   [
   \delta_k=\operatorname{atan2}!\left(d\cdot(t_k\times t_{k+1}),,t_k\cdot t_{k+1}\right).
   ]
4. Let
   [
   A=\sum|\delta_k|,\qquad T=\left|\sum\delta_k\right|,\qquad C=T/A.
   ]
   (C=0) when (A=0).

Fit orthogonal TLS lines to profile samples in the two flank intervals:

* left: (s/W\in[-1,-0.55]);
* right: (s/W\in[0.55,1]).

Orient each TLS direction in increasing profile arclength. Let (\phi) be their directed angle in ([0^\circ,180^\circ]).

A station has a **valid localized turn** only when all conditions hold:

* both flank RMS values are at most (\tau_f);
* (10^\circ\le\phi\le170^\circ);
* (T\ge10^\circ);
* (C\ge0.85);
* (|T-\phi|\le\max(6^\circ,0.25\phi)).

Let (\sigma=\operatorname{sign}(\sum\delta_k)), with a zero tie assigned (+1), and define positive turn mass

[
m_k=\max(0,\sigma\delta_k).
]

The 3-D turn center is

[
g_j=\frac{\sum m_k (y_k+y_{k+1})/2}{\sum m_k}.
]

The turn width is the shortest contiguous arclength interval containing 80% of (\sum m_k), with an earlier-start tie-break. It must not exceed (0.55R).

Define:

* (\rho_j=|g_j-q_j|);
* (F_{0j}): fraction of positive turn mass whose midpoint has (|s|\le\tau_c);
* (A_{0j}): total absolute turn within (|s|\le\tau_c).

A valid-turn station is classified:

* **CENTERED** when
  (\rho_j\le\tau_c) and (F_{0j}\ge0.45).

* **OFFSET_FLAT_AXIS** when
  (\tau_o\le\rho_j\le0.75R),
  (F_{0j}\le0.10), and
  (A_{0j}\le3^\circ).

* **VALID_OTHER** otherwise.

### 3.6 Longitudinal turn-locus fit

For CENTERED or OFFSET_FLAT_AXIS stations, fit a 3-D TLS line to their (g_j) points.

* Principal direction sign is aligned to (d).
* Direction error is the unoriented angle to (d).
* RMS is the orthogonal RMS of the (g_j) points to the fitted line.
* Ties in the eigensolve use the lexicographically smallest canonical eigenvector.

No fitted turn-locus geometry is output as a prediction.

### 3.7 V52 pruning decision

Every V52 line is **kept by default**.

Prune it only when all of the following hold:

1. the line satisfies (L\ge L_{\min});
2. at least 8 of 9 stations are OFFSET_FLAT_AXIS;
3. the longest consecutive OFFSET_FLAT_AXIS run is at least 7;
4. zero stations are CENTERED;
5. the fitted offset turn locus differs from (d) by at most (2.5^\circ);
6. turn-locus RMS is at most (\tau_\ell);
7. for (v_j=(g_j-q_j)/|g_j-q_j|),
   [
   \left|\frac{1}{n}\sum v_j\right|\ge0.95;
   ]
8. the offset-distance ratio (q_{90}(\rho)/q_{10}(\rho)\le1.50);
9. median (\rho\ge\tau_o);
10. (q_{90}(\rho)\le0.75R).

This is the only pruning certificate. A flat profile with no positively localized alternative turn is insufficient to prune. Any ambiguity, boundary loss, section failure, shallow turn, topology problem, or insufficient station count keeps the incumbent.

### 3.8 Addition decision

A frozen pool candidate becomes an addition proposal only when:

1. it is not bitwise geometry-identical to any original V52 line;
2. (L\ge L_{\min});
3. at least 7 stations have valid localized turns;
4. at least 7 stations are CENTERED;
5. the longest CENTERED run is at least 5;
6. the CENTERED (g_j) locus differs from (d) by at most (2.5^\circ);
7. turn-locus RMS is at most (\tau_\ell);
8. median (\rho\le\tau_c);
9. (q_{90}(\rho)\le\tau_{c90}).

Proposal ordering is lexicographic:

1. descending CENTERED station count;
2. descending longest CENTERED run;
3. ascending (q_{90}(\rho)/R);
4. ascending turn-locus RMS divided by (R);
5. descending (L/B);
6. ascending frozen candidate ordinal.

Process this list once. Accept a proposal only when it does not conflict with any retained V52 line or earlier accepted addition.

Two lines conflict only if all hold:

* direction angle at most (3^\circ);
* shortest infinite-line separation at most (\max(4h,0.015B));
* projected segment overlap along their normalized mean direction is at least 0.50 of the shorter segment.

## 4. Source inputs, populations, ordering, and compact cache

### Inputs permitted

Only:

* frozen source vertex coordinates;
* frozen triangular faces and their adjacency implied by shared edges;
* frozen V52 line geometry and order;
* frozen candidate line geometry and ordinal;
* integrity hashes and manifest order.

MSTLR must not read:

* V52 or candidate support banks;
* source-owner maps;
* V53 route records;
* V54 provenance records, partial additions, station results, or normal-turn cache;
* source labels, truth, family identity, filenames, expected counts, CAD, nominal geometry, A/B identity, or benchmark results.

### Population

* **Pruning population:** every V52 prediction.
* **Addition population:** every frozen pool candidate except a bitwise geometry duplicate of an original V52 prediction.
* No family or scan-specific filtering is allowed.

### Scan order

Use the exact frozen 30-scan manifest order for both primary and replay. Manifest position is harness metadata and may not enter any geometric calculation.

### Compact cache schema

Per scan, serialize only:

* schema version;
* source, V52, pool, code, constants, and precommit hashes;
* (B) and (h);
* for each V52 line: ordinal, keep/prune decision, reason code, usable/centered/offset 9-bit masks, longest run, median and (q_{90}) normalized offset, median turn angle, locus angle, locus RMS, offset concentration;
* for each accepted addition: the same scalar summaries and candidate ordinal;
* aggregate rejection counts by reason;
* final line array with source kind, frozen ordinal, and exact segment endpoints.

Nine-bit station masks are permitted because they describe nine fixed tests, not source membership. No source point, vertex, edge, face, section segment, profile sample, turn-center coordinate, support set, or membership bitset may be serialized.

Maximum canonical output size is 2 MiB per scan and 64 MiB for the complete run.

## 5. Direct transverse-localization certificate

The decisive quantity is not “how much turn evidence belongs to this line.” It is:

[
\rho_j=|\text{measured profile-turn center }g_j-\text{candidate-axis point }q_j|.
]

An addition requires the actual profile’s tangent rotation to be centered at (q_j) at at least seven longitudinal stations.

A prune requires stronger, contrary evidence: at least eight stations must show that:

* the candidate axis lies on a locally straight part of the same section profile;
* a localized, coherent bend exists away from it;
* the displaced bend centers form a stable line parallel to the candidate;
* the displacement is consistently in the same transverse direction.

For the V54 S010/S012 failure mode, sections with the candidate at (x=0) and the real turn at (x\approx+8) to (+9) mm, or farther away, would produce OFFSET_FLAT_AXIS rather than CENTERED. The test uses the actual mesh-section coordinates, not turn mass borrowed from candidate support banks. 

## 6. Abstention rules

### Candidate- or station-level abstention

The following do **not** terminate the scan:

* line shorter than (L_{\min});
* insufficient scale separation;
* no attached section component;
* open section with insufficient two-sided coverage;
* section branch or local non-manifold topology;
* more than 8,192 local triangles;
* insufficient segment count;
* non-localized or distributed curvature;
* fewer than the required usable stations.

For a pool candidate, abstention means no addition.

For a V52 incumbent, abstention means KEEP_DEFAULT. Pruning may never be inferred from missing or unusable evidence.

### Scan-level abstention

The scan must abstain, and the entire experiment terminates immediately, on:

* source, V52, pool, code, constants, or precommit hash mismatch;
* malformed or non-finite frozen line geometry;
* invalid (B) or (h);
* more than 2,048 frozen candidates;
* failure to construct the deterministic section index;
* forbidden-field access;
* canonical serializer failure;
* output-schema overflow;
* wall-time or memory guard;
* an exception not already mapped to a candidate/station abstention.

A valid zero-change scan is not an abstention.

## 7. Immediate failure gates and one-shot execution

### Gate 0: fixed source-free preflight

Constants and code are hashed before fixture execution. Run exactly these fixtures once:

| Fixture                            | Frozen expected result                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `PF01_CENTERED_SHARP90`            | centered candidate passes 9/9; incumbent kept                                                  |
| `PF02_CENTERED_ROUND60`            | centered candidate passes at least 8/9; incumbent kept                                         |
| `PF03_SINGLE_PLANE`                | no addition; incumbent kept                                                                    |
| `PF04_OFFSET_FOLD_0P40R`           | flat-panel incumbent pruned; true centered candidate accepted                                  |
| `PF05_DOUBLE_FOLD_PM0P40R`         | midpoint candidate neither added nor pruned; both centered fold candidates remain eligible     |
| `PF06_DISTRIBUTED_CYLINDER`        | rejected by turn-width/localization gate                                                       |
| `PF07_OPEN_BOUNDARY`               | candidate abstains; incumbent kept                                                             |
| `PF08_DIAGONAL_CROSS_PANEL`        | diagonal candidate rejected; true fold candidate accepted                                      |
| `PF09_TWO_PARALLEL_FOLDS_0P08B`    | both true candidates accepted and not conflict-suppressed                                      |
| `PF10_CANONICAL_SERIALIZER_REPLAY` | native and library scalar types canonicalize to identical bytes; non-finite values fail closed |

The geometric fixtures use deterministic analytic meshes, mesh spacing (h=0.5), (B=200), line span (L=120), and no noise or random sampling. Relative offsets and radii are exactly those in the fixture names or the frozen constants.

Any failed assertion writes the terminal token before the sealed lane is opened.

### Gate 1: primary

Run all 30 scans in frozen order. Stop immediately on the first scan-level abstention or immediate failure.

### Gate 2: replay

Only after a complete primary, start a clean process and execute the same 30 scans in the same order. Require byte identity for:

* all decisions;
* all scalar summaries;
* all final line arrays;
* every per-scan canonical cache;
* the complete manifest.

### Gate 3: freeze and audit

After replay identity:

1. seal outputs and hashes;
2. generate all required source-only visuals;
3. complete and sign the blind visual audit;
4. open truth exactly once;
5. score exactly once;
6. apply the frozen retention gate.

Immediate failures include:

* mutation of the V52 artifact;
* geometric alteration of any retained V52 line;
* output of geometry not present in V52 or the frozen pool;
* any V53/V54 import;
* truth or identity opened before output and visual-audit freeze;
* non-finite result;
* canonicalization failure;
* peak RSS reaching 15.5 GiB;
* any per-scan wall time reaching 285 seconds;
* output exceeding its size cap;
* replay mismatch.

The lower resource guards provide margin beneath the packet’s hard 16 GiB and 300-second limits. 

## 8. Minimum positive retention gate

Retain V55 only if **every** condition holds:

| Gate                    |                             Required result |
| ----------------------- | ------------------------------------------: |
| Correct additions       |                               at least 6 TP |
| False additions         |                                exactly 0 FP |
| Correct prunes          |             at least 6 inherited FP removed |
| Incorrect prunes        |              exactly 0 inherited TP removed |
| Final TP                |                                 at least 58 |
| Final FP                |                                  at most 21 |
| Final FN                |                                  at most 42 |
| Precision               |                               at least 0.73 |
| Recall                  |                               at least 0.58 |
| F1                      |                              at least 0.645 |
| Exact-count scans       |                              at least 12/30 |
| Topology-family breadth | correct change in at least 5 of 15 families |
| Replay                  |                        30/30 byte-identical |
| Visual gate             |                every mandatory audit passes |

For S005, S006, S007, S013, S026, and S027, the complete prediction array must be exactly identical to V52 in count, geometry, and order. No prune and no addition is allowed in those six scans. They are evaluation-side zero-regression gates and are not exposed during inference. Their status as the six exact V52 cases is frozen in the packet. 

No family may lose a TP or gain an FP. One failed condition retires the entire V55 result; partial retention is prohibited.

## 9. Mandatory native-resolution visual signatures

Render, without truth overlays:

* every scan containing an addition or prune;
* all six exact V52 scans;
* S011, S016, S024, and S029;
* every changed line separately when a scan has multiple changes.

Each changed-line suite must contain:

1. full native-resolution raw-source overview;
2. six-view V52/V55 comparison;
3. nine axis-normal sections at the exact detector stations;
4. candidate axis fixed at section coordinate (x=0);
5. the traversed source profile;
6. the measured turn-center marker;
7. (\pm\tau_c) centering limits;
8. fitted left and right flank lines;
9. the measured turn-width interval.

### Required visual signature for an addition

Across at least seven stations:

* the actual rounded or sharp surface transition crosses (x=0);
* the two locally planar flanks transition through the reported axis;
* no stronger parallel fold remains consistently displaced from (x=0);
* the line follows the same physical bend longitudinally;
* the marker is neither floating nor diagonal across panel interiors.

### Required visual signature for a prune

Across at least eight stations:

* the old V52 axis lies on a visibly straight local profile;
* the visible turn is displaced in the same transverse direction;
* the displaced turn remains longitudinally coherent and parallel;
* there is no visible shallow or sharp transition at the pruned axis.

### Automatic visual failures

Retire V55 if any of these occurs:

* an added line lies on a flat profile in two or more usable sections;
* a visible turn is outside the displayed center tolerance in two or more usable sections;
* a pruned line visibly lands on a physical fold in any two sections;
* the supposed displaced fold changes side, disappears, or belongs to unrelated geometry;
* a changed line is floating, truncated, duplicated, or diagonally crosses panels;
* a required section is unreadable, clipped, or rendered at reduced resolution;
* any of the six exact V52 cases changes;
* any changed marker is not render-admissible.

The visual gate overrides aggregate metrics. V54 demonstrated why: both failed additions had strong numeric longitudinal evidence, and only axis-normal sections exposed the localization error. 

## 10. Leakage, determinism, latency, and memory audits

### Leakage

* No truth, labels, CAD, drawings, nominal geometry, expected counts, case name, filename, family identity, A/B identity, part number, or benchmark feedback enters inference.
* The candidate and V52 ordinals are ordering/integrity references only.
* Scan IDs used to request mandatory review are unavailable inside the detector.
* No S010/S012-specific coordinate, offset, topology, or rule is allowed.
* V54’s partial outputs and provenance audit are not detector inputs.
* Truth and family identity remain closed until primary, replay, output freeze, and blind source-only visual judgments are complete.

### Determinism

* No random sampling.
* BVH construction, section traversal, eigenvector signs, component ties, line ordering, and conflict decisions all have explicit tie-breaks.
* Triangle and segment reductions occur in ascending frozen ordinal.
* Parallel execution may occur only across independent candidates or scans; floating-point reductions within a candidate remain serial and ordinally fixed.
* BLAS and eigensolver thread counts are fixed at one.
* Canonical serialization recursively converts scalar-library booleans, integers, and floats to native scalar types, rejects non-finite values, sorts keys, emits no optional whitespace, and uses shortest round-trip float representation.
* Primary and replay run in clean processes.

### Resource bounds

* Deterministic BVH storage is linear in source faces.
* At most one 65-point profile and one bounded section graph are live per station.
* Source section nodes are released before the next station.
* Hard RSS guard: 15.5 GiB.
* Hard per-scan wall guard: 285 seconds.
* Maximum 8,192 local triangles per station.
* Maximum 2,048 candidates and 2 MiB serialized output per scan.

Thirty complete primary scans and thirty complete replay scans are required; this lane cannot establish the enterprise 99.5% job-success claim.

## 11. Causal justification

V55 attacks two current defects with one geometric observable.

First, it attacks the pool-contained false negatives. Thirty-nine of the 48 V52 misses are, in principle, represented inside the 91-truth candidate ceiling. Those candidates can now be selected only when the actual cross-sectional turn is directly centered on their axes, rather than when remote evidence merely accumulates around their support banks. 

Second, it begins the scientifically unavoidable removal of inherited false positives. The prune is deliberately asymmetric: uncertainty keeps V52; only repeated positive evidence of a coherent displaced bend permits deletion. This is needed because append-only selection cannot approach the precision objective while the 27 inherited false positives remain.

This does not retest V53:

* there is no second route;
* no adjacency/lattice agreement;
* no source-bank agreement;
* no route-pair population.

This does not retest or repair V54:

* no support-bank ownership;
* no longitudinal evidence bins;
* no exclusive evidence assignment;
* no normal-turn edge aggregation;
* no narrower radius applied to the same statistic;
* no provenance re-resolution;
* no correction of S013 followed by rerun.

V54 failed partly because source evidence could not always be re-resolved, despite the underlying records existing; V55 never requests that mapping. 

Most importantly, V54 treated longitudinally strong turn mass as evidence that the turn was centered. V55 measures the transverse location of the turn itself. The hypothesis under test is therefore:

> A true frozen bend candidate, or a safely removable false incumbent, can be distinguished by the repeated location of a localized source-profile tangent transition relative to the reported axis.

That hypothesis is narrow, causal, and falsifiable in one run.

## 12. Research grounding

**Established ideas.** Sharp-feature extraction from sampled geometry commonly combines local differential evidence with spatial or graph continuity; Demarsin et al. use normal estimation and graph theory for sharp-edge extraction. Neighborhood-integrated or covariance-based geometry has also been developed specifically to make curvature and sharp-feature estimation less sensitive to sampling noise and discretization. ([ScienceDirect][1])

Robust model-fitting research distinguishes consensus-supported structure from averages contaminated by unrelated observations. The original RANSAC formulation also explicitly notes that deterministic selection may replace random sampling when problem structure supplies a fixed rationale. MSTLR uses that principle only in a bounded deterministic form: fixed profile sections, fixed flank fits, and repeated station consensus. 

**Sherman-specific hypothesis.** The literature does not establish that nine sections, these radii, these centering constants, or this asymmetric prune/add rule will recover Sherman bends. Nor does it establish that the frozen candidate axes coincide with planar-section turn centroids. Those are the preregistered V55 assumptions. The single sealed execution must either demonstrate broad, zero-error corrections under the frozen gates or retire the mechanism without rescue.

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0010448506002260?utm_source=chatgpt.com "Detection of closed sharp edges in point clouds using ..."

